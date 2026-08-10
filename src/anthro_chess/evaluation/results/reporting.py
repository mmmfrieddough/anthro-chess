"""Compact delta views over the results store.

The delta view is the primary agent-facing surface, which is why it is small
by default. An agent that has to read a full artifact to answer "did this get
worse" spends most of its context on irrelevant JSON.

Everything the default view leaves out stays available behind an explicit
option: slices by family or metric, provenance, and full per-series history.
A family with nothing to report is named with a reason rather than dropped,
because a silently missing family reads as "fine" when it usually means "not
measured".
"""

from __future__ import annotations

import json
import math
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from anthro_chess.evaluation.results.comparability import (
    UNSCOPED_WORKLOAD,
    Attribution,
    AxisChange,
    BridgeIndex,
    Comparability,
    ProvenanceDifference,
    attribute,
    condition_differences,
    environment_differences,
    measurements_by_workload,
    provenance_differences,
    recorded_series,
)
from anthro_chess.evaluation.results.metrics import (
    MetricDefinition,
    MetricDirection,
    MetricFamily,
    MetricRegistryError,
    metric_definition,
    registered_families,
    registered_metrics,
)
from anthro_chess.evaluation.results.noise import combined_floor
from anthro_chess.evaluation.results.records import (
    Measurement,
    ResultEnvelope,
)
from anthro_chess.evaluation.results.store import (
    checkpoint_labels,
    results_for_checkpoint,
)

#: Absence reason for a family registered ahead of the benchmark that fills it.
#: Named rather than inlined so the renderer can tell it apart from an absence
#: that is about the checkpoint being reported.
UNREGISTERED_FAMILY_ABSENCE = "no metric is registered for this family yet"

#: The default view is read in a terminal beside other output, so it wraps
#: rather than relying on the reader's window.
MAXIMUM_LINE_WIDTH = 120


def _delta_header(metric_width: int) -> str:
    """Return the delta table's header row at one identifier-column width.

    Defined beside the widths rather than with the other renderers because the
    budget below is measured from it. The header is the widest fixed part of
    the table, so what it spends is what the identifier column has left.
    """

    return (
        f"  {'metric':<{metric_width}} {'better':<6} "
        f"{'baseline':>11} {'current':>11} {'delta':>11}  {'change':<9} noise"
    )


#: Any width past the header's own ``metric`` label, so subtracting it from the
#: rendered header leaves exactly the fixed part.
_HEADER_PROBE_WIDTH = 64

#: Everything the header spends on something other than the identifier: the
#: indent, the fixed columns, and the separators between them. Measured from
#: the header rather than written down, so resizing a column cannot leave the
#: budget below it stale.
_HEADER_OVERHEAD = len(_delta_header(_HEADER_PROBE_WIDTH)) - _HEADER_PROBE_WIDTH

#: The identifier column never renders narrower than this, so an ordinary
#: report keeps the shape a reader is used to instead of reflowing to whatever
#: names happen to be in it.
MINIMUM_METRIC_COLUMN_WIDTH = 38

#: The longest identifier the header can carry inside ``MAXIMUM_LINE_WIDTH``.
#: The renderer widens to whatever it is given rather than truncating here,
#: because a truncated identifier is not one a reader can look up; this is the
#: budget the registry is held to, so nothing longer reaches the report.
#:
#: Measured on the header rather than on a row, because a row's noise
#: annotation is optional and long enough that bounding it would force the
#: column back to a width the registry has already outgrown. A row carrying
#: one can therefore run past the line, as it does today wherever a long
#: identifier is reported.
MAXIMUM_METRIC_COLUMN_WIDTH = MAXIMUM_LINE_WIDTH - _HEADER_OVERHEAD

#: How much of one workload value a series label shows. The label exists to
#: tell two cells of a matrix apart, and the whole value is in the envelope for
#: the reader who needs it.
WORKLOAD_VALUE_WIDTH = 28

#: Series label for the group of a family's metrics that declare no workload.
#: Only reachable where a family mixes workload-scoped metrics with metrics
#: whose value does not depend on execution.
UNSCOPED_SERIES_LABEL = "no declared workload"


class ReportError(ValueError):
    """Raised when a report cannot be built from the requested selection."""


class Movement(StrEnum):
    """Whether a delta is an improvement, judged only by declared direction."""

    BETTER = "better"
    WORSE = "worse"
    #: The delta is zero, or inside the floor that qualifies it. The two are one
    #: statement to act on: nothing here has been shown to move.
    UNCHANGED = "unchanged"
    INFORMATIONAL = "informational"
    UNKNOWN = "unknown"
    #: The delta is real and interpretable, but something other than the model
    #: moved as well, so it is not a verdict on the model. Distinct from
    #: ``UNKNOWN``, which means there was nothing to compare against.
    #:
    #: This is the field automation keys on, which is why the honesty lives
    #: here rather than in a withheld ``delta``: a reader holding both operands
    #: can always subtract them, so hiding the arithmetic protects nobody.
    CONFOUNDED = "confounded"


class ReportPivot(StrEnum):
    """Which coordinate a report varies, and therefore which it asks about."""

    #: Vary the checkpoint, hold the environment still. "Did the model change
    #: make this slower?"
    CHECKPOINT = "checkpoint"
    #: Vary the environment, hold the model still. "Did the upgrade help?"
    ENVIRONMENT = "environment"


class NoiseVerdict(StrEnum):
    """Whether a delta is larger than the noise in the measurement."""

    CLEARED = "cleared"
    WITHIN = "within"
    UNKNOWN = "unknown"
    #: No floor can exist for this metric, because what it counts is not what
    #: any characterization resamples. Distinct from ``UNKNOWN``, which means a
    #: floor could exist and none was found: only one of the two is waiting on
    #: work somebody could do.
    UNQUALIFIABLE = "unqualifiable"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class MetricDelta:
    """One metric's default-view row."""

    metric: str
    family: str
    direction: MetricDirection
    baseline: float | None
    current: float | None
    delta: float | None
    comparability: Comparability
    movement: Movement
    noise: NoiseVerdict
    #: The floor this delta faces, combined from the spread each of the two
    #: readings measured for itself. A cleared delta is larger than benchmark
    #: noise and is **not** thereby established as caused by the change: two
    #: arms differ by their initialization seeds as well, and nothing a reading
    #: measures about itself can see that.
    noise_floor: float | None
    #: How the two readings described the spreads the floor combines, baseline
    #: first. A floor combined from a thousand-game reading and a forty-game one
    #: is neither reading's, and which side contributed the width is the first
    #: thing a reader chasing a wide floor wants.
    noise_floor_source: str | None
    bridges: tuple[str, ...]
    note: str | None
    #: Which coordinates moved. ``None`` for a metric with no execution
    #: context, where the model is the only thing that can have moved.
    attribution: Attribution | None = None
    #: The execution coordinates that differ, when the environment moved.
    environment: tuple[ProvenanceDifference, ...] = ()
    #: The benchmark-declared coordinates that differ, such as the model
    #: architecture or the training corpus. These confound a delta rather than
    #: invalidating it, so they are named beside the number, not instead of it.
    conditions: tuple[ProvenanceDifference, ...] = ()
    #: The series this row reports, so automation reading the rows of a matrix
    #: benchmark can tell them apart without reconstructing the grouping.
    series: str | None = None

    def as_record(self) -> dict[str, object]:
        """Return the machine-readable row."""

        return {
            "metric": self.metric,
            "family": self.family,
            "series": self.series,
            "direction": self.direction.value,
            "baseline": self.baseline,
            "current": self.current,
            "delta": self.delta,
            "comparability": self.comparability.value,
            "movement": self.movement.value,
            "noise": self.noise.value,
            "noise_floor": self.noise_floor,
            "noise_floor_source": self.noise_floor_source,
            "bridges": list(self.bridges),
            "note": self.note,
            "attribution": (
                None if self.attribution is None else self.attribution.as_record()
            ),
            "environment_differences": _difference_records(self.environment),
            "condition_differences": _difference_records(self.conditions),
        }


@dataclass(frozen=True)
class SeriesGroup:
    """One declared workload's rows within a family.

    A benchmark that varies a dial across a matrix writes one result per cell,
    and every cell is its own series. Grouping the rows by the workload they
    were measured under is what keeps a matrix readable: the dial is stated
    once above the rows it applies to, rather than one cell being shown as
    though it were the checkpoint's value.
    """

    #: The declared workload's digest, or ``None`` where the metrics in this
    #: group declare no workload.
    workload: str | None
    #: How this group differs from the others in its family, or ``None`` when
    #: it is the only one and there is nothing to tell apart.
    label: str | None
    metrics: tuple[MetricDelta, ...]
    #: The workload fields that differ across the family's groups, which is
    #: what ``label`` renders.
    coordinates: tuple[tuple[str, str], ...] = ()
    #: Execution differences shared by every row in this group. Rendered once
    #: as a header rather than repeated per row, because execution is a
    #: property of the result the whole group was recorded in.
    environment: tuple[ProvenanceDifference, ...] = ()
    #: Declared-coordinate differences, shared by the group for the same
    #: reason.
    conditions: tuple[ProvenanceDifference, ...] = ()

    def as_record(self) -> dict[str, object]:
        """Return the machine-readable group section."""

        return {
            "workload": self.workload,
            "label": self.label,
            "coordinates": [
                {"field": field, "value": value} for field, value in self.coordinates
            ],
            "environment_differences": _difference_records(self.environment),
            "condition_differences": _difference_records(self.conditions),
            "metrics": [metric.as_record() for metric in self.metrics],
        }


@dataclass(frozen=True)
class FamilyReport:
    """One family's rows, grouped by series, or the reason it has none."""

    family: MetricFamily
    series: tuple[SeriesGroup, ...]
    absence: str | None

    @property
    def metrics(self) -> tuple[MetricDelta, ...]:
        """Return every row in the family, across its series."""

        return tuple(row for group in self.series for row in group.metrics)

    def as_record(self) -> dict[str, object]:
        """Return the machine-readable family section."""

        return {
            "family": self.family.identifier,
            "title": self.family.title,
            "absence": self.absence,
            "series": [group.as_record() for group in self.series],
        }


@dataclass(frozen=True)
class CheckpointSelection:
    """Which recorded checkpoint one side of a comparison refers to."""

    label: str
    recorded_at: datetime
    results: int

    def as_record(self) -> dict[str, object]:
        """Return the machine-readable selection."""

        return {
            "label": self.label,
            "recorded_at": self.recorded_at.isoformat(),
            "results": self.results,
        }


@dataclass(frozen=True)
class DeltaReport:
    """A whole default view: two checkpoints, by family, by metric."""

    baseline: CheckpointSelection | None
    current: CheckpointSelection
    families: tuple[FamilyReport, ...]
    provenance: tuple[ProvenanceDifference, ...]
    pivot: ReportPivot = ReportPivot.CHECKPOINT

    def as_record(self) -> dict[str, object]:
        """Return the machine-readable report."""

        return {
            "pivot": self.pivot.value,
            "baseline": None if self.baseline is None else self.baseline.as_record(),
            "current": self.current.as_record(),
            "families": [family.as_record() for family in self.families],
            "provenance": _difference_records(self.provenance),
        }


@dataclass(frozen=True)
class HistoryPoint:
    """One recorded value in a metric's history."""

    recorded_at: datetime
    checkpoint: str
    value: float
    fingerprint: str
    series: str
    starts_new_series: bool
    bridged_from_previous: bool
    #: Where this point was measured, for an efficiency metric. The series is
    #: continuous across machines by design, so the line stays readable as a
    #: long-run trend and the annotation is what keeps it honest.
    environment: str | None = None
    environment_changed: bool = False

    def as_record(self) -> dict[str, object]:
        """Return the machine-readable point."""

        return {
            "recorded_at": self.recorded_at.isoformat(),
            "checkpoint": self.checkpoint,
            "value": self.value,
            "fingerprint": self.fingerprint,
            "series": self.series,
            "starts_new_series": self.starts_new_series,
            "bridged_from_previous": self.bridged_from_previous,
            "environment": self.environment,
            "environment_changed": self.environment_changed,
        }


@dataclass(frozen=True)
class MetricHistory:
    """One metric's whole recorded history, with its series seams marked."""

    metric: str
    direction: MetricDirection
    points: tuple[HistoryPoint, ...]

    def as_record(self) -> dict[str, object]:
        """Return the machine-readable history."""

        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "points": [point.as_record() for point in self.points],
        }


def build_delta_report(
    results: Sequence[ResultEnvelope],
    bridges: BridgeIndex,
    *,
    current: str | None = None,
    baseline: str | None = None,
    families: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
) -> DeltaReport:
    """Build the default compact view between two recorded checkpoints."""

    pivot = ReportPivot.CHECKPOINT
    labels = checkpoint_labels(results)
    if not labels:
        raise ReportError("the results store has no recorded results")

    current_label = current if current is not None else labels[-1]
    if current_label not in labels:
        raise ReportError(f"no result is recorded for checkpoint {current_label!r}")
    if baseline is None:
        earlier = [label for label in labels if label != current_label]
        baseline_label = earlier[-1] if earlier else None
    else:
        if baseline not in labels:
            raise ReportError(f"no result is recorded for checkpoint {baseline!r}")
        baseline_label = baseline
    if baseline_label == current_label:
        raise ReportError("the baseline and current checkpoints must differ")

    current_results = results_for_checkpoint(results, current_label)
    baseline_results = (
        results_for_checkpoint(results, baseline_label)
        if baseline_label is not None
        else ()
    )

    sliced = families is not None or metrics is not None
    selected = _selected_metrics(families, metrics)
    sections: list[FamilyReport] = []
    for family in registered_families():
        family_metrics = tuple(
            definition
            for definition in registered_metrics(family.identifier)
            if definition.identifier in selected
        )
        # An explicit slice asked about specific metrics; naming every other
        # family as absent would bury the answer it asked for.
        if sliced and not family_metrics:
            continue
        sections.append(
            _family_report(
                family,
                family_metrics,
                current_results,
                baseline_results,
                bridges,
                current_label,
                baseline_label,
                pivot,
            )
        )

    return DeltaReport(
        baseline=(
            _selection(baseline_label, baseline_results)
            if baseline_label is not None
            else None
        ),
        current=_selection(current_label, current_results),
        families=tuple(sections),
        provenance=_report_provenance(baseline_results, current_results),
        pivot=pivot,
    )


def build_environment_report(
    results: Sequence[ResultEnvelope],
    bridges: BridgeIndex,
    *,
    checkpoint: str | None = None,
    families: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
) -> DeltaReport:
    """Compare one checkpoint's efficiency across two environments.

    The mirror image of the default view: the model is pinned and the machine
    varies, which is the question an optimization asks. Pinning is by
    ``parameter_sha256`` rather than by label, because a reused label would
    quietly turn a model change into an apparent hardware win.
    """

    pivot = ReportPivot.ENVIRONMENT
    measured = [envelope for envelope in results if envelope.execution is not None]
    if not measured:
        raise ReportError(
            "no recorded result carries an execution record, so there is no "
            "environment to compare"
        )

    label = (
        checkpoint if checkpoint is not None else _latest_multi_environment(measured)
    )
    selected = [envelope for envelope in measured if envelope.checkpoint.label == label]
    if not selected:
        raise ReportError(f"no efficiency result is recorded for checkpoint {label!r}")
    _require_one_model(selected, label)

    groups = _by_environment(selected)
    if len(groups) < 2:
        raise ReportError(
            f"checkpoint {label!r} was only measured in one environment; there "
            "is nothing to compare it against"
        )
    ordered = sorted(
        groups.values(),
        key=lambda group: max(envelope.recorded_at for envelope in group),
    )
    baseline_results = ordered[-2]
    current_results = ordered[-1]

    sliced = families is not None or metrics is not None
    selected_metrics = _selected_metrics(families, metrics)
    sections: list[FamilyReport] = []
    for family in registered_families():
        family_metrics = tuple(
            definition
            for definition in registered_metrics(family.identifier)
            if definition.identifier in selected_metrics
            # An environment comparison is only meaningful for a metric whose
            # value depends on execution. Reporting move loss here would invite
            # reading an unchanged number as evidence about the machine.
            and definition.execution_sensitive
        )
        if not family_metrics:
            if sliced:
                continue
            continue
        sections.append(
            _family_report(
                family,
                family_metrics,
                current_results,
                baseline_results,
                bridges,
                _environment_name(current_results),
                _environment_name(baseline_results),
                pivot,
            )
        )
    if not sections:
        raise ReportError(
            "no execution-sensitive metric was selected; an environment "
            "comparison has nothing to report about other metrics"
        )

    return DeltaReport(
        baseline=_environment_selection(baseline_results),
        current=_environment_selection(current_results),
        families=tuple(sections),
        provenance=_report_provenance(baseline_results, current_results),
        pivot=pivot,
    )


def build_history(
    results: Sequence[ResultEnvelope],
    bridges: BridgeIndex,
    metric: str,
) -> MetricHistory:
    """Build one metric's full history, marking every series seam."""

    try:
        definition = metric_definition(metric)
    except MetricRegistryError as error:
        raise ReportError(str(error)) from error

    points: list[HistoryPoint] = []
    previous_fingerprint: str | None = None
    previous_environment: dict[str, str | None] | None = None
    for envelope in sorted(
        results,
        key=lambda item: (item.recorded_at, item.result_id),
    ):
        found = envelope.measurement(definition.identifier)
        if found is None:
            continue
        comparison = (
            bridges.compare(previous_fingerprint, found.fingerprint)
            if previous_fingerprint is not None
            else None
        )
        execution = envelope.execution
        environment = None if execution is None else execution.environment()
        points.append(
            HistoryPoint(
                recorded_at=envelope.recorded_at,
                checkpoint=envelope.checkpoint.label,
                value=found.value,
                fingerprint=found.fingerprint,
                series=bridges.series(found.fingerprint),
                starts_new_series=(
                    comparison is not None
                    and comparison.comparability is Comparability.INCOMPARABLE
                ),
                bridged_from_previous=(
                    comparison is not None
                    and comparison.comparability is Comparability.BRIDGED
                ),
                environment=(
                    None if execution is None else execution.environment_label()
                ),
                environment_changed=(
                    environment is not None
                    and previous_environment is not None
                    and environment != previous_environment
                ),
            )
        )
        previous_fingerprint = found.fingerprint
        if environment is not None:
            previous_environment = environment
    return MetricHistory(
        metric=definition.identifier,
        direction=definition.direction,
        points=tuple(points),
    )


def metric_column_width(identifiers: Iterable[str]) -> int:
    """Return the identifier-column width for the identifiers being rendered.

    Sized from what is present rather than fixed, because registered names
    range across more than forty characters. A column fixed to the longest
    spends that width on every report that does not contain it, and one fixed
    to the common case pushes the whole of a longer row out of alignment.
    """

    longest = max((len(identifier) for identifier in identifiers), default=0)
    return max(longest, MINIMUM_METRIC_COLUMN_WIDTH)


def render_report(report: DeltaReport) -> str:
    """Render the compact default view as text."""

    noun = "checkpoint" if report.pivot is ReportPivot.CHECKPOINT else "environment"
    lines = [
        f"Current:  {report.current.label} "
        f"({report.current.results} result(s), "
        f"{report.current.recorded_at.date().isoformat()})"
    ]
    if report.baseline is None:
        lines.append(f"Baseline: none; no earlier {noun} is recorded")
    else:
        lines.append(
            f"Baseline: {report.baseline.label} "
            f"({report.baseline.results} result(s), "
            f"{report.baseline.recorded_at.date().isoformat()})"
        )
    if report.pivot is ReportPivot.ENVIRONMENT:
        lines.append("Model held fixed; the environment is what varies.")
    width = metric_column_width(
        metric.metric for family in report.families for metric in family.metrics
    )
    lines.append("")
    lines.append(_delta_header(width))

    # Families awaiting their first metric are collapsed onto one line. They are
    # a statement about the plan rather than about this checkpoint, and giving
    # each two lines would let the default view grow with every family
    # registered ahead of the benchmark that fills it. Absences that *are*
    # about this checkpoint stay on their own line.
    unregistered: list[str] = []
    for family in report.families:
        if family.absence == UNREGISTERED_FAMILY_ABSENCE:
            unregistered.append(family.family.identifier)
            continue
        lines.append(family.family.identifier)
        if family.absence is not None:
            lines.append(f"  absent: {family.absence}")
            continue
        for group in family.series:
            lines.extend(_render_group_header(group))
            for metric in group.metrics:
                lines.append(_render_metric(metric, width))
    if unregistered:
        lines.extend(
            textwrap.wrap(
                f"awaiting a first metric: {', '.join(unregistered)}",
                width=MAXIMUM_LINE_WIDTH,
                subsequent_indent="  ",
            )
        )
    lines.extend(["", *_confounded_legend(report), *_noise_legend(report)])
    return "\n".join(lines).rstrip() + "\n"


def _render_group_header(group: SeriesGroup) -> list[str]:
    """Name the series, and the execution change every row in it shares.

    Execution belongs to the result rather than to a metric, so repeating it
    on seven rows would say the same thing seven times in the width the
    numbers need. The series line is absent when the family has one group, so
    an ordinary single-envelope benchmark renders exactly as it always did.
    """

    entries: list[str] = []
    if group.label is not None:
        entries.append(f"  [series: {group.label}]")
    for label, differences in (
        ("environment changed", group.environment),
        ("conditions changed", group.conditions),
    ):
        if not differences:
            continue
        changes = "; ".join(
            f"{difference.field} {_short(difference.baseline)} \u2192 "
            f"{_short(difference.current)}"
            for difference in differences
        )
        entries.append(f"  [{label}: {changes}]")
    return [
        line
        for entry in entries
        for line in textwrap.wrap(
            entry,
            width=MAXIMUM_LINE_WIDTH,
            subsequent_indent="   ",
        )
        or [entry]
    ]


def _confounded_legend(report: DeltaReport) -> list[str]:
    """Explain the confounded verdict, naming what actually moved."""

    confounded = [
        metric
        for family in report.families
        for metric in family.metrics
        if metric.movement is Movement.CONFOUNDED
    ]
    if not confounded:
        return []
    moved: list[str] = []
    if any(metric.conditions for metric in confounded):
        moved.append("the declared conditions")
    if any(metric.environment for metric in confounded):
        moved.append("the environment")
    if not moved:
        moved.append(
            "the environment" if report.pivot is ReportPivot.CHECKPOINT else "the model"
        )
    subject = "model" if report.pivot is ReportPivot.CHECKPOINT else "environment"
    return textwrap.wrap(
        f"confound: the delta is real and interpretable, but "
        f"{' and '.join(moved)} moved as well, so it is not a verdict on the "
        f"{subject}.",
        width=MAXIMUM_LINE_WIDTH,
        subsequent_indent="  ",
    )


def _noise_legend(report: DeltaReport) -> list[str]:
    """Explain the noise column, when any row actually carries one."""

    rows = [metric for family in report.families for metric in family.metrics]
    verdicts = {
        metric.noise
        for metric in rows
        if metric.noise is not NoiseVerdict.NOT_APPLICABLE
    }
    if not verdicts:
        return []
    legend = (
        "noise: a delta is judged against the floor combined from the spread "
        "each of the two readings measured for itself; 'within' means the "
        "delta is inside it and reads as 'unchanged' in the change column, and "
        "'unknown' means at least one of the two readings measured none."
    )
    if NoiseVerdict.CLEARED in verdicts:
        legend = (
            f"{legend} 'cleared' means larger than benchmark noise, not that "
            "the change caused it: two arms differ by their initialization "
            "seeds as well, which no floor read off a reading's own units can "
            "see."
        )
    if NoiseVerdict.UNQUALIFIABLE in verdicts:
        legend = (
            f"{legend} 'unqualifiable' means resampling that metric's units "
            "cannot estimate its dispersion, so no sampling floor can exist "
            "for it; the metric registry says why."
        )
    return textwrap.wrap(
        legend,
        width=MAXIMUM_LINE_WIDTH,
        subsequent_indent="  ",
    )


def render_history(history: MetricHistory) -> str:
    """Render one metric's history as text, marking series seams."""

    lines = [f"{history.metric} ({history.direction.value})"]
    if not history.points:
        lines.append("  no recorded values")
        return "\n".join(lines) + "\n"
    annotated = False
    for point in history.points:
        seam = "  "
        if point.starts_new_series:
            seam = "| "
        elif point.bridged_from_previous:
            seam = "~ "
        row = (
            f"  {seam}{point.recorded_at.date().isoformat()}  "
            f"{point.checkpoint:<24} {_format(point.value):>12}  "
            f"series {point.series[:12]}"
        )
        if point.environment_changed:
            annotated = True
            row = f"{row}  * now on {point.environment}"
        lines.append(row)
    lines.append("")
    legend = "  | series break    ~ bridged seam"
    if annotated:
        legend = f"{legend}    * environment changed"
    lines.append(legend)
    return "\n".join(lines) + "\n"


def render_provenance(report: DeltaReport) -> str:
    """Render how the two compared checkpoints were produced differently."""

    if not report.provenance:
        return "Provenance: no recorded differences\n"
    lines = ["Provenance differences"]
    for difference in report.provenance:
        lines.append(f"  {difference.field}")
        lines.append(f"    baseline: {difference.baseline}")
        lines.append(f"    current:  {difference.current}")
    return "\n".join(lines) + "\n"


#: Which way the metric has to move, in the width a table column allows.
_DIRECTION_LABELS = {
    MetricDirection.LOWER_IS_BETTER: "lower",
    MetricDirection.HIGHER_IS_BETTER: "higher",
    MetricDirection.INFORMATIONAL: "-",
}

#: Movement values whose enum name does not fit the change column.
_MOVEMENT_LABELS = {
    Movement.INFORMATIONAL: "-",
    Movement.CONFOUNDED: "confound",
}


def _render_metric(metric: MetricDelta, width: int) -> str:
    change = _MOVEMENT_LABELS.get(metric.movement, metric.movement.value)
    row = (
        f"  {metric.metric:<{width}} "
        f"{_DIRECTION_LABELS[metric.direction]:<6} "
        f"{_format(metric.baseline):>11} "
        f"{_format(metric.current):>11} "
        f"{_format(metric.delta, signed=True):>11}  "
        f"{change:<9}"
    )
    if metric.noise is not NoiseVerdict.NOT_APPLICABLE:
        row = f"{row} noise {metric.noise.value}"
    if metric.note is not None:
        row = f"{row.rstrip()}  ({metric.note})"
    return row.rstrip()


def _short(value: str | None) -> str:
    """Render a coordinate value, abbreviating a digest to a readable prefix.

    A model or dataset coordinate is a full hex digest. Printed whole it pushes
    the line that explains a confound past any terminal, so the prefix stands
    in; the machine-readable record keeps the full value.
    """

    if value is None:
        return "-"
    if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
        return value[:12]
    return value


def _format(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    if signed and value == 0.0:
        return "0"
    return f"{value:+.6g}" if signed else f"{value:.6g}"


def _selected_metrics(
    families: Sequence[str] | None,
    metrics: Sequence[str] | None,
) -> frozenset[str]:
    if metrics is not None:
        chosen = set()
        for identifier in metrics:
            try:
                chosen.add(metric_definition(identifier).identifier)
            except MetricRegistryError as error:
                raise ReportError(str(error)) from error
        return frozenset(chosen)
    if families is not None:
        try:
            return frozenset(
                definition.identifier
                for family in families
                for definition in registered_metrics(family)
            )
        except MetricRegistryError as error:
            raise ReportError(str(error)) from error
    return frozenset(definition.identifier for definition in registered_metrics())


def _family_report(
    family: MetricFamily,
    definitions: Sequence[MetricDefinition],
    current_results: Sequence[ResultEnvelope],
    baseline_results: Sequence[ResultEnvelope],
    bridges: BridgeIndex,
    current_label: str,
    baseline_label: str | None,
    pivot: ReportPivot,
) -> FamilyReport:
    if not definitions:
        return FamilyReport(
            family=family,
            series=(),
            absence=UNREGISTERED_FAMILY_ABSENCE,
        )

    current_readings = {
        definition.identifier: measurements_by_workload(
            current_results,
            definition.identifier,
            workload_scoped=definition.execution_sensitive,
        )
        for definition in definitions
    }
    baseline_readings = {
        definition.identifier: measurements_by_workload(
            baseline_results,
            definition.identifier,
            workload_scoped=definition.execution_sensitive,
        )
        for definition in definitions
    }

    pairings = _group_pairings(current_readings, baseline_readings)
    workloads = _group_workloads(pairings, current_readings, baseline_readings)
    labels = _series_labels(workloads)
    groups: list[SeriesGroup] = []
    for key in _ordered_groups(labels):
        rows: list[MetricDelta] = []
        environment: tuple[ProvenanceDifference, ...] = ()
        conditions: tuple[ProvenanceDifference, ...] = ()
        for definition in definitions:
            current = current_readings[definition.identifier].get(key)
            baseline = baseline_readings[definition.identifier].get(pairings[key])
            if current is None and baseline is None:
                continue
            row = _metric_delta(
                definition,
                baseline=baseline,
                current=current,
                bridges=bridges,
                current_label=current_label,
                baseline_label=baseline_label,
                pivot=pivot,
                hidden_series=_hidden_series(
                    current_results,
                    definition,
                    bridges,
                ),
            )
            if row.environment:
                environment = row.environment
            if row.conditions:
                conditions = row.conditions
            rows.append(row)
        if not rows:
            continue
        groups.append(
            SeriesGroup(
                workload=None if key == UNSCOPED_WORKLOAD else key,
                # One group has nothing to be told apart from, so the family
                # renders exactly as it did before any benchmark wrote a matrix.
                label=labels[key] if len(labels) > 1 else None,
                metrics=tuple(rows),
                coordinates=workloads.get(key, ()),
                environment=environment,
                conditions=conditions,
            )
        )
    if not groups:
        return FamilyReport(
            family=family,
            series=(),
            absence=f"no result recorded for {current_label}",
        )
    return FamilyReport(family=family, series=tuple(groups), absence=None)


Readings = Mapping[str, Mapping[str, tuple[ResultEnvelope, Measurement]]]


def _hidden_series(
    results: Sequence[ResultEnvelope],
    definition: MetricDefinition,
    bridges: BridgeIndex,
) -> int:
    """Return how many series one checkpoint recorded a metric on.

    Only a metric that stayed in one group can be hiding a second series behind
    the reading shown, since a workload-scoped one has already been split into
    a group per series.
    """

    if definition.execution_sensitive:
        return 1
    return len(recorded_series(results, definition.identifier, bridges))


def _group_pairings(
    current_readings: Readings,
    baseline_readings: Readings,
) -> dict[str, str]:
    """Return each group of a family, and the baseline workload it reads.

    Normally a workload identifies itself on both sides and the pairing is the
    identity. The exception is the family measured under one workload before a
    change and one after: pairing those two is what lets the report say the
    declared workload moved, rather than reporting two half-rows that never
    name the cause. It is only applied where exactly one workload is unmatched
    on each side, since anything looser would be guessing which cell of a
    matrix succeeded which.
    """

    current_keys = _workload_keys(current_readings)
    baseline_keys = _workload_keys(baseline_readings)
    shared = current_keys & baseline_keys
    current_only = sorted(current_keys - shared)
    baseline_only = sorted(baseline_keys - shared)
    if len(current_only) == 1 and len(baseline_only) == 1:
        return {key: key for key in sorted(shared)} | {
            current_only[0]: baseline_only[0]
        }
    return {key: key for key in sorted(current_keys | baseline_keys)}


def _workload_keys(readings: Readings) -> set[str]:
    return {key for by_workload in readings.values() for key in by_workload}


def _group_workloads(
    pairings: Mapping[str, str],
    current_readings: Readings,
    baseline_readings: Readings,
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Return the flattened declared workload behind each group of a family.

    The current side is preferred where both recorded a group, so the label a
    reader sees names the run they are asking about.
    """

    recorded: dict[str, tuple[tuple[str, str], ...]] = {}
    for readings in (baseline_readings, current_readings):
        for by_workload in readings.values():
            for key, (envelope, _) in by_workload.items():
                execution = envelope.execution
                if key == UNSCOPED_WORKLOAD or execution is None:
                    recorded[key] = ()
                    continue
                recorded[key] = _flatten(execution.workload)
    return {
        key: recorded.get(key, recorded.get(baseline_key, ()))
        for key, baseline_key in pairings.items()
    }


def _series_labels(
    workloads: Mapping[str, tuple[tuple[str, str], ...]],
) -> dict[str, str]:
    """Name each group by the workload fields that actually tell them apart.

    Rendering the whole declared workload would put a dozen fields above every
    cell of a matrix that varies two of them. Only the fields that differ carry
    information, so only those are shown; the envelope keeps the rest.
    """

    scoped = {key: dict(fields) for key, fields in workloads.items() if key}
    fields = sorted({field for entry in scoped.values() for field in entry})
    differing = [
        field
        for field in fields
        if len({entry.get(field) for entry in scoped.values()}) > 1
    ]
    labels: dict[str, str] = {}
    for key in workloads:
        if key == UNSCOPED_WORKLOAD:
            labels[key] = UNSCOPED_SERIES_LABEL
            continue
        entry = scoped[key]
        rendered = " ".join(
            f"{field}={_clip(entry.get(field, '-'))}" for field in differing
        )
        # Two workloads with the same fields cannot share a digest, so an empty
        # label means the only difference is in a field one of them omits
        # entirely. The digest prefix is then the honest discriminator.
        labels[key] = rendered or f"workload {key[:12]}"
    return labels


def _ordered_groups(labels: Mapping[str, str]) -> tuple[str, ...]:
    """Order a family's groups so a report does not depend on record order."""

    return tuple(sorted(labels, key=lambda key: (labels[key], key)))


def _flatten(workload: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    """Render a declared workload as comparable dotted scalar fields.

    Flattened rather than compared whole, because a nested position source that
    differs in one field would otherwise label a group with the entire mapping.
    """

    flattened: list[tuple[str, str]] = []

    def walk(prefix: str, value: object) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                walk(f"{prefix}.{key}" if prefix else str(key), value[key])
            return
        flattened.append((prefix, _workload_value(value)))

    walk("", workload)
    return tuple(flattened)


def _workload_value(value: object) -> str:
    """Render one workload field as a string two groups can be compared on."""

    if isinstance(value, str):
        return value
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, int | float):
        return f"{value:g}"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _clip(value: str) -> str:
    """Shorten one workload value to the width a series label allows."""

    if len(value) <= WORKLOAD_VALUE_WIDTH:
        return value
    return f"{value[: WORKLOAD_VALUE_WIDTH - 1]}…"


def _difference_records(
    differences: Sequence[ProvenanceDifference],
) -> list[dict[str, str | None]]:
    """Return the machine-readable form of a provenance difference list."""

    return [
        {
            "field": difference.field,
            "baseline": difference.baseline,
            "current": difference.current,
        }
        for difference in differences
    ]


def _metric_delta(
    definition: MetricDefinition,
    *,
    baseline: tuple[ResultEnvelope, Measurement] | None,
    current: tuple[ResultEnvelope, Measurement] | None,
    bridges: BridgeIndex,
    current_label: str,
    baseline_label: str | None,
    pivot: ReportPivot,
    hidden_series: int = 1,
) -> MetricDelta:
    if current is None:
        assert baseline is not None  # the caller skips a metric with neither
        return _incomparable_delta(
            definition,
            baseline=baseline[1].value,
            current=None,
            comparability=Comparability.INCOMPARABLE,
            note=f"not measured for {current_label}",
            series=bridges.series(baseline[1].fingerprint),
        )
    if baseline is None:
        return _incomparable_delta(
            definition,
            baseline=None,
            current=current[1].value,
            comparability=Comparability.INCOMPARABLE,
            note=_annotations(
                "no baseline recorded"
                if baseline_label is None
                else f"not measured for {baseline_label}",
                _hidden_note(hidden_series),
            ),
            series=bridges.series(current[1].fingerprint),
        )

    baseline_envelope, baseline_measurement = baseline
    current_envelope, current_measurement = current
    comparison = bridges.compare_measurements(
        baseline_measurement,
        current_measurement,
    )
    attribution = (
        attribute(baseline_envelope, current_envelope)
        if definition.execution_sensitive
        else None
    )
    if not comparison.is_comparable:
        # For an efficiency metric this can only be a workload change, since
        # the environment is not in the fingerprint. That is the case where
        # the delta really is meaningless rather than merely confounded.
        return _incomparable_delta(
            definition,
            baseline=baseline_measurement.value,
            current=current_measurement.value,
            comparability=comparison.comparability,
            note=_annotations(
                "different measurement; the declared workload changed"
                if attribution is not None
                and attribution.workload is AxisChange.CHANGED
                else "incomparable; these results are not on the same series",
                _hidden_note(hidden_series),
            ),
            attribution=attribution,
            series=bridges.series(current_measurement.fingerprint),
        )

    conditions = (
        condition_differences(baseline_envelope, current_envelope)
        if definition.execution_sensitive
        else ()
    )
    delta = current_measurement.value - baseline_measurement.value
    floor, floor_source = _delta_floor(
        definition, baseline_measurement, current_measurement
    )
    environment = (
        environment_differences(baseline_envelope, current_envelope)
        if definition.execution_sensitive
        else ()
    )
    noise = _noise_verdict(definition, delta, floor)
    return MetricDelta(
        metric=definition.identifier,
        family=definition.family,
        direction=definition.direction,
        baseline=baseline_measurement.value,
        current=current_measurement.value,
        delta=delta,
        comparability=comparison.comparability,
        movement=_pivoted_movement(definition, delta, attribution, pivot, noise),
        noise=noise,
        noise_floor=floor,
        noise_floor_source=floor_source,
        bridges=tuple(bridge.bridge_id for bridge in comparison.bridges),
        attribution=attribution,
        environment=environment,
        conditions=conditions,
        series=bridges.series(current_measurement.fingerprint),
        note=_annotations(
            "bridged series seam"
            if comparison.comparability is Comparability.BRIDGED
            else None,
            _hidden_note(hidden_series),
        ),
    )


def _annotations(*notes: str | None) -> str | None:
    """Join the annotations a row carries, or ``None`` when it carries none."""

    return "; ".join(note for note in notes if note is not None) or None


def _hidden_note(hidden_series: int) -> str | None:
    """Say when the row shown is one of several series recorded.

    A metric that declares no workload still ends up on more than one series
    when the inputs underneath it change — a regenerated pool is the usual
    cause. There is nothing to compare across those series, so the most recent
    is the right one to show; what would be wrong is showing it as though it
    were the only one.
    """

    if hidden_series <= 1:
        return None
    return (
        f"{hidden_series} series recorded for this checkpoint; showing the most recent"
    )


def _incomparable_delta(
    definition: MetricDefinition,
    *,
    baseline: float | None,
    current: float | None,
    comparability: Comparability,
    note: str | None,
    series: str | None = None,
    attribution: Attribution | None = None,
) -> MetricDelta:
    """Return a row with no delta, and therefore nothing to judge against noise.

    Reserved for a delta that carries no meaning at all: a metric measured
    over different games, or an efficiency metric measured under a different
    workload. A merely confounded delta is reported with its value, because it
    does mean something.
    """

    return MetricDelta(
        metric=definition.identifier,
        family=definition.family,
        direction=definition.direction,
        baseline=baseline,
        current=current,
        delta=None,
        comparability=comparability,
        movement=Movement.UNKNOWN,
        noise=NoiseVerdict.NOT_APPLICABLE,
        noise_floor=None,
        noise_floor_source=None,
        bridges=(),
        note=note,
        attribution=attribution,
        series=series,
    )


def _pivoted_movement(
    definition: MetricDefinition,
    delta: float,
    attribution: Attribution | None,
    pivot: ReportPivot,
    noise: NoiseVerdict,
) -> Movement:
    """Return the verdict, given what the report holds fixed.

    The checkpoint pivot asks whether the *model* improved, so any environment
    movement makes that unanswerable. The environment pivot asks whether the
    *environment* is faster with the model pinned, so there the environment
    moving is the point and the model moving is what would confound it.

    Declared conditions confound either pivot. A corpus regenerated underneath
    a comparison moves the number without answering either question, and it is
    the confounder most likely to pass unnoticed, because nothing about the
    machine or the checkpoint label changed.
    """

    if attribution is None:
        return _movement(definition.direction, delta, noise)
    if pivot is ReportPivot.CHECKPOINT:
        # The environment has to have provably held still; an unverifiable
        # claim of sameness is exactly what must not pass. A condition change
        # confounds too, and is the one most likely to go unnoticed.
        if attribution.environment is not AxisChange.UNCHANGED:
            return Movement.CONFOUNDED
        if attribution.conditions is AxisChange.CHANGED:
            return Movement.CONFOUNDED
        return _movement(definition.direction, delta, noise)

    # The environment pivot asks whether the machine got faster, so what has to
    # hold still is the thing being measured on it. Where a benchmark declares
    # its conditions those are that thing, because two runs of one training
    # configuration never share weights and pinning on the parameter digest
    # would make the question unanswerable rather than rigorous.
    held = (
        attribution.conditions
        if attribution.conditions is not AxisChange.UNKNOWN
        else attribution.model
    )
    if held is not AxisChange.UNCHANGED:
        return Movement.CONFOUNDED
    return _movement(definition.direction, delta, noise)


def _delta_floor(
    definition: MetricDefinition,
    baseline: Measurement,
    current: Measurement,
) -> tuple[float | None, str | None]:
    """Return the floor this delta faces, and how the two readings described it.

    Both readings have to carry a dispersion. One that only one side measured
    describes that operand rather than the difference, and nothing here says the
    side carrying none is the narrower, so the delta is reported as unqualified
    rather than floored by half of itself.

    A metric that declares why resampling its units cannot estimate its
    dispersion is refused a floor outright, rather than only a resampled one.
    The declaration says the quantity has no spread worth stating, so a floor
    from any estimator would answer a question the metric does not pose.

    Two readings that describe themselves identically are named once, which is
    what the ordered deduplication is for.
    """

    if definition.no_sampling_floor_reason is not None:
        return None, None
    if baseline.dispersion is None or current.dispersion is None:
        return None, None
    named = [
        source
        for source in (baseline.dispersion.source, current.dispersion.source)
        if source is not None
    ]
    return (
        combined_floor(baseline.dispersion, current.dispersion),
        " + ".join(dict.fromkeys(named)) or None,
    )


def _movement(
    direction: MetricDirection,
    delta: float,
    noise: NoiseVerdict,
) -> Movement:
    """Return the verdict on the delta, which noise can withdraw.

    A delta inside its floor is not a smaller improvement than one that clears
    it; it is a number that has not been shown to have moved. Reporting the sign
    of it as ``better`` is how a row comes to read ``better`` and ``within`` at
    once, which is the reading that sends somebody to defend a change the
    benchmark cannot distinguish from its own noise.
    """

    if direction is MetricDirection.INFORMATIONAL:
        return Movement.INFORMATIONAL
    if noise is NoiseVerdict.WITHIN or math.isclose(
        delta, 0.0, abs_tol=0.0, rel_tol=0.0
    ):
        return Movement.UNCHANGED
    improved = (
        delta < 0.0 if direction is MetricDirection.LOWER_IS_BETTER else delta > 0.0
    )
    return Movement.BETTER if improved else Movement.WORSE


def _noise_verdict(
    definition: MetricDefinition,
    delta: float,
    floor: float | None,
) -> NoiseVerdict:
    """Judge one delta, distinguishing a missing floor from an impossible one."""

    if floor is None:
        if definition.no_sampling_floor_reason is not None:
            return NoiseVerdict.UNQUALIFIABLE
        return NoiseVerdict.UNKNOWN
    if abs(delta) <= floor:
        return NoiseVerdict.WITHIN
    return NoiseVerdict.CLEARED


def _selection(
    label: str,
    results: Sequence[ResultEnvelope],
) -> CheckpointSelection:
    recorded = max(envelope.recorded_at for envelope in results)
    return CheckpointSelection(label=label, recorded_at=recorded, results=len(results))


def _report_provenance(
    baseline_results: Sequence[ResultEnvelope],
    current_results: Sequence[ResultEnvelope],
) -> tuple[ProvenanceDifference, ...]:
    if not baseline_results or not current_results:
        return ()
    baseline = _newest(baseline_results)
    current = _newest(current_results)
    return provenance_differences(baseline, current)


def _newest(results: Iterable[ResultEnvelope]) -> ResultEnvelope:
    return max(results, key=lambda envelope: (envelope.recorded_at, envelope.result_id))


def _by_environment(
    results: Sequence[ResultEnvelope],
) -> dict[tuple[tuple[str, str | None], ...], list[ResultEnvelope]]:
    """Group results by the environment coordinates they were measured in."""

    groups: dict[tuple[tuple[str, str | None], ...], list[ResultEnvelope]] = {}
    for envelope in results:
        assert envelope.execution is not None  # filtered by the caller
        key = tuple(sorted(envelope.execution.environment().items()))
        groups.setdefault(key, []).append(envelope)
    return groups


def _latest_multi_environment(results: Sequence[ResultEnvelope]) -> str:
    """Return the most recent checkpoint measured in more than one environment."""

    by_label: dict[str, list[ResultEnvelope]] = {}
    for envelope in results:
        by_label.setdefault(envelope.checkpoint.label, []).append(envelope)
    candidates = [
        (max(group, key=lambda item: item.recorded_at).recorded_at, label)
        for label, group in by_label.items()
        if len(_by_environment(group)) > 1
    ]
    if not candidates:
        raise ReportError(
            "no checkpoint has been measured in more than one environment yet; "
            "record the same checkpoint elsewhere to compare them"
        )
    return max(candidates)[1]


def _require_one_model(results: Sequence[ResultEnvelope], label: str) -> None:
    """Reject a comparison whose two sides are not the same model.

    Pinned by ``parameter_sha256`` where the benchmark scores one checkpoint on
    two machines, because a reused label would otherwise turn a model change
    into an apparent hardware win.

    A benchmark that declares conditions pins on those instead. Training
    efficiency is the case: the same configuration trained on two machines
    produces two different sets of weights, so requiring identical parameters
    would make the environment question unaskable rather than rigorous. The
    declared architecture and corpus are what has to hold still there, and they
    are checked exactly as strictly.
    """

    declared = [
        envelope.execution.declared_coordinates()
        for envelope in results
        if envelope.execution is not None and envelope.execution.coordinates
    ]
    if declared:
        if len(declared) != len(results):
            raise ReportError(
                f"checkpoint {label!r} mixes results that declare conditions "
                "with results that do not, so an environment comparison cannot "
                "prove they were measured under the same ones"
            )
        if any(entry != declared[0] for entry in declared[1:]):
            raise ReportError(
                f"checkpoint {label!r} covers more than one set of declared "
                "conditions; an environment comparison needs them held fixed"
            )
        return

    digests = {envelope.checkpoint.parameter_sha256 for envelope in results}
    if None in digests:
        raise ReportError(
            f"checkpoint {label!r} has a result with no parameter digest, so "
            "an environment comparison cannot prove the model was held fixed"
        )
    if len(digests) > 1:
        raise ReportError(
            f"checkpoint label {label!r} covers more than one set of weights; "
            "an environment comparison needs the model held fixed"
        )


def _environment_name(results: Sequence[ResultEnvelope]) -> str:
    execution = _newest(results).execution
    assert execution is not None  # filtered by the caller
    return execution.environment_label()


def _environment_selection(results: Sequence[ResultEnvelope]) -> CheckpointSelection:
    return CheckpointSelection(
        label=_environment_name(results),
        recorded_at=max(envelope.recorded_at for envelope in results),
        results=len(results),
    )
