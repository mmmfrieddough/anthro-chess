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

import math
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from anthro_chess.evaluation.results.comparability import (
    BridgeIndex,
    Comparability,
    ProvenanceDifference,
    latest_measurement,
    provenance_differences,
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
from anthro_chess.evaluation.results.noise import NoiseFloorIndex
from anthro_chess.evaluation.results.records import (
    Measurement,
    NoiseFloor,
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


class ReportError(ValueError):
    """Raised when a report cannot be built from the requested selection."""


class Movement(StrEnum):
    """Whether a delta is an improvement, judged only by declared direction."""

    BETTER = "better"
    WORSE = "worse"
    UNCHANGED = "unchanged"
    INFORMATIONAL = "informational"
    UNKNOWN = "unknown"


class NoiseVerdict(StrEnum):
    """Whether a delta is larger than the noise in the measurement."""

    CLEARED = "cleared"
    WITHIN = "within"
    UNKNOWN = "unknown"
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
    #: The binding floor: the largest of the characterized floors that apply,
    #: because a delta has to clear every noise source to be a finding.
    noise_floor: float | None
    noise_floor_kind: str | None
    #: Every applicable floor, so a reader who knows which noise source their
    #: comparison is actually exposed to can read past the binding one.
    noise_floors: tuple[NoiseFloor, ...]
    bridges: tuple[str, ...]
    note: str | None

    def as_record(self) -> dict[str, object]:
        """Return the machine-readable row."""

        return {
            "metric": self.metric,
            "family": self.family,
            "direction": self.direction.value,
            "baseline": self.baseline,
            "current": self.current,
            "delta": self.delta,
            "comparability": self.comparability.value,
            "movement": self.movement.value,
            "noise": self.noise.value,
            "noise_floor": self.noise_floor,
            "noise_floor_kind": self.noise_floor_kind,
            "noise_floors": [
                floor.model_dump(mode="json") for floor in self.noise_floors
            ],
            "bridges": list(self.bridges),
            "note": self.note,
        }


@dataclass(frozen=True)
class FamilyReport:
    """One family's rows, or the reason it has none."""

    family: MetricFamily
    metrics: tuple[MetricDelta, ...]
    absence: str | None

    def as_record(self) -> dict[str, object]:
        """Return the machine-readable family section."""

        return {
            "family": self.family.identifier,
            "title": self.family.title,
            "absence": self.absence,
            "metrics": [metric.as_record() for metric in self.metrics],
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

    def as_record(self) -> dict[str, object]:
        """Return the machine-readable report."""

        return {
            "baseline": None if self.baseline is None else self.baseline.as_record(),
            "current": self.current.as_record(),
            "families": [family.as_record() for family in self.families],
            "provenance": [
                {
                    "field": difference.field,
                    "baseline": difference.baseline,
                    "current": difference.current,
                }
                for difference in self.provenance
            ],
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
    floors: NoiseFloorIndex | None = None,
    current: str | None = None,
    baseline: str | None = None,
    families: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
) -> DeltaReport:
    """Build the default compact view between two recorded checkpoints."""

    resolved_floors = floors if floors is not None else NoiseFloorIndex()
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
                resolved_floors,
                current_label,
                baseline_label,
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
            )
        )
        previous_fingerprint = found.fingerprint
    return MetricHistory(
        metric=definition.identifier,
        direction=definition.direction,
        points=tuple(points),
    )


def render_report(report: DeltaReport) -> str:
    """Render the compact default view as text."""

    lines = [
        f"Current:  {report.current.label} "
        f"({report.current.results} result(s), "
        f"{report.current.recorded_at.date().isoformat()})"
    ]
    if report.baseline is None:
        lines.append("Baseline: none; no earlier checkpoint is recorded")
    else:
        lines.append(
            f"Baseline: {report.baseline.label} "
            f"({report.baseline.results} result(s), "
            f"{report.baseline.recorded_at.date().isoformat()})"
        )
    lines.append("")
    lines.append(
        f"  {'metric':<38} {'better':<6} "
        f"{'baseline':>11} {'current':>11} {'delta':>11}  {'change':<9} noise"
    )

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
        for metric in family.metrics:
            lines.append(_render_metric(metric))
    if unregistered:
        lines.extend(
            textwrap.wrap(
                f"awaiting a first metric: {', '.join(unregistered)}",
                width=MAXIMUM_LINE_WIDTH,
                subsequent_indent="  ",
            )
        )
    lines.extend(["", *_noise_legend(report)])
    return "\n".join(lines).rstrip() + "\n"


def _noise_legend(report: DeltaReport) -> list[str]:
    """Explain the noise column, when any row actually carries one."""

    verdicts = {
        metric.noise
        for family in report.families
        for metric in family.metrics
        if metric.noise is not NoiseVerdict.NOT_APPLICABLE
    }
    if not verdicts:
        return []
    return textwrap.wrap(
        "noise: a delta is judged against the widest characterized floor that "
        "applies, named in the column; 'within' means the delta is inside it "
        "and 'unknown' means no floor is characterized for that series.",
        width=MAXIMUM_LINE_WIDTH,
        subsequent_indent="  ",
    )


def render_history(history: MetricHistory) -> str:
    """Render one metric's history as text, marking series seams."""

    lines = [f"{history.metric} ({history.direction.value})"]
    if not history.points:
        lines.append("  no recorded values")
        return "\n".join(lines) + "\n"
    for point in history.points:
        seam = "  "
        if point.starts_new_series:
            seam = "| "
        elif point.bridged_from_previous:
            seam = "~ "
        lines.append(
            f"  {seam}{point.recorded_at.date().isoformat()}  "
            f"{point.checkpoint:<24} {_format(point.value):>12}  "
            f"series {point.series[:12]}"
        )
    lines.append("")
    lines.append("  | series break    ~ bridged seam")
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

#: Stored kind names, in the width the noise column allows. The machine-readable
#: row carries the unabbreviated kind and the floor's value.
_NOISE_KIND_LABELS = {
    "evaluation": "eval",
    "data-sampling": "sampling",
    "training": "training",
}


def _render_metric(metric: MetricDelta) -> str:
    change = "-" if metric.movement is Movement.INFORMATIONAL else metric.movement.value
    row = (
        f"  {metric.metric:<38} "
        f"{_DIRECTION_LABELS[metric.direction]:<6} "
        f"{_format(metric.baseline):>11} "
        f"{_format(metric.current):>11} "
        f"{_format(metric.delta, signed=True):>11}  "
        f"{change:<9}"
    )
    if metric.noise is not NoiseVerdict.NOT_APPLICABLE:
        row = f"{row} noise {_render_noise(metric)}"
    if metric.note is not None:
        row = f"{row.rstrip()}  ({metric.note})"
    return row.rstrip()


def _render_noise(metric: MetricDelta) -> str:
    """Render the verdict together with the noise source it was judged against.

    A bare "within" is unreadable without naming the source that produced the
    floor, because the three are not interchangeable and a reader has to know
    whether their comparison is even exposed to it.
    """

    if metric.noise_floor_kind is None:
        return metric.noise.value
    kind = _NOISE_KIND_LABELS.get(metric.noise_floor_kind, metric.noise_floor_kind)
    return f"{metric.noise.value} ({kind})"


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
    floors: NoiseFloorIndex,
    current_label: str,
    baseline_label: str | None,
) -> FamilyReport:
    if not definitions:
        return FamilyReport(
            family=family,
            metrics=(),
            absence=UNREGISTERED_FAMILY_ABSENCE,
        )

    rows: list[MetricDelta] = []
    for definition in definitions:
        current = latest_measurement(current_results, definition.identifier)
        baseline = latest_measurement(baseline_results, definition.identifier)
        if current is None and baseline is None:
            continue
        rows.append(
            _metric_delta(
                definition,
                baseline=None if baseline is None else baseline[1],
                current=None if current is None else current[1],
                bridges=bridges,
                floors=floors,
                current_label=current_label,
                baseline_label=baseline_label,
            )
        )
    if not rows:
        return FamilyReport(
            family=family,
            metrics=(),
            absence=f"no result recorded for {current_label}",
        )
    return FamilyReport(family=family, metrics=tuple(rows), absence=None)


def _metric_delta(
    definition: MetricDefinition,
    *,
    baseline: Measurement | None,
    current: Measurement | None,
    bridges: BridgeIndex,
    floors: NoiseFloorIndex,
    current_label: str,
    baseline_label: str | None,
) -> MetricDelta:
    if current is None:
        return _incomparable_delta(
            definition,
            baseline=None if baseline is None else baseline.value,
            current=None,
            comparability=Comparability.INCOMPARABLE,
            note=f"not measured for {current_label}",
        )
    if baseline is None:
        return _incomparable_delta(
            definition,
            baseline=None,
            current=current.value,
            comparability=Comparability.INCOMPARABLE,
            note=(
                "no baseline recorded"
                if baseline_label is None
                else f"not measured for {baseline_label}"
            ),
        )

    comparison = bridges.compare_measurements(baseline, current)
    if not comparison.is_comparable:
        return _incomparable_delta(
            definition,
            baseline=baseline.value,
            current=current.value,
            comparability=comparison.comparability,
            note="incomparable; these results are not on the same series",
        )

    delta = current.value - baseline.value
    applicable = _applicable_floors(definition.identifier, baseline, current, floors)
    binding = max(applicable, key=lambda floor: floor.value, default=None)
    return MetricDelta(
        metric=definition.identifier,
        family=definition.family,
        direction=definition.direction,
        baseline=baseline.value,
        current=current.value,
        delta=delta,
        comparability=comparison.comparability,
        movement=_movement(definition.direction, delta),
        noise=_noise_verdict(delta, None if binding is None else binding.value),
        noise_floor=None if binding is None else binding.value,
        noise_floor_kind=None if binding is None else binding.kind,
        noise_floors=applicable,
        bridges=tuple(bridge.bridge_id for bridge in comparison.bridges),
        note=(
            "bridged series seam"
            if comparison.comparability is Comparability.BRIDGED
            else None
        ),
    )


def _incomparable_delta(
    definition: MetricDefinition,
    *,
    baseline: float | None,
    current: float | None,
    comparability: Comparability,
    note: str,
) -> MetricDelta:
    """Return a row with no delta, and therefore nothing to judge against noise."""

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
        noise_floor_kind=None,
        noise_floors=(),
        bridges=(),
        note=note,
    )


def _applicable_floors(
    metric: str,
    baseline: Measurement,
    current: Measurement,
    floors: NoiseFloorIndex,
) -> tuple[NoiseFloor, ...]:
    """Return one floor per noise kind that applies to this comparison.

    Two sources can supply a floor. A benchmark whose floor is a function of
    its own configuration attaches it to the measurement, which is the only
    place it can be right; a calibration pass characterizes a floor for the
    whole series. Where both exist for one kind, the wider one is kept, since
    a floor that understates the noise is worse than one that overstates it.
    """

    widest: dict[str, NoiseFloor] = {}
    candidates = [
        floor
        for floor in (current.noise_floor, baseline.noise_floor)
        if floor is not None
    ]
    candidates.extend(floors.floors(metric, current.fingerprint))
    for floor in candidates:
        existing = widest.get(floor.kind)
        if existing is None or floor.value > existing.value:
            widest[floor.kind] = floor
    return tuple(widest[kind] for kind in sorted(widest))


def _movement(direction: MetricDirection, delta: float) -> Movement:
    if direction is MetricDirection.INFORMATIONAL:
        return Movement.INFORMATIONAL
    if math.isclose(delta, 0.0, abs_tol=0.0, rel_tol=0.0):
        return Movement.UNCHANGED
    improved = (
        delta < 0.0 if direction is MetricDirection.LOWER_IS_BETTER else delta > 0.0
    )
    return Movement.BETTER if improved else Movement.WORSE


def _noise_verdict(delta: float, floor: float | None) -> NoiseVerdict:
    if floor is None:
        return NoiseVerdict.UNKNOWN
    return NoiseVerdict.CLEARED if abs(delta) > floor else NoiseVerdict.WITHIN


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
