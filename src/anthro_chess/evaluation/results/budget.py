"""Quality against the budget that bought it.

A throughput ranking is not a training-efficiency verdict. A step that runs
twice as fast while learning less per position is a regression that every
efficiency metric in isolation reports as a win, so the question worth asking
is what held-out quality a run reached for a given number of processed
positions or a given amount of wall clock.

That question spans two families — ``training-efficiency`` supplies the budget
axes and ``held-out-prediction`` supplies the quality — so this is a **report
joining two families** rather than a third family duplicating both. It is a
view over the store like every other report, and it needs no benchmark of its
own: the points already exist, written by the run as it trained and by the
cadence readings taken alongside them.

The join is by checkpoint label, which is why
:func:`anthro_chess.evaluation.results.records.default_checkpoint_label` has to
be the one name every reading of the same parameters agrees on.

Two comparability rules apply, and both are refusals rather than annotations,
because a curve is read as one line:

- every point's quality must sit on one series, so a view or pool change
  cannot masquerade as learning; and
- every point's efficiency must sit on one declared workload, so a batch-size
  change cannot masquerade as a faster machine.

The environment is not a refusal. It is recorded per point and surfaced, so a
curve measured partly on a laptop is legible as such rather than incomparable,
which is the same posture decision 0018 takes everywhere else.

The two budget axes survive a resume differently, and the report says so rather
than letting the difference pass. Processed positions are checkpointed, so they
accumulate across a restart; wall clock is not, because a resumed run starts a
new process. A curve whose seconds fall while its positions rise has crossed a
resume, which is reported, and a wall-clock budget over such a curve is refused
because it has no single span to be measured against.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from anthro_chess.evaluation.results.comparability import ProvenanceDifference
from anthro_chess.evaluation.results.metrics import (
    MetricDefinition,
    MetricDirection,
    MetricRegistryError,
    metric_definition,
)
from anthro_chess.evaluation.results.records import Measurement, ResultEnvelope
from anthro_chess.evaluation.results.reporting import ReportError

#: The budget axes a point carries. Both are recorded by the same efficiency
#: reading, so a point either has both or is not a point.
PROCESSED_POSITIONS_METRIC = "training.processed_positions"
TRAINING_SECONDS_METRIC = "training.training_seconds"

#: The quality metric a budget report reads unless the caller names another.
DEFAULT_QUALITY_METRIC = "held_out.move_loss"


@dataclass(frozen=True)
class BudgetPoint:
    """One checkpoint's quality and the budget that reached it."""

    checkpoint: str
    recorded_at: datetime
    processed_positions: int
    training_seconds: float
    quality: float
    #: Where the training that produced this point ran, when the efficiency
    #: reading carried an execution record.
    environment: str | None = None
    view: str | None = None

    def as_record(self) -> dict[str, object]:
        """Return the machine-readable point."""

        return {
            "checkpoint": self.checkpoint,
            "recorded_at": self.recorded_at.isoformat(),
            "processed_positions": self.processed_positions,
            "training_seconds": self.training_seconds,
            "quality": self.quality,
            "environment": self.environment,
            "view": self.view,
        }


@dataclass(frozen=True)
class BudgetAnswer:
    """The best quality a run reached within one declared budget."""

    #: ``positions`` or ``seconds``.
    axis: str
    budget: float
    point: BudgetPoint | None

    def as_record(self) -> dict[str, object]:
        """Return the machine-readable answer."""

        return {
            "axis": self.axis,
            "budget": self.budget,
            "point": None if self.point is None else self.point.as_record(),
        }


@dataclass(frozen=True)
class BudgetReport:
    """A whole quality-versus-budget view over one run's recorded points."""

    metric: str
    direction: MetricDirection
    points: tuple[BudgetPoint, ...]
    answers: tuple[BudgetAnswer, ...]
    environment_changes: tuple[ProvenanceDifference, ...]
    #: Checkpoints where the training clock restarted. Processed positions
    #: survive a resume because the counter is checkpointed; wall clock does
    #: not, because a resumed process starts a new one. The positions axis
    #: stays whole across a resume and the time axis does not.
    clock_restarts: tuple[str, ...] = ()

    def as_record(self) -> dict[str, object]:
        """Return the machine-readable report."""

        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "points": [point.as_record() for point in self.points],
            "answers": [answer.as_record() for answer in self.answers],
            "clock_restarts": list(self.clock_restarts),
            "environment_changes": [
                {
                    "field": difference.field,
                    "baseline": difference.baseline,
                    "current": difference.current,
                }
                for difference in self.environment_changes
            ],
        }


def build_budget_report(
    results: Sequence[ResultEnvelope],
    *,
    metric: str = DEFAULT_QUALITY_METRIC,
    position_budgets: Sequence[int] = (),
    time_budgets: Sequence[float] = (),
) -> BudgetReport:
    """Join efficiency budgets to held-out quality, by checkpoint label."""

    definition = _quality_definition(metric)
    efficiency = _efficiency_by_checkpoint(results)
    quality = _quality_by_checkpoint(results, definition)
    labels = sorted(set(efficiency) & set(quality))
    if not labels:
        raise ReportError(
            f"no checkpoint carries both a training-efficiency reading and "
            f"{definition.identifier}; a budget report needs the run to have "
            "recorded efficiency alongside its cadence readings"
        )

    _require_one_series(
        [quality[label][1].fingerprint for label in labels],
        (
            f"{definition.identifier} was measured on more than one series, so "
            "these points do not describe one curve"
        ),
    )
    _require_one_series(
        [_workload(efficiency[label][0]) for label in labels],
        (
            "the efficiency readings declare more than one workload, so these "
            "points do not describe one run configuration"
        ),
    )

    points = tuple(
        sorted(
            (_point(label, efficiency[label], quality[label]) for label in labels),
            key=lambda point: (point.processed_positions, point.recorded_at),
        )
    )
    restarts = _clock_restarts(points)
    if restarts and time_budgets:
        raise ReportError(
            "the training clock restarts at "
            f"{', '.join(restarts)}, so seconds do not accumulate across these "
            "points and a wall-clock budget cannot be answered over them. "
            "Processed positions survive a resume; use a position budget, or "
            "report over one uninterrupted run"
        )

    return BudgetReport(
        metric=definition.identifier,
        direction=definition.direction,
        points=points,
        answers=tuple(
            [
                BudgetAnswer(
                    axis="positions",
                    budget=float(budget),
                    point=_best_within(
                        points,
                        definition,
                        lambda point: point.processed_positions,
                        budget,
                    ),
                )
                for budget in position_budgets
            ]
            + [
                BudgetAnswer(
                    axis="seconds",
                    budget=float(budget),
                    point=_best_within(
                        points,
                        definition,
                        lambda point: point.training_seconds,
                        budget,
                    ),
                )
                for budget in time_budgets
            ]
        ),
        environment_changes=_environment_changes(points),
        clock_restarts=restarts,
    )


def _clock_restarts(points: Sequence[BudgetPoint]) -> tuple[str, ...]:
    """Return the points where training seconds fell as positions rose.

    A resumed run restores its processed-position counter from the checkpoint
    but starts a new process clock, so the two budget axes disagree about
    whether the run began at the resume. Detecting it from the recorded points
    costs nothing and is what keeps the time axis from being read as one
    continuous span.
    """

    restarts: list[str] = []
    previous: float | None = None
    for point in points:
        if previous is not None and point.training_seconds < previous:
            restarts.append(point.checkpoint)
        previous = point.training_seconds
    return tuple(restarts)


def render_budget_report(report: BudgetReport) -> str:
    """Render the quality-versus-budget view as text."""

    lines = [
        f"{report.metric} ({report.direction.value}) against training budget",
        "",
        f"  {'checkpoint':<28} {'positions':>14} {'seconds':>12} {'quality':>12}",
    ]
    for point in report.points:
        row = (
            f"  {point.checkpoint:<28} {point.processed_positions:>14} "
            f"{point.training_seconds:>12.3f} {point.quality:>12.6g}"
        )
        lines.append(row)
    if report.clock_restarts:
        lines.append("")
        lines.append(
            "  the training clock restarts at "
            f"{', '.join(report.clock_restarts)}; positions accumulate across "
            "a resume and seconds do not"
        )
    if report.environment_changes:
        lines.append("")
        lines.append("  environment changed during this curve:")
        for difference in report.environment_changes:
            lines.append(
                f"    {difference.field}: {difference.baseline} → {difference.current}"
            )
    if report.answers:
        lines.append("")
        lines.append("  best quality within budget")
        for answer in report.answers:
            if answer.point is None:
                lines.append(
                    f"    {answer.axis} ≤ {answer.budget:g}: no recorded point fits"
                )
                continue
            lines.append(
                f"    {answer.axis} ≤ {answer.budget:g}: "
                f"{answer.point.quality:.6g} at {answer.point.checkpoint}"
            )
    return "\n".join(lines) + "\n"


def _quality_definition(metric: str) -> MetricDefinition:
    try:
        definition = metric_definition(metric)
    except MetricRegistryError as error:
        raise ReportError(str(error)) from error
    if definition.execution_sensitive:
        raise ReportError(
            f"{definition.identifier} measures execution rather than quality, "
            "so plotting it against a training budget would put the same "
            "measurement on both axes"
        )
    if definition.direction is MetricDirection.INFORMATIONAL:
        raise ReportError(
            f"{definition.identifier} declares no direction, so there is no "
            "best value within a budget to report"
        )
    return definition


def _efficiency_by_checkpoint(
    results: Sequence[ResultEnvelope],
) -> dict[str, tuple[ResultEnvelope, int, float]]:
    """Return the newest efficiency budget point per checkpoint label."""

    found: dict[str, tuple[ResultEnvelope, int, float]] = {}
    for envelope in sorted(
        results,
        key=lambda item: (item.recorded_at, item.result_id),
    ):
        positions = envelope.measurement(PROCESSED_POSITIONS_METRIC)
        seconds = envelope.measurement(TRAINING_SECONDS_METRIC)
        if positions is None or seconds is None:
            continue
        found[envelope.checkpoint.label] = (
            envelope,
            int(positions.value),
            seconds.value,
        )
    return found


def _quality_by_checkpoint(
    results: Sequence[ResultEnvelope],
    definition: MetricDefinition,
) -> dict[str, tuple[ResultEnvelope, Measurement]]:
    found: dict[str, tuple[ResultEnvelope, Measurement]] = {}
    for envelope in sorted(
        results,
        key=lambda item: (item.recorded_at, item.result_id),
    ):
        value = envelope.measurement(definition.identifier)
        if value is None:
            continue
        found[envelope.checkpoint.label] = (envelope, value)
    return found


def _point(
    label: str,
    efficiency: tuple[ResultEnvelope, int, float],
    quality: tuple[ResultEnvelope, Measurement],
) -> BudgetPoint:
    envelope, positions, seconds = efficiency
    quality_envelope, value = quality
    execution = envelope.execution
    return BudgetPoint(
        checkpoint=label,
        recorded_at=quality_envelope.recorded_at,
        processed_positions=positions,
        training_seconds=seconds,
        quality=value.value,
        environment=None if execution is None else execution.environment_label(),
        view=None if quality_envelope.data is None else quality_envelope.data.view,
    )


def _workload(envelope: ResultEnvelope) -> str:
    execution = envelope.execution
    if execution is None:
        raise ReportError(
            f"the efficiency reading for {envelope.checkpoint.label!r} carries "
            "no execution record, so its declared workload is unknown"
        )
    return execution.workload_sha256


def _require_one_series(values: Sequence[str], message: str) -> None:
    if len(set(values)) > 1:
        raise ReportError(message)


def _best_within(
    points: Sequence[BudgetPoint],
    definition: MetricDefinition,
    axis: Callable[[BudgetPoint], float],
    budget: float,
) -> BudgetPoint | None:
    """Return the best recorded quality reached without exceeding a budget.

    Deliberately the best *recorded* point rather than an interpolation. A
    curve between two cadence readings is not measured, and inventing a value
    there would put a number in a report that no run produced.
    """

    eligible = [point for point in points if axis(point) <= budget]
    if not eligible:
        return None
    better = min if definition.direction is MetricDirection.LOWER_IS_BETTER else max
    return better(eligible, key=lambda point: point.quality)


def _environment_changes(
    points: Sequence[BudgetPoint],
) -> tuple[ProvenanceDifference, ...]:
    """Return every environment change observed along the curve."""

    changes: list[ProvenanceDifference] = []
    previous: str | None = None
    for point in points:
        if point.environment is None:
            continue
        if previous is not None and point.environment != previous:
            changes.append(
                ProvenanceDifference(
                    field=f"environment at {point.checkpoint}",
                    baseline=previous,
                    current=point.environment,
                )
            )
        previous = point.environment
    return tuple(changes)


__all__ = [
    "DEFAULT_QUALITY_METRIC",
    "PROCESSED_POSITIONS_METRIC",
    "TRAINING_SECONDS_METRIC",
    "BudgetAnswer",
    "BudgetPoint",
    "BudgetReport",
    "build_budget_report",
    "render_budget_report",
]
