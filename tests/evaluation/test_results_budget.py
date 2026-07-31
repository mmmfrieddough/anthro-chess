"""The quality-versus-budget report joining two families by checkpoint label."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
    ExecutionRecord,
    ResultEnvelope,
    build_result,
    execution_reference,
    measurement,
)
from anthro_chess.evaluation.results.budget import (
    BudgetReport,
    build_budget_report,
    render_budget_report,
)
from anthro_chess.evaluation.results.reporting import ReportError

RECORDED_AT = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)

EfficiencyFactory = Callable[..., ResultEnvelope]


def _execution(**overrides: Any) -> ExecutionRecord:
    workload: dict[str, Any] = {"benchmark_version": 1, "batch_size": 4}
    workload.update(overrides.pop("workload", {}))
    fields: dict[str, Any] = {
        "device": "cpu",
        "device_name": "fixture-cpu",
        "precision": "float32",
        "torch_version": "2.7.0",
        "platform_key": "Linux-x86_64",
        "platform": "Linux-6.1-x86_64",
        "cpu_threads": 4,
        "workload": workload,
    }
    fields.update(overrides)
    return execution_reference(**fields)


@pytest.fixture
def efficiency_result() -> EfficiencyFactory:
    """Return a factory for one training-efficiency budget point."""

    def build(
        *,
        label: str,
        step: int,
        positions: int,
        seconds: float,
        execution: ExecutionRecord | None = None,
        recorded_at: datetime | None = None,
    ) -> ResultEnvelope:
        resolved = execution if execution is not None else _execution()
        workload = resolved.workload_component()
        return build_result(
            kind="training-efficiency",
            benchmark=BenchmarkReference(name="training-efficiency", version=1),
            checkpoint=CheckpointReference(label=label, step=step),
            execution=resolved,
            measurements=[
                measurement(
                    "training.processed_positions",
                    float(positions),
                    workload=workload,
                ),
                measurement(
                    "training.training_seconds",
                    seconds,
                    workload=workload,
                ),
            ],
            recorded_at=recorded_at or RECORDED_AT,
        )

    return build


def _curve(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    component: DataComponent,
) -> list[ResultEnvelope]:
    """Return three checkpoints, each with a budget point and a quality point."""

    results: list[ResultEnvelope] = []
    for step, positions, seconds, loss in (
        (100, 1_000, 10.0, 4.0),
        (200, 2_000, 20.0, 3.2),
        (300, 3_000, 30.0, 3.4),
    ):
        label = f"run-step-{step:08d}"
        results.append(
            efficiency_result(
                label=label,
                step=step,
                positions=positions,
                seconds=seconds,
            )
        )
        results.append(
            recorded_result(label=label, step=step, move_loss=loss, component=component)
        )
    return results


def _quality(report: BudgetReport, checkpoint: str) -> float:
    for point in report.points:
        if point.checkpoint == checkpoint:
            return point.quality
    raise AssertionError(f"{checkpoint} is not in the report")


def test_the_curve_joins_both_families_by_checkpoint_label(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    component = move_prediction_component()

    report = build_budget_report(_curve(efficiency_result, recorded_result, component))

    assert report.metric == "held_out.move_loss"
    assert [point.checkpoint for point in report.points] == [
        "run-step-00000100",
        "run-step-00000200",
        "run-step-00000300",
    ]
    assert [point.processed_positions for point in report.points] == [1000, 2000, 3000]
    assert [point.training_seconds for point in report.points] == [10.0, 20.0, 30.0]
    assert _quality(report, "run-step-00000200") == pytest.approx(3.2)
    # The resolved view the quality was measured over travels with the point.
    assert all(point.view == "canonical" for point in report.points)
    assert all(point.environment is not None for point in report.points)


def test_points_are_ordered_by_budget_rather_than_by_record_order(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    component = move_prediction_component()
    results = _curve(efficiency_result, recorded_result, component)

    report = build_budget_report(list(reversed(results)))

    assert [point.processed_positions for point in report.points] == [1000, 2000, 3000]


def test_a_budget_answer_reports_the_best_recorded_point_within_it(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    component = move_prediction_component()

    report = build_budget_report(
        _curve(efficiency_result, recorded_result, component),
        position_budgets=(2_500, 500),
        time_budgets=(35.0,),
    )

    answers = {(answer.axis, answer.budget): answer for answer in report.answers}
    # Lower is better, so the 3.2 at two thousand positions wins over the 4.0.
    within = answers[("positions", 2500.0)]
    assert within.point is not None
    assert within.point.checkpoint == "run-step-00000200"
    # No point fits the tightest budget, which is reported rather than guessed.
    assert answers[("positions", 500.0)].point is None
    # The later checkpoint is inside the time budget but scored worse, so the
    # answer is the best quality reached rather than the last one.
    generous = answers[("seconds", 35.0)]
    assert generous.point is not None
    assert generous.point.checkpoint == "run-step-00000200"


def test_a_higher_is_better_metric_picks_the_largest_value(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    component = move_prediction_component()
    results: list[ResultEnvelope] = []
    for step, positions, accuracy in ((100, 1_000, 0.20), (200, 2_000, 0.35)):
        label = f"run-step-{step:08d}"
        results.append(
            efficiency_result(
                label=label,
                step=step,
                positions=positions,
                seconds=float(step),
            )
        )
        results.append(
            recorded_result(
                label=label,
                step=step,
                component=component,
                measurements=[
                    measurement(
                        "held_out.top1_accuracy",
                        accuracy,
                        data=component,
                    )
                ],
            )
        )

    report = build_budget_report(
        results,
        metric="held_out.top1_accuracy",
        position_budgets=(2_000,),
    )

    assert report.answers[0].point is not None
    assert report.answers[0].point.quality == pytest.approx(0.35)


def test_a_checkpoint_with_only_one_side_is_not_a_point(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    component = move_prediction_component()
    results = _curve(efficiency_result, recorded_result, component)
    results.append(
        recorded_result(
            label="run-step-00000400",
            step=400,
            move_loss=3.0,
            component=component,
        )
    )

    report = build_budget_report(results)

    assert "run-step-00000400" not in {point.checkpoint for point in report.points}


def test_a_store_with_no_joinable_checkpoint_is_refused(
    recorded_result: Callable[..., ResultEnvelope],
) -> None:
    with pytest.raises(ReportError, match="training-efficiency reading"):
        build_budget_report([recorded_result(label="run-step-00000100")])


def test_quality_measured_on_two_series_is_refused(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
    scored_row: Callable[..., dict[str, Any]],
) -> None:
    """A pool or view change is a different measurement, not more of the curve."""

    first = move_prediction_component()
    second = move_prediction_component([scored_row(7), scored_row(8), scored_row(9)])
    results = [
        efficiency_result(
            label="run-step-00000100", step=100, positions=1_000, seconds=10.0
        ),
        recorded_result(label="run-step-00000100", step=100, component=first),
        efficiency_result(
            label="run-step-00000200", step=200, positions=2_000, seconds=20.0
        ),
        recorded_result(label="run-step-00000200", step=200, component=second),
    ]

    with pytest.raises(ReportError, match="more than one series"):
        build_budget_report(results)


def test_two_declared_workloads_are_refused(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    """A batch-size change is a different run configuration, not a faster one."""

    component = move_prediction_component()
    results = [
        efficiency_result(
            label="run-step-00000100", step=100, positions=1_000, seconds=10.0
        ),
        recorded_result(label="run-step-00000100", step=100, component=component),
        efficiency_result(
            label="run-step-00000200",
            step=200,
            positions=2_000,
            seconds=20.0,
            execution=_execution(workload={"batch_size": 64}),
        ),
        recorded_result(label="run-step-00000200", step=200, component=component),
    ]

    with pytest.raises(ReportError, match="more than one workload"):
        build_budget_report(results)


def test_an_environment_change_is_annotated_rather_than_refused(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    """Decision 0018's posture: a machine change is coordinates, not a break."""

    component = move_prediction_component()
    results = [
        efficiency_result(
            label="run-step-00000100", step=100, positions=1_000, seconds=10.0
        ),
        recorded_result(label="run-step-00000100", step=100, component=component),
        efficiency_result(
            label="run-step-00000200",
            step=200,
            positions=2_000,
            seconds=5.0,
            execution=_execution(device="cuda", device_name="fixture-gpu"),
        ),
        recorded_result(label="run-step-00000200", step=200, component=component),
    ]

    report = build_budget_report(results)

    assert len(report.points) == 2
    assert len(report.environment_changes) == 1
    change = report.environment_changes[0]
    assert "run-step-00000200" in change.field
    assert change.baseline is not None
    assert "fixture-gpu" in str(change.current)
    assert "environment changed" in render_budget_report(report)


def test_a_resumed_run_is_reported_rather_than_plotted_as_one_span(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    """Positions are checkpointed across a resume; the process clock is not."""

    component = move_prediction_component()
    results: list[ResultEnvelope] = []
    # The third point resumed, so its positions keep climbing while its
    # training seconds start again from a fresh process.
    for step, positions, seconds in (
        (100, 1_000, 10.0),
        (200, 2_000, 20.0),
        (300, 3_000, 4.0),
    ):
        label = f"run-step-{step:08d}"
        results.append(
            efficiency_result(
                label=label, step=step, positions=positions, seconds=seconds
            )
        )
        results.append(recorded_result(label=label, step=step, component=component))

    report = build_budget_report(results, position_budgets=(3_000,))

    assert report.clock_restarts == ("run-step-00000300",)
    assert "training clock restarts" in render_budget_report(report)
    # A position budget still answers, because that axis did survive.
    assert report.answers[0].point is not None

    with pytest.raises(ReportError, match="wall-clock budget cannot be answered"):
        build_budget_report(results, time_budgets=(30.0,))


def test_an_efficiency_metric_cannot_be_the_quality_axis(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    with pytest.raises(ReportError, match="both axes"):
        build_budget_report(
            _curve(efficiency_result, recorded_result, move_prediction_component()),
            metric="training.step_seconds",
        )


def test_a_directionless_metric_has_no_best_value_within_a_budget(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    with pytest.raises(ReportError, match="no direction"):
        build_budget_report(
            _curve(efficiency_result, recorded_result, move_prediction_component()),
            metric="held_out.uniform_over_legal_move_loss",
        )


def test_an_unknown_metric_is_refused(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    with pytest.raises(ReportError, match="unknown metric"):
        build_budget_report(
            _curve(efficiency_result, recorded_result, move_prediction_component()),
            metric="held_out.invented_metric",
        )


def test_the_rendered_report_shows_both_axes_and_the_answers(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    report = build_budget_report(
        _curve(efficiency_result, recorded_result, move_prediction_component()),
        position_budgets=(2_500, 500),
    )

    text = render_budget_report(report)

    assert "held_out.move_loss (lower_is_better)" in text
    assert "positions" in text
    assert "seconds" in text
    assert "run-step-00000200" in text
    assert "no recorded point fits" in text


def test_the_machine_readable_report_round_trips_every_point(
    efficiency_result: EfficiencyFactory,
    recorded_result: Callable[..., ResultEnvelope],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    report = build_budget_report(
        _curve(efficiency_result, recorded_result, move_prediction_component()),
        time_budgets=(25.0,),
    )

    record = report.as_record()

    assert record["metric"] == "held_out.move_loss"
    assert record["direction"] == "lower_is_better"
    assert len(record["points"]) == 3  # type: ignore[arg-type]
    assert len(record["answers"]) == 1  # type: ignore[arg-type]
    assert record["environment_changes"] == []
