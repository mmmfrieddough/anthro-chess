"""The compact delta view, its absences, and its comparability verdicts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from anthro_chess.evaluation.results import (
    BridgeIndex,
    Comparability,
    DataComponent,
    DeltaReport,
    FamilyReport,
    MetricDelta,
    Movement,
    NoiseFloor,
    NoiseVerdict,
    ReportError,
    ResultEnvelope,
    build_bridge,
    build_delta_report,
    build_history,
    measurement,
    render_history,
    render_provenance,
    render_report,
)

ResultFactory = Callable[..., ResultEnvelope]
RowFactory = Callable[..., dict[str, Any]]
Digest = Callable[..., DataComponent]

BASELINE_AT = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
CURRENT_AT = datetime(2026, 7, 8, 12, 0, tzinfo=UTC)


def _row(report: DeltaReport, metric: str) -> MetricDelta:
    for family in report.families:
        for row in family.metrics:
            if row.metric == metric:
                return row
    raise AssertionError(f"{metric} is not in the report")


def _family(report: DeltaReport, identifier: str) -> FamilyReport:
    for family in report.families:
        if family.family.identifier == identifier:
            return family
    raise AssertionError(f"{identifier} is not in the report")


@pytest.fixture
def two_checkpoints(
    recorded_result: ResultFactory,
) -> tuple[ResultEnvelope, ResultEnvelope]:
    """Return two results measured over identical inputs."""

    baseline = recorded_result(
        label="checkpoint-a",
        move_loss=3.5,
        mask_penalty=0.75,
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        move_loss=3.2,
        mask_penalty=0.8,
        recorded_at=CURRENT_AT,
    )
    return baseline, current


def test_default_view_compares_the_two_most_recent_checkpoints(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    baseline, current = two_checkpoints

    report = build_delta_report([baseline, current], BridgeIndex())

    assert report.current.label == "checkpoint-b"
    assert report.baseline is not None
    assert report.baseline.label == "checkpoint-a"

    move_loss = _row(report, "held_out.move_loss")
    assert move_loss.baseline == pytest.approx(3.5)
    assert move_loss.current == pytest.approx(3.2)
    assert move_loss.delta == pytest.approx(-0.3)
    assert move_loss.movement is Movement.BETTER
    assert move_loss.comparability is Comparability.SAME_SERIES

    penalty = _row(report, "legality.mask_penalty")
    assert penalty.movement is Movement.WORSE


def test_direction_of_improvement_is_declared_rather_than_inferred(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    values = [
        measurement("legality.legal_mass", value, data=component)
        for value in (0.5, 0.9)
    ]
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[values[0]],
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        measurements=[values[1]],
        recorded_at=CURRENT_AT,
    )

    report = build_delta_report([baseline, current], BridgeIndex())

    assert _row(report, "legality.legal_mass").movement is Movement.BETTER


def test_a_family_with_no_supporting_result_is_named_with_a_reason(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    """A silently dropped family reads as "fine" when it means "not measured"."""

    report = build_delta_report(list(two_checkpoints), BridgeIndex())

    training_health = _family(report, "training-health")
    assert training_health.metrics == ()
    assert training_health.absence == "no result recorded for checkpoint-b"

    rating = _family(report, "rating-behavior")
    assert rating.absence == "no metric is registered for this family yet"


def test_mismatched_fingerprints_are_reported_as_incomparable(
    recorded_result: ResultFactory,
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    baseline = recorded_result(
        label="checkpoint-a",
        move_loss=3.5,
        component=move_prediction_component([scored_row(1)]),
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        move_loss=3.2,
        component=move_prediction_component([scored_row(1), scored_row(2)]),
        recorded_at=CURRENT_AT,
    )

    report = build_delta_report([baseline, current], BridgeIndex())
    row = _row(report, "held_out.move_loss")

    assert row.comparability is Comparability.INCOMPARABLE
    assert row.delta is None
    assert row.movement is Movement.UNKNOWN
    assert row.note is not None
    assert "incomparable" in row.note
    assert "incomparable" in render_report(report)


def test_a_bridge_rejoins_a_series_and_renders_as_a_seam(
    recorded_result: ResultFactory,
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    baseline = recorded_result(
        label="checkpoint-a",
        move_loss=3.5,
        component=move_prediction_component([scored_row(1)]),
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        move_loss=3.2,
        component=move_prediction_component([scored_row(1), scored_row(2)]),
        recorded_at=CURRENT_AT,
    )
    bridge = build_bridge(
        from_fingerprint=baseline.measurements[0].fingerprint,
        to_fingerprint=current.measurements[0].fingerprint,
        reason="storage format change only",
        author="maintainer",
        recorded_at=CURRENT_AT,
    )

    report = build_delta_report(
        [baseline, current],
        BridgeIndex([bridge]),
    )
    row = _row(report, "held_out.move_loss")

    assert row.comparability is Comparability.BRIDGED
    assert row.delta == pytest.approx(-0.3)
    assert row.bridges == (bridge.bridge_id,)
    assert row.note == "bridged series seam"


def test_bridging_is_symmetric_and_transitive() -> None:
    """A report's answer must not depend on which side was the baseline."""

    first = build_bridge(
        from_fingerprint="a" * 64,
        to_fingerprint="b" * 64,
        reason="format",
        author="maintainer",
        recorded_at=BASELINE_AT,
    )
    second = build_bridge(
        from_fingerprint="b" * 64,
        to_fingerprint="c" * 64,
        reason="format",
        author="maintainer",
        recorded_at=CURRENT_AT,
    )
    index = BridgeIndex([first, second])

    assert index.compare("a" * 64, "c" * 64).comparability is Comparability.BRIDGED
    assert index.compare("c" * 64, "a" * 64).comparability is Comparability.BRIDGED
    assert index.series("a" * 64) == index.series("c" * 64)
    assert index.compare("a" * 64, "d" * 64).comparability is (
        Comparability.INCOMPARABLE
    )
    assert index.compare("a" * 64, "c" * 64).bridges == (first, second)


def test_a_delta_is_annotated_against_its_noise_floor(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    floor = NoiseFloor(value=0.1, kind="training", source="five seeds")
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[
            measurement("held_out.move_loss", 3.5, data=component, noise_floor=floor)
        ],
        recorded_at=BASELINE_AT,
    )
    quiet = recorded_result(
        label="checkpoint-b",
        measurements=[
            measurement("held_out.move_loss", 3.45, data=component, noise_floor=floor)
        ],
        recorded_at=CURRENT_AT,
    )
    loud = recorded_result(
        label="checkpoint-c",
        measurements=[
            measurement("held_out.move_loss", 3.0, data=component, noise_floor=floor)
        ],
        recorded_at=datetime(2026, 7, 9, tzinfo=UTC),
    )

    within = build_delta_report([baseline, quiet], BridgeIndex())
    cleared = build_delta_report(
        [baseline, quiet, loud],
        BridgeIndex(),
        baseline="checkpoint-a",
    )

    assert _row(within, "held_out.move_loss").noise is NoiseVerdict.WITHIN
    assert _row(cleared, "held_out.move_loss").noise is NoiseVerdict.CLEARED


def test_an_unknown_noise_floor_is_stated_rather_than_assumed(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    report = build_delta_report(list(two_checkpoints), BridgeIndex())

    assert _row(report, "held_out.move_loss").noise is NoiseVerdict.UNKNOWN


def test_the_first_recorded_checkpoint_reports_without_a_baseline(
    recorded_result: ResultFactory,
) -> None:
    report = build_delta_report([recorded_result()], BridgeIndex())
    row = _row(report, "held_out.move_loss")

    assert report.baseline is None
    assert row.baseline is None
    assert row.delta is None
    assert row.note == "no baseline recorded"
    assert "Baseline: none" in render_report(report)


def test_a_slice_reports_only_what_it_was_asked_about(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    report = build_delta_report(
        list(two_checkpoints),
        BridgeIndex(),
        families=["legality"],
    )

    assert [family.family.identifier for family in report.families] == ["legality"]

    by_metric = build_delta_report(
        list(two_checkpoints),
        BridgeIndex(),
        metrics=["held_out.move_loss"],
    )
    assert [row.metric for row in by_metric.families[0].metrics] == [
        "held_out.move_loss"
    ]


def test_an_unknown_selection_is_refused(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    with pytest.raises(ReportError, match="no result is recorded"):
        build_delta_report(
            list(two_checkpoints),
            BridgeIndex(),
            current="checkpoint-z",
        )
    with pytest.raises(ReportError, match="unknown metric"):
        build_delta_report(
            list(two_checkpoints),
            BridgeIndex(),
            metrics=["held_out.nonexistent"],
        )
    with pytest.raises(ReportError, match="no recorded results"):
        build_delta_report([], BridgeIndex())


def test_history_marks_series_breaks_and_bridged_seams(
    recorded_result: ResultFactory,
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    small = move_prediction_component([scored_row(1)])
    grown = move_prediction_component([scored_row(1), scored_row(2)])
    first = recorded_result(
        label="checkpoint-a",
        move_loss=3.5,
        component=small,
        recorded_at=BASELINE_AT,
    )
    second = recorded_result(
        label="checkpoint-b",
        move_loss=3.4,
        component=small,
        recorded_at=datetime(2026, 7, 4, tzinfo=UTC),
    )
    third = recorded_result(
        label="checkpoint-c",
        move_loss=3.3,
        component=grown,
        recorded_at=CURRENT_AT,
    )

    broken = build_history([first, second, third], BridgeIndex(), "held_out.move_loss")
    assert [point.starts_new_series for point in broken.points] == [
        False,
        False,
        True,
    ]

    bridge = build_bridge(
        from_fingerprint=second.measurements[0].fingerprint,
        to_fingerprint=third.measurements[0].fingerprint,
        reason="storage format change only",
        author="maintainer",
        recorded_at=CURRENT_AT,
    )
    bridged = build_history(
        [first, second, third],
        BridgeIndex([bridge]),
        "held_out.move_loss",
    )
    assert [point.bridged_from_previous for point in bridged.points] == [
        False,
        False,
        True,
    ]
    rendered = render_history(bridged)
    assert "~ " in rendered
    assert "bridged seam" in rendered


def test_machine_readable_output_is_deterministic(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    first = build_delta_report(list(two_checkpoints), BridgeIndex()).as_record()
    second = build_delta_report(
        list(reversed(two_checkpoints)),
        BridgeIndex(),
    ).as_record()

    assert first == second
    assert first["current"] == {
        "label": "checkpoint-b",
        "recorded_at": CURRENT_AT.isoformat(),
        "results": 1,
    }


def test_provenance_differences_are_available_behind_an_option(
    recorded_result: ResultFactory,
) -> None:
    baseline = recorded_result(label="checkpoint-a", recorded_at=BASELINE_AT)
    current = recorded_result(
        label="checkpoint-b",
        kind="held-out-prediction",
        recorded_at=CURRENT_AT,
    )
    changed = current.model_copy(
        update={
            "environment": current.environment.model_copy(
                update={"package_version": "9.9.9"}
            )
        }
    )

    report = build_delta_report([baseline, changed], BridgeIndex())
    rendered = render_provenance(report)

    assert [difference.field for difference in report.provenance] == ["package_version"]
    assert "9.9.9" in rendered


def test_the_default_text_view_stays_readable(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    rendered = render_report(build_delta_report(list(two_checkpoints), BridgeIndex()))

    assert "held_out.move_loss" in rendered
    assert "legality" in rendered
    assert "absent: no metric is registered for this family yet" in rendered
    assert max(len(line) for line in rendered.splitlines()) <= 120
    assert len(rendered.splitlines()) <= 20
