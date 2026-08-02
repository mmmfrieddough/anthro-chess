"""The compact delta view, its absences, and its comparability verdicts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from anthro_chess.evaluation.results import (
    PAIRED_CONTRIBUTIONS_KEY,
    BenchmarkReference,
    BridgeIndex,
    CheckpointReference,
    Comparability,
    DataComponent,
    DeltaReport,
    DetailStore,
    FamilyReport,
    FloorEntry,
    MetricDelta,
    MetricDirection,
    MetricFamily,
    Movement,
    NoiseFloor,
    NoiseFloorIndex,
    NoiseVerdict,
    PairedContributions,
    PairedFloorIndex,
    ReportError,
    ResultEnvelope,
    SeriesGroup,
    build_bridge,
    build_characterization,
    build_delta_report,
    build_history,
    build_result,
    execution_reference,
    measurement,
    paired_contributions,
    render_history,
    render_provenance,
    render_report,
    series_fingerprint,
)
from anthro_chess.evaluation.results.reporting import (
    MAXIMUM_LINE_WIDTH,
    MAXIMUM_METRIC_COLUMN_WIDTH,
    MINIMUM_METRIC_COLUMN_WIDTH,
    CheckpointSelection,
)

ResultFactory = Callable[..., ResultEnvelope]
RowFactory = Callable[..., dict[str, Any]]
Digest = Callable[..., DataComponent]

BASELINE_AT = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
CURRENT_AT = datetime(2026, 7, 8, 12, 0, tzinfo=UTC)

GENERATED_PLIES = "generated_play.mean_game_plies"


def _row(report: DeltaReport, metric: str) -> MetricDelta:
    for family in report.families:
        for row in family.metrics:
            if row.metric == metric:
                return row
    raise AssertionError(f"{metric} is not in the report")


def _groups(report: DeltaReport, metric: str) -> tuple[SeriesGroup, ...]:
    """Return the series a metric was reported on, in report order."""

    return tuple(
        group
        for family in report.families
        for group in family.series
        if any(row.metric == metric for row in group.metrics)
    )


def _rollout_cell(
    label: str,
    plies: float,
    *,
    arm: str = "standard-start",
    rating: int = 1500,
    temperature: float = 0.9,
    recorded_at: datetime,
) -> ResultEnvelope:
    """Build one recorded cell of a generated-play matrix.

    A rollout writes one envelope per cell, so a single run records several
    results for one checkpoint that share a metric identifier and differ only
    by declared workload.
    """

    execution = execution_reference(
        device="cpu",
        device_name="fixture",
        precision="float32",
        torch_version="2.7.0",
        platform_key="Fixture-x86",
        platform="fixture-1.2.3",
        workload={
            "positions": {"kind": arm, "count": 8},
            "target_rating": rating,
            "temperature": temperature,
            "maximum_generated_plies": 200,
        },
    )
    return build_result(
        kind="generated-play",
        benchmark=BenchmarkReference(name="generated-play", version=1),
        checkpoint=CheckpointReference(label=label, step=1),
        execution=execution,
        measurements=[
            measurement(
                GENERATED_PLIES,
                plies,
                workload=execution.workload_component(),
            )
        ],
        recorded_at=recorded_at,
    )


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
    assert rating.absence == "no result recorded for checkpoint-b"


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


def test_a_paired_floor_replaces_independent_sampling_floors(
    tmp_path: Path,
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    independent = NoiseFloor(
        value=1.0,
        kind="data-sampling",
        source="independent benchmark draws",
    )
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[
            measurement(
                "held_out.move_loss",
                3.5,
                data=component,
                noise_floor=independent,
            )
        ],
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        measurements=[
            measurement(
                "held_out.move_loss",
                3.3,
                data=component,
                noise_floor=independent,
            )
        ],
        recorded_at=CURRENT_AT,
    )
    detail = DetailStore(tmp_path / "detail")

    # Enough units to support a floor. The paired bootstrap's dispersion is
    # bounded for the units behind it, so a two-unit payload would produce a
    # floor an order of magnitude above the delta no matter how well the two
    # sides agreed, and would be testing the bound rather than the pairing.
    units = tuple(f"game-{index}" for index in range(40))

    def retained(values: list[float]) -> dict[str, object]:
        return {
            PAIRED_CONTRIBUTIONS_KEY: paired_contributions(
                unit="fixture-game",
                unit_ids=units,
                metrics={"held_out.move_loss": values},
                resamples=1_000,
                seed=0,
                coverage=1.96,
            ).as_record()
        }

    # The two checkpoints differ by the same small amount on every unit, which
    # is the case a paired floor exists to see: the spread across units is
    # large and common to both, and the spread of their difference is not.
    # Centred so the retained values reproduce the recorded means of 3.5 and
    # 3.3, which the store checks before it will bootstrap them.
    offsets = [0.05 * (index - (len(units) - 1) / 2) for index in range(len(units))]
    baseline_values = [3.5 + offset for offset in offsets]
    current_values = [value - 0.2 for value in baseline_values]
    baseline = baseline.model_copy(
        update={"detail": detail.write("baseline.json", retained(baseline_values))}
    )
    current = current.model_copy(
        update={"detail": detail.write("current.json", retained(current_values))}
    )

    report = build_delta_report(
        [baseline, current],
        BridgeIndex(),
        comparison_floors=PairedFloorIndex(detail),
    )
    row = _row(report, "held_out.move_loss")

    assert row.noise is NoiseVerdict.CLEARED
    assert row.noise_floor is not None
    assert row.noise_floor < 0.2
    assert row.noise_floors[0].source is not None
    assert "paired bootstrap" in row.noise_floors[0].source


def test_a_paired_floor_preserves_the_benchmark_strata(
    tmp_path: Path,
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[measurement("held_out.move_loss", 0.0, data=component)],
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        measurements=[measurement("held_out.move_loss", 0.5, data=component)],
        recorded_at=CURRENT_AT,
    )
    detail = DetailStore(tmp_path / "detail")

    def retained(values: list[float]) -> dict[str, object]:
        return {
            PAIRED_CONTRIBUTIONS_KEY: paired_contributions(
                unit="fixture-game",
                unit_ids=("a", "b", "c", "d"),
                stratum="rating",
                strata=("low", "low", "high", "high"),
                metrics={"held_out.move_loss": values},
                resamples=1_000,
                seed=0,
                coverage=1.96,
            ).as_record()
        }

    baseline = baseline.model_copy(
        update={
            "detail": detail.write(
                "stratified-baseline.json",
                retained([0.0, 0.0, 0.0, 0.0]),
            )
        }
    )
    current = current.model_copy(
        update={
            "detail": detail.write(
                "stratified-current.json",
                retained([0.0, 0.0, 1.0, 1.0]),
            )
        }
    )

    report = build_delta_report(
        [baseline, current],
        BridgeIndex(),
        comparison_floors=PairedFloorIndex(detail),
    )
    row = _row(report, "held_out.move_loss")

    # Each stratum's delta is constant. Resampling within the fixed allocation
    # therefore adds no composition variance between the two strata.
    assert row.noise_floor == 0.0
    assert row.noise_floors[0].source is not None
    assert "stratified paired bootstrap" in row.noise_floors[0].source


def test_a_weighted_paired_floor_reproduces_a_mean_over_positions(
    tmp_path: Path,
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    """A dependency degradation is a per-position mean carried by games.

    Games hold different numbers of positions, so the retained per-game values
    only reproduce the recorded measurement once each is weighted by the
    positions behind it. Without the weights the store rejects the payload
    rather than bootstrapping a quantity nobody measured.
    """

    component = move_prediction_component()
    units = tuple(f"game-{index}" for index in range(40))
    # Alternating sizes, so an unweighted mean and a weighted one differ.
    weights = [1.0 + 9.0 * (index % 2) for index in range(len(units))]
    offsets = [0.05 * (index - (len(units) - 1) / 2) for index in range(len(units))]
    baseline_values = [0.5 + offset for offset in offsets]
    current_values = [value + 0.2 for value in baseline_values]

    def weighted_mean(values: list[float]) -> float:
        return sum(
            weight * value for weight, value in zip(weights, values, strict=True)
        ) / sum(weights)

    metric = "dependency.rating_absent_degradation"
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[
            measurement(metric, weighted_mean(baseline_values), data=component)
        ],
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        measurements=[
            measurement(metric, weighted_mean(current_values), data=component)
        ],
        recorded_at=CURRENT_AT,
    )
    detail = DetailStore(tmp_path / "detail")

    def retained(values: list[float]) -> dict[str, object]:
        return {
            PAIRED_CONTRIBUTIONS_KEY: paired_contributions(
                unit="pool-game",
                unit_ids=units,
                metrics={metric: values},
                weights=weights,
                resamples=1_000,
                seed=0,
                coverage=1.96,
            ).as_record()
        }

    baseline = baseline.model_copy(
        update={"detail": detail.write("baseline.json", retained(baseline_values))}
    )
    current = current.model_copy(
        update={"detail": detail.write("current.json", retained(current_values))}
    )

    report = build_delta_report(
        [baseline, current],
        BridgeIndex(),
        comparison_floors=PairedFloorIndex(detail),
    )
    row = _row(report, metric)

    assert row.delta == pytest.approx(0.2)
    assert row.noise is NoiseVerdict.CLEARED
    assert row.noise_floor is not None
    assert row.noise_floor < 0.2
    assert row.noise_floors[0].source is not None
    assert "weighted paired bootstrap" in row.noise_floors[0].source


def test_a_paired_floor_is_withheld_when_the_weights_disagree(
    tmp_path: Path,
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    """Differently weighted sides do not describe one delta.

    A weight is a property of the frozen view rather than of the checkpoint, so
    two readings that disagree about it did not average over the same thing,
    and their difference is not the delta either measurement reports.
    """

    component = move_prediction_component()
    metric = "dependency.rating_absent_degradation"
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[measurement(metric, 0.5, data=component)],
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        measurements=[measurement(metric, 0.5, data=component)],
        recorded_at=CURRENT_AT,
    )
    detail = DetailStore(tmp_path / "detail")

    def retained(weights: list[float]) -> dict[str, object]:
        return {
            PAIRED_CONTRIBUTIONS_KEY: paired_contributions(
                unit="pool-game",
                unit_ids=("a", "b", "c", "d"),
                metrics={metric: [0.5, 0.5, 0.5, 0.5]},
                weights=weights,
                resamples=1_000,
                seed=0,
                coverage=1.96,
            ).as_record()
        }

    baseline = baseline.model_copy(
        update={"detail": detail.write("baseline.json", retained([1.0, 1.0, 2.0, 2.0]))}
    )
    current = current.model_copy(
        update={"detail": detail.write("current.json", retained([1.0, 1.0, 1.0, 1.0]))}
    )

    report = build_delta_report(
        [baseline, current],
        BridgeIndex(),
        comparison_floors=PairedFloorIndex(detail),
    )
    row = _row(report, metric)

    assert row.noise is NoiseVerdict.UNKNOWN
    assert row.noise_floors == ()


def test_a_payload_written_before_weights_existed_still_resolves() -> None:
    """Retained contributions outlive the build that wrote them.

    Adding the weight vector must not strand the payloads already sitting in
    machine-local detail directories, which is a property of the stored bytes
    rather than of anything the current code path produces. So this validates
    a literal record rather than one built through ``paired_contributions``.
    """

    stored = {
        "version": 2,
        "unit": "puzzle-source-game",
        "unit_ids": ["a", "b", "c", "d"],
        "stratum": "puzzle-rating",
        "strata": ["1000", "1000", "1800", "1800"],
        "metrics": {"puzzle.greedy_first_move_accuracy": [0.0, 1.0, 0.0, 1.0]},
        "resamples": 1000,
        "seed": 0,
        "coverage": 1.96,
        "confidence": 0.95,
    }

    restored = PairedContributions.model_validate(stored)

    assert restored.version == 2
    # No weights means every unit counts once, which is what version 2 meant.
    assert restored.weights is None
    # The ceiling still refuses a payload this build cannot read.
    with pytest.raises(ValueError, match="version 4"):
        PairedContributions.model_validate({**stored, "version": 4})


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1.0, 2.0, 3.0], "3 weights for 4 units"),
        ([1.0, 1.0, 0.0, 1.0], "finite and positive"),
        ([1.0, 1.0, -1.0, 1.0], "finite and positive"),
    ],
)
def test_paired_contribution_weights_are_validated(
    weights: list[float],
    message: str,
) -> None:
    """A weight decides how much of the metric a unit accounts for.

    A missing or non-positive one silently reweights the reading rather than
    failing, so it is rejected where it is written instead of surfacing as a
    floor nobody can explain.
    """

    with pytest.raises(ValueError, match=message):
        paired_contributions(
            unit="pool-game",
            unit_ids=("a", "b", "c", "d"),
            metrics={"held_out.move_loss": [1.0, 1.0, 1.0, 1.0]},
            weights=weights,
            resamples=1_000,
            seed=0,
            coverage=1.96,
        )


def test_a_declared_metric_still_takes_a_floor_the_declaration_does_not_cover(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    """The declaration rules out one estimator, not every noise source.

    Both declared reasons argue that resampling the units a reading scored
    cannot estimate the metric's dispersion. Evaluation noise is read from
    repeated measurements instead, so it describes such a metric perfectly
    well, and a report that discarded it would be withholding a floor it has.
    """

    component = move_prediction_component()
    metric = "dependency.rating_cross_conditioning_match_rate"
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[measurement(metric, 0.25, data=component)],
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        measurements=[measurement(metric, 0.75, data=component)],
        recorded_at=CURRENT_AT,
    )
    floors = NoiseFloorIndex(
        [
            build_characterization(
                kind="evaluation",
                method="repeat-measurement",
                replicates=8,
                source="eight re-measurements of one checkpoint",
                floors=[
                    FloorEntry(
                        metric=metric,
                        fingerprint=series_fingerprint(metric, component),
                        floor=0.75,
                        dispersion=0.2,
                        dispersion_bound=0.2,
                        degrees_of_freedom=7,
                        sampling_units=8,
                    )
                ],
                recorded_at=BASELINE_AT,
            )
        ]
    )

    report = build_delta_report([baseline, current], BridgeIndex(), floors=floors)
    row = _row(report, metric)

    # The delta is 0.5, inside an evaluation floor of 0.75.
    assert row.noise is NoiseVerdict.WITHIN
    assert row.noise_floor_kind == "evaluation"
    assert [floor.kind for floor in row.noise_floors] == ["evaluation"]


def test_a_metric_that_can_carry_no_sampling_floor_says_so_rather_than_unknown(
    tmp_path: Path,
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    """ "Nobody characterized one" and "one cannot exist" are different states.

    The cross-conditioning match rate counts rating slices, so no resampling of
    games estimates its dispersion. Reporting that as ``unknown`` beside a
    metric merely awaiting a characterization would set a reader to work that
    cannot be done.
    """

    component = move_prediction_component()
    metric = "dependency.rating_cross_conditioning_match_rate"
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[measurement(metric, 0.25, data=component)],
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        measurements=[measurement(metric, 0.75, data=component)],
        recorded_at=CURRENT_AT,
    )
    detail = DetailStore(tmp_path / "detail")

    def retained(values: list[float]) -> dict[str, object]:
        return {
            PAIRED_CONTRIBUTIONS_KEY: paired_contributions(
                unit="pool-game",
                unit_ids=("a", "b", "c", "d"),
                metrics={metric: values},
                resamples=1_000,
                seed=0,
                coverage=1.96,
            ).as_record()
        }

    baseline = baseline.model_copy(
        update={"detail": detail.write("baseline.json", retained([0.25] * 4))}
    )
    current = current.model_copy(
        update={"detail": detail.write("current.json", retained([0.75] * 4))}
    )

    report = build_delta_report(
        [baseline, current],
        BridgeIndex(),
        comparison_floors=PairedFloorIndex(detail),
    )
    row = _row(report, metric)

    # A sampling floor was retained for it and is still refused, because the
    # declaration is about what resampling can estimate rather than about
    # whether anybody produced a number.
    assert row.noise is NoiseVerdict.UNQUALIFIABLE
    assert row.noise_floor is None
    assert row.noise_floors == ()
    rendered = render_report(report)
    assert "unqualifiable" in rendered
    # The legend wraps to the terminal width, so it is read unwrapped here.
    assert "no sampling floor can exist for it" in " ".join(rendered.split())


def test_an_unknown_noise_floor_is_stated_rather_than_assumed(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    report = build_delta_report(list(two_checkpoints), BridgeIndex())
    row = _row(report, "held_out.move_loss")

    assert row.noise is NoiseVerdict.UNKNOWN
    # An uncharacterized floor is not a floor of zero, which would license
    # every delta as a finding.
    assert row.noise_floor is None
    assert row.noise_floor_kind is None
    assert row.noise_floors == ()
    assert "unknown" in render_report(report)


def test_a_characterized_floor_applies_without_being_attached_to_the_result(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    floors = NoiseFloorIndex(
        [
            build_characterization(
                kind="data-sampling",
                method="bootstrap-over-games",
                replicates=1_000,
                source="one pool",
                floors=[
                    FloorEntry(
                        metric="held_out.move_loss",
                        fingerprint=series_fingerprint(
                            "held_out.move_loss",
                            component,
                        ),
                        floor=0.5,
                        dispersion=0.18,
                        dispersion_bound=0.18,
                        degrees_of_freedom=199,
                        sampling_units=200,
                    )
                ],
                recorded_at=BASELINE_AT,
            )
        ]
    )

    report = build_delta_report(list(two_checkpoints), BridgeIndex(), floors=floors)
    row = _row(report, "held_out.move_loss")

    # The recorded delta is -0.3, inside a floor of 0.5.
    assert row.noise is NoiseVerdict.WITHIN
    assert row.noise_floor_kind == "data-sampling"
    assert "within (sampling)" in render_report(report)


def test_a_delta_is_judged_against_the_widest_floor_that_applies(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    # Sampling noise and training noise are different questions, and a delta
    # has to clear every source that applies before it is a finding.
    component = move_prediction_component()
    fingerprint = series_fingerprint("held_out.move_loss", component)
    floors = NoiseFloorIndex(
        [
            build_characterization(
                kind="data-sampling",
                method="bootstrap-over-games",
                replicates=1_000,
                source="one pool",
                floors=[
                    FloorEntry(
                        metric="held_out.move_loss",
                        fingerprint=fingerprint,
                        floor=0.05,
                        dispersion=0.02,
                        dispersion_bound=0.02,
                        degrees_of_freedom=199,
                        sampling_units=200,
                    )
                ],
                recorded_at=BASELINE_AT,
            ),
            build_characterization(
                kind="training",
                method="independent-replicates",
                replicates=4,
                source="four seeds",
                floors=[
                    FloorEntry(
                        metric="held_out.move_loss",
                        fingerprint=fingerprint,
                        floor=0.4,
                        dispersion=0.14,
                        dispersion_bound=0.14,
                        degrees_of_freedom=3,
                    )
                ],
                recorded_at=BASELINE_AT,
            ),
        ]
    )
    baseline = recorded_result(
        label="checkpoint-a",
        move_loss=3.5,
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        move_loss=3.2,
        recorded_at=CURRENT_AT,
    )

    report = build_delta_report([baseline, current], BridgeIndex(), floors=floors)
    row = _row(report, "held_out.move_loss")

    assert row.noise_floor == pytest.approx(0.4)
    assert row.noise_floor_kind == "training"
    assert {floor.kind for floor in row.noise_floors} == {
        "data-sampling",
        "training",
    }
    # A delta of -0.3 clears the sampling floor but not the training one, so
    # the reported verdict is the conservative reading.
    assert row.noise is NoiseVerdict.WITHIN


def test_a_delta_inside_its_floor_is_shown_rather_than_hidden(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    # A consistent small regression only stays visible if a within-floor delta
    # is still reported with its value.
    component = move_prediction_component()
    floors = NoiseFloorIndex(
        [
            build_characterization(
                kind="training",
                method="independent-replicates",
                replicates=3,
                source="three seeds",
                floors=[
                    FloorEntry(
                        metric="held_out.move_loss",
                        fingerprint=series_fingerprint(
                            "held_out.move_loss",
                            component,
                        ),
                        floor=10.0,
                        dispersion=3.6,
                        dispersion_bound=3.6,
                        degrees_of_freedom=2,
                    )
                ],
                recorded_at=BASELINE_AT,
            )
        ]
    )
    baseline = recorded_result(
        label="checkpoint-a",
        move_loss=3.5,
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        move_loss=3.2,
        recorded_at=CURRENT_AT,
    )

    report = build_delta_report([baseline, current], BridgeIndex(), floors=floors)
    row = _row(report, "held_out.move_loss")

    assert row.noise is NoiseVerdict.WITHIN
    assert row.delta == pytest.approx(-0.3)
    assert row.movement is Movement.BETTER
    assert "-0.3" in render_report(report)


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


def test_every_cell_of_a_matrix_is_reported_rather_than_one_of_them() -> None:
    """One envelope per cell must not collapse into one arbitrary row.

    A suite shares one ``recorded_at``, so choosing the most recent measurement
    of a metric broke the tie on ``result_id`` and rendered one cell as though
    it were the checkpoint's value.
    """

    report = build_delta_report(
        [
            _rollout_cell(
                "checkpoint-a", 60.0, arm="standard-start", recorded_at=BASELINE_AT
            ),
            _rollout_cell(
                "checkpoint-a", 90.0, arm="human-prefix", recorded_at=BASELINE_AT
            ),
            _rollout_cell(
                "checkpoint-b", 64.0, arm="standard-start", recorded_at=CURRENT_AT
            ),
            _rollout_cell(
                "checkpoint-b", 88.0, arm="human-prefix", recorded_at=CURRENT_AT
            ),
        ],
        BridgeIndex(),
        metrics=[GENERATED_PLIES],
    )

    groups = _groups(report, GENERATED_PLIES)
    assert [group.label for group in groups] == [
        "positions.kind=human-prefix",
        "positions.kind=standard-start",
    ]
    values = {
        group.label: (group.metrics[0].baseline, group.metrics[0].current)
        for group in groups
    }
    assert values["positions.kind=human-prefix"] == (90.0, 88.0)
    assert values["positions.kind=standard-start"] == (60.0, 64.0)
    # Each cell is its own series, so the rows have to be distinguishable in
    # the record without reconstructing the grouping.
    assert len({group.metrics[0].series for group in groups}) == 2


def test_a_series_label_names_only_what_tells_the_cells_apart() -> None:
    """A cell's whole declared workload above every row carries no information."""

    report = build_delta_report(
        [
            _rollout_cell("checkpoint-a", 60.0, rating=1200, recorded_at=BASELINE_AT),
            _rollout_cell("checkpoint-a", 70.0, rating=1800, recorded_at=BASELINE_AT),
            _rollout_cell("checkpoint-b", 62.0, rating=1200, recorded_at=CURRENT_AT),
            _rollout_cell("checkpoint-b", 74.0, rating=1800, recorded_at=CURRENT_AT),
        ],
        BridgeIndex(),
        metrics=[GENERATED_PLIES],
    )

    labels = [group.label for group in _groups(report, GENERATED_PLIES)]

    assert labels == ["target_rating=1200", "target_rating=1800"]
    # The fields the two cells share stay in the envelope rather than in a
    # label that has to fit above a table.
    assert not any("temperature" in (label or "") for label in labels)


def test_a_cell_measured_for_only_one_checkpoint_is_named_rather_than_dropped() -> None:
    report = build_delta_report(
        [
            _rollout_cell("checkpoint-a", 60.0, rating=1200, recorded_at=BASELINE_AT),
            _rollout_cell("checkpoint-b", 62.0, rating=1200, recorded_at=CURRENT_AT),
            _rollout_cell("checkpoint-b", 74.0, rating=1800, recorded_at=CURRENT_AT),
        ],
        BridgeIndex(),
        metrics=[GENERATED_PLIES],
    )

    added = next(
        group
        for group in _groups(report, GENERATED_PLIES)
        if group.label == "target_rating=1800"
    )
    row = added.metrics[0]

    assert row.current == 74.0
    assert row.baseline is None
    assert row.delta is None
    assert row.note == "not measured for checkpoint-a"


def test_one_workload_replacing_another_is_reported_as_a_workload_change() -> None:
    """The sequential case keeps its diagnosis rather than becoming two halves.

    Splitting a family whose workload simply moved would report two half-rows
    that never name the cause, so exactly one unmatched workload on each side
    is paired instead. Anything looser would be guessing which cell of a matrix
    succeeded which.
    """

    report = build_delta_report(
        [
            _rollout_cell("checkpoint-a", 60.0, rating=1200, recorded_at=BASELINE_AT),
            _rollout_cell("checkpoint-b", 74.0, rating=1800, recorded_at=CURRENT_AT),
        ],
        BridgeIndex(),
        metrics=[GENERATED_PLIES],
    )

    (group,) = _groups(report, GENERATED_PLIES)
    row = group.metrics[0]

    assert group.label is None
    assert row.comparability is Comparability.INCOMPARABLE
    assert row.delta is None
    assert row.note is not None
    assert "the declared workload changed" in row.note


def test_a_metric_on_several_series_says_so_rather_than_showing_one(
    recorded_result: ResultFactory,
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    """A metric that declares no workload can still be hiding a second series.

    Nothing can be compared across them, so the most recent is the right one to
    show; what would be wrong is showing it as though it were the only one.
    """

    small = move_prediction_component([scored_row(1)])
    grown = move_prediction_component([scored_row(1), scored_row(2)])
    report = build_delta_report(
        [
            recorded_result(
                label="checkpoint-a",
                move_loss=3.5,
                component=grown,
                recorded_at=BASELINE_AT,
            ),
            recorded_result(
                label="checkpoint-b",
                move_loss=9.9,
                component=small,
                recorded_at=CURRENT_AT,
            ),
            recorded_result(
                label="checkpoint-b",
                move_loss=3.2,
                component=grown,
                recorded_at=datetime(2026, 7, 9, 12, 0, tzinfo=UTC),
            ),
        ],
        BridgeIndex(),
        metrics=["held_out.move_loss"],
    )
    row = _row(report, "held_out.move_loss")

    assert row.current == pytest.approx(3.2)
    assert row.note == "2 series recorded for this checkpoint; showing the most recent"


def test_the_text_view_states_which_series_each_row_belongs_to() -> None:
    report = build_delta_report(
        [
            _rollout_cell(
                "checkpoint-a", 60.0, arm="standard-start", recorded_at=BASELINE_AT
            ),
            _rollout_cell(
                "checkpoint-a", 90.0, arm="human-prefix", recorded_at=BASELINE_AT
            ),
            _rollout_cell(
                "checkpoint-b", 64.0, arm="standard-start", recorded_at=CURRENT_AT
            ),
            _rollout_cell(
                "checkpoint-b", 88.0, arm="human-prefix", recorded_at=CURRENT_AT
            ),
        ],
        BridgeIndex(),
        metrics=[GENERATED_PLIES],
    )

    rendered = render_report(report)

    assert "[series: positions.kind=standard-start]" in rendered
    assert "[series: positions.kind=human-prefix]" in rendered
    assert rendered.count(GENERATED_PLIES) == 2
    assert max(len(line) for line in rendered.splitlines()) <= 120


def test_a_single_series_family_renders_without_a_series_header(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    """The benchmark writing one result per checkpoint gains no new lines."""

    report = build_delta_report(list(two_checkpoints), BridgeIndex())

    assert all(
        group.label is None for family in report.families for group in family.series
    )
    assert "[series:" not in render_report(report)


def test_the_default_text_view_stays_readable(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    rendered = render_report(build_delta_report(list(two_checkpoints), BridgeIndex()))

    assert "held_out.move_loss" in rendered
    assert "legality" in rendered
    # A family still awaiting its first metric is named rather than dropped,
    # but collapsed onto one wrapped line so registering families ahead of
    # their benchmarks cannot grow the default view.
    assert "awaiting a first metric:" in rendered
    assert "rating-behavior" in rendered
    # An absence that is about this checkpoint rather than about the plan keeps
    # its own line, because it is the actionable one.
    assert "absent: no result recorded for" in rendered
    assert max(len(line) for line in rendered.splitlines()) <= 120
    # A ratchet rather than a round number: it is the current height, so a
    # change that grows the default view has to be a deliberate one. Two lines
    # each belong to decision decomposition, the puzzle-backed rating family,
    # generated play, game termination, novelty, and training efficiency, which
    # have metrics but no result in this fixture; a family gets its own
    # actionable absence section as soon as it has a metric to be absent.
    assert len(rendered.splitlines()) <= 32


def _width_report(*identifiers: str) -> DeltaReport:
    """Build a report whose rows carry exactly the given identifiers.

    Assembled directly rather than measured, because the property under test
    belongs to the renderer: what the column has to do is hold whatever
    identifiers it is handed, including ones no benchmark has registered yet.
    """

    rows = tuple(
        MetricDelta(
            metric=identifier,
            family="legality",
            direction=MetricDirection.LOWER_IS_BETTER,
            baseline=0.5,
            current=0.4,
            delta=-0.1,
            comparability=Comparability.SAME_SERIES,
            movement=Movement.BETTER,
            noise=NoiseVerdict.NOT_APPLICABLE,
            noise_floor=None,
            noise_floor_kind=None,
            noise_floors=(),
            bridges=(),
            note=None,
        )
        for identifier in identifiers
    )
    return DeltaReport(
        baseline=CheckpointSelection(
            label="checkpoint-a", recorded_at=BASELINE_AT, results=1
        ),
        current=CheckpointSelection(
            label="checkpoint-b", recorded_at=CURRENT_AT, results=1
        ),
        families=(
            FamilyReport(
                family=MetricFamily(
                    identifier="legality",
                    title="Legality",
                    summary="fixture family",
                ),
                series=(SeriesGroup(workload=None, label=None, metrics=rows),),
                absence=None,
            ),
        ),
        provenance=(),
    )


def _header(rendered: str) -> str:
    return next(
        line for line in rendered.splitlines() if line.lstrip().startswith("metric ")
    )


def test_a_long_identifier_does_not_push_its_row_out_of_alignment() -> None:
    """The column widens to the longest name present, header included.

    Every column right of the identifier shifts on an overflowing row, so the
    default view stops scanning cleanly exactly where a reader is comparing
    numbers down a column.
    """

    rendered = render_report(
        _width_report(
            "legality.mask_penalty",
            "x" * MAXIMUM_METRIC_COLUMN_WIDTH,
        )
    )

    # The direction column is the first thing right of the identifier, so it is
    # where a shifted row shows up first.
    directions = {
        line.index("lower") for line in rendered.splitlines() if "lower" in line
    }

    assert directions == {_header(rendered).index("better")}


def test_the_widest_permitted_identifier_still_fits_the_line() -> None:
    """The registry's budget is exactly what keeps the header on one line."""

    rendered = render_report(_width_report("x" * MAXIMUM_METRIC_COLUMN_WIDTH))

    assert len(_header(rendered)) == MAXIMUM_LINE_WIDTH


def test_short_identifiers_keep_the_column_at_its_minimum() -> None:
    """A report of short names keeps the table's usual shape.

    Sizing from what is present is about making room, not about reflowing the
    common report to whichever family happens to be selected.
    """

    rendered = render_report(_width_report("legality.legal_mass"))

    indent, separator = len("  "), len(" ")
    assert _header(rendered).index("better") == (
        indent + MINIMUM_METRIC_COLUMN_WIDTH + separator
    )
