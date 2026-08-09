"""Noise floor estimation, storage, lookup, and the sizing question."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest

from anthro_chess.evaluation.noise import (
    GameTotals,
    MetricTotal,
    NoiseConfig,
    bootstrap_dispersions,
    sampling_dispersions,
)
from anthro_chess.evaluation.results import (
    BOOTSTRAP_METHOD,
    DEFAULT_CONFIDENCE,
    DEFAULT_COVERAGE,
    PROCESS_REPLICATE_METHOD,
    BridgeIndex,
    DataComponent,
    ExecutionRecord,
    FloorEntry,
    MetricDispersion,
    NoiseCharacterization,
    NoiseCharacterizationError,
    NoiseFloorIndex,
    ResultsStore,
    ResultsStoreError,
    bounded_floor,
    build_bridge,
    build_characterization,
    combined_floor,
    dispersion_bound,
    execution_reference,
    games_to_resolve,
    measured_dispersion,
    process_dispersion,
    process_replicate_floors,
    replicate_dispersion,
    replicate_floors,
    self_combined_floor,
    series_fingerprint,
    training_scope,
)
from anthro_chess.evaluation.results.metrics import (
    INFERENCE_MOVE_LATENCY_BY_PERCENTILE,
)
from anthro_chess.evaluation.results.records import ResultEnvelope

Digest = Callable[..., DataComponent]

RECORDED_AT = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
#: Any digest serves; a scope test turns only on whether the two sides of a
#: delta carry the same one.
TRAINING_SCOPE = "1a" * 32
OTHER_TRAINING_SCOPE = "2b" * 32
METRIC = "held_out.move_loss"
OTHER_METRIC = "legality.mask_penalty"
EFFICIENCY_METRIC = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier


def _only(dispersions: Mapping[str, MetricDispersion]) -> MetricDispersion:
    """Return the one dispersion a single-metric bootstrap produced."""

    (dispersion,) = dispersions.values()
    return dispersion


def _totals(values: list[float], *, positions: int = 4) -> tuple[GameTotals, ...]:
    """Return per-game totals for one metric at a fixed positions-per-game."""

    return tuple(
        GameTotals(
            game_id=index,
            metrics={METRIC: MetricTotal(total=value * positions, positions=positions)},
        )
        for index, value in enumerate(values)
    )


def _entry(
    component: DataComponent,
    *,
    metric: str = METRIC,
    floor: float = 0.1,
    dispersion: float = 0.05,
    dispersion_bound: float | None = None,
    degrees_of_freedom: int = 5,
) -> FloorEntry:
    return FloorEntry(
        metric=metric,
        fingerprint=series_fingerprint(metric, component),
        floor=floor,
        dispersion=dispersion,
        dispersion_bound=dispersion if dispersion_bound is None else dispersion_bound,
        degrees_of_freedom=degrees_of_freedom,
    )


def test_a_floor_is_the_delta_two_independent_measurements_produce() -> None:
    # A floor has to be comparable to a reported delta, and the difference of
    # two independent measurements is wider than either one by sqrt(2). Held at
    # a confidence of one half, where the chi-squared bound is close enough to
    # the estimate to leave the sqrt(2) visible on its own.
    bound = dispersion_bound(0.1, degrees_of_freedom=200, confidence=0.5)
    _, floor = bounded_floor(0.1, degrees_of_freedom=200, coverage=1.0, confidence=0.5)
    assert floor == pytest.approx(math.sqrt(2) * bound)
    assert bounded_floor(0.0, degrees_of_freedom=5)[1] == 0.0
    single = bounded_floor(0.1, degrees_of_freedom=5, coverage=1.0)[1]
    doubled = bounded_floor(0.1, degrees_of_freedom=5, coverage=2.0)[1]
    assert doubled == pytest.approx(2 * single)


def test_a_delta_floor_combines_the_two_readings_it_compares() -> None:
    # The variance of a difference is the sum of the two variances, so two
    # readings that happen to agree reduce to the sqrt(2) the old arithmetic
    # assumed, and two that do not are dominated by the noisier of them rather
    # than by whichever one the report happened to look up.
    narrow = measured_dispersion(0.1, kind="data-sampling", degrees_of_freedom=200)
    wide = measured_dispersion(1.19, kind="data-sampling", degrees_of_freedom=200)

    assert combined_floor(narrow, narrow) == pytest.approx(
        DEFAULT_COVERAGE * math.sqrt(2) * narrow.bound
    )
    assert combined_floor(narrow, wide) == pytest.approx(
        DEFAULT_COVERAGE * math.hypot(narrow.bound, wide.bound)
    )
    # The equal-dispersion assumption understates the combined floor whenever
    # the two readings differ, and by most where they differ by most.
    assert combined_floor(narrow, wide) > combined_floor(narrow, narrow)
    assert combined_floor(narrow, wide) < combined_floor(wide, wide)


def test_one_reading_alone_reports_the_floor_a_matching_reading_would_face() -> None:
    # A single reading resolves nothing on its own, so a display holding one
    # shows what a delta against a reading like it would have to clear.
    dispersion = measured_dispersion(0.3, kind="evaluation", degrees_of_freedom=9)

    assert self_combined_floor(dispersion) == pytest.approx(
        combined_floor(dispersion, dispersion)
    )


def test_a_floor_is_built_from_a_bound_rather_than_the_measured_dispersion() -> None:
    # The measured dispersion sits in the middle of its own sampling
    # distribution, so a floor built directly on it is too narrow about half
    # the time. Every floor is built from an upper limit instead.
    assert bounded_floor(0.1, degrees_of_freedom=5)[1] > 1.96 * math.sqrt(2) * 0.1


def test_a_thinner_estimate_is_bounded_further_from_what_it_measured() -> None:
    # Fewer replicates do not make a spread smaller, only less certain, so the
    # bound has to widen as the degrees of freedom fall. This is what makes
    # more replicates the only honest way to narrow a floor.
    bounds = [dispersion_bound(1.0, degrees_of_freedom=df) for df in (2, 5, 9, 29)]

    assert bounds == sorted(bounds, reverse=True)
    assert all(bound > 1.0 for bound in bounds)


def test_the_dispersion_bound_matches_the_chi_squared_limit() -> None:
    # Checked against published chi-squared quantiles rather than against the
    # implementation, since the whole value of the bound is that it is the
    # right number and not merely a consistently larger one.
    assert dispersion_bound(1.0, degrees_of_freedom=5) == pytest.approx(
        math.sqrt(5 / 1.1454763), rel=1e-6
    )
    assert dispersion_bound(1.0, degrees_of_freedom=9) == pytest.approx(
        math.sqrt(9 / 3.3251129), rel=1e-6
    )
    assert dispersion_bound(1.0, degrees_of_freedom=5, confidence=0.9) == pytest.approx(
        math.sqrt(5 / 1.6103080), rel=1e-6
    )


def test_a_bound_needs_a_spread_to_bound() -> None:
    with pytest.raises(NoiseCharacterizationError, match="degree of freedom"):
        dispersion_bound(0.1, degrees_of_freedom=0)
    with pytest.raises(NoiseCharacterizationError, match="between zero and one"):
        dispersion_bound(0.1, degrees_of_freedom=5, confidence=1.0)


def test_one_replicate_cannot_produce_a_floor() -> None:
    with pytest.raises(NoiseCharacterizationError, match="at least two"):
        replicate_dispersion([3.5])


def test_replicate_dispersion_is_the_spread_across_seeds() -> None:
    dispersion = replicate_dispersion([3.0, 3.2, 3.4])

    assert dispersion == pytest.approx(0.2)


def test_bootstrap_resamples_games_rather_than_positions(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    # Every game agrees, so no resample of games can move the mean at all.
    identical = bootstrap_dispersions(
        _totals([2.0] * 20),
        component=component,
        seed=7,
        source="identical games",
        resamples=200,
    )
    spread = _only(
        bootstrap_dispersions(
            _totals([1.0, 2.0, 3.0, 4.0] * 5),
            component=component,
            seed=7,
            source="spread games",
            resamples=200,
        )
    )

    # No dispersion at all rather than one of zero: the resample observed that
    # it could not move this metric, not that a wider draw could not, and a zero
    # would clear every later delta.
    assert identical == {}
    assert spread.value > 0.0
    assert spread.kind == "data-sampling"


def test_a_bootstrap_bound_rests_on_the_games_rather_than_the_resamples(
    move_prediction_component: Digest,
) -> None:
    # Resamples are drawn for free from the same games, so counting them as
    # replicates would buy near-certainty about the dispersion by raising a
    # number the caller chose. Only more games may narrow the bound.
    component = move_prediction_component()
    totals = _totals([1.0, 2.0, 3.0, 4.0] * 5)
    draw = partial(
        bootstrap_dispersions, component=component, seed=7, source="resample count"
    )

    few = _only(draw(totals, resamples=200))
    many = _only(draw(totals, resamples=2_000))
    wider = _only(draw(_totals([1.0, 2.0, 3.0, 4.0] * 20), resamples=200))

    # The measured spread is what more resamples read more finely; the bound
    # over it is what only more games narrow.
    assert many.bound == pytest.approx(few.bound, rel=0.05)
    assert self_combined_floor(wider) < self_combined_floor(few)


def test_a_same_weights_delta_stays_inside_a_bounded_floor() -> None:
    # The case that motivated the bound, with the dispersions three separate
    # characterizations of one checkpoint's p50 latency actually produced at
    # six readings across three processes, against the value twelve readings
    # put the truth near. The unluckiest of the three understates the spread by
    # a third, and a floor built straight on it is narrower than the delta that
    # same machine produces with nothing changed.
    truth = 0.21
    measured = (0.14, 0.29, 0.72)
    honest = 1.96 * math.sqrt(2) * truth

    for estimate in measured:
        naive = 1.96 * math.sqrt(2) * estimate
        bounded = bounded_floor(estimate, degrees_of_freedom=5)[1]
        assert bounded >= honest
        if estimate < truth:
            assert naive < honest


def test_bootstrap_output_is_deterministic_for_one_seed(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    totals = _totals([1.0, 2.0, 3.5, 0.5, 2.5, 4.0])
    draw = partial(bootstrap_dispersions, totals, component=component, source="seeded")

    first = draw(seed=11, resamples=150)
    again = draw(seed=11, resamples=150)
    different = draw(seed=12, resamples=150)

    assert first == again
    assert _only(first).value != _only(different).value


def test_a_bootstrap_floor_lands_on_the_series_it_qualifies(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()

    dispersions = bootstrap_dispersions(
        _totals([1.0, 2.0, 3.0]),
        component=component,
        seed=3,
        source="one series",
        resamples=100,
    )

    assert set(dispersions) == {series_fingerprint(METRIC, component)}


def test_bootstrapping_needs_more_than_one_game(
    move_prediction_component: Digest,
) -> None:
    with pytest.raises(NoiseCharacterizationError, match="at least two scored games"):
        bootstrap_dispersions(
            _totals([1.0]),
            component=move_prediction_component(),
            seed=1,
            source="one game",
            resamples=100,
        )


def test_a_metric_absent_from_a_game_is_still_bootstrapped(
    move_prediction_component: Digest,
) -> None:
    # A rule-case slice only counts the games that realized it. Its floor is
    # estimated from those games rather than diluted by the ones that did not.
    component = move_prediction_component()
    totals = (
        GameTotals(
            game_id=1,
            metrics={
                METRIC: MetricTotal(total=8.0, positions=4),
                OTHER_METRIC: MetricTotal(total=1.0, positions=1),
            },
        ),
        GameTotals(game_id=2, metrics={METRIC: MetricTotal(total=4.0, positions=4)}),
        GameTotals(
            game_id=3,
            metrics={
                METRIC: MetricTotal(total=12.0, positions=4),
                OTHER_METRIC: MetricTotal(total=3.0, positions=1),
            },
        ),
    )

    dispersions = bootstrap_dispersions(
        totals,
        component=component,
        seed=5,
        source="sliced games",
        resamples=200,
    )

    assert set(dispersions) == {
        series_fingerprint(METRIC, component),
        series_fingerprint(OTHER_METRIC, component),
    }
    assert dispersions[series_fingerprint(OTHER_METRIC, component)].value > 0.0


def test_sampling_noise_sizes_the_games_an_axis_needs() -> None:
    dispersion = measured_dispersion(
        0.04, kind="data-sampling", degrees_of_freedom=999, units=1_000
    )
    floor = self_combined_floor(dispersion)

    # A floor shrinks with the square root of the games behind it, so resolving
    # a quarter of the measured floor takes sixteen times the games.
    assert games_to_resolve(dispersion, effect=floor) == 1_000
    assert games_to_resolve(dispersion, effect=floor / 4) == 16_000


def test_a_spread_that_does_not_scale_cannot_size_an_input() -> None:
    # A spread read over games a reading generated, rather than over a draw
    # from a population, does not shrink with a larger pool, so extrapolating
    # it by the inverse square root would answer a question nobody asked.
    scaling = measured_dispersion(
        0.04, kind="data-sampling", degrees_of_freedom=9, units=10
    )
    unscaled = measured_dispersion(0.04, kind="evaluation", degrees_of_freedom=9)

    with pytest.raises(NoiseCharacterizationError, match="does not scale"):
        games_to_resolve(unscaled, effect=0.01)
    with pytest.raises(NoiseCharacterizationError, match="finite and positive"):
        games_to_resolve(scaling, effect=0.0)


def test_a_characterization_round_trips_through_the_store(
    tmp_path: Path,
    move_prediction_component: Digest,
) -> None:
    store = ResultsStore(tmp_path / "results")
    characterization = build_characterization(
        kind="training",
        method="independent-replicates",
        replicates=5,
        source="five smoke-scale seeds",
        training=TRAINING_SCOPE,
        floors=[_entry(move_prediction_component())],
        recorded_at=RECORDED_AT,
    )

    path = store.append_characterization(characterization)
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent == store.floors_directory
    assert raw["kind"] == "training"
    assert store.characterizations() == (characterization,)
    # Recording the same characterization twice is idempotent.
    assert store.append_characterization(characterization) == path


def test_a_different_characterization_cannot_take_a_recorded_identity(
    tmp_path: Path,
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    store = ResultsStore(tmp_path / "results")
    characterization = build_characterization(
        kind="training",
        method="independent-replicates",
        replicates=5,
        source="five seeds",
        training=TRAINING_SCOPE,
        floors=[_entry(component)],
        recorded_at=RECORDED_AT,
    )
    store.append_characterization(characterization)
    forged = characterization.model_copy(
        update={"floors": (_entry(component, floor=99.0),)}
    )

    with pytest.raises(ResultsStoreError, match="already recorded"):
        store.append_characterization(forged)


def test_a_floor_is_found_through_the_series_it_was_characterized_on(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    index = NoiseFloorIndex(
        [
            build_characterization(
                kind="training",
                method="independent-replicates",
                replicates=3,
                source="three seeds",
                training=TRAINING_SCOPE,
                floors=[_entry(component, floor=0.2)],
                recorded_at=RECORDED_AT,
            )
        ]
    )

    found = index.floors(
        METRIC,
        series_fingerprint(METRIC, component),
        trainings=(TRAINING_SCOPE, TRAINING_SCOPE),
    )

    assert [floor.value for floor in found] == [0.2]
    assert found[0].kind == "training"
    assert found[0].source == "three seeds"


def test_a_stale_floor_stops_applying_rather_than_lingering(
    move_prediction_component: Digest,
    scored_row: Callable[..., dict[str, object]],
) -> None:
    # A floor characterized on one pool must not be applied to a measurement
    # taken over different games; the fingerprint is what makes that automatic.
    original = move_prediction_component()
    regenerated = move_prediction_component(
        [scored_row(1), scored_row(2), scored_row(3)]
    )
    index = NoiseFloorIndex(
        [
            build_characterization(
                kind="data-sampling",
                method="bootstrap-over-games",
                replicates=1_000,
                source="the previous pool generation",
                floors=[_entry(original)],
                recorded_at=RECORDED_AT,
            )
        ]
    )

    assert index.floors(METRIC, series_fingerprint(METRIC, original))
    assert index.floors(METRIC, series_fingerprint(METRIC, regenerated)) == ()


def test_a_bridged_series_keeps_the_floor_characterized_on_either_side(
    move_prediction_component: Digest,
    scored_row: Callable[..., dict[str, object]],
) -> None:
    original = move_prediction_component()
    regenerated = move_prediction_component(
        [scored_row(1), scored_row(2), scored_row(3)]
    )
    bridges = BridgeIndex(
        [
            build_bridge(
                from_fingerprint=series_fingerprint(METRIC, original),
                to_fingerprint=series_fingerprint(METRIC, regenerated),
                reason="the digest moved for a reason independent of the metric",
                author="maintainer",
            )
        ]
    )
    index = NoiseFloorIndex(
        [
            build_characterization(
                kind="data-sampling",
                method="bootstrap-over-games",
                replicates=1_000,
                source="the earlier generation",
                floors=[_entry(original)],
                recorded_at=RECORDED_AT,
            )
        ],
        bridges,
    )

    assert index.floors(METRIC, series_fingerprint(METRIC, regenerated))


def test_a_re_characterized_floor_replaces_the_older_one(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    index = NoiseFloorIndex(
        [
            build_characterization(
                kind="training",
                method="independent-replicates",
                replicates=3,
                source="three seeds",
                training=TRAINING_SCOPE,
                floors=[_entry(component, floor=0.5)],
                recorded_at=RECORDED_AT,
            ),
            build_characterization(
                kind="training",
                method="independent-replicates",
                replicates=6,
                source="six seeds",
                training=TRAINING_SCOPE,
                floors=[_entry(component, floor=0.2)],
                recorded_at=datetime(2026, 7, 8, 12, 0, tzinfo=UTC),
            ),
        ]
    )

    found = index.floors(
        METRIC,
        series_fingerprint(METRIC, component),
        trainings=(TRAINING_SCOPE, TRAINING_SCOPE),
    )

    assert [floor.value for floor in found] == [0.2]


def test_kinds_are_kept_apart_rather_than_conflated(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    index = NoiseFloorIndex(
        [
            build_characterization(
                kind="data-sampling",
                method="bootstrap-over-games",
                replicates=1_000,
                source="one pool",
                floors=[_entry(component, floor=0.01)],
                recorded_at=RECORDED_AT,
            ),
            build_characterization(
                kind="training",
                method="independent-replicates",
                replicates=4,
                source="four seeds",
                training=TRAINING_SCOPE,
                floors=[_entry(component, floor=0.3)],
                recorded_at=RECORDED_AT,
            ),
        ]
    )

    found = index.floors(
        METRIC,
        series_fingerprint(METRIC, component),
        trainings=(TRAINING_SCOPE, TRAINING_SCOPE),
    )

    assert {floor.kind for floor in found} == {"data-sampling", "training"}


def test_an_uncharacterized_series_resolves_to_nothing(
    move_prediction_component: Digest,
) -> None:
    index = NoiseFloorIndex()

    assert (
        index.floors(METRIC, series_fingerprint(METRIC, move_prediction_component()))
        == ()
    )


def test_replicate_floors_carry_the_series_they_were_measured_on(
    move_prediction_component: Digest,
) -> None:
    fingerprint = series_fingerprint(METRIC, move_prediction_component())

    entries = replicate_floors({METRIC: [(fingerprint, 3.0), (fingerprint, 3.4)]})

    assert entries[0].metric == METRIC
    assert entries[0].fingerprint == fingerprint
    assert entries[0].dispersion == pytest.approx(replicate_dispersion([3.0, 3.4]))


def test_sampling_noise_travels_on_the_reading_that_measured_it(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()

    dispersions = sampling_dispersions(
        _totals([1.0, 2.0, 3.0, 4.0]),
        component=component,
        config=NoiseConfig(resamples=200, seed=4),
        source="bootstrap over 4 game(s)",
    )

    dispersion = dispersions[series_fingerprint(METRIC, component)]
    assert dispersion.kind == "data-sampling"
    assert dispersion.estimator == BOOTSTRAP_METHOD
    assert dispersion.source == "bootstrap over 4 game(s)"
    # Both quantities are kept: the measured spread describes the sample, and
    # the bound is what a floor is combined from.
    assert dispersion.bound > dispersion.value
    assert dispersion.bound == pytest.approx(
        dispersion_bound(
            dispersion.value, degrees_of_freedom=3, confidence=DEFAULT_CONFIDENCE
        )
    )


def test_a_bound_below_the_dispersion_it_bounds_is_refused(
    move_prediction_component: Digest,
) -> None:
    # A stored bound is only meaningful if it is conservative. Writing one
    # under the dispersion would record a floor claiming a confidence its own
    # inputs contradict.
    component = move_prediction_component()

    with pytest.raises(ValueError, match="not a conservative limit"):
        FloorEntry(
            metric=METRIC,
            fingerprint=series_fingerprint(METRIC, component),
            floor=0.4,
            dispersion=0.2,
            dispersion_bound=0.1,
            degrees_of_freedom=5,
        )


def _execution_scope(*, device_name: str = "fixture-laptop") -> ExecutionRecord:
    """Return one machine's execution record for the efficiency workload."""

    return execution_reference(
        device="cpu",
        device_name=device_name,
        precision="float32",
        torch_version="2.7.0",
        platform_key="Fixture-x86",
        platform="fixture-1.2.3",
        cpu_threads=8,
        workload={"latency_reference_plies": 40},
    )


def _execution_characterization(
    *,
    device_name: str = "fixture-laptop",
    floor: float = 0.4,
) -> NoiseCharacterization:
    execution = _execution_scope(device_name=device_name)
    return build_characterization(
        kind="execution",
        method=PROCESS_REPLICATE_METHOD,
        replicates=6,
        processes=3,
        source="three processes on this machine",
        execution=execution,
        floors=[
            FloorEntry(
                metric=EFFICIENCY_METRIC,
                fingerprint=series_fingerprint(
                    EFFICIENCY_METRIC,
                    None,
                    execution.workload_component(),
                ),
                floor=floor,
                dispersion=floor / 2,
                dispersion_bound=floor / 2,
                degrees_of_freedom=2,
                within_process_dispersion=floor / 8,
            )
        ],
        recorded_at=RECORDED_AT,
    )


def _efficiency_entry(floor: float = 0.4) -> FloorEntry:
    """Return one efficiency floor already on its own workload-scoped series."""

    return FloorEntry(
        metric=EFFICIENCY_METRIC,
        fingerprint=series_fingerprint(
            EFFICIENCY_METRIC,
            None,
            _execution_scope().workload_component(),
        ),
        floor=floor,
        dispersion=floor / 2,
        dispersion_bound=floor / 2,
        degrees_of_freedom=2,
    )


def test_an_execution_floor_must_record_the_machine_it_describes() -> None:
    """Without it the floor would be applied to every machine that ran the series."""

    with pytest.raises(NoiseCharacterizationError, match="machine and a workload"):
        build_characterization(
            kind="execution",
            method=PROCESS_REPLICATE_METHOD,
            replicates=6,
            processes=3,
            source="three processes",
            floors=[_efficiency_entry()],
            recorded_at=RECORDED_AT,
        )


def test_a_floor_that_does_not_depend_on_the_machine_carries_no_scope() -> None:
    with pytest.raises(NoiseCharacterizationError, match="no execution scope"):
        build_characterization(
            kind="training",
            method="independent-replicates",
            replicates=3,
            source="three seeds",
            execution=_execution_scope(),
            floors=[_efficiency_entry()],
            recorded_at=RECORDED_AT,
        )


def _training_characterization(
    component: DataComponent,
    *,
    training: str = TRAINING_SCOPE,
    floor: float = 0.3,
) -> NoiseCharacterization:
    return build_characterization(
        kind="training",
        method="independent-replicates",
        replicates=4,
        source="four seeds of one configuration",
        training=training,
        floors=[_entry(component, floor=floor)],
        recorded_at=RECORDED_AT,
    )


def test_a_training_floor_must_record_the_configuration_it_describes(
    move_prediction_component: Digest,
) -> None:
    """Without it the floor would qualify every configuration on the pool.

    Decisions 0018 and 0021 keep the training run out of series identity on
    purpose, so the fingerprint alone cannot stop one configuration's seed
    variance from qualifying a delta between models of another size entirely.
    """

    with pytest.raises(NoiseCharacterizationError, match="replicates shared"):
        build_characterization(
            kind="training",
            method="independent-replicates",
            replicates=4,
            source="four seeds",
            floors=[_entry(move_prediction_component())],
            recorded_at=RECORDED_AT,
        )


def test_a_floor_that_does_not_depend_on_what_was_trained_carries_no_scope(
    move_prediction_component: Digest,
) -> None:
    with pytest.raises(NoiseCharacterizationError, match="no training scope"):
        build_characterization(
            kind="data-sampling",
            method="bootstrap-over-games",
            replicates=1_000,
            source="one pool",
            training=TRAINING_SCOPE,
            floors=[_entry(move_prediction_component())],
            recorded_at=RECORDED_AT,
        )


def test_a_training_floor_does_not_travel_to_another_configuration(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    index = NoiseFloorIndex([_training_characterization(component)])
    fingerprint = series_fingerprint(METRIC, component)

    assert index.floors(METRIC, fingerprint, trainings=(TRAINING_SCOPE,) * 2)
    assert (
        index.floors(METRIC, fingerprint, trainings=(OTHER_TRAINING_SCOPE,) * 2) == ()
    )


def test_a_training_floor_qualifies_a_delta_against_the_configuration_it_measured(
    move_prediction_component: Digest,
) -> None:
    """The control-arm comparison, whose two sides differ by construction.

    The change under test is what makes the treatment arm's identity differ, so
    requiring both sides to match would refuse the one delta a training floor
    exists to qualify.
    """

    component = move_prediction_component()
    index = NoiseFloorIndex([_training_characterization(component)])
    fingerprint = series_fingerprint(METRIC, component)

    control_then_treatment = index.floors(
        METRIC, fingerprint, trainings=(TRAINING_SCOPE, OTHER_TRAINING_SCOPE)
    )
    treatment_then_control = index.floors(
        METRIC, fingerprint, trainings=(OTHER_TRAINING_SCOPE, TRAINING_SCOPE)
    )

    assert [floor.kind for floor in control_then_treatment] == ["training"]
    # Which operand is the baseline is a recording order, not a claim about
    # which configuration the floor describes.
    assert treatment_then_control == control_then_treatment


def test_replicates_that_do_not_share_a_configuration_characterize_nothing(
    recorded_result: Callable[..., ResultEnvelope],
) -> None:
    """The scope on a stored floor is a fact about its replicates, not a claim."""

    here = recorded_result(training_sha256=TRAINING_SCOPE)
    elsewhere = recorded_result(training_sha256=OTHER_TRAINING_SCOPE)
    unrecorded = recorded_result(training_sha256=None)

    assert training_scope([here, here]) == TRAINING_SCOPE
    with pytest.raises(NoiseCharacterizationError, match="records no training"):
        training_scope([here, unrecorded])
    with pytest.raises(NoiseCharacterizationError, match="different configurations"):
        training_scope([here, elsewhere])


def test_a_reading_with_no_training_identity_borrows_no_training_floor(
    move_prediction_component: Digest,
) -> None:
    """Unknown noise is the honest answer for a reading recorded before the scope."""

    component = move_prediction_component()
    index = NoiseFloorIndex([_training_characterization(component)])
    fingerprint = series_fingerprint(METRIC, component)

    assert index.floors(METRIC, fingerprint) == ()
    assert index.floors(METRIC, fingerprint, trainings=(None, None)) == ()


def test_a_scope_mismatch_withholds_one_kind_rather_than_the_reading(
    move_prediction_component: Digest,
) -> None:
    """Only the scoped kind is refused; an unscoped one is valid on its series."""

    component = move_prediction_component()
    index = NoiseFloorIndex(
        [
            _training_characterization(component),
            build_characterization(
                kind="data-sampling",
                method=BOOTSTRAP_METHOD,
                replicates=1_000,
                source="one pool",
                floors=[_entry(component, floor=0.01)],
                recorded_at=RECORDED_AT,
            ),
        ]
    )
    fingerprint = series_fingerprint(METRIC, component)

    found = index.floors(
        METRIC,
        fingerprint,
        trainings=(OTHER_TRAINING_SCOPE,) * 2,
    )

    assert [floor.kind for floor in found] == ["data-sampling"]


def test_an_execution_floor_applies_where_it_was_measured() -> None:
    characterization = _execution_characterization()
    index = NoiseFloorIndex([characterization])
    execution = _execution_scope()
    fingerprint = series_fingerprint(
        EFFICIENCY_METRIC,
        None,
        execution.workload_component(),
    )

    found = index.floors(
        EFFICIENCY_METRIC,
        fingerprint,
        executions=(execution, execution),
    )

    assert [floor.kind for floor in found] == ["execution"]
    assert found[0].value == pytest.approx(0.4)


def test_an_execution_floor_does_not_travel_to_another_machine() -> None:
    """Decision 0018 keeps the machine out of the series; the noise is the machine.

    The series is deliberately continuous across hardware so long-run drift
    stays answerable, which means the fingerprint alone cannot stop a laptop's
    floor from qualifying a workstation's delta. Reporting the noise as unknown
    there is the honest answer.
    """

    index = NoiseFloorIndex([_execution_characterization(device_name="laptop")])
    elsewhere = _execution_scope(device_name="workstation")
    fingerprint = series_fingerprint(
        EFFICIENCY_METRIC,
        None,
        elsewhere.workload_component(),
    )

    assert index.floors(EFFICIENCY_METRIC, fingerprint) == ()
    assert (
        index.floors(
            EFFICIENCY_METRIC,
            fingerprint,
            executions=(elsewhere, elsewhere),
        )
        == ()
    )


def test_a_delta_spanning_two_machines_has_no_execution_floor() -> None:
    index = NoiseFloorIndex([_execution_characterization(device_name="laptop")])
    here = _execution_scope(device_name="laptop")
    there = _execution_scope(device_name="workstation")
    fingerprint = series_fingerprint(
        EFFICIENCY_METRIC,
        None,
        here.workload_component(),
    )

    assert index.floors(EFFICIENCY_METRIC, fingerprint, executions=(here, here))
    assert index.floors(EFFICIENCY_METRIC, fingerprint, executions=(here, there)) == ()


def test_an_execution_floor_still_needs_both_sides_of_the_delta() -> None:
    """A machine is a condition, not a null distribution, so the rules differ.

    A training floor describes the spread one operand would have shown under
    another seed, and a delta asks whether the other fell outside it. Nothing
    equivalent is true of a machine: a delta spanning two of them is described
    by neither machine's floor.
    """

    index = NoiseFloorIndex([_execution_characterization(device_name="laptop")])
    here = _execution_scope(device_name="laptop")
    there = _execution_scope(device_name="workstation")
    fingerprint = series_fingerprint(EFFICIENCY_METRIC, None, here.workload_component())

    assert index.floors(EFFICIENCY_METRIC, fingerprint, executions=(here, here))
    assert index.floors(EFFICIENCY_METRIC, fingerprint, executions=(here, there)) == ()


def test_execution_dispersion_needs_more_than_one_process() -> None:
    with pytest.raises(NoiseCharacterizationError, match="at least two processes"):
        process_dispersion([[10.0, 10.2, 10.1]])


def test_execution_dispersion_separates_the_process_from_the_repeat() -> None:
    """The total is what a report compares; the within is where the noise lives."""

    dispersion = process_dispersion([[10.0, 10.2], [12.0, 12.2]])

    assert dispersion.within is not None
    assert dispersion.within == pytest.approx(0.2 / math.sqrt(2))
    assert dispersion.total == pytest.approx(
        replicate_dispersion([10.0, 10.2, 12.0, 12.2])
    )
    assert dispersion.total > dispersion.within


def test_execution_floors_land_on_the_series_they_qualify() -> None:
    execution = _execution_scope()
    fingerprint = series_fingerprint(
        EFFICIENCY_METRIC,
        None,
        execution.workload_component(),
    )

    (entry,) = process_replicate_floors(
        {EFFICIENCY_METRIC: [[10.0, 10.2], [12.0, 12.2]]},
        fingerprints={EFFICIENCY_METRIC: fingerprint},
    )

    assert entry.fingerprint == fingerprint
    # Four readings, but two processes: the process is the independent
    # replicate, so the bound rests on one degree of freedom rather than three.
    assert entry.degrees_of_freedom == 1
    assert entry.floor == pytest.approx(
        bounded_floor(
            replicate_dispersion([10.0, 10.2, 12.0, 12.2]),
            degrees_of_freedom=1,
        )[1]
    )


def test_a_floor_with_no_series_named_for_it_is_refused() -> None:
    with pytest.raises(NoiseCharacterizationError, match="no series fingerprint"):
        process_replicate_floors(
            {EFFICIENCY_METRIC: [[10.0], [12.0]]},
            fingerprints={},
        )
