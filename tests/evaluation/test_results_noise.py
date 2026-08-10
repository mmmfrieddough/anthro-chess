"""Dispersion estimation, the floor built from it, and the sizing question."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import partial

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
    DataComponent,
    MetricDispersion,
    NoiseCharacterizationError,
    bounded_floor,
    combined_floor,
    dispersion_bound,
    games_to_resolve,
    measured_dispersion,
    process_dispersion,
    replicate_dispersion,
    self_combined_floor,
    series_fingerprint,
)
from anthro_chess.evaluation.results.metrics import (
    INFERENCE_MOVE_LATENCY_BY_PERCENTILE,
)

Digest = Callable[..., DataComponent]

RECORDED_AT = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
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


def test_a_floor_is_the_delta_two_independent_measurements_produce() -> None:
    # A floor has to be comparable to a reported delta, and the difference of
    # two independent measurements is wider than either one by sqrt(2). Held at
    # a confidence of one half, where the chi-squared bound is close enough to
    # the estimate to leave the sqrt(2) visible on its own.
    bound = dispersion_bound(0.1, degrees_of_freedom=200, confidence=0.5)
    floor = bounded_floor(0.1, degrees_of_freedom=200, coverage=1.0, confidence=0.5)
    assert floor == pytest.approx(math.sqrt(2) * bound)
    assert bounded_floor(0.0, degrees_of_freedom=5) == 0.0
    single = bounded_floor(0.1, degrees_of_freedom=5, coverage=1.0)
    doubled = bounded_floor(0.1, degrees_of_freedom=5, coverage=2.0)
    assert doubled == pytest.approx(2 * single)


def test_a_delta_floor_combines_the_two_readings_it_compares() -> None:
    # The variance of a difference is the sum of the two variances, so two
    # readings that happen to agree reduce to sqrt(2) times either one, and two
    # that do not are dominated by the noisier of them.
    narrow = measured_dispersion(0.1, degrees_of_freedom=200)
    wide = measured_dispersion(1.19, degrees_of_freedom=200)

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
    dispersion = measured_dispersion(0.3, degrees_of_freedom=9)

    assert self_combined_floor(dispersion) == pytest.approx(
        combined_floor(dispersion, dispersion)
    )


def test_a_floor_is_built_from_a_bound_rather_than_the_measured_dispersion() -> None:
    # The measured dispersion sits in the middle of its own sampling
    # distribution, so a floor built directly on it is too narrow about half
    # the time. Every floor is built from an upper limit instead.
    assert bounded_floor(0.1, degrees_of_freedom=5) > 1.96 * math.sqrt(2) * 0.1


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
    assert spread.estimator == BOOTSTRAP_METHOD


@pytest.mark.parametrize("positions", [(3, 3, 3), (121, 42, 149)])
def test_agreeing_games_are_omitted_even_where_their_rate_is_inexact(
    move_prediction_component: Digest,
    positions: tuple[int, ...],
) -> None:
    # The same omission as test_bootstrap_resamples_games_rather_than_positions,
    # at a rate binary floating point cannot hold.
    # Every resample recomputes the one value from differently-rounded sums, so
    # it lands within a few ulp of itself instead of on itself, and a test for
    # exactly zero would record a floor no comparison could fail to clear.
    # Unequal position counts are the case that reaches several distinct
    # quotients rather than one, so identity is not the test either.
    rate = 0.9233579079706695
    agreeing = tuple(
        GameTotals(
            game_id=game_id,
            metrics={METRIC: MetricTotal(total=rate * count, positions=count)},
        )
        for game_id, count in enumerate(positions)
    )

    identical = bootstrap_dispersions(
        agreeing,
        component=move_prediction_component(),
        seed=5,
        source="agreeing games at an inexact rate",
        resamples=200,
    )

    assert identical == {}


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
        bounded = bounded_floor(estimate, degrees_of_freedom=5)
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
    sliced = dispersions[series_fingerprint(OTHER_METRIC, component)]
    pooled = dispersions[series_fingerprint(METRIC, component)]
    assert sliced.value > 0.0
    assert (sliced.units, pooled.units) == (2, 3)
    assert sliced.bound == pytest.approx(
        dispersion_bound(sliced.value, degrees_of_freedom=1)
    )
    assert pooled.bound == pytest.approx(
        dispersion_bound(pooled.value, degrees_of_freedom=2)
    )


def test_a_metric_one_game_realized_reports_no_dispersion(
    move_prediction_component: Digest,
) -> None:
    # One game is one replicate, and a single replicate observes no spread for
    # a bound to rest on.
    component = move_prediction_component()
    totals = (
        GameTotals(
            game_id=1,
            metrics={
                METRIC: MetricTotal(total=3.0, positions=3),
                OTHER_METRIC: MetricTotal(total=1.0, positions=3),
            },
        ),
        GameTotals(game_id=2, metrics={METRIC: MetricTotal(total=9.0, positions=3)}),
        GameTotals(game_id=3, metrics={METRIC: MetricTotal(total=15.0, positions=3)}),
    )

    dispersions = bootstrap_dispersions(
        totals,
        component=component,
        seed=0,
        source="one realizing game",
        resamples=200,
    )

    assert set(dispersions) == {series_fingerprint(METRIC, component)}


def test_sampling_noise_sizes_the_games_an_axis_needs() -> None:
    dispersion = measured_dispersion(0.04, degrees_of_freedom=999, units=1_000)
    floor = self_combined_floor(dispersion)

    # A floor shrinks with the square root of the games behind it, so resolving
    # a quarter of the measured floor takes sixteen times the games.
    assert games_to_resolve(dispersion, effect=floor) == 1_000
    assert games_to_resolve(dispersion, effect=floor / 4) == 16_000


def test_a_spread_that_does_not_scale_cannot_size_an_input() -> None:
    # A spread read over games a reading generated, rather than over a draw
    # from a population, does not shrink with a larger pool, so extrapolating
    # it by the inverse square root would answer a question nobody asked.
    scaling = measured_dispersion(0.04, degrees_of_freedom=9, units=10)
    unscaled = measured_dispersion(0.04, degrees_of_freedom=9)

    with pytest.raises(NoiseCharacterizationError, match="does not scale"):
        games_to_resolve(unscaled, effect=0.01)
    with pytest.raises(NoiseCharacterizationError, match="finite and positive"):
        games_to_resolve(scaling, effect=0.0)


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


def test_a_bound_below_the_dispersion_it_bounds_is_refused() -> None:
    # A stored bound is only meaningful if it is conservative. Writing one
    # under the dispersion would record a floor claiming a confidence its own
    # inputs contradict.
    with pytest.raises(ValueError, match="not a conservative limit"):
        MetricDispersion(value=0.2, bound=0.1)


def test_execution_dispersion_needs_more_than_one_process() -> None:
    # One process cannot observe what a second would pay for again, which is
    # the whole component this estimate exists to see.
    with pytest.raises(NoiseCharacterizationError, match="at least two processes"):
        process_dispersion([10.0])


def test_execution_dispersion_is_the_spread_across_processes() -> None:
    assert process_dispersion([10.0, 12.0]) == pytest.approx(
        replicate_dispersion([10.0, 12.0])
    )
