"""The human-reference curve comparison shape several benchmarks share."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence

import numpy as np
import pytest

from anthro_chess.evaluation.curves import (
    CURVE_BOOTSTRAP_METHOD,
    CURVE_COMPARISON_VERSION,
    CURVE_DETERMINISTIC_METHOD,
    HUMAN_REFERENCE_LABEL,
    CurveComparison,
    CurveComparisonError,
    CurveMetrics,
    CurveQuantity,
    CurveSpec,
    Observation,
    PairedRateObservation,
    RatingResponse,
    _categories,
    _read,
    _reduce,
    _Side,
    compare_curves,
    compare_reference_rate,
    curve_overlays,
    distribution_distance,
    estimate_curve,
    rating_grid,
    select_neighbours,
)
from anthro_chess.evaluation.results import (
    BridgeIndex,
    DataComponent,
    MetricCost,
    MetricDefinition,
    MetricDirection,
    build_bridge,
    register_metric,
)
from anthro_chess.evaluation.results.metrics import (
    GENERATED_PLAY_FAMILY,
    MOVE_PREDICTION_PROJECTION,
)

Digest = Callable[..., DataComponent]

#: Tricube weight at half the local radius, which is what the exact-arithmetic
#: tests below expect a neighbourhood to give a game halfway out.
TRICUBE_AT_HALF = (1.0 - 0.5**3) ** 3

GRID = rating_grid(1200.0, 2000.0, 5)

SCALAR_SPEC = CurveSpec(
    name="game-length",
    version=1,
    quantity=CurveQuantity.SCALAR,
    neighbours=20,
    grid=GRID,
)

CATEGORICAL_SPEC = CurveSpec(
    name="opening-repertoire",
    version=1,
    quantity=CurveQuantity.CATEGORICAL,
    neighbours=25,
    grid=GRID,
)

CONDITIONAL_METRIC = "generated_play.length_curve_distance"
POOLED_METRIC = "generated_play.length_pooled_distance"
VARIATION_METRIC = "generated_play.length_rating_response"


def _length(rating: float, *, slope: float = 0.02, intercept: float = 40.0) -> float:
    """Return the human game length at one rating, before noise."""

    return intercept + slope * (rating - 1600.0)


def _human_reference(
    *,
    seed: int = 1,
    games: int = 500,
    slope: float = 0.02,
) -> tuple[Observation, ...]:
    """Return a reference whose density over the rating range is uneven.

    Real human density peaks in the middle of the range and thins at both
    ends, which is the whole reason the bandwidth has to adapt.
    """

    generator = random.Random(seed)
    observations = []
    for _ in range(games):
        rating = min(2400.0, max(800.0, generator.gauss(1600.0, 220.0)))
        value = _length(rating, slope=slope) + generator.gauss(0.0, 3.0)
        observations.append(Observation(rating, value))
    return tuple(observations)


def _generated(
    value: Callable[[float, random.Random], float],
    *,
    seed: int = 2,
    per_rating: int = 60,
    grid: Sequence[float] = GRID,
) -> tuple[Observation, ...]:
    """Return generated games at each configured rating, uniformly spread."""

    generator = random.Random(seed)
    return tuple(
        Observation(rating, value(rating, generator))
        for rating in grid
        for _ in range(per_rating)
    )


def _compare(
    model: Sequence[Observation],
    *,
    human: Sequence[Observation] | None = None,
    resamples: int = 80,
    references: bool = True,
    model_varies: bool = True,
) -> CurveComparison:
    return compare_curves(
        spec=SCALAR_SPEC,
        human=human if human is not None else _human_reference(),
        model=model,
        resamples=resamples,
        seed=17,
        model_varies=model_varies,
        references=references,
    )


def _register_curve_metrics() -> None:
    """Register the metrics a benchmark would declare for one comparison."""

    for identifier, direction in (
        (CONDITIONAL_METRIC, MetricDirection.LOWER_IS_BETTER),
        (POOLED_METRIC, MetricDirection.LOWER_IS_BETTER),
        (VARIATION_METRIC, MetricDirection.INFORMATIONAL),
    ):
        register_metric(
            MetricDefinition(
                identifier=identifier,
                family=GENERATED_PLAY_FAMILY.identifier,
                direction=direction,
                definition_version=1,
                summary="Fixture metric for one curve comparison.",
                cost=MetricCost.GENERATED,
                projection=MOVE_PREDICTION_PROJECTION.name,
            )
        )


def test_curve_spec_rejects_a_shape_that_cannot_be_measured() -> None:
    with pytest.raises(CurveComparisonError, match="lowercase"):
        CurveSpec(
            name="Game Length",
            version=1,
            quantity=CurveQuantity.SCALAR,
            neighbours=8,
            grid=GRID,
        )
    with pytest.raises(CurveComparisonError, match="two neighbours"):
        CurveSpec(
            name="game-length",
            version=1,
            quantity=CurveQuantity.SCALAR,
            neighbours=1,
            grid=GRID,
        )
    with pytest.raises(CurveComparisonError, match="strictly increase"):
        CurveSpec(
            name="game-length",
            version=1,
            quantity=CurveQuantity.SCALAR,
            neighbours=8,
            grid=(1600.0, 1200.0),
        )
    with pytest.raises(CurveComparisonError, match="two rating points"):
        CurveSpec(
            name="game-length",
            version=1,
            quantity=CurveQuantity.SCALAR,
            neighbours=8,
            grid=(1600.0,),
        )


def test_point_reference_rate_keeps_game_clustered_support() -> None:
    comparison = compare_reference_rate(
        (
            PairedRateObservation(1, True, False, 0.2),
            PairedRateObservation(1, False, True, 0.8),
            PairedRateObservation(2, True, True, 0.6),
        )
    )

    assert comparison.games == 2
    assert comparison.opportunities == 3
    assert comparison.effective_sample_size == pytest.approx(9 / 5)
    assert comparison.human_rate == pytest.approx(2 / 3)
    assert comparison.model_rate == pytest.approx(2 / 3)
    assert comparison.model_probability_mass == pytest.approx(1.6 / 3)
    assert comparison.human_gap == pytest.approx(0.0)


def test_rating_grid_is_evenly_spaced_and_rejects_an_empty_range() -> None:
    assert rating_grid(1200.0, 2000.0, 5) == (1200.0, 1400.0, 1600.0, 1800.0, 2000.0)
    with pytest.raises(CurveComparisonError, match="increasing range"):
        rating_grid(2000.0, 1200.0, 5)
    with pytest.raises(CurveComparisonError, match="at least two points"):
        rating_grid(1200.0, 2000.0, 1)


def test_curve_estimation_smooths_locally_over_the_declared_neighbours() -> None:
    """A point reads its neighbourhood, weighted, and reports how much of one."""

    spec = CurveSpec(
        name="game-length",
        version=1,
        quantity=CurveQuantity.SCALAR,
        neighbours=5,
        grid=(1200.0, 1600.0),
    )
    human = [Observation(1000.0 + 100.0 * step, 10.0 + step) for step in range(11)]
    model = [Observation(1200.0, 12.0), Observation(1600.0, 16.0)]

    comparison = compare_curves(spec=spec, human=human, model=model, resamples=0)
    point = comparison.points[0]

    # Five neighbours of 1200 sit at 0, 100, 100, 200 and 200 rating apart, so
    # the fifth forces a radius of 200 and the two games on it weigh nothing.
    assert point.bandwidth == pytest.approx(200.0)
    assert point.human.games == 3
    assert point.human.value == pytest.approx(12.0)
    expected_effective = (1.0 + 2.0 * TRICUBE_AT_HALF) ** 2 / (
        1.0 + 2.0 * TRICUBE_AT_HALF**2
    )
    assert point.human.effective_sample_size == pytest.approx(expected_effective)
    assert point.human.effective_sample_size < spec.neighbours


def test_effective_sample_size_falls_where_the_reference_is_thin() -> None:
    comparison = _compare(_generated(lambda rating, _: _length(rating)), resamples=0)
    dense = comparison.points[2]
    thin = comparison.points[-1]

    assert dense.rating == 1600.0
    assert thin.rating == 2000.0
    assert dense.bandwidth < thin.bandwidth
    assert dense.human.effective_sample_size > thin.human.effective_sample_size


def test_both_curves_are_estimated_at_the_bandwidth_the_human_side_forces() -> None:
    """The model side is read through the human radius, not its own density."""

    spec = CurveSpec(
        name="game-length",
        version=1,
        quantity=CurveQuantity.SCALAR,
        neighbours=5,
        grid=(1200.0, 1600.0),
    )
    human = [Observation(1000.0 + 100.0 * step, 10.0 + step) for step in range(11)]
    # Two generated games only. Smoothed at their own five-neighbour span they
    # would blend into every point; at the human radius of 200 each is visible
    # at one point and invisible at the other.
    model = [Observation(1150.0, 100.0), Observation(1500.0, 0.0)]

    comparison = compare_curves(spec=spec, human=human, model=model, resamples=0)
    low, high = comparison.points

    assert low.bandwidth == pytest.approx(200.0)
    assert low.model.games == 1
    assert low.model.value == pytest.approx(100.0)
    assert low.distance == pytest.approx(88.0)
    assert high.model.games == 1
    assert high.model.value == pytest.approx(0.0)
    assert high.distance == pytest.approx(16.0)
    assert comparison.conditional_distance == pytest.approx(52.0)


def test_the_bandwidth_does_not_move_with_the_generated_sample() -> None:
    human = _human_reference()
    sparse = _compare(
        _generated(lambda rating, _: _length(rating), per_rating=2),
        human=human,
        resamples=0,
    )
    dense = _compare(
        _generated(lambda rating, _: _length(rating), per_rating=200),
        human=human,
        resamples=0,
    )

    assert [point.bandwidth for point in sparse.points] == [
        point.bandwidth for point in dense.points
    ]


def test_replaying_the_generated_side_reads_as_the_sample_it_replayed() -> None:
    """What licenses collapsing a deterministic suite's replicates.

    A greedy cell's replicates are copies of one game, and a copy carries
    nothing the original did not. The human side forces the bandwidth, so
    duplicating every generated observation scales the weights inside each
    neighbourhood uniformly and leaves both distances where they were.
    """

    model = _generated(
        lambda rating, generator: _length(rating) + generator.gauss(0.0, 3.0)
    )
    once = _compare(model, resamples=0)
    replayed = _compare(model * 3, resamples=0)

    assert replayed.conditional_distance == pytest.approx(once.conditional_distance)
    assert replayed.pooled_distance == pytest.approx(once.pooled_distance)


def test_a_grid_point_no_generated_game_reaches_is_reported_not_averaged() -> None:
    comparison = _compare(
        _generated(lambda rating, _: _length(rating), grid=GRID[:2]),
        resamples=0,
    )

    assert comparison.unsupported_points > 0
    assert comparison.points[0].distance is not None
    assert comparison.points[-1].distance is None
    assert comparison.points[-1].model.games == 0
    assert comparison.points[-1].model.value is None
    # The reported distance is still the mean of what could be compared, and a
    # missing point never counts as agreement.
    assert math.isfinite(comparison.conditional_distance)


def test_a_matching_model_reads_as_matching() -> None:
    comparison = _compare(
        _generated(
            lambda rating, generator: _length(rating) + generator.gauss(0.0, 3.0)
        )
    )

    assert comparison.references is not None
    assert comparison.conditional_distance <= comparison.references.conditional
    assert comparison.pooled_distance <= comparison.references.pooled
    assert comparison.response is RatingResponse.MATCHES


def test_a_peak_survives_the_comparison_rather_than_becoming_a_difference() -> None:
    """The reason both curves take the bandwidth the human side forces.

    Smoothing flattens a peak in proportion to the bandwidth. The human side is
    dense at the peak and thin at the edges while the generated side is uniform
    everywhere, so per-side bandwidths would flatten the two curves by
    different amounts and leave the difference behind as a finding.
    """

    def peaked(rating: float) -> float:
        return 40.0 + 8.0 * math.exp(-(((rating - 1600.0) / 150.0) ** 2))

    generator = random.Random(4)
    human = tuple(
        Observation(rating, peaked(rating) + generator.gauss(0.0, 2.0))
        for rating in (
            min(2400.0, max(800.0, generator.gauss(1600.0, 220.0))) for _ in range(800)
        )
    )
    model = _generated(
        lambda rating, rng: peaked(rating) + rng.gauss(0.0, 2.0),
        per_rating=120,
    )

    comparison = _compare(model, human=human)
    peak = comparison.points[2]

    assert comparison.references is not None
    assert peak.rating == 1600.0
    assert peak.human.value is not None
    assert peak.model.value is not None
    # Both curves are flattened by the same amount at the peak, so what is left
    # of the difference there is noise rather than smoothing.
    assert peak.distance is not None
    assert peak.distance < comparison.references.conditional
    assert comparison.conditional_distance <= comparison.references.conditional
    assert comparison.response is RatingResponse.MATCHES


def test_a_flat_curve_over_the_pooled_human_average_reads_as_such() -> None:
    """The behavioral form of a rating dependency test."""

    comparison = _compare(
        _generated(lambda _, generator: _length(1600.0) + generator.gauss(0.0, 3.0))
    )

    assert comparison.references is not None
    # It behaves like the average human overall...
    assert comparison.pooled_distance <= comparison.references.pooled
    # ...while ignoring the rating input it was given. Flatness is read against
    # the wiggle a response-free model of this size would show anyway, not
    # against zero.
    assert comparison.model_variation <= comparison.references.flat
    assert comparison.human_variation > comparison.references.flat
    assert comparison.conditional_distance > comparison.references.conditional
    assert comparison.response is RatingResponse.AVERAGE_HUMAN


def test_a_shifted_distribution_reads_as_a_pooled_mismatch() -> None:
    comparison = _compare(
        _generated(
            lambda rating, generator: _length(rating) + 8.0 + generator.gauss(0.0, 3.0)
        )
    )

    assert comparison.references is not None
    assert comparison.pooled_distance > comparison.references.pooled
    assert comparison.response is RatingResponse.MISMATCH


def test_a_model_that_responds_the_wrong_way_is_not_reported_as_flat() -> None:
    comparison = _compare(
        _generated(
            lambda rating, generator: (
                _length(rating, slope=-0.02) + generator.gauss(0.0, 3.0)
            )
        )
    )

    assert comparison.references is not None
    assert comparison.pooled_distance <= comparison.references.pooled
    assert comparison.model_variation > comparison.references.flat
    assert comparison.conditional_distance > comparison.references.conditional
    assert comparison.response is RatingResponse.DIVERGENT_RESPONSE


def test_pooled_and_conditional_readings_come_from_one_pass() -> None:
    comparison = _compare(
        _generated(lambda _, generator: _length(1600.0) + generator.gauss(0.0, 3.0)),
        resamples=0,
    )

    # The rating-free reading averages each side's own curve over the grid, so
    # both sides are pooled over one rating mixture rather than over the human
    # corpus on one side and the benchmark's rating grid on the other.
    human_points = [point.human.value for point in comparison.points]
    assert None not in human_points
    assert comparison.human_pooled.value == pytest.approx(
        sum(value for value in human_points if value is not None) / len(human_points)
    )
    assert comparison.human_pooled.games > 0
    # The pooled reading agrees while the conditional one does not, which is
    # the dissociation the shape exists to surface.
    assert comparison.pooled_distance < comparison.conditional_distance


def test_a_categorical_curve_estimates_the_local_distribution() -> None:
    spec = CurveSpec(
        name="opening-repertoire",
        version=1,
        quantity=CurveQuantity.CATEGORICAL,
        neighbours=3,
        grid=(1200.0, 2000.0),
    )
    human = [
        Observation(
            1000.0 + 100.0 * step,
            "sicilian" if 1000.0 + 100.0 * step >= 1600.0 else "other",
        )
        for step in range(11)
    ]
    model = [
        Observation(1200.0, "sicilian"),
        Observation(1200.0, "other"),
        Observation(2000.0, "sicilian"),
    ]

    comparison = compare_curves(spec=spec, human=human, model=model, resamples=0)
    low, high = comparison.points

    assert low.human.value is None
    assert low.human.distribution == {"other": pytest.approx(1.0), "sicilian": 0.0}
    assert low.model.distribution == {
        "other": pytest.approx(0.5),
        "sicilian": pytest.approx(0.5),
    }
    # Total variation, so a half-and-half split against a pure one reads 0.5.
    assert low.distance == pytest.approx(0.5)
    assert high.human.distribution == {"other": 0.0, "sicilian": pytest.approx(1.0)}
    assert high.distance == pytest.approx(0.0)
    assert comparison.conditional_distance == pytest.approx(0.25)


def test_a_categorical_curve_counts_a_family_only_the_model_plays() -> None:
    spec = CurveSpec(
        name="opening-repertoire",
        version=1,
        quantity=CurveQuantity.CATEGORICAL,
        neighbours=2,
        grid=(1200.0, 1600.0),
    )
    human = [Observation(1200.0 + 100.0 * step, "other") for step in range(5)]
    model = [Observation(1200.0, "kings-gambit"), Observation(1600.0, "other")]

    comparison = compare_curves(spec=spec, human=human, model=model, resamples=0)

    assert comparison.points[0].model.distribution == {
        "kings-gambit": pytest.approx(1.0),
        "other": 0.0,
    }
    assert comparison.points[0].distance == pytest.approx(1.0)


def test_a_comparison_carries_the_floor_its_own_distances_are_read_against() -> None:
    comparison = _compare(
        _generated(
            lambda rating, generator: _length(rating) + generator.gauss(0.0, 3.0)
        )
    )

    assert comparison.dispersions is not None
    assert comparison.dispersions.resamples == 80
    for spread in (
        comparison.dispersions.conditional,
        comparison.dispersions.pooled,
        comparison.dispersions.model_variation,
    ):
        assert spread.kind == "evaluation"
        assert spread.value > 0.0
        assert spread.bound >= spread.value
        assert spread.source is not None
        assert SCALAR_SPEC.name in spread.source


@pytest.mark.parametrize("quantity", list(CurveQuantity))
@pytest.mark.parametrize("per_rating,resamples", [(8, 4), (60, 24)])
def test_the_bootstrap_reads_one_human_curve_across_every_resample(
    quantity: CurveQuantity,
    per_rating: int,
    resamples: int,
) -> None:
    """The floor pass broadcasts the point pass's human curve, not a copy of it.

    Only the model side is resampled, so the human curve the floor is read
    against is one row against the model's many. This reaches past the public
    interface because a mis-broadcast produces a floor that is wrong rather than
    absent, and no fixture comparison has an oracle for a floor's value.
    """

    scalar = quantity is CurveQuantity.SCALAR
    spec = SCALAR_SPEC if scalar else CATEGORICAL_SPEC
    generator = random.Random(31)

    def value(rating: float) -> float | str:
        if scalar:
            return _length(rating) + generator.gauss(0.0, 3.0)
        return "sicilian" if generator.random() < rating / 4000.0 else "other"

    human = tuple(
        Observation(item.rating, value(item.rating)) for item in _human_reference()
    )
    model = tuple(
        Observation(rating, value(rating))
        for rating in GRID
        for _ in range(per_rating if rating < GRID[-1] else 2)
    )
    categories = _categories(spec, human, model)
    human_side = _Side.prepare(human, spec, categories)
    model_side = _Side.prepare(model, spec, categories)
    model_weights = (
        np.random.default_rng(13)
        .multinomial(
            model_side.size,
            np.full(model_side.size, 1.0 / model_side.size),
            size=resamples,
        )
        .astype(np.float64)
    )

    point = _read(
        spec,
        human_side,
        model_side,
        np.ones((1, human_side.size)),
        np.ones((1, model_side.size)),
    )
    broadcast = _reduce(spec, point.radii, point.human, model_side, model_weights)
    repeated = _read(
        spec,
        human_side,
        model_side,
        np.ones((resamples, human_side.size)),
        model_weights,
    )

    # A zero stride is the whole optimization: the human curve was estimated
    # once and repeated by view, rather than computed for every replicate.
    assert broadcast.human.values.strides[0] == 0
    # The model side is thin at the top of the grid so that resampling drops
    # that point in some replicates and not others, which is the only case
    # where the one human row meets a support mask varying beneath it. Assert
    # the fixture achieved that rather than trusting it to.
    assert 0 < int(broadcast.supported[:, -1].sum()) < resamples
    # Not bitwise: numpy dispatches the local fit's matrix product differently
    # at one replicate than at many, so the two paths agree to floating-point
    # noise rather than exactly. The floor is a spread over these readings, so
    # only their agreement matters here.
    assert broadcast.conditional == pytest.approx(repeated.conditional, rel=1e-12)
    assert broadcast.pooled == pytest.approx(repeated.pooled, rel=1e-12)
    assert broadcast.model_variation == pytest.approx(
        repeated.model_variation, rel=1e-12
    )


def test_a_model_side_that_cannot_vary_reports_a_floor_of_exactly_zero() -> None:
    """Re-measuring a deterministic reading replays it, so nothing moves.

    Bootstrapping the games instead answers how far a different *draw* would
    have landed, which no delta between two checkpoints read on these same
    games is exposed to.
    """

    model = _generated(
        lambda rating, generator: _length(rating) + generator.gauss(0.0, 3.0)
    )
    comparison = _compare(model, model_varies=False)

    assert comparison.dispersions is not None
    assert comparison.dispersions.method == CURVE_DETERMINISTIC_METHOD
    assert comparison.dispersions.resamples == 0
    for spread in (
        comparison.dispersions.conditional,
        comparison.dispersions.pooled,
        comparison.dispersions.model_variation,
    ):
        assert spread.value == 0.0
        assert spread.bound == 0.0
        assert spread.kind == "evaluation"
        assert spread.source is not None
        assert CURVE_DETERMINISTIC_METHOD in spread.source


def test_a_stated_floor_survives_a_model_side_too_thin_to_bootstrap() -> None:
    """Being replayed is not a sample-size question, so nothing thins it.

    A bootstrap needs games to resample and gives up below two. A reading that
    another run reproduces move for move has a floor of zero however few games
    it played, and withholding one there would report the exactly-known case as
    unknown.
    """

    thin = _generated(lambda rating, _: _length(rating), per_rating=1, grid=GRID[:1])
    comparison = _compare(thin, model_varies=False)

    assert comparison.dispersions is not None
    assert comparison.dispersions.conditional.value == 0.0
    assert comparison.references is None


def test_declaring_the_model_side_fixed_leaves_the_reading_itself_alone() -> None:
    """Only the floor's claim changes, which is the point of the distinction.

    The distances, the null levels, and the curve are all properties of the
    games played. Saying that another run would replay them says nothing about
    what they read at.

    Read at a thin sample, which is where issue #257's defect showed: the
    bootstrap reports a floor of whole plies against a distance that a second
    run would reproduce exactly.
    """

    model = _generated(
        lambda rating, generator: _length(rating) + generator.gauss(0.0, 3.0),
        per_rating=2,
    )
    varying = _compare(model)
    fixed = _compare(model, model_varies=False)

    assert fixed.conditional_distance == varying.conditional_distance
    assert fixed.pooled_distance == varying.pooled_distance
    assert fixed.model_variation == varying.model_variation
    assert fixed.references is not None
    assert varying.references is not None
    assert fixed.references.conditional == varying.references.conditional
    assert varying.dispersions is not None
    assert varying.dispersions.conditional_floor > 1.0


def test_the_reference_size_does_not_move_the_floor() -> None:
    """A floor qualifies a delta, and both checkpoints share one reference.

    Two checkpoints are compared against the identical human games, so the
    reference's own sampling error is common-mode and cancels in their
    difference. Letting it into the floor would inflate every one of them and
    hide real movement, so a reference four times the size must leave the floor
    essentially where it was — only the generated side may move it.
    """

    model = _generated(
        lambda rating, generator: _length(rating) + generator.gauss(0.0, 3.0)
    )
    small = _compare(model, human=_human_reference(games=250))
    large = _compare(model, human=_human_reference(games=1000))

    assert small.dispersions is not None
    assert large.dispersions is not None
    # Not identical: a different reference shifts the bandwidth and so the
    # curve itself. But the spread must not scale with reference size.
    assert large.dispersions.conditional.value == pytest.approx(
        small.dispersions.conditional.value, rel=0.5
    )


def test_a_floor_shrinks_as_the_games_behind_it_grow() -> None:
    small = compare_curves(
        spec=SCALAR_SPEC,
        human=_human_reference(games=200),
        model=_generated(
            lambda rating, generator: _length(rating) + generator.gauss(0.0, 3.0),
            per_rating=20,
        ),
        resamples=80,
        seed=17,
    )
    large = compare_curves(
        spec=SCALAR_SPEC,
        human=_human_reference(games=2000),
        model=_generated(
            lambda rating, generator: _length(rating) + generator.gauss(0.0, 3.0),
            per_rating=200,
        ),
        resamples=80,
        seed=17,
    )

    assert small.dispersions is not None
    assert large.dispersions is not None
    assert large.dispersions.pooled.value < small.dispersions.pooled.value


def test_without_replicates_no_floor_is_invented() -> None:
    comparison = _compare(
        _generated(lambda rating, _: _length(rating)),
        resamples=0,
    )

    assert comparison.dispersions is None
    assert comparison.references is None
    assert comparison.response is RatingResponse.UNKNOWN


def test_measurements_carry_their_floor_into_the_summary_tier(
    move_prediction_component: Digest,
) -> None:
    _register_curve_metrics()
    component = move_prediction_component()
    comparison = _compare(
        _generated(
            lambda rating, generator: _length(rating) + generator.gauss(0.0, 3.0)
        )
    )

    measurements = comparison.measurements(
        CurveMetrics(
            conditional=CONDITIONAL_METRIC,
            pooled=POOLED_METRIC,
            model_variation=VARIATION_METRIC,
        ),
        data=component,
    )

    assert [entry.metric for entry in measurements] == [
        CONDITIONAL_METRIC,
        POOLED_METRIC,
        VARIATION_METRIC,
    ]
    assert measurements[0].value == pytest.approx(comparison.conditional_distance)
    assert measurements[1].value == pytest.approx(comparison.pooled_distance)
    assert measurements[2].value == pytest.approx(comparison.model_variation)
    for entry in measurements:
        assert entry.dispersion is not None
        assert entry.dispersion.kind == "evaluation"
        assert entry.sample_size == comparison.human_games + comparison.model_games


def test_the_model_variation_metric_is_optional(
    move_prediction_component: Digest,
) -> None:
    _register_curve_metrics()
    comparison = _compare(_generated(lambda rating, _: _length(rating)), resamples=0)

    measurements = comparison.measurements(
        CurveMetrics(conditional=CONDITIONAL_METRIC, pooled=POOLED_METRIC),
        data=move_prediction_component(),
    )

    assert [entry.metric for entry in measurements] == [
        CONDITIONAL_METRIC,
        POOLED_METRIC,
    ]
    assert all(entry.dispersion is None for entry in measurements)


def test_curve_points_are_stored_as_data_for_the_detail_tier() -> None:
    comparison = _compare(
        _generated(
            lambda rating, generator: _length(rating) + generator.gauss(0.0, 3.0)
        )
    )

    record = comparison.as_detail_record()
    encoded = json.loads(json.dumps(record))

    assert encoded["version"] == CURVE_COMPARISON_VERSION
    assert encoded["spec"] == SCALAR_SPEC.as_record()
    assert len(encoded["points"]) == len(SCALAR_SPEC.grid)
    assert [point["rating"] for point in encoded["points"]] == list(SCALAR_SPEC.grid)
    first = encoded["points"][0]
    assert first["human"]["effective_sample_size"] > 0.0
    assert first["bandwidth"] > 0.0
    assert encoded["response"] == comparison.response.value
    assert encoded["references"]["flat"] == pytest.approx(
        comparison.references.flat if comparison.references else None
    )
    assert encoded["dispersions"]["conditional"]["kind"] == "evaluation"
    # How the floor was arrived at, because a bootstrap over plentiful games
    # and a deterministic reading's exact zero are not distinguishable from the
    # value alone.
    assert encoded["dispersions"]["method"] == CURVE_BOOTSTRAP_METHOD
    assert encoded["human_games"] == comparison.human_games
    assert encoded["pooled"]["human"]["games"] > 0


def test_curves_on_different_series_are_not_one_continuous_comparison() -> None:
    comparison = _compare(_generated(lambda rating, _: _length(rating)), resamples=0)
    first = comparison.traces(model_label="checkpoint-a", fingerprint="a" * 64)
    second = comparison.traces(model_label="checkpoint-b", fingerprint="b" * 64)

    overlays = curve_overlays([*first, *second])

    assert len(overlays) == 2
    assert {overlay.series for overlay in overlays} == {"a" * 64, "b" * 64}
    for overlay in overlays:
        # One human reference per series, however many checkpoints carried it.
        labels = [trace.label for trace in overlay.traces]
        assert labels.count(HUMAN_REFERENCE_LABEL) == 1
        assert len(labels) == 2


def test_checkpoints_on_one_series_overlay_together() -> None:
    comparison = _compare(_generated(lambda rating, _: _length(rating)), resamples=0)
    first = comparison.traces(model_label="checkpoint-a", fingerprint="a" * 64)
    second = comparison.traces(model_label="checkpoint-b", fingerprint="a" * 64)

    (overlay,) = curve_overlays([*first, *second])

    assert [trace.label for trace in overlay.traces] == [
        HUMAN_REFERENCE_LABEL,
        "checkpoint-a",
        "checkpoint-b",
    ]
    assert len(overlay.traces[1].points) == len(SCALAR_SPEC.grid)
    assert overlay.traces[1].quantity is CurveQuantity.SCALAR


def test_a_recorded_bridge_rejoins_two_curve_series() -> None:
    comparison = _compare(_generated(lambda rating, _: _length(rating)), resamples=0)
    first = comparison.traces(model_label="checkpoint-a", fingerprint="a" * 64)
    second = comparison.traces(model_label="checkpoint-b", fingerprint="b" * 64)
    bridges = BridgeIndex(
        [
            build_bridge(
                from_fingerprint="a" * 64,
                to_fingerprint="b" * 64,
                reason="storage format change only",
                author="maintainer",
            )
        ]
    )

    (overlay,) = curve_overlays([*first, *second], bridges=bridges)

    assert [trace.label for trace in overlay.traces] == [
        HUMAN_REFERENCE_LABEL,
        "checkpoint-a",
        "checkpoint-b",
    ]


def test_bandwidth_selection_prefers_the_span_that_predicts_best() -> None:
    generator = random.Random(5)
    noisy = [
        Observation(rating, 40.0 + generator.gauss(0.0, 5.0))
        for rating in (1000.0 + index for index in range(300))
    ]
    detailed = [
        Observation(float(index), 10.0 * math.sin(index / 2.0)) for index in range(300)
    ]

    smooth = select_neighbours(
        noisy,
        quantity=CurveQuantity.SCALAR,
        candidates=(2, 4, 8, 16, 32, 64),
    )
    sharp = select_neighbours(
        detailed,
        quantity=CurveQuantity.SCALAR,
        candidates=(2, 4, 8, 16, 32, 64),
    )

    # Pure noise is best predicted by averaging widely; structure that turns
    # over inside a few rating points is destroyed by it.
    assert smooth.neighbours >= 16
    assert sharp.neighbours == 2
    assert smooth.observations == 300
    assert dict(smooth.candidates)[smooth.neighbours] == smooth.error
    assert smooth.as_record()["neighbours"] == smooth.neighbours


def test_bandwidth_selection_works_on_a_categorical_quantity() -> None:
    observations = [
        Observation(float(index), "sicilian" if index >= 150 else "other")
        for index in range(300)
    ]

    selection = select_neighbours(
        observations,
        quantity=CurveQuantity.CATEGORICAL,
        candidates=(2, 8, 32, 128),
    )

    assert selection.neighbours in (2, 8, 32, 128)
    assert selection.error >= 0.0


def test_bandwidth_selection_rejects_a_reference_it_cannot_cross_validate() -> None:
    observations = [Observation(float(index), float(index)) for index in range(4)]

    with pytest.raises(CurveComparisonError, match="at least three"):
        select_neighbours(
            observations[:2],
            quantity=CurveQuantity.SCALAR,
            candidates=(2,),
        )
    with pytest.raises(CurveComparisonError, match="no candidate bandwidth"):
        select_neighbours(
            observations,
            quantity=CurveQuantity.SCALAR,
            candidates=(64,),
        )


def test_a_comparison_needs_enough_reference_to_fill_its_bandwidth() -> None:
    human = [Observation(1200.0 + index, float(index)) for index in range(5)]

    with pytest.raises(CurveComparisonError, match="human reference has 5"):
        compare_curves(
            spec=SCALAR_SPEC,
            human=human,
            model=[Observation(1200.0, 1.0)],
            resamples=0,
        )
    with pytest.raises(CurveComparisonError, match="no generated games"):
        compare_curves(
            spec=SCALAR_SPEC,
            human=_human_reference(),
            model=[],
            resamples=0,
        )


def test_a_curve_rejects_values_of_the_wrong_kind() -> None:
    with pytest.raises(CurveComparisonError, match="needs a number"):
        compare_curves(
            spec=CATEGORICAL_SPEC.__class__(
                name="game-length",
                version=1,
                quantity=CurveQuantity.SCALAR,
                neighbours=2,
                grid=(1200.0, 1600.0),
            ),
            human=[Observation(1200.0, "sicilian"), Observation(1600.0, "other")],
            model=[Observation(1200.0, 1.0)],
            resamples=0,
        )
    with pytest.raises(CurveComparisonError, match="category label"):
        compare_curves(
            spec=CurveSpec(
                name="opening-repertoire",
                version=1,
                quantity=CurveQuantity.CATEGORICAL,
                neighbours=2,
                grid=(1200.0, 1600.0),
            ),
            human=[Observation(1200.0, "sicilian"), Observation(1600.0, 4.0)],
            model=[Observation(1200.0, "sicilian")],
            resamples=0,
        )


def test_the_category_drilldown_shows_mass_beside_the_delta() -> None:
    """Family granularity is uneven, so a delta alone is not a finding."""

    spec = CurveSpec(
        name="opening-repertoire",
        version=1,
        quantity=CurveQuantity.CATEGORICAL,
        neighbours=3,
        grid=(1200.0, 1600.0),
    )
    # One narrow category against a broad one, which is exactly the pair a
    # reader needs the mass column to tell apart.
    human = [
        Observation(1200.0 + 100.0 * step, "novelty" if step == 7 else "sicilian")
        for step in range(8)
    ]
    model = [Observation(1200.0, "novelty"), Observation(1600.0, "novelty")]

    shares = compare_curves(
        spec=spec, human=human, model=model, resamples=0
    ).category_shares()

    # Ordered by the reference's own mass, so the broad family leads.
    assert [share.category for share in shares] == ["sicilian", "novelty"]
    assert shares[0].mass == shares[0].human
    assert shares[0].delta == pytest.approx(shares[0].model - shares[0].human)
    assert sum(abs(share.delta) for share in shares) == pytest.approx(
        2.0 * comparison_pooled(spec, human, model)
    )


def comparison_pooled(
    spec: CurveSpec,
    human: Sequence[Observation],
    model: Sequence[Observation],
) -> float:
    """Return the pooled distance, for checking the drill-down sums to it."""

    return compare_curves(
        spec=spec, human=human, model=model, resamples=0
    ).pooled_distance


def test_a_scalar_comparison_has_no_category_drilldown() -> None:
    comparison = _compare(_generated(lambda rating, _: _length(rating)))

    assert comparison.category_shares() == ()


def test_a_floor_only_comparison_skips_the_null_levels() -> None:
    """A sweep recomputing a distance per ply pays for spreads, not for nulls."""

    comparison = _compare(
        _generated(
            lambda rating, generator: _length(rating) + generator.gauss(0.0, 3.0)
        ),
        references=False,
    )

    assert comparison.dispersions is not None
    assert comparison.references is None
    assert comparison.response is RatingResponse.UNKNOWN


def test_a_floor_only_comparison_reports_the_same_distances() -> None:
    """Skipping the nulls is a cost decision, not a different measurement.

    That covers the spreads as well as the distances, which is why the model
    side draws its resampling weights before the human side draws the ones only
    the nulls consume. Draw them the other way round and a sweep that skips the
    nulls to save time reports a different spread for its trouble.
    """

    generated = _generated(
        lambda rating, generator: _length(rating) + generator.gauss(0.0, 3.0)
    )

    full = _compare(generated)
    spreads_only = _compare(generated, references=False)

    assert spreads_only.conditional_distance == pytest.approx(full.conditional_distance)
    assert spreads_only.pooled_distance == pytest.approx(full.pooled_distance)
    assert full.dispersions is not None
    assert spreads_only.dispersions is not None
    assert spreads_only.dispersions == full.dispersions


def test_one_side_can_be_estimated_without_a_counterpart() -> None:
    """A model side computed exactly has no games to put through a comparison."""

    spec = CurveSpec(
        name="opening-repertoire",
        version=1,
        quantity=CurveQuantity.CATEGORICAL,
        neighbours=3,
        grid=(1200.0, 2000.0),
    )
    human = [
        Observation(
            1000.0 + 100.0 * step,
            "sicilian" if 1000.0 + 100.0 * step >= 1600.0 else "other",
        )
        for step in range(11)
    ]

    curve = estimate_curve(spec, human)

    assert [point.rating for point in curve.points] == [1200.0, 2000.0]
    assert curve.distribution_at(1200.0) == {
        "other": pytest.approx(1.0),
        "sicilian": 0.0,
    }
    assert curve.distribution_at(2000.0) == {
        "other": 0.0,
        "sicilian": pytest.approx(1.0),
    }
    assert curve.distribution_at(1400.0) is None


def test_an_estimated_curve_keeps_room_for_a_category_it_never_saw() -> None:
    """A category only the counterpart produces must read zero, not vanish."""

    spec = CurveSpec(
        name="opening-repertoire",
        version=1,
        quantity=CurveQuantity.CATEGORICAL,
        neighbours=2,
        grid=(1200.0, 1600.0),
    )
    human = [Observation(1200.0 + 100.0 * step, "other") for step in range(5)]

    curve = estimate_curve(spec, human, categories=("kings-gambit",))

    assert curve.categories == ("kings-gambit", "other")
    assert curve.distribution_at(1200.0) == {
        "kings-gambit": 0.0,
        "other": pytest.approx(1.0),
    }


def test_an_estimated_curve_needs_its_declared_bandwidth() -> None:
    spec = CurveSpec(
        name="opening-repertoire",
        version=1,
        quantity=CurveQuantity.CATEGORICAL,
        neighbours=10,
        grid=(1200.0, 1600.0),
    )

    with pytest.raises(CurveComparisonError, match="smooths over"):
        estimate_curve(spec, [Observation(1200.0, "other")])


def test_the_distribution_distance_is_the_comparison_shape_s_own() -> None:
    """An exact reading and a sampled one have to land on one scale."""

    assert distribution_distance({"a": 1.0}, {"a": 1.0}) == pytest.approx(0.0)
    assert distribution_distance({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    assert distribution_distance({"a": 0.5, "b": 0.5}, {"a": 1.0}) == pytest.approx(0.5)
