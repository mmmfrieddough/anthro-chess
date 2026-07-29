"""The human-reference curve comparison shape several benchmarks share."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence

import pytest

from anthro_chess.evaluation.curves import (
    CURVE_COMPARISON_VERSION,
    HUMAN_REFERENCE_LABEL,
    CurveComparison,
    CurveComparisonError,
    CurveMetrics,
    CurveQuantity,
    CurveSpec,
    Observation,
    RatingResponse,
    compare_curves,
    curve_overlays,
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
) -> CurveComparison:
    return compare_curves(
        spec=SCALAR_SPEC,
        human=human if human is not None else _human_reference(),
        model=model,
        resamples=resamples,
        seed=17,
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

    assert comparison.floors is not None
    assert comparison.floors.resamples == 80
    for floor in (
        comparison.floors.conditional,
        comparison.floors.pooled,
        comparison.floors.model_variation,
    ):
        assert floor.kind == "data-sampling"
        assert floor.value > 0.0
        assert floor.source is not None
        assert SCALAR_SPEC.name in floor.source


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

    assert small.floors is not None
    assert large.floors is not None
    assert large.floors.pooled.value < small.floors.pooled.value


def test_without_replicates_no_floor_is_invented() -> None:
    comparison = _compare(
        _generated(lambda rating, _: _length(rating)),
        resamples=0,
    )

    assert comparison.floors is None
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
        assert entry.noise_floor is not None
        assert entry.noise_floor.kind == "data-sampling"
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
    assert all(entry.noise_floor is None for entry in measurements)


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
    assert encoded["floors"]["conditional"]["kind"] == "data-sampling"
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
