"""Noise floor estimation, storage, lookup, and the sizing question."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from anthro_chess.evaluation.noise import (
    GameTotals,
    MetricTotal,
    NoiseConfig,
    bootstrap_floors,
    characterize_sampling_noise,
)
from anthro_chess.evaluation.results import (
    BridgeIndex,
    DataComponent,
    FloorEntry,
    NoiseCharacterizationError,
    NoiseFloorIndex,
    ResultsStore,
    ResultsStoreError,
    build_bridge,
    build_characterization,
    floor_from_dispersion,
    games_to_resolve,
    replicate_dispersion,
    replicate_floors,
    series_fingerprint,
)

Digest = Callable[..., DataComponent]

RECORDED_AT = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
METRIC = "held_out.move_loss"
OTHER_METRIC = "legality.mask_penalty"


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
    sampling_units: int | None = None,
) -> FloorEntry:
    return FloorEntry(
        metric=metric,
        fingerprint=series_fingerprint(metric, component),
        floor=floor,
        dispersion=dispersion,
        sampling_units=sampling_units,
    )


def test_a_floor_is_the_delta_two_independent_measurements_produce() -> None:
    # A floor has to be comparable to a reported delta, and the difference of
    # two independent measurements is wider than either one by sqrt(2).
    assert floor_from_dispersion(0.1, coverage=1.0) == pytest.approx(math.sqrt(2) * 0.1)
    assert floor_from_dispersion(0.0) == 0.0
    assert floor_from_dispersion(0.1, coverage=2.0) == pytest.approx(
        2 * floor_from_dispersion(0.1, coverage=1.0)
    )


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
    identical = bootstrap_floors(
        _totals([2.0] * 20),
        component=component,
        seed=7,
        resamples=200,
    )
    spread = bootstrap_floors(
        _totals([1.0, 2.0, 3.0, 4.0] * 5),
        component=component,
        seed=7,
        resamples=200,
    )

    assert identical[0].floor == 0.0
    assert spread[0].floor > 0.0
    assert spread[0].sampling_units == 20


def test_bootstrap_output_is_deterministic_for_one_seed(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    totals = _totals([1.0, 2.0, 3.5, 0.5, 2.5, 4.0])

    first = bootstrap_floors(totals, component=component, seed=11, resamples=150)
    again = bootstrap_floors(totals, component=component, seed=11, resamples=150)
    different = bootstrap_floors(totals, component=component, seed=12, resamples=150)

    assert first == again
    assert first[0].floor != different[0].floor


def test_a_bootstrap_floor_lands_on_the_series_it_qualifies(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()

    entries = bootstrap_floors(
        _totals([1.0, 2.0, 3.0]),
        component=component,
        seed=3,
        resamples=100,
    )

    assert entries[0].fingerprint == series_fingerprint(METRIC, component)


def test_bootstrapping_needs_more_than_one_game(
    move_prediction_component: Digest,
) -> None:
    with pytest.raises(NoiseCharacterizationError, match="at least two scored games"):
        bootstrap_floors(
            _totals([1.0]),
            component=move_prediction_component(),
            seed=1,
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

    entries = {
        entry.metric: entry
        for entry in bootstrap_floors(
            totals,
            component=component,
            seed=5,
            resamples=200,
        )
    }

    assert set(entries) == {METRIC, OTHER_METRIC}
    assert entries[OTHER_METRIC].floor > 0.0


def test_sampling_noise_sizes_the_games_an_axis_needs(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    entry = _entry(component, floor=0.04, sampling_units=1_000)

    # A floor shrinks with the square root of the games behind it, so resolving
    # a quarter of the measured floor takes sixteen times the games.
    assert games_to_resolve(entry, 0.04) == 1_000
    assert games_to_resolve(entry, 0.01) == 16_000


def test_a_floor_that_does_not_scale_cannot_size_an_input(
    move_prediction_component: Digest,
) -> None:
    entry = _entry(move_prediction_component(), sampling_units=None)

    with pytest.raises(NoiseCharacterizationError, match="does not scale"):
        games_to_resolve(entry, 0.01)


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
                floors=[_entry(component, floor=0.2)],
                recorded_at=RECORDED_AT,
            )
        ]
    )

    found = index.floors(METRIC, series_fingerprint(METRIC, component))

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
                floors=[_entry(component, floor=0.5)],
                recorded_at=RECORDED_AT,
            ),
            build_characterization(
                kind="training",
                method="independent-replicates",
                replicates=6,
                source="six seeds",
                floors=[_entry(component, floor=0.2)],
                recorded_at=datetime(2026, 7, 8, 12, 0, tzinfo=UTC),
            ),
        ]
    )

    found = index.floors(METRIC, series_fingerprint(METRIC, component))

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
                floors=[_entry(component, floor=0.3)],
                recorded_at=RECORDED_AT,
            ),
        ]
    )

    found = index.floors(METRIC, series_fingerprint(METRIC, component))

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
    assert entries[0].sampling_units is None


def test_characterizing_sampling_noise_produces_a_recordable_record(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()

    characterization = characterize_sampling_noise(
        _totals([1.0, 2.0, 3.0, 4.0]),
        component=component,
        config=NoiseConfig(resamples=200, seed=4),
        source="bootstrap over 4 game(s)",
        recorded_at=RECORDED_AT,
    )

    assert characterization is not None
    assert characterization.kind == "data-sampling"
    assert characterization.method == "bootstrap-over-games"
    assert characterization.replicates == 200
    entry = characterization.entry(METRIC)
    assert entry is not None
    assert entry.sampling_units == 4
