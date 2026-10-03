import pytest

from anthro_chess.data.artifacts import DataLoadingError
from anthro_chess.data.composition import RatingComposition, fit_rating_composition
from anthro_chess.data.config import RatingCompositionConfig

#: Far enough apart that the smoothed density of one band does not reach the
#: other, so each band's weight is its own count's.
_COMMON = 1500
_RARE = 2500


def _fit(
    strength: float,
    *,
    maximum_weight: float = 100.0,
    common: int = 900,
    rare: int = 100,
    length: int = 40,
) -> RatingComposition:
    ratings = [_COMMON] * common + [_RARE] * rare
    return fit_rating_composition(
        ratings,
        ratings,
        [length] * len(ratings),
        ["blitz"] * common + ["classical"] * rare,
        RatingCompositionConfig(strength=strength, maximum_weight=maximum_weight),
        population_games=len(ratings),
    )


def _share(composition: RatingComposition, bucket: str) -> float:
    shares = composition.report["decision_share_by_rating"]
    return float(shares[bucket]["composed"])


def test_strength_zero_takes_the_population_as_it_comes() -> None:
    composition = _fit(0.0)

    assert composition.game_acceptance(_COMMON, _COMMON, 40) == 1.0
    assert composition.game_acceptance(_RARE, _RARE, 40) == 1.0
    assert composition.report["retained_decision_share"] == pytest.approx(1.0)
    assert composition.report["effective_sample_share"] == pytest.approx(1.0)


def test_full_strength_draws_every_rating_equally_often() -> None:
    composition = _fit(1.0)

    assert composition.game_acceptance(_RARE, _RARE, 40) == pytest.approx(1.0)
    assert composition.game_acceptance(_COMMON, _COMMON, 40) == pytest.approx(
        1 / 9, rel=1e-3
    )
    assert _share(composition, "1400_to_1599") == pytest.approx(0.5, rel=1e-3)
    assert _share(composition, "2400_to_2599") == pytest.approx(0.5, rel=1e-3)


def test_strength_between_corrects_partially() -> None:
    composition = _fit(0.5)

    assert composition.game_acceptance(_COMMON, _COMMON, 40) == pytest.approx(
        1 / 3, rel=1e-3
    )
    assert 0.1 < _share(composition, "2400_to_2599") < 0.5


def test_the_clip_bounds_the_weight_and_so_the_retained_share() -> None:
    composition = _fit(1.0, maximum_weight=3.0)

    assert composition.game_acceptance(_COMMON, _COMMON, 40) == pytest.approx(1 / 3)
    assert composition.report["retained_decision_share"] >= 1 / 3
    assert composition.report["clipped_decision_share"] == pytest.approx(
        _share(composition, "2400_to_2599")
    )


@pytest.mark.parametrize("maximum_weight", [3.0, 100.0])
def test_no_rating_is_kept_more_often_than_the_heaviest_fitted_one(
    maximum_weight: float,
) -> None:
    """Where the clip never binds, the heaviest rating is below it, not at it."""

    composition = _fit(0.5, maximum_weight=maximum_weight)
    rare = composition.game_acceptance(_RARE, _RARE, 40)

    assert composition.game_acceptance(3900, 3900, 40) == rare
    assert composition.game_acceptance(2000, 2000, 40) == rare
    assert composition.game_acceptance(3900, _COMMON, 40) == pytest.approx(
        composition.game_acceptance(_RARE, _COMMON, 40)
    )


def test_the_speed_mix_is_reported_as_it_moves() -> None:
    shares = _fit(1.0).report["decision_share_by_speed"]

    assert shares["classical"]["as_drawn"] == pytest.approx(0.1)
    assert shares["classical"]["composed"] == pytest.approx(0.5, rel=1e-3)


def test_a_game_is_weighted_by_whose_decisions_it_holds() -> None:
    """White moves first, so an odd-length game holds one more of its decisions."""

    composition = _fit(1.0)
    common = composition.game_acceptance(_COMMON, _COMMON, 3)
    rare = composition.game_acceptance(_RARE, _RARE, 3)

    assert composition.game_acceptance(_RARE, _COMMON, 3) == pytest.approx(
        (2 * rare + common) / 3
    )


def test_effective_sample_share_falls_as_the_correction_strengthens() -> None:
    shares = [
        _fit(strength).report["effective_sample_share"] for strength in (0.0, 0.5, 1.0)
    ]

    assert shares == sorted(shares, reverse=True)
    assert shares[-1] < 1.0


def test_thinning_keeps_the_share_its_acceptance_names() -> None:
    composition = _fit(1.0)
    kept = sum(
        composition.keeps("seed", game_id, _COMMON, _COMMON, 40)
        for game_id in range(20_000)
    )

    assert kept / 20_000 == pytest.approx(1 / 9, abs=0.01)


def test_a_fit_from_no_games_is_refused() -> None:
    with pytest.raises(DataLoadingError, match="needs games"):
        fit_rating_composition(
            [],
            [],
            [],
            [],
            RatingCompositionConfig(strength=1.0, maximum_weight=4.0),
            population_games=0,
        )
