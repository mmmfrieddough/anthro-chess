import pytest

from anthro_chess.data.artifacts import DataLoadingError
from anthro_chess.data.composition import RatingComposition, fit_rating_composition

#: Far enough apart that the smoothed density of one band does not reach the
#: other, so each band's weight is its own count's.
_COMMON = 1500
_RARE = 2500


def _fit(
    balance: float,
    *,
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
        balance,
        population_games=len(ratings),
    )


def _share(composition: RatingComposition, bucket: str) -> float:
    shares = composition.report["decision_share_by_rating"]
    return float(shares[bucket]["composed"])


def test_a_balance_of_one_takes_the_population_as_it_comes() -> None:
    composition = _fit(1.0)

    assert composition.game_acceptance(_COMMON, _COMMON, 40) == 1.0
    assert composition.game_acceptance(_RARE, _RARE, 40) == 1.0
    assert composition.report["retained_decision_share"] == pytest.approx(1.0)
    assert composition.report["effective_sample_share"] == pytest.approx(1.0)


def test_a_balance_past_the_rarity_draws_every_rating_equally_often() -> None:
    composition = _fit(100.0)

    assert composition.game_acceptance(_RARE, _RARE, 40) == pytest.approx(1.0)
    assert composition.game_acceptance(_COMMON, _COMMON, 40) == pytest.approx(
        1 / 9, rel=1e-3
    )
    assert _share(composition, "1400_to_1599") == pytest.approx(0.5, rel=1e-3)
    assert _share(composition, "2400_to_2599") == pytest.approx(0.5, rel=1e-3)


def test_a_balance_short_of_the_rarity_boosts_the_rare_rating_by_itself() -> None:
    composition = _fit(3.0)

    assert composition.game_acceptance(_COMMON, _COMMON, 40) == pytest.approx(1 / 3)
    assert _share(composition, "2400_to_2599") == pytest.approx(1 / 4, rel=1e-3)
    assert composition.report["retained_decision_share"] >= 1 / 3
    assert composition.report["clipped_decision_share"] == pytest.approx(
        _share(composition, "2400_to_2599")
    )


def test_the_ratings_drawn_equally_often_are_reported_as_a_range() -> None:
    ratings = list(range(1000, 2001))
    weights = [1 + min(rating - 1000, 2000 - rating) for rating in ratings]
    sample = [
        rating
        for rating, weight in zip(ratings, weights, strict=True)
        for _ in range(weight)
    ]
    composition = fit_rating_composition(
        sample,
        sample,
        [2] * len(sample),
        ["blitz"] * len(sample),
        2.0,
        population_games=len(sample),
    )

    low, high = composition.report["equalized_ratings"]
    assert 1000 < low < 1500 < high < 2000


def test_no_rating_is_kept_more_often_than_the_heaviest_fitted_one() -> None:
    composition = _fit(3.0)
    rare = composition.game_acceptance(_RARE, _RARE, 40)

    assert composition.game_acceptance(3900, 3900, 40) == rare
    assert composition.game_acceptance(2000, 2000, 40) == rare


def test_a_game_is_weighted_by_whose_decisions_it_holds() -> None:
    """White moves first, so an odd-length game holds one more of its decisions."""

    composition = _fit(100.0)
    common = composition.game_acceptance(_COMMON, _COMMON, 3)
    rare = composition.game_acceptance(_RARE, _RARE, 3)

    assert composition.game_acceptance(_RARE, _COMMON, 3) == pytest.approx(
        (2 * rare + common) / 3
    )


def test_effective_sample_share_falls_as_the_balance_rises() -> None:
    shares = [
        _fit(balance).report["effective_sample_share"] for balance in (1.0, 3.0, 9.0)
    ]

    assert shares == sorted(shares, reverse=True)
    assert shares[-1] < 1.0


def test_the_speed_mix_is_reported_as_it_moves() -> None:
    shares = _fit(100.0).report["decision_share_by_speed"]

    assert shares["classical"]["as_drawn"] == pytest.approx(0.1)
    assert shares["classical"]["composed"] == pytest.approx(0.5, rel=1e-3)


def test_thinning_keeps_the_share_its_acceptance_names() -> None:
    composition = _fit(100.0)
    kept = sum(
        composition.keeps("seed", game_id, _COMMON, _COMMON, 40)
        for game_id in range(20_000)
    )

    assert kept / 20_000 == pytest.approx(1 / 9, abs=0.01)


def test_a_fit_from_no_games_is_refused() -> None:
    with pytest.raises(DataLoadingError, match="needs games"):
        fit_rating_composition([], [], [], [], 4.0, population_games=0)
