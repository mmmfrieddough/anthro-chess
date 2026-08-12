"""Tests for deterministic per-benchmark views over an evaluation pool."""

from datetime import date

import pytest

from anthro_chess.data import Speed
from anthro_chess.evaluation import PoolGame, ViewConfig, apply_view
from anthro_chess.evaluation.views import excluded_summary


def _game(
    game_id: int,
    *,
    plies: int = 40,
    ratings: bool = True,
    speed: Speed | None = Speed.BLITZ,
    source_date: date | None = date(2019, 6, 15),
) -> PoolGame:
    return PoolGame(
        game_id=game_id,
        ply_count=plies,
        result="1-0",
        has_ratings=ratings,
        speed=speed,
        source_date=source_date,
    )


def _pool(count: int = 200) -> tuple[PoolGame, ...]:
    return tuple(_game(game_id) for game_id in range(1, count + 1))


def test_a_view_without_filters_selects_the_whole_pool() -> None:
    selection = apply_view(_pool(20), ViewConfig(name="all"))

    assert selection.selected_games == 20
    assert selection.eligible_games == 20
    assert selection.excluded_games == {}
    assert selection.game_ids == tuple(range(1, 21))


def test_subsampling_is_deterministic_and_reproducible() -> None:
    config = ViewConfig(name="fast", maximum_games=25)

    first = apply_view(_pool(), config)
    second = apply_view(_pool(), config)

    assert first.game_ids == second.game_ids
    assert first.selected_games == 25
    assert first.eligible_games == 200


def test_a_different_seed_selects_a_different_subsample() -> None:
    pool = _pool()

    default = apply_view(pool, ViewConfig(name="fast", maximum_games=25))
    reseeded = apply_view(pool, ViewConfig(name="fast", maximum_games=25, seed="other"))

    assert default.game_ids != reseeded.game_ids


def test_subsampling_ranks_by_game_id_so_growth_does_not_reshuffle() -> None:
    """Rank ordering is a pure function of the game id, like the split itself."""

    config = ViewConfig(name="fast", maximum_games=10)

    small = apply_view(_pool(50), config)
    grown = apply_view(_pool(500), config)

    ranked_small = apply_view(_pool(50), ViewConfig(name="fast"))
    assert small.game_ids == tuple(
        sorted(
            game_id
            for game_id in ranked_small.game_ids
            if game_id in set(small.game_ids)
        )
    )
    assert grown.selected_games == 10


def test_selection_is_order_independent() -> None:
    config = ViewConfig(name="fast", maximum_games=15)

    forward = apply_view(_pool(60), config)
    reversed_pool = apply_view(tuple(reversed(_pool(60))), config)

    assert forward.game_ids == reversed_pool.game_ids


def test_filters_exclude_games_and_report_why() -> None:
    pool = (
        _game(1, plies=4),
        _game(2, plies=40),
        _game(3, plies=400),
        _game(4, ratings=False),
    )

    selection = apply_view(
        pool,
        ViewConfig(
            name="rated",
            minimum_plies=10,
            maximum_plies=100,
            require_ratings=True,
        ),
    )

    assert selection.game_ids == (2,)
    assert selection.excluded_games == {
        "below_minimum_plies": 1,
        "above_maximum_plies": 1,
        "missing_ratings": 1,
    }


def test_a_prefix_view_excludes_games_shorter_than_the_prefix() -> None:
    pool = (_game(1, plies=8), _game(2, plies=40))

    selection = apply_view(pool, ViewConfig(name="prefixes", prefix_plies=16))

    assert selection.game_ids == (2,)
    assert selection.prefix_plies == 16
    assert selection.excluded_games == {"shorter_than_prefix": 1}


def test_a_view_takes_one_speed_class_and_reports_the_rest() -> None:
    pool = (
        _game(1, speed=Speed.BULLET),
        _game(2, speed=Speed.BLITZ),
        _game(3, speed=Speed.RAPID),
        _game(4, speed=None),
    )

    selection = apply_view(pool, ViewConfig(name="blitz", speed=Speed.BLITZ))

    assert selection.game_ids == (2,)
    assert selection.excluded_games == {
        "speed_mismatch": 2,
        "missing_time_control": 1,
    }
    assert selection.as_record()["speed"] == "blitz"


def test_a_selection_summarizes_what_it_left_out() -> None:
    """An empty selection looks the same however it got there."""

    pool = (_game(1, plies=4), _game(2, speed=Speed.BULLET))

    filtered = apply_view(
        pool, ViewConfig(name="blitz", minimum_plies=10, speed=Speed.BLITZ)
    )
    whole = apply_view(pool, ViewConfig(name="all"))

    assert (
        excluded_summary(filtered.excluded_games)
        == "1 below_minimum_plies, 1 speed_mismatch"
    )
    assert excluded_summary(whole.excluded_games) == "nothing excluded"


def test_the_class_filters_before_the_cap_takes_its_games() -> None:
    """A cap is the declared size, so it must take that many of the class."""

    pool = tuple(
        _game(game_id, speed=Speed.BLITZ if game_id % 2 else Speed.BULLET)
        for game_id in range(1, 41)
    )

    selection = apply_view(
        pool,
        ViewConfig(name="blitz", maximum_games=10, speed=Speed.BLITZ),
    )

    assert selection.selected_games == 10
    assert all(game_id % 2 for game_id in selection.game_ids)


def test_the_spec_record_identifies_exactly_which_games_were_measured() -> None:
    selection = apply_view(_pool(30), ViewConfig(name="fast", maximum_games=5))

    record = selection.as_record()

    assert record["name"] == "fast-5"
    assert record["selected_games"] == 5
    assert record["eligible_games"] == 30
    assert len(str(record["game_ids_sha256"])) == 64
    assert (
        record["game_ids_sha256"]
        != apply_view(_pool(30), ViewConfig(name="fast", maximum_games=6)).as_record()[
            "game_ids_sha256"
        ]
    )


def test_a_view_that_took_everything_records_the_name_it_declared() -> None:
    selection = apply_view(_pool(30), ViewConfig(name="canonical"))

    assert selection.as_record()["name"] == "canonical"


def test_a_cap_that_bit_nothing_leaves_the_declared_name_alone() -> None:
    selection = apply_view(_pool(5), ViewConfig(name="canonical", maximum_games=50))

    assert selection.as_record()["name"] == "canonical"


def test_contradictory_ply_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be below minimum_plies"):
        apply_view(
            _pool(5),
            ViewConfig(name="bad", minimum_plies=50, maximum_plies=10),
        )


def test_a_view_that_matches_nothing_reports_an_empty_selection() -> None:
    selection = apply_view(_pool(5), ViewConfig(name="none", minimum_plies=1000))

    assert selection.selected_games == 0
    assert selection.eligible_games == 0
    assert selection.excluded_games == {"below_minimum_plies": 5}


def test_a_date_range_takes_one_era_of_a_pool_that_spans_several() -> None:
    """The bounds are inclusive, which is what names an era by its months."""

    pool = (
        _game(1, source_date=date(2019, 12, 31)),
        _game(2, source_date=date(2020, 3, 1)),
        _game(3, source_date=date(2020, 10, 31)),
        _game(4, source_date=date(2020, 11, 1)),
    )

    selection = apply_view(
        pool,
        ViewConfig(
            name="pandemic",
            minimum_date=date(2020, 3, 1),
            maximum_date=date(2020, 10, 31),
        ),
    )

    assert selection.game_ids == (2, 3)
    assert selection.excluded_games == {
        "before_minimum_date": 1,
        "after_maximum_date": 1,
    }


def test_a_game_with_no_date_is_excluded_by_a_date_bound() -> None:
    """An era reading must not quietly count games of unknown era."""

    pool = (_game(1, source_date=None), _game(2, source_date=date(2020, 3, 1)))

    assert apply_view(
        pool, ViewConfig(name="from", minimum_date=date(2020, 1, 1))
    ).excluded_games == {"missing_date": 1}
    assert apply_view(
        pool, ViewConfig(name="until", maximum_date=date(2020, 12, 31))
    ).excluded_games == {"missing_date": 1}


def test_an_undated_game_survives_a_view_that_asks_for_no_era() -> None:
    pool = (_game(1, source_date=None),)

    assert apply_view(pool, ViewConfig(name="all")).game_ids == (1,)


def test_contradictory_date_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be before minimum_date"):
        apply_view(
            _pool(5),
            ViewConfig(
                name="bad",
                minimum_date=date(2021, 1, 1),
                maximum_date=date(2020, 1, 1),
            ),
        )
