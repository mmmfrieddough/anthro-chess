"""Tests for deterministic per-benchmark views over an evaluation pool."""

import pytest

from anthro_chess.evaluation import PoolGame, ViewConfig, apply_view


def _game(
    game_id: int,
    *,
    plies: int = 40,
    ratings: bool = True,
) -> PoolGame:
    return PoolGame(
        game_id=game_id,
        ply_count=plies,
        result="1-0",
        has_ratings=ratings,
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
