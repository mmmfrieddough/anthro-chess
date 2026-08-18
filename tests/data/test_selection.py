"""Load-time selection within one prepared corpus.

Selection is what makes the value of data measurable: two runs that differ only
in what they trained on can be scored against one evaluation reference, which
preparing two narrower corpora can never do. These tests cover the axes, the
subsample dial, and the recorded identity that lets a later run confirm it
selected the same games.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from anthro_chess.data import (
    DataLoadingError,
    SelectionConfig,
    SequenceDataLoader,
    SequenceDataset,
    SequenceLoaderConfig,
    Speed,
    StreamingLoaderConfig,
    StreamingSequenceDataLoader,
    resolve_sharded_selection,
)
from anthro_chess.data.accounts import account_row_digest
from anthro_chess.data.artifacts import (
    game_ids_sha256,
    normalized_shard_paths,
    validate_manifest_outputs,
)

BLITZ_MS = 300_000
BULLET_MS = 60_000


def test_filters_within_one_corpus_by_time_control(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    games = _corpus(
        tmp_path,
        normalized_row,
        write_corpus,
        [
            (1, BULLET_MS, 1500),
            (2, BLITZ_MS, 1500),
            (3, BLITZ_MS, 1500),
        ],
    )

    dataset = SequenceDataset.from_parquet(
        games,
        split="train",
        selection=SelectionConfig(minimum_time_initial_ms=BLITZ_MS),
    )

    assert _loaded(dataset) == tuple(sorted((fixture_game_id(2), fixture_game_id(3))))
    assert dataset.resolution.excluded_games == {"below_minimum_time_initial": 1}


def test_a_speed_class_selects_a_diagonal_the_clock_bounds_cannot_draw(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    # 5+0 and 1+3 are blitz while 1+0 is bullet, so no bounds over the two
    # clock columns keep the first two rows and drop the third.
    rows = [
        normalized_row(1, split="train", time_initial_ms=BLITZ_MS),
        normalized_row(
            2, split="train", time_initial_ms=BULLET_MS, time_increment_ms=3_000
        ),
        normalized_row(3, split="train", time_initial_ms=BULLET_MS),
        normalized_row(4, split="train", time_initial_ms=None, time_increment_ms=None),
    ]
    normalized_directory, _ = write_corpus(tmp_path, rows)

    dataset = SequenceDataset.from_parquet(
        normalized_directory / "games.parquet",
        split="train",
        selection=SelectionConfig(speed=Speed.BLITZ),
    )

    assert _loaded(dataset) == tuple(sorted((fixture_game_id(1), fixture_game_id(2))))
    assert dataset.resolution.excluded_games == {
        "speed_mismatch": 1,
        "missing_time_control": 1,
    }


def test_rating_band_requires_both_players_inside_it(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    rows = [
        normalized_row(1, split="train", ratings=(1500, 1520)),
        normalized_row(2, split="train", ratings=(1500, 2400)),
        normalized_row(3, split="train", ratings=(1500, None)),
    ]
    normalized_directory, _ = write_corpus(tmp_path, rows)

    dataset = SequenceDataset.from_parquet(
        normalized_directory / "games.parquet",
        split="train",
        selection=SelectionConfig(minimum_rating=1400, maximum_rating=1600),
    )

    assert _loaded(dataset) == (fixture_game_id(1),)
    assert dataset.resolution.excluded_games == {
        "above_maximum_rating": 1,
        "missing_ratings": 1,
    }


def test_a_missing_axis_value_is_excluded_rather_than_treated_as_zero(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    rows = [
        normalized_row(1, split="train", time_initial_ms=None),
        normalized_row(2, split="train", time_initial_ms=BLITZ_MS),
    ]
    normalized_directory, _ = write_corpus(tmp_path, rows)

    dataset = SequenceDataset.from_parquet(
        normalized_directory / "games.parquet",
        split="train",
        selection=SelectionConfig(minimum_time_initial_ms=0),
    )

    assert _loaded(dataset) == (fixture_game_id(2),)
    assert dataset.resolution.excluded_games == {"missing_time_control": 1}


def test_subsampling_is_deterministic_and_a_smaller_dial_is_a_subset(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    games = _corpus(
        tmp_path,
        normalized_row,
        write_corpus,
        [(game_id, BLITZ_MS, 1500) for game_id in range(1, 201)],
    )

    half = _selected(games, SelectionConfig(fraction=0.5))
    half_again = _selected(games, SelectionConfig(fraction=0.5))
    fifth = _selected(games, SelectionConfig(fraction=0.2))
    whole = _selected(games, SelectionConfig(fraction=1.0))

    # Reproducible on any machine, and nested, because a share is a cut of the
    # rank space each game's own digest places it in rather than a shuffle of
    # whatever order the shards held.
    assert half == half_again
    assert set(fifth) < set(half) < set(whole)
    assert len(whole) == 200
    # The size follows the share it asked for rather than matching it, because
    # a cut needs no count of the candidates and so cannot land on one exactly.
    assert 80 <= len(half) <= 120
    assert 20 <= len(fifth) <= 60


def test_maximum_games_caps_a_selection_after_filtering(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    games = _corpus(
        tmp_path,
        normalized_row,
        write_corpus,
        [(game_id, BLITZ_MS, 1500) for game_id in range(1, 11)],
    )

    capped = _selected(games, SelectionConfig(maximum_games=3))
    halved_then_capped = _selected(
        games, SelectionConfig(fraction=0.5, maximum_games=3)
    )

    assert len(capped) == 3
    assert capped == halved_then_capped


def test_a_selection_matching_nothing_fails_instead_of_training_on_nothing(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    games = _corpus(tmp_path, normalized_row, write_corpus, [(1, BLITZ_MS, 1500)])

    with pytest.raises(DataLoadingError, match="no normalized games matched"):
        SequenceDataset.from_parquet(
            games,
            split="train",
            selection=SelectionConfig(minimum_rating=3000),
        )


def test_a_selection_never_reaches_outside_the_split_it_loads(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    rows = [
        normalized_row(1, split="train"),
        normalized_row(2, split="validation"),
        normalized_row(3, split="test"),
    ]
    normalized_directory, _ = write_corpus(tmp_path, rows)

    dataset = SequenceDataset.from_parquet(
        normalized_directory / "games.parquet",
        split="train",
        selection=SelectionConfig(fraction=1.0),
    )

    assert _loaded(dataset) == (fixture_game_id(1),)
    assert dataset.resolution.eligible_games == 1


def test_the_resolved_selection_reproduces_the_same_games_by_identity(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    games = _corpus(
        tmp_path,
        normalized_row,
        write_corpus,
        [(game_id, BLITZ_MS, 1500) for game_id in range(1, 201)],
    )
    selection = SelectionConfig(fraction=0.5)

    first = SequenceDataset.from_parquet(games, split="train", selection=selection)
    second = SequenceDataset.from_parquet(games, split="train", selection=selection)
    narrower = SequenceDataset.from_parquet(
        games,
        split="train",
        selection=SelectionConfig(fraction=0.25),
    )

    record = first.resolution.as_record()
    assert record["spec"] == selection.model_dump(mode="json")
    assert record["eligible_games"] == 200
    assert record["excluded_games"] == {}
    assert record == second.resolution.as_record()
    # Two selections over one corpus stay distinguishable by what they realized,
    # not only by the configuration that asked for it.
    assert record["selected_games"] != narrower.resolution.as_record()["selected_games"]
    assert first.identity_sha256 == second.identity_sha256
    assert first.identity_sha256 != narrower.identity_sha256


def test_a_loader_rejects_a_dataset_built_for_a_different_selection(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    games = _corpus(
        tmp_path,
        normalized_row,
        write_corpus,
        [(game_id, BLITZ_MS, 1500) for game_id in range(1, 5)],
    )
    dataset = SequenceDataset.from_parquet(
        games,
        split="train",
        selection=SelectionConfig(fraction=0.5),
    )

    with pytest.raises(DataLoadingError, match="loader selection does not match"):
        SequenceDataLoader(
            dataset,
            SequenceLoaderConfig(
                split="train",
                batch_size=1,
                selection=SelectionConfig(fraction=0.25),
            ),
        )


def test_a_changed_selection_changes_the_loader_configuration_identity(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    games = _corpus(
        tmp_path,
        normalized_row,
        write_corpus,
        [(game_id, BLITZ_MS, 1500) for game_id in range(1, 5)],
    )
    broad = SequenceLoaderConfig(split="train", batch_size=1, shuffle=False)
    narrow = broad.model_copy(update={"selection": SelectionConfig(fraction=0.5)})

    assert (
        SequenceDataLoader.from_parquet(games, broad).configuration_sha256
        != SequenceDataLoader.from_parquet(games, narrow).configuration_sha256
    )


@pytest.mark.parametrize(
    ("selection", "digest"),
    [
        (
            SelectionConfig(),
            "b96d1bb0c4fc37a8c19fdc17e5fbc701a8737f17121167dc43af92a556b13d35",
        ),
        (
            SelectionConfig(fraction=0.5),
            "755deaddf342ec7abab42be561159513cacff304083c44f7c3696a5f38ddff76",
        ),
        (
            SelectionConfig(fraction=0.25),
            "4221a20eacc3d0ad6289b158d94131173be6f5b21270483d0addbc2a8bf21f7d",
        ),
        (
            SelectionConfig(maximum_games=5),
            "7eefe287e079a9e7c6305fc92fe2a9823260271fc3d220296f07c29e904ab12b",
        ),
        (
            SelectionConfig(fraction=0.5, seed="frozen-digest-seed"),
            "d3e2655ad361d4f8d628bb925beda7f53e42998b6a1dbdadddbad4cff90b87da",
        ),
    ],
    ids=["everything", "half", "quarter", "capped", "other-seed"],
)
def test_a_corpus_and_spec_still_select_the_games_they_always_have(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    selection: SelectionConfig,
    digest: str,
) -> None:
    """A selection is reproducible only against something outside itself.

    Every other test here compares one selection to another resolved by the
    same code, so a rewritten rank, floor, or ordering agrees with itself and
    passes. A run recorded a year ago cannot be re-resolved to check, which is
    what these constants stand in for. A failure means this corpus and spec now
    select different games: decide which of the two moved before updating it.

    The constants are read off the games the loader encoded rather than off any
    figure it recorded about them, so a resolution that miscounted its own
    selection fails here too.
    """

    rows = [
        normalized_row(game_id, split="train", time_initial_ms=BLITZ_MS, rating=1500)
        for game_id in range(1, 17)
    ]
    normalized, manifest_path = write_corpus(tmp_path, rows)
    shards = validate_manifest_outputs(
        json.loads(manifest_path.read_bytes()),
        manifest_path,
        normalized_shard_paths(normalized),
    )

    eager = SequenceDataset.from_parquet(
        [shard.path for shard in shards], split="train", selection=selection
    )

    assert game_ids_sha256(_loaded(eager)) == digest


def test_selection_bounds_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="maximum_rating must not be below"):
        SelectionConfig(minimum_rating=1800, maximum_rating=1200)


def test_a_marked_account_of_either_colour_leaves_what_a_run_reads(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    """`0041` rejects on the account, so which colour it played is immaterial."""

    games = _corpus(
        tmp_path,
        normalized_row,
        write_corpus,
        [(1, BLITZ_MS, 1500), (2, BLITZ_MS, 1500), (3, BLITZ_MS, 1500)],
    )
    marked = frozenset({account_row_digest("white1"), account_row_digest("black2")})

    dataset = SequenceDataset.from_parquet(
        games, split="train", selection=_naming_a_snapshot(), marked_digests=marked
    )

    assert _loaded(dataset) == (fixture_game_id(3),)
    assert dataset.resolution.excluded_games == {"marked_account": 2}


def test_both_loaders_reject_the_same_accounts(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """A run picks one loader by configuration, so they must hold one corpus.

    The shard-backed loader reads its own projected columns per row group and
    the eager one reads whole rows, which is where a filter reaching only one
    of them would go unnoticed.
    """

    rows = [
        normalized_row(game_id, split="train", time_initial_ms=BLITZ_MS, rating=1500)
        for game_id in range(1, 17)
    ]
    normalized, manifest_path = write_corpus(tmp_path, rows)
    manifest_bytes = manifest_path.read_bytes()
    shards = validate_manifest_outputs(
        json.loads(manifest_bytes), manifest_path, normalized_shard_paths(normalized)
    )
    marked = frozenset(
        {account_row_digest(f"white{game_id}") for game_id in (2, 5, 11)}
    )

    eager = SequenceDataset.from_parquet(
        [shard.path for shard in shards],
        split="train",
        selection=_naming_a_snapshot(),
        marked_digests=marked,
    )
    streaming = StreamingSequenceDataLoader(
        resolve_sharded_selection(
            shards,
            split="train",
            selection=_naming_a_snapshot(),
            manifest_sha256=sha256(manifest_bytes).hexdigest(),
            marked_digests=marked,
        ),
        SequenceLoaderConfig(split="train", selection=_naming_a_snapshot()),
        StreamingLoaderConfig(),
    )
    try:
        read = {
            int(batch.game_ids[row][0])
            for batch in streaming
            for row in range(batch.batch_size)
        }
    finally:
        streaming.close()

    assert eager.resolution.selected_games == 13
    assert read == set(_loaded(eager))
    assert eager.resolution.excluded_games == {"marked_account": 3}
    # The shard-backed loader rejects the same accounts and never counts them:
    # a tally over a corpus-scale split costs a read of every row of it.
    assert streaming.resolution.excluded_games is None


def test_the_rejection_runs_before_the_subsample_that_sizes_an_arm(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """`maximum_games` holds two runs differing only in this dial to one count.

    Rejecting after the subsample would hand the filtered run fewer games than
    the unfiltered one, so a comparison between them would confound the filter
    with how much data each saw. `fraction` cannot do the same job.
    """

    games = _corpus(
        tmp_path,
        normalized_row,
        write_corpus,
        [(game_id, BLITZ_MS, 1500) for game_id in range(1, 21)],
    )
    marked = frozenset(
        {account_row_digest(f"white{game_id}") for game_id in range(1, 8)}
    )
    unfiltered = SequenceDataset.from_parquet(
        games, split="train", selection=SelectionConfig(maximum_games=10)
    )
    filtered = SequenceDataset.from_parquet(
        games,
        split="train",
        selection=_naming_a_snapshot(maximum_games=10),
        marked_digests=marked,
    )

    assert len(_loaded(unfiltered)) == len(_loaded(filtered)) == 10
    assert set(_loaded(filtered)) != set(_loaded(unfiltered))

    # A fraction takes its share of whatever survived, so the two arms differ in
    # how much data each saw as well as in which games.
    unfiltered_share = _loaded(
        SequenceDataset.from_parquet(
            games, split="train", selection=SelectionConfig(fraction=0.5)
        )
    )
    filtered_share = _loaded(
        SequenceDataset.from_parquet(
            games,
            split="train",
            selection=_naming_a_snapshot(fraction=0.5),
            marked_digests=marked,
        )
    )
    assert set(filtered_share) < set(unfiltered_share)


def _naming_a_snapshot(
    *,
    fraction: float | None = None,
    maximum_games: int | None = None,
) -> SelectionConfig:
    """Return a selection declaring a snapshot no caller here opens."""

    return SelectionConfig(
        marked_accounts=Path("marked-accounts.txt"),
        fraction=fraction,
        maximum_games=maximum_games,
    )


def _corpus(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    games: list[tuple[int, int, int]],
) -> Path:
    rows = [
        normalized_row(
            game_id,
            split="train",
            time_initial_ms=time_initial_ms,
            rating=rating,
        )
        for game_id, time_initial_ms, rating in games
    ]
    normalized_directory, _ = write_corpus(tmp_path, rows)
    return normalized_directory / "games.parquet"


def _selected(games: Path, selection: SelectionConfig) -> tuple[int, ...]:
    return _loaded(
        SequenceDataset.from_parquet(games, split="train", selection=selection)
    )


def _loaded(dataset: SequenceDataset) -> tuple[int, ...]:
    """Return the games a dataset actually encoded, in ascending order.

    Read from the examples rather than from the resolution, which records how
    many games it kept and not which. That leaves the count uncheckable from
    any one test, so it is checked here instead, on every selection any test in
    this file resolves.
    """

    game_ids = tuple(sorted({example.game_id for example in dataset}))
    assert dataset.resolution.selected_games == len(game_ids)
    return game_ids
