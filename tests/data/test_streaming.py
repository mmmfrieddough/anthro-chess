import json
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from anthro_chess.data import (
    DataLoadingError,
    SelectionConfig,
    SequenceBatch,
    SequenceDataLoader,
    SequenceLoaderConfig,
    StreamingLoaderConfig,
    StreamingSequenceDataLoader,
    resolve_sharded_selection,
)
from anthro_chess.data.accounts import account_row_digest
from anthro_chess.data.artifacts import (
    ShardIdentity,
    normalized_shard_paths,
    validate_manifest_outputs,
)
from anthro_chess.data.streaming import ShardedSelection

Corpus = tuple[tuple[ShardIdentity, ...], str]

#: Lengths that straddle a bucket boundary of four, so a fixture corpus has
#: more than one bucket to fill and flush.
_PLY_COUNTS = (3, 4, 5, 6, 7, 8, 9, 10)


def _corpus(
    write_corpus: Callable[..., tuple[Path, Path]],
    directory: Path,
    rows: list[dict[str, Any]],
    **layout: Any,
) -> Corpus:
    """Write a corpus and return it the way a verified manifest names it."""

    normalized, manifest_path = write_corpus(directory, rows, **layout)
    manifest_bytes = manifest_path.read_bytes()
    shards = validate_manifest_outputs(
        json.loads(manifest_bytes),
        manifest_path,
        normalized_shard_paths(normalized),
    )
    return shards, sha256(manifest_bytes).hexdigest()


def _loader(
    corpus: Corpus,
    config: SequenceLoaderConfig,
    streaming: StreamingLoaderConfig | None = None,
    *,
    legal_actions: bool = True,
) -> StreamingSequenceDataLoader:
    shards, manifest_sha256 = corpus
    selection = resolve_sharded_selection(
        shards,
        split=config.split,
        selection=config.selection,
        chunk_length=config.chunk_length,
        manifest_sha256=manifest_sha256,
    )
    return StreamingSequenceDataLoader(
        selection,
        config,
        StreamingLoaderConfig() if streaming is None else streaming,
        legal_actions=legal_actions,
    )


def _sequences(batch: SequenceBatch) -> list[tuple[int, tuple[int, ...]]]:
    """Return each row of a batch as its game and the plies it actually holds."""

    sequences: list[tuple[int, tuple[int, ...]]] = []
    for row in range(batch.batch_size):
        present = [
            index
            for index, occupied in enumerate(batch.attention_mask[row])
            if occupied
        ]
        sequences.append(
            (
                int(batch.game_ids[row][0]),
                tuple(int(batch.ply_indices[row][index]) for index in present),
            )
        )
    return sequences


def _decoded(batch: SequenceBatch) -> list[tuple[Any, ...]]:
    """Return each row of a batch as enough of its decode to identify the row.

    A game id alone is not: the plan chooses a row and the id is derived from
    whichever row was gathered, so a batch built from the wrong row would still
    be named consistently with itself.
    """

    inputs = batch.inputs
    decoded: list[tuple[Any, ...]] = []
    for row in range(batch.batch_size):
        held = sum(1 for occupied in batch.attention_mask[row] if occupied)
        decoded.append(
            (
                int(batch.game_ids[row][0]),
                batch.chunk_start_plies[row],
                batch.ply_indices[row][:held].tolist(),
                inputs.piece_ids[row][:held].tolist(),
                batch.action_targets[row][:held].tolist(),
                inputs.player_clock_ms.values[row][:held].tolist(),
                inputs.player_clock_ms.present[row][:held].tolist(),
                inputs.target_rating.values[row][:held].tolist(),
                inputs.target_rating.present[row][:held].tolist(),
            )
        )
    return decoded


def _drain(loader: StreamingSequenceDataLoader) -> list[tuple[int, tuple[int, ...]]]:
    """Return every sequence one epoch produced, in the order it produced them."""

    try:
        return [sequence for batch in loader for sequence in _sequences(batch)]
    finally:
        loader.close()


def _rows(
    normalized_row: Callable[..., dict[str, Any]],
    count: int = 24,
    *,
    split: str = "train",
    vary_ratings: bool = False,
) -> list[dict[str, Any]]:
    """Return a fixture corpus, optionally giving every game its own ratings.

    The ratings are constant by default because the tests that filter on them
    choose their own bounds against a known value.
    """

    return [
        normalized_row(
            game_id,
            split=split,
            plies=_PLY_COUNTS[game_id % len(_PLY_COUNTS)],
            ratings=(1200 + game_id, 2400 - game_id) if vary_ratings else None,
        )
        for game_id in range(1, count + 1)
    ]


def test_streams_exactly_the_games_the_eager_loader_decodes(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """The same games, and the same values decoded out of them.

    Ratings vary per game, so two games of one length are told apart by what
    they decode to rather than only by how long they are.
    """

    rows = _rows(normalized_row, vary_ratings=True)
    corpus = _corpus(write_corpus, tmp_path, rows, games_per_shard=8, row_group_size=4)
    config = SequenceLoaderConfig(split="train", batch_size=3, length_bucket_width=4)

    loader = _loader(corpus, config, legal_actions=False)
    try:
        streamed = [sequence for batch in loader for sequence in _decoded(batch)]
    finally:
        loader.close()
    eager = SequenceDataLoader.from_parquet(
        [shard.path for shard in corpus[0]],
        config,
        legal_actions=False,
    )

    assert sorted(streamed) == sorted(
        sequence for batch in eager for sequence in _decoded(batch)
    )
    assert len(streamed) == len(rows)


def test_pads_masks_and_indexes_every_sequence_it_emits(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row))
    loader = _loader(
        corpus,
        SequenceLoaderConfig(
            split="train",
            batch_size=4,
            length_bucket_width=None,
            shuffle=False,
        ),
    )

    batch = next(loader)
    loader.close()

    legal_action_ids = batch.legal_action_ids
    assert legal_action_ids is not None
    for row in range(batch.batch_size):
        mask = batch.attention_mask[row].tolist()
        held = sum(mask)
        assert mask == [True] * held + [False] * (len(mask) - held)
        assert batch.action_loss_mask[row].tolist() == mask
        assert batch.ply_indices[row][:held].tolist() == list(range(held))
        assert batch.action_targets[row][held:].tolist() == [0] * (len(mask) - held)
        assert legal_action_ids[row][held:] == ((),) * (len(mask) - held)
        assert all(legal_action_ids[row][:held])


def test_a_loader_asked_for_no_legal_actions_ships_none_through_its_workers(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """The flag has to travel in the job rather than be consulted in the parent.

    Both the decode this skips and the payload it shrinks happen in the worker,
    so this drains a real pool rather than the in-process path.
    """

    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row))
    config = SequenceLoaderConfig(
        split="train",
        batch_size=4,
        length_bucket_width=None,
        shuffle=False,
    )
    streaming = StreamingLoaderConfig(workers=2, prefetch_batches=2)

    training = _loader(corpus, config, streaming, legal_actions=False)
    scoring = _loader(corpus, config, streaming)
    try:
        training_batch = next(training)
        scoring_batch = next(scoring)
    finally:
        training.close()
        scoring.close()

    assert training_batch.legal_action_ids is None
    assert scoring_batch.legal_action_ids is not None
    # Everything a training step reads is the same batch either way.
    assert (
        training_batch.action_targets.tolist() == scoring_batch.action_targets.tolist()
    )
    assert (
        training_batch.attention_mask.tolist() == scoring_batch.attention_mask.tolist()
    )
    assert training_batch.ply_indices.tolist() == scoring_batch.ply_indices.tolist()


def test_length_buckets_keep_a_batch_to_one_bucket(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row, 48))
    loader = _loader(
        corpus,
        SequenceLoaderConfig(split="train", batch_size=3, length_bucket_width=4),
    )

    try:
        for batch in loader:
            lengths = {len(plies) for _, plies in _sequences(batch)}
            assert len({(length - 1) // 4 for length in lengths}) == 1
    finally:
        loader.close()


def test_chunks_stay_contiguous_and_cover_every_ply(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    rows = _rows(normalized_row, 12)
    corpus = _corpus(write_corpus, tmp_path, rows)

    streamed = _drain(
        _loader(
            corpus,
            SequenceLoaderConfig(split="train", batch_size=2, chunk_length=3),
        )
    )

    covered: dict[int, list[int]] = {}
    for game_id, plies in streamed:
        assert plies == tuple(range(plies[0], plies[0] + len(plies)))
        assert len(plies) <= 3
        covered.setdefault(game_id, []).extend(plies)
    eager = SequenceDataLoader.from_parquet(
        [shard.path for shard in corpus[0]],
        SequenceLoaderConfig(split="train", batch_size=2, chunk_length=3),
    )
    expected: dict[int, list[int]] = {}
    for batch in eager:
        for game_id, plies in _sequences(batch):
            expected.setdefault(game_id, []).extend(plies)
    assert {game: sorted(plies) for game, plies in covered.items()} == {
        game: sorted(plies) for game, plies in expected.items()
    }


def test_epoch_order_is_stable_within_an_epoch_and_moves_between_them(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    corpus = _corpus(
        write_corpus, tmp_path, _rows(normalized_row, 32), games_per_shard=8
    )
    config = SequenceLoaderConfig(split="train", batch_size=2, length_bucket_width=4)

    first = _drain(_loader(corpus, config))
    repeated = _drain(_loader(corpus, config))
    later = _loader(corpus, config)
    later.start_epoch(3)
    third = _drain(later)

    assert first == repeated
    assert first != third
    assert sorted(first) == sorted(third)


def test_a_planning_window_bounds_which_examples_share_a_batch(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row, 32))
    config = SequenceLoaderConfig(
        split="train",
        batch_size=8,
        length_bucket_width=None,
        shuffle=False,
    )

    narrow = _loader(corpus, config, StreamingLoaderConfig(planning_window_examples=4))
    try:
        assert all(batch.batch_size <= 4 for batch in narrow)
    finally:
        narrow.close()

    wide = _loader(corpus, config, StreamingLoaderConfig(planning_window_examples=32))
    try:
        assert next(wide).batch_size == 8
    finally:
        wide.close()


def test_drop_last_drops_only_the_batches_a_window_left_short(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row, 30))
    config = SequenceLoaderConfig(
        split="train",
        batch_size=4,
        length_bucket_width=None,
        shuffle=False,
    )

    kept = _drain(_loader(corpus, config))
    dropped = _drain(_loader(corpus, config.model_copy(update={"drop_last": True})))

    assert len(kept) == 30
    assert len(dropped) == 28
    assert set(dropped) < set(kept)


def test_resume_continues_the_epoch_from_the_saved_cursor(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    corpus = _corpus(
        write_corpus, tmp_path, _rows(normalized_row, 32), games_per_shard=8
    )
    config = SequenceLoaderConfig(split="train", batch_size=2, length_bucket_width=4)

    complete = _drain(_loader(corpus, config))
    interrupted = _loader(corpus, config)
    consumed = [
        sequence for _ in range(3) for sequence in _sequences(next(interrupted))
    ]
    saved = interrupted.state().as_record()
    interrupted.close()

    resumed = _loader(corpus, config)
    resumed.load_state(saved)

    assert consumed + _drain(resumed) == complete


def test_resume_across_epochs_restores_that_epoch_order(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row, 24))
    config = SequenceLoaderConfig(split="train", batch_size=2, length_bucket_width=4)

    reference = _loader(corpus, config)
    reference.start_epoch(2)
    expected = _drain(reference)

    interrupted = _loader(corpus, config)
    interrupted.start_epoch(2)
    consumed = [
        sequence for _ in range(2) for sequence in _sequences(next(interrupted))
    ]
    saved = interrupted.state()
    interrupted.close()

    resumed = _loader(corpus, config)
    resumed.load_state(saved)

    assert saved.epoch == 2
    assert consumed + _drain(resumed) == expected


def test_resume_rejects_a_cursor_from_different_data_or_configuration(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    rows = _rows(normalized_row, 16)
    corpus = _corpus(write_corpus, tmp_path / "one", rows)
    other = _corpus(write_corpus, tmp_path / "two", rows[:8], source_id="other")
    config = SequenceLoaderConfig(split="train", batch_size=2)

    loader = _loader(corpus, config)
    next(loader)
    saved = loader.state()
    loader.close()

    elsewhere = _loader(other, config)
    with pytest.raises(DataLoadingError, match="different sequence data"):
        elsewhere.load_state(saved)
    elsewhere.close()

    rebatched = _loader(corpus, config.model_copy(update={"batch_size": 4}))
    with pytest.raises(DataLoadingError, match="different loader configuration"):
        rebatched.load_state(saved)
    rebatched.close()


def test_resume_rejects_a_cursor_from_the_eager_loader(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row, 16))
    config = SequenceLoaderConfig(split="train", batch_size=2)
    eager = SequenceDataLoader.from_parquet(
        [shard.path for shard in corpus[0]],
        config,
    )
    next(eager)

    loader = _loader(corpus, config)
    with pytest.raises(DataLoadingError, match="different sequence data"):
        loader.load_state(eager.state())
    loader.close()


def test_resume_rejects_a_window_that_would_replan_the_epoch(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row, 16))
    config = SequenceLoaderConfig(split="train", batch_size=2)

    loader = _loader(corpus, config, StreamingLoaderConfig(planning_window_examples=8))
    next(loader)
    saved = loader.state()
    loader.close()

    replanned = _loader(
        corpus,
        config,
        StreamingLoaderConfig(planning_window_examples=4),
    )
    with pytest.raises(DataLoadingError, match="different loader configuration"):
        replanned.load_state(saved)
    replanned.close()


def test_worker_count_and_prefetch_depth_leave_the_cursor_alone(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row, 16))
    config = SequenceLoaderConfig(split="train", batch_size=2)

    alone = _loader(corpus, config, StreamingLoaderConfig(workers=0))
    next(alone)
    saved = alone.state()
    alone.close()

    parallel = _loader(
        corpus,
        config,
        StreamingLoaderConfig(workers=2, prefetch_batches=6),
    )
    parallel.load_state(saved)
    parallel.close()

    assert saved.configuration_sha256 == parallel.state().configuration_sha256


@pytest.mark.parametrize("workers", [1, 2])
def test_workers_produce_the_batches_the_plan_named(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    workers: int,
) -> None:
    corpus = _corpus(
        write_corpus, tmp_path, _rows(normalized_row, 16), games_per_shard=8
    )
    config = SequenceLoaderConfig(split="train", batch_size=2, length_bucket_width=4)

    alone = _drain(_loader(corpus, config, StreamingLoaderConfig(workers=0)))
    shared = _drain(
        _loader(
            corpus,
            config,
            StreamingLoaderConfig(workers=workers, prefetch_batches=3),
        )
    )

    assert shared == alone


def test_a_pool_larger_than_the_prefetch_depth_still_has_jobs_outstanding(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """One outstanding job per worker, and the prefetch depth on top of them.

    Read off the deque, because what a pool is being given is not observable
    from the batches that come back — only from timing, which a test cannot
    hold still.
    """

    corpus = _corpus(
        write_corpus, tmp_path, _rows(normalized_row, 64), games_per_shard=64
    )
    config = SequenceLoaderConfig(split="train", batch_size=2, length_bucket_width=4)
    streaming = StreamingLoaderConfig(workers=3, prefetch_batches=2)

    loader = _loader(corpus, config, streaming)
    try:
        next(loader)
        outstanding = len(loader._inflight)
    finally:
        loader.close()

    assert outstanding == streaming.workers + streaming.prefetch_batches


def test_identity_follows_the_manifest_the_shards_the_split_and_the_selection(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    rows = [
        *_rows(normalized_row, 12),
        *_rows(normalized_row, 4, split="validation"),
    ]
    shards, manifest_sha256 = _corpus(write_corpus, tmp_path, rows)

    def identity(**overrides: Any) -> str:
        arguments: dict[str, Any] = {
            "split": "train",
            "selection": SelectionConfig(),
            "chunk_length": None,
            "manifest_sha256": manifest_sha256,
        }
        arguments.update(overrides)
        return resolve_sharded_selection(shards, **arguments).identity_sha256

    baseline = identity()
    assert identity() == baseline
    assert identity(split="validation") != baseline
    assert identity(chunk_length=4) != baseline
    assert identity(selection=SelectionConfig(minimum_rating=1400)) != baseline
    assert identity(manifest_sha256="0" * 64) != baseline
    assert (
        resolve_sharded_selection(
            (replace(shards[0], sha256="0" * 64),),
            split="train",
            selection=SelectionConfig(),
            manifest_sha256=manifest_sha256,
        ).identity_sha256
        != baseline
    )


def test_identity_costs_no_decode_of_the_corpus(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row, 16))

    def refuse(*arguments: Any, **keywords: Any) -> None:
        raise AssertionError("opening the corpus decoded a game")

    monkeypatch.setattr("anthro_chess.data.streaming.encode_game", refuse)
    shards, manifest_sha256 = corpus
    selection = resolve_sharded_selection(
        shards,
        split="train",
        selection=SelectionConfig(),
        manifest_sha256=manifest_sha256,
    )

    assert selection.identity_sha256
    assert selection.resolution.selected_games == 16


def test_a_filter_keeps_the_games_the_eager_loader_keeps(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """One filter, two loaders, and only one of them counts what it rejected.

    So agreement is read off the games rather than off the record: the
    shard-backed loader drops rows as the epoch reaches them and never learns
    how many the split held.
    """

    rows = [
        *[normalized_row(index, split="train", rating=1200) for index in range(1, 9)],
        *[normalized_row(index, split="train", rating=1900) for index in range(9, 17)],
    ]
    corpus = _corpus(write_corpus, tmp_path, rows, games_per_shard=4)
    selection = SelectionConfig(minimum_rating=1500)
    config = SequenceLoaderConfig(split="train", batch_size=2, selection=selection)

    streaming = _loader(corpus, config)
    eager = SequenceDataLoader.from_parquet(
        [shard.path for shard in corpus[0]],
        config,
    )

    assert {game for game, _ in _drain(streaming)} == {
        example.game_id for example in eager.dataset
    }
    assert eager.resolution.excluded_games == {"below_minimum_rating": 8}
    assert streaming.resolution.excluded_games is None
    assert streaming.resolution.eligible_games is None
    streaming.close()


def test_a_selection_rejecting_nothing_opens_without_reading_a_row_group(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation counted every split, so opening one does not count it again.

    This is the whole reason a corpus opens in seconds, and a corpus holding
    more than one split is where it would break: a count read for the wrong
    split, or not read at all, sends the open down the counting path instead.
    """

    rows = [
        normalized_row(game_id, split=split, plies=6)
        for split, game_ids in (
            ("train", range(1, 9)),
            ("validation", range(9, 13)),
            ("test", range(13, 17)),
        )
        for game_id in game_ids
    ]
    corpus = _corpus(write_corpus, tmp_path, rows, games_per_shard=4)

    def refuse(*arguments: Any, **keywords: Any) -> None:
        raise AssertionError("opening the corpus read a row group")

    monkeypatch.setattr("anthro_chess.data.streaming.read_normalized_row_group", refuse)
    shards, manifest_sha256 = corpus
    selection = resolve_sharded_selection(
        shards,
        split="train",
        selection=SelectionConfig(),
        manifest_sha256=manifest_sha256,
    )

    assert selection.resolution.eligible_games == 8


def test_an_unsubsampled_selection_reads_every_eligible_game(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """The recorded count is not derived from the games, so it has to match.

    Nothing holds the ids any more, so a plan that skipped or repeated a game
    would leave a run training on a set its own record misdescribes.
    """

    corpus = _corpus(
        write_corpus, tmp_path, _rows(normalized_row, 16), games_per_shard=4
    )
    loader = _loader(corpus, SequenceLoaderConfig(split="train"))

    read = {game for game, _ in _drain(loader)}

    assert len(read) == 16
    assert loader.resolution.selected_games == 16
    assert loader.resolution.eligible_games == 16


def test_two_snapshots_rejecting_the_same_count_resolve_different_identities(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """The resolved record counts what a snapshot removed and not who.

    So the identity a resumed run is compared against has to carry the accounts
    itself. Without that, a run continues against a snapshot rejecting a
    different set of the same size and nothing notices.
    """

    shards, manifest_sha256 = _corpus(write_corpus, tmp_path, _rows(normalized_row, 16))
    selection = SelectionConfig(marked_accounts=Path("marked-accounts.txt"))

    def identity(*marked: int) -> ShardedSelection:
        return resolve_sharded_selection(
            shards,
            split="train",
            selection=selection,
            manifest_sha256=manifest_sha256,
            marked_digests=frozenset(marked),
        )

    first = identity(*(account_row_digest(f"white{game}") for game in (1, 2, 3)))
    second = identity(*(account_row_digest(f"white{game}") for game in (4, 5, 6)))

    assert first.resolution.excluded_games == second.resolution.excluded_games
    assert first.resolution.selected_games == second.resolution.selected_games
    assert first.identity_sha256 != second.identity_sha256


@pytest.mark.parametrize(
    "selection",
    [SelectionConfig(fraction=0.5), SelectionConfig(maximum_games=5)],
    ids=["fraction", "capped"],
)
def test_a_subsample_is_refused_rather_than_approximated(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    selection: SelectionConfig,
) -> None:
    """A cutoff over the rank keeps whatever share of a corpus it happens to.

    Refusing is what keeps the recorded size honest. Approximating would let a
    run record a count nobody trained on, and nothing holds the ids to notice.
    """

    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row, 16))

    with pytest.raises(DataLoadingError, match="cannot subsample"):
        _loader(corpus, SequenceLoaderConfig(split="train", selection=selection))


def test_a_split_the_corpus_does_not_hold_is_refused_at_the_open(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """The manifest counted every split, so this one costs nothing to catch."""

    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row, 8))

    with pytest.raises(DataLoadingError, match="holds no test games"):
        _loader(corpus, SequenceLoaderConfig(split="test"))


def test_a_filter_matching_nobody_yields_no_batches(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """Opening reads nothing, so this is found where the rows are.

    A run turns it into a refusal at its first step rather than at its first
    second, which `tests/training/test_runner.py` covers.
    """

    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row, 8))
    loader = _loader(
        corpus,
        SequenceLoaderConfig(
            split="train", selection=SelectionConfig(minimum_rating=3000)
        ),
    )

    assert list(loader) == []
    loader.close()


def test_an_unreadable_row_group_names_the_shard_it_could_not_be_read_from(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """Whoever reads this failure is the one who has to find the shard and
    rebuild it, and what the read itself raises locates no file at all.
    """

    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row, 8))
    shard = corpus[0][0].path
    raw = bytearray(shard.read_bytes())
    # A read parses the footer first, so corrupting only the data pages ahead
    # of it fails the row group rather than the open.
    footer = len(raw) - 8 - int.from_bytes(raw[-8:-4], "little")
    raw[4:footer] = bytes(footer - 4)
    shard.write_bytes(bytes(raw))
    loader = _loader(corpus, SequenceLoaderConfig(split="train"))

    with pytest.raises(DataLoadingError, match=str(shard)):
        list(loader)


def test_a_game_that_decodes_to_another_length_than_planned_fails_clearly(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    rows = _rows(normalized_row, 4)
    rows[0]["ply_count"] = rows[0]["ply_count"] + 2
    corpus = _corpus(write_corpus, tmp_path, rows)

    loader = _loader(
        corpus,
        SequenceLoaderConfig(split="train", batch_size=4, shuffle=False),
    )

    with pytest.raises(DataLoadingError, match="decodes to"):
        next(loader)
    loader.close()
