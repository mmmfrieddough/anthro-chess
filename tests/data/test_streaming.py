import json
from collections.abc import Callable
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
    build_sharded_index,
)
from anthro_chess.data.artifacts import (
    ShardIdentity,
    game_ids_sha256,
    normalized_shard_paths,
    validate_manifest_outputs,
)

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
    index = build_sharded_index(
        shards,
        split=config.split,
        selection=config.selection,
        chunk_length=config.chunk_length,
        manifest_sha256=manifest_sha256,
    )
    return StreamingSequenceDataLoader(
        index,
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
) -> list[dict[str, Any]]:
    return [
        normalized_row(
            game_id,
            split=split,
            plies=_PLY_COUNTS[game_id % len(_PLY_COUNTS)],
        )
        for game_id in range(1, count + 1)
    ]


def test_streams_exactly_the_games_the_eager_loader_selects(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    rows = _rows(normalized_row)
    corpus = _corpus(write_corpus, tmp_path, rows, games_per_shard=8, row_group_size=4)
    config = SequenceLoaderConfig(split="train", batch_size=3, length_bucket_width=4)

    streamed = _drain(_loader(corpus, config))
    eager = SequenceDataLoader.from_parquet(
        [shard.path for shard in corpus[0]],
        config,
    )

    assert sorted(streamed) == sorted(
        sequence for batch in eager for sequence in _sequences(batch)
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
        return build_sharded_index(shards, **arguments).identity_sha256

    baseline = identity()
    assert identity() == baseline
    assert identity(split="validation") != baseline
    assert identity(chunk_length=4) != baseline
    assert identity(selection=SelectionConfig(minimum_rating=1400)) != baseline
    assert identity(manifest_sha256="0" * 64) != baseline
    assert (
        build_sharded_index(
            (ShardIdentity(path=shards[0].path, sha256="0" * 64),),
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
        raise AssertionError("indexing decoded a game")

    monkeypatch.setattr("anthro_chess.data.streaming.encode_game", refuse)
    shards, manifest_sha256 = corpus
    index = build_sharded_index(
        shards,
        split="train",
        selection=SelectionConfig(),
        manifest_sha256=manifest_sha256,
    )

    assert index.identity_sha256
    assert index.games == 16


def test_selection_resolves_the_games_and_reasons_the_eager_loader_does(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    rows = [
        *[normalized_row(index, split="train", rating=1200) for index in range(1, 9)],
        *[normalized_row(index, split="train", rating=1900) for index in range(9, 17)],
    ]
    corpus = _corpus(write_corpus, tmp_path, rows, games_per_shard=4)
    selection = SelectionConfig(minimum_rating=1500, fraction=0.5)
    config = SequenceLoaderConfig(split="train", batch_size=2, selection=selection)

    streaming = _loader(corpus, config)
    eager = SequenceDataLoader.from_parquet(
        [shard.path for shard in corpus[0]],
        config,
    )

    assert streaming.resolution.as_record() == eager.resolution.as_record()
    assert streaming.resolution.excluded_games == {"below_minimum_rating": 8}
    assert streaming.resolution.selected_games == 4
    streaming.close()


@pytest.mark.parametrize(
    "selection",
    [
        SelectionConfig(),
        SelectionConfig(fraction=1.0),
        SelectionConfig(fraction=0.5),
        SelectionConfig(maximum_games=5),
    ],
    ids=["everything", "whole-fraction", "half", "capped"],
)
def test_the_resolved_digest_names_the_games_the_index_actually_holds(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    selection: SelectionConfig,
) -> None:
    """The digest is all that is kept of the ids, so it has to be theirs.

    A subsample that the index disagreed with would otherwise record a set of
    games no run ever read, and nothing downstream holds the ids to notice.
    """

    shards, manifest_sha256 = _corpus(
        write_corpus, tmp_path, _rows(normalized_row, 16), games_per_shard=4
    )

    index = build_sharded_index(
        shards,
        split="train",
        selection=selection,
        manifest_sha256=manifest_sha256,
    )

    held = [game_id for group in index.groups for game_id in group.game_ids]
    assert index.resolution.selected_games == len(held)
    assert index.resolution.game_ids_sha256 == game_ids_sha256(held)


def test_an_empty_selection_fails_instead_of_starting_a_run_on_nothing(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    corpus = _corpus(write_corpus, tmp_path, _rows(normalized_row, 8))

    with pytest.raises(DataLoadingError, match="no normalized games matched"):
        _loader(
            corpus,
            SequenceLoaderConfig(
                split="train",
                selection=SelectionConfig(minimum_rating=3000),
            ),
        )


def test_a_game_that_decodes_to_another_length_than_indexed_fails_clearly(
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
