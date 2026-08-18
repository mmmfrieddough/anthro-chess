"""Bounded-memory shard-backed sequence loading for corpus-scale training.

The eager loader reconstructs and retains every selected ply before the first
batch. That is right for fixtures and for a bounded proof slice, and it does
not scale: a per-ply encoding is orders of magnitude larger than the normalized
row it came from, so the whole-corpus materialization a million-game selection
would need does not fit in host memory and would be paid before the first
optimizer step regardless.

This loader keeps the same promises and holds almost none of it. What is
resident is one row group's projected columns, the batches currently in
flight, and a list naming which row groups the corpus has. Everything else is
decoded on the way past and released.

Nothing per game is resident. A batch's examples all come from one row group,
so which rows a row group contributes and how long each one decodes to are
derived from that row group at the moment it is reached, which makes what a run
pays to plan follow what it reads rather than what the corpus holds.

Opening a corpus therefore reads a footer per shard and nothing else, as long
as the selection rejects nothing: preparation already counted every split, and
a count it recorded is a count nobody has to take again. A selection that
filters has to look, and that is the one pass here whose cost follows corpus
size.

A resumed run replays the plan to its saved cursor, which re-derives the row
groups it passes over rather than decoding any game in them.
"""

from __future__ import annotations

import json
import logging
from array import array
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any

from anthro_chess.data.artifacts import (
    DataLoadingError,
    ShardIdentity,
    materialize_rows,
    normalized_row_group_count,
    open_normalized_shard,
    read_normalized_row_group,
    row_group_column,
    take_rows,
)
from anthro_chess.data.config import (
    SelectionConfig,
    SequenceLoaderConfig,
    StreamingLoaderConfig,
)
from anthro_chess.data.encoding import PlyEncoding, encode_game
from anthro_chess.data.loading import (
    _LOADER_COLUMNS,
    _MARKED_COLUMNS,
    LOADER_STATE_VERSION,
    SelectionResolution,
    SequenceBatch,
    SequenceBatchSource,
    SequenceExample,
    SequenceLoaderState,
    _exclusion_reason,
    _game_from_row,
    _length_bucket,
    _state_from_record,
    collate_sequences,
    loader_configuration_sha256,
    require_resolved_snapshot,
    subsample_size,
)
from anthro_chess.data.schema import (
    NormalizedColumn,
    row_game_id,
)
from anthro_chess.data.termination import TerminalActionStatus

#: Bumped when the shard-backed identity or plan changes shape, so a checkpoint
#: written by an earlier one is refused rather than silently replanned.
STREAMING_IDENTITY_VERSION = 2
#: How the loader names itself in identities and run records.
STREAMING_LOADER_NAME = "shard-backed"

logger = logging.getLogger(__name__)

#: How many shards are opened at once to learn how many row groups they hold.
#: The read is a footer parse rather than a scan, so this is bound by the drive.
_FOOTER_READERS = 8

#: What every pass over a row group reads, because both of them start by
#: dropping the rows of other splits.
_SPLIT_COLUMNS = (NormalizedColumn.SPLIT,)
#: What planning adds to that. ``ply_count`` counts moves and
#: ``terminal_action_status`` says whether one further action was appended,
#: which together give a game's encoded length without touching its actions.
_LENGTH_COLUMNS = (
    NormalizedColumn.PLY_COUNT,
    NormalizedColumn.TERMINAL_ACTION_STATUS,
)
#: Joined to that projection only when the selection filters on them.
_FILTER_COLUMNS = (
    NormalizedColumn.WHITE_NORMALIZED_RATING,
    NormalizedColumn.BLACK_NORMALIZED_RATING,
    NormalizedColumn.TIME_INITIAL_MS,
    NormalizedColumn.TIME_INCREMENT_MS,
)


@dataclass(frozen=True)
class _RowGroup:
    """One row group of one shard, as the epoch order names it."""

    shard: int
    row_group: int


@dataclass(frozen=True)
class _Example:
    """One planned full game or contiguous chunk, by its row in a row group.

    ``game_length`` is the whole game's encoded length rather than this
    example's, because it is what a decoded game is checked against and a chunk
    is not the game.
    """

    position: int
    start_ply: int
    length: int
    game_length: int


@dataclass(frozen=True)
class _PlannedBatch:
    """One batch's examples, which all come from a single row group."""

    shard: int
    row_group: int
    examples: tuple[_Example, ...]


@dataclass(frozen=True)
class _BatchJob:
    """Everything a worker needs to turn planned examples into one batch.

    The rows travel with the job rather than being re-read in the worker. They
    are small next to what they decode into, and sending them keeps every
    Parquet read sequential in one process instead of having every worker seek
    into the same shard.

    They travel as the table they were gathered into, and the worker is what
    turns them into rows of Python values. That conversion costs an object per
    field of every game in the batch, so leaving it in the parent would make
    the one process every batch passes through the slowest part of the loader
    at a wide batch, however many workers were decoding behind it.
    """

    shard: int
    path: str
    row_table: Any
    lengths: tuple[int, ...]
    entries: tuple[tuple[int, int, int], ...]
    legal_actions: bool


@dataclass(frozen=True)
class ShardedSelection:
    """One split of a prepared corpus, as the shard-backed loader reads it.

    Nothing here is per game. The epoch order is a function of which row groups
    exist, and which rows a row group contributes is derived from that row
    group when the plan reaches it.
    """

    shards: tuple[ShardIdentity, ...]
    row_groups: tuple[_RowGroup, ...]
    split: str
    chunk_length: int | None
    selection: SelectionConfig
    marked_digests: frozenset[int] | None
    resolution: SelectionResolution
    identity_sha256: str

    @property
    def games(self) -> int:
        """Return how many games the selection kept."""

        return self.resolution.selected_games


def resolve_sharded_selection(
    shards: Sequence[ShardIdentity],
    *,
    split: str,
    selection: SelectionConfig,
    chunk_length: int | None = None,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    marked_digests: frozenset[int] | None = None,
) -> ShardedSelection:
    """Open one split of a prepared corpus without reading any game."""

    require_resolved_snapshot(selection, marked_digests)
    if not shards:
        raise DataLoadingError("at least one normalized shard is required")
    row_groups = _enumerate_row_groups(shards)
    eligible, excluded = _resolve_counts(
        shards,
        row_groups,
        split=split,
        selection=selection,
        manifest=manifest,
        marked_digests=marked_digests,
    )
    if not eligible:
        raise DataLoadingError("no normalized games matched the loader selection")
    _reject_subsample(eligible, selection)
    resolution = SelectionResolution(
        spec=selection.model_dump(mode="json"),
        eligible_games=eligible,
        selected_games=eligible,
        excluded_games=excluded,
    )
    identity = {
        "version": STREAMING_IDENTITY_VERSION,
        "loader": STREAMING_LOADER_NAME,
        "split": split,
        "chunk_length": chunk_length,
        "manifest_sha256": manifest_sha256,
        "shards": [
            {"name": shard.path.name, "sha256": shard.sha256} for shard in shards
        ],
        "selection": resolution.as_identity_record(),
    }
    logger.info(
        "Opened %s of %s eligible %s game(s) across %s row group(s) in %s shard(s)",
        resolution.selected_games,
        resolution.eligible_games,
        split,
        len(row_groups),
        len(shards),
    )
    return ShardedSelection(
        shards=tuple(shards),
        row_groups=row_groups,
        split=split,
        chunk_length=chunk_length,
        selection=selection,
        marked_digests=marked_digests,
        resolution=resolution,
        identity_sha256=sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def _enumerate_row_groups(shards: Sequence[ShardIdentity]) -> tuple[_RowGroup, ...]:
    """Return every row group of every shard, in shard order."""

    readers = ThreadPoolExecutor(max_workers=_FOOTER_READERS)
    try:
        counts = list(readers.map(_shard_row_group_count, shards))
    finally:
        readers.shutdown(wait=False, cancel_futures=True)
    return tuple(
        _RowGroup(shard=shard_index, row_group=row_group)
        for shard_index, count in enumerate(counts)
        for row_group in range(count)
    )


def _shard_row_group_count(shard: ShardIdentity) -> int:
    return normalized_row_group_count(open_normalized_shard(shard.path))


def _filters_rows(
    selection: SelectionConfig,
    marked_digests: frozenset[int] | None,
) -> bool:
    """Say whether the selection can reject a row of the split it names."""

    return marked_digests is not None or any(
        (
            selection.speed is not None,
            selection.require_ratings,
            selection.minimum_time_initial_ms is not None,
            selection.maximum_time_initial_ms is not None,
            selection.minimum_time_increment_ms is not None,
            selection.maximum_time_increment_ms is not None,
            selection.minimum_rating is not None,
            selection.maximum_rating is not None,
        )
    )


def _resolve_counts(
    shards: Sequence[ShardIdentity],
    row_groups: Sequence[_RowGroup],
    *,
    split: str,
    selection: SelectionConfig,
    manifest: Mapping[str, Any],
    marked_digests: frozenset[int] | None,
) -> tuple[int, dict[str, int]]:
    """Return how many games the split offers, and why the rest were rejected.

    A selection that rejects nothing is answered from the manifest, which
    already counted every split when the corpus was written. One that filters
    has to look, and looking costs a projected read and a check per row of the
    split.
    """

    if not _filters_rows(selection, marked_digests):
        recorded = _manifest_split_games(manifest, split)
        if recorded is not None:
            return recorded, {}
    eligible = 0
    excluded: dict[str, int] = {}
    columns = (
        _SPLIT_COLUMNS + _FILTER_COLUMNS + (_MARKED_COLUMNS if marked_digests else ())
    )
    for group in row_groups:
        reader = open_normalized_shard(shards[group.shard].path)
        table = read_normalized_row_group(reader, group.row_group, columns)
        values = {
            column.value: row_group_column(table, column.value) for column in columns
        }
        splits = values[NormalizedColumn.SPLIT]
        for position in range(len(splits)):
            if splits[position] != split:
                continue
            row = {
                column: column_values[position]
                for column, column_values in values.items()
            }
            reason = _exclusion_reason(row, selection, marked_digests)
            if reason is None:
                eligible += 1
            else:
                excluded[reason] = excluded.get(reason, 0) + 1
    return eligible, excluded


def _manifest_split_games(manifest: Mapping[str, Any], split: str) -> int | None:
    """Return what a manifest recorded for one split, or ``None`` if it cannot.

    Preparation counts every split as it writes each shard, so a selection that
    rejects nothing has already been counted and does not have to be counted
    again. A manifest predating those per-shard counts answers nothing here
    rather than answering wrongly.
    """

    output = manifest.get("output")
    shards = output.get("shards") if isinstance(output, Mapping) else None
    if not isinstance(shards, list) or not shards:
        return None
    total = 0
    for shard in shards:
        counts = shard.get("split_counts") if isinstance(shard, Mapping) else None
        if not isinstance(counts, Mapping):
            return None
        recorded = counts.get(split)
        if type(recorded) is not int:
            return None
        total += recorded
    return total


def _reject_subsample(eligible: int, selection: SelectionConfig) -> None:
    """Refuse a subsample this loader could not resolve without reading it all.

    Taking the lowest-ranked share of a split means ranking every candidate,
    which needs each one's identity, which is a digest per game over the whole
    corpus. That is the pass this loader exists not to make, and a cutoff over
    the same rank does not stand in for it: the count it lands on is the count
    it lands on, so a run would record a size it did not train on.

    A dial that keeps the whole split ranks nothing, so it is admitted. Only a
    selection asking for fewer games than it found has to choose which, and the
    eager loader still ranks, which is where a selection small enough to hold
    belongs.
    """

    if subsample_size(eligible, selection) >= eligible:
        return
    raise DataLoadingError(
        "the shard-backed loader cannot subsample a selection: ranking every "
        "candidate needs a digest per game of the whole corpus. Select with "
        "the filters instead, or read a selection small enough for the eager "
        "loader"
    )


class StreamingSequenceDataLoader(SequenceBatchSource):
    """Deterministic shard-backed batch iterator with explicit resume state.

    An epoch orders row groups, then the games inside each one, then cuts that
    stream into planning windows. A window is where length buckets fill and
    where they are flushed, so every example in a batch comes from one row
    group and a batch is read with a single columnar take.

    That is a different order from the eager loader's global shuffle, and
    deliberately so: a global shuffle over a corpus means a seek per example.
    The identities differ accordingly, so a run cannot resume across the two
    and quietly train on a different order than it recorded.
    """

    def __init__(
        self,
        selection: ShardedSelection,
        config: SequenceLoaderConfig,
        streaming: StreamingLoaderConfig,
        *,
        legal_actions: bool = True,
    ) -> None:
        if config.split != selection.split:
            raise DataLoadingError("loader split does not match the sequence selection")
        if config.chunk_length != selection.chunk_length:
            raise DataLoadingError(
                "loader chunk_length does not match the sequence selection"
            )
        if config.selection != selection.selection:
            raise DataLoadingError(
                "loader selection does not match the sequence selection"
            )
        self.corpus = selection
        self.config = config
        self.streaming = streaming
        # Outside the configuration digest for the same reason as in the eager
        # loader: it decides what a decoded ply carries, not which games the
        # selection holds.
        self.legal_actions = legal_actions
        self.configuration_sha256 = sha256(
            json.dumps(
                {
                    "loader": loader_configuration_sha256(config),
                    # Only the window, because only the window decides which
                    # examples share a batch. Worker count and prefetch depth
                    # are what the machine can afford, and a run resumed on
                    # another machine should be free to afford differently.
                    "planning_window_examples": streaming.planning_window_examples,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self._pool: ProcessPoolExecutor | None = None
        self._reader: Any | None = None
        self._reader_shard: int | None = None
        self._table: Any | None = None
        self._table_group: tuple[int, int] | None = None
        self._inflight: deque[Future[SequenceBatch]] = deque()
        self._epoch = 0
        self._position = 0
        self._plan = self._plan_epoch(self._epoch)

    def __iter__(self) -> StreamingSequenceDataLoader:
        return self

    def __next__(self) -> SequenceBatch:
        self._fill()
        if not self._inflight:
            raise StopIteration
        batch = self._inflight.popleft().result()
        self._position += 1
        self._fill()
        return batch

    @property
    def identity_sha256(self) -> str:
        """Return the manifest-derived identity of the selected corpus."""

        return self.corpus.identity_sha256

    @property
    def resolution(self) -> SelectionResolution:
        """Return which games the configured selection kept."""

        return self.corpus.resolution

    def state(self) -> SequenceLoaderState:
        """Return the exact next-batch cursor for checkpointing."""

        return SequenceLoaderState(
            version=LOADER_STATE_VERSION,
            dataset_sha256=self.corpus.identity_sha256,
            configuration_sha256=self.configuration_sha256,
            epoch=self._epoch,
            position=self._position,
        )

    def load_state(self, state: SequenceLoaderState | Mapping[str, object]) -> None:
        """Restore a compatible saved cursor and deterministic epoch order.

        Replaying the plan re-derives every row group it passes, which is a
        projected read per row group and no decoding at all. An epoch over a
        corpus is millions of batches, so a cursor a run actually reached sits
        a few row groups in.
        """

        parsed = (
            state
            if isinstance(state, SequenceLoaderState)
            else _state_from_record(state)
        )
        if parsed.version != LOADER_STATE_VERSION:
            raise DataLoadingError(
                f"unsupported loader state version: {parsed.version}"
            )
        if parsed.dataset_sha256 != self.corpus.identity_sha256:
            raise DataLoadingError("loader state belongs to different sequence data")
        if parsed.configuration_sha256 != self.configuration_sha256:
            raise DataLoadingError("loader state uses different loader configuration")
        self.start_epoch(parsed.epoch)
        for _ in range(parsed.position):
            if next(self._plan, None) is None:
                raise DataLoadingError(
                    "loader state position is outside the epoch plan"
                )
        self._position = parsed.position

    def start_epoch(self, epoch: int) -> None:
        """Start a deterministic epoch from its first example."""

        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self._drain()
        self._epoch = epoch
        self._position = 0
        self._plan = self._plan_epoch(epoch)

    def close(self) -> None:
        """Release the worker pool and the row group currently resident."""

        self._drain()
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None
        self._reader = None
        self._reader_shard = None
        self._table = None
        self._table_group = None

    def _plan_epoch(self, epoch: int) -> Iterator[_PlannedBatch]:
        """Yield one epoch's batches as plans, decoding nothing.

        Generated rather than materialized so a corpus-scale epoch costs a
        cursor instead of a list of every example in it, and so a resumed run
        can skip forward through the plan at the price of one projected read
        per row group it passes.
        """

        order = list(self.corpus.row_groups)
        if self.config.shuffle:
            order.sort(key=partial(_group_key, self.config.seed, epoch))
        for group in order:
            yield from self._plan_row_group(group, epoch)

    def _plan_row_group(
        self,
        group: _RowGroup,
        epoch: int,
    ) -> Iterator[_PlannedBatch]:
        """Yield the batches one row group contributes to an epoch."""

        positions, lengths = self._eligible_rows(group)
        rows = list(zip(positions, lengths, strict=True))
        if self.config.shuffle:
            rows.sort(key=partial(_game_key, self.config.seed, epoch, group))
        examples = [
            example
            for position, length in rows
            for example in _examples_for(position, length, self.corpus.chunk_length)
        ]
        window = self.streaming.planning_window_examples
        for start in range(0, len(examples), window):
            for batch in _window_batches(
                examples[start : start + window], self.config, epoch
            ):
                yield _PlannedBatch(
                    shard=group.shard,
                    row_group=group.row_group,
                    examples=batch,
                )

    def _eligible_rows(self, group: _RowGroup) -> tuple[array[int], array[int]]:
        """Return which rows of one row group the selection keeps, and how long.

        Reading this here rather than before the run is what keeps the whole
        loader free of per-game state: the answer is wanted once, in the order
        the epoch visits row groups, and it is derived from columns cheap
        enough to project.
        """

        selection = self.corpus.selection
        marked = self.corpus.marked_digests
        filtered = _filters_rows(selection, marked)
        columns = (
            _SPLIT_COLUMNS
            + _LENGTH_COLUMNS
            + (_FILTER_COLUMNS if filtered else ())
            + (_MARKED_COLUMNS if marked else ())
        )
        table = read_normalized_row_group(
            self._shard_reader(group.shard), group.row_group, columns
        )
        values = {
            column.value: row_group_column(table, column.value) for column in columns
        }
        splits = values[NormalizedColumn.SPLIT]
        ply_counts = values[NormalizedColumn.PLY_COUNT]
        terminal = values[NormalizedColumn.TERMINAL_ACTION_STATUS]
        positions: array[int] = array("I")
        lengths: array[int] = array("i")
        for position in range(len(splits)):
            if splits[position] != self.corpus.split:
                continue
            if filtered:
                row = {
                    column: column_values[position]
                    for column, column_values in values.items()
                }
                if _exclusion_reason(row, selection, marked) is not None:
                    continue
            appended = terminal[position] == TerminalActionStatus.APPENDED
            positions.append(position)
            lengths.append(ply_counts[position] + (1 if appended else 0))
        return positions, lengths

    def _fill(self) -> None:
        # One job per worker plus the declared depth, never the depth alone:
        # that would leave a larger pool with nothing to do. The reason the two
        # add is in `StreamingLoaderConfig`.
        workers = self.streaming.workers
        depth = workers + self.streaming.prefetch_batches if workers else 1
        while len(self._inflight) < depth:
            planned = next(self._plan, None)
            if planned is None:
                return
            self._inflight.append(self._submit(planned))

    def _submit(self, planned: _PlannedBatch) -> Future[SequenceBatch]:
        job = self._job(planned)
        if self.streaming.workers:
            return self._executor().submit(_materialize_batch, job)
        immediate: Future[SequenceBatch] = Future()
        immediate.set_result(_materialize_batch(job))
        return immediate

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            # Whatever the platform starts processes with, which is what every
            # other dataloader on it uses. Naming a method instead would trade
            # one hazard for another: forking is the risk on a host whose
            # runtime dislikes it, and spawning re-imports the entry point, so
            # it fails outright wherever the entry point is not importable.
            # A worker touches Parquet and chess logic and no accelerator.
            self._pool = ProcessPoolExecutor(max_workers=self.streaming.workers)
        return self._pool

    def _job(self, planned: _PlannedBatch) -> _BatchJob:
        table = self._row_group_table(planned.shard, planned.row_group)
        rows = sorted({example.position for example in planned.examples})
        row_index = {position: index for index, position in enumerate(rows)}
        game_lengths = {
            example.position: example.game_length for example in planned.examples
        }
        return _BatchJob(
            shard=planned.shard,
            path=str(self.corpus.shards[planned.shard].path),
            row_table=take_rows(table, rows),
            lengths=tuple(game_lengths[position] for position in rows),
            entries=tuple(
                (row_index[example.position], example.start_ply, example.length)
                for example in planned.examples
            ),
            legal_actions=self.legal_actions,
        )

    def _shard_reader(self, shard: int) -> Any:
        if self._reader_shard != shard:
            self._reader = open_normalized_shard(self.corpus.shards[shard].path)
            self._reader_shard = shard
        return self._reader

    def _row_group_table(self, shard: int, row_group: int) -> Any:
        if self._table_group == (shard, row_group):
            return self._table
        reader = self._shard_reader(shard)
        # One row group at a time, and the previous one released before the
        # next is read. This is the loader's largest resident structure and the
        # only one preparation's shard sizing decides.
        self._table = None
        self._table = read_normalized_row_group(reader, row_group, _LOADER_COLUMNS)
        self._table_group = (shard, row_group)
        return self._table

    def _drain(self) -> None:
        for pending in self._inflight:
            pending.cancel()
        self._inflight.clear()


def _materialize_batch(job: _BatchJob) -> SequenceBatch:
    """Decode one batch's games and pack them, in a worker or in place."""

    path = Path(job.path)
    rows = materialize_rows(job.row_table)
    decoded: dict[int, tuple[PlyEncoding, ...]] = {}
    examples: list[SequenceExample] = []
    for row_index, start_ply, length in job.entries:
        plies = decoded.get(row_index)
        if plies is None:
            row = rows[row_index]
            plies = encode_game(
                _game_from_row(row, path, row_game_id(row)),
                legal_actions=job.legal_actions,
            )
            if len(plies) != job.lengths[row_index]:
                raise DataLoadingError(
                    f"{path} game {row_game_id(row)} "
                    f"decodes to {len(plies)} action(s) where its ply count and "
                    f"terminal action status describe {job.lengths[row_index]}"
                )
            decoded[row_index] = plies
        chunk = plies[start_ply : start_ply + length]
        examples.append(
            SequenceExample(
                shard_index=job.shard,
                game_id=chunk[0].game_id,
                start_ply=chunk[0].ply_index,
                plies=chunk,
            )
        )
    return collate_sequences(examples)


def _examples_for(
    position: int,
    length: int,
    chunk_length: int | None,
) -> Iterator[_Example]:
    """Expand one game into the full sequence or the chunks it is cut into."""

    if chunk_length is None:
        yield _Example(
            position=position, start_ply=0, length=length, game_length=length
        )
        return
    if type(chunk_length) is not int or chunk_length < 1:
        raise ValueError("chunk_length must be a positive integer or None")
    for start in range(0, length, chunk_length):
        yield _Example(
            position=position,
            start_ply=start,
            length=min(chunk_length, length - start),
            game_length=length,
        )


def _window_batches(
    window: Sequence[_Example],
    config: SequenceLoaderConfig,
    epoch: int,
) -> Iterator[tuple[_Example, ...]]:
    """Fill length buckets across one window and flush what is left of it."""

    buckets: dict[int, list[_Example]] = {}
    batches: list[tuple[_Example, ...]] = []
    for example in window:
        bucket = buckets.setdefault(
            _length_bucket(example.length, config.length_bucket_width),
            [],
        )
        bucket.append(example)
        if len(bucket) == config.batch_size:
            batches.append(tuple(bucket))
            bucket.clear()
    for key in sorted(buckets):
        remainder = buckets[key]
        if not remainder:
            continue
        if config.drop_last and len(remainder) < config.batch_size:
            continue
        batches.append(tuple(remainder))
    if config.shuffle:
        batches.sort(key=partial(_batch_key, config.seed, epoch))
    yield from batches


def _group_key(seed: str, epoch: int, group: _RowGroup) -> bytes:
    return sha256(
        f"{seed}\0{epoch}\0group\0{group.shard}\0{group.row_group}".encode()
    ).digest()


def _game_key(seed: str, epoch: int, group: _RowGroup, row: tuple[int, int]) -> bytes:
    return sha256(
        f"{seed}\0{epoch}\0{group.shard}\0{group.row_group}\0{row[0]}".encode()
    ).digest()


def _batch_key(seed: str, epoch: int, batch: Sequence[_Example]) -> bytes:
    members = ",".join(f"{example.position}:{example.start_ply}" for example in batch)
    return sha256(f"{seed}\0{epoch}\0batch\0{members}".encode()).digest()


__all__ = [
    "STREAMING_IDENTITY_VERSION",
    "STREAMING_LOADER_NAME",
    "ShardedSelection",
    "StreamingSequenceDataLoader",
    "resolve_sharded_selection",
]
