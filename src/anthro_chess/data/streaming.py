"""Bounded-memory shard-backed sequence loading for corpus-scale training.

The eager loader reconstructs and retains every selected ply before the first
batch. That is right for fixtures and for a bounded proof slice, and it does
not scale: a per-ply encoding is orders of magnitude larger than the normalized
row it came from, so the whole-corpus materialization a million-game selection
would need does not fit in host memory and would be paid before the first
optimizer step regardless.

This loader keeps the same promises and holds almost none of it. What is
resident is one row group's projected columns, an index naming which games each
row group holds and how long each one is, and the batches currently in flight.
Everything else is decoded on the way past and released.

The index is what makes that possible. Sequence length is derivable from the
normalized schema without decoding a game, so the epoch's whole batch plan is a
pure function of columns cheap enough to read for a corpus. A resumed run
replays that plan to its saved cursor without reading or decoding anything,
and the plan is what a batch's rows are read against rather than the other way
around.
"""

from __future__ import annotations

import json
import logging
from array import array
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any

from anthro_chess.data.artifacts import (
    DataLoadingError,
    ShardIdentity,
    normalized_row_group_count,
    open_normalized_shard,
    read_normalized_row_group,
    row_group_column,
    rows_at_positions,
)
from anthro_chess.data.config import (
    SelectionConfig,
    SequenceLoaderConfig,
    StreamingLoaderConfig,
)
from anthro_chess.data.encoding import PlyEncoding, encode_game
from anthro_chess.data.loading import (
    _LOADER_COLUMNS,
    LOADER_STATE_VERSION,
    SelectionResolution,
    SequenceBatch,
    SequenceBatchSource,
    SequenceExample,
    SequenceLoaderState,
    _exclusion_reason,
    _game_from_row,
    _length_bucket,
    _rank_key,
    _state_from_record,
    collate_sequences,
)
from anthro_chess.data.schema import NormalizedColumn
from anthro_chess.data.termination import TerminalActionStatus

#: Bumped when the shard-backed identity or plan changes shape, so a checkpoint
#: written by an earlier one is refused rather than silently replanned.
STREAMING_IDENTITY_VERSION = 1
#: How the loader names itself in identities and run records.
STREAMING_LOADER_NAME = "shard-backed"

logger = logging.getLogger(__name__)

#: What the index pass reads. Every column here is fixed-width or dictionary
#: encoded, so a pass over a corpus-scale shard costs a projection rather than
#: a decode. ``ply_count`` counts moves and ``terminal_action_status`` says
#: whether one further action was appended, which together give the encoded
#: length of a game without touching its actions.
_INDEX_COLUMNS = (
    NormalizedColumn.GAME_ID,
    NormalizedColumn.PLY_COUNT,
    NormalizedColumn.TERMINAL_ACTION_STATUS,
    NormalizedColumn.WHITE_NORMALIZED_RATING,
    NormalizedColumn.BLACK_NORMALIZED_RATING,
    NormalizedColumn.TIME_INITIAL_MS,
    NormalizedColumn.TIME_INCREMENT_MS,
    NormalizedColumn.SPLIT,
)


@dataclass(frozen=True)
class _RowGroupIndex:
    """Which selected games one row group holds, in file order.

    The three parallel arrays are typed storage rather than tuples of objects
    on purpose: at corpus scale this index is resident for the whole run, and
    the difference between twelve bytes a game and a hundred is the difference
    between a rounding error and a real share of host memory.
    """

    shard: int
    row_group: int
    positions: array[int]
    game_ids: array[int]
    lengths: array[int]

    def __len__(self) -> int:
        return len(self.game_ids)


@dataclass(frozen=True)
class _Example:
    """One planned full game or contiguous chunk, named by where it lives."""

    group: int
    slot: int
    start_ply: int
    length: int


@dataclass(frozen=True)
class _BatchJob:
    """Everything a worker needs to turn planned examples into one batch.

    The rows travel with the job rather than being re-read in the worker. They
    are small next to what they decode into, and sending them keeps every
    Parquet read sequential in one process instead of having every worker seek
    into the same shard.
    """

    shard: int
    path: str
    rows: tuple[Mapping[str, Any], ...]
    lengths: tuple[int, ...]
    entries: tuple[tuple[int, int, int], ...]
    legal_actions: bool


@dataclass(frozen=True)
class ShardedSequenceIndex:
    """Where every selected game lives, and how long it is when decoded.

    This is the whole corpus as the loader knows it. Building it reads the
    projected index columns once; nothing here required decoding a game or
    hashing a shard, because the manifest already established what the shards
    are.
    """

    shards: tuple[ShardIdentity, ...]
    groups: tuple[_RowGroupIndex, ...]
    split: str
    chunk_length: int | None
    selection: SelectionConfig
    resolution: SelectionResolution
    identity_sha256: str

    @property
    def games(self) -> int:
        """Return how many games the selection kept."""

        return self.resolution.selected_games


def build_sharded_index(
    shards: Sequence[ShardIdentity],
    *,
    split: str,
    selection: SelectionConfig,
    chunk_length: int | None = None,
    manifest_sha256: str,
) -> ShardedSequenceIndex:
    """Index one split of a prepared corpus without decoding any game."""

    if not shards:
        raise DataLoadingError("at least one normalized shard is required")
    logger.info("Indexing %s shard(s) for the %s split", len(shards), split)

    scanned: list[_ScannedGroup] = []
    eligible: array[int] = array("Q")
    excluded: dict[str, int] = {}
    for shard_index, shard in enumerate(shards):
        reader = open_normalized_shard(shard.path)
        for row_group in range(normalized_row_group_count(reader)):
            group = _scan_row_group(
                reader,
                shard_index=shard_index,
                row_group=row_group,
                split=split,
                selection=selection,
                excluded=excluded,
            )
            eligible.extend(group.game_ids)
            scanned.append(group)

    kept = _kept_game_ids(eligible, selection)
    groups = tuple(_selected_groups(scanned, kept))
    resolution = SelectionResolution(
        spec=selection.model_dump(mode="json"),
        game_ids=tuple(sorted(kept)),
        eligible_games=len(eligible),
        excluded_games=excluded,
    )
    if not groups:
        raise DataLoadingError("no normalized games matched the loader selection")

    identity = {
        "version": STREAMING_IDENTITY_VERSION,
        "loader": STREAMING_LOADER_NAME,
        "split": split,
        "chunk_length": chunk_length,
        "manifest_sha256": manifest_sha256,
        "shards": [
            {"name": shard.path.name, "sha256": shard.sha256} for shard in shards
        ],
        "selection": resolution.as_record(),
    }
    logger.info(
        "Indexed %s of %s eligible %s game(s) across %s row group(s)",
        resolution.selected_games,
        resolution.eligible_games,
        split,
        len(groups),
    )
    return ShardedSequenceIndex(
        shards=tuple(shards),
        groups=groups,
        split=split,
        chunk_length=chunk_length,
        selection=selection,
        resolution=resolution,
        identity_sha256=sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
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
        index: ShardedSequenceIndex,
        config: SequenceLoaderConfig,
        streaming: StreamingLoaderConfig,
        *,
        legal_actions: bool = True,
    ) -> None:
        if config.split != index.split:
            raise DataLoadingError("loader split does not match the sequence index")
        if config.chunk_length != index.chunk_length:
            raise DataLoadingError(
                "loader chunk_length does not match the sequence index"
            )
        if config.selection != index.selection:
            raise DataLoadingError("loader selection does not match the sequence index")
        self.index = index
        self.config = config
        self.streaming = streaming
        # Outside the configuration digest for the same reason as in the eager
        # loader: it decides what a batch carries, not which games it holds.
        self.legal_actions = legal_actions
        self.configuration_sha256 = sha256(
            json.dumps(
                {
                    "loader": config.model_dump(mode="json"),
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
        self._table_group: int | None = None
        self._inflight: deque[Future[SequenceBatch]] = deque()
        self._epoch = 0
        self._position = 0
        self._plan = _plan_epoch(index, config, streaming, self._epoch)

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
        """Return the manifest-derived identity of the indexed corpus."""

        return self.index.identity_sha256

    @property
    def resolution(self) -> SelectionResolution:
        """Return which games the configured selection kept."""

        return self.index.resolution

    def state(self) -> SequenceLoaderState:
        """Return the exact next-batch cursor for checkpointing."""

        return SequenceLoaderState(
            version=LOADER_STATE_VERSION,
            dataset_sha256=self.index.identity_sha256,
            configuration_sha256=self.configuration_sha256,
            epoch=self._epoch,
            position=self._position,
        )

    def load_state(self, state: SequenceLoaderState | Mapping[str, object]) -> None:
        """Restore a compatible saved cursor and deterministic epoch order."""

        parsed = (
            state
            if isinstance(state, SequenceLoaderState)
            else _state_from_record(state)
        )
        if parsed.version != LOADER_STATE_VERSION:
            raise DataLoadingError(
                f"unsupported loader state version: {parsed.version}"
            )
        if parsed.dataset_sha256 != self.index.identity_sha256:
            raise DataLoadingError("loader state belongs to different sequence data")
        if parsed.configuration_sha256 != self.configuration_sha256:
            raise DataLoadingError("loader state uses different loader configuration")
        self.start_epoch(parsed.epoch)
        # Replaying the plan touches the index alone, so a cursor deep into a
        # corpus-scale epoch is restored without reading or decoding anything.
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
        self._plan = _plan_epoch(self.index, self.config, self.streaming, epoch)

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

    def _fill(self) -> None:
        depth = self.streaming.prefetch_batches if self.streaming.workers else 1
        while len(self._inflight) < depth:
            planned = next(self._plan, None)
            if planned is None:
                return
            self._inflight.append(self._submit(planned))

    def _submit(self, planned: tuple[_Example, ...]) -> Future[SequenceBatch]:
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

    def _job(self, planned: tuple[_Example, ...]) -> _BatchJob:
        group = self.index.groups[planned[0].group]
        table = self._row_group_table(planned[0].group)
        slots = sorted({example.slot for example in planned})
        rows = rows_at_positions(table, [group.positions[slot] for slot in slots])
        row_index = {slot: index for index, slot in enumerate(slots)}
        return _BatchJob(
            shard=group.shard,
            path=str(self.index.shards[group.shard].path),
            rows=tuple(rows),
            lengths=tuple(group.lengths[slot] for slot in slots),
            entries=tuple(
                (row_index[example.slot], example.start_ply, example.length)
                for example in planned
            ),
            legal_actions=self.legal_actions,
        )

    def _row_group_table(self, group_index: int) -> Any:
        if self._table_group == group_index:
            return self._table
        group = self.index.groups[group_index]
        if self._reader_shard != group.shard:
            self._reader = open_normalized_shard(self.index.shards[group.shard].path)
            self._reader_shard = group.shard
        # One row group at a time, and the previous one released before the
        # next is read. This is the loader's largest resident structure and the
        # only one preparation's shard sizing decides.
        self._table = None
        self._table = read_normalized_row_group(
            self._reader,
            group.row_group,
            _LOADER_COLUMNS,
        )
        self._table_group = group_index
        return self._table

    def _drain(self) -> None:
        for pending in self._inflight:
            pending.cancel()
        self._inflight.clear()


def _materialize_batch(job: _BatchJob) -> SequenceBatch:
    """Decode one batch's games and pack them, in a worker or in place."""

    path = Path(job.path)
    decoded: dict[int, tuple[PlyEncoding, ...]] = {}
    examples: list[SequenceExample] = []
    for row_index, start_ply, length in job.entries:
        plies = decoded.get(row_index)
        if plies is None:
            plies = encode_game(_game_from_row(job.rows[row_index], path))
            if len(plies) != job.lengths[row_index]:
                raise DataLoadingError(
                    f"{path} game {job.rows[row_index][NormalizedColumn.GAME_ID]} "
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
    return collate_sequences(examples, legal_actions=job.legal_actions)


@dataclass(frozen=True)
class _ScannedGroup:
    """One row group's eligible games, before the subsample cut is known."""

    shard: int
    row_group: int
    positions: array[int]
    game_ids: array[int]
    lengths: array[int]


def _scan_row_group(
    reader: Any,
    *,
    shard_index: int,
    row_group: int,
    split: str,
    selection: SelectionConfig,
    excluded: dict[str, int],
) -> _ScannedGroup:
    """Read one row group's index columns and apply the selection filters."""

    table = read_normalized_row_group(reader, row_group, _INDEX_COLUMNS)
    columns = {
        column.value: row_group_column(table, column.value) for column in _INDEX_COLUMNS
    }
    positions: array[int] = array("I")
    game_ids: array[int] = array("Q")
    lengths: array[int] = array("i")
    splits = columns[NormalizedColumn.SPLIT]
    ply_counts = columns[NormalizedColumn.PLY_COUNT]
    terminal = columns[NormalizedColumn.TERMINAL_ACTION_STATUS]
    for position in range(len(splits)):
        if splits[position] != split:
            continue
        row = {column: values[position] for column, values in columns.items()}
        reason = _exclusion_reason(row, selection)
        if reason is not None:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        appended = terminal[position] == TerminalActionStatus.APPENDED
        positions.append(position)
        game_ids.append(columns[NormalizedColumn.GAME_ID][position])
        lengths.append(ply_counts[position] + (1 if appended else 0))
    return _ScannedGroup(
        shard=shard_index,
        row_group=row_group,
        positions=positions,
        game_ids=game_ids,
        lengths=lengths,
    )


def _kept_game_ids(eligible: array[int], selection: SelectionConfig) -> frozenset[int]:
    """Return the game ids a subsample keeps, ranked the same way everywhere."""

    if selection.fraction is None and selection.maximum_games is None:
        return frozenset(eligible)
    ordered = sorted(eligible, key=lambda game_id: _rank_key(selection.seed, game_id))
    kept = len(ordered)
    if selection.fraction is not None:
        kept = int(len(ordered) * selection.fraction)
    if selection.maximum_games is not None:
        kept = min(kept, selection.maximum_games)
    return frozenset(ordered[:kept])


def _selected_groups(
    scanned: Sequence[_ScannedGroup],
    kept: frozenset[int],
) -> Iterator[_RowGroupIndex]:
    """Drop the games a subsample excluded, and any row group left empty."""

    for group in scanned:
        positions: array[int] = array("I")
        game_ids: array[int] = array("Q")
        lengths: array[int] = array("i")
        for slot, game_id in enumerate(group.game_ids):
            if game_id not in kept:
                continue
            positions.append(group.positions[slot])
            game_ids.append(game_id)
            lengths.append(group.lengths[slot])
        if not game_ids:
            continue
        yield _RowGroupIndex(
            shard=group.shard,
            row_group=group.row_group,
            positions=positions,
            game_ids=game_ids,
            lengths=lengths,
        )


def _plan_epoch(
    index: ShardedSequenceIndex,
    config: SequenceLoaderConfig,
    streaming: StreamingLoaderConfig,
    epoch: int,
) -> Iterator[tuple[_Example, ...]]:
    """Yield one epoch's batches as plans, reading nothing.

    Generated rather than materialized so a corpus-scale epoch costs a cursor
    instead of a list of every example in it, and so a resumed run can skip
    forward through the plan at the price of arithmetic.
    """

    for window in _plan_windows(index, config, streaming, epoch):
        yield from _window_batches(window, config, epoch)


def _plan_windows(
    index: ShardedSequenceIndex,
    config: SequenceLoaderConfig,
    streaming: StreamingLoaderConfig,
    epoch: int,
) -> Iterator[tuple[_Example, ...]]:
    """Yield the epoch's planning windows in deterministic order."""

    order = list(range(len(index.groups)))
    if config.shuffle:
        order.sort(
            key=lambda position: _group_key(config.seed, epoch, index.groups[position])
        )
    window = streaming.planning_window_examples
    for group_index in order:
        group = index.groups[group_index]
        slots = list(range(len(group)))
        if config.shuffle:
            slots.sort(key=partial(_game_key, config.seed, epoch, group))
        examples = [
            example
            for slot in slots
            for example in _examples_for(group_index, slot, group, index.chunk_length)
        ]
        for start in range(0, len(examples), window):
            yield tuple(examples[start : start + window])


def _examples_for(
    group_index: int,
    slot: int,
    group: _RowGroupIndex,
    chunk_length: int | None,
) -> Iterator[_Example]:
    """Expand one game into the full sequence or the chunks it is cut into."""

    length = group.lengths[slot]
    if chunk_length is None:
        yield _Example(group=group_index, slot=slot, start_ply=0, length=length)
        return
    if type(chunk_length) is not int or chunk_length < 1:
        raise ValueError("chunk_length must be a positive integer or None")
    for start in range(0, length, chunk_length):
        yield _Example(
            group=group_index,
            slot=slot,
            start_ply=start,
            length=min(chunk_length, length - start),
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
        batches.sort(key=lambda batch: _batch_key(config.seed, epoch, batch))
    yield from batches


def _group_key(seed: str, epoch: int, group: _RowGroupIndex) -> bytes:
    return sha256(
        f"{seed}\0{epoch}\0group\0{group.shard}\0{group.row_group}".encode()
    ).digest()


def _game_key(seed: str, epoch: int, group: _RowGroupIndex, slot: int) -> bytes:
    return sha256(
        f"{seed}\0{epoch}\0{group.shard}\0{group.game_ids[slot]}".encode()
    ).digest()


def _batch_key(seed: str, epoch: int, batch: Sequence[_Example]) -> bytes:
    members = ",".join(
        f"{example.group}:{example.slot}:{example.start_ply}" for example in batch
    )
    return sha256(f"{seed}\0{epoch}\0batch\0{members}".encode()).digest()


__all__ = [
    "STREAMING_IDENTITY_VERSION",
    "STREAMING_LOADER_NAME",
    "ShardedSequenceIndex",
    "StreamingSequenceDataLoader",
    "build_sharded_index",
]
