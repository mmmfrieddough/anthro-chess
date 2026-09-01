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

Opening a corpus therefore reads nothing, as long as the selection rejects
nothing: preparation counted every split when it wrote each shard, and the
check that admitted the corpus carried those counts here. A selection that
filters has to look, and that is the one pass here whose cost follows corpus
size.

A resumed run replays the plan to its saved cursor, which re-derives the row
groups it passes over rather than decoding any game in them.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any

from anthro_chess.data.artifacts import (
    DataLoadingError,
    ShardIdentity,
    materialize_rows,
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
    collate_packed,
    collate_sequences,
    loader_configuration_sha256,
    packed_cuts,
    require_resolved_snapshot,
    subsample_threshold,
    within_subsample,
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

#: What every pass over a row group reads, because both of them start by
#: dropping the rows of other splits.
_SPLIT_COLUMNS = (NormalizedColumn.SPLIT,)
#: Joined only when a subsample cuts the rank space, because a game's place in
#: it is the one thing that follows from its identity rather than its position.
_IDENTITY_COLUMNS = (
    NormalizedColumn.SOURCE_ID,
    NormalizedColumn.SOURCE_GAME_KEY,
)
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

    group: _RowGroup
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
    #: Set when the entries are laid end to end in one row of this width
    #: rather than given a padded row each.
    packed_width: int | None


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
    subsample_threshold: int | None
    resolution: SelectionResolution
    identity_sha256: str


def resolve_sharded_selection(
    shards: Sequence[ShardIdentity],
    *,
    split: str,
    selection: SelectionConfig,
    chunk_length: int | None = None,
    manifest_sha256: str,
    marked_digests: frozenset[int] | None = None,
) -> ShardedSelection:
    """Open one split of a prepared corpus without reading any game."""

    require_resolved_snapshot(selection, marked_digests)
    if not shards:
        raise DataLoadingError("at least one normalized shard is required")
    split_games = sum(shard.split_counts.get(split, 0) for shard in shards)
    if not split_games:
        raise DataLoadingError(f"the corpus holds no {split} games")
    if selection.maximum_games is not None:
        raise DataLoadingError(
            "the shard-backed loader cannot hold a selection to a game count: "
            "delivering an exact one means ranking every candidate, which is a "
            "digest per game of the whole corpus. Use fraction, which cuts the "
            "same rank space and needs no count"
        )
    row_groups = _enumerate_row_groups(shards)
    filtered = _filters_rows(selection, marked_digests)
    threshold = subsample_threshold(selection)
    resolution = SelectionResolution(
        spec=selection.model_dump(mode="json"),
        # Counted only where the manifest already counted it. A filter is
        # applied as the epoch reaches each row and a subsample cuts a rank
        # space rather than a list, so neither knows its own size without
        # reading every row of the split for a number nothing computes from.
        eligible_games=None if filtered else split_games,
        selected_games=None if filtered or threshold is not None else split_games,
        excluded_games=None if filtered else {},
    )
    identity = {
        "version": STREAMING_IDENTITY_VERSION,
        "loader": STREAMING_LOADER_NAME,
        "split": split,
        "split_games": split_games,
        "chunk_length": chunk_length,
        "manifest_sha256": manifest_sha256,
        "shards": [
            {"name": shard.path.name, "sha256": shard.sha256} for shard in shards
        ],
        "selection": resolution.as_identity_record(),
        "subsample_threshold": threshold,
        "marked_accounts": _marked_accounts_sha256(marked_digests),
    }
    logger.info(
        "Opened the %s split of %s shard(s), %s game(s) before selection",
        split,
        len(shards),
        split_games,
    )
    return ShardedSelection(
        shards=tuple(shards),
        row_groups=row_groups,
        split=split,
        chunk_length=chunk_length,
        selection=selection,
        marked_digests=marked_digests,
        subsample_threshold=threshold,
        resolution=resolution,
        identity_sha256=sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def _enumerate_row_groups(shards: Sequence[ShardIdentity]) -> tuple[_RowGroup, ...]:
    """Return every row group of every shard, in shard order."""

    return tuple(
        _RowGroup(shard=shard_index, row_group=row_group)
        for shard_index, shard in enumerate(shards)
        for row_group in range(shard.row_groups)
    )


def _marked_accounts_sha256(marked_digests: frozenset[int] | None) -> str | None:
    """Return what a snapshot rejected, as something a resume can compare.

    The resolved record counts what it removed and does not say who, and two
    snapshots rejecting the same number of games are a set of games apart. The
    snapshot's own path cannot stand in either, because the same file sits at
    different paths on two machines. This is bounded by the account census
    rather than by the corpus, so it is affordable where a digest per game is
    not.
    """

    if marked_digests is None:
        return None
    digest = sha256()
    for account in sorted(marked_digests):
        digest.update(account.to_bytes(8, "big"))
    return digest.hexdigest()


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


def _scan_row_group(
    reader: Any,
    group: _RowGroup,
    *,
    split: str,
    selection: SelectionConfig,
    marked_digests: frozenset[int] | None,
    threshold: int | None = None,
    lengths: bool = False,
) -> Iterator[tuple[int, str | None, int]]:
    """Yield each row of one row group that is in the split, and its verdict.

    Both passes over a row group start the same way, and only one of them ever
    wants a game's length, so the projection follows what the caller asked for
    rather than the union of the two.
    """

    filtered = _filters_rows(selection, marked_digests)
    columns = (
        _SPLIT_COLUMNS
        + (_LENGTH_COLUMNS if lengths else ())
        + (_FILTER_COLUMNS if filtered else ())
        + (_MARKED_COLUMNS if marked_digests else ())
        + (_IDENTITY_COLUMNS if threshold is not None else ())
    )
    table = read_normalized_row_group(reader, group.row_group, columns)
    values = {column.value: row_group_column(table, column.value) for column in columns}
    splits = values[NormalizedColumn.SPLIT]
    ply_counts = values[NormalizedColumn.PLY_COUNT] if lengths else ()
    terminal = values[NormalizedColumn.TERMINAL_ACTION_STATUS] if lengths else ()
    for position in range(len(splits)):
        if splits[position] != split:
            continue
        reason = None
        if filtered or threshold is not None:
            row = {
                column: column_values[position]
                for column, column_values in values.items()
            }
            if filtered:
                reason = _exclusion_reason(row, selection, marked_digests)
            if (
                reason is None
                and threshold is not None
                and not within_subsample(row_game_id(row), selection.seed, threshold)
            ):
                continue
        if not lengths or reason is not None:
            yield position, reason, 0
            continue
        appended = terminal[position] == TerminalActionStatus.APPENDED
        yield position, None, ply_counts[position] + (1 if appended else 0)


def shard_loader_configuration_sha256(
    config: SequenceLoaderConfig,
    streaming: StreamingLoaderConfig,
) -> str:
    """Return the loader identity a shard-backed run records.

    Wider than :func:`loader_configuration_sha256`, which describes the
    selection alone. Only the planning window joins it, because only the window
    decides which examples share a batch; worker count and prefetch depth are
    what the machine can afford, and a run resumed on another machine should be
    free to afford differently.
    """

    return sha256(
        json.dumps(
            {
                "loader": loader_configuration_sha256(config),
                "planning_window_examples": streaming.planning_window_examples,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


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
        self.configuration_sha256 = shard_loader_configuration_sha256(config, streaming)
        self._pool: ProcessPoolExecutor | None = None
        self._reader: Any | None = None
        self._reader_shard: int | None = None
        self._table: Any | None = None
        self._table_group: _RowGroup | None = None
        self._inflight: deque[tuple[int, Future[SequenceBatch]]] = deque()
        self._epoch = 0
        self._position = 0
        self._group_index = 0
        self._group_position = 0
        self._plan = self._plan_epoch(self._epoch)

    def __iter__(self) -> StreamingSequenceDataLoader:
        return self

    def __next__(self) -> SequenceBatch:
        self._fill()
        if not self._inflight:
            raise StopIteration
        ordinal, pending = self._inflight.popleft()
        batch = pending.result()
        self._position += 1
        # Tracked on the way out rather than on the way in, because the cursor
        # has to name the batch a resumed run reads next and the plan runs
        # ahead of that by the prefetch depth.
        self._group_position = (
            self._group_position + 1 if ordinal == self._group_index else 1
        )
        self._group_index = ordinal
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
            group_index=self._group_index,
            group_position=self._group_position,
        )

    def load_state(self, state: SequenceLoaderState | Mapping[str, object]) -> None:
        """Restore a compatible saved cursor and deterministic epoch order.

        The saved row group is reached by arithmetic over the epoch order and
        only that one is planned, so a cursor deep into a corpus-scale epoch
        costs one projected read rather than one per row group before it.
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
        self._plan = self._plan_epoch(parsed.epoch, start=parsed.group_index)
        self._group_index = parsed.group_index
        for _ in range(parsed.group_position):
            if next(self._plan, None) is None:
                raise DataLoadingError(
                    "loader state position is outside the epoch plan"
                )
        self._position = parsed.position
        self._group_position = parsed.group_position

    def start_epoch(self, epoch: int) -> None:
        """Start a deterministic epoch from its first example."""

        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self._drain()
        self._epoch = epoch
        self._position = 0
        self._group_index = 0
        self._group_position = 0
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

    def _plan_epoch(
        self,
        epoch: int,
        start: int = 0,
    ) -> Iterator[tuple[int, _PlannedBatch]]:
        """Yield one epoch's batches as plans, with their place in the order.

        Generated rather than materialized so a corpus-scale epoch costs a
        cursor instead of a list of every example in it. ``start`` skips whole
        row groups by arithmetic, which is what lets a resumed run reach a
        cursor deep into an epoch without planning what came before it.
        """

        order = list(self.corpus.row_groups)
        if self.config.shuffle:
            order.sort(key=partial(_group_key, self.config.seed, epoch))
        for ordinal in range(start, len(order)):
            for planned in self._plan_row_group(order[ordinal], epoch):
                yield ordinal, planned

    def _plan_row_group(
        self,
        group: _RowGroup,
        epoch: int,
    ) -> Iterator[_PlannedBatch]:
        """Yield the batches one row group contributes to an epoch."""

        rows = self._eligible_rows(group)
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
                yield _PlannedBatch(group=group, examples=batch)

    def _eligible_rows(self, group: _RowGroup) -> list[tuple[int, int]]:
        """Return which rows of one row group the selection keeps, and how long.

        The answer is wanted once, in the order the epoch visits row groups,
        and it is derived from columns cheap enough to project.
        """

        return [
            (position, length)
            for position, reason, length in _scan_row_group(
                self._shard_reader(group.shard),
                group,
                split=self.corpus.split,
                selection=self.corpus.selection,
                marked_digests=self.corpus.marked_digests,
                threshold=self.corpus.subsample_threshold,
                lengths=True,
            )
            if reason is None
        ]

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
            ordinal, batch = planned
            self._inflight.append((ordinal, self._submit(batch)))

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
        table = self._row_group_table(planned.group)
        game_lengths = {
            example.position: example.game_length for example in planned.examples
        }
        rows = sorted(game_lengths)
        row_index = {position: index for index, position in enumerate(rows)}
        return _BatchJob(
            shard=planned.group.shard,
            path=str(self.corpus.shards[planned.group.shard].path),
            row_table=take_rows(table, rows),
            lengths=tuple(game_lengths[position] for position in rows),
            entries=tuple(
                (row_index[example.position], example.start_ply, example.length)
                for example in planned.examples
            ),
            legal_actions=self.legal_actions,
            packed_width=self.config.positions_per_batch,
        )

    def _shard_reader(self, shard: int) -> Any:
        if self._reader_shard != shard:
            self._reader = open_normalized_shard(self.corpus.shards[shard].path)
            self._reader_shard = shard
        return self._reader

    def _row_group_table(self, group: _RowGroup) -> Any:
        if self._table_group == group:
            return self._table
        reader = self._shard_reader(group.shard)
        # One row group at a time, and the previous one released before the
        # next is read. This is the loader's largest resident structure and the
        # only one preparation's shard sizing decides.
        self._table = None
        self._table = read_normalized_row_group(
            reader, group.row_group, _LOADER_COLUMNS
        )
        self._table_group = group
        return self._table

    def _drain(self) -> None:
        for _, pending in self._inflight:
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
    if job.packed_width is None:
        return collate_sequences(examples)
    return collate_packed(examples, job.packed_width)


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
    """Cut one window into the batches it contributes, in the configured shape."""

    width = config.positions_per_batch
    batches = (
        _bucketed_window_batches(window, config)
        if width is None
        else _packed_window_batches(window, width, drop_last=config.drop_last)
    )
    if config.shuffle:
        batches.sort(key=partial(_batch_key, config.seed, epoch))
    yield from batches


def _bucketed_window_batches(
    window: Sequence[_Example],
    config: SequenceLoaderConfig,
) -> list[tuple[_Example, ...]]:
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
    return batches


def _packed_window_batches(
    window: Sequence[_Example],
    width: int,
    *,
    drop_last: bool,
) -> list[tuple[_Example, ...]]:
    """Lay one window's examples end to end and cut fixed-width batches out.

    A window is where a cut game's two halves can still meet, so the tail of a
    window is the only place a batch comes up short.
    """

    return [
        tuple(
            replace(example, start_ply=example.start_ply + start, length=taken)
            for example, start, taken in cuts
        )
        for cuts in packed_cuts(
            [(example, example.length) for example in window],
            width,
            drop_last=drop_last,
        )
    ]


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
