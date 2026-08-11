"""Deterministic full-game and fixed-length sequence loading."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from heapq import nsmallest
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, overload

from anthro_chess.data.artifacts import (
    DataLoadingError,
    read_normalized_rows,
    sorted_game_ids_sha256,
)
from anthro_chess.data.config import SelectionConfig, SequenceLoaderConfig
from anthro_chess.data.encoding import (
    BOARD_SQUARE_COUNT,
    GameEncodingInput,
    PlyEncoding,
    en_passant_token,
    encode_game,
    previous_action_token,
)
from anthro_chess.data.schema import (
    SCHEMA_VERSION,
    NormalizedColumn,
    clock_remaining_ms,
    row_game_id,
)
from anthro_chess.data.speed import speed_from_clock_ms

if TYPE_CHECKING:
    import numpy as np

LOADER_STATE_VERSION = 4
SELECTION_SPEC_VERSION = 1
logger = logging.getLogger(__name__)
#: The columns a selection filters on. Reading only these keeps the pass that
#: resolves which games to keep far cheaper than the pass that encodes them.
_SELECTION_COLUMNS = (
    NormalizedColumn.SOURCE_ID,
    NormalizedColumn.SOURCE_GAME_KEY,
    NormalizedColumn.WHITE_NORMALIZED_RATING,
    NormalizedColumn.BLACK_NORMALIZED_RATING,
    NormalizedColumn.TIME_INITIAL_MS,
    NormalizedColumn.TIME_INCREMENT_MS,
    NormalizedColumn.SPLIT,
)
_LOADER_COLUMNS = (
    NormalizedColumn.SCHEMA_VERSION,
    NormalizedColumn.SOURCE_ID,
    NormalizedColumn.SOURCE_GAME_KEY,
    NormalizedColumn.RULESET,
    NormalizedColumn.INITIAL_POSITION,
    NormalizedColumn.ACTION_IDS,
    NormalizedColumn.WHITE_NORMALIZED_RATING,
    NormalizedColumn.BLACK_NORMALIZED_RATING,
    NormalizedColumn.TIME_INITIAL_MS,
    NormalizedColumn.TIME_INCREMENT_MS,
    NormalizedColumn.CLOCK_REMAINING_DELTA_MS,
    NormalizedColumn.SPLIT,
)

LegalActionTensor: TypeAlias = tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class SequenceExample:
    """One full game or contiguous fixed-length game chunk."""

    shard_index: int
    game_id: int
    start_ply: int
    plies: tuple[PlyEncoding, ...]

    def __post_init__(self) -> None:
        if not self.plies:
            raise ValueError("a sequence example needs at least one ply")
        if self.plies[0].game_id != self.game_id:
            raise ValueError("sequence game_id must match its encoded plies")
        if self.plies[0].ply_index != self.start_ply:
            raise ValueError("sequence start_ply must match its first encoded ply")


@dataclass(frozen=True)
class OptionalIntBatch:
    """Packed nullable integers with an explicit presence mask."""

    values: np.ndarray
    present: np.ndarray


@dataclass(frozen=True)
class SequenceInputs:
    """Framework-neutral numeric model inputs shaped batch by sequence.

    ``en_passant_token`` and ``previous_action_token`` are the embedding rows
    :mod:`anthro_chess.data.encoding` assigns, absence included, rather than a
    value beside a presence flag.
    """

    piece_ids: np.ndarray
    side_to_move: np.ndarray
    castling_rights: np.ndarray
    en_passant_token: np.ndarray
    halfmove_clock: np.ndarray
    fullmove_number: np.ndarray
    previous_action_token: np.ndarray
    target_rating: OptionalIntBatch
    time_initial_ms: OptionalIntBatch
    time_increment_ms: OptionalIntBatch
    player_clock_ms: OptionalIntBatch
    opponent_clock_ms: OptionalIntBatch


@dataclass(frozen=True)
class SequenceBatch:
    """Padded causal batch with aligned targets, masks, and slice metadata.

    Every field is a contiguous array narrow enough to hold what it carries,
    which is what lets a consumer wrap it rather than walk it: the tensor
    boundary is a buffer view and a device copy, and a batch crossing a
    process boundary travels as buffers rather than as objects.

    Boolean attention values are ``True`` where a timestep is present.

    Causality is a property of the model rather than of the batch, so no
    sequence-by-sequence mask travels here. Padding is right-aligned, which is
    what makes that safe: a real query attends only to earlier timesteps, and
    every timestep earlier than a real one is itself real.

    ``legal_action_ids`` is absent when nothing downstream will read it. Policy
    scoring and the batch's own legality check are its only consumers and
    training is neither, so a training batch carries none and the encoding that
    fed it never built one. It is a ragged Python structure because each
    timestep enables a different number of actions.
    """

    inputs: SequenceInputs
    action_targets: np.ndarray
    action_loss_mask: np.ndarray
    attention_mask: np.ndarray
    legal_action_ids: LegalActionTensor | None
    game_ids: np.ndarray
    ply_indices: np.ndarray
    chunk_start_plies: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        """Return the number of sequences in the batch."""

        return int(self.action_targets.shape[0])

    @property
    def sequence_length(self) -> int:
        """Return the padded sequence length."""

        return int(self.attention_mask.shape[1])


@dataclass(frozen=True)
class SelectionResolution:
    """Which games a load-time selection kept, and how to reproduce that set.

    ``game_ids_sha256`` is what makes the record reproducible rather than
    merely descriptive: a later run over the same corpus with the same spec
    resolves the same digest, and a corpus that has since grown resolves a
    different one.

    It is also all that survives of the ids themselves. Confirming that a later
    run reproduced the same games is the only thing anything downstream reads
    them for, the digest answers exactly that, and a corpus-scale selection's
    ids are tens of gigabytes of Python object held for the length of a run.
    """

    spec: dict[str, object]
    eligible_games: int
    selected_games: int
    game_ids_sha256: str
    excluded_games: dict[str, int]

    def as_record(self) -> dict[str, object]:
        """Return the resolved-selection record stored in run artifacts."""

        return {
            "version": SELECTION_SPEC_VERSION,
            "spec": dict(sorted(self.spec.items())),
            "eligible_games": self.eligible_games,
            "selected_games": self.selected_games,
            "excluded_games": dict(sorted(self.excluded_games.items())),
            "game_ids_sha256": self.game_ids_sha256,
        }


@dataclass(frozen=True)
class SequenceLoaderState:
    """Serializable exact next-batch cursor for one deterministic loader epoch."""

    version: int
    dataset_sha256: str
    configuration_sha256: str
    epoch: int
    position: int

    def __post_init__(self) -> None:
        if type(self.version) is not int:
            raise DataLoadingError("loader state version must be an integer")
        if not isinstance(self.dataset_sha256, str) or not self.dataset_sha256:
            raise DataLoadingError("loader state dataset identity must be a string")
        if (
            not isinstance(self.configuration_sha256, str)
            or not self.configuration_sha256
        ):
            raise DataLoadingError(
                "loader state configuration identity must be a string"
            )
        if type(self.epoch) is not int or self.epoch < 0:
            raise DataLoadingError("loader state epoch must be nonnegative")
        if type(self.position) is not int or self.position < 0:
            raise DataLoadingError("loader state position must be nonnegative")

    def as_record(self) -> dict[str, object]:
        """Return the JSON-serializable checkpoint representation."""

        return {
            "version": self.version,
            "dataset_sha256": self.dataset_sha256,
            "configuration_sha256": self.configuration_sha256,
            "epoch": self.epoch,
            "position": self.position,
        }


class SequenceBatchSource(Iterator[SequenceBatch], ABC):
    """What a training run and its checkpoints require of any loader.

    Two loaders implement it and a run picks one by configuration. The eager
    loader reconstructs every selected ply before the first batch, which suits
    fixtures and bounded proof slices. The shard-backed loader in
    ``anthro_chess.data.streaming`` decodes a batch at a time, which is the
    only way a corpus-scale selection starts at all.

    They differ in what they hold and not in what they promise: a deterministic
    epoch order, an exact next-batch cursor, and identities that tell a resumed
    run whether it is continuing the same work.
    """

    config: SequenceLoaderConfig
    configuration_sha256: str

    @property
    @abstractmethod
    def identity_sha256(self) -> str:
        """Return what this source is reading, for checkpoint compatibility."""

    @property
    @abstractmethod
    def resolution(self) -> SelectionResolution:
        """Return which games the configured selection kept."""

    @abstractmethod
    def state(self) -> SequenceLoaderState:
        """Return the exact next-batch cursor for checkpointing."""

    @abstractmethod
    def load_state(self, state: SequenceLoaderState | Mapping[str, object]) -> None:
        """Restore a compatible saved cursor and deterministic epoch order."""

    @abstractmethod
    def start_epoch(self, epoch: int) -> None:
        """Start a deterministic epoch from its first example."""

    def close(self) -> None:
        """Release whatever the source holds open. Idempotent."""


class SequenceDataset(Sequence[SequenceExample]):
    """In-memory sequence view over one or more normalized Parquet shards."""

    def __init__(
        self,
        examples: Sequence[SequenceExample],
        *,
        identity_sha256: str,
        split: str = "train",
        chunk_length: int | None = None,
        selection: SelectionConfig | None = None,
        resolution: SelectionResolution | None = None,
    ) -> None:
        if not examples:
            raise DataLoadingError("no normalized games matched the loader selection")
        self._examples = tuple(examples)
        self.identity_sha256 = identity_sha256
        self.split = split
        self.chunk_length = chunk_length
        self.selection = SelectionConfig() if selection is None else selection
        # A dataset built directly from examples never ran a resolution pass,
        # so it reports the games it holds rather than claiming it filtered.
        self.resolution = (
            resolution
            if resolution is not None
            else _resolution_for_examples(self._examples, self.selection)
        )

    def __len__(self) -> int:
        return len(self._examples)

    @overload
    def __getitem__(self, index: int) -> SequenceExample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[SequenceExample]: ...

    def __getitem__(
        self, index: int | slice
    ) -> SequenceExample | Sequence[SequenceExample]:
        return self._examples[index]

    @classmethod
    def from_parquet(
        cls,
        paths: str | Path | Sequence[str | Path],
        *,
        split: str,
        chunk_length: int | None = None,
        selection: SelectionConfig | None = None,
        legal_actions: bool = True,
    ) -> SequenceDataset:
        """Load, validate, encode, and optionally chunk normalized games.

        Resolving the selection first means only the selected games are
        encoded, which is where nearly all of the load cost is.
        """

        selection = SelectionConfig() if selection is None else selection
        normalized_paths = _normalize_paths(paths)
        logger.info(
            "Loading normalized %s split from %s shard(s)",
            split,
            len(normalized_paths),
        )
        resolution, selected = _resolve_selection(
            normalized_paths,
            split=split,
            selection=selection,
        )
        examples: list[SequenceExample] = []
        identity_records: list[dict[str, object]] = []
        for shard_index, path in enumerate(normalized_paths):
            for row in read_normalized_rows(path, _LOADER_COLUMNS):
                if row[NormalizedColumn.SPLIT] != split:
                    continue
                game_id = row_game_id(row)
                if game_id not in selected:
                    continue
                game = _game_from_row(row, path, game_id)
                plies = encode_game(game, legal_actions=legal_actions)
                chunks = _chunk_plies(plies, chunk_length)
                for chunk in chunks:
                    examples.append(
                        SequenceExample(
                            shard_index=shard_index,
                            game_id=game.game_id,
                            start_ply=chunk[0].ply_index,
                            plies=chunk,
                        )
                    )
                identity_records.append(
                    {
                        "shard": shard_index,
                        "game_id": game.game_id,
                        "ply_count": len(plies),
                        "content_sha256": _normalized_game_sha256(row),
                    }
                )

        examples.sort(key=lambda item: (item.shard_index, item.game_id, item.start_ply))
        identity = {
            "version": LOADER_STATE_VERSION,
            "split": split,
            "chunk_length": chunk_length,
            "games": identity_records,
        }
        identity_sha256 = sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        logger.info(
            "Loaded %s sequence example(s) from %s of %s eligible %s game(s)",
            len(examples),
            resolution.selected_games,
            resolution.eligible_games,
            split,
        )
        return cls(
            examples,
            identity_sha256=identity_sha256,
            split=split,
            chunk_length=chunk_length,
            selection=selection,
            resolution=resolution,
        )


class SequenceDataLoader(SequenceBatchSource):
    """Stateful deterministic batch iterator with explicit resume state."""

    def __init__(
        self,
        dataset: SequenceDataset,
        config: SequenceLoaderConfig,
    ) -> None:
        if config.split != dataset.split:
            raise DataLoadingError("loader split does not match the sequence dataset")
        if config.chunk_length != dataset.chunk_length:
            raise DataLoadingError(
                "loader chunk_length does not match the sequence dataset"
            )
        if config.selection != dataset.selection:
            raise DataLoadingError(
                "loader selection does not match the sequence dataset"
            )
        self.dataset = dataset
        self.config = config
        self.configuration_sha256 = sha256(
            json.dumps(
                config.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self._epoch = 0
        self._position = 0
        self._batches = self._batches_for_epoch(self._epoch)

    def __iter__(self) -> SequenceDataLoader:
        return self

    def __next__(self) -> SequenceBatch:
        if self._position >= len(self._batches):
            raise StopIteration
        indices = self._batches[self._position]
        self._position += 1
        return collate_sequences(tuple(self.dataset[index] for index in indices))

    @property
    def identity_sha256(self) -> str:
        """Return the decoded-content identity of the retained examples."""

        return self.dataset.identity_sha256

    @property
    def resolution(self) -> SelectionResolution:
        """Return which games the configured selection kept."""

        return self.dataset.resolution

    def state(self) -> SequenceLoaderState:
        """Return the exact next-batch cursor for checkpointing."""

        return SequenceLoaderState(
            version=LOADER_STATE_VERSION,
            dataset_sha256=self.dataset.identity_sha256,
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
        if parsed.dataset_sha256 != self.dataset.identity_sha256:
            raise DataLoadingError("loader state belongs to different sequence data")
        if parsed.configuration_sha256 != self.configuration_sha256:
            raise DataLoadingError("loader state uses different loader configuration")
        batches = self._batches_for_epoch(parsed.epoch)
        if not 0 <= parsed.position <= len(batches):
            raise DataLoadingError("loader state position is outside the epoch plan")
        self._epoch = parsed.epoch
        self._position = parsed.position
        self._batches = batches

    @classmethod
    def from_parquet(
        cls,
        paths: str | Path | Sequence[str | Path],
        config: SequenceLoaderConfig,
        *,
        legal_actions: bool = True,
    ) -> SequenceDataLoader:
        """Build a configured loader directly from normalized Parquet shards.

        ``legal_actions`` reaches the encoding rather than the collation.
        Deliberately outside the configuration digest: which games this loader
        holds is unchanged either way, so a run resumed against the same
        declared loader is reading the same data.
        """

        return cls(
            SequenceDataset.from_parquet(
                paths,
                split=config.split,
                chunk_length=config.chunk_length,
                selection=config.selection,
                legal_actions=legal_actions,
            ),
            config,
        )

    def start_epoch(self, epoch: int) -> None:
        """Start a deterministic epoch from its first example."""

        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self._epoch = epoch
        self._position = 0
        self._batches = self._batches_for_epoch(epoch)

    def _batches_for_epoch(self, epoch: int) -> tuple[tuple[int, ...], ...]:
        buckets: dict[int, list[int]] = {}
        for index, example in enumerate(self.dataset):
            bucket = _length_bucket(len(example.plies), self.config.length_bucket_width)
            buckets.setdefault(bucket, []).append(index)

        batches: list[tuple[int, ...]] = []
        for bucket in sorted(buckets):
            indices = buckets[bucket]
            if self.config.shuffle:
                indices.sort(
                    key=lambda index: _shuffle_key(
                        self.config.seed,
                        epoch,
                        self.dataset[index],
                    )
                )
            for start in range(0, len(indices), self.config.batch_size):
                batch = tuple(indices[start : start + self.config.batch_size])
                if self.config.drop_last and len(batch) < self.config.batch_size:
                    continue
                batches.append(batch)

        if self.config.shuffle:
            batches.sort(
                key=lambda batch: _batch_shuffle_key(
                    self.config.seed,
                    epoch,
                    batch,
                )
            )
        return tuple(batches)


def collate_sequences(examples: Sequence[SequenceExample]) -> SequenceBatch:
    """Pad and pack sequence examples into arrays, inventing no targets.

    Every column is written into a prefilled array of its own width, so a
    padded timestep costs a memset rather than a Python object and the result
    is what a tensor can wrap. Each width is what that column's values need and
    no more, because this is what a worker sends and what crosses to a device.
    NumPy raises on a value too large for its column rather than wrapping it,
    so a corpus that outgrew one fails at the batch that first carried it.

    Each timestep's legal action ids are packed when the encoding built them.
    """

    # Deferred, the way `anthro_chess.data.artifacts` defers pyarrow: importing
    # an array library at module scope would take `anthro machine` — a
    # diagnostic that has to run wherever the package is installed — down with
    # it on an install carrying no extras.
    import numpy as np

    if not examples:
        raise ValueError("cannot collate an empty sequence collection")
    lengths = [len(example.plies) for example in examples]
    shape = (len(examples), max(lengths))

    def required(
        getter: Callable[[PlyEncoding], int],
        dtype: type[np.generic],
        *,
        padding: int = 0,
    ) -> np.ndarray:
        values = np.full(shape, padding, dtype=dtype)
        for index, example in enumerate(examples):
            values[index, : lengths[index]] = [getter(ply) for ply in example.plies]
        return values

    def optional(
        getter: Callable[[PlyEncoding], int | None],
        dtype: type[np.generic],
    ) -> OptionalIntBatch:
        values = np.zeros(shape, dtype=dtype)
        present = np.zeros(shape, dtype=np.bool_)
        for index, example in enumerate(examples):
            observed: list[int] = []
            seen: list[bool] = []
            for ply in example.plies:
                value = getter(ply)
                observed.append(0 if value is None else value)
                seen.append(value is not None)
            values[index, : lengths[index]] = observed
            present[index, : lengths[index]] = seen
        return OptionalIntBatch(values, present)

    piece_ids = np.zeros((*shape, BOARD_SQUARE_COUNT), dtype=np.uint8)
    attention_mask = np.zeros(shape, dtype=np.bool_)
    for index, example in enumerate(examples):
        length = lengths[index]
        piece_ids[index, :length] = np.frombuffer(
            b"".join([ply.board.piece_ids for ply in example.plies]),
            dtype=np.uint8,
        ).reshape(length, BOARD_SQUARE_COUNT)
        attention_mask[index, :length] = True

    # One ply decides for the batch. A collection is encoded by one caller, so
    # its plies agree; a hand-built one that does not raises from the accessor
    # below rather than being reconciled here.
    packs_legal_actions = examples[0].plies[0].legal_action_ids is not None
    inputs = SequenceInputs(
        piece_ids=piece_ids,
        side_to_move=required(lambda ply: ply.board.side_to_move, np.uint8),
        castling_rights=required(lambda ply: ply.board.castling_rights, np.uint8),
        # A padded timestep has no en-passant square and no previous action, so
        # each is filled with the row that names absence. Only the second one
        # differs from a zero fill, and both say so rather than one relying on
        # a coincidence.
        en_passant_token=required(
            lambda ply: en_passant_token(ply.board.en_passant_square),
            np.uint8,
            padding=en_passant_token(None),
        ),
        halfmove_clock=required(lambda ply: ply.board.halfmove_clock, np.int16),
        fullmove_number=required(lambda ply: ply.board.fullmove_number, np.int16),
        previous_action_token=required(
            lambda ply: previous_action_token(ply.previous_action_id),
            np.int16,
            padding=previous_action_token(None),
        ),
        target_rating=optional(lambda ply: ply.target_rating, np.int16),
        time_initial_ms=optional(lambda ply: ply.time_initial_ms, np.int32),
        time_increment_ms=optional(lambda ply: ply.time_increment_ms, np.int32),
        player_clock_ms=optional(lambda ply: ply.player_clock_ms, np.int32),
        opponent_clock_ms=optional(lambda ply: ply.opponent_clock_ms, np.int32),
    )
    return SequenceBatch(
        inputs=inputs,
        action_targets=required(lambda ply: ply.target_action_id, np.int16),
        action_loss_mask=attention_mask,
        attention_mask=attention_mask,
        legal_action_ids=(
            tuple(
                tuple(ply.enabled_actions() for ply in example.plies)
                + ((),) * (shape[1] - lengths[index])
                for index, example in enumerate(examples)
            )
            if packs_legal_actions
            else None
        ),
        game_ids=required(lambda ply: ply.game_id, np.uint64),
        ply_indices=required(lambda ply: ply.ply_index, np.int16),
        chunk_start_plies=tuple(example.start_ply for example in examples),
    )


def _normalize_paths(
    paths: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    normalized: tuple[Path, ...]
    if isinstance(paths, (str, Path)):
        normalized = (Path(paths),)
    else:
        normalized = tuple(Path(path) for path in paths)
    if not normalized:
        raise DataLoadingError("at least one normalized Parquet path is required")
    missing = tuple(path for path in normalized if not path.is_file())
    if missing:
        raise DataLoadingError(f"normalized Parquet file does not exist: {missing[0]}")
    return tuple(sorted(normalized, key=lambda path: str(path.resolve())))


def _resolve_selection(
    paths: Sequence[Path],
    *,
    split: str,
    selection: SelectionConfig,
) -> tuple[SelectionResolution, frozenset[int]]:
    """Decide which games in one split the configured selection keeps."""

    eligible: list[int] = []
    excluded: dict[str, int] = {}
    for path in paths:
        for row in read_normalized_rows(path, _SELECTION_COLUMNS):
            if row[NormalizedColumn.SPLIT] != split:
                continue
            reason = _exclusion_reason(row, selection)
            if reason is None:
                eligible.append(row_game_id(row))
            else:
                excluded[reason] = excluded.get(reason, 0) + 1

    kept = subsample_size(len(eligible), selection)
    selected = sorted(nsmallest(kept, eligible, key=partial(_rank_key, selection.seed)))
    return (
        SelectionResolution(
            spec=selection.model_dump(mode="json"),
            eligible_games=len(eligible),
            selected_games=len(selected),
            game_ids_sha256=sorted_game_ids_sha256(selected),
            excluded_games=excluded,
        ),
        frozenset(selected),
    )


def subsample_size(eligible_games: int, selection: SelectionConfig) -> int:
    """Return how many of the eligible games the configured dials keep."""

    kept = eligible_games
    if selection.fraction is not None:
        # Floor rather than round up: a fraction small enough to select nothing
        # should fail as an empty selection instead of quietly training on one
        # game.
        kept = int(eligible_games * selection.fraction)
    if selection.maximum_games is not None:
        kept = min(kept, selection.maximum_games)
    return kept


def _resolution_for_examples(
    examples: Sequence[SequenceExample],
    selection: SelectionConfig,
) -> SelectionResolution:
    game_ids = sorted({example.game_id for example in examples})
    return SelectionResolution(
        spec=selection.model_dump(mode="json"),
        eligible_games=len(game_ids),
        selected_games=len(game_ids),
        game_ids_sha256=sorted_game_ids_sha256(game_ids),
        excluded_games={},
    )


def _exclusion_reason(row: Mapping[str, Any], selection: SelectionConfig) -> str | None:
    time_initial = row[NormalizedColumn.TIME_INITIAL_MS]
    time_increment = row[NormalizedColumn.TIME_INCREMENT_MS]
    ratings = (
        row[NormalizedColumn.WHITE_NORMALIZED_RATING],
        row[NormalizedColumn.BLACK_NORMALIZED_RATING],
    )
    bounds_time_initial = (
        selection.minimum_time_initial_ms is not None
        or selection.maximum_time_initial_ms is not None
    )
    bounds_time_increment = (
        selection.minimum_time_increment_ms is not None
        or selection.maximum_time_increment_ms is not None
    )
    bounds_rating = (
        selection.minimum_rating is not None or selection.maximum_rating is not None
    )

    if (bounds_time_initial and time_initial is None) or (
        bounds_time_increment and time_increment is None
    ):
        return "missing_time_control"
    if (selection.require_ratings or bounds_rating) and any(
        rating is None for rating in ratings
    ):
        return "missing_ratings"

    if selection.speed is not None:
        speed = speed_from_clock_ms(time_initial, time_increment)
        if speed is None:
            return "missing_time_control"
        if speed != selection.speed:
            return "speed_mismatch"

    checks: tuple[tuple[str, int | None, int | None, int | None], ...] = (
        (
            "time_initial",
            time_initial,
            selection.minimum_time_initial_ms,
            selection.maximum_time_initial_ms,
        ),
        (
            "time_increment",
            time_increment,
            selection.minimum_time_increment_ms,
            selection.maximum_time_increment_ms,
        ),
    )
    for name, value, minimum, maximum in checks:
        if value is None:
            continue
        if minimum is not None and value < minimum:
            return f"below_minimum_{name}"
        if maximum is not None and value > maximum:
            return f"above_maximum_{name}"

    for rating in ratings:
        if rating is None:
            continue
        if selection.minimum_rating is not None and rating < selection.minimum_rating:
            return "below_minimum_rating"
        if selection.maximum_rating is not None and rating > selection.maximum_rating:
            return "above_maximum_rating"
    return None


def _rank_key(seed: str, game_id: int) -> bytes:
    """Rank uniformly by game id so a subsample stays representative."""

    return sha256(f"{seed}\0{game_id}".encode()).digest()


def _game_from_row(
    row: Mapping[str, Any],
    path: Path,
    game_id: int,
) -> GameEncodingInput:
    if row[NormalizedColumn.SCHEMA_VERSION] != SCHEMA_VERSION:
        raise DataLoadingError(
            f"{path} uses normalized schema version "
            f"{row[NormalizedColumn.SCHEMA_VERSION]}; "
            f"expected {SCHEMA_VERSION}"
        )
    try:
        action_ids = tuple(row[NormalizedColumn.ACTION_IDS])
        clocks = clock_remaining_ms(row)
        return GameEncodingInput(
            game_id=game_id,
            ruleset=row[NormalizedColumn.RULESET],
            initial_position=row[NormalizedColumn.INITIAL_POSITION],
            action_ids=action_ids,
            white_normalized_rating=row[NormalizedColumn.WHITE_NORMALIZED_RATING],
            black_normalized_rating=row[NormalizedColumn.BLACK_NORMALIZED_RATING],
            time_initial_ms=row[NormalizedColumn.TIME_INITIAL_MS],
            time_increment_ms=row[NormalizedColumn.TIME_INCREMENT_MS],
            clock_remaining_ms=clocks,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DataLoadingError(f"invalid normalized game in {path}: {error}") from error


def maximum_position_bound(maximum_game_plies: int, chunk_length: int | None) -> int:
    """Return the furthest ``MoveModelBatch.position_bound`` a corpus can reach.

    Both loaders cut non-overlapping chunks, so a game of ``L`` plies starts
    its last chunk at ``(L - 1) // C * C`` and has no chunk wider than
    ``min(C, L)``. Both grow with ``L``, so the longest game in a corpus bounds
    a batch that mixes its last chunk with any other game's widest one, and
    that worst case is what this returns. Unchunked, a row is a whole game and
    the reach is ``L``.
    """

    if chunk_length is None:
        return maximum_game_plies
    last_chunk_start = (maximum_game_plies - 1) // chunk_length * chunk_length
    return last_chunk_start + min(chunk_length, maximum_game_plies)


def _chunk_plies(
    plies: tuple[PlyEncoding, ...],
    chunk_length: int | None,
) -> tuple[tuple[PlyEncoding, ...], ...]:
    if chunk_length is None:
        return (plies,)
    if type(chunk_length) is not int or chunk_length < 1:
        raise ValueError("chunk_length must be a positive integer or None")
    return tuple(
        plies[start : start + chunk_length]
        for start in range(0, len(plies), chunk_length)
    )


def _normalized_game_sha256(row: Mapping[str, Any]) -> str:
    content = {column: row[column] for column in _LOADER_COLUMNS}
    return sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _shuffle_key(seed: str, epoch: int, example: SequenceExample) -> bytes:
    return sha256(
        (
            f"{seed}\0{epoch}\0{example.shard_index}\0"
            f"{example.game_id}\0{example.start_ply}"
        ).encode()
    ).digest()


def _batch_shuffle_key(seed: str, epoch: int, batch: tuple[int, ...]) -> bytes:
    indices = ",".join(str(index) for index in batch)
    return sha256(f"{seed}\0{epoch}\0batch\0{indices}".encode()).digest()


def _length_bucket(sequence_length: int, bucket_width: int | None) -> int:
    if bucket_width is None:
        return 0
    return (sequence_length - 1) // bucket_width


def _state_from_record(record: Mapping[str, object]) -> SequenceLoaderState:
    expected_keys = {
        "version",
        "dataset_sha256",
        "configuration_sha256",
        "epoch",
        "position",
    }
    if set(record) != expected_keys:
        raise DataLoadingError("loader state fields are incomplete or unknown")
    version = record["version"]
    dataset_sha256 = record["dataset_sha256"]
    configuration_sha256 = record["configuration_sha256"]
    epoch = record["epoch"]
    position = record["position"]
    if (
        type(version) is not int
        or not isinstance(dataset_sha256, str)
        or not isinstance(configuration_sha256, str)
        or type(epoch) is not int
        or type(position) is not int
    ):
        raise DataLoadingError("loader state fields have invalid types")
    return SequenceLoaderState(
        version,
        dataset_sha256,
        configuration_sha256,
        epoch,
        position,
    )
