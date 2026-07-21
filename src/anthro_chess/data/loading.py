"""Deterministic full-game and fixed-length sequence loading."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeAlias, cast, overload

import chess

from anthro_chess.data.config import SequenceLoaderConfig
from anthro_chess.data.encoding import (
    BOARD_SQUARE_COUNT,
    GameEncodingInput,
    PlyEncoding,
    encode_game,
)
from anthro_chess.data.schema import SCHEMA_VERSION, NormalizedColumn

LOADER_STATE_VERSION = 3
_LOADER_COLUMNS = (
    NormalizedColumn.SCHEMA_VERSION,
    NormalizedColumn.GAME_ID,
    NormalizedColumn.RULESET,
    NormalizedColumn.INITIAL_POSITION,
    NormalizedColumn.ACTION_IDS,
    NormalizedColumn.WHITE_NORMALIZED_RATING,
    NormalizedColumn.BLACK_NORMALIZED_RATING,
    NormalizedColumn.TIME_INITIAL_MS,
    NormalizedColumn.TIME_INCREMENT_MS,
    NormalizedColumn.CLOCK_REMAINING_MS,
    NormalizedColumn.SPLIT,
)

IntMatrix: TypeAlias = tuple[tuple[int, ...], ...]
BoolMatrix: TypeAlias = tuple[tuple[bool, ...], ...]
PieceTensor: TypeAlias = tuple[tuple[tuple[int, ...], ...], ...]
LegalActionTensor: TypeAlias = tuple[tuple[tuple[int, ...], ...], ...]


class DataLoadingError(ValueError):
    """Raised when normalized data or saved loader state is incompatible."""


@dataclass(frozen=True)
class SequenceExample:
    """One full game or contiguous fixed-length game chunk."""

    shard_index: int
    game_id: int
    controlled_color: int
    start_ply: int
    plies: tuple[PlyEncoding, ...]

    def __post_init__(self) -> None:
        if not self.plies:
            raise ValueError("a sequence example needs at least one ply")
        if self.plies[0].game_id != self.game_id:
            raise ValueError("sequence game_id must match its encoded plies")
        if self.plies[0].ply_index != self.start_ply:
            raise ValueError("sequence start_ply must match its first encoded ply")
        if any(ply.controlled_color != self.controlled_color for ply in self.plies):
            raise ValueError("sequence controlled_color must match its encoded plies")
        if not any(
            ply.board.side_to_move == self.controlled_color for ply in self.plies
        ):
            raise ValueError("sequence needs a controlled-player action target")


@dataclass(frozen=True)
class OptionalIntBatch:
    """Packed nullable integers with an explicit presence mask."""

    values: IntMatrix
    present: BoolMatrix


@dataclass(frozen=True)
class SequenceInputs:
    """Framework-neutral numeric model inputs shaped batch by sequence."""

    piece_ids: PieceTensor
    side_to_move: IntMatrix
    castling_rights: IntMatrix
    en_passant_square: OptionalIntBatch
    halfmove_clock: IntMatrix
    fullmove_number: IntMatrix
    previous_action_id: OptionalIntBatch
    target_rating: OptionalIntBatch
    controlled_color: IntMatrix
    time_initial_ms: OptionalIntBatch
    time_increment_ms: OptionalIntBatch
    player_clock_ms: OptionalIntBatch
    opponent_clock_ms: OptionalIntBatch


@dataclass(frozen=True)
class SequenceBatch:
    """Padded causal batch with aligned targets, masks, and slice metadata.

    Boolean attention values are ``True`` where a timestep is present, and
    causal values are ``True`` where a query may attend to a key.
    """

    inputs: SequenceInputs
    action_targets: IntMatrix
    action_loss_mask: BoolMatrix
    attention_mask: BoolMatrix
    causal_attention_mask: BoolMatrix
    legal_action_ids: LegalActionTensor
    game_ids: IntMatrix
    ply_indices: IntMatrix
    chunk_start_plies: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        """Return the number of sequences in the batch."""

        return len(self.action_targets)

    @property
    def sequence_length(self) -> int:
        """Return the padded sequence length."""

        return len(self.causal_attention_mask)


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


class SequenceDataset(Sequence[SequenceExample]):
    """In-memory sequence view over one or more normalized Parquet shards."""

    def __init__(
        self,
        examples: Sequence[SequenceExample],
        *,
        identity_sha256: str,
        split: str = "train",
        chunk_length: int | None = None,
    ) -> None:
        if not examples:
            raise DataLoadingError("no normalized games matched the loader selection")
        self._examples = tuple(examples)
        self.identity_sha256 = identity_sha256
        self.split = split
        self.chunk_length = chunk_length

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
    ) -> SequenceDataset:
        """Load, validate, encode, and optionally chunk normalized games."""

        normalized_paths = _normalize_paths(paths)
        examples: list[SequenceExample] = []
        identity_records: list[dict[str, object]] = []
        for shard_index, path in enumerate(normalized_paths):
            for row in _read_normalized_rows(path):
                if row[NormalizedColumn.SPLIT] != split:
                    continue
                game = _game_from_row(row, path)
                for controlled_color in (chess.WHITE, chess.BLACK):
                    color_token = 0 if controlled_color == chess.WHITE else 1
                    plies = encode_game(game, controlled_color=controlled_color)
                    chunks = _chunk_plies(plies, chunk_length)
                    for chunk in chunks:
                        if not any(
                            ply.board.side_to_move == color_token for ply in chunk
                        ):
                            continue
                        examples.append(
                            SequenceExample(
                                shard_index=shard_index,
                                game_id=game.game_id,
                                controlled_color=color_token,
                                start_ply=chunk[0].ply_index,
                                plies=chunk,
                            )
                        )
                    identity_records.append(
                        {
                            "shard": shard_index,
                            "game_id": game.game_id,
                            "controlled_color": color_token,
                            "ply_count": len(plies),
                            "content_sha256": _normalized_game_sha256(row),
                        }
                    )

        examples.sort(
            key=lambda item: (
                item.shard_index,
                item.game_id,
                item.controlled_color,
                item.start_ply,
            )
        )
        identity = {
            "version": LOADER_STATE_VERSION,
            "split": split,
            "chunk_length": chunk_length,
            "games": identity_records,
        }
        identity_sha256 = sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            examples,
            identity_sha256=identity_sha256,
            split=split,
            chunk_length=chunk_length,
        )


class SequenceDataLoader(Iterator[SequenceBatch]):
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
    ) -> SequenceDataLoader:
        """Build a configured loader directly from normalized Parquet shards."""

        return cls(
            SequenceDataset.from_parquet(
                paths,
                split=config.split,
                chunk_length=config.chunk_length,
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
    """Pad and pack sequence examples without introducing fake targets."""

    if not examples:
        raise ValueError("cannot collate an empty sequence collection")
    sequence_length = max(len(example.plies) for example in examples)
    padded = tuple(
        tuple(example.plies) + (None,) * (sequence_length - len(example.plies))
        for example in examples
    )
    attention_mask = tuple(tuple(ply is not None for ply in row) for row in padded)

    inputs = SequenceInputs(
        piece_ids=tuple(
            tuple(
                ply.board.piece_ids if ply is not None else (0,) * BOARD_SQUARE_COUNT
                for ply in row
            )
            for row in padded
        ),
        side_to_move=_pack_required(padded, lambda ply: ply.board.side_to_move),
        castling_rights=_pack_required(padded, lambda ply: ply.board.castling_rights),
        en_passant_square=_pack_optional(
            padded, lambda ply: ply.board.en_passant_square
        ),
        halfmove_clock=_pack_required(padded, lambda ply: ply.board.halfmove_clock),
        fullmove_number=_pack_required(padded, lambda ply: ply.board.fullmove_number),
        previous_action_id=_pack_optional(padded, lambda ply: ply.previous_action_id),
        target_rating=_pack_optional(padded, lambda ply: ply.target_rating),
        controlled_color=_pack_required(padded, lambda ply: ply.controlled_color),
        time_initial_ms=_pack_optional(padded, lambda ply: ply.time_initial_ms),
        time_increment_ms=_pack_optional(padded, lambda ply: ply.time_increment_ms),
        player_clock_ms=_pack_optional(padded, lambda ply: ply.player_clock_ms),
        opponent_clock_ms=_pack_optional(padded, lambda ply: ply.opponent_clock_ms),
    )
    return SequenceBatch(
        inputs=inputs,
        action_targets=_pack_required(padded, lambda ply: ply.target_action_id),
        action_loss_mask=tuple(
            tuple(
                ply is not None and ply.board.side_to_move == ply.controlled_color
                for ply in row
            )
            for row in padded
        ),
        attention_mask=attention_mask,
        causal_attention_mask=tuple(
            tuple(key_index <= query_index for key_index in range(sequence_length))
            for query_index in range(sequence_length)
        ),
        legal_action_ids=tuple(
            tuple(ply.legal_action_ids if ply is not None else () for ply in row)
            for row in padded
        ),
        game_ids=_pack_required(padded, lambda ply: ply.game_id),
        ply_indices=_pack_required(padded, lambda ply: ply.ply_index),
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


def _read_normalized_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - exercised by wheel smoke only
        raise DataLoadingError(
            "Parquet support is unavailable; install anthro-chess[data]"
        ) from error
    try:
        table = pq.read_table(path, columns=list(_LOADER_COLUMNS))
    except (OSError, ValueError) as error:
        raise DataLoadingError(
            f"cannot read normalized data {path}: {error}"
        ) from error
    return cast(list[dict[str, Any]], table.to_pylist())


def _game_from_row(row: Mapping[str, Any], path: Path) -> GameEncodingInput:
    if row[NormalizedColumn.SCHEMA_VERSION] != SCHEMA_VERSION:
        raise DataLoadingError(
            f"{path} uses normalized schema version "
            f"{row[NormalizedColumn.SCHEMA_VERSION]}; "
            f"expected {SCHEMA_VERSION}"
        )
    try:
        action_ids = tuple(row[NormalizedColumn.ACTION_IDS])
        clocks = tuple(row[NormalizedColumn.CLOCK_REMAINING_MS])
        return GameEncodingInput(
            game_id=row[NormalizedColumn.GAME_ID],
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


def _pack_required(
    padded: Sequence[Sequence[PlyEncoding | None]],
    getter: Callable[[PlyEncoding], int],
) -> IntMatrix:
    return tuple(
        tuple(getter(ply) if ply is not None else 0 for ply in row) for row in padded
    )


def _pack_optional(
    padded: Sequence[Sequence[PlyEncoding | None]],
    getter: Callable[[PlyEncoding], int | None],
) -> OptionalIntBatch:
    values: list[tuple[int, ...]] = []
    present: list[tuple[bool, ...]] = []
    for row in padded:
        row_values: list[int] = []
        row_present: list[bool] = []
        for ply in row:
            value = getter(ply) if ply is not None else None
            row_values.append(value if value is not None else 0)
            row_present.append(value is not None)
        values.append(tuple(row_values))
        present.append(tuple(row_present))
    return OptionalIntBatch(tuple(values), tuple(present))


def _shuffle_key(seed: str, epoch: int, example: SequenceExample) -> bytes:
    return sha256(
        (
            f"{seed}\0{epoch}\0{example.shard_index}\0"
            f"{example.game_id}\0{example.controlled_color}\0{example.start_ply}"
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
