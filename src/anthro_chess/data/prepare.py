"""Normalize standard-chess PGN games into deterministic Parquet artifacts."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, TextIO
from urllib.parse import urlparse

import chess
import chess.pgn

from anthro_chess.chess import action_vocabulary_identity, encode_move
from anthro_chess.config import ResolvedConfig
from anthro_chess.data.config import PrepareConfig

SCHEMA_VERSION = 1
PREPROCESSING_VERSION = 1
FieldStatus = Literal["present", "unavailable", "rejected"]
_STATUS_PRESENT: FieldStatus = "present"
_STATUS_UNAVAILABLE: FieldStatus = "unavailable"
_STATUS_REJECTED: FieldStatus = "rejected"
_CLOCK_RE = re.compile(r"\[%clk\s+([^\]\s]+)\]")
_CLOCK_CENTISECONDS_RE = re.compile(r"\[%clkc\s+([^\]\s]+)\]")
_CLOCK_SHAPED_RE = re.compile(r"\[%clk(?:c)?\b")
_TIME_CONTROL_RE = re.compile(r"(\d+)\+(\d+)")
_SOURCE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class DataPreparationError(ValueError):
    """Raised when a data-preparation run cannot produce a valid artifact."""


@dataclass(frozen=True)
class PreparationResult:
    """Paths and counts produced by one preparation run."""

    normalized_path: Path
    manifest_path: Path
    accepted_games: int
    rejected_games: int
    split_counts: dict[str, int]


@dataclass(frozen=True)
class _OptionalInteger:
    value: int | None
    status: FieldStatus


@dataclass(frozen=True)
class _ParsedClock:
    value: int | None
    status: FieldStatus
    precision_ms: int | None


@dataclass(frozen=True)
class _ParsedGame:
    record: dict[str, object] | None
    rejection: str | None


def prepare_pgn(
    input_path: str | Path,
    output_directory: str | Path,
    resolved_config: ResolvedConfig[PrepareConfig],
) -> PreparationResult:
    """Prepare a PGN stream and write normalized and manifest artifacts."""
    source_path = Path(input_path)
    output_path = Path(output_directory)
    if not source_path.is_file():
        raise DataPreparationError(f"input PGN does not exist: {source_path}")

    input_sha256 = _file_sha256(source_path)
    records: list[dict[str, object]] = []
    rejections: Counter[str] = Counter()
    seen_game_ids: set[int] = set()

    try:
        with source_path.open("r", encoding="utf-8") as pgn_file:
            for game in _read_games(pgn_file):
                parsed = _parse_game(game, resolved_config.value)
                if parsed.record is None:
                    assert parsed.rejection is not None
                    rejections[parsed.rejection] += 1
                else:
                    game_id = _record_game_id(parsed.record)
                    if game_id in seen_game_ids:
                        rejections["duplicate_game"] += 1
                        continue
                    seen_game_ids.add(game_id)
                    records.append(parsed.record)
    except (OSError, UnicodeError) as error:
        raise DataPreparationError(
            f"cannot read input PGN {source_path}: {error}"
        ) from error

    if not records:
        detail = ", ".join(
            f"{reason}={count}" for reason, count in sorted(rejections.items())
        )
        raise DataPreparationError(
            "no games passed preparation filters" + (f" ({detail})" if detail else "")
        )

    records.sort(key=_record_game_id)
    normalized_directory = output_path / "normalized"
    manifest_directory = output_path / "manifests"
    normalized_directory.mkdir(parents=True, exist_ok=True)
    manifest_directory.mkdir(parents=True, exist_ok=True)
    normalized_path = normalized_directory / "games.parquet"
    manifest_path = manifest_directory / "manifest.json"

    _write_parquet(records, normalized_path)
    normalized_sha256 = _file_sha256(normalized_path)
    observed_splits = Counter(str(record["split"]) for record in records)
    split_counts = {
        split_name: observed_splits[split_name]
        for split_name in ("train", "validation")
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "source": resolved_config.value.source.model_dump(mode="json"),
        "input": {
            "file_name": source_path.name,
            "sha256": input_sha256,
        },
        "output": {
            "path": "normalized/games.parquet",
            "sha256": normalized_sha256,
            "format": "parquet",
            "compression": "zstd",
        },
        "action_vocabulary": action_vocabulary_identity(),
        "split": {
            "algorithm": "sha256-threshold-v1",
            "seed": resolved_config.value.split.seed,
            "validation_fraction": resolved_config.value.split.validation_fraction,
            "counts": split_counts,
        },
        "games": {
            "accepted": len(records),
            "rejected": sum(rejections.values()),
            "rejection_reasons": dict(sorted(rejections.items())),
        },
        "resolved_config": resolved_config.as_record(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PreparationResult(
        normalized_path=normalized_path,
        manifest_path=manifest_path,
        accepted_games=len(records),
        rejected_games=sum(rejections.values()),
        split_counts=split_counts,
    )


def _read_games(pgn_file: TextIO) -> Iterator[chess.pgn.Game]:
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            return
        yield game


def _parse_game(game: chess.pgn.Game, config: PrepareConfig) -> _ParsedGame:
    if game.errors:
        return _ParsedGame(None, "pgn_parse_error")
    variant = game.headers.get("Variant", "Standard")
    if variant.casefold() not in {"standard", "from position"}:
        return _ParsedGame(None, "unsupported_variant")
    if game.headers.get("SetUp") == "1" or game.headers.get("FEN"):
        return _ParsedGame(None, "nonstandard_initial_position")
    if config.filters.exclude_bots and _has_bot_player(game):
        return _ParsedGame(None, "bot_game")
    if (
        config.filters.require_rated
        and "rated" not in game.headers.get("Event", "").casefold()
    ):
        return _ParsedGame(None, "unrated_game")

    source_game_key = _source_game_key(game.headers.get("Site"))
    if source_game_key is None:
        return _ParsedGame(None, "missing_source_game_key")

    white_rating = _parse_nonnegative_integer(game.headers.get("WhiteElo"))
    black_rating = _parse_nonnegative_integer(game.headers.get("BlackElo"))
    time_initial, time_increment = _parse_time_control(game.headers.get("TimeControl"))
    actions: list[int] = []
    clock_values: list[int | None] = []
    clock_statuses: list[FieldStatus] = []
    clock_precisions: list[int | None] = []
    board = game.board()
    for node in game.mainline():
        move = node.move
        if move not in board.legal_moves:
            return _ParsedGame(None, "illegal_move")
        actions.append(encode_move(move))
        clock = _parse_clock(node.comment)
        clock_values.append(clock.value)
        clock_statuses.append(clock.status)
        clock_precisions.append(clock.precision_ms)
        board.push(move)

    if len(actions) < config.filters.minimum_plies:
        return _ParsedGame(None, "too_short")

    result = game.headers.get("Result")
    if result not in {"1-0", "0-1", "1/2-1/2", "*"}:
        return _ParsedGame(None, "invalid_result")

    termination, termination_status = _parse_text(game.headers.get("Termination"))
    game_id = _game_id(config.source.id, source_game_key)
    split = _split_name(
        game_id,
        seed=config.split.seed,
        validation_fraction=config.split.validation_fraction,
    )
    normalized_white = (
        white_rating.value if config.source.ratings_are_normalized else None
    )
    normalized_black = (
        black_rating.value if config.source.ratings_are_normalized else None
    )
    normalized_white_status: FieldStatus = (
        white_rating.status
        if config.source.ratings_are_normalized
        else _STATUS_UNAVAILABLE
    )
    normalized_black_status: FieldStatus = (
        black_rating.status
        if config.source.ratings_are_normalized
        else _STATUS_UNAVAILABLE
    )
    return _ParsedGame(
        {
            "schema_version": SCHEMA_VERSION,
            "game_id": game_id,
            "source_id": config.source.id,
            "source_game_key": source_game_key,
            "ruleset": "standard",
            "initial_position": chess.STARTING_FEN,
            "result": result,
            "termination": termination,
            "termination_status": termination_status,
            "ply_count": len(actions),
            "action_ids": actions,
            "white_source_rating": white_rating.value,
            "white_source_rating_status": white_rating.status,
            "black_source_rating": black_rating.value,
            "black_source_rating_status": black_rating.status,
            "source_rating_namespace": config.source.rating_namespace,
            "source_rating_system": config.source.rating_system,
            "white_normalized_rating": normalized_white,
            "white_normalized_rating_status": normalized_white_status,
            "black_normalized_rating": normalized_black,
            "black_normalized_rating_status": normalized_black_status,
            "time_initial_ms": time_initial.value,
            "time_initial_status": time_initial.status,
            "time_increment_ms": time_increment.value,
            "time_increment_status": time_increment.status,
            "clock_remaining_ms": clock_values,
            "clock_status": clock_statuses,
            "clock_precision_ms": clock_precisions,
            "split": split,
        },
        None,
    )


def _has_bot_player(game: chess.pgn.Game) -> bool:
    return any(
        game.headers.get(header, "").casefold() == "bot"
        for header in ("WhiteTitle", "BlackTitle")
    )


def _source_game_key(site: str | None) -> str | None:
    if site is None:
        return None
    parsed = urlparse(site)
    candidate = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if not candidate or not _SOURCE_KEY_RE.fullmatch(candidate):
        return None
    return candidate


def _parse_nonnegative_integer(value: str | None) -> _OptionalInteger:
    if value is None or not value.strip():
        return _OptionalInteger(None, _STATUS_UNAVAILABLE)
    try:
        parsed = int(value)
    except ValueError:
        return _OptionalInteger(None, _STATUS_REJECTED)
    if parsed < 0:
        return _OptionalInteger(None, _STATUS_REJECTED)
    return _OptionalInteger(parsed, _STATUS_PRESENT)


def _parse_time_control(
    value: str | None,
) -> tuple[_OptionalInteger, _OptionalInteger]:
    if value is None or not value.strip() or value in {"-", "?"}:
        unavailable = _OptionalInteger(None, _STATUS_UNAVAILABLE)
        return unavailable, unavailable
    match = _TIME_CONTROL_RE.fullmatch(value.strip())
    if match is None:
        rejected = _OptionalInteger(None, _STATUS_REJECTED)
        return rejected, rejected
    return (
        _OptionalInteger(int(match.group(1)) * 1000, _STATUS_PRESENT),
        _OptionalInteger(int(match.group(2)) * 1000, _STATUS_PRESENT),
    )


def _parse_clock(comment: str) -> _ParsedClock:
    clock_match = _CLOCK_RE.search(comment)
    if clock_match is not None:
        clock_text = clock_match.group(1)
        value = _clock_text_to_milliseconds(clock_text)
        return (
            _ParsedClock(value, _STATUS_PRESENT, _clock_precision_ms(clock_text))
            if value is not None
            else _ParsedClock(None, _STATUS_REJECTED, None)
        )
    centiseconds_match = _CLOCK_CENTISECONDS_RE.search(comment)
    if centiseconds_match is not None:
        try:
            centiseconds = int(centiseconds_match.group(1))
        except ValueError:
            return _ParsedClock(None, _STATUS_REJECTED, None)
        if centiseconds < 0:
            return _ParsedClock(None, _STATUS_REJECTED, None)
        return _ParsedClock(centiseconds * 10, _STATUS_PRESENT, 10)
    if _CLOCK_SHAPED_RE.search(comment):
        return _ParsedClock(None, _STATUS_REJECTED, None)
    return _ParsedClock(None, _STATUS_UNAVAILABLE, None)


def _clock_text_to_milliseconds(value: str) -> int | None:
    parts = value.split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return None
    milliseconds = round((hours * 3600 + minutes * 60 + seconds) * 1000)
    return milliseconds


def _clock_precision_ms(value: str) -> int:
    seconds = value.rsplit(":", 1)[-1]
    decimal_places = len(seconds.partition(".")[2])
    return (1000, 100, 10, 1)[min(decimal_places, 3)]


def _parse_text(value: str | None) -> tuple[str | None, FieldStatus]:
    if value is None or not value.strip():
        return None, _STATUS_UNAVAILABLE
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not normalized:
        return None, _STATUS_REJECTED
    return normalized, _STATUS_PRESENT


def _game_id(source_id: str, source_game_key: str) -> int:
    digest = sha256(f"{source_id}\0{source_game_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _split_name(
    game_id: int, *, seed: str, validation_fraction: float
) -> Literal["train", "validation"]:
    digest = sha256(f"{seed}\0{game_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if fraction < validation_fraction else "train"


def _write_parquet(records: list[dict[str, object]], path: Path) -> None:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - exercised by wheel smoke only
        raise DataPreparationError(
            "Parquet support is unavailable; install anthro-chess[data]"
        ) from error

    schema = pa.schema(
        [
            pa.field("schema_version", pa.int16(), nullable=False),
            pa.field("game_id", pa.uint64(), nullable=False),
            pa.field("source_id", pa.string(), nullable=False),
            pa.field("source_game_key", pa.string(), nullable=False),
            pa.field("ruleset", pa.string(), nullable=False),
            pa.field("initial_position", pa.string(), nullable=False),
            pa.field("result", pa.string(), nullable=False),
            pa.field("termination", pa.string()),
            pa.field("termination_status", pa.string(), nullable=False),
            pa.field("ply_count", pa.int32(), nullable=False),
            pa.field("action_ids", pa.list_(pa.uint16()), nullable=False),
            pa.field("white_source_rating", pa.int32()),
            pa.field("white_source_rating_status", pa.string(), nullable=False),
            pa.field("black_source_rating", pa.int32()),
            pa.field("black_source_rating_status", pa.string(), nullable=False),
            pa.field("source_rating_namespace", pa.string()),
            pa.field("source_rating_system", pa.string()),
            pa.field("white_normalized_rating", pa.int32()),
            pa.field("white_normalized_rating_status", pa.string(), nullable=False),
            pa.field("black_normalized_rating", pa.int32()),
            pa.field("black_normalized_rating_status", pa.string(), nullable=False),
            pa.field("time_initial_ms", pa.int32()),
            pa.field("time_initial_status", pa.string(), nullable=False),
            pa.field("time_increment_ms", pa.int32()),
            pa.field("time_increment_status", pa.string(), nullable=False),
            pa.field("clock_remaining_ms", pa.list_(pa.int32()), nullable=False),
            pa.field("clock_status", pa.list_(pa.string()), nullable=False),
            pa.field("clock_precision_ms", pa.list_(pa.int32()), nullable=False),
            pa.field("split", pa.string(), nullable=False),
        ],
        metadata={
            b"anthro_schema_version": str(SCHEMA_VERSION).encode(),
            b"anthro_preprocessing_version": str(PREPROCESSING_VERSION).encode(),
        },
    )
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def _record_game_id(record: dict[str, object]) -> int:
    game_id = record["game_id"]
    if not isinstance(game_id, int):  # pragma: no cover - internal invariant
        raise TypeError("normalized game id must be an integer")
    return game_id


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
