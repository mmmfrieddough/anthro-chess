"""Load and validate the owned, versioned Lichess puzzle set."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from importlib.resources import files
from typing import Any

import chess

PUZZLE_FILE_NAME = "puzzles.csv"
PUZZLE_METADATA_FILE_NAME = "set.json"
_DATA_PACKAGE = "anthro_chess.evaluation.puzzles"


class PuzzleSetError(ValueError):
    """Raised when the vendored puzzle set cannot be loaded or trusted."""


@dataclass(frozen=True)
class Puzzle:
    """One puzzle rooted before the opponent's setup move."""

    puzzle_id: str
    initial_fen: str
    moves: tuple[chess.Move, ...]
    rating: int
    source_game_key: str

    @property
    def solution_moves(self) -> tuple[chess.Move, ...]:
        """Return the player moves, excluding the opponent's forced replies."""

        return self.moves[1::2]

    def as_projection_record(self) -> dict[str, object]:
        """Return the stable content the benchmark series fingerprints."""

        return {
            "game_id": _puzzle_numeric_id(self.puzzle_id),
            "puzzle_id": self.puzzle_id,
            "initial_fen": self.initial_fen,
            "moves": [move.uci() for move in self.moves],
            "rating": self.rating,
            "source_game_key": self.source_game_key,
        }


@dataclass(frozen=True)
class PuzzleSet:
    """A checked-in puzzle set plus its external-source provenance."""

    name: str
    version: int
    content_sha256: str
    source: Mapping[str, Any]
    license: Mapping[str, Any]
    selection: Mapping[str, Any]
    puzzles: tuple[Puzzle, ...]

    def identity(self) -> dict[str, object]:
        """Return the set identity carried by benchmark artifacts."""

        return {
            "name": self.name,
            "version": self.version,
            "entries": len(self.puzzles),
            "sha256": self.content_sha256,
        }


@cache
def load_puzzle_set() -> PuzzleSet:
    """Load the packaged puzzle set, verifying its checksum and chess lines."""

    metadata = _load_metadata()
    content = _read_data_file(PUZZLE_FILE_NAME)
    observed = sha256(content.encode("utf-8")).hexdigest()
    expected = metadata.get("puzzles_sha256")
    if observed != expected:
        raise PuzzleSetError(
            f"{PUZZLE_FILE_NAME} does not match the checksum recorded in "
            f"{PUZZLE_METADATA_FILE_NAME}: expected {expected}, observed {observed}"
        )

    puzzles = tuple(_parse_puzzles(content))
    if not puzzles:
        raise PuzzleSetError(f"{PUZZLE_FILE_NAME} contains no puzzles")
    if metadata.get("entries") != len(puzzles):
        raise PuzzleSetError(
            f"{PUZZLE_METADATA_FILE_NAME} records {metadata.get('entries')} "
            f"puzzle(s) but {PUZZLE_FILE_NAME} contains {len(puzzles)}"
        )
    identifiers = [puzzle.puzzle_id for puzzle in puzzles]
    if identifiers != sorted(identifiers) or len(set(identifiers)) != len(identifiers):
        raise PuzzleSetError("puzzles must have unique identifiers in sorted order")

    return PuzzleSet(
        name=str(metadata["name"]),
        version=int(metadata["version"]),
        content_sha256=observed,
        source=dict(metadata["source"]),
        license=dict(metadata["license"]),
        selection=dict(metadata["selection"]),
        puzzles=puzzles,
    )


def puzzle_set_identity() -> dict[str, object]:
    """Return the identity for artifacts that carry puzzle results."""

    return load_puzzle_set().identity()


def _parse_puzzles(content: str) -> Iterator[Puzzle]:
    reader = csv.DictReader(content.splitlines())
    expected = [
        "puzzle_id",
        "initial_fen",
        "moves",
        "rating",
        "source_game_key",
    ]
    if reader.fieldnames != expected:
        raise PuzzleSetError(
            f"{PUZZLE_FILE_NAME} must have {', '.join(expected)} columns; "
            f"found {reader.fieldnames}"
        )
    for number, row in enumerate(reader, start=2):
        puzzle_id = (row["puzzle_id"] or "").strip()
        source_game_key = (row["source_game_key"] or "").strip()
        try:
            rating = int(row["rating"])
            board = chess.Board(row["initial_fen"])
        except (TypeError, ValueError) as error:
            raise PuzzleSetError(
                f"{PUZZLE_FILE_NAME} line {number} has invalid metadata: {error}"
            ) from error
        if not puzzle_id or not source_game_key or rating < 0:
            raise PuzzleSetError(
                f"{PUZZLE_FILE_NAME} line {number} has empty or invalid metadata"
            )
        moves: list[chess.Move] = []
        for ply, token in enumerate((row["moves"] or "").split()):
            try:
                move = chess.Move.from_uci(token)
            except ValueError as error:
                raise PuzzleSetError(
                    f"{PUZZLE_FILE_NAME} line {number} has unreadable move "
                    f"{token!r}: {error}"
                ) from error
            if not board.is_legal(move):
                raise PuzzleSetError(
                    f"{PUZZLE_FILE_NAME} line {number} plays illegal move "
                    f"{token!r} at ply {ply}"
                )
            moves.append(move)
            board.push(move)
        if len(moves) < 2 or len(moves) % 2 != 0:
            raise PuzzleSetError(
                f"{PUZZLE_FILE_NAME} line {number} needs an opponent setup move "
                "followed by one or more complete solution pairs"
            )
        yield Puzzle(
            puzzle_id=puzzle_id,
            initial_fen=row["initial_fen"],
            moves=tuple(moves),
            rating=rating,
            source_game_key=source_game_key,
        )


def _load_metadata() -> Mapping[str, Any]:
    try:
        metadata = json.loads(_read_data_file(PUZZLE_METADATA_FILE_NAME))
    except json.JSONDecodeError as error:
        raise PuzzleSetError(
            f"{PUZZLE_METADATA_FILE_NAME} is not valid JSON: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise PuzzleSetError(f"{PUZZLE_METADATA_FILE_NAME} must be a JSON object")
    required = {
        "name",
        "version",
        "entries",
        "puzzles_sha256",
        "source",
        "license",
        "selection",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise PuzzleSetError(
            f"{PUZZLE_METADATA_FILE_NAME} is missing {', '.join(missing)}"
        )
    return metadata


def _read_data_file(name: str) -> str:
    resource = files(_DATA_PACKAGE).joinpath("data", name)
    try:
        return resource.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise PuzzleSetError(f"the packaged puzzle set is missing {name}") from error


def _puzzle_numeric_id(puzzle_id: str) -> int:
    """Map a source id onto the uint64 key the shared digest boundary expects."""

    return int.from_bytes(sha256(puzzle_id.encode()).digest()[:8], "big")
