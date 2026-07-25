"""Shared normalized-artifact fixtures for evaluation tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chess
import pytest

from anthro_chess.chess import action_vocabulary_identity, encode_move
from anthro_chess.data.schema import (
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    normalized_parquet_schema,
)

OPENING_MOVES = (
    "e2e4",
    "e7e5",
    "g1f3",
    "b8c6",
    "f1b5",
    "a7a6",
    "b5a4",
    "g8f6",
    "e1g1",
    "f8e7",
)


def _action_ids(moves: tuple[str, ...]) -> tuple[int, ...]:
    """Return action ids for UCI move strings."""

    return tuple(encode_move(chess.Move.from_uci(move)) for move in moves)


def _normalized_row(
    game_id: int,
    *,
    split: str = "test",
    plies: int = 6,
    rating: int | None = 1500,
    clocks: bool = True,
    result: str = "1-0",
) -> dict[str, Any]:
    """Return one canonical normalized game row."""

    moves = OPENING_MOVES[:plies]
    status = "present"
    return {
        "schema_version": SCHEMA_VERSION,
        "game_id": game_id,
        "source_id": "fixture",
        "source_game_key": f"game{game_id}",
        "ruleset": "standard",
        "initial_position": chess.STARTING_FEN,
        "result": result,
        "termination": "normal",
        "termination_status": status,
        "ply_count": len(moves),
        "action_ids": list(_action_ids(moves)),
        "white_source_rating": rating,
        "white_source_rating_status": status if rating else "unavailable",
        "black_source_rating": rating,
        "black_source_rating_status": status if rating else "unavailable",
        "source_rating_namespace": "fixture_blitz",
        "source_rating_system": "glicko2",
        "white_normalized_rating": rating,
        "white_normalized_rating_status": status if rating else "unavailable",
        "black_normalized_rating": rating,
        "black_normalized_rating_status": status if rating else "unavailable",
        "time_initial_ms": 300_000,
        "time_initial_status": status,
        "time_increment_ms": 0,
        "time_increment_status": status,
        "clock_remaining_ms": [290_000] * len(moves) if clocks else [None] * len(moves),
        "clock_status": [status if clocks else "unavailable"] * len(moves),
        "clock_precision_ms": [100 if clocks else None] * len(moves),
        "split": split,
    }


def _write_corpus(directory: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    """Write a normalized shard plus a matching manifest, returning both paths."""

    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    normalized_directory = directory / "normalized"
    manifest_directory = directory / "manifests"
    normalized_directory.mkdir(parents=True, exist_ok=True)
    manifest_directory.mkdir(parents=True, exist_ok=True)

    games_path = normalized_directory / "games.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=normalized_parquet_schema()),
        games_path,
        compression="zstd",
    )

    manifest_path = manifest_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "preprocessing_version": PREPROCESSING_VERSION,
                "action_vocabulary": action_vocabulary_identity(),
                "source": {"id": "fixture", "version": "v1"},
                "input": {"file_name": "fixture.pgn", "sha256": "0" * 64},
                "split": {"algorithm": "sha256-threshold-v2", "seed": "fixture"},
                "selection": {"algorithm": "fixture"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return normalized_directory, manifest_path


@pytest.fixture
def action_ids() -> Callable[[tuple[str, ...]], tuple[int, ...]]:
    """Return a helper converting UCI move strings into action ids."""

    return _action_ids


@pytest.fixture
def normalized_row() -> Callable[..., dict[str, Any]]:
    """Return a factory for canonical normalized game rows."""

    return _normalized_row


@pytest.fixture
def write_corpus() -> Callable[[Path, list[dict[str, Any]]], tuple[Path, Path]]:
    """Return a factory writing a normalized shard plus matching manifest."""

    return _write_corpus
