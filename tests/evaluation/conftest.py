"""Shared normalized-artifact and results-store fixtures for evaluation tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
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
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
    EnvironmentRecord,
    Measurement,
    ResultEnvelope,
    build_result,
    dataset_reference,
    measurement,
    projection_content_digest,
    registry_snapshot,
    restore_registry,
)
from anthro_chess.evaluation.results.metrics import MOVE_PREDICTION_PROJECTION

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


FIXTURE_ENVIRONMENT = EnvironmentRecord(
    package_version="0.0.0-fixture",
    git_revision="0" * 40,
    python_version="3.11.0",
    platform="fixture-platform",
    dependencies={"torch": "2.7.0"},
)


@pytest.fixture(autouse=True)
def restored_registry() -> Iterator[None]:
    """Undo any metric registered by a test, keeping the registry global."""

    snapshot = registry_snapshot()
    try:
        yield
    finally:
        restore_registry(snapshot)


def _scored_row(game_id: int, **overrides: Any) -> dict[str, Any]:
    """Return one projected row as a benchmark would hand it to a digest."""

    row: dict[str, Any] = {
        "game_id": game_id,
        "ruleset": "standard",
        "initial_position": "startpos",
        "action_ids": [1, 2, 3],
        "white_normalized_rating": 1500,
        "black_normalized_rating": 1500,
        # Deliberately outside the move-prediction projection.
        "clock_remaining_ms": [300_000, 299_000],
        "clock_status": ["present", "present"],
        "time_initial_ms": 300_000,
        "time_increment_ms": 0,
        "result": "1-0",
    }
    row.update(overrides)
    return row


@pytest.fixture
def scored_row() -> Callable[..., dict[str, Any]]:
    """Return a factory for one projected row of a scored game."""

    return _scored_row


@pytest.fixture
def move_prediction_component() -> Callable[..., DataComponent]:
    """Return a factory for the move-prediction content digest."""

    def build(rows: Sequence[dict[str, Any]] | None = None) -> DataComponent:
        return projection_content_digest(
            rows if rows is not None else [_scored_row(1), _scored_row(2)],
            MOVE_PREDICTION_PROJECTION,
        )

    return build


@pytest.fixture
def recorded_result(
    move_prediction_component: Callable[..., DataComponent],
) -> Callable[..., ResultEnvelope]:
    """Return a factory for a complete held-out prediction result."""

    def build(
        *,
        label: str = "checkpoint-a",
        step: int = 8000,
        move_loss: float = 3.5,
        mask_penalty: float = 0.75,
        component: DataComponent | None = None,
        measurements: Sequence[Measurement] | None = None,
        recorded_at: datetime | None = None,
        kind: str = "held-out-prediction",
    ) -> ResultEnvelope:
        data = component if component is not None else move_prediction_component()
        values = (
            list(measurements)
            if measurements is not None
            else [
                measurement("held_out.move_loss", move_loss, data=data),
                measurement("legality.mask_penalty", mask_penalty, data=data),
            ]
        )
        return build_result(
            kind=kind,
            benchmark=BenchmarkReference(name="move-validation", version=1),
            checkpoint=CheckpointReference(label=label, step=step),
            data=dataset_reference(
                pool_id="fixture-pool",
                pool_version=1,
                view="canonical",
                selected_games=data.games,
                game_ids_sha256="a" * 64,
                components=[data],
            ),
            measurements=values,
            environment=FIXTURE_ENVIRONMENT,
            recorded_at=recorded_at or datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
        )

    return build
