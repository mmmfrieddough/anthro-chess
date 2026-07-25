"""Build and load the frozen evaluation pool drawn from the test split.

The pool is a regenerable pipeline output, not committed data. What is checked
in is the selection configuration plus, once a pool exists, its expected
identity digest, so a rebuild on another machine is verifiable rather than
merely plausible.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import Field

from anthro_chess.chess import action_vocabulary_identity
from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import encoding_identity
from anthro_chess.data.artifacts import (
    DataLoadingError,
    file_sha256,
    normalized_shard_paths,
    read_normalized_rows,
    validate_manifest_compatibility,
    write_normalized_rows,
)
from anthro_chess.data.schema import (
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    NormalizedColumn,
    SplitName,
)
from anthro_chess.evaluation.coverage import pool_coverage

BENCHMARK_VERSION = 1
POOL_GAMES_FILE_NAME = "games.parquet"
POOL_MANIFEST_FILE_NAME = "manifest.json"
logger = logging.getLogger(__name__)


class EvaluationPoolError(ValueError):
    """Raised when a pool cannot be built from or loaded for a selection."""


def game_ids_sha256(game_ids: Sequence[int]) -> str:
    """Return the order-independent identity digest for a set of game ids."""

    joined = ",".join(str(game_id) for game_id in sorted(game_ids))
    return sha256(joined.encode()).hexdigest()


class PoolConfig(ConfigModel):
    """Code-owned schema for ``anthro eval freeze``."""

    pool_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    pool_version: int = Field(default=1, ge=1)
    normalized: Path
    manifest: Path
    split: SplitName = "test"
    expected_game_ids_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )


@dataclass(frozen=True)
class PoolGame:
    """Game-level facts a view needs, without loading full encodings."""

    game_id: int
    ply_count: int
    result: str
    has_clocks: bool
    has_ratings: bool
    content_sha256: str


@dataclass(frozen=True)
class FrozenPool:
    """A loaded pool artifact and its manifest."""

    games_path: Path
    manifest: dict[str, Any]
    games: tuple[PoolGame, ...]

    @property
    def game_ids(self) -> tuple[int, ...]:
        """Return the pool's game ids in ascending order."""

        return tuple(sorted(game.game_id for game in self.games))


@dataclass(frozen=True)
class PoolResult:
    """Paths and counts produced by one freeze run."""

    games_path: Path
    manifest_path: Path
    games: int
    plies: int
    game_ids_sha256: str


def freeze_pool(
    resolved_config: ResolvedConfig[PoolConfig],
    output_directory: str | Path,
) -> PoolResult:
    """Materialize the configured split as a checksummed evaluation pool.

    The shared artifact helpers are used by preparation and training too, so
    they raise the data package's error. Convert it here to keep one exception
    type at this module's boundary.
    """

    try:
        return _freeze_pool(resolved_config, output_directory)
    except DataLoadingError as error:
        raise EvaluationPoolError(str(error)) from error


def _freeze_pool(
    resolved_config: ResolvedConfig[PoolConfig],
    output_directory: str | Path,
) -> PoolResult:
    config = resolved_config.value
    output_path = Path(output_directory)
    source_paths = normalized_shard_paths(config.normalized)
    manifest_path = Path(config.manifest)
    if not manifest_path.is_file():
        raise EvaluationPoolError(f"source manifest does not exist: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    source_manifest = json.loads(manifest_bytes)
    if not isinstance(source_manifest, dict):
        raise EvaluationPoolError("source manifest must contain a JSON object")
    validate_manifest_compatibility(source_manifest, manifest_path)

    logger.info(
        "Freezing the %s split of %s shard(s) into an evaluation pool",
        config.split,
        len(source_paths),
    )
    selected: list[dict[str, Any]] = []
    train_game_ids: set[int] = set()
    for path in source_paths:
        for row in read_normalized_rows(path):
            split = row[NormalizedColumn.SPLIT]
            if split == config.split:
                selected.append(row)
            elif split == "train":
                train_game_ids.add(int(row[NormalizedColumn.GAME_ID]))

    if not selected:
        raise EvaluationPoolError(
            f"no normalized games are assigned to the {config.split} split"
        )

    selected.sort(key=lambda row: int(row[NormalizedColumn.GAME_ID]))
    games = tuple(_pool_game(row) for row in selected)
    game_ids = tuple(game.game_id for game in games)
    identity = game_ids_sha256(game_ids)
    if (
        config.expected_game_ids_sha256 is not None
        and identity != config.expected_game_ids_sha256
    ):
        raise EvaluationPoolError(
            "rebuilt evaluation pool does not match its expected identity: "
            f"expected {config.expected_game_ids_sha256}, observed {identity}"
        )

    overlap = sorted(set(game_ids) & train_game_ids)
    if overlap:
        raise EvaluationPoolError(
            f"{len(overlap)} pool game(s) also appear in the train split; "
            f"first offending game id is {overlap[0]}"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    games_path = output_path / POOL_GAMES_FILE_NAME
    write_normalized_rows(selected, games_path)

    coverage = pool_coverage(selected)
    pool_manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "pool": {
            "id": config.pool_id,
            "version": config.pool_version,
            "split": config.split,
        },
        "schema_version": SCHEMA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "action_vocabulary": action_vocabulary_identity(),
        "encoding": encoding_identity(),
        "source": {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_bytes).hexdigest(),
            "source": source_manifest.get("source"),
            "input": source_manifest.get("input"),
            "split": source_manifest.get("split"),
            "selection": source_manifest.get("selection"),
        },
        "output": {
            "format": "parquet",
            "compression": "zstd",
            "path": games_path.name,
            "sha256": file_sha256(games_path),
            "games": len(games),
        },
        "identity": {
            "algorithm": "sorted-game-id-sha256-v1",
            "game_ids_sha256": identity,
            "games": [
                {"game_id": game.game_id, "content_sha256": game.content_sha256}
                for game in games
            ],
        },
        "leakage": {
            "algorithm": "game-id-intersection-v1",
            "compared_split": "train",
            "compared_games": len(train_game_ids),
            "overlapping_games": len(overlap),
        },
        "coverage": coverage,
        "resolved_config": resolved_config.as_record(),
    }
    manifest_output_path = output_path / POOL_MANIFEST_FILE_NAME
    manifest_output_path.write_text(
        json.dumps(pool_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    total_plies = int(coverage["plies"]["total"])
    logger.info(
        "Froze %s game(s) and %s ply/plies into %s",
        len(games),
        total_plies,
        games_path,
    )
    return PoolResult(
        games_path=games_path,
        manifest_path=manifest_output_path,
        games=len(games),
        plies=total_plies,
        game_ids_sha256=identity,
    )


def load_pool(directory: str | Path) -> FrozenPool:
    """Load a frozen pool and verify it against its recorded identity."""

    try:
        return _load_pool(directory)
    except DataLoadingError as error:
        raise EvaluationPoolError(str(error)) from error


def _load_pool(directory: str | Path) -> FrozenPool:
    pool_path = Path(directory)
    games_path = pool_path / POOL_GAMES_FILE_NAME
    manifest_path = pool_path / POOL_MANIFEST_FILE_NAME
    if not games_path.is_file():
        raise EvaluationPoolError(f"evaluation pool games do not exist: {games_path}")
    if not manifest_path.is_file():
        raise EvaluationPoolError(
            f"evaluation pool manifest does not exist: {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise EvaluationPoolError("evaluation pool manifest must be a JSON object")
    if manifest.get("benchmark_version") != BENCHMARK_VERSION:
        raise EvaluationPoolError(
            f"{manifest_path} uses benchmark version "
            f"{manifest.get('benchmark_version')}; expected {BENCHMARK_VERSION}"
        )
    if manifest.get("action_vocabulary") != action_vocabulary_identity():
        raise EvaluationPoolError(
            f"{manifest_path} uses an incompatible action vocabulary"
        )
    if manifest.get("encoding") != encoding_identity():
        raise EvaluationPoolError(f"{manifest_path} uses an incompatible encoding")

    output = manifest.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("sha256"), str):
        raise EvaluationPoolError(f"{manifest_path} has no output checksum")
    observed = file_sha256(games_path)
    if observed != output["sha256"]:
        raise EvaluationPoolError(f"evaluation pool checksum mismatch: {games_path}")

    games = tuple(_pool_game(row) for row in read_normalized_rows(games_path))
    recorded = manifest.get("identity")
    if not isinstance(recorded, Mapping):
        raise EvaluationPoolError(f"{manifest_path} has no identity record")
    identity = game_ids_sha256([game.game_id for game in games])
    if recorded.get("game_ids_sha256") != identity:
        raise EvaluationPoolError(
            f"evaluation pool contents do not match the recorded identity: {games_path}"
        )
    return FrozenPool(games_path=games_path, manifest=manifest, games=games)


def _pool_game(row: Mapping[str, Any]) -> PoolGame:
    clock_statuses = row[NormalizedColumn.CLOCK_STATUS]
    return PoolGame(
        game_id=int(row[NormalizedColumn.GAME_ID]),
        ply_count=int(row[NormalizedColumn.PLY_COUNT]),
        result=str(row[NormalizedColumn.RESULT]),
        has_clocks=any(status == "present" for status in clock_statuses),
        has_ratings=(
            row[NormalizedColumn.WHITE_NORMALIZED_RATING] is not None
            and row[NormalizedColumn.BLACK_NORMALIZED_RATING] is not None
        ),
        content_sha256=_row_sha256(row),
    )


def _row_sha256(row: Mapping[str, Any]) -> str:
    content = {
        str(column): row[column]
        for column in (
            NormalizedColumn.GAME_ID,
            NormalizedColumn.RULESET,
            NormalizedColumn.INITIAL_POSITION,
            NormalizedColumn.ACTION_IDS,
            NormalizedColumn.RESULT,
            NormalizedColumn.WHITE_NORMALIZED_RATING,
            NormalizedColumn.BLACK_NORMALIZED_RATING,
            NormalizedColumn.CLOCK_REMAINING_MS,
        )
    }
    return sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
