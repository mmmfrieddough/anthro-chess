"""Build and load the frozen evaluation pool drawn from the test split.

The pool is a regenerable pipeline output, not committed data. What is checked
in is the selection configuration plus, once a pool exists, its expected
identity digest, so a rebuild on another machine is verifiable rather than
merely plausible.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
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
    game_ids_sha256,
    materialize_rows,
    normalized_row_group_count,
    normalized_shard_paths,
    open_normalized_shard,
    read_normalized_row_group,
    read_normalized_rows,
    row_group_column,
    take_rows,
    validate_manifest_compatibility,
    write_normalized_rows,
)
from anthro_chess.data.schema import (
    GAME_IDENTITY_COLUMNS,
    NORMALIZED_COLUMNS,
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    NormalizedColumn,
    SplitName,
    derive_game_id,
    row_game_id,
)
from anthro_chess.evaluation.coverage import pool_coverage
from anthro_chess.evaluation.results.records import Sha256Hex

BENCHMARK_VERSION = 1
POOL_GAMES_FILE_NAME = "games.parquet"
POOL_MANIFEST_FILE_NAME = "manifest.json"

#: The seed the pool's own admission ranks under. Separate from any view seed so
#: that what a view takes first is independent of what the pool admitted, and a
#: constant rather than a configured field because a different seed redraws
#: membership and would drop games an earlier generation contains.
POOL_SAMPLE_SEED = "anthro-evaluation-pool-v1"

logger = logging.getLogger(__name__)

#: What deciding one game's admission reads. A freeze scans every corpus row to
#: keep one split's admitted share of it, so reading the rest of the schema for
#: the rows it discards is most of what a freeze costs.
_FREEZE_SCAN_COLUMNS: tuple[NormalizedColumn, ...] = (
    NormalizedColumn.SPLIT,
    *GAME_IDENTITY_COLUMNS,
)

#: The columns a :class:`PoolGame` is derived from. Reading the schema's
#: remaining columns only to discard them is most of what a load costs.
_POOL_GAME_COLUMNS = (
    NormalizedColumn.SOURCE_ID.value,
    NormalizedColumn.SOURCE_GAME_KEY.value,
    NormalizedColumn.PLY_COUNT.value,
    NormalizedColumn.RESULT.value,
    NormalizedColumn.WHITE_NORMALIZED_RATING.value,
    NormalizedColumn.BLACK_NORMALIZED_RATING.value,
)

#: Games parsed by an earlier load in this process, keyed on the artifact
#: checksum every load computes anyway. One sweep loads the same pool five to
#: twelve times and the parse is nearly the whole cost. Keying on content
#: rather than on the path makes a hit proof that the same bytes were parsed,
#: so a pool rewritten in place is reloaded rather than remembered.
_POOL_GAMES: dict[str, tuple[PoolGame, ...]] = {}


class EvaluationPoolError(ValueError):
    """Raised when a pool cannot be built from or loaded for a selection."""


class PoolGenerationPin(ConfigModel):
    """The pool generation a benchmark selection is defined over.

    Held by a type rather than by repetition, because the cost workload has to
    know which configured field names realized data identity so it can leave it
    out: a pool joins that digest as the artifact it names, and a pin moving at
    every generation cut would otherwise start a fresh cost line at each one.
    ``docs/decisions/0048-a-pinned-pool-generation-is-not-a-cost-workload.md``
    owns that rule, so a later field naming realized data belongs here too.

    Inherited rather than nested, for the reason
    :mod:`anthro_chess.evaluation.selection` gives: a field typed as another
    model is a table in the TOML, so composing this in would move the pin under
    one and rewrite every shipped selection.
    """

    #: Absent loads whatever the configured path holds, which is what a pool
    #: with no designated generation needs.
    expected_pool_game_ids_sha256: Sha256Hex | None = None


class PoolConfig(ConfigModel):
    """Code-owned schema for ``anthro eval freeze``."""

    pool_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    pool_version: int = Field(default=1, ge=1)
    normalized: Path
    manifest: Path
    split: SplitName = "test"
    #: How much of the split the pool admits, absent to admit all of it, which
    #: every shipped selection still does. Raising it later is a generation cut
    #: and lowering it drops games earlier generations hold, so it only ever
    #: rises — a direction nothing here can enforce.
    #: ``docs/decisions/0052-a-bounded-pool-is-a-fixed-admission-fraction.md``
    #: owns why the bound is a fraction rather than a game count.
    sample_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    expected_game_ids_sha256: Sha256Hex | None = None


@dataclass(frozen=True)
class PoolGame:
    """Game-level facts a view needs, without loading full encodings."""

    game_id: int
    ply_count: int
    result: str
    has_ratings: bool


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


def pool_rows(
    pool: FrozenPool,
    game_ids: Sequence[int],
    columns: Sequence[str],
    *,
    error: type[Exception],
) -> tuple[dict[str, Any], ...]:
    """Read a view's selected games, projected to what the caller consumes.

    Naming the columns is the point of gathering the read here: the pool
    carries the whole normalized schema, a benchmark reads a third of it at
    most, and the per-ply clock and action columns it skips are most of what a
    full read costs. A required argument rather than an optional one, so a new
    benchmark cannot pay for the whole schema by saying nothing.
    """

    wanted = set(game_ids)
    try:
        rows = [
            row
            for row in read_normalized_rows(pool.games_path, columns)
            if row_game_id(row) in wanted
        ]
    except DataLoadingError as loading_error:
        raise error(str(loading_error)) from loading_error
    if len(rows) != len(wanted):
        raise error("the evaluation pool does not contain every selected game")
    return tuple(sorted(rows, key=lambda row: row_game_id(row)))


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
    admits = _admission(config.sample_fraction)
    selected: list[dict[str, Any]] = []
    split_games = 0
    # Only an admitted game can reach the pool, so a train id that is not
    # admitted cannot collide with one and holding it would grow this set with
    # the corpus for nothing. The narrowing is the pool's to accept: what a
    # freeze can still see is whether the games it wrote are held out.
    train_game_ids: set[int] = set()
    for path in source_paths:
        shard = open_normalized_shard(path)
        for row_group in range(normalized_row_group_count(shard)):
            scan = read_normalized_row_group(shard, row_group, _FREEZE_SCAN_COLUMNS)
            identities = zip(
                row_group_column(scan, NormalizedColumn.SPLIT),
                row_group_column(scan, NormalizedColumn.SOURCE_ID),
                row_group_column(scan, NormalizedColumn.SOURCE_GAME_KEY),
                strict=True,
            )
            positions: list[int] = []
            for position, (split, source_id, game_key) in enumerate(identities):
                if split == config.split:
                    split_games += 1
                    if admits(derive_game_id(source_id, game_key)):
                        positions.append(position)
                elif split == "train":
                    train_game_id = derive_game_id(source_id, game_key)
                    if admits(train_game_id):
                        train_game_ids.add(train_game_id)
            if positions:
                # A projection does not reorder a row group, so a position the
                # scan found names the same game in the full read.
                full = read_normalized_row_group(shard, row_group, NORMALIZED_COLUMNS)
                selected.extend(materialize_rows(take_rows(full, positions)))

    if not selected:
        if split_games == 0:
            message = f"no normalized games are assigned to the {config.split} split"
        else:
            message = (
                f"a sample fraction of {config.sample_fraction} admitted none of "
                f"the {split_games} game(s) in the {config.split} split"
            )
        raise EvaluationPoolError(message)

    selected.sort(key=lambda row: row_game_id(row))
    games = tuple(pool_game(row) for row in selected)
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
    # A corpus prepared before manifests spanned archives records one `input`,
    # and freezing from one is legitimate where appending to it is not, so its
    # provenance is carried rather than dropped.
    source_inputs = source_manifest.get("inputs")
    if source_inputs is None and "input" in source_manifest:
        source_inputs = [source_manifest["input"]]
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
            "inputs": source_inputs,
            "split": source_manifest.get("split"),
            "selection": source_manifest.get("selection"),
        },
        "sampling": {
            "algorithm": "sha256-prefix-fraction-v1",
            "seed": POOL_SAMPLE_SEED,
            "fraction": config.sample_fraction,
            "split_games": split_games,
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
                {
                    "game_id": row_game_id(row),
                    "content_sha256": _row_sha256(row),
                }
                for row in selected
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


def load_pool(
    directory: str | Path,
    *,
    expected_game_ids_sha256: str | None = None,
) -> FrozenPool:
    """Load a frozen pool and verify it against its recorded identity.

    A load repeated in the same process reuses the games an earlier one
    parsed, but skips none of the verification: the manifest is re-read, the
    artifact re-checksummed, and the recorded identity re-derived every time.

    Every check but one asks whether the pool is intact and readable by this
    code, which any pool of any generation can be. ``expected_game_ids_sha256``
    is what the caller's configuration expected to find here, and it is the
    only check that can tell a superseded pool left on disk from the one the
    reading is defined over.
    """

    try:
        return _load_pool(directory, expected_game_ids_sha256)
    except DataLoadingError as error:
        raise EvaluationPoolError(str(error)) from error


def _load_pool(
    directory: str | Path,
    expected_game_ids_sha256: str | None,
) -> FrozenPool:
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

    games = _POOL_GAMES.get(observed)
    if games is None:
        games = tuple(
            pool_game(row)
            for row in read_normalized_rows(games_path, _POOL_GAME_COLUMNS)
        )
        _POOL_GAMES[observed] = games

    recorded = manifest.get("identity")
    if not isinstance(recorded, Mapping):
        raise EvaluationPoolError(f"{manifest_path} has no identity record")
    identity = game_ids_sha256([game.game_id for game in games])
    if recorded.get("game_ids_sha256") != identity:
        raise EvaluationPoolError(
            f"evaluation pool contents do not match the recorded identity: {games_path}"
        )
    if expected_game_ids_sha256 is not None and identity != expected_game_ids_sha256:
        raise EvaluationPoolError(
            f"the evaluation pool at {pool_path} is not the one this "
            f"configuration is defined over: expected {expected_game_ids_sha256}, "
            f"loaded {identity}"
        )
    return FrozenPool(games_path=games_path, manifest=manifest, games=games)


def rank_key(seed: str, game_id: int) -> bytes:
    """Rank uniformly by game id so a sample stays representative."""

    return sha256(f"{seed}\0{game_id}".encode()).digest()


def _admission(fraction: float | None) -> Callable[[int], bool]:
    """Return the predicate a configured sample fraction admits games by.

    A fraction and not a count, so that a game is decided on its id alone and
    growth only ever adds.
    ``docs/decisions/0052-a-bounded-pool-is-a-fixed-admission-fraction.md``
    holds why a count cannot be made to work here.

    Split assignment reaches for the same arithmetic and is deliberately not
    shared with it: that one is versioned in every corpus manifest and may be
    redrawn, and this one may not. What the two have in common is the property,
    not the code, and each is a pure function of the game id.

    Where that one divides every candidate into a ratio, this compares whole
    integers, so the one float conversion happens per freeze rather than per
    game.
    """

    if fraction is None:
        return lambda game_id: True
    threshold = int(fraction * 2**64)
    return lambda game_id: (
        int.from_bytes(rank_key(POOL_SAMPLE_SEED, game_id)[:8], "big") < threshold
    )


def pool_game(row: Mapping[str, Any]) -> PoolGame:
    """Return the game-level facts a view selects on, from one row.

    A view ranks and subsamples games, and the validation split needs the
    same derivation the frozen pool uses so an in-training preview selects
    games the same way a canonical reading does.
    """

    return PoolGame(
        game_id=row_game_id(row),
        ply_count=int(row[NormalizedColumn.PLY_COUNT]),
        result=str(row[NormalizedColumn.RESULT]),
        has_ratings=(
            row[NormalizedColumn.WHITE_NORMALIZED_RATING] is not None
            and row[NormalizedColumn.BLACK_NORMALIZED_RATING] is not None
        ),
    )


def _row_sha256(row: Mapping[str, Any]) -> str:
    """Digest one game's content for the freeze-time identity block.

    Nothing reads this after a pool is written.
    """

    content = {
        str(column): row[column]
        for column in (
            NormalizedColumn.SOURCE_ID,
            NormalizedColumn.SOURCE_GAME_KEY,
            NormalizedColumn.RULESET,
            NormalizedColumn.INITIAL_POSITION,
            NormalizedColumn.ACTION_IDS,
            NormalizedColumn.RESULT,
            NormalizedColumn.WHITE_NORMALIZED_RATING,
            NormalizedColumn.BLACK_NORMALIZED_RATING,
            NormalizedColumn.CLOCK_REMAINING_DELTA_MS,
        )
    }
    return sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
