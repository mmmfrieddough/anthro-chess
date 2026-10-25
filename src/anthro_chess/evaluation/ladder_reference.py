"""The human games a rating ladder's pairings are read against.

For each pair of ratings on a grid, how often the higher-rated human scores in
held-out games. A ladder seat is asked for one exact rating, so a human only
stands in for a grid rating when their own rating lies within a narrow window
of it: binning each player to the nearest grid rating instead lets a nominal
gap span twice the grid step and flattens the curve the ladder is compared to.

The surface is kept over both ratings rather than pooled by gap, because a
pooled curve is dominated by the middle of the range and hides where a model
stops separating players. Every pairing a ladder plays is played from both
sides, so each cell averages the higher-rated player's score as white and as
black rather than taking whatever colour mix the humans happened to play.

The artifact stores counts, and every score is derived from them on load. Its
identity is a digest of the declared selection and those counts, so two
machines building it from one corpus agree on it and a ladder reading can name
the reference it was read against.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import Speed, speed_from_clock_ms
from anthro_chess.data.accounts import MarkedAccountError, snapshot_for_corpus
from anthro_chess.data.artifacts import (
    DataLoadingError,
    normalized_row_group_count,
    normalized_shard_paths,
    open_normalized_shard,
    read_normalized_row_group,
    row_group_column,
    take_rows,
    validate_manifest_compatibility,
)
from anthro_chess.data.schema import NormalizedColumn, SplitName
from anthro_chess.data.termination import TerminationCategory
from anthro_chess.evaluation.results.records import Sha256Hex

#: Bumped when the same counts would derive a different surface, or the same
#: corpus would count differently.
LADDER_REFERENCE_VERSION = 1
LADDER_REFERENCE_FILE_NAME = "reference.json"

logger = logging.getLogger(__name__)

_SCAN_COLUMNS: tuple[NormalizedColumn, ...] = (
    NormalizedColumn.SPLIT,
    NormalizedColumn.RESULT,
    NormalizedColumn.TERMINATION_CATEGORY,
    NormalizedColumn.WHITE_NORMALIZED_RATING,
    NormalizedColumn.BLACK_NORMALIZED_RATING,
    NormalizedColumn.TIME_INITIAL_MS,
    NormalizedColumn.TIME_INCREMENT_MS,
    NormalizedColumn.SOURCE_RATING_NAMESPACE,
    NormalizedColumn.WHITE_PLAYER_DIGEST,
    NormalizedColumn.BLACK_PLAYER_DIGEST,
)

#: The outcome each result is counted under, indexed as the count planes are.
_OUTCOMES = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}
_COUNT_PLANES = ("white_wins", "draws", "black_wins")


class LadderReferenceError(ValueError):
    """Raised when a ladder reference cannot be built or loaded as configured."""


class LadderReferenceConfig(ConfigModel):
    """Code-owned schema for ``anthro eval ladder-reference``."""

    reference_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    normalized: Path
    manifest: Path
    split: SplitName = "test"
    speed: Speed
    rating_namespace: str = Field(min_length=1)
    target_ratings: tuple[int, ...] = Field(min_length=2)
    #: How far a player's rating may sit from a grid rating and still stand in
    #: for it, inclusive.
    rating_window: int = Field(ge=0)
    excluded_terminations: tuple[TerminationCategory, ...] = ()
    #: Games each colour of a cell needs before the cell is read. Below it the
    #: cell reports its count and no score.
    minimum_games: int = Field(ge=2)
    #: Relative to the selection naming it, as a pool selection's is.
    marked_accounts: Path | None = None
    expected_surface_sha256: Sha256Hex | None = None

    @model_validator(mode="after")
    def _check_grid(self) -> LadderReferenceConfig:
        ratings = self.target_ratings
        if list(ratings) != sorted(set(ratings)):
            raise ValueError("target_ratings must be strictly increasing")
        closest = min(
            high - low for low, high in zip(ratings, ratings[1:], strict=False)
        )
        if 2 * self.rating_window >= closest:
            raise ValueError(
                f"a rating window of {self.rating_window} lets one player stand "
                f"in for two grid ratings {closest} apart"
            )
        return self


@dataclass(frozen=True)
class ReferenceCell:
    """How often the higher-rated of two grid ratings scores against the lower."""

    higher_rating: int
    lower_rating: int
    #: Games at both colours, counted whether or not the cell is read.
    games: int
    #: The higher-rated player's score as white and as black, averaged. ``None``
    #: when either colour holds fewer games than the reference declares.
    score: float | None
    standard_error: float | None

    @property
    def thin(self) -> bool:
        """Return whether the cell holds too few games to be read."""

        return self.score is None


@dataclass(frozen=True)
class LadderReference:
    """A loaded reference surface and the identity a reading names it by."""

    reference_id: str
    surface_sha256: str
    target_ratings: tuple[int, ...]
    cells: Mapping[tuple[int, int], ReferenceCell]

    def cell(self, first: int, second: int) -> ReferenceCell:
        """Return the cell for two distinct grid ratings, in either order."""

        key = (max(first, second), min(first, second))
        if key not in self.cells:
            raise LadderReferenceError(
                f"the reference holds no cell for {first} against {second}"
            )
        return self.cells[key]

    def identity(self) -> dict[str, str]:
        """Return what a reading records about the reference it was read against."""

        return {"id": self.reference_id, "surface_sha256": self.surface_sha256}


@dataclass(frozen=True)
class LadderReferenceResult:
    """What one build wrote and how much of the split reached the surface."""

    path: Path
    surface_sha256: str
    scanned: Mapping[str, int]
    reference: LadderReference


def build_ladder_reference(
    resolved_config: ResolvedConfig[LadderReferenceConfig],
    output_directory: str | Path,
    *,
    workers: int = 1,
) -> LadderReferenceResult:
    """Count the configured split's games onto the grid and write the surface."""

    try:
        return _build(resolved_config, Path(output_directory), max(workers, 1))
    except DataLoadingError as error:
        raise LadderReferenceError(str(error)) from error


def load_ladder_reference(
    directory: str | Path,
    *,
    expected_surface_sha256: str | None = None,
) -> LadderReference:
    """Load a built surface, re-deriving its identity from what it holds."""

    path = Path(directory) / LADDER_REFERENCE_FILE_NAME
    if not path.is_file():
        raise LadderReferenceError(f"ladder reference does not exist: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("version") != LADDER_REFERENCE_VERSION:
        raise LadderReferenceError(
            f"{path} was built as ladder reference version "
            f"{record.get('version')}; expected {LADDER_REFERENCE_VERSION}"
        )
    observed = _surface_sha256(record["selection"], record["counts"])
    if observed != record["surface_sha256"]:
        raise LadderReferenceError(
            f"{path} does not hold the surface it records: expected "
            f"{record['surface_sha256']}, observed {observed}"
        )
    if expected_surface_sha256 is not None and observed != expected_surface_sha256:
        raise LadderReferenceError(
            f"{path} is not the reference this selection reads: expected "
            f"{expected_surface_sha256}, observed {observed}"
        )
    return _reference(record["reference_id"], observed, record)


def _build(
    resolved_config: ResolvedConfig[LadderReferenceConfig],
    output_path: Path,
    workers: int,
) -> LadderReferenceResult:
    config = resolved_config.value
    source_paths = normalized_shard_paths(config.normalized)
    manifest_path = Path(config.manifest)
    if not manifest_path.is_file():
        raise LadderReferenceError(f"source manifest does not exist: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    source_manifest = json.loads(manifest_bytes)
    validate_manifest_compatibility(source_manifest, manifest_path)
    try:
        snapshot = snapshot_for_corpus(
            config.marked_accounts,
            resolved_config.provenance.source,
            source_manifest,
            manifest_path,
        )
    except MarkedAccountError as error:
        raise LadderReferenceError(str(error)) from error
    marked = frozenset[int]() if snapshot is None else snapshot.accounts.row_digests()

    logger.info(
        "Counting the %s split of %s shard(s) onto a %s-rating grid, %s at a time",
        config.split,
        len(source_paths),
        len(config.target_ratings),
        workers,
    )
    rule = _ScanRule.of(config, marked)
    size = len(config.target_ratings)
    counts = [[[0] * size for _ in range(size)] for _ in _COUNT_PLANES]
    scanned: dict[str, int] = {}
    for shard in _scan_shards(source_paths, rule, workers):
        for plane, shard_plane in zip(counts, shard.counts, strict=True):
            for row, shard_row in zip(plane, shard_plane, strict=True):
                for index, count in enumerate(shard_row):
                    row[index] += count
        for name, count in shard.tallies.items():
            scanned[name] = scanned.get(name, 0) + count

    if not scanned.get("surface_games"):
        raise LadderReferenceError(
            f"no {config.speed} game of the {config.split} split in "
            f"{config.rating_namespace} has both players on the rating grid"
        )

    selection = {
        "split": config.split,
        "speed": config.speed.value,
        "rating_namespace": config.rating_namespace,
        "target_ratings": list(config.target_ratings),
        "rating_window": config.rating_window,
        "excluded_terminations": [
            category.value for category in config.excluded_terminations
        ],
        "minimum_games": config.minimum_games,
        "marked_accounts_sha256": None if snapshot is None else snapshot.sha256,
    }
    count_record = dict(zip(_COUNT_PLANES, counts, strict=True))
    identity = _surface_sha256(selection, count_record)
    if (
        config.expected_surface_sha256 is not None
        and identity != config.expected_surface_sha256
    ):
        raise LadderReferenceError(
            "rebuilt ladder reference does not match its expected identity: "
            f"expected {config.expected_surface_sha256}, observed {identity}"
        )

    tallies = dict(sorted(scanned.items()))
    record = {
        "version": LADDER_REFERENCE_VERSION,
        "reference_id": config.reference_id,
        "surface_sha256": identity,
        "selection": selection,
        "counts": count_record,
        "scanned": tallies,
        "source": {
            "manifest_sha256": sha256(manifest_bytes).hexdigest(),
            "source": source_manifest.get("source"),
            "split": source_manifest.get("split"),
        },
        "marked_accounts": None if snapshot is None else snapshot.as_record(),
        "resolved_config": resolved_config.as_record(),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / LADDER_REFERENCE_FILE_NAME
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Counted %s game(s) onto the grid into %s", scanned["surface_games"], path
    )
    return LadderReferenceResult(
        path=path,
        surface_sha256=identity,
        scanned=tallies,
        reference=_reference(config.reference_id, identity, record),
    )


def _surface_sha256(
    selection: Mapping[str, Any],
    counts: Mapping[str, Any],
) -> str:
    content = {
        "version": LADDER_REFERENCE_VERSION,
        "selection": selection,
        "counts": counts,
    }
    return sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reference(
    reference_id: str,
    surface_sha256: str,
    record: Mapping[str, Any],
) -> LadderReference:
    ratings = tuple(int(rating) for rating in record["selection"]["target_ratings"])
    minimum = int(record["selection"]["minimum_games"])
    white_wins, draws, black_wins = (record["counts"][plane] for plane in _COUNT_PLANES)
    cells: dict[tuple[int, int], ReferenceCell] = {}
    for higher in range(len(ratings)):
        for lower in range(higher):
            # The higher-rated player as white, then as black, each as
            # (their wins, draws, games).
            colours = (
                (
                    white_wins[higher][lower],
                    draws[higher][lower],
                    white_wins[higher][lower]
                    + draws[higher][lower]
                    + black_wins[higher][lower],
                ),
                (
                    black_wins[lower][higher],
                    draws[lower][higher],
                    white_wins[lower][higher]
                    + draws[lower][higher]
                    + black_wins[lower][higher],
                ),
            )
            score: float | None = None
            standard_error: float | None = None
            if all(games >= minimum for _, _, games in colours):
                readings = [_colour_score(*colour) for colour in colours]
                score = sum(mean for mean, _ in readings) / 2
                standard_error = math.sqrt(sum(se**2 for _, se in readings)) / 2
            cells[(ratings[higher], ratings[lower])] = ReferenceCell(
                higher_rating=ratings[higher],
                lower_rating=ratings[lower],
                games=sum(games for _, _, games in colours),
                score=score,
                standard_error=standard_error,
            )
    return LadderReference(
        reference_id=reference_id,
        surface_sha256=surface_sha256,
        target_ratings=ratings,
        cells=cells,
    )


def _colour_score(wins: int, draws: int, games: int) -> tuple[float, float]:
    """Return one colour's mean score and its standard error.

    The variance is the games' own, since a draw scores a half and a binomial
    variance would overstate it wherever players draw.
    """

    score = (wins + draws / 2) / games
    second_moment = (wins + draws / 4) / games
    variance = max(second_moment - score**2, 0.0) * games / (games - 1)
    return score, math.sqrt(variance / games)


@dataclass(frozen=True)
class _ScanRule:
    """What deciding one game's cell needs, installed once per scan worker."""

    split: str
    speed: Speed
    rating_namespace: str
    excluded_terminations: frozenset[str]
    marked_digests: frozenset[int]
    grid_index: Mapping[int, int]
    size: int

    @classmethod
    def of(cls, config: LadderReferenceConfig, marked: frozenset[int]) -> _ScanRule:
        window = config.rating_window
        return cls(
            split=config.split,
            speed=config.speed,
            rating_namespace=config.rating_namespace,
            excluded_terminations=frozenset(
                category.value for category in config.excluded_terminations
            ),
            marked_digests=marked,
            grid_index={
                rating: index
                for index, target in enumerate(config.target_ratings)
                for rating in range(target - window, target + window + 1)
            },
            size=len(config.target_ratings),
        )


@dataclass(frozen=True)
class _ShardCounts:
    counts: list[list[list[int]]]
    tallies: dict[str, int]


_SCAN_RULE: _ScanRule | None = None


def _scan_shards(
    paths: Sequence[Path],
    rule: _ScanRule,
    workers: int,
) -> Iterator[_ShardCounts]:
    if workers == 1:
        _init_scan(rule)
        yield from (_scan_shard(path) for path in paths)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_scan,
        initargs=(rule,),
    ) as pool:
        yield from pool.map(_scan_shard, paths)


def _init_scan(rule: _ScanRule) -> None:
    global _SCAN_RULE
    _SCAN_RULE = rule


def _scan_shard(path: Path) -> _ShardCounts:
    rule = _SCAN_RULE
    if rule is None:  # pragma: no cover - the initializer precedes every task
        raise LadderReferenceError("a reference scan worker was started without a rule")
    counts = [[[0] * rule.size for _ in range(rule.size)] for _ in _COUNT_PLANES]
    tallies = {
        "split_games": 0,
        "selected_games": 0,
        "excluded_termination_games": 0,
        "marked_games": 0,
        "surface_games": 0,
    }
    shard = open_normalized_shard(path)
    for row_group in range(normalized_row_group_count(shard)):
        table = read_normalized_row_group(shard, row_group, _SCAN_COLUMNS)
        positions = [
            position
            for position, split in enumerate(
                row_group_column(table, NormalizedColumn.SPLIT)
            )
            if split == rule.split
        ]
        tallies["split_games"] += len(positions)
        if not positions:
            continue
        held_out = take_rows(table, positions)
        columns = [row_group_column(held_out, column) for column in _SCAN_COLUMNS[1:]]
        for (
            result,
            termination,
            white_rating,
            black_rating,
            initial_ms,
            increment_ms,
            namespace,
            white_digest,
            black_digest,
        ) in zip(*columns, strict=True):
            if (
                namespace != rule.rating_namespace
                or speed_from_clock_ms(initial_ms, increment_ms) != rule.speed
                or result not in _OUTCOMES
            ):
                continue
            tallies["selected_games"] += 1
            if termination in rule.excluded_terminations:
                tallies["excluded_termination_games"] += 1
                continue
            if (
                white_digest in rule.marked_digests
                or black_digest in rule.marked_digests
            ):
                tallies["marked_games"] += 1
                continue
            white = rule.grid_index.get(white_rating)
            black = rule.grid_index.get(black_rating)
            if white is None or black is None or white == black:
                continue
            counts[_OUTCOMES[result]][white][black] += 1
            tallies["surface_games"] += 1
    return _ShardCounts(counts, tallies)
