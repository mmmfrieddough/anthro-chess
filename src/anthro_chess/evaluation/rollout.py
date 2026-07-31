"""What whole generated games look like, across a declared rollout matrix.

Offline prediction metrics score positions a human already reached. This plays
new games instead, which is the only way to see the behaviors that only exist
across a whole trajectory: how long games run, how they end, and whether the
model falls into cycles it can never be shown falling into by a per-position
metric.

The suite is a matrix rather than a run. One trajectory proves nothing — a
single deterministic game is one sample of a distribution — so the core axes
are explicit seeds, both color assignments, and independent rating and
temperature grids. Each grid cell is its own measurement and its own stored
result; the seeds inside a cell are replicates of it.

Two position sources, which the harness treats identically: the standard
starting position, and frozen human prefixes projected out of the evaluation
pool through the shared view layer. Human prefixes matter most early, when a
checkpoint may not yet build a coherent game from move one but can still be
asked what it does with a real opening.

The arms are not interchangeable for every reading. On the prefix arm the
opening distribution is a property of the *view*, since the prefix already
decided the opening before the model moved: measured on a real checkpoint, the
prefix arm reported the identical opening counts at every rating. Repertoire is
therefore only a statement about the model on the standard-start arm. The prefix
arm's opening labels are there to slice its other readings by opening, not to be
read as the model's choices.

Nothing here waits in wall-clock time or draws from an uncontrolled source. A
cell reproduces exactly from its declared seeds, and a single game reproduces on
its own from the seed its record carries.

Series identity is the declared generation recipe, not the games generated.
Rating, temperature, ply limit, and position source say what was measured;
how many games were played says only how precisely. That split is decision
0020, which generalizes the workload scoping decision 0018 introduced for
efficiency. Metrics reach the committed tier; the games themselves are bulk
diagnostics and stay in the machine-local detail tier, where a later analysis
pass can recompute new features over them without replaying anything.

A run produces two kinds of record, because it measures two kinds of thing. A
**cell** is one point of the matrix and carries the raw rollout scalars: how
long its games ran, how they ended, how much they repeated. A **reading** spans
one arm's whole rating grid at one temperature and carries the distances against
matched human play.

The reading is what makes any of this a verdict. A draw rate or a game length
has no right value the project is willing to name, so those scalars stay
informational; the distance to humans of the same rating does have a direction,
because playing like a human is the goal. The comparison uses the shape in
``anthro_chess.evaluation.curves``, whose bandwidth is selected once from the
real corpus and frozen in ``anthro_chess.evaluation.reference`` rather than
chosen per run — re-selecting it would mean two checkpoints were measured
differently.

The grid is the curve's axis, which is why a reading is not per cell: a single
cell has one rating and therefore no curve. It also means a sparse rating grid
leaves evaluation points with no generated game nearby; those drop out of the
conditional reading rather than being interpolated, and the count is reported so
a narrow reading is not mistaken for a complete one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from statistics import stdev
from typing import Annotated, Any

import torch
from pydantic import Field, StrictBool, StrictInt, model_validator

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data.artifacts import DataLoadingError, read_normalized_rows
from anthro_chess.data.schema import NormalizedColumn
from anthro_chess.evaluation.curves import (
    CurveComparison,
    CurveComparisonError,
    CurveMetrics,
    compare_curves,
)
from anthro_chess.evaluation.execution import execution_record
from anthro_chess.evaluation.games import (
    GAME_ANALYSIS_VERSION,
    GENERATION_VERSION,
    GameDistribution,
    GameFeatures,
    GameRecord,
    GameTermination,
    GenerationConfig,
    GenerationError,
    ModelPlayer,
    PlayerError,
    StartPosition,
    analyze_games,
    generate_games,
    prefix_positions,
    standard_positions,
    summarize_games,
)
from anthro_chess.evaluation.openings import (
    OpeningBook,
    OpeningBookError,
    OpeningLevel,
    load_book,
)
from anthro_chess.evaluation.pool import (
    EvaluationPoolError,
    FrozenPool,
    load_pool,
)
from anthro_chess.evaluation.reference import (
    CURVE_SPEC_VERSION,
    DECLARED_NEIGHBOURS,
    ComparableGame,
    ComparedQuantity,
    HumanReference,
    ReferenceConfig,
    ReferenceError,
    curve_spec,
    generated_games,
    human_reference,
    iter_quantities,
)
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DatasetReference,
    DetailReference,
    DetailStore,
    ExecutionRecord,
    Measurement,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    ResultsStoreError,
    build_result,
    configuration_reference,
    dataset_reference,
    default_checkpoint_label,
    measurement,
)
from anthro_chess.evaluation.results.fingerprints import (
    FingerprintError,
    projection_content_digest,
)
from anthro_chess.evaluation.results.metrics import (
    GENERATED_PLAY_CONDITIONAL_DISTANCE,
    GENERATED_PLAY_DECISIVE_GAME_RATE,
    GENERATED_PLAY_DISTINCT_GAME_FRACTION,
    GENERATED_PLAY_MEAN_CYCLE_PLY_FRACTION,
    GENERATED_PLAY_MEAN_DISTINCT_MOVE_FRACTION,
    GENERATED_PLAY_MEAN_FIRST_REPETITION_PLY,
    GENERATED_PLAY_MEAN_GAME_PLIES,
    GENERATED_PLAY_MEAN_GENERATED_PLIES,
    GENERATED_PLAY_POOLED_DISTANCE,
    GENERATED_PLAY_RATING_VARIATION,
    GENERATED_PLAY_REPEATED_POSITION_GAME_RATE,
    GENERATED_PLAY_RESIGNATION_RATE,
    GENERATED_PLAY_THREEFOLD_CLAIMABLE_GAME_RATE,
    GENERATED_PLAY_UNFINISHED_GAME_RATE,
    GENERATED_PLAY_WHITE_SCORE,
    MOVE_PREDICTION_PROJECTION,
)
from anthro_chess.evaluation.results.noise import floor_from_dispersion
from anthro_chess.evaluation.views import ViewConfig, ViewSelection, apply_view
from anthro_chess.inference import CheckpointModelRunner, ModelRunnerConfig
from anthro_chess.inference.runner import ModelRunnerError
from anthro_chess.runtime import ActionModelRunner, RuntimeConfig

ROLLOUT_BENCHMARK_VERSION = 1

ROLLOUT_KIND = "generated-play"
ROLLOUT_BENCHMARK = BenchmarkReference(
    name="generated-play-rollout",
    version=ROLLOUT_BENCHMARK_VERSION,
)

logger = logging.getLogger(__name__)


class RolloutBenchmarkError(ValueError):
    """Raised when generated play cannot be measured as configured."""


class RolloutArm(StrEnum):
    """Where the games of one cell start.

    Two arms rather than two benchmarks: they answer the same questions about
    the same checkpoint and differ only in the position source, which is one
    field of the declared workload. Keeping them in one run is also what makes
    them comparable, since they then share a checkpoint, a grid, and a seed
    derivation.
    """

    #: Whole games from the standard starting position.
    STANDARD_START = "standard-start"
    #: Continuations of frozen human openings drawn from the evaluation pool.
    HUMAN_PREFIX = "human-prefix"


class RolloutGridConfig(ConfigModel):
    """The matrix a rollout suite plays.

    Rating and temperature are independent axes because they are independent
    dials: decision 0008 keeps temperature out of the rating scale, and a grid
    that moved them together could not tell a rating effect from a sampling
    one.
    """

    #: Conditioning ratings to play at. Each is its own series.
    target_ratings: tuple[StrictInt, ...] = (1500,)
    #: Sampling temperatures to play at. Each is its own series.
    temperatures: tuple[float, ...] = (1.0,)
    #: Base seeds, each producing a whole independent suite inside its cell.
    #: Several is the point: one seed cannot distinguish a deterministic
    #: trajectory from stable behavior, and the spread across seeds is what a
    #: later evaluation-noise characterization reads.
    seeds: tuple[StrictInt, ...] = (0, 1, 2)

    @model_validator(mode="after")
    def _validate_axes(self) -> RolloutGridConfig:
        for name, values in (
            ("target_ratings", self.target_ratings),
            ("temperatures", self.temperatures),
            ("seeds", self.seeds),
        ):
            if not values:
                raise ValueError(f"a rollout grid needs at least one {name} value")
            if len(set(values)) != len(values):
                raise ValueError(f"a rollout grid must not repeat a {name} value")
        if any(rating < 0 for rating in self.target_ratings):
            raise ValueError("a conditioning rating cannot be negative")
        if any(not 0.0 <= value <= 3.0 for value in self.temperatures):
            raise ValueError("a rollout temperature must be between zero and three")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("a rollout seed cannot be negative")
        return self


class RolloutPrefixConfig(ConfigModel):
    """Where frozen human prefixes come from, and how deep they run."""

    view: ViewConfig = ViewConfig(name="rollout-prefix", maximum_games=32)
    #: How many plies of each source game are replayed before the seats decide.
    #: Twelve is a real opening rather than a first move, and shallow enough
    #: that a mid-training checkpoint still has a game to play.
    plies: Annotated[StrictInt, Field(ge=1)] = 12


class RolloutReferenceConfig(ReferenceConfig):
    """Which human games the comparison is measured against, and how many.

    A separate view from the prefix one because the two want opposite things.
    Prefixes are a handful of openings to continue; the reference is as much
    matched human play as can be afforded, since it forces the bandwidth and
    the noise floors at every evaluation point.
    """

    view: ViewConfig = ViewConfig(name="rollout-reference", require_ratings=True)


class RolloutDetailConfig(ConfigModel):
    """What the machine-local detail tier keeps beside a committed summary."""

    #: Whole game records. Large, and the input to every later distribution
    #: feature: recomputing a feature over retained games is seconds where
    #: regenerating them is hours. Retained by default for that reason.
    retain_games: StrictBool = True


class RolloutBenchmarkConfig(ConfigModel):
    """Code-owned schema for ``anthro eval rollout``."""

    model: ModelRunnerConfig = ModelRunnerConfig()
    checkpoint_label: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    #: The base runtime settings every seat plays under. Rating, temperature,
    #: and seed are supplied per cell and per game, so setting them here would
    #: be overridden; the rest, such as whether resignation is enabled, applies
    #: to the whole suite and joins the declared workload.
    runtime: RuntimeConfig = RuntimeConfig()
    grid: RolloutGridConfig = RolloutGridConfig()
    generation: GenerationConfig = GenerationConfig()
    arms: tuple[RolloutArm, ...] = (RolloutArm.STANDARD_START,)
    #: A frozen evaluation pool. The human-prefix arm reads its openings from
    #: it and the human-reference comparison reads its reference from it, so
    #: either one without a pool is a configuration error rather than a silent
    #: skip.
    pool: Path | None = None
    prefix: RolloutPrefixConfig = RolloutPrefixConfig()
    reference: RolloutReferenceConfig = RolloutReferenceConfig()
    #: Granularity the opening distribution is aggregated at. Family groups
    #: transpositions together, which literal move prefixes cannot.
    opening_level: OpeningLevel = OpeningLevel.FAMILY
    detail: RolloutDetailConfig = RolloutDetailConfig()

    @model_validator(mode="after")
    def _validate_arms(self) -> RolloutBenchmarkConfig:
        if not self.arms:
            raise ValueError("a rollout suite needs at least one arm")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("a rollout suite must not repeat an arm")
        if self.pool is None:
            if RolloutArm.HUMAN_PREFIX in self.arms:
                raise ValueError(
                    "the human-prefix arm needs pool to point at a frozen "
                    "evaluation pool"
                )
            if self.reference.enabled:
                raise ValueError(
                    "comparing generated play against human play needs pool to "
                    "point at a frozen evaluation pool; set "
                    "reference.enabled to false to record the rollout scalars "
                    "alone"
                )
        return self


@dataclass(frozen=True)
class RolloutCell:
    """One measured point of the matrix: an arm at a rating and temperature."""

    arm: RolloutArm
    target_rating: int
    temperature: float
    #: Positions the cell's games started from, before color assignment.
    positions: int
    seeds: tuple[int, ...]
    #: Each seed's own distribution, so the spread across seeds is readable as
    #: this cell's evaluation noise rather than having to be regenerated.
    per_seed: tuple[tuple[int, GameDistribution], ...]
    #: The cell's reading, pooled over every seed.
    distribution: GameDistribution
    execution: ExecutionRecord
    #: Per-game features, always kept: the human-reference curve reads one
    #: observation per game from them, so discarding them would make the
    #: comparison depend on whether whole games happened to be retained.
    features: tuple[GameFeatures, ...] = field(default=(), repr=False)
    #: The same features grouped by the seed that produced them. A curve's
    #: floor is estimated by bootstrapping the generated games; the spread of
    #: the distance across independent seeds is the quantity that bootstrap
    #: approximates, so keeping the grouping lets the two be compared instead
    #: of the estimate being trusted on its own.
    features_by_seed: tuple[tuple[int, tuple[GameFeatures, ...]], ...] = field(
        default=(), repr=False
    )
    #: The games behind the reading, retained only when the detail tier is
    #: keeping them. Bulk diagnostics: they never reach the committed summary,
    #: and every later distribution feature is recomputed from them rather than
    #: by regenerating the suite.
    records: tuple[GameRecord, ...] = field(default=(), repr=False)

    @property
    def label(self) -> str:
        """Return a short human label for this cell."""

        return (
            f"{self.arm.value} rating={self.target_rating} "
            f"temperature={self.temperature:g}"
        )

    def as_record(self) -> dict[str, Any]:
        """Return the cell record stored in the detail tier."""

        return {
            "arm": self.arm.value,
            "target_rating": self.target_rating,
            "temperature": self.temperature,
            "positions": self.positions,
            "seeds": list(self.seeds),
            "workload_sha256": self.execution.workload_sha256,
            "workload": dict(self.execution.workload),
            "distribution": self.distribution.as_record(),
            "per_seed": [
                {"seed": seed, "distribution": distribution.as_record()}
                for seed, distribution in self.per_seed
            ],
        }


@dataclass(frozen=True)
class SeedSpread:
    """One quantity's distance measured independently at each seed.

    Independent replicates of the same measurement, which is the definition of
    evaluation noise. Two seeds are enough to report the values and too few to
    derive a floor from, so the floor is absent below three rather than
    invented.

    Recorded as a diagnostic rather than used as a floor, and the distinction
    matters. Each seed plays only its share of the suite's games, so a per-seed
    distance is a smaller-sample reading: it is both noisier than the pooled one
    and biased away from the reference, since a distributional distance
    estimated from few games per rating point reads high. Comparing this spread
    against the pooled reading's bootstrap floor compares two different sample
    sizes and makes the bootstrap look about the square root of the seed count
    too narrow.

    What it is good for is showing whether the seeds disagree wildly, which no
    single-run estimate can reveal, and it is the raw material for a proper
    like-for-like check.
    """

    distances: tuple[tuple[int, float], ...]
    pooled: tuple[tuple[int, float], ...] = ()

    @property
    def floor(self) -> float | None:
        """Return the delta floor the observed conditional spread implies."""

        return self._floor(self.distances)

    @property
    def pooled_floor(self) -> float | None:
        """Return the delta floor the observed pooled spread implies."""

        return self._floor(self.pooled)

    @staticmethod
    def _floor(values: Sequence[tuple[int, float]]) -> float | None:
        if len(values) < 3:
            return None
        return floor_from_dispersion(stdev([value for _, value in values]))

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one quantity's seed spread."""

        return {
            "distances": [
                {"seed": seed, "conditional_distance": value}
                for seed, value in self.distances
            ],
            "pooled": [
                {"seed": seed, "pooled_distance": value} for seed, value in self.pooled
            ],
            "floor": self.floor,
            "pooled_floor": self.pooled_floor,
        }


@dataclass(frozen=True)
class RolloutReading:
    """One arm's whole rating grid, compared against matched human play.

    The unit is deliberately not the cell. A curve's axis *is* the rating, so a
    single cell has one x value and no curve at all; the reading spans every
    rating of one arm at one temperature. Temperature stays fixed across it
    because it is a separate dial rather than a point on this axis, and mixing
    two temperatures into one curve would report the sampling setting as a
    rating effect.
    """

    arm: RolloutArm
    temperature: float
    ratings: tuple[int, ...]
    model_games: int
    human_games: int
    comparisons: dict[ComparedQuantity, CurveComparison]
    execution: ExecutionRecord
    #: Each seed's own conditional distance per quantity, and the floor that
    #: spread implies. The bootstrap floor carried on every measurement is an
    #: estimate of this quantity; recording both is what makes it checkable
    #: rather than trusted.
    seed_spread: dict[ComparedQuantity, SeedSpread] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Return a short human label for this reading."""

        return f"{self.arm.value} temperature={self.temperature:g}"

    def as_record(self) -> dict[str, Any]:
        """Return the curve payload stored in the detail tier."""

        return {
            "arm": self.arm.value,
            "temperature": self.temperature,
            "ratings": list(self.ratings),
            "model_games": self.model_games,
            "human_games": self.human_games,
            "workload_sha256": self.execution.workload_sha256,
            "workload": dict(self.execution.workload),
            "seed_spread": {
                quantity.value: spread.as_record()
                for quantity, spread in self.seed_spread.items()
            },
            "comparisons": {
                quantity.value: comparison.as_detail_record()
                for quantity, comparison in self.comparisons.items()
            },
        }


@dataclass(frozen=True)
class RolloutBenchmarkResult:
    """Everything one rollout suite measured, and where it was written."""

    checkpoint: CheckpointReference
    cells: tuple[RolloutCell, ...]
    #: The prefix view a human-prefix arm selected, absent when no arm used one.
    view: ViewSelection | None
    dataset: DatasetReference | None
    #: One per arm and temperature, absent when no human reference was read.
    readings: tuple[RolloutReading, ...] = ()
    reference: HumanReference | None = None
    reference_view: ViewSelection | None = None
    envelopes: tuple[ResultEnvelope, ...] = ()
    recorded_paths: tuple[Path, ...] = ()
    detail_paths: tuple[Path, ...] = ()

    @property
    def games(self) -> int:
        """Return how many games the whole suite played."""

        return sum(cell.distribution.games for cell in self.cells)

    def cell(self, arm: RolloutArm, rating: int, temperature: float) -> RolloutCell:
        """Return one measured cell of the matrix."""

        for candidate in self.cells:
            if (
                candidate.arm is arm
                and candidate.target_rating == rating
                and candidate.temperature == temperature
            ):
                return candidate
        raise RolloutBenchmarkError(
            f"no cell was measured for {arm.value} at rating {rating} and "
            f"temperature {temperature:g}"
        )

    def as_record(self) -> dict[str, Any]:
        """Return the full structured result, detail tier included."""

        return {
            "version": ROLLOUT_BENCHMARK_VERSION,
            "generation_version": GENERATION_VERSION,
            "analysis_version": GAME_ANALYSIS_VERSION,
            "checkpoint": self.checkpoint.model_dump(mode="json"),
            "view": None if self.view is None else self.view.as_record(),
            "dataset": (
                None if self.dataset is None else self.dataset.model_dump(mode="json")
            ),
            "games": self.games,
            "cells": [cell.as_record() for cell in self.cells],
            "reference": (
                None if self.reference is None else self.reference.as_record()
            ),
            "reference_view": (
                None if self.reference_view is None else self.reference_view.as_record()
            ),
            "readings": [reading.as_record() for reading in self.readings],
            "recorded": [str(path) for path in self.recorded_paths],
        }

    def reading(self, arm: RolloutArm, temperature: float) -> RolloutReading:
        """Return one arm's curve reading at one temperature."""

        for candidate in self.readings:
            if candidate.arm is arm and candidate.temperature == temperature:
                return candidate
        raise RolloutBenchmarkError(
            f"no curve reading was measured for {arm.value} at temperature "
            f"{temperature:g}"
        )


@dataclass(frozen=True)
class _PositionSource:
    """One arm's resolved roots, and how to describe them in a workload."""

    arm: RolloutArm
    positions: tuple[StartPosition, ...]
    identity: dict[str, Any]


def benchmark_rollout(
    resolved_config: ResolvedConfig[RolloutBenchmarkConfig],
    *,
    run_root: Path | None = None,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
    runner: ActionModelRunner | None = None,
    checkpoint: CheckpointReference | None = None,
) -> RolloutBenchmarkResult:
    """Play the configured matrix and report what the generated games look like.

    Passing no ``store`` measures everything and records nothing, which is what
    an exploratory reading wants: a rollout taken to look at one temperature is
    real but does not belong in the committed history.

    A ``runner`` may be supplied to measure an already-loaded checkpoint, in
    which case ``checkpoint`` identifies it. Otherwise both are resolved from
    the configuration.
    """

    config = resolved_config.value
    loaded, identity = _resolve_model(config, runner, checkpoint, run_root)
    book = _load_book()
    sources, view, dataset = _position_sources(config, book=book)

    cells: list[RolloutCell] = []
    for source in sources:
        for target_rating in config.grid.target_ratings:
            for temperature in config.grid.temperatures:
                cells.append(
                    _measure_cell(
                        config,
                        loaded,
                        identity,
                        source,
                        book=book,
                        target_rating=target_rating,
                        temperature=temperature,
                    )
                )

    reference: HumanReference | None = None
    reference_view: ViewSelection | None = None
    readings: tuple[RolloutReading, ...] = ()
    if config.reference.enabled:
        reference, reference_view = _load_reference(config, book=book)
        readings = _curve_readings(
            config,
            cells,
            reference,
            device=_device(loaded),
        )

    result = RolloutBenchmarkResult(
        checkpoint=identity,
        cells=tuple(cells),
        view=view,
        dataset=dataset,
        readings=readings,
        reference=reference,
        reference_view=reference_view,
    )
    return _record(result, resolved_config, store=store, detail=detail)


def _measure_cell(
    config: RolloutBenchmarkConfig,
    runner: ActionModelRunner,
    checkpoint: CheckpointReference,
    source: _PositionSource,
    *,
    book: OpeningBook | None,
    target_rating: int,
    temperature: float,
) -> RolloutCell:
    """Play every seed of one cell and pool them into its reading."""

    runtime = config.runtime.model_copy(
        update={"target_rating": target_rating, "temperature": temperature}
    )
    player = ModelPlayer(
        runner,
        label=f"{checkpoint.label}-r{target_rating}-t{temperature:g}",
        config=runtime,
        checkpoint=checkpoint,
    )
    features: list[GameFeatures] = []
    per_seed: list[tuple[int, GameDistribution]] = []
    by_seed: list[tuple[int, tuple[GameFeatures, ...]]] = []
    records: list[GameRecord] = []
    for seed in config.grid.seeds:
        generation = config.generation.model_copy(update={"seed": seed})
        played = _generate(player, source.positions, generation)
        seed_features = analyze_games(played, book=book)
        per_seed.append(
            (seed, summarize_games(seed_features, level=config.opening_level))
        )
        by_seed.append((seed, seed_features))
        features.extend(seed_features)
        if config.detail.retain_games:
            records.extend(played)
    cell = RolloutCell(
        arm=source.arm,
        target_rating=target_rating,
        temperature=temperature,
        positions=len(source.positions),
        seeds=tuple(config.grid.seeds),
        per_seed=tuple(per_seed),
        distribution=summarize_games(features, level=config.opening_level),
        execution=_execution_record(config, runner, source, target_rating, temperature),
        features=tuple(features),
        features_by_seed=tuple(by_seed),
        records=tuple(records),
    )
    logger.info(
        "Rollout cell %s: %s game(s), mean %.1f plies, %s unfinished",
        cell.label,
        cell.distribution.games,
        cell.distribution.mean_ply_count,
        _termination_count(cell.distribution, GameTermination.PLY_LIMIT),
    )
    return cell


def _generate(
    player: ModelPlayer,
    positions: Sequence[StartPosition],
    generation: GenerationConfig,
) -> tuple[GameRecord, ...]:
    """Play one seed's suite as self-play, with one configuration in both seats.

    Both seats are the same player, which is what makes this a statement about
    one configuration rather than about a matchup. The harness has no notion of
    self-play; it is simply the same configuration passed twice.
    """

    try:
        return tuple(generate_games(player, player, positions, config=generation))
    except (GenerationError, PlayerError) as error:
        raise RolloutBenchmarkError(f"cannot generate games: {error}") from error


def _position_sources(
    config: RolloutBenchmarkConfig,
    *,
    book: OpeningBook | None,
) -> tuple[
    tuple[_PositionSource, ...],
    ViewSelection | None,
    DatasetReference | None,
]:
    """Resolve every arm's roots, loading the pool at most once."""

    view: ViewSelection | None = None
    dataset: DatasetReference | None = None
    sources: list[_PositionSource] = []
    for arm in config.arms:
        if arm is RolloutArm.STANDARD_START:
            sources.append(
                _PositionSource(
                    arm=arm,
                    positions=standard_positions(label=arm.value),
                    identity={"kind": arm.value},
                )
            )
            continue
        positions, view, dataset = _load_prefix_positions(config, book=book)
        sources.append(
            _PositionSource(
                arm=arm,
                positions=positions,
                identity={
                    "kind": arm.value,
                    "prefix_plies": config.prefix.plies,
                    "pool_id": dataset.pool_id,
                    "pool_version": dataset.pool_version,
                    "view": dataset.view,
                    "game_ids_sha256": dataset.game_ids_sha256,
                },
            )
        )
    return tuple(sources), view, dataset


def _load_prefix_positions(
    config: RolloutBenchmarkConfig,
    *,
    book: OpeningBook | None,
) -> tuple[tuple[StartPosition, ...], ViewSelection, DatasetReference]:
    """Project the selected pool games onto the roots the seats continue from."""

    pool = _load_pool(config)
    # The prefix depth is this benchmark's dial rather than the pool's, so it
    # is pushed into the view: a game shorter than the prefix is excluded by
    # the same mechanism that excludes every other ineligible game, and the
    # exclusion is recorded in the view spec instead of being dropped silently.
    view_config = config.prefix.view.model_copy(
        update={"prefix_plies": config.prefix.plies}
    )
    selection = apply_view(pool.games, view_config)
    if not selection.game_ids:
        raise RolloutBenchmarkError(
            f"view {view_config.name!r} selected no games from the pool"
        )

    wanted = set(selection.game_ids)
    try:
        rows = [
            _prefix_row(row, config.prefix.plies)
            for row in read_normalized_rows(pool.games_path)
            if int(row[NormalizedColumn.GAME_ID]) in wanted
        ]
    except DataLoadingError as error:
        raise RolloutBenchmarkError(str(error)) from error
    if len(rows) != len(wanted):
        raise RolloutBenchmarkError(
            "the evaluation pool does not contain every selected game"
        )
    games = sorted(
        (
            int(row[NormalizedColumn.GAME_ID.value]),
            tuple(int(value) for value in row[NormalizedColumn.ACTION_IDS.value]),
        )
        for row in rows
    )
    try:
        positions = prefix_positions(games, plies=config.prefix.plies)
    except GenerationError as error:
        raise RolloutBenchmarkError(str(error)) from error
    return positions, selection, _dataset_reference(pool, selection, rows)


def _load_pool(config: RolloutBenchmarkConfig) -> FrozenPool:
    """Load the frozen pool both the prefix arm and the reference read from."""

    if config.pool is None:  # pragma: no cover - validated on the config
        raise RolloutBenchmarkError("this suite needs a frozen evaluation pool")
    try:
        return load_pool(config.pool)
    except EvaluationPoolError as error:
        raise RolloutBenchmarkError(str(error)) from error


def _load_reference(
    config: RolloutBenchmarkConfig,
    *,
    book: OpeningBook | None,
) -> tuple[HumanReference, ViewSelection]:
    """Read the matched human play every curve is compared against."""

    pool = _load_pool(config)
    selection = apply_view(pool.games, config.reference.view)
    if not selection.game_ids:
        raise RolloutBenchmarkError(
            f"view {config.reference.view.name!r} selected no human games to "
            "compare against"
        )
    wanted = set(selection.game_ids)
    try:
        rows = [
            row
            for row in read_normalized_rows(pool.games_path)
            if int(row[NormalizedColumn.GAME_ID]) in wanted
        ]
    except DataLoadingError as error:
        raise RolloutBenchmarkError(str(error)) from error
    try:
        reference = human_reference(rows, config.reference, book=book)
    except ReferenceError as error:
        raise RolloutBenchmarkError(str(error)) from error
    return reference, selection


def _curve_readings(
    config: RolloutBenchmarkConfig,
    cells: Sequence[RolloutCell],
    reference: HumanReference,
    *,
    device: torch.device,
) -> tuple[RolloutReading, ...]:
    """Compare each arm's rating grid against matched human play.

    One reading per arm and temperature, because the curve's axis is the rating
    and temperature is a separate dial rather than a point on it.
    """

    readings: list[RolloutReading] = []
    for arm in config.arms:
        for temperature in config.grid.temperatures:
            selected = [
                cell
                for cell in cells
                if cell.arm is arm and cell.temperature == temperature
            ]
            if not selected:  # pragma: no cover - the matrix is complete
                continue
            readings.append(
                _curve_reading(
                    config, arm, temperature, selected, reference, device=device
                )
            )
    return tuple(readings)


def _curve_reading(
    config: RolloutBenchmarkConfig,
    arm: RolloutArm,
    temperature: float,
    cells: Sequence[RolloutCell],
    reference: HumanReference,
    *,
    device: torch.device,
) -> RolloutReading:
    """Compare one arm's whole rating grid at one temperature."""

    model: list[ComparableGame] = []
    for cell in cells:
        model.extend(generated_games(cell.features, rating=cell.target_rating))
    ratings = tuple(sorted({cell.target_rating for cell in cells}))
    comparisons: dict[ComparedQuantity, CurveComparison] = {}
    for quantity in iter_quantities():
        try:
            spec = curve_spec(quantity, ratings)
        except ReferenceError as error:
            raise RolloutBenchmarkError(str(error)) from error
        human = reference.observations(quantity, level=config.opening_level)
        generated = tuple(
            game.observation(quantity, level=config.opening_level) for game in model
        )
        try:
            comparisons[quantity] = compare_curves(
                spec=spec,
                human=human,
                model=generated,
                resamples=config.reference.resamples,
                seed=config.reference.seed,
            )
        except CurveComparisonError as error:
            raise RolloutBenchmarkError(
                f"cannot compare {quantity.value} against human play: {error}"
            ) from error
    return RolloutReading(
        arm=arm,
        temperature=temperature,
        ratings=ratings,
        model_games=len(model),
        human_games=len(reference.games),
        comparisons=comparisons,
        seed_spread=_seed_spread(config, cells, reference, ratings),
        execution=_reading_execution_record(
            config, cells, temperature, ratings, device=device
        ),
    )


def _seed_spread(
    config: RolloutBenchmarkConfig,
    cells: Sequence[RolloutCell],
    reference: HumanReference,
    ratings: Sequence[int],
) -> dict[ComparedQuantity, SeedSpread]:
    """Re-measure each quantity from each seed's games alone.

    The seeds are independent replicates of the same measurement, so the spread
    of a distance across them is evaluation noise directly rather than an
    estimate of it. Resampling is switched off here: only the point estimate is
    wanted, and paying for floors on every seed would multiply the cost of the
    reading for nothing.
    """

    seeds = sorted({seed for cell in cells for seed, _ in cell.features_by_seed})
    if len(seeds) < 2:
        return {}
    spreads: dict[ComparedQuantity, list[tuple[int, float]]] = {
        quantity: [] for quantity in iter_quantities()
    }
    pooled: dict[ComparedQuantity, list[tuple[int, float]]] = {
        quantity: [] for quantity in iter_quantities()
    }
    for seed in seeds:
        games: list[ComparableGame] = []
        for cell in cells:
            for candidate, features in cell.features_by_seed:
                if candidate == seed:
                    games.extend(generated_games(features, rating=cell.target_rating))
        if not games:  # pragma: no cover - every cell plays every seed
            continue
        for quantity in iter_quantities():
            try:
                comparison = compare_curves(
                    spec=curve_spec(quantity, ratings),
                    human=reference.observations(quantity, level=config.opening_level),
                    model=tuple(
                        game.observation(quantity, level=config.opening_level)
                        for game in games
                    ),
                    resamples=0,
                )
            except (CurveComparisonError, ReferenceError):
                # One seed too thin to estimate is a reason to report fewer
                # replicates, not to fail the reading the seeds support.
                continue
            spreads[quantity].append((seed, comparison.conditional_distance))
            pooled[quantity].append((seed, comparison.pooled_distance))
    return {
        quantity: SeedSpread(distances=tuple(values), pooled=tuple(pooled[quantity]))
        for quantity, values in spreads.items()
        if values
    }


def _reading_execution_record(
    config: RolloutBenchmarkConfig,
    cells: Sequence[RolloutCell],
    temperature: float,
    ratings: Sequence[int],
    *,
    device: torch.device,
) -> ExecutionRecord:
    """Declare what one curve reading measured.

    A curve's workload is a cell's with the single rating replaced by the whole
    grid, because the grid is the axis rather than a setting held fixed. The
    curve shape joins it too: a distance estimated at one bandwidth and one at
    another are not the same quantity, which is exactly why the bandwidth is
    declared and frozen rather than configured.
    """

    workload = dict(cells[0].execution.workload)
    workload.pop("target_rating", None)
    workload.update(
        {
            "target_ratings": list(ratings),
            "temperature": temperature,
            "curve_spec_version": CURVE_SPEC_VERSION,
            "curve_neighbours": {
                quantity.value: DECLARED_NEIGHBOURS[quantity]
                for quantity in iter_quantities()
            },
            "reference_maximum_rating_gap": config.reference.maximum_rating_gap,
        }
    )
    return execution_record(device, workload)


def _prefix_row(row: Mapping[str, Any], plies: int) -> dict[str, Any]:
    """Truncate one pool game to the prefix the seats were actually given.

    The truncation reaches the rows the provenance digest is computed over, so
    a twelve-ply prefix and a twenty-ply prefix over the same games do not
    claim to have read the same content.
    """

    updated = dict(row)
    values = updated.get(NormalizedColumn.ACTION_IDS.value)
    if values is not None:
        updated[NormalizedColumn.ACTION_IDS.value] = list(values)[:plies]
    return updated


def _dataset_reference(
    pool: FrozenPool,
    selection: ViewSelection,
    rows: Sequence[Mapping[str, Any]],
) -> DatasetReference:
    """Describe the human games the prefixes were projected out of.

    This is provenance, not series identity. A generated-play metric declares
    no projection, so this digest never enters a fingerprint; it is here so a
    reader can tell which human openings a rollout continued.
    """

    identity = pool.manifest.get("pool")
    if not isinstance(identity, Mapping):
        raise RolloutBenchmarkError("evaluation pool manifest has no pool identity")
    try:
        component = projection_content_digest(rows, MOVE_PREDICTION_PROJECTION)
    except FingerprintError as error:
        raise RolloutBenchmarkError(str(error)) from error
    record = selection.as_record()
    return dataset_reference(
        pool_id=str(identity["id"]),
        pool_version=int(identity["version"]),
        view=selection.name,
        selected_games=selection.selected_games,
        game_ids_sha256=str(record["game_ids_sha256"]),
        components=[component],
    )


def _execution_record(
    config: RolloutBenchmarkConfig,
    runner: ActionModelRunner,
    source: _PositionSource,
    target_rating: int,
    temperature: float,
) -> ExecutionRecord:
    """Declare what this cell measured, and record where it ran.

    The workload is what decides the quantity: the arm, the two dials, how long
    a game is allowed to run, and whether the harness settles endings the seats
    cannot. Seed count, games per position, and concurrency are deliberately
    absent. More games estimate the same distribution more precisely, and
    concurrency only changes which kernels resolve a decision, so putting
    either in identity would end a series for a throughput change.
    """

    return execution_record(
        _device(runner),
        {
            "generation_version": GENERATION_VERSION,
            "positions": source.identity,
            "target_rating": target_rating,
            "temperature": temperature,
            "maximum_generated_plies": config.generation.maximum_generated_plies,
            "swap_colors": config.generation.swap_colors,
            "claim_draws": config.generation.claim_draws,
            "resignation_enabled": config.runtime.resignation_enabled,
            "opening_level": config.opening_level.value,
        },
    )


def _record(
    result: RolloutBenchmarkResult,
    resolved_config: ResolvedConfig[RolloutBenchmarkConfig],
    *,
    store: ResultsStore | None,
    detail: DetailStore | None,
) -> RolloutBenchmarkResult:
    """Write one envelope per cell, plus one per curve reading.

    Two kinds of record from one run, because they are two units. A cell is one
    point of the matrix and carries the raw rollout scalars; a reading spans one
    arm's whole rating grid and carries the distances against human play. They
    have different workloads and so different series, which is why neither can
    stand in for the other.
    """

    configuration = configuration_reference(
        resolved_config.as_record(),
        source=resolved_config.provenance.source,
        overrides=resolved_config.provenance.overrides,
    )
    recorded_at = datetime.now(tz=UTC)
    detail_paths: list[Path] = []
    envelopes: list[ResultEnvelope] = []
    try:
        for cell in result.cells:
            detail_reference = _write_detail(
                detail,
                checkpoint=result.checkpoint,
                slug=(
                    f"{cell.arm.value}-r{cell.target_rating}-t{_slug(cell.temperature)}"
                ),
                description=f"Generated-play rollout cell: {cell.label}",
                payload=_cell_payload(cell),
                recorded_at=recorded_at,
                paths=detail_paths,
            )
            envelopes.append(
                build_result(
                    kind=ROLLOUT_KIND,
                    benchmark=ROLLOUT_BENCHMARK,
                    checkpoint=result.checkpoint,
                    configuration=configuration,
                    # Provenance rather than series identity: the pool games
                    # were an input to generation, not the content measured,
                    # so they reach the fingerprint through the workload's
                    # position source instead of through a data component.
                    data=result.dataset
                    if cell.arm is RolloutArm.HUMAN_PREFIX
                    else None,
                    execution=cell.execution,
                    measurements=_measurements(cell),
                    detail=detail_reference,
                    recorded_at=recorded_at,
                )
            )
        for reading in result.readings:
            detail_reference = _write_detail(
                detail,
                checkpoint=result.checkpoint,
                slug=f"{reading.arm.value}-curves-t{_slug(reading.temperature)}",
                description=f"Human-reference curves: {reading.label}",
                payload=_reading_payload(result, reading),
                recorded_at=recorded_at,
                paths=detail_paths,
            )
            envelopes.append(
                build_result(
                    kind=ROLLOUT_KIND,
                    benchmark=ROLLOUT_BENCHMARK,
                    checkpoint=result.checkpoint,
                    configuration=configuration,
                    # The reference view is provenance here for the same reason
                    # the prefix view is on a cell: the human games shaped what
                    # was compared against, and the workload carries identity.
                    data=result.dataset,
                    execution=reading.execution,
                    measurements=_curve_measurements(reading),
                    detail=detail_reference,
                    recorded_at=recorded_at,
                )
            )
    except ResultRecordError as error:
        raise RolloutBenchmarkError(str(error)) from error

    recorded_paths: list[Path] = []
    if store is not None:
        try:
            recorded_paths = [store.append(envelope) for envelope in envelopes]
        except (ResultRecordError, ResultsStoreError) as error:
            raise RolloutBenchmarkError(str(error)) from error

    return replace(
        result,
        envelopes=tuple(envelopes),
        recorded_paths=tuple(recorded_paths),
        detail_paths=tuple(detail_paths),
    )


def _measurements(cell: RolloutCell) -> tuple[Measurement, ...]:
    """Return one cell's committed measurements.

    Every metric is scoped by the cell's workload rather than by a view. Sample
    size is per metric rather than shared, because three of these are averaged
    over a subset: a cycle depth is estimated only from the games that repeated,
    and a decisive rate only from the games that ended. Reporting the suite's
    game count for those would overstate their precision to every reader that
    consumes it, the noise-floor layer included. A subset that is empty reports
    no sample size at all, which is honest: the value is a defined zero but
    nothing was averaged.
    """

    distribution = cell.distribution
    games = distribution.games
    repeated = distribution.repeated_games
    finished = _finished_games(distribution)
    workload = cell.execution.workload_component()
    decisive = distribution.result_counts.get(
        "1-0", 0
    ) + distribution.result_counts.get("0-1", 0)
    values: tuple[tuple[str, float, int], ...] = (
        (
            GENERATED_PLAY_MEAN_GAME_PLIES.identifier,
            distribution.mean_ply_count,
            games,
        ),
        (
            GENERATED_PLAY_MEAN_GENERATED_PLIES.identifier,
            distribution.mean_generated_plies,
            games,
        ),
        # White's score is over the games that produced a result, which is the
        # same denominator the decisive rate uses.
        (GENERATED_PLAY_WHITE_SCORE.identifier, _white_score(distribution), finished),
        # Decisiveness is a share of the games that actually ended, since an
        # unfinished game has no result to be decisive or drawn. Its own rate
        # below is what says how many those were.
        (
            GENERATED_PLAY_DECISIVE_GAME_RATE.identifier,
            _fraction(decisive, finished),
            finished,
        ),
        (
            GENERATED_PLAY_UNFINISHED_GAME_RATE.identifier,
            _fraction(
                _termination_count(distribution, GameTermination.PLY_LIMIT), games
            ),
            games,
        ),
        (
            GENERATED_PLAY_RESIGNATION_RATE.identifier,
            _fraction(
                _termination_count(distribution, GameTermination.RESIGNATION), games
            ),
            games,
        ),
        (
            GENERATED_PLAY_REPEATED_POSITION_GAME_RATE.identifier,
            _fraction(repeated, games),
            games,
        ),
        (
            GENERATED_PLAY_THREEFOLD_CLAIMABLE_GAME_RATE.identifier,
            _fraction(distribution.threefold_claimable_games, games),
            games,
        ),
        (
            GENERATED_PLAY_MEAN_FIRST_REPETITION_PLY.identifier,
            distribution.mean_first_repetition_ply,
            repeated,
        ),
        (
            GENERATED_PLAY_MEAN_CYCLE_PLY_FRACTION.identifier,
            distribution.mean_cycle_ply_fraction,
            repeated,
        ),
        (
            GENERATED_PLAY_MEAN_DISTINCT_MOVE_FRACTION.identifier,
            distribution.mean_distinct_move_fraction,
            games,
        ),
        (
            GENERATED_PLAY_DISTINCT_GAME_FRACTION.identifier,
            distribution.distinct_game_fraction,
            games,
        ),
    )
    return tuple(
        measurement(
            identifier,
            value,
            workload=workload,
            sample_size=sample_size if sample_size > 0 else None,
        )
        for identifier, value, sample_size in values
    )


def _curve_measurements(reading: RolloutReading) -> tuple[Measurement, ...]:
    """Return one reading's committed distances, each with the floor to beat.

    The floor is the comparison's own bootstrap over the games this reading
    generated, and deliberately not the spread across seeds. Both estimate
    evaluation noise, but only the bootstrap estimates it *at the sample size
    the reading was taken at*: each seed plays a fraction of the games, so the
    spread across seeds measures the noise of a much smaller reading and runs
    roughly the square root of the seed count too wide. Measured against forty
    independent draws, the bootstrap reproduces the true spread to within a few
    percent at fixed size, so the seeds stay a diagnostic rather than a floor.
    """

    workload = reading.execution.workload_component()
    measurements: list[Measurement] = []
    for quantity, comparison in reading.comparisons.items():
        measurements.extend(
            comparison.measurements(
                CurveMetrics(
                    conditional=GENERATED_PLAY_CONDITIONAL_DISTANCE[
                        quantity.value
                    ].identifier,
                    pooled=GENERATED_PLAY_POOLED_DISTANCE[quantity.value].identifier,
                    model_variation=GENERATED_PLAY_RATING_VARIATION[
                        quantity.value
                    ].identifier,
                ),
                workload=workload,
            )
        )
    return tuple(measurements)


def _write_detail(
    detail: DetailStore | None,
    *,
    checkpoint: CheckpointReference,
    slug: str,
    description: str,
    payload: Mapping[str, Any],
    recorded_at: datetime,
    paths: list[Path],
) -> DetailReference | None:
    """Write one bulk payload and return its summary-tier reference."""

    if detail is None:
        return None
    stamp = recorded_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    relative = Path(ROLLOUT_KIND) / checkpoint.label / f"{stamp}-{slug}.json"
    reference = detail.write(relative, dict(payload), description=description)
    paths.append(detail.root / relative)
    return reference


def _cell_payload(cell: RolloutCell) -> dict[str, Any]:
    """Return one cell's diagnostics, games included when they were retained."""

    payload = cell.as_record()
    payload["games_detail"] = [record.as_record() for record in cell.records]
    payload["features"] = [feature.as_record() for feature in cell.features]
    return payload


def _reading_payload(
    result: RolloutBenchmarkResult,
    reading: RolloutReading,
) -> dict[str, Any]:
    """Return one curve reading's points and the reference behind them.

    Curve points are stored as data rather than as a rendered image, so drawing
    several checkpoints against one human reference later is a query over
    payloads that already exist.
    """

    payload = reading.as_record()
    payload["reference"] = (
        None if result.reference is None else result.reference.as_record()
    )
    payload["reference_view"] = (
        None if result.reference_view is None else result.reference_view.as_record()
    )
    return payload


def _slug(value: float) -> str:
    """Return a filename-safe form of a temperature."""

    return str(value).replace(".", "_")


def _resolve_model(
    config: RolloutBenchmarkConfig,
    runner: ActionModelRunner | None,
    checkpoint: CheckpointReference | None,
    run_root: Path | None,
) -> tuple[ActionModelRunner, CheckpointReference]:
    """Return the runner to play with and the checkpoint identity to record."""

    if runner is not None:
        if checkpoint is None:
            raise RolloutBenchmarkError(
                "an explicitly supplied runner needs a checkpoint reference to "
                "record its results against"
            )
        return runner, checkpoint
    try:
        loaded = CheckpointModelRunner.load(config.model, run_root=run_root)
    except ModelRunnerError as error:
        raise RolloutBenchmarkError(str(error)) from error
    run_id = loaded.selection.run_path.name
    label = config.checkpoint_label or default_checkpoint_label(
        run_id,
        loaded.global_step,
    )
    return loaded, CheckpointReference(
        label=label,
        step=loaded.global_step,
        run_id=run_id,
        parameter_sha256=loaded.parameter_sha256(),
    )


def _load_book() -> OpeningBook:
    """Load the versioned opening book once for the whole suite."""

    try:
        return load_book()
    except OpeningBookError as error:
        raise RolloutBenchmarkError(str(error)) from error


def _device(runner: ActionModelRunner) -> torch.device:
    """Return the device a runner executes on, defaulting to the CPU.

    A stand-in runner need not carry a device. The environment half of the
    record is attribution rather than identity, so a missing device is recorded
    as the CPU rather than failing the measurement.
    """

    device = getattr(runner, "device", None)
    return device if isinstance(device, torch.device) else torch.device("cpu")


def _finished_games(distribution: GameDistribution) -> int:
    """Return how many games produced a result.

    Counted from the results rather than as the complement of the ply limit, so
    a later adjudicated ending that also has no result cannot quietly inflate
    the denominator of every rate computed over finished games.
    """

    counts = distribution.result_counts
    return counts.get("1-0", 0) + counts.get("0-1", 0) + counts.get("1/2-1/2", 0)


def _white_score(distribution: GameDistribution) -> float:
    """Return white's score per finished game, counting a draw as a half."""

    counts = distribution.result_counts
    points = counts.get("1-0", 0) + 0.5 * counts.get("1/2-1/2", 0)
    return _quotient(points, _finished_games(distribution))


def _termination_count(
    distribution: GameDistribution,
    termination: GameTermination,
) -> int:
    """Return how many games ended for one reason."""

    return distribution.termination_counts.get(termination.value, 0)


def _fraction(part: int, whole: int) -> float:
    return _quotient(part, whole)


def _quotient(part: float, whole: int) -> float:
    return part / whole if whole else 0.0


__all__ = [
    "ROLLOUT_BENCHMARK",
    "ROLLOUT_BENCHMARK_VERSION",
    "ROLLOUT_KIND",
    "RolloutArm",
    "RolloutBenchmarkConfig",
    "RolloutBenchmarkError",
    "RolloutBenchmarkResult",
    "RolloutCell",
    "RolloutDetailConfig",
    "RolloutGridConfig",
    "RolloutPrefixConfig",
    "RolloutReading",
    "RolloutReferenceConfig",
    "benchmark_rollout",
]
