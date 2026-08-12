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
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from pathlib import Path
from statistics import stdev
from typing import Annotated, Any

import chess
import torch
from pydantic import Field, StrictBool, StrictInt, model_validator

from anthro_chess.chess import MOVE_ACTION_COUNT, decode_move
from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data.schema import (
    NormalizedColumn,
    row_game_id,
)
from anthro_chess.evaluation.curves import (
    CurveComparison,
    CurveComparisonError,
    CurveMetrics,
    Observation,
    compare_curves,
    distribution_distance,
    estimate_curve,
)
from anthro_chess.evaluation.execution import execution_record, runner_device
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
    collapse_replicates,
    generate_games,
    prefix_positions,
    replicates_vary,
    standard_positions,
    summarize_games,
)
from anthro_chess.evaluation.openings import (
    ActionPolicy,
    OpeningBook,
    OpeningBookError,
    OpeningLabel,
    OpeningLevel,
    OpeningTreeError,
    RepertoireWalk,
    load_book,
    progression_for_action_ids,
    walk_repertoire,
)
from anthro_chess.evaluation.pool import (
    EvaluationPoolError,
    FrozenPool,
    PoolGenerationPin,
    load_pool,
    pool_rows,
)
from anthro_chess.evaluation.recording import (
    ResultRecording,
    pool_dataset_reference,
    resolve_model,
)
from anthro_chess.evaluation.reference import (
    CURVE_SPEC_VERSION,
    DECLARED_NEIGHBOURS,
    REFERENCE_COLUMNS,
    ComparableGame,
    ComparedQuantity,
    HumanReference,
    ReferenceConfig,
    ReferenceError,
    curve_spec,
    generated_games,
    human_reference,
    iter_quantities,
    minimum_reference_games,
    observations_for,
    reference_workload,
    validate_reference_size,
)
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DatasetReference,
    ExecutionRecord,
    Measurement,
    ResultEnvelope,
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
    GENERATED_PLAY_EXACT_REPERTOIRE_CONDITIONAL_DISTANCE,
    GENERATED_PLAY_EXACT_REPERTOIRE_POOLED_DISTANCE,
    GENERATED_PLAY_EXACT_REPERTOIRE_PRUNED_MASS,
    GENERATED_PLAY_EXACT_REPERTOIRE_UNSETTLED_MASS,
    GENERATED_PLAY_EXACT_REPERTOIRE_WAYPOINT_MASS,
    GENERATED_PLAY_MEAN_AVAILABLE_PLY,
    GENERATED_PLAY_MEAN_BOOK_PLY,
    GENERATED_PLAY_MEAN_CONSUMED_FRACTION,
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
    GENERATED_PLAY_WAYPOINT_GAME_RATE,
    GENERATED_PLAY_WHITE_SCORE,
    MOVE_PREDICTION_PROJECTION,
)
from anthro_chess.evaluation.results.noise import bounded_floor
from anthro_chess.evaluation.selection import CheckpointSelection
from anthro_chess.evaluation.views import (
    ViewConfig,
    ViewSelection,
    apply_view,
    excluded_summary,
)
from anthro_chess.runtime import ActionModelRunner, RuntimeConfig
from anthro_chess.runtime.session import (
    BatchedActionModelRunner,
    DecisionRuntimeError,
    GameSession,
)

ROLLOUT_BENCHMARK_VERSION = 1

ROLLOUT_KIND = "generated-play"
ROLLOUT_BENCHMARK = BenchmarkReference(
    name="generated-play-rollout",
    version=ROLLOUT_BENCHMARK_VERSION,
)

logger = logging.getLogger(__name__)

#: What the prefix arm reads from a pool row: the moves it continues from,
#: and nothing beyond what the provenance digest already declares.
_PREFIX_COLUMNS = (
    NormalizedColumn.SOURCE_ID.value,
    NormalizedColumn.SOURCE_GAME_KEY.value,
    *MOVE_PREDICTION_PROJECTION.columns,
)


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
    #: later evaluation-noise characterization reads. A cell at temperature
    #: zero plays the first of them alone, because greedy seats make every
    #: replicate the same game.
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
    Prefixes are a handful of openings to continue; the reference is what forces
    the bandwidth and the noise floors at every evaluation point.

    Its size is not a sample count. The bandwidth is a neighbour count, so the
    reference size decides the rating span those neighbours occupy: shrinking it
    widens every neighbourhood until the grid points are estimated from the same
    games. That is why a reduced sweep leaves this view alone, and why the size
    belongs in the declared workload rather than in provenance.
    """

    view: ViewConfig = ViewConfig(name="rollout-reference", require_ratings=True)


class RepertoireWalkConfig(ConfigModel):
    """The exact shallow repertoire reading, computed rather than sampled.

    A model's policy at a fixed position is one forward pass, so the shallow
    end of the repertoire distribution can be enumerated with no sampling noise
    at all. Only the shallow end: the branching factor past a handful of plies
    beats any threshold, which is why the deep reading stays sampled.
    """

    enabled: StrictBool = True
    #: How deep to enumerate. Eight plies covers the depth at which the openings
    #: people actually name their repertoire after become nameable — the Ruy
    #: Lopez, Italian, and Scotch split at five — while staying inside what an
    #: exhaustive walk can afford.
    plies: Annotated[StrictInt, Field(ge=1)] = 8
    #: Cumulative-probability threshold below which a line stops being expanded.
    #: Its reciprocal bounds the positions evaluated per ply, and the mass it
    #: leaves uncommitted is reported as the reading's error bound.
    #:
    #: Chosen by measurement rather than by feel. On a real checkpoint the
    #: leading family share reads 0.281 at 1e-2, 0.260 at 1e-3, and 0.258 at
    #: 1e-4: the distribution has converged by 1e-3, and the order of magnitude
    #: past it buys 0.002 of movement for roughly nine times the work. Cost at
    #: this value is a few seconds per rating, against a suite that spends
    #: minutes generating games.
    threshold: Annotated[float, Field(gt=0.0, le=1.0)] = 0.001


class DivergenceDepthConfig(ConfigModel):
    """The divergence-versus-depth diagnostic, recomputed at each book ply.

    A diagnostic rather than a headline: it says *where* the model departs from
    human play rather than whether it does, and its category count grows with
    depth, so its floor grows with depth too. Detail tier only, and never
    stored without the floor each point was estimated against.
    """

    enabled: StrictBool = True
    #: Resamples behind each depth's floor. Lower than a headline comparison's,
    #: because this runs once per ply and only the floor is read from it — the
    #: null levels are skipped entirely.
    resamples: Annotated[StrictInt, Field(ge=2)] = 64
    #: Deepest ply to recompute at. Defaults to the deepest match either side
    #: actually reached, so a shallow suite does not pay for plies no game got
    #: near.
    maximum_ply: Annotated[StrictInt, Field(ge=1)] | None = None


class RolloutDetailConfig(ConfigModel):
    """What the machine-local detail tier keeps beside a committed summary."""

    #: Whole game records. Large, and the input to every later distribution
    #: feature: recomputing a feature over retained games is seconds where
    #: regenerating them is hours. Retained by default for that reason.
    retain_games: StrictBool = True


class RolloutBenchmarkConfig(CheckpointSelection, PoolGenerationPin):
    """Code-owned schema for ``anthro eval rollout``."""

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
    walk: RepertoireWalkConfig = RepertoireWalkConfig()
    divergence: DivergenceDepthConfig = DivergenceDepthConfig()
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
            if self.expected_pool_game_ids_sha256 is not None:
                raise ValueError(
                    "expected_pool_game_ids_sha256 names the generation of a "
                    "pool this selection does not read"
                )
        if self.reference.enabled:
            validate_reference_size(
                self.reference.view,
                self.grid.target_ratings,
                max(DECLARED_NEIGHBOURS.values()),
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
    #: The seeds the cell actually played, which ``collapse_replicates`` cuts
    #: to one at temperature zero.
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
    invented, and absent again where the seeds all agreed.

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
        dispersion = stdev([value for _, value in values])
        if dispersion == 0.0:
            # ``bounded_floor`` refuses a zero, and absent is the right answer
            # anyway, for the same reason as below three seeds: the seeds
            # agreeing says this draw of them could not move the distance, not
            # that a re-run would not.
            return None
        # The seeds are the replicates, so a suite run at the usual handful of
        # them leaves few degrees of freedom and a wide bound. That is the
        # honest reading of a spread measured this thinly, and it is another
        # reason this stays a diagnostic rather than a floor a report applies.
        return bounded_floor(dispersion, degrees_of_freedom=len(values) - 1)

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
class DepthDivergence:
    """One point of the divergence-versus-depth diagnostic.

    The repertoire distance recomputed with classification truncated at one
    ply. Raw labels rather than destinations, because at a truncated depth
    every line is still en route: the waypoint distinction is a statement about
    where a game *stopped*, which has no meaning at a depth the reading imposed.

    The floor travels with the point and is not optional. Category count grows
    with depth, so the distance a model with nothing wrong would still show
    grows with it, and the raw number read alone says that the model diverges
    more deeply than it does.
    """

    ply: int
    categories: int
    conditional_distance: float
    conditional_floor: float | None
    pooled_distance: float
    pooled_floor: float | None

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one depth point."""

        return {
            "ply": self.ply,
            "categories": self.categories,
            "conditional_distance": self.conditional_distance,
            "conditional_floor": self.conditional_floor,
            "pooled_distance": self.pooled_distance,
            "pooled_floor": self.pooled_floor,
        }


@dataclass(frozen=True)
class ExactRepertoire:
    """The shallow repertoire enumerated exactly, against the human reference.

    One walk per conditioning rating, since the policy is conditioned on it.
    The human side is the same reference every other curve reads, classified at
    the walk's own depth and smoothed at the declared bandwidth, so the two
    sides meet on one definition.
    """

    plies: int
    threshold: float
    walks: tuple[tuple[int, RepertoireWalk], ...]
    #: Distance at each rating the walk ran at, and the mean over them.
    distances: tuple[tuple[int, float], ...]
    conditional_distance: float
    pooled_distance: float
    #: Worst pruned mass across the ratings, which is the bound that applies to
    #: the reading as a whole rather than to its most favorable point.
    pruned_mass: float
    #: The same, for the bound that discounts mass the book has already
    #: committed. This is the one to read; ``pruned_mass`` saturates near one on
    #: any real policy and says almost nothing.
    unsettled_mass: float
    waypoint_mass: float
    #: Deepest ply any rating's walk actually expanded to, which is below
    #: ``plies`` whenever the threshold bit first.
    deepest_expanded_ply: int
    execution: ExecutionRecord

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one exact repertoire reading."""

        return {
            "plies": self.plies,
            "threshold": self.threshold,
            "conditional_distance": self.conditional_distance,
            "pooled_distance": self.pooled_distance,
            "pruned_mass": self.pruned_mass,
            "unsettled_mass": self.unsettled_mass,
            "waypoint_mass": self.waypoint_mass,
            "deepest_expanded_ply": self.deepest_expanded_ply,
            "workload_sha256": self.execution.workload_sha256,
            "workload": dict(self.execution.workload),
            "ratings": [
                {
                    "target_rating": rating,
                    "distance": distance,
                    "walk": walk.as_record(),
                }
                for (rating, walk), (_, distance) in zip(
                    self.walks, self.distances, strict=True
                )
            ],
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
    #: spread implies. Where a measurement carries a bootstrap floor it is an
    #: estimate of this quantity; recording both is what makes it checkable
    #: rather than trusted. A temperature-zero reading has one seed and states
    #: its floor instead, so neither number is present there.
    seed_spread: dict[ComparedQuantity, SeedSpread] = field(default_factory=dict)
    #: Quantities no game on one side had a value for, with the reason. A real
    #: state rather than an error: a checkpoint whose every game leaves the book
    #: on a waypoint has made no repertoire choice to compare, and reporting
    #: that is more useful than failing the whole reading or writing a zero.
    unavailable: dict[ComparedQuantity, str] = field(default_factory=dict)
    #: The repertoire distance recomputed at each book ply. Detail tier only.
    divergence: tuple[DepthDivergence, ...] = ()
    #: The shallow repertoire enumerated exactly, when the walk ran.
    exact: ExactRepertoire | None = None

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
            "unavailable": {
                quantity.value: reason
                for quantity, reason in sorted(self.unavailable.items())
            },
            "divergence_by_depth": [point.as_record() for point in self.divergence],
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
            "exact_repertoire": [
                reading.exact.as_record()
                for reading in self.readings
                if reading.exact is not None
            ],
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
    recording: ResultRecording,
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
    loaded, identity = resolve_model(
        config.model,
        runner,
        checkpoint,
        label=config.checkpoint_label,
        run_root=run_root,
        error=RolloutBenchmarkError,
    )
    book = _load_book()
    sources, view, dataset = _position_sources(config, book=book)

    # Read before the matrix is played, not after. The reference is a pass over
    # frozen rows and can fail on its own — a view the rating gap guts cannot
    # carry the declared bandwidth — so a suite that is going to fail on it
    # should fail in the pool pass rather than an hour of generation later.
    reference: HumanReference | None = None
    reference_view: ViewSelection | None = None
    if config.reference.enabled:
        reference, reference_view = _load_reference(config, book=book)

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

    readings: tuple[RolloutReading, ...] = ()
    if reference is not None and reference_view is not None:
        readings = _curve_readings(
            config,
            cells,
            reference,
            reference_view,
            book=book,
            runner=loaded,
            device=runner_device(loaded),
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
    _record(result, recording)
    return result


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
    """Play every replicate of one cell and pool them into its reading."""

    runtime = config.runtime.model_copy(
        update={"target_rating": target_rating, "temperature": temperature}
    )
    player = ModelPlayer(
        runner,
        label=f"{checkpoint.label}-r{target_rating}-t{temperature:g}",
        config=runtime,
        checkpoint=checkpoint,
    )
    # One configuration in both seats, so the cell's own temperature is the
    # whole of what decides whether its replicates differ.
    seeds, generation = collapse_replicates(
        config.grid.seeds, config.generation, temperatures=(temperature,)
    )
    features: list[GameFeatures] = []
    per_seed: list[tuple[int, GameDistribution]] = []
    by_seed: list[tuple[int, tuple[GameFeatures, ...]]] = []
    records: list[GameRecord] = []
    for seed in seeds:
        played = _generate(
            player, source.positions, generation.model_copy(update={"seed": seed})
        )
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
        seeds=seeds,
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
            f"view {view_config.name!r} selected no games from the pool "
            f"({excluded_summary(selection.excluded_games)})"
        )

    rows = [
        _prefix_row(row, config.prefix.plies)
        for row in pool_rows(
            pool,
            selection.game_ids,
            _PREFIX_COLUMNS,
            error=RolloutBenchmarkError,
        )
    ]
    games = sorted(
        (
            row_game_id(row),
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
        return load_pool(
            config.pool,
            expected_game_ids_sha256=config.expected_pool_game_ids_sha256,
        )
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
            f"compare against ({excluded_summary(selection.excluded_games)})"
        )
    rows = pool_rows(
        pool,
        selection.game_ids,
        REFERENCE_COLUMNS,
        error=RolloutBenchmarkError,
    )
    try:
        reference = human_reference(rows, config.reference, book=book)
    except ReferenceError as error:
        raise RolloutBenchmarkError(str(error)) from error
    required = minimum_reference_games(
        config.grid.target_ratings, max(DECLARED_NEIGHBOURS.values())
    )
    # The declared cap clears this floor by construction; what the cap cannot
    # promise is how much of the view survives the rating-gap filter, which
    # depends on the pool's rating composition rather than on configuration.
    if len(reference.games) < required:
        raise RolloutBenchmarkError(
            f"view {config.reference.view.name!r} left "
            f"{len(reference.games)} usable human game(s) of "
            f"{selection.selected_games} selected "
            f"({excluded_summary(reference.excluded)}), "
            f"below the {required} a curve over this rating grid needs at the "
            "declared bandwidth; widen the view or the rating gap"
        )
    return reference, selection


def _curve_readings(
    config: RolloutBenchmarkConfig,
    cells: Sequence[RolloutCell],
    reference: HumanReference,
    reference_view: ViewSelection,
    *,
    book: OpeningBook | None,
    runner: ActionModelRunner,
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
                    config,
                    arm,
                    temperature,
                    selected,
                    reference,
                    reference_view,
                    book=book,
                    runner=runner,
                    device=device,
                )
            )
    return tuple(readings)


def _curve_reading(
    config: RolloutBenchmarkConfig,
    arm: RolloutArm,
    temperature: float,
    cells: Sequence[RolloutCell],
    reference: HumanReference,
    reference_view: ViewSelection,
    *,
    book: OpeningBook | None,
    runner: ActionModelRunner,
    device: torch.device,
) -> RolloutReading:
    """Compare one arm's whole rating grid at one temperature."""

    model: list[ComparableGame] = []
    for cell in cells:
        model.extend(generated_games(cell.features, rating=cell.target_rating))
    ratings = tuple(sorted({cell.target_rating for cell in cells}))
    # Both seats play at the reading's own temperature, so this is the whole of
    # what decides whether another run of it would produce different games —
    # and therefore whether it has any evaluation noise for a floor to bound.
    varies = replicates_vary((temperature,))
    comparisons: dict[ComparedQuantity, CurveComparison] = {}
    unavailable: dict[ComparedQuantity, str] = {}
    for quantity in iter_quantities():
        try:
            spec = curve_spec(quantity, ratings)
        except ReferenceError as error:
            raise RolloutBenchmarkError(str(error)) from error
        human = reference.observations(quantity, level=config.opening_level)
        generated = observations_for(model, quantity, level=config.opening_level)
        # A quantity some games simply do not have is a reading about the
        # checkpoint, not a broken run: every generated game leaving the book on
        # a waypoint means there is no repertoire to compare, and the waypoint
        # rate is where that shows up.
        if not generated or not human:
            side = "generated" if not generated else "human"
            unavailable[quantity] = f"no {side} game carries this quantity"
            continue
        try:
            comparisons[quantity] = compare_curves(
                spec=spec,
                human=human,
                model=generated,
                resamples=config.reference.resamples,
                seed=config.reference.seed,
                model_varies=varies,
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
        unavailable=unavailable,
        divergence=_divergence_by_depth(
            config, cells, reference, ratings, book=book, model_varies=varies
        ),
        exact=_exact_repertoire(
            config,
            arm,
            temperature,
            cells,
            reference,
            reference_view,
            ratings,
            book=book,
            runner=runner,
            device=device,
        ),
        execution=_reading_execution_record(
            config, cells, temperature, ratings, reference_view, device=device
        ),
    )


def _divergence_by_depth(
    config: RolloutBenchmarkConfig,
    cells: Sequence[RolloutCell],
    reference: HumanReference,
    ratings: Sequence[int],
    *,
    book: OpeningBook | None,
    model_varies: bool,
) -> tuple[DepthDivergence, ...]:
    """Recompute the opening distance with classification truncated per ply.

    Says where the model departs from human play rather than whether it does.
    Every point carries the floor it was estimated against, because category
    count grows with depth and so does the distance two identical populations
    would still show.

    The distance is over raw labels rather than destinations. At an imposed
    truncation every line is still on its way somewhere, so excluding waypoints
    would exclude nearly everything at the shallow end and would be measuring
    the truncation rather than the model.
    """

    if not config.divergence.enabled:
        return ()
    level = config.opening_level
    human_games = reference.games
    model_games = [
        game
        for cell in cells
        for game in generated_games(cell.features, rating=cell.target_rating)
    ]
    if not model_games:  # pragma: no cover - a cell always plays games
        return ()
    deepest = config.divergence.maximum_ply or max(
        (game.trajectory.opening.matched_ply for game in (*human_games, *model_games)),
        default=0,
    )
    if deepest < 1:
        return ()

    try:
        spec = curve_spec(ComparedQuantity.REPERTOIRE, ratings)
    except ReferenceError as error:  # pragma: no cover - the grid is validated
        raise RolloutBenchmarkError(str(error)) from error

    human_labels = _label_progressions(human_games, deepest, book=book)
    model_labels = _label_progressions(model_games, deepest, book=book)

    points: list[DepthDivergence] = []
    for ply in range(1, deepest + 1):
        human = _depth_observations(human_games, human_labels, ply, level)
        model = _depth_observations(model_games, model_labels, ply, level)
        if not human or not model:  # pragma: no cover - both sides start named
            continue
        try:
            comparison = compare_curves(
                spec=spec,
                human=human,
                model=model,
                resamples=config.divergence.resamples,
                seed=config.reference.seed,
                model_varies=model_varies,
                references=False,
            )
        except CurveComparisonError:
            # One depth the reference cannot support is a reason to report
            # fewer points, not to fail a diagnostic the rest of them support.
            continue
        spreads = comparison.dispersions
        points.append(
            DepthDivergence(
                ply=ply,
                categories=len({observation.value for observation in (*human, *model)}),
                conditional_distance=comparison.conditional_distance,
                conditional_floor=(
                    None if spreads is None else spreads.conditional_floor
                ),
                pooled_distance=comparison.pooled_distance,
                pooled_floor=None if spreads is None else spreads.pooled_floor,
            )
        )
    return tuple(points)


def _label_progressions(
    games: Sequence[ComparableGame],
    deepest: int,
    *,
    book: OpeningBook | None,
) -> tuple[tuple[OpeningLabel, ...], ...]:
    """Classify every game once, keeping the label it held at each ply.

    One replay per game rather than one per game and depth. The sweep asks the
    same question at thirty-odd plies, and reclassifying from the start each
    time would make a diagnostic cost more than the games it reads.
    """

    return tuple(
        progression_for_action_ids(
            game.trajectory.action_ids,
            initial_position=game.trajectory.initial_position,
            book=book,
            plies=deepest,
        )
        for game in games
    )


def _depth_observations(
    games: Sequence[ComparableGame],
    progressions: Sequence[tuple[OpeningLabel, ...]],
    ply: int,
    level: OpeningLevel,
) -> tuple[Observation, ...]:
    """Return one depth's raw opening labels, one observation per game."""

    return tuple(
        Observation(rating=game.rating, value=progression[ply - 1].label(level))
        for game, progression in zip(games, progressions, strict=True)
    )


def _exact_repertoire(
    config: RolloutBenchmarkConfig,
    arm: RolloutArm,
    temperature: float,
    cells: Sequence[RolloutCell],
    reference: HumanReference,
    reference_view: ViewSelection,
    ratings: Sequence[int],
    *,
    book: OpeningBook | None,
    runner: ActionModelRunner,
    device: torch.device,
) -> ExactRepertoire | None:
    """Enumerate the shallow repertoire exactly, and meet the human reference.

    Standard-start only. On the prefix arm the opening was decided by the view
    before the model moved, so a walk from the standard start would be
    answering a question that arm never asked.

    One walk per conditioning rating, because the policy is conditioned on it
    and the whole point of the reading is whether that dial moves the openings
    it plays.
    """

    if not config.walk.enabled or arm is not RolloutArm.STANDARD_START:
        return None
    level = config.opening_level
    plies = config.walk.plies

    human_labels = [
        progression[plies - 1]
        for progression in _label_progressions(reference.games, plies, book=book)
    ]
    observations = tuple(
        Observation(rating=game.rating, value=destination)
        for game, label in zip(reference.games, human_labels, strict=True)
        if (destination := label.destination(level)) is not None
    )
    if not observations:
        logger.info(
            "Exact repertoire skipped: no reference game reaches a destination "
            "within %s plies",
            plies,
        )
        return None

    walks: list[tuple[int, RepertoireWalk]] = []
    for rating in ratings:
        runtime = config.runtime.model_copy(
            update={"target_rating": rating, "temperature": temperature}
        )
        try:
            walks.append(
                (
                    rating,
                    walk_repertoire(
                        _walk_policy(runner, runtime),
                        plies=plies,
                        threshold=config.walk.threshold,
                        level=level,
                        book=book,
                    ),
                )
            )
        except (OpeningTreeError, DecisionRuntimeError) as error:
            raise RolloutBenchmarkError(
                f"cannot walk the opening tree at rating {rating}: {error}"
            ) from error

    model_categories = sorted(
        {label for _, walk in walks for label in walk.destinations}
    )
    try:
        curve = estimate_curve(
            curve_spec(ComparedQuantity.REPERTOIRE, ratings),
            observations,
            categories=model_categories,
        )
    except (CurveComparisonError, ReferenceError) as error:
        raise RolloutBenchmarkError(
            f"cannot estimate the human repertoire reference: {error}"
        ) from error

    distances: list[tuple[int, float]] = []
    for rating, walk in walks:
        human = curve.distribution_at(float(rating))
        if human is None:
            continue
        distances.append((rating, distribution_distance(walk.repertoire(), human)))
    if not distances:  # pragma: no cover - the grid is the reference's own
        return None

    pooled_model: dict[str, float] = {}
    for _, walk in walks:
        for label, share in walk.repertoire().items():
            pooled_model[label] = pooled_model.get(label, 0.0) + share / len(walks)
    return ExactRepertoire(
        plies=plies,
        threshold=config.walk.threshold,
        walks=tuple(walks),
        distances=tuple(distances),
        conditional_distance=sum(value for _, value in distances) / len(distances),
        pooled_distance=distribution_distance(
            pooled_model, curve.pooled.distribution or {}
        ),
        pruned_mass=max(walk.pruned_mass for _, walk in walks),
        unsettled_mass=max(walk.unsettled_mass for _, walk in walks),
        waypoint_mass=sum(walk.waypoint_mass for _, walk in walks) / len(walks),
        deepest_expanded_ply=max(walk.deepest_expanded_ply for _, walk in walks),
        execution=_walk_execution_record(
            config, cells, temperature, ratings, reference_view, device=device
        ),
    )


def _walk_policy(
    runner: ActionModelRunner,
    runtime: RuntimeConfig,
) -> ActionPolicy:
    """Return the policy an opening-tree walk asks for its move distributions.

    Each prefix gets its own session, so the legal mask, the encoding, and the
    temperature all come from the runtime the games were played under rather
    than from a second implementation. A whole depth is encoded and predicted in
    one batch, which is what keeps an exhaustive walk affordable.
    """

    def policy(
        prefixes: Sequence[tuple[chess.Move, ...]],
    ) -> tuple[dict[chess.Move, float], ...]:
        sessions = [
            GameSession(runner, config=runtime, moves=prefix) for prefix in prefixes
        ]
        # A line that is already over contributes no continuation. It is not an
        # error: eight plies is long enough to reach a fool's mate.
        live = [
            index for index, session in enumerate(sessions) if not session.is_terminal
        ]
        contexts = [sessions[index].decision_context() for index in live]
        if isinstance(runner, BatchedActionModelRunner) and len(contexts) > 1:
            logits = runner.predict_batch(contexts)
            if len(logits) != len(contexts):
                raise RolloutBenchmarkError(
                    "model runner returned the wrong number of predictions"
                )
        else:
            logits = tuple(runner.predict(context) for context in contexts)

        distributions: list[dict[chess.Move, float]] = [{} for _ in prefixes]
        for index, row in zip(live, logits, strict=True):
            enabled, probabilities = sessions[index].selection_distribution(row)
            moves: dict[chess.Move, float] = {}
            for action_id, probability in zip(
                enabled, probabilities.tolist(), strict=True
            ):
                # Mass on a non-move action ends the line rather than
                # continuing it, so it is simply left out of the sub-
                # distribution the walk expands.
                if action_id < MOVE_ACTION_COUNT and probability > 0.0:
                    moves[decode_move(action_id)] = float(probability)
            distributions[index] = moves
        return tuple(distributions)

    return policy


def _walk_execution_record(
    config: RolloutBenchmarkConfig,
    cells: Sequence[RolloutCell],
    temperature: float,
    ratings: Sequence[int],
    reference_view: ViewSelection,
    *,
    device: torch.device,
) -> ExecutionRecord:
    """Declare what the exact walk measured.

    Its own workload rather than the reading's, because the walk's depth and
    threshold decide the quantity: a repertoire enumerated to eight plies and
    one enumerated to twelve are different readings, and a looser threshold is
    a different approximation of each. Keeping them here rather than on the
    reading is what stops a change to the walk from ending every other curve
    series in the same run.
    """

    workload = dict(cells[0].execution.workload)
    workload.pop("target_rating", None)
    workload.update(
        {
            "target_ratings": list(ratings),
            "temperature": temperature,
            "walk_plies": config.walk.plies,
            "walk_threshold": config.walk.threshold,
            "curve_spec_version": CURVE_SPEC_VERSION,
            "curve_neighbours": DECLARED_NEIGHBOURS[ComparedQuantity.REPERTOIRE],
            "reference": reference_workload(config.reference, reference_view),
        }
    )
    return execution_record(device, workload)


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
                    model=observations_for(games, quantity, level=config.opening_level),
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
    reference_view: ViewSelection,
    *,
    device: torch.device,
) -> ExecutionRecord:
    """Declare what one curve reading measured.

    A curve's workload is a cell's with the single rating replaced by the whole
    grid, because the grid is the axis rather than a setting held fixed. The
    curve shape joins it too: a distance estimated at one bandwidth and one at
    another are not the same quantity, which is exactly why the bandwidth is
    declared and frozen rather than configured. So does the human reference the
    bandwidth is expressed against, for the reason ``reference_workload``
    gives.
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
            "reference": reference_workload(config.reference, reference_view),
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

    try:
        component = projection_content_digest(rows, MOVE_PREDICTION_PROJECTION)
    except FingerprintError as error:
        raise RolloutBenchmarkError(str(error)) from error
    return pool_dataset_reference(
        pool,
        selection,
        component,
        error=RolloutBenchmarkError,
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
        runner_device(runner),
        {
            "generation_version": GENERATION_VERSION,
            "positions": source.identity,
            "target_rating": target_rating,
            "temperature": temperature,
            "maximum_generated_plies": config.generation.maximum_generated_plies,
            "swap_colors": config.generation.swap_colors,
            "claim_draws": config.generation.claim_draws,
            "resignation_enabled": config.runtime.resignation_enabled,
            "draw_claim_enabled": config.runtime.draw_claim_enabled,
            "opening_level": config.opening_level.value,
        },
    )


def _record(
    result: RolloutBenchmarkResult,
    recording: ResultRecording,
) -> None:
    """Write one envelope per cell, per curve reading, and per exact walk.

    Three kinds of record from one run, because they are three units. A cell is
    one point of the matrix and carries the raw rollout scalars; a reading spans
    one arm's whole rating grid and carries the distances against human play; an
    exact walk enumerates the shallow repertoire instead of playing it. Each has
    its own declared workload and so its own series, which is why none can stand
    in for another — and why changing the walk's depth cannot end the curve
    series recorded beside it.
    """

    recorder = recording.measuring(
        result.checkpoint,
        kind=ROLLOUT_KIND,
        benchmark=ROLLOUT_BENCHMARK,
    )
    for cell in result.cells:
        recorder.add(
            _measurements(cell),
            payload=partial(_cell_payload, cell),
            description=f"Generated-play rollout cell: {cell.label}",
            slug=(f"{cell.arm.value}-r{cell.target_rating}-t{_slug(cell.temperature)}"),
            # Provenance rather than series identity: the pool games were
            # an input to generation, not the content measured, so they
            # reach the fingerprint through the workload's position source
            # instead of through a data component.
            data=result.dataset if cell.arm is RolloutArm.HUMAN_PREFIX else None,
            execution=cell.execution,
        )
    for reading in result.readings:
        recorder.add(
            _curve_measurements(reading),
            payload=partial(_reading_payload, result, reading),
            description=f"Human-reference curves: {reading.label}",
            slug=f"{reading.arm.value}-curves-t{_slug(reading.temperature)}",
            # The reference view is provenance here for the same reason the
            # prefix view is on a cell: the human games shaped what was
            # compared against, and the workload carries identity.
            data=result.dataset,
            execution=reading.execution,
        )
        if reading.exact is None:
            continue
        recorder.add(
            _exact_measurements(reading),
            payload=reading.exact.as_record,
            description=f"Exact shallow repertoire: {reading.label}",
            slug=(f"{reading.arm.value}-repertoire-t{_slug(reading.temperature)}"),
            data=result.dataset,
            execution=reading.exact.execution,
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
        (
            GENERATED_PLAY_WAYPOINT_GAME_RATE.identifier,
            distribution.waypoint_game_rate,
            games,
        ),
        # The three depth readings are averaged over the games the book named
        # at all, since an unnamed game has no depth to average in.
        (
            GENERATED_PLAY_MEAN_BOOK_PLY.identifier,
            distribution.mean_book_ply,
            distribution.classified_games,
        ),
        (
            GENERATED_PLAY_MEAN_AVAILABLE_PLY.identifier,
            distribution.mean_available_ply,
            distribution.classified_games,
        ),
        (
            GENERATED_PLAY_MEAN_CONSUMED_FRACTION.identifier,
            distribution.mean_consumed_fraction,
            distribution.classified_games,
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
    generated, and deliberately not the spread across
    seeds. Both estimate evaluation noise, but only the bootstrap estimates it
    *at the sample size the reading was taken at*: each seed plays a fraction of
    the games, so the spread across seeds measures the noise of a much smaller
    reading and runs roughly the square root of the seed count too wide.
    Measured against forty independent draws, the bootstrap reproduces the true
    spread to within a few percent at fixed size, so the seeds stay a diagnostic
    rather than a floor.

    A temperature-zero reading has no draw to estimate from at all. Its floors
    are stated at zero rather than bootstrapped, per decision 0032, and neither
    estimator applies.
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


def _exact_measurements(reading: RolloutReading) -> tuple[Measurement, ...]:
    """Return the exactly computed repertoire distances, when the walk ran.

    Its own workload rather than the reading's: the walk's depth and threshold
    decide what these numbers mean, and a change to either should end this
    series without touching the sampled ones beside it.

    No sample size. The model side was enumerated rather than drawn, so there
    is no count of games behind it that a precision estimate could use; the
    pruned mass is the reading's error bound and is recorded as a metric of its
    own instead.
    """

    exact = reading.exact
    if exact is None:
        return ()
    workload = exact.execution.workload_component()
    values = (
        (
            GENERATED_PLAY_EXACT_REPERTOIRE_CONDITIONAL_DISTANCE.identifier,
            exact.conditional_distance,
        ),
        (
            GENERATED_PLAY_EXACT_REPERTOIRE_POOLED_DISTANCE.identifier,
            exact.pooled_distance,
        ),
        (
            GENERATED_PLAY_EXACT_REPERTOIRE_PRUNED_MASS.identifier,
            exact.pruned_mass,
        ),
        (
            GENERATED_PLAY_EXACT_REPERTOIRE_UNSETTLED_MASS.identifier,
            exact.unsettled_mass,
        ),
        (
            GENERATED_PLAY_EXACT_REPERTOIRE_WAYPOINT_MASS.identifier,
            exact.waypoint_mass,
        ),
    )
    return tuple(
        measurement(identifier, value, workload=workload)
        for identifier, value in values
    )


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


def _load_book() -> OpeningBook:
    """Load the versioned opening book once for the whole suite."""

    try:
        return load_book()
    except OpeningBookError as error:
        raise RolloutBenchmarkError(str(error)) from error


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
