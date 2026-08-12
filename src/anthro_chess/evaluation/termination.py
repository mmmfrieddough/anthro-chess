"""How games end, measured against the human termination mix.

Ending a game is a decision the model makes, and no aggregate over moves can
see it. A checkpoint that never resigns and one that resigns while winning post
the same move cross-entropy, so this family exists to make the difference
visible before resignation is enabled by default.

Four readings, deliberately kept apart because they answer different questions
and cost wildly different amounts:

- the **mix**, a human-reference curve comparison over derived termination
  categories, sliced by rating and by the time control of the human population
  it is read against;
- **held-out resignation prediction**, the mass the policy puts on resigning at
  the plies where a human actually resigned and at the plies where one moved
  instead. One pass over frozen games, no rollouts, cheap enough to read often;
- the **resignation deficit**, how far behind the model was when it resigned,
  against the human distribution for the same rating band;
- the **guardrails**, whose failure modes are not symmetric and which a
  distributional distance therefore averages away rather than reports.

Both sides of the mix are counted over one vocabulary, formed as the union of
the derived human categories and the harness's own. Neither side can produce
every term: abandonment is a human-only bucket because the model has no way to
walk away, and the ply limit is a model-only bucket because human games do not
have one. They stay visible as their own categories rather than being folded
into a neighbour, so a permanent gap reads as a permanent gap instead of
distorting a category a checkpoint could actually move. See
``docs/decisions/0017-derived-termination-and-terminal-actions.md``.

Nothing here reports a target rate. Human endings are the reference for the
mix and for the deficit, and the two guardrails that do carry a direction —
premature resignation and non-termination — are defects rather than style
choices.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

import chess
from pydantic import StrictBool, StrictInt, model_validator
from torch import Tensor

from anthro_chess.chess import (
    DRAW_CLAIM_ACTION_ID,
    RESIGNATION_ACTION_ID,
    draw_claim_available,
    is_terminal_action,
)
from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import SequenceDataLoader, Speed, speed_from_clock_ms
from anthro_chess.data.schema import NormalizedColumn
from anthro_chess.data.termination import TERMINATION_CATEGORIES, TerminationCategory
from anthro_chess.evaluation.curves import (
    CurveComparison,
    CurveComparisonError,
    CurveMetrics,
    CurveQuantity,
    CurveSpec,
    Observation,
    compare_curves,
    distribution_distance,
)
from anthro_chess.evaluation.execution import execution_record, runner_device
from anthro_chess.evaluation.games import (
    GENERATION_VERSION,
    GameRecord,
    GameTermination,
    GenerationConfig,
    GenerationError,
    ModelPlayer,
    PlayerError,
    StartPosition,
    collapse_replicates,
    generate_games,
    replicates_vary,
    standard_positions,
)
from anthro_chess.evaluation.policy import (
    POLICY_SCORING_VERSION,
    TerminalActionPolicy,
    active_batch,
    score_terminal_actions,
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
    ReferenceConfig,
    minimum_reference_games,
    reference_workload,
    validate_reference_size,
)
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
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
    TERMINATION_MIX_CONDITIONAL_DISTANCE,
    TERMINATION_MIX_POOLED_DISTANCE,
    TERMINATION_MIX_RATING_VARIATION,
    TERMINATION_PREDICTION_PROJECTION,
    TERMINATION_PREMATURE_RESIGNATION_HUMAN_RATE,
    TERMINATION_PREMATURE_RESIGNATION_RATE,
    TERMINATION_RESIGNATION_DEFICIT_DISTANCE,
    TERMINATION_RESIGNATION_DEFICIT_GAP,
    TERMINATION_RESIGNATION_DEFICIT_MEDIAN,
    TERMINATION_RESIGNATION_MASS_AT_MOVES,
    TERMINATION_RESIGNATION_MASS_AT_RESIGNATION,
    TERMINATION_RESIGNATION_MASS_SEPARATION,
    TERMINATION_SILENT_TERMINAL_ACTIONS,
    TERMINATION_UNTIMED_NON_TERMINATION_RATE,
)
from anthro_chess.evaluation.scoring import (
    SCORED_COLUMNS,
    EvaluationLoaderConfig,
    ScoringError,
    build_scoring_inputs,
    rows_identity_sha256,
)
from anthro_chess.evaluation.selection import CheckpointSelection
from anthro_chess.evaluation.slices import material_balance, rating_band_name
from anthro_chess.evaluation.views import ViewConfig, ViewSelection, apply_view
from anthro_chess.models import MoveModelBatch
from anthro_chess.runtime import ActionModelRunner, RuntimeConfig

TERMINATION_BENCHMARK_VERSION = 1

TERMINATION_KIND = "game-termination"
TERMINATION_BENCHMARK = BenchmarkReference(
    name="game-termination",
    version=TERMINATION_BENCHMARK_VERSION,
)

logger = logging.getLogger(__name__)

#: What both views read from a pool row: the encoder's inputs for the held-out
#: scoring pass, the derived termination the human mix is built from, and the
#: result the resignation deficit is replayed against.
_ENDING_COLUMNS = (
    *SCORED_COLUMNS,
    NormalizedColumn.RESULT.value,
    NormalizedColumn.TERMINATION_CATEGORY.value,
    NormalizedColumn.TERMINAL_ACTION_STATUS.value,
)

#: Endings only a human platform produces. The model has no clock to lose, no
#: opponent to agree with, and no way to walk away, so these can only ever be
#: human mass. They are reported as themselves rather than merged into a
#: comparable category: merging would hide a permanent gap inside a number a
#: checkpoint is supposed to be able to move.
HUMAN_ONLY_CATEGORIES: tuple[str, ...] = (
    TerminationCategory.ABANDONMENT.value,
    TerminationCategory.CLOCK_EXPIRY.value,
    TerminationCategory.DRAW_AGREEMENT.value,
    TerminationCategory.UNKNOWN.value,
)

#: The mirror image: a generated game the harness stopped has no human
#: counterpart, because human games have no ply limit. Kept in the vocabulary
#: for the same reason, and read beside the mix rather than instead of it.
MODEL_ONLY_CATEGORIES: tuple[str, ...] = (GameTermination.PLY_LIMIT.value,)

#: The one vocabulary both sides are counted over.
TERMINATION_MIX_CATEGORIES: tuple[str, ...] = tuple(
    sorted({*TERMINATION_CATEGORIES, *(value.value for value in GameTermination)})
)

#: Version of the mix comparison's declared shape. Bumping it ends every
#: termination-mix series, so it changes only when the bandwidth does.
MIX_CURVE_SPEC_VERSION = 1

#: Bandwidth for the mix curve, declared and frozen rather than selected per
#: run: re-selecting it would mean two checkpoints were measured differently.
#: It inherits the value the generated-play comparisons declare, for the reason
#: that single shared value exists — selection needs the real pool, and a later
#: selection that moves the number is a spec version bump rather than a silent
#: merge. Reproduce a selection with ``anthro eval curve-bandwidth``.
DECLARED_MIX_NEIGHBOURS = 1024

#: Deficit buckets, in pawns behind from the resigning player's point of view.
#: Chosen against the population decision 0017 measured, where resignations sat
#: at a median of six pawns down with 73% at least three down: the boundaries
#: separate resigning while not behind, resigning a pawn or two down, and the
#: ordinary range, rather than resolving the far tail nobody reads.
DEFICIT_BUCKET_EDGES: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 9.0)


class TerminationBenchmarkError(ValueError):
    """Raised when game termination cannot be measured as configured."""


class ScoringModelRunner(Protocol):
    """The runner surface the held-out reading needs.

    Narrower than the checkpoint runner and wider than the decision runtime's:
    this benchmark plays games *and* scores stored positions, which are two
    different entry points into the same loaded model.
    """

    def action_logits(self, batch: MoveModelBatch) -> Tensor:
        """Return raw action logits for one aligned evaluation batch."""


#: One human population the mix is read against, named by the speed
#: :func:`~anthro_chess.data.speed_from_clock_ms` bands its games into, or
#: ``all`` for the undivided reference. The generated side has no clock — the
#: harness plays untimed — so a class slices the *reference* rather than the
#: model, which is the useful direction anyway: the question is which human
#: population the model's endings resemble, and a corpus of blitz games ends
#: very differently from a corpus of classical ones.
#:
#: Reusing the data layer's vocabulary is what makes this reading and a
#: selection trained at that speed one population. A reference game played
#: without a clock bands into no speed at all, so only the undivided reading
#: holds it, and a ``correspondence`` class reads only the games whose clock
#: reached that band.
TimeControlClass: TypeAlias = Speed | Literal["all"]


class TerminationGridConfig(ConfigModel):
    """The rating and temperature grid the generated side is played over."""

    #: The curve's axis, so at least two are needed for a mix reading.
    target_ratings: tuple[StrictInt, ...] = (1200, 1500, 1800)
    #: Each temperature is its own reading, because temperature is a separate
    #: dial rather than a point on the rating axis.
    temperatures: tuple[float, ...] = (1.0,)
    #: Replicates of one rating's reading, and precision alone. A temperature
    #: of zero plays the first of them alone, because greedy seats replay one
    #: game per position.
    seeds: tuple[StrictInt, ...] = (0, 1, 2)

    @model_validator(mode="after")
    def _validate_axes(self) -> TerminationGridConfig:
        for name, values in (
            ("target_ratings", self.target_ratings),
            ("temperatures", self.temperatures),
            ("seeds", self.seeds),
        ):
            if not values:
                raise ValueError(f"a termination grid needs at least one {name} value")
            if len(set(values)) != len(values):
                raise ValueError(f"a termination grid must not repeat a {name} value")
        if any(rating < 0 for rating in self.target_ratings):
            raise ValueError("a conditioning rating cannot be negative")
        if any(not 0.0 <= value <= 3.0 for value in self.temperatures):
            raise ValueError("a termination temperature must be between zero and three")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("a termination seed cannot be negative")
        return self


class TerminationReferenceConfig(ReferenceConfig):
    """Which human games the mix and the deficit are read against.

    Its size is not a sample count. The mix bandwidth is a neighbour count, so
    the reference size decides the rating span every neighbourhood covers:
    shrinking it widens them until the grid points are estimated from the same
    games. That is why a reduced sweep leaves this view alone.
    """

    view: ViewConfig = ViewConfig(name="termination-reference", require_ratings=True)


class HeldOutResignationConfig(ConfigModel):
    """The cheap half: resignation mass on games humans already played."""

    enabled: StrictBool = True
    #: Its own view rather than the reference's. The reference wants as much
    #: matched play as it can afford because it forces the curve's bandwidth;
    #: this one scores every ply of what it selects, so it is sized by scoring
    #: cost instead.
    view: ViewConfig = ViewConfig(name="termination-held-out", maximum_games=256)
    loader: EvaluationLoaderConfig = EvaluationLoaderConfig()


class GuardrailConfig(ConfigModel):
    """Thresholds the two directional guardrails are judged against."""

    #: Material balance, in pawns and from the resigning player's point of
    #: view, at or above which a resignation counts as premature. Zero means
    #: "resigned while level or ahead", which is the egregious case material
    #: alone can honestly claim. Part of the declared workload, because moving
    #: it measures a different quantity.
    premature_material_balance: float = 0.0


class TerminationDetailConfig(ConfigModel):
    """What the machine-local detail tier keeps beside a committed summary."""

    #: Whole game records. Large, and the input to any later ending feature.
    retain_games: StrictBool = False
    #: Per-decision resignation mass from the held-out pass. One row per scored
    #: ply, so it is the largest payload this benchmark can produce and the one
    #: that makes an unexpected reading explainable.
    retain_decisions: StrictBool = True


class TerminationBenchmarkConfig(CheckpointSelection, PoolGenerationPin):
    """Code-owned schema for ``anthro eval termination``."""

    #: The base runtime settings every seat plays under. Whether the terminal
    #: actions are enabled at all is the setting that matters most here, and it
    #: joins the declared workload: a suite played with resignation disabled
    #: measures a model that was never offered the choice.
    runtime: RuntimeConfig = RuntimeConfig(
        resignation_enabled=True,
        draw_claim_enabled=True,
    )
    grid: TerminationGridConfig = TerminationGridConfig()
    generation: GenerationConfig = GenerationConfig()
    #: A frozen evaluation pool. Every reading here is read against human
    #: endings, so a suite without one has nothing to compare against.
    pool: Path
    reference: TerminationReferenceConfig = TerminationReferenceConfig()
    held_out: HeldOutResignationConfig = HeldOutResignationConfig()
    #: One reading per class. Reading against the whole reference is the right
    #: default for a corpus prepared at one speed.
    time_controls: tuple[TimeControlClass, ...] = ("all",)
    guardrails: GuardrailConfig = GuardrailConfig()
    detail: TerminationDetailConfig = TerminationDetailConfig()

    @model_validator(mode="after")
    def _validate_time_controls(self) -> TerminationBenchmarkConfig:
        if not self.time_controls:
            raise ValueError(
                "a termination suite needs at least one time-control class"
            )
        if len(set(self.time_controls)) != len(self.time_controls):
            raise ValueError("a termination suite must not repeat a time-control name")
        validate_reference_size(
            self.reference.view, self.grid.target_ratings, DECLARED_MIX_NEIGHBOURS
        )
        return self


@dataclass(frozen=True)
class GeneratedEnding:
    """How one generated game ended, reduced to what this family reads.

    ``resignation_deficit`` is in pawns and positive when the resigning player
    was behind, which is the direction decision 0017 measured the human
    population in.
    """

    rating: int
    category: str
    resignation_deficit: float | None
    #: Whether a draw was ever claimable and the game still never ended. The
    #: failure the claim action exists to prevent, and only meaningful in
    #: untimed play, where no clock resolves the position instead.
    claimable_and_unfinished: bool
    #: Terminal actions this game's seats actually selected.
    selected_terminal_actions: frozenset[int]

    def as_record(self) -> dict[str, Any]:
        """Return the per-game record stored in the detail tier."""

        return {
            "rating": self.rating,
            "category": self.category,
            "resignation_deficit": self.resignation_deficit,
            "claimable_and_unfinished": self.claimable_and_unfinished,
            "selected_terminal_actions": sorted(self.selected_terminal_actions),
        }


@dataclass(frozen=True)
class HumanEnding:
    """How one frozen human game ended, in the same terms.

    Rating follows the rule the generated-play reference already uses: a game
    sits at the mean of its two players' ratings, and a lopsided game is
    excluded rather than averaged into the middle, because its behavior belongs
    to neither player's level.
    """

    rating: float
    category: str
    speed: Speed | None
    resignation_deficit: float | None


@dataclass(frozen=True)
class DeficitBand:
    """One rating band's resignation-deficit comparison."""

    band: str
    model_games: int
    human_games: int
    model_median: float | None
    human_median: float | None
    distance: float | None
    model_distribution: dict[str, float]
    human_distribution: dict[str, float]

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one band's drill-down row."""

        return {
            "band": self.band,
            "model_games": self.model_games,
            "human_games": self.human_games,
            "model_median": self.model_median,
            "human_median": self.human_median,
            "distance": self.distance,
            "model_distribution": dict(self.model_distribution),
            "human_distribution": dict(self.human_distribution),
        }


@dataclass(frozen=True)
class ResignationDeficit:
    """How far behind each side was when it resigned.

    Bands rather than a curve, because a resignation deficit only exists for
    the games that ended in one: a curve smoothed over a subset that thin would
    report the subset's density rather than the model's behavior.
    """

    bands: tuple[DeficitBand, ...]
    model_games: int
    human_games: int
    model_median: float | None
    human_median: float | None
    #: Mean of the per-band distances over the bands both sides populate. None
    #: when no band has both, which is a real state rather than a zero.
    distance: float | None

    @property
    def median_gap(self) -> float | None:
        """Return model median minus human median, when both exist."""

        if self.model_median is None or self.human_median is None:
            return None
        return self.model_median - self.human_median

    def as_record(self) -> dict[str, Any]:
        """Return the deficit payload stored in the detail tier."""

        return {
            "bucket_edges": list(DEFICIT_BUCKET_EDGES),
            "model_games": self.model_games,
            "human_games": self.human_games,
            "model_median": self.model_median,
            "human_median": self.human_median,
            "distance": self.distance,
            "median_gap": self.median_gap,
            "bands": [band.as_record() for band in self.bands],
        }


@dataclass(frozen=True)
class Guardrails:
    """The readings whose failure modes are not symmetric.

    Each is reported explicitly rather than inferred from a distance, because a
    distributional distance treats resigning too often and resigning while
    winning as the same size of error, and they are not.
    """

    games: int
    resignations: int
    premature_resignations: int
    human_resignations: int
    human_premature_resignations: int
    #: Terminal actions the runtime enabled, and the subset never selected.
    enabled_terminal_actions: tuple[int, ...]
    silent_terminal_actions: tuple[int, ...]
    claimable_unfinished_games: int

    @property
    def premature_rate(self) -> float | None:
        """Return the share of the model's resignations that were premature."""

        return _rate(self.premature_resignations, self.resignations)

    @property
    def human_premature_rate(self) -> float | None:
        """Return the same rate over the human reference's resignations."""

        return _rate(self.human_premature_resignations, self.human_resignations)

    @property
    def untimed_non_termination_rate(self) -> float | None:
        """Return the share of games that could have ended and did not."""

        return _rate(self.claimable_unfinished_games, self.games)

    def as_record(self) -> dict[str, Any]:
        """Return the guardrail payload stored in the detail tier."""

        return {
            "games": self.games,
            "resignations": self.resignations,
            "premature_resignations": self.premature_resignations,
            "premature_rate": self.premature_rate,
            "human_resignations": self.human_resignations,
            "human_premature_resignations": self.human_premature_resignations,
            "human_premature_rate": self.human_premature_rate,
            "enabled_terminal_actions": list(self.enabled_terminal_actions),
            "silent_terminal_actions": list(self.silent_terminal_actions),
            "claimable_unfinished_games": self.claimable_unfinished_games,
            "untimed_non_termination_rate": self.untimed_non_termination_rate,
        }


@dataclass(frozen=True)
class TerminationMix:
    """One temperature's mix comparison against one human time-control class."""

    time_control: TimeControlClass
    temperature: float
    ratings: tuple[int, ...]
    comparison: CurveComparison
    execution: ExecutionRecord
    human_games: int
    model_games: int

    @property
    def label(self) -> str:
        """Return a short human label for this reading."""

        return f"{self.time_control} temperature={self.temperature:g}"

    def as_record(self) -> dict[str, Any]:
        """Return the mix payload stored in the detail tier."""

        return {
            "time_control": str(self.time_control),
            "temperature": self.temperature,
            "ratings": list(self.ratings),
            "human_games": self.human_games,
            "model_games": self.model_games,
            "human_only_categories": list(HUMAN_ONLY_CATEGORIES),
            "model_only_categories": list(MODEL_ONLY_CATEGORIES),
            "workload_sha256": self.execution.workload_sha256,
            "workload": dict(self.execution.workload),
            "comparison": self.comparison.as_detail_record(),
        }


@dataclass(frozen=True)
class GeneratedReading:
    """One temperature's whole generated side: guardrails and deficit."""

    temperature: float
    ratings: tuple[int, ...]
    games: int
    category_counts: dict[str, int]
    deficit: ResignationDeficit
    guardrails: Guardrails
    execution: ExecutionRecord
    #: Readings that had nothing to measure, with the reason. A real state:
    #: a model that never resigned has no deficit distribution, and reporting
    #: that is more useful than a zero a reader would take for a measurement.
    unavailable: dict[str, str] = field(default_factory=dict)
    endings: tuple[GeneratedEnding, ...] = field(default=(), repr=False)
    records: tuple[GameRecord, ...] = field(default=(), repr=False)

    @property
    def label(self) -> str:
        """Return a short human label for this reading."""

        return f"generated temperature={self.temperature:g}"

    def as_record(self) -> dict[str, Any]:
        """Return the generated payload stored in the detail tier."""

        return {
            "temperature": self.temperature,
            "ratings": list(self.ratings),
            "games": self.games,
            "category_counts": dict(sorted(self.category_counts.items())),
            "deficit": self.deficit.as_record(),
            "guardrails": self.guardrails.as_record(),
            "unavailable": dict(sorted(self.unavailable.items())),
            "workload_sha256": self.execution.workload_sha256,
            "workload": dict(self.execution.workload),
            "endings": [ending.as_record() for ending in self.endings],
            "games_detail": [record.as_record() for record in self.records],
        }


@dataclass(frozen=True)
class HeldOutResignation:
    """What the policy says about resigning at positions humans reached."""

    view: ViewSelection
    dataset: DatasetReference
    #: The projected content this pass scored. Series identity rather than
    #: provenance: nothing was generated here, so what the reading is scoped by
    #: is exactly the human decisions it read.
    data: DataComponent
    games: int
    resignation_plies: int
    move_plies: int
    mass_at_resignation: float | None
    mass_at_moves: float | None
    unavailable: dict[str, str] = field(default_factory=dict)
    decisions: tuple[TerminalActionPolicy, ...] = field(default=(), repr=False)

    @property
    def separation(self) -> float | None:
        """Return how much more mass sits where humans resigned."""

        if self.mass_at_resignation is None or self.mass_at_moves is None:
            return None
        return self.mass_at_resignation - self.mass_at_moves

    def as_record(self) -> dict[str, Any]:
        """Return the held-out payload stored in the detail tier."""

        return {
            "policy_scoring_version": POLICY_SCORING_VERSION,
            "view": self.view.as_record(),
            "games": self.games,
            "resignation_plies": self.resignation_plies,
            "move_plies": self.move_plies,
            "mass_at_resignation": self.mass_at_resignation,
            "mass_at_moves": self.mass_at_moves,
            "separation": self.separation,
            "unavailable": dict(sorted(self.unavailable.items())),
            "decisions": [decision.as_record() for decision in self.decisions],
        }


@dataclass(frozen=True)
class TerminationBenchmarkResult:
    """Everything one termination suite measured, and where it was written."""

    checkpoint: CheckpointReference
    reference_view: ViewSelection
    reference_games: int
    reference_excluded: dict[str, int]
    dataset: DatasetReference
    generated: tuple[GeneratedReading, ...]
    mixes: tuple[TerminationMix, ...] = ()
    held_out: HeldOutResignation | None = None
    unavailable: dict[str, str] = field(default_factory=dict)
    envelopes: tuple[ResultEnvelope, ...] = ()
    recorded_paths: tuple[Path, ...] = ()
    detail_paths: tuple[Path, ...] = ()

    @property
    def games(self) -> int:
        """Return how many games the whole suite generated."""

        return sum(reading.games for reading in self.generated)

    def mix(self, time_control: str, temperature: float) -> TerminationMix:
        """Return one time-control class's mix reading at one temperature."""

        for candidate in self.mixes:
            if (
                candidate.time_control == time_control
                and candidate.temperature == temperature
            ):
                return candidate
        raise TerminationBenchmarkError(
            f"no termination mix was measured for {time_control!r} at "
            f"temperature {temperature:g}"
        )

    def reading(self, temperature: float) -> GeneratedReading:
        """Return the generated reading at one temperature."""

        for candidate in self.generated:
            if candidate.temperature == temperature:
                return candidate
        raise TerminationBenchmarkError(
            f"no generated reading was measured at temperature {temperature:g}"
        )

    def as_record(self) -> dict[str, Any]:
        """Return the full structured result, detail tier included."""

        return {
            "version": TERMINATION_BENCHMARK_VERSION,
            "generation_version": GENERATION_VERSION,
            "categories": list(TERMINATION_MIX_CATEGORIES),
            "checkpoint": self.checkpoint.model_dump(mode="json"),
            "reference_view": self.reference_view.as_record(),
            "reference_games": self.reference_games,
            "reference_excluded": dict(sorted(self.reference_excluded.items())),
            "games": self.games,
            "generated": [reading.as_record() for reading in self.generated],
            "mixes": [mix.as_record() for mix in self.mixes],
            "held_out": None if self.held_out is None else self.held_out.as_record(),
            "unavailable": dict(sorted(self.unavailable.items())),
            "recorded": [str(path) for path in self.recorded_paths],
        }


def benchmark_termination(
    resolved_config: ResolvedConfig[TerminationBenchmarkConfig],
    *,
    run_root: Path | None = None,
    recording: ResultRecording,
    runner: ActionModelRunner | None = None,
    checkpoint: CheckpointReference | None = None,
) -> TerminationBenchmarkResult:
    """Measure how a checkpoint ends games, against how humans end theirs.

    Passing no ``store`` measures everything and records nothing, which is what
    a shakedown reading wants: real evidence about the instrument that does not
    belong in the committed history.
    """

    config = resolved_config.value
    loaded, identity = resolve_model(
        config.model,
        runner,
        checkpoint,
        label=config.checkpoint_label,
        run_root=run_root,
        error=TerminationBenchmarkError,
    )
    pool = _load_pool(config)
    reference, reference_view, dataset, reference_excluded = _load_reference(
        config, pool
    )

    generated = tuple(
        _measure_generated(config, loaded, identity, reference, temperature)
        for temperature in config.grid.temperatures
    )

    unavailable: dict[str, str] = {}
    mixes = _mix_readings(
        config, loaded, reference, reference_view, generated, unavailable
    )
    held_out = (
        _held_out_resignation(config, loaded, pool) if config.held_out.enabled else None
    )
    if held_out is None:
        unavailable["held_out_resignation"] = "the held-out reading is disabled"

    result = TerminationBenchmarkResult(
        checkpoint=identity,
        reference_view=reference_view,
        reference_games=len(reference),
        reference_excluded=reference_excluded,
        dataset=dataset,
        generated=generated,
        mixes=mixes,
        held_out=held_out,
        unavailable=unavailable,
    )
    _record(result, recording)
    return result


def mix_curve_spec(ratings: Sequence[int]) -> CurveSpec:
    """Return the mix comparison's frozen shape over the ratings played.

    The evaluation points are the conditioning ratings the suite generated
    games at, for the reason the generated-play comparison gives: a point the
    model was never asked to play has a human curve and nothing to compare it
    against. The bandwidth stays declared, because it is the smoothing rather
    than the points.
    """

    points = tuple(sorted(float(rating) for rating in dict.fromkeys(ratings)))
    if len(points) < 2:
        raise TerminationBenchmarkError(
            "a termination mix needs at least two conditioning ratings to be "
            "estimated over; a suite that played one rating has a point, not a "
            "curve"
        )
    return CurveSpec(
        name="game-termination-mix",
        version=MIX_CURVE_SPEC_VERSION,
        quantity=CurveQuantity.CATEGORICAL,
        neighbours=DECLARED_MIX_NEIGHBOURS,
        grid=points,
    )


def generated_ending(record: GameRecord) -> GeneratedEnding:
    """Reduce one generated game to the ending quantities this family reads.

    Replaying is what makes the deficit and the claim guardrail exact rather
    than inferred from the outcome the harness wrote: the material behind a
    resignation is a property of the final position, and whether a draw was
    ever claimable is a property of the whole trajectory.
    """

    rating = _seat_rating(record)
    board = chess.Board(record.initial_position)
    claimable = False
    for action_id in record.action_ids:
        if is_terminal_action(action_id):
            break
        if draw_claim_available(board):
            claimable = True
        board.push(_move(action_id))
    if draw_claim_available(board):
        claimable = True

    termination = record.outcome.termination
    deficit: float | None = None
    if termination is GameTermination.RESIGNATION:
        # The terminal action belongs to the player holding the move, so the
        # position the game stopped at is the resigning player's own.
        deficit = float(-material_balance(board, board.turn))
    return GeneratedEnding(
        rating=rating,
        category=termination.value,
        resignation_deficit=deficit,
        claimable_and_unfinished=(
            claimable and termination is GameTermination.PLY_LIMIT
        ),
        selected_terminal_actions=frozenset(
            action_id
            for action_id in record.action_ids
            if is_terminal_action(action_id)
        ),
    )


def human_ending(
    row: Mapping[str, Any],
    config: ReferenceConfig,
) -> tuple[HumanEnding | None, str | None]:
    """Reduce one frozen pool row to the same quantities, or say why not."""

    white = row.get(NormalizedColumn.WHITE_NORMALIZED_RATING.value)
    black = row.get(NormalizedColumn.BLACK_NORMALIZED_RATING.value)
    if white is None or black is None:
        return None, "missing_ratings"
    if abs(int(white) - int(black)) > config.maximum_rating_gap:
        return None, "rating_gap"
    action_ids = row.get(NormalizedColumn.ACTION_IDS.value)
    if not action_ids:
        return None, "no_moves"
    category = row.get(NormalizedColumn.TERMINATION_CATEGORY.value)
    if not category:
        return None, "missing_termination_category"

    deficit: float | None = None
    if category == TerminationCategory.RESIGNATION.value:
        loser = _loser(str(row[NormalizedColumn.RESULT.value]))
        if loser is not None:
            board = _replayed_board(row)
            deficit = float(-material_balance(board, loser))
    return (
        HumanEnding(
            rating=(float(white) + float(black)) / 2.0,
            category=str(category),
            speed=speed_from_clock_ms(
                row.get(NormalizedColumn.TIME_INITIAL_MS.value),
                row.get(NormalizedColumn.TIME_INCREMENT_MS.value),
            ),
            resignation_deficit=deficit,
        ),
        None,
    )


def _measure_generated(
    config: TerminationBenchmarkConfig,
    runner: ActionModelRunner,
    checkpoint: CheckpointReference,
    reference: Sequence[HumanEnding],
    temperature: float,
) -> GeneratedReading:
    """Play one temperature's whole rating grid and read its endings."""

    endings: list[GeneratedEnding] = []
    records: list[GameRecord] = []
    positions = standard_positions(label="standard-start")
    for rating in config.grid.target_ratings:
        runtime = config.runtime.model_copy(
            update={"target_rating": rating, "temperature": temperature}
        )
        player = ModelPlayer(
            runner,
            label=f"{checkpoint.label}-r{rating}-t{temperature:g}",
            config=runtime,
            checkpoint=checkpoint,
        )
        seeds, generation = collapse_replicates(
            config.grid.seeds, config.generation, temperatures=(temperature,)
        )
        for seed in seeds:
            played = _generate(
                player, positions, generation.model_copy(update={"seed": seed})
            )
            endings.extend(generated_ending(record) for record in played)
            if config.detail.retain_games:
                records.extend(played)

    unavailable: dict[str, str] = {}
    deficit = _resignation_deficit(endings, reference)
    if deficit.model_games == 0:
        unavailable["resignation_deficit"] = (
            "the model never resigned, so it has no deficit distribution; the "
            "silent-terminal-action count is the reading that says so"
        )
    elif deficit.distance is None:
        unavailable["resignation_deficit_distance"] = (
            "no rating band holds both a model and a human resignation"
        )
    guardrails = _guardrails(config, endings, reference)
    if not guardrails.enabled_terminal_actions:
        unavailable["silent_terminal_actions"] = (
            "the runtime enabled no terminal action, so there is none the model "
            "could have left unused"
        )

    reading = GeneratedReading(
        temperature=temperature,
        ratings=tuple(config.grid.target_ratings),
        games=len(endings),
        category_counts=_counts(ending.category for ending in endings),
        deficit=deficit,
        guardrails=guardrails,
        execution=_generated_execution(config, runner, temperature),
        unavailable=unavailable,
        endings=tuple(endings),
        records=tuple(records),
    )
    logger.info(
        "Termination reading %s: %s game(s), %s resignation(s), %s premature",
        reading.label,
        reading.games,
        guardrails.resignations,
        guardrails.premature_resignations,
    )
    return reading


def _mix_readings(
    config: TerminationBenchmarkConfig,
    runner: ActionModelRunner,
    reference: Sequence[HumanEnding],
    reference_view: ViewSelection,
    generated: Sequence[GeneratedReading],
    unavailable: dict[str, str],
) -> tuple[TerminationMix, ...]:
    """Compare each temperature's endings against each human time-control class."""

    spec = mix_curve_spec(config.grid.target_ratings)
    mixes: list[TerminationMix] = []
    for time_control in config.time_controls:
        human = tuple(
            Observation(rating=ending.rating, value=ending.category)
            for ending in reference
            if time_control == "all" or ending.speed == time_control
        )
        if len(human) < spec.neighbours:
            unavailable[f"mix:{time_control}"] = (
                f"the {time_control} reference holds {len(human)} game(s) "
                f"and the declared bandwidth smooths over {spec.neighbours}"
            )
            continue
        for reading in generated:
            model = tuple(
                Observation(rating=float(ending.rating), value=ending.category)
                for ending in reading.endings
            )
            try:
                comparison = compare_curves(
                    spec=spec,
                    human=human,
                    model=model,
                    resamples=config.reference.resamples,
                    seed=config.reference.seed,
                    model_varies=replicates_vary((reading.temperature,)),
                )
            except CurveComparisonError as error:
                unavailable[f"mix:{time_control}:t{reading.temperature:g}"] = str(error)
                continue
            mixes.append(
                TerminationMix(
                    time_control=time_control,
                    temperature=reading.temperature,
                    ratings=reading.ratings,
                    comparison=comparison,
                    execution=_mix_execution(
                        config,
                        runner,
                        time_control,
                        reading.temperature,
                        reference_view,
                    ),
                    human_games=len(human),
                    model_games=len(model),
                )
            )
    return tuple(mixes)


def _held_out_resignation(
    config: TerminationBenchmarkConfig,
    runner: ActionModelRunner,
    pool: FrozenPool,
) -> HeldOutResignation:
    """Score every ply of the held-out view for its resignation mass."""

    selection = apply_view(pool.games, config.held_out.view)
    if not selection.game_ids:
        raise TerminationBenchmarkError(
            f"view {config.held_out.view.name!r} selected no human games to score"
        )
    rows = _rows_for(pool, selection.game_ids)
    scorer = _scoring_runner(runner)
    decisions = _score_decisions(config, scorer, rows, selection)
    component = _projection_component(rows)

    at_resignation = [
        decision.resignation_mass
        for decision in decisions
        if decision.target_action_id == RESIGNATION_ACTION_ID
    ]
    at_moves = [
        decision.resignation_mass
        for decision in decisions
        if not decision.target_is_terminal
    ]
    unavailable: dict[str, str] = {}
    if not at_resignation:
        unavailable["resignation_mass_at_resignation"] = (
            "no game in the held-out view carries a resignation action, so the "
            "policy has no human resignation to be scored against"
        )
    return HeldOutResignation(
        view=selection,
        dataset=_dataset_reference(pool, selection, rows),
        data=component,
        games=len({decision.game_id for decision in decisions}),
        resignation_plies=len(at_resignation),
        move_plies=len(at_moves),
        mass_at_resignation=_mean(at_resignation),
        mass_at_moves=_mean(at_moves),
        unavailable=unavailable,
        decisions=tuple(decisions) if config.detail.retain_decisions else (),
    )


def _score_decisions(
    config: TerminationBenchmarkConfig,
    runner: ScoringModelRunner,
    rows: Sequence[Mapping[str, Any]],
    selection: ViewSelection,
) -> tuple[TerminalActionPolicy, ...]:
    """Run one deterministic pass over the view and keep the terminal mass."""

    try:
        inputs = build_scoring_inputs(
            rows,
            split="test",
            batch_size=config.held_out.loader.batch_size,
            length_bucket_width=config.held_out.loader.length_bucket_width,
            identity_sha256=rows_identity_sha256(rows, context=selection.name),
        )
    except ScoringError as error:
        raise TerminationBenchmarkError(str(error)) from error

    device = runner_device(runner)
    scored: list[TerminalActionPolicy] = []
    loader = SequenceDataLoader(inputs.dataset, inputs.loader_config)
    for sequence_batch in loader:
        batch = MoveModelBatch.from_sequence_batch(sequence_batch, device=device)
        scored.extend(
            score_terminal_actions(active_batch(runner.action_logits(batch), batch))
        )
    if not scored:
        raise TerminationBenchmarkError(
            "the configured held-out view selected no positions to score"
        )
    return tuple(scored)


def _resignation_deficit(
    endings: Sequence[GeneratedEnding],
    reference: Sequence[HumanEnding],
) -> ResignationDeficit:
    """Compare the two deficit distributions band by band."""

    model_by_band: dict[str, list[float]] = {}
    human_by_band: dict[str, list[float]] = {}
    for ending in endings:
        if ending.resignation_deficit is not None:
            model_by_band.setdefault(_band(ending.rating), []).append(
                ending.resignation_deficit
            )
    for human in reference:
        if human.resignation_deficit is not None:
            human_by_band.setdefault(_band(human.rating), []).append(
                human.resignation_deficit
            )

    bands: list[DeficitBand] = []
    distances: list[float] = []
    for name in sorted({*model_by_band, *human_by_band}):
        model_values = model_by_band.get(name, [])
        human_values = human_by_band.get(name, [])
        model_shares = _bucket_shares(model_values)
        human_shares = _bucket_shares(human_values)
        distance = (
            distribution_distance(human_shares, model_shares)
            if model_values and human_values
            else None
        )
        if distance is not None:
            distances.append(distance)
        bands.append(
            DeficitBand(
                band=name,
                model_games=len(model_values),
                human_games=len(human_values),
                model_median=_median(model_values),
                human_median=_median(human_values),
                distance=distance,
                model_distribution=model_shares,
                human_distribution=human_shares,
            )
        )

    model_all = [value for values in model_by_band.values() for value in values]
    human_all = [value for values in human_by_band.values() for value in values]
    return ResignationDeficit(
        bands=tuple(bands),
        model_games=len(model_all),
        human_games=len(human_all),
        model_median=_median(model_all),
        human_median=_median(human_all),
        distance=_mean(distances),
    )


def _guardrails(
    config: TerminationBenchmarkConfig,
    endings: Sequence[GeneratedEnding],
    reference: Sequence[HumanEnding],
) -> Guardrails:
    """Count the three guardrails over one temperature's games."""

    threshold = config.guardrails.premature_material_balance
    model_deficits = [
        ending.resignation_deficit
        for ending in endings
        if ending.resignation_deficit is not None
    ]
    human_deficits = [
        human.resignation_deficit
        for human in reference
        if human.resignation_deficit is not None
    ]
    enabled = tuple(
        action_id
        for action_id, allowed in (
            (RESIGNATION_ACTION_ID, config.runtime.resignation_enabled),
            (DRAW_CLAIM_ACTION_ID, config.runtime.draw_claim_enabled),
        )
        if allowed
    )
    selected = frozenset(
        action_id
        for ending in endings
        for action_id in ending.selected_terminal_actions
    )
    return Guardrails(
        games=len(endings),
        resignations=len(model_deficits),
        # "Not lost" by the material proxy: the resigning player was level or
        # ahead, which is the egregious case material alone can honestly claim.
        premature_resignations=sum(
            1 for deficit in model_deficits if -deficit >= threshold
        ),
        human_resignations=len(human_deficits),
        human_premature_resignations=sum(
            1 for deficit in human_deficits if -deficit >= threshold
        ),
        enabled_terminal_actions=enabled,
        silent_terminal_actions=tuple(
            action_id for action_id in enabled if action_id not in selected
        ),
        claimable_unfinished_games=sum(
            1 for ending in endings if ending.claimable_and_unfinished
        ),
    )


def _load_pool(config: TerminationBenchmarkConfig) -> FrozenPool:
    """Load the frozen pool every reading here is measured against."""

    try:
        return load_pool(
            config.pool,
            expected_game_ids_sha256=config.expected_pool_game_ids_sha256,
        )
    except EvaluationPoolError as error:
        raise TerminationBenchmarkError(str(error)) from error


def _load_reference(
    config: TerminationBenchmarkConfig,
    pool: FrozenPool,
) -> tuple[tuple[HumanEnding, ...], ViewSelection, DatasetReference, dict[str, int]]:
    """Read the human endings the mix and the deficit are measured against.

    Why a game was dropped is returned alongside the endings rather than
    recomputed for the record: deriving it a second time re-read every pool
    row to reach the same answer this pass already had.
    """

    selection = apply_view(pool.games, config.reference.view)
    if not selection.game_ids:
        raise TerminationBenchmarkError(
            f"view {config.reference.view.name!r} selected no human games to "
            "compare against"
        )
    rows = _rows_for(pool, selection.game_ids)
    endings: list[HumanEnding] = []
    excluded: dict[str, int] = {}
    for row in rows:
        ending, reason = human_ending(row, config.reference)
        if ending is None:
            excluded[reason or "unusable"] = excluded.get(reason or "unusable", 0) + 1
            continue
        endings.append(ending)
    if not endings:
        raise TerminationBenchmarkError(
            "no human game in the reference view carries the ratings and derived "
            "termination a mix comparison needs"
        )
    logger.info(
        "Human termination reference: %s game(s), %s excluded",
        len(endings),
        sum(excluded.values()),
    )
    required = minimum_reference_games(
        config.grid.target_ratings, DECLARED_MIX_NEIGHBOURS
    )
    # Said in the pool pass rather than only in the per-class unavailable lines
    # the run ends with. The declared cap clears this floor by construction, so
    # reaching here means the pool or the rating-gap filter did it, which is a
    # property of the corpus rather than of configuration. Not an error: the
    # guardrails, the deficit, and the held-out reading need no curve at all,
    # and discarding them would lose more than the mix is worth.
    if len(endings) < required:
        logger.warning(
            "Human termination reference holds %s usable game(s), below the %s "
            "a mix curve over this rating grid needs at a bandwidth of %s "
            "neighbours; the mix will report as unavailable",
            len(endings),
            required,
            DECLARED_MIX_NEIGHBOURS,
        )
    return (
        tuple(endings),
        selection,
        _dataset_reference(pool, selection, rows),
        excluded,
    )


def _rows_for(pool: FrozenPool, game_ids: Sequence[int]) -> tuple[dict[str, Any], ...]:
    """Return the selected pool rows, in ascending game-id order."""

    return pool_rows(
        pool,
        game_ids,
        _ENDING_COLUMNS,
        error=TerminationBenchmarkError,
    )


def _dataset_reference(
    pool: FrozenPool,
    selection: ViewSelection,
    rows: Sequence[Mapping[str, Any]],
) -> DatasetReference:
    """Describe the human games one reading read."""

    return pool_dataset_reference(
        pool,
        selection,
        _projection_component(rows),
        error=TerminationBenchmarkError,
    )


def _projection_component(rows: Sequence[Mapping[str, Any]]) -> DataComponent:
    """Return the digest of the content one reading actually consumed."""

    try:
        return projection_content_digest(rows, TERMINATION_PREDICTION_PROJECTION)
    except FingerprintError as error:
        raise TerminationBenchmarkError(str(error)) from error


def _generated_execution(
    config: TerminationBenchmarkConfig,
    runner: ActionModelRunner,
    temperature: float,
) -> ExecutionRecord:
    """Declare what one temperature's generated readings measured.

    Whether the terminal actions were enabled belongs here rather than in the
    coordinates, and it is the field that matters most: a suite played with
    resignation disabled measures a model that was never offered the choice,
    and its silent-non-use reading would otherwise sit on the same series as
    one that was.
    """

    return execution_record(
        runner_device(runner),
        {
            "generation_version": GENERATION_VERSION,
            "positions": {"kind": "standard-start"},
            "target_ratings": list(config.grid.target_ratings),
            "temperature": temperature,
            "maximum_generated_plies": config.generation.maximum_generated_plies,
            "swap_colors": config.generation.swap_colors,
            "claim_draws": config.generation.claim_draws,
            "resignation_enabled": config.runtime.resignation_enabled,
            "draw_claim_enabled": config.runtime.draw_claim_enabled,
            "premature_material_balance": (
                config.guardrails.premature_material_balance
            ),
            "deficit_bucket_edges": list(DEFICIT_BUCKET_EDGES),
        },
    )


def _mix_execution(
    config: TerminationBenchmarkConfig,
    runner: ActionModelRunner,
    time_control: TimeControlClass,
    temperature: float,
    reference_view: ViewSelection,
) -> ExecutionRecord:
    """Declare what one mix comparison measured.

    The time-control class joins the workload because it decides which human
    population the distance is to. Two classes are two different questions
    rather than two samples of one, so they must not share a series.

    The reference joins it for a related but stronger reason, which
    ``reference_workload`` gives: the bandwidth is a neighbour count, so two
    reference sizes are two smoothings rather than two samples of one.
    """

    return execution_record(
        runner_device(runner),
        {
            "generation_version": GENERATION_VERSION,
            "positions": {"kind": "standard-start"},
            "target_ratings": list(config.grid.target_ratings),
            "temperature": temperature,
            "maximum_generated_plies": config.generation.maximum_generated_plies,
            "swap_colors": config.generation.swap_colors,
            "claim_draws": config.generation.claim_draws,
            "resignation_enabled": config.runtime.resignation_enabled,
            "draw_claim_enabled": config.runtime.draw_claim_enabled,
            "time_control": str(time_control),
            "curve_spec_version": MIX_CURVE_SPEC_VERSION,
            "neighbours": DECLARED_MIX_NEIGHBOURS,
            "reference": reference_workload(config.reference, reference_view),
        },
    )


def _record(
    result: TerminationBenchmarkResult,
    recording: ResultRecording,
) -> None:
    """Write one envelope per generated reading, per mix, and per held-out pass.

    Three units and therefore three records. A generated reading is scoped by
    the recipe its games were played under; a mix additionally by the human
    population it was compared against; the held-out pass by the content it
    scored, since it generated nothing at all.
    """

    recorder = recording.measuring(
        result.checkpoint,
        kind=TERMINATION_KIND,
        benchmark=TERMINATION_BENCHMARK,
    )
    for reading in result.generated:
        recorder.add(
            _generated_measurements(reading),
            payload=reading.as_record,
            description=f"Game termination: {reading.label}",
            slug=f"generated-t{_slug(reading.temperature)}",
            # Provenance rather than identity: the human reference shaped
            # the guardrail's comparison rate and the deficit's bands, but
            # what identifies the series is the recipe the games were
            # played under, per decision 0020.
            data=result.dataset,
            execution=reading.execution,
        )
    for mix in result.mixes:
        recorder.add(
            _mix_measurements(mix),
            payload=mix.as_record,
            description=f"Termination mix: {mix.label}",
            slug=f"mix-{mix.time_control}-t{_slug(mix.temperature)}",
            data=result.dataset,
            execution=mix.execution,
        )
    held_out = result.held_out
    if held_out is not None:
        # A pass that measured nothing still writes its payload, which is
        # what `add` does with empty measurements: the reading itself is
        # the evidence for why there is nothing to commit.
        recorder.add(
            _held_out_measurements(held_out),
            payload=held_out.as_record,
            description="Held-out resignation prediction",
            slug="held-out-resignation",
            # Series identity here, not provenance: this reading is a
            # deterministic pass over fixed content and generated nothing,
            # so the content is what it is scoped by.
            data=held_out.dataset,
        )


def _generated_measurements(reading: GeneratedReading) -> tuple[Measurement, ...]:
    """Return one temperature's guardrails and deficit readings.

    A reading with nothing behind it is omitted rather than recorded as a zero.
    That is the whole point of the unavailable map beside it: a model that never
    resigned has no median deficit, and a zero there would read as resigning
    while exactly level.
    """

    workload = reading.execution.workload_component()
    deficit = reading.deficit
    guardrails = reading.guardrails
    values: tuple[tuple[str, float | None, int | None], ...] = (
        (
            TERMINATION_RESIGNATION_DEFICIT_MEDIAN.identifier,
            deficit.model_median,
            deficit.model_games,
        ),
        (
            TERMINATION_RESIGNATION_DEFICIT_DISTANCE.identifier,
            deficit.distance,
            deficit.model_games,
        ),
        (
            TERMINATION_RESIGNATION_DEFICIT_GAP.identifier,
            deficit.median_gap,
            deficit.model_games,
        ),
        (
            TERMINATION_PREMATURE_RESIGNATION_RATE.identifier,
            guardrails.premature_rate,
            guardrails.resignations,
        ),
        (
            TERMINATION_PREMATURE_RESIGNATION_HUMAN_RATE.identifier,
            guardrails.human_premature_rate,
            guardrails.human_resignations,
        ),
        (
            TERMINATION_SILENT_TERMINAL_ACTIONS.identifier,
            (
                None
                if not guardrails.enabled_terminal_actions
                else float(len(guardrails.silent_terminal_actions))
            ),
            guardrails.games,
        ),
        (
            TERMINATION_UNTIMED_NON_TERMINATION_RATE.identifier,
            guardrails.untimed_non_termination_rate,
            guardrails.games,
        ),
    )
    return tuple(
        measurement(
            identifier,
            value,
            workload=workload,
            sample_size=sample_size if sample_size else None,
        )
        for identifier, value, sample_size in values
        if value is not None
    )


def _mix_measurements(mix: TerminationMix) -> tuple[Measurement, ...]:
    """Return one mix comparison's distances, each with the floor to beat."""

    return mix.comparison.measurements(
        CurveMetrics(
            conditional=TERMINATION_MIX_CONDITIONAL_DISTANCE.identifier,
            pooled=TERMINATION_MIX_POOLED_DISTANCE.identifier,
            model_variation=TERMINATION_MIX_RATING_VARIATION.identifier,
        ),
        workload=mix.execution.workload_component(),
    )


def _held_out_measurements(reading: HeldOutResignation) -> tuple[Measurement, ...]:
    """Return the held-out resignation readings the summary tier records.

    Sample sizes are per metric rather than shared. The two halves are averaged
    over populations that differ by orders of magnitude — a handful of
    resignation plies against every move ply in the view — and reporting the
    larger count for both would overstate the first one's precision to every
    reader, the noise-floor layer included.
    """

    data = reading.data
    values: tuple[tuple[str, float | None, int], ...] = (
        (
            TERMINATION_RESIGNATION_MASS_AT_RESIGNATION.identifier,
            reading.mass_at_resignation,
            reading.resignation_plies,
        ),
        (
            TERMINATION_RESIGNATION_MASS_AT_MOVES.identifier,
            reading.mass_at_moves,
            reading.move_plies,
        ),
        (
            TERMINATION_RESIGNATION_MASS_SEPARATION.identifier,
            reading.separation,
            reading.resignation_plies,
        ),
    )
    return tuple(
        measurement(
            identifier,
            value,
            data=data,
            sample_size=sample_size if sample_size else None,
        )
        for identifier, value, sample_size in values
        if value is not None
    )


def _scoring_runner(runner: ActionModelRunner) -> ScoringModelRunner:
    """Return the runner as a batch scorer, or say it cannot be one."""

    if not callable(getattr(runner, "action_logits", None)):
        raise TerminationBenchmarkError(
            "the held-out resignation reading needs a runner that scores whole "
            "batches; set held_out.enabled to false to measure generated play "
            "alone"
        )
    return runner  # type: ignore[return-value]


def _generate(
    player: ModelPlayer,
    positions: Sequence[StartPosition],
    generation: GenerationConfig,
) -> tuple[GameRecord, ...]:
    """Play one seed's suite as self-play, one configuration in both seats."""

    try:
        return tuple(generate_games(player, player, positions, config=generation))
    except (GenerationError, PlayerError) as error:
        raise TerminationBenchmarkError(f"cannot generate games: {error}") from error


def _seat_rating(record: GameRecord) -> int:
    """Return the conditioning rating a self-play record was generated at."""

    for seat in (record.white, record.black):
        rating = seat.configuration.get("target_rating")
        if isinstance(rating, int):
            return rating
    raise TerminationBenchmarkError(
        "a generated game carries no conditioning rating to place it at"
    )


def _replayed_board(row: Mapping[str, Any]) -> chess.Board:
    """Return the final position of one normalized game, with its move stack."""

    board = chess.Board(str(row[NormalizedColumn.INITIAL_POSITION.value]))
    for action_id in row[NormalizedColumn.ACTION_IDS.value]:
        if is_terminal_action(int(action_id)):
            break
        board.push(_move(int(action_id)))
    return board


def _move(action_id: int) -> chess.Move:
    # Imported here rather than at module scope so the codec is not pulled in
    # by a reader that only wants this module's category vocabulary.
    from anthro_chess.chess import decode_move

    return decode_move(action_id)


def _loser(result: str) -> chess.Color | None:
    """Return which color lost a decided game."""

    return {"1-0": chess.BLACK, "0-1": chess.WHITE}.get(result)


def _band(rating: float) -> str:
    """Return the default rating band one game sits in.

    The default bands rather than a set of this benchmark's own, so a deficit
    drill-down lines up with every other band-sliced reading in the store.
    """

    return rating_band_name(int(rating)) or "unknown"


def _bucket_shares(values: Sequence[float]) -> dict[str, float]:
    """Return the deficit distribution over declared pawn buckets."""

    if not values:
        return {}
    counts: dict[str, int] = {}
    for value in values:
        counts[_bucket(value)] = counts.get(_bucket(value), 0) + 1
    return {name: count / len(values) for name, count in sorted(counts.items())}


def _bucket(deficit: float) -> str:
    """Return the bucket one deficit falls in, named by its own bounds."""

    edges = DEFICIT_BUCKET_EDGES
    if deficit < edges[0]:
        return f"below-{_edge(edges[0])}"
    for low, high in zip(edges, edges[1:], strict=False):
        if deficit < high:
            return f"{_edge(low)}-to-{_edge(high)}"
    return f"{_edge(edges[-1])}-and-above"


def _edge(value: float) -> str:
    """Return a stable, filename-safe name for one bucket boundary."""

    return f"{value:g}".replace("-", "minus").replace(".", "_")


def _counts(values: Iterator[str]) -> dict[str, int]:
    """Return how often each value occurred, in ascending key order."""

    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mean(values: Sequence[float]) -> float | None:
    """Return the mean, or ``None`` when nothing was averaged."""

    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    """Return the median, or ``None`` when nothing was measured."""

    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _rate(part: int, whole: int) -> float | None:
    """Return a share, or ``None`` when there was nothing to take a share of."""

    return part / whole if whole else None


def _slug(value: float) -> str:
    """Return a filename-safe form of a temperature."""

    return str(value).replace(".", "_")


__all__ = [
    "DECLARED_MIX_NEIGHBOURS",
    "DEFICIT_BUCKET_EDGES",
    "HUMAN_ONLY_CATEGORIES",
    "MIX_CURVE_SPEC_VERSION",
    "MODEL_ONLY_CATEGORIES",
    "TERMINATION_BENCHMARK",
    "TERMINATION_BENCHMARK_VERSION",
    "TERMINATION_KIND",
    "TERMINATION_MIX_CATEGORIES",
    "DeficitBand",
    "GeneratedEnding",
    "GeneratedReading",
    "GuardrailConfig",
    "Guardrails",
    "HeldOutResignation",
    "HeldOutResignationConfig",
    "HumanEnding",
    "ResignationDeficit",
    "TerminationBenchmarkConfig",
    "TerminationBenchmarkError",
    "TerminationBenchmarkResult",
    "TerminationDetailConfig",
    "TerminationGridConfig",
    "TerminationMix",
    "TerminationReferenceConfig",
    "TimeControlClass",
    "benchmark_termination",
    "generated_ending",
    "human_ending",
    "mix_curve_spec",
]
