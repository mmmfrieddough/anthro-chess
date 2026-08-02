"""Capability retention under a controlled novelty dose.

The model degrades on positions unlike those it trained on. Slicing the pool by
a familiarity proxy cannot measure that, because the pool is human games and is
therefore in distribution nearly by construction. Perturbation replaces
detection: deriving positions by perturbing pool games supplies novelty at a
known dose, so there is nothing to detect, validate, or hold stable across
checkpoints.

Three properties of the derivation carry the whole design.

**It is one-sided.** Only the opponent's moves are replaced, which is the
situation being measured — someone playing garbage to knock the engine out of
distribution — where the model still chooses its own moves. Perturbing both
sides measures a situation nobody will create.

**Divergence is absorbing.** Once one opponent move has been replaced, the
human's later opponent moves belong to a game that no longer exists, so every
later opponent move in the window is random too. The configured dose is the
per-move rate at which divergence starts, and the realized share of replaced
moves is reported beside every reading.

**The human side is replayed while it stays legal.** The player's continuation
is the human's own, which keeps the arm model-independent, and it ends the
moment the human's move is no longer legal in the diverged position. That
truncation is a real selection effect rather than a defect, so the share of the
control arm's positions that survived is a reported metric.

What can be measured on the derived arms is limited by the same divergence.
Held-out move prediction is undefined once a prefix stops being what the humans
played, so it is reported on the control arm only. Legality needs no target and
survives; so do the predicates exact chess logic resolves. On perturbed arms the
reference is the model's own unperturbed rate on the same positions, so results
there are reported as retention rather than as an absolute rate.

See ``docs/evaluation.md`` and
``docs/decisions/0024-one-sided-perturbation-derived-novelty.md``.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any, cast

import chess
from pydantic import Field, StrictBool, StrictInt, model_validator

from anthro_chess.chess import decode_move, encode_move, is_terminal_action
from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import DataLoadingError, SequenceDataLoader
from anthro_chess.data.artifacts import read_normalized_rows
from anthro_chess.data.schema import SPLIT_NAMES, NormalizedColumn, SplitName
from anthro_chess.evaluation.adjudication import action_sets, merge_game_totals
from anthro_chess.evaluation.aggregation import PHASE_DIMENSION, SliceTable
from anthro_chess.evaluation.checkpoint import (
    CheckpointEvaluationError,
    DetailConfig,
    LeakageConfig,
)
from anthro_chess.evaluation.cost import benchmark_cost_result
from anthro_chess.evaluation.execution import execution_record
from anthro_chess.evaluation.leakage import LeakageCheck, check_leakage
from anthro_chess.evaluation.noise import (
    GameTotals,
    MetricTotal,
    NoiseConfig,
    characterize_sampling_noise,
)
from anthro_chess.evaluation.policy import (
    POLICY_SCORING_VERSION,
    ActionSetPolicy,
    PositionPolicy,
    score_action_sets,
    score_positions,
)
from anthro_chess.evaluation.pool import EvaluationPoolError, FrozenPool, load_pool
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
    DatasetReference,
    DetailReference,
    DetailStore,
    ExecutionRecord,
    Measurement,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    ResultsStoreError,
    WorkloadComponent,
    build_result,
    configuration_reference,
    dataset_reference,
    default_checkpoint_label,
    measurement,
    projection_content_digest,
)
from anthro_chess.evaluation.results.metrics import (
    MOVE_PREDICTION_PROJECTION,
    NOVELTY_DERIVED_PLY_RETENTION,
    NOVELTY_LEGAL_MASS,
    NOVELTY_LEGAL_MASS_RETENTION,
    NOVELTY_MASK_PENALTY,
    NOVELTY_MASK_PENALTY_BY_PHASE,
    NOVELTY_MASK_PENALTY_RATIO,
    NOVELTY_PREDICATE_BEST_RANK,
    NOVELTY_PREDICATE_POLICY_MASS,
    NOVELTY_PREDICATE_RETENTION,
    NOVELTY_PREDICATE_SELECTED_RATE,
    NOVELTY_REALIZED_DOSE,
    MetricDefinition,
)
from anthro_chess.evaluation.results.noise import (
    NoiseCharacterization,
    NoiseCharacterizationError,
)
from anthro_chess.evaluation.scoring import (
    EvaluationLoaderConfig,
    ScoringInputs,
    aggregate_positions,
    build_scoring_inputs,
    rows_identity_sha256,
)
from anthro_chess.evaluation.slices import SLICE_SCHEME_VERSION, PositionPredicate
from anthro_chess.evaluation.views import ViewConfig, ViewSelection, apply_view
from anthro_chess.inference import CheckpointModelRunner, ModelRunnerConfig
from anthro_chess.inference.runner import ModelRunnerError
from anthro_chess.models import MoveModelBatch

#: Bumped when the derivation changes what a dose means. Results are not
#: comparable across recipe versions, which is why this joins the workload.
NOVELTY_RECIPE_VERSION = 1
NOVELTY_BENCHMARK_VERSION = 1
NOVELTY_KIND = "novelty-dose-response"
NOVELTY_BENCHMARK = BenchmarkReference(
    name=NOVELTY_KIND,
    version=NOVELTY_BENCHMARK_VERSION,
)

#: The dose at which no opponent move is replaced.
CONTROL_DOSE = 0.0

logger = logging.getLogger(__name__)


class NoveltyBenchmarkError(ValueError):
    """Raised when a novelty dose response cannot be measured safely."""


class PerturbationRecipe(StrEnum):
    """How novelty is introduced into a derived continuation.

    Only one recipe exists, and sampling from the model's own low-probability
    tail is deliberately not among the candidates: it would make the derived
    positions model-dependent, which defeats the multi-checkpoint trend this
    benchmark exists to report.
    """

    RANDOM_LEGAL_OPPONENT = "random-legal-opponent"


class PerturbationConfig(ConfigModel):
    """The derivation every arm of one suite shares, apart from its dose."""

    recipe: PerturbationRecipe = PerturbationRecipe.RANDOM_LEGAL_OPPONENT
    seed: str = Field(default="anthro-novelty-v1", min_length=1)
    #: Where the measurement window opens. Positions before it are the human
    #: game on every arm, which is what keeps the arms comparable.
    onset_plies: StrictInt = Field(default=16, ge=1)
    #: How many of the opponent's moves the window covers. Each one is followed
    #: by the player reply this benchmark scores.
    window_moves: StrictInt = Field(default=8, ge=1)
    #: The sweep. Zero is the control arm and is required, so every perturbed
    #: reading has a baseline rather than an absolute scale.
    doses: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 1.0)

    @model_validator(mode="after")
    def _validate_doses(self) -> PerturbationConfig:
        if len(set(self.doses)) != len(self.doses):
            raise ValueError("each perturbation dose may appear once")
        if any(not 0.0 <= dose <= 1.0 for dose in self.doses):
            raise ValueError("a perturbation dose is a rate between zero and one")
        if CONTROL_DOSE not in self.doses:
            raise ValueError(
                "a novelty sweep must include the unperturbed control arm at "
                "dose 0.0; without it a perturbed reading has no reference"
            )
        if len(self.doses) < 2:
            raise ValueError("a dose sweep needs a perturbed arm beside the control")
        return self


class NoveltyBenchmarkConfig(ConfigModel):
    """Code-owned schema for ``anthro eval novelty``."""

    pool: Path
    view: ViewConfig = ViewConfig(name="novelty")
    model: ModelRunnerConfig = ModelRunnerConfig()
    loader: EvaluationLoaderConfig = EvaluationLoaderConfig()
    checkpoint_label: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    perturbation: PerturbationConfig = PerturbationConfig()
    leakage: LeakageConfig = LeakageConfig()
    detail: DetailConfig = DetailConfig()
    noise: NoiseConfig = NoiseConfig()


PositionKey = tuple[int, int]


@dataclass(frozen=True)
class DerivedGame:
    """One pool game projected onto one arm of the sweep."""

    game_id: int
    #: The color whose decisions this benchmark scores.
    player_color: chess.Color
    row: dict[str, Any]
    #: The player-to-move plies inside the measurement window, in order.
    measured_plies: tuple[int, ...]
    window_opponent_moves: int
    perturbed_opponent_moves: int
    truncated: StrictBool

    @property
    def diverged(self) -> bool:
        """Return whether any opponent move in the window was replaced."""

        return self.perturbed_opponent_moves > 0

    def as_record(self) -> dict[str, object]:
        """Return the per-game derivation provenance kept in the detail tier."""

        return {
            "game_id": self.game_id,
            "player_color": "white" if self.player_color else "black",
            "measured_plies": list(self.measured_plies),
            "window_opponent_moves": self.window_opponent_moves,
            "perturbed_opponent_moves": self.perturbed_opponent_moves,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class PredicateReading:
    """One predicate's reading on one arm."""

    predicate: PositionPredicate
    opportunities: int
    selected_rate: float
    policy_mass: float
    mean_best_rank: float | None
    rankable_opportunities: int

    def as_record(self) -> dict[str, object]:
        return {
            "opportunities": self.opportunities,
            "selected_rate": self.selected_rate,
            "policy_mass": self.policy_mass,
            "mean_best_rank": self.mean_best_rank,
            "rankable_opportunities": self.rankable_opportunities,
        }


@dataclass(frozen=True)
class ArmReading:
    """Everything one dose measured, before it is joined to the control."""

    dose: float
    games: tuple[DerivedGame, ...]
    slices: SliceTable
    predicates: Mapping[PositionPredicate, PredicateReading]
    positions: tuple[PositionPolicy, ...]
    opportunities: Mapping[PositionPredicate, tuple[_Opportunity, ...]]
    scored_positions: int

    @property
    def is_control(self) -> bool:
        """Return whether this arm replaced nothing."""

        return self.dose == CONTROL_DOSE

    @property
    def measured_keys(self) -> frozenset[PositionKey]:
        """Return the positions this arm actually scored."""

        return frozenset(
            (position.game_id, position.ply_index) for position in self.positions
        )

    def paired_legality(self, keys: frozenset[PositionKey]) -> _Legality:
        """Return legality over only the positions in ``keys``.

        Retention is a paired quantity. A perturbed arm ends where the human's
        reply stopped being legal, so it scores a subset of the control's
        positions, and comparing its mean against the control's mean over
        everything reports that composition difference as a novelty effect.
        Measured on a real checkpoint the artifact was large enough to invert
        the reading, making legality look *better* under perturbation.
        """

        kept = [
            position
            for position in self.positions
            if (position.game_id, position.ply_index) in keys
        ]
        if not kept:
            return _Legality(legal_mass=0.0, mask_penalty=0.0, positions=0)
        return _Legality(
            legal_mass=sum(item.legal_mass for item in kept) / len(kept),
            mask_penalty=sum(item.mask_penalty for item in kept) / len(kept),
            positions=len(kept),
        )

    def paired_predicate(
        self,
        predicate: PositionPredicate,
        keys: frozenset[PositionKey],
    ) -> PredicateReading | None:
        """Return one predicate's reading over only the positions in ``keys``."""

        kept = tuple(
            item for item in self.opportunities.get(predicate, ()) if item.key in keys
        )
        return None if not kept else _reduce(predicate, kept)

    @property
    def realized_dose(self) -> float:
        """Return the share of window opponent moves actually replaced."""

        available = sum(game.window_opponent_moves for game in self.games)
        if available == 0:
            return 0.0
        return sum(game.perturbed_opponent_moves for game in self.games) / available

    @property
    def truncated_games(self) -> int:
        """Return how many derived games ended on an illegal human reply."""

        return sum(1 for game in self.games if game.truncated)

    def as_record(self) -> dict[str, object]:
        return {
            "dose": self.dose,
            "realized_dose": self.realized_dose,
            "scored_positions": self.scored_positions,
            "truncated_games": self.truncated_games,
            "slices": self.slices.as_record(),
            "predicates": {
                predicate.value: reading.as_record()
                for predicate, reading in sorted(
                    self.predicates.items(), key=lambda item: item[0].value
                )
            },
            "games": [game.as_record() for game in self.games],
        }


@dataclass(frozen=True)
class NoveltyBenchmarkResult:
    """One checkpoint's dose response, and where it was written."""

    checkpoint: CheckpointReference
    dataset: DatasetReference
    view: ViewSelection
    leakage: LeakageCheck
    arms: tuple[ArmReading, ...]
    envelopes: tuple[ResultEnvelope, ...]
    recorded_paths: tuple[Path, ...]
    detail_paths: tuple[Path, ...]

    @property
    def control(self) -> ArmReading:
        """Return the unperturbed arm every other one is read against."""

        for arm in self.arms:
            if arm.is_control:
                return arm
        raise NoveltyBenchmarkError("the suite recorded no control arm")

    def as_record(self) -> dict[str, object]:
        return {
            "version": NOVELTY_BENCHMARK_VERSION,
            "recipe_version": NOVELTY_RECIPE_VERSION,
            "policy_scoring_version": POLICY_SCORING_VERSION,
            "slice_scheme_version": SLICE_SCHEME_VERSION,
            "checkpoint": self.checkpoint.model_dump(mode="json"),
            "dataset": self.dataset.model_dump(mode="json"),
            "view": self.view.as_record(),
            "leakage": self.leakage.as_record(),
            "arms": [arm.as_record() for arm in self.arms],
            "recorded": [str(path) for path in self.recorded_paths],
        }


def derive_arm(
    rows: Sequence[Mapping[str, Any]],
    *,
    dose: float,
    config: PerturbationConfig,
) -> tuple[DerivedGame, ...]:
    """Derive one arm of the sweep from the source games, deterministically.

    Every draw is keyed by the seed, the recipe, the game, and the window
    index, so an arm reproduces from its recorded workload alone and no arm
    depends on the order games were processed in.
    """

    if not 0.0 <= dose <= 1.0:
        raise NoveltyBenchmarkError(
            "a perturbation dose is a rate between zero and one"
        )
    if config.recipe is not PerturbationRecipe.RANDOM_LEGAL_OPPONENT:
        raise NoveltyBenchmarkError(f"unsupported perturbation recipe: {config.recipe}")
    return tuple(
        derived
        for row in sorted(rows, key=lambda item: int(item[NormalizedColumn.GAME_ID]))
        if (derived := _derive_game(row, dose=dose, config=config)) is not None
    )


def benchmark_novelty(
    resolved_config: ResolvedConfig[NoveltyBenchmarkConfig],
    *,
    run_root: Path | None = None,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> NoveltyBenchmarkResult:
    """Measure one checkpoint's dose response and optionally record it."""

    config = resolved_config.value
    started = time.perf_counter()
    try:
        pool, selection, source_rows = _load_inputs(config)
        runner = CheckpointModelRunner.load(config.model, run_root=run_root)
    except (
        DataLoadingError,
        EvaluationPoolError,
        ModelRunnerError,
        OSError,
        ValueError,
    ) as error:
        if isinstance(error, NoveltyBenchmarkError):
            raise
        raise NoveltyBenchmarkError(str(error)) from error

    leakage = check_leakage(
        pool,
        runner.metadata,
        training_normalized=config.leakage.training_normalized,
    )

    split = _pool_split(pool)
    arms: list[ArmReading] = []
    for dose in sorted(config.perturbation.doses):
        logger.info("Deriving and scoring the novelty arm at dose %.3f", dose)
        try:
            arms.append(
                _score_arm(
                    source_rows,
                    runner,
                    dose=dose,
                    config=config,
                    selection=selection,
                    split=split,
                )
            )
        except (CheckpointEvaluationError, ValueError) as error:
            if isinstance(error, NoveltyBenchmarkError):
                raise
            raise NoveltyBenchmarkError(str(error)) from error

    component = projection_content_digest(source_rows, MOVE_PREDICTION_PROJECTION)
    checkpoint = _checkpoint_reference(config, runner)
    data = _dataset_reference(pool, selection, component)
    recorded_at = datetime.now(tz=UTC)
    configuration = configuration_reference(
        resolved_config.as_record(),
        source=resolved_config.provenance.source,
        overrides=resolved_config.provenance.overrides,
    )

    result = NoveltyBenchmarkResult(
        checkpoint=checkpoint,
        dataset=data,
        view=selection,
        leakage=leakage,
        arms=tuple(arms),
        envelopes=(),
        recorded_paths=(),
        detail_paths=(),
    )
    control = result.control

    envelopes: list[ResultEnvelope] = []
    characterizations: list[NoiseCharacterization] = []
    detail_paths: list[Path] = []
    try:
        for arm in result.arms:
            execution = _execution_record(runner, config.perturbation, arm.dose)
            workload = execution.workload_component()
            arm_detail = _write_detail(
                detail,
                arm,
                checkpoint=checkpoint,
                recorded_at=recorded_at,
                per_position=config.detail.per_position,
                paths=detail_paths,
            )
            envelopes.append(
                build_result(
                    kind=NOVELTY_KIND,
                    benchmark=NOVELTY_BENCHMARK,
                    checkpoint=checkpoint,
                    configuration=configuration,
                    data=data,
                    execution=execution,
                    measurements=_arm_measurements(arm, control, component, workload),
                    detail=arm_detail,
                    recorded_at=recorded_at,
                )
            )
            characterization = _characterize_noise(
                arm,
                config.noise,
                component=component,
                workload=workload,
                recorded_at=recorded_at,
            )
            if characterization is not None:
                characterizations.append(characterization)
        # After the bootstrap floors above, which this invocation also paid for.
        envelopes.append(
            benchmark_cost_result(
                benchmark=NOVELTY_BENCHMARK,
                checkpoint=checkpoint,
                configuration=configuration,
                config=config,
                device=runner.device,
                seconds=time.perf_counter() - started,
                recorded_at=recorded_at,
            )
        )
    except (ResultRecordError, ResultsStoreError) as error:
        raise NoveltyBenchmarkError(str(error)) from error

    recorded_paths: list[Path] = []
    if store is not None:
        try:
            recorded_paths = [store.append(envelope) for envelope in envelopes]
            recorded_paths.extend(
                store.append_characterization(item) for item in characterizations
            )
        except (ResultRecordError, ResultsStoreError) as error:
            raise NoveltyBenchmarkError(str(error)) from error

    return replace(
        result,
        envelopes=tuple(envelopes),
        recorded_paths=tuple(recorded_paths),
        detail_paths=tuple(detail_paths),
    )


def _derive_game(
    row: Mapping[str, Any],
    *,
    dose: float,
    config: PerturbationConfig,
) -> DerivedGame | None:
    """Derive one game, or ``None`` when its window never opens.

    A game too short to reach the onset contributes nothing rather than
    contributing a shorter window, which would make the arms incomparable.
    """

    game_id = int(row[NormalizedColumn.GAME_ID])
    action_ids = [int(value) for value in row[NormalizedColumn.ACTION_IDS]]
    board = _initial_board(row)
    player = _player_color(config.seed, game_id)

    derived: list[int] = []
    measured: list[int] = []
    window_opponent_moves = 0
    perturbed_opponent_moves = 0
    diverged = False
    truncated = False
    window_open = False

    for ply_index, action_id in enumerate(action_ids):
        if len(measured) >= config.window_moves:
            break
        is_opponent = board.turn != player
        if not window_open and is_opponent and ply_index >= config.onset_plies:
            window_open = True

        if window_open and is_opponent:
            replacement = _opponent_action(
                board,
                diverged=diverged,
                dose=dose,
                seed=config.seed,
                game_id=game_id,
                window_index=window_opponent_moves,
            )
            window_opponent_moves += 1
            if replacement is not None:
                diverged = True
                perturbed_opponent_moves += 1
                derived.append(replacement)
                board.push(decode_move(replacement))
                continue

        # Everything else is the human's own move: every ply before the window
        # opens, every opponent move the draw left alone, and every player
        # reply. Once the line has diverged the human's move may no longer be
        # legal, and that is where the derived game ends.
        if is_terminal_action(action_id):
            truncated = diverged
            break
        move = decode_move(action_id)
        if move not in board.legal_moves:
            truncated = True
            break
        if window_open and not is_opponent:
            measured.append(ply_index)
        derived.append(action_id)
        board.push(move)

    if not measured:
        return None
    return DerivedGame(
        game_id=game_id,
        player_color=player,
        row=_project_row(row, derived),
        measured_plies=tuple(measured),
        window_opponent_moves=window_opponent_moves,
        perturbed_opponent_moves=perturbed_opponent_moves,
        truncated=truncated,
    )


def _opponent_action(
    board: chess.Board,
    *,
    diverged: bool,
    dose: float,
    seed: str,
    game_id: int,
    window_index: int,
) -> int | None:
    """Return a replacement opponent action, or ``None`` to replay the human's.

    Divergence is absorbing. After the first replacement the human's later
    opponent moves belong to a game that no longer exists, so every subsequent
    opponent move in the window is drawn rather than replayed. The configured
    dose is therefore the per-move rate at which divergence *starts*, and the
    realized share of replaced moves is reported beside every reading.
    """

    if dose <= 0.0:
        return None
    stream = _stream(seed, game_id, window_index)
    if not diverged and stream.random() >= dose:
        return None
    moves = sorted(board.legal_moves, key=lambda move: move.uci())
    if not moves:
        return None
    return encode_move(stream.choice(moves))


def _stream(seed: str, game_id: int, window_index: int) -> Random:
    """Return the deterministic stream for one game's window position."""

    digest = sha256(
        f"{seed}\0{PerturbationRecipe.RANDOM_LEGAL_OPPONENT.value}\0"
        f"{NOVELTY_RECIPE_VERSION}\0{game_id}\0{window_index}".encode()
    ).digest()
    return Random(int.from_bytes(digest, "big"))


def _player_color(seed: str, game_id: int) -> chess.Color:
    """Assign the measured color, stably across every dose of one suite.

    The assignment cannot depend on the dose, or two arms would measure
    different sides of the same game and stop being paired.
    """

    digest = sha256(f"{seed}\0color\0{game_id}".encode()).digest()
    return chess.WHITE if digest[0] & 1 else chess.BLACK


def _initial_board(row: Mapping[str, Any]) -> chess.Board:
    initial = row[NormalizedColumn.INITIAL_POSITION]
    return chess.Board() if initial is None else chess.Board(str(initial))


def _project_row(row: Mapping[str, Any], actions: Sequence[int]) -> dict[str, Any]:
    """Return the derived row carrying this arm's own action sequence.

    Everything else about the game is the source's, truncated to the plies the
    derivation reached. The clock trace is projected rather than recomputed:
    this benchmark reads no timing, and inventing move times for moves nobody
    played would put fabricated data in the one place a later timing benchmark
    would trust.
    """

    updated = dict(row)
    plies = len(actions)
    updated[NormalizedColumn.ACTION_IDS.value] = list(actions)
    for column in (
        NormalizedColumn.CLOCK_REMAINING_MS,
        NormalizedColumn.CLOCK_STATUS,
        NormalizedColumn.CLOCK_PRECISION_MS,
    ):
        values = updated.get(column.value)
        if values is not None:
            updated[column.value] = list(values)[:plies]
    updated[NormalizedColumn.PLY_COUNT.value] = plies
    return updated


def _score_arm(
    source_rows: Sequence[Mapping[str, Any]],
    runner: CheckpointModelRunner,
    *,
    dose: float,
    config: NoveltyBenchmarkConfig,
    selection: ViewSelection,
    split: SplitName,
) -> ArmReading:
    """Derive one arm and score the player decisions inside its window."""

    games = derive_arm(source_rows, dose=dose, config=config.perturbation)
    if not games:
        raise NoveltyBenchmarkError(
            f"the novelty arm at dose {dose} derived no measurable position; "
            "the view's games are shorter than the configured onset"
        )
    rows = [game.row for game in games]
    inputs = build_scoring_inputs(
        rows,
        split=split,
        batch_size=config.loader.batch_size,
        length_bucket_width=config.loader.length_bucket_width,
        identity_sha256=rows_identity_sha256(
            rows,
            context=(selection.as_record(), config.perturbation.recipe.value, dose),
        ),
    )
    measured: set[PositionKey] = {
        (game.game_id, ply_index) for game in games for ply_index in game.measured_plies
    }

    positions: list[PositionPolicy] = []
    predicate_scores: list[ActionSetPolicy] = []
    subsets = action_sets(inputs)
    for batch in _batches(inputs, runner):
        logits = runner.action_logits(batch)
        positions.extend(
            item
            for item in score_positions(logits, batch)
            if (item.game_id, item.ply_index) in measured
        )
        predicate_scores.extend(
            item
            for item in score_action_sets(logits, batch, subsets)
            if (item.game_id, item.ply_index) in measured
        )
    if not positions:
        raise NoveltyBenchmarkError(
            f"the novelty arm at dose {dose} scored no position inside its window"
        )

    opportunities = _predicate_opportunities(predicate_scores, inputs)
    return ArmReading(
        dose=dose,
        games=games,
        slices=aggregate_positions(positions, inputs),
        predicates=_reduce_opportunities(opportunities),
        positions=tuple(positions),
        opportunities=opportunities,
        scored_positions=len(positions),
    )


def _batches(
    inputs: ScoringInputs,
    runner: CheckpointModelRunner,
) -> Iterator[MoveModelBatch]:
    loader = SequenceDataLoader(inputs.dataset, inputs.loader_config)
    for sequence_batch in loader:
        yield MoveModelBatch.from_sequence_batch(
            sequence_batch,
            device=runner.device,
        )


@dataclass(frozen=True)
class _Legality:
    """Legality over one selection of an arm's scored positions."""

    legal_mass: float
    mask_penalty: float
    positions: int


@dataclass(frozen=True)
class _Opportunity:
    """One predicate opportunity on a derived arm, with no human reference."""

    key: PositionKey
    success: bool
    policy_mass: float
    best_rank: int | None


def _predicate_opportunities(
    scored: Sequence[ActionSetPolicy],
    inputs: ScoringInputs,
) -> Mapping[PositionPredicate, tuple[_Opportunity, ...]]:
    """Group the scored predicate opportunities, keeping their positions.

    The positions are kept rather than summed away because retention has to be
    computed over the ones a perturbed arm actually reached, not over every
    opportunity the control arm had.

    No human rate appears here even on the control arm. The control's own human
    reference is the checkpoint runner's job; this benchmark's question is what
    the model retains relative to itself.
    """

    grouped: dict[PositionPredicate, list[_Opportunity]] = defaultdict(list)
    for item in scored:
        key = (item.game_id, item.ply_index)
        predicate = PositionPredicate(item.name)
        match = inputs.predicates[key].get(predicate)
        if match is None:
            continue
        grouped[predicate].append(
            _Opportunity(
                key=key,
                success=item.selected_action_id in match.successful_action_ids,
                policy_mass=item.raw_probability_mass,
                best_rank=item.best_rank,
            )
        )
    return {predicate: tuple(items) for predicate, items in grouped.items()}


def _reduce(
    predicate: PositionPredicate,
    items: Sequence[_Opportunity],
) -> PredicateReading:
    """Reduce one predicate's opportunities to a single reading."""

    ranks = [item.best_rank for item in items if item.best_rank is not None]
    return PredicateReading(
        predicate=predicate,
        opportunities=len(items),
        selected_rate=sum(1.0 for item in items if item.success) / len(items),
        policy_mass=sum(item.policy_mass for item in items) / len(items),
        mean_best_rank=(sum(ranks) / len(ranks) if ranks else None),
        rankable_opportunities=len(ranks),
    )


def _reduce_opportunities(
    opportunities: Mapping[PositionPredicate, Sequence[_Opportunity]],
) -> Mapping[PositionPredicate, PredicateReading]:
    return {
        predicate: _reduce(predicate, items)
        for predicate, items in opportunities.items()
        if items
    }


def _arm_measurements(
    arm: ArmReading,
    control: ArmReading,
    component: DataComponent,
    workload: WorkloadComponent,
) -> tuple[Measurement, ...]:
    """Return one arm's committed measurements, retention included."""

    overall = arm.slices.overall
    values: list[Measurement] = [
        _measure(
            NOVELTY_LEGAL_MASS,
            overall.legal_mass,
            component,
            workload,
            overall.position_count,
        ),
        _measure(
            NOVELTY_MASK_PENALTY,
            overall.mask_penalty,
            component,
            workload,
            overall.position_count,
        ),
        _measure(
            NOVELTY_REALIZED_DOSE,
            arm.realized_dose,
            component,
            workload,
            len(arm.games),
        ),
        _measure(
            NOVELTY_DERIVED_PLY_RETENTION,
            _ratio(arm.scored_positions, control.scored_positions),
            component,
            workload,
            len(arm.games),
        ),
    ]
    for phase, definition in NOVELTY_MASK_PENALTY_BY_PHASE.items():
        summary = arm.slices.slice_summary(PHASE_DIMENSION, phase)
        if summary is None:
            continue
        values.append(
            _measure(
                definition,
                summary.mask_penalty,
                component,
                workload,
                summary.position_count,
            )
        )

    # Retention is paired on position: the control is read over the positions
    # this arm actually reached, never over everything it had.
    keys = arm.measured_keys
    if not arm.is_control:
        reference = control.paired_legality(keys)
        observed = arm.paired_legality(keys)
        values.append(
            _measure(
                NOVELTY_LEGAL_MASS_RETENTION,
                _ratio(observed.legal_mass, reference.legal_mass),
                component,
                workload,
                observed.positions,
            )
        )
        values.append(
            _measure(
                NOVELTY_MASK_PENALTY_RATIO,
                _ratio(observed.mask_penalty, reference.mask_penalty),
                component,
                workload,
                observed.positions,
            )
        )

    for predicate, reading in sorted(
        arm.predicates.items(), key=lambda item: item[0].value
    ):
        name = predicate.value
        values.append(
            _measure(
                NOVELTY_PREDICATE_SELECTED_RATE[name],
                reading.selected_rate,
                component,
                workload,
                reading.opportunities,
            )
        )
        values.append(
            _measure(
                NOVELTY_PREDICATE_POLICY_MASS[name],
                reading.policy_mass,
                component,
                workload,
                reading.opportunities,
            )
        )
        if reading.mean_best_rank is not None:
            values.append(
                _measure(
                    NOVELTY_PREDICATE_BEST_RANK[name],
                    reading.mean_best_rank,
                    component,
                    workload,
                    reading.rankable_opportunities,
                )
            )
        if arm.is_control:
            continue
        # A predicate is realized by the position, and a perturbed position is
        # a different one, so this pairs on the positions rather than on the
        # opportunities: the control's rate is read over the plies this arm
        # reached, whether or not the predicate fired there for the control.
        paired = control.paired_predicate(predicate, keys)
        if paired is None:
            continue
        values.append(
            _measure(
                NOVELTY_PREDICATE_RETENTION[name],
                _ratio(reading.selected_rate, paired.selected_rate),
                component,
                workload,
                reading.opportunities,
            )
        )
    return tuple(values)


def _measure(
    definition: MetricDefinition,
    value: float,
    component: DataComponent,
    workload: WorkloadComponent,
    sample_size: int,
) -> Measurement:
    return measurement(
        definition.identifier,
        value,
        data=component,
        workload=workload,
        sample_size=max(sample_size, 1),
    )


def _ratio(value: float, reference: float) -> float:
    """Return a retention ratio, with an absent reference reported as zero.

    A control arm that never realized the quantity leaves retention undefined
    rather than infinite. Zero is the honest floor: nothing was retained
    because there was nothing to retain, and the opportunity counts beside it
    say so.
    """

    if reference == 0.0:
        return 0.0
    return value / reference


def _arm_game_totals(arm: ArmReading) -> tuple[GameTotals, ...]:
    """Return each derived game's contribution to this arm's own metrics."""

    by_game: dict[int, list[PositionPolicy]] = defaultdict(list)
    for position in arm.positions:
        by_game[position.game_id].append(position)

    totals: list[GameTotals] = []
    for game_id, positions in sorted(by_game.items()):
        count = len(positions)
        totals.append(
            GameTotals(
                game_id=game_id,
                metrics={
                    NOVELTY_LEGAL_MASS.identifier: MetricTotal(
                        total=sum(item.legal_mass for item in positions),
                        positions=count,
                    ),
                    NOVELTY_MASK_PENALTY.identifier: MetricTotal(
                        total=sum(item.mask_penalty for item in positions),
                        positions=count,
                    ),
                },
            )
        )
    return tuple(totals)


def _characterize_noise(
    arm: ArmReading,
    config: NoiseConfig,
    *,
    component: DataComponent,
    workload: WorkloadComponent,
    recorded_at: datetime,
) -> NoiseCharacterization | None:
    """Bootstrap this arm's own data-sampling floor over its derived games."""

    if not config.enabled:
        return None
    try:
        return characterize_sampling_noise(
            merge_game_totals(_arm_game_totals(arm)),
            component=component,
            config=config,
            source=(
                f"bootstrap over {len(arm.games)} derived game(s) at novelty "
                f"dose {arm.dose}"
            ),
            recorded_at=recorded_at,
            workload=workload,
        )
    except NoiseCharacterizationError as error:
        logger.warning(
            "Skipping the novelty floor at dose %s: %s",
            arm.dose,
            error,
        )
        return None


def _execution_record(
    runner: CheckpointModelRunner,
    config: PerturbationConfig,
    dose: float,
) -> ExecutionRecord:
    """Return the declared derivation this arm was measured under.

    Every field here is a realized input to the value rather than a condition
    it could be subtracted across, which is why the whole recipe joins series
    identity. Two doses are two series, and a recipe change ends both.
    """

    return execution_record(
        runner.device,
        {
            "benchmark": NOVELTY_KIND,
            "benchmark_version": NOVELTY_BENCHMARK_VERSION,
            "recipe": config.recipe.value,
            "recipe_version": NOVELTY_RECIPE_VERSION,
            "seed": config.seed,
            "onset_plies": config.onset_plies,
            "window_moves": config.window_moves,
            "dose": dose,
        },
    )


def _load_inputs(
    config: NoveltyBenchmarkConfig,
) -> tuple[FrozenPool, ViewSelection, tuple[dict[str, Any], ...]]:
    pool = load_pool(config.pool)
    selection = apply_view(pool.games, config.view)
    if not selection.game_ids:
        raise NoveltyBenchmarkError(
            f"view {config.view.name!r} selected no games from the pool"
        )
    wanted = set(selection.game_ids)
    rows = tuple(
        dict(row)
        for row in read_normalized_rows(pool.games_path)
        if int(row[NormalizedColumn.GAME_ID]) in wanted
    )
    if len(rows) != len(wanted):
        raise NoveltyBenchmarkError(
            "the evaluation pool does not contain every selected game"
        )
    return pool, selection, rows


def _checkpoint_reference(
    config: NoveltyBenchmarkConfig,
    runner: CheckpointModelRunner,
) -> CheckpointReference:
    run_id = runner.selection.run_path.name
    label = config.checkpoint_label or default_checkpoint_label(
        run_id,
        runner.global_step,
    )
    return CheckpointReference(
        label=label,
        step=runner.global_step,
        run_id=run_id,
        parameter_sha256=runner.parameter_sha256(),
    )


def _dataset_reference(
    pool: FrozenPool,
    selection: ViewSelection,
    component: DataComponent,
) -> DatasetReference:
    record = pool.manifest.get("pool")
    if not isinstance(record, Mapping):
        raise NoveltyBenchmarkError("evaluation pool manifest has no pool identity")
    view_record = selection.as_record()
    return dataset_reference(
        pool_id=str(record["id"]),
        pool_version=int(record["version"]),
        view=selection.name,
        selected_games=selection.selected_games,
        game_ids_sha256=str(view_record["game_ids_sha256"]),
        components=[component],
    )


def _pool_split(pool: FrozenPool) -> SplitName:
    record = pool.manifest.get("pool")
    split = record.get("split") if isinstance(record, Mapping) else None
    if split not in SPLIT_NAMES:
        raise NoveltyBenchmarkError(
            f"evaluation pool manifest names an unknown split: {split!r}"
        )
    return cast(SplitName, split)


def _write_detail(
    detail: DetailStore | None,
    arm: ArmReading,
    *,
    checkpoint: CheckpointReference,
    recorded_at: datetime,
    per_position: bool,
    paths: list[Path],
) -> DetailReference | None:
    if detail is None:
        return None
    stamp = recorded_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload: dict[str, object] = dict(arm.as_record())
    if per_position:
        payload["positions"] = [position.as_record() for position in arm.positions]
    dose_slug = f"{arm.dose:.4f}".replace(".", "-")
    relative = Path(NOVELTY_KIND) / checkpoint.label / f"dose-{dose_slug}-{stamp}.json"
    reference = detail.write(
        relative,
        payload,
        description=(
            "Per-arm derivation provenance, slice tables, and predicate "
            "readings for one novelty dose."
        ),
    )
    paths.append(detail.root / relative)
    return reference


__all__ = [
    "CONTROL_DOSE",
    "NOVELTY_BENCHMARK_VERSION",
    "NOVELTY_KIND",
    "NOVELTY_RECIPE_VERSION",
    "ArmReading",
    "DerivedGame",
    "NoveltyBenchmarkConfig",
    "NoveltyBenchmarkError",
    "NoveltyBenchmarkResult",
    "PerturbationConfig",
    "PerturbationRecipe",
    "PredicateReading",
    "benchmark_novelty",
    "derive_arm",
]
