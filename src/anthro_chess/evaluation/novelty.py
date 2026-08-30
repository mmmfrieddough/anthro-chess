"""What a controlled novelty dose costs the policy.

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
played. Legality needs no target and survives. So does material gain, which is
read at a fixed size of win: whether a position offers one and how large is a
property of the board, and a random opponent hands over larger ones, so an
average across sizes reports the mix rather than the model.

See ``docs/evaluation.md`` and
``docs/decisions/0024-one-sided-perturbation-derived-novelty.md``.
"""

from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from random import Random
from statistics import fmean
from typing import Any, cast

import chess
from pydantic import Field, StrictBool, StrictInt, model_validator

from anthro_chess.chess import decode_move, encode_move, is_terminal_action
from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import (
    DataLoadingError,
    PlyEncoding,
    SequenceDataLoader,
    encode_game,
)
from anthro_chess.data.schema import (
    SPLIT_NAMES,
    NormalizedColumn,
    SplitName,
    row_game_id,
)
from anthro_chess.evaluation.adjudication import merge_game_totals
from anthro_chess.evaluation.checkpoint import (
    CheckpointEvaluationError,
    DetailConfig,
    LeakageConfig,
)
from anthro_chess.evaluation.execution import execution_record
from anthro_chess.evaluation.leakage import LeakageCheck, check_leakage
from anthro_chess.evaluation.noise import (
    GameTotals,
    MetricTotal,
    NoiseConfig,
    sampling_dispersions,
)
from anthro_chess.evaluation.policy import (
    POLICY_SCORING_VERSION,
    ActionSetPolicy,
    PositionPolicy,
    active_batch,
    score_action_sets,
    score_positions,
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
    checkpoint_reference,
    pool_dataset_reference,
)
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
    DatasetReference,
    ExecutionRecord,
    Measurement,
    MetricDispersion,
    ResultEnvelope,
    WorkloadComponent,
    measurement,
    projection_content_digest,
)
from anthro_chess.evaluation.results.metrics import (
    MOVE_PREDICTION_PROJECTION,
    NOVELTY_DERIVED_PLY_RETENTION,
    NOVELTY_MASK_PENALTY,
    NOVELTY_MASK_PENALTY_DELTA,
    NOVELTY_MATERIAL_GAIN_OPPORTUNITY_SHARE,
    NOVELTY_MATERIAL_GAIN_POLICY_MASS,
    NOVELTY_MATERIAL_GAIN_SELECTED_RATE,
    NOVELTY_REALIZED_DOSE,
    MetricDefinition,
)
from anthro_chess.evaluation.results.noise import NoiseCharacterizationError
from anthro_chess.evaluation.scoring import (
    SCORED_COLUMNS,
    EvaluationLoaderConfig,
    ScoringInputs,
    build_scoring_inputs,
    encoding_input,
    rows_identity_sha256,
)
from anthro_chess.evaluation.selection import CheckpointSelection
from anthro_chess.evaluation.slices import (
    MATERIAL_GAIN_BAND_FLOORS,
    SLICE_SCHEME_VERSION,
    board_from_encoding,
    material_winning_moves,
)
from anthro_chess.evaluation.views import ViewConfig, ViewSelection, apply_view
from anthro_chess.inference import CheckpointModelRunner
from anthro_chess.inference.runner import ModelRunnerError
from anthro_chess.models import MoveModelBatch

#: Bumped when the derivation changes what a dose means. Results are not
#: comparable across recipe versions, which is why this joins the workload.
NOVELTY_RECIPE_VERSION = 1
NOVELTY_BENCHMARK_VERSION = 2
NOVELTY_KIND = "novelty-dose-response"
NOVELTY_BENCHMARK = BenchmarkReference(
    name=NOVELTY_KIND,
    version=NOVELTY_BENCHMARK_VERSION,
)

#: The dose at which no opponent move is replaced.
CONTROL_DOSE = 0.0

#: Workers preparing arms. Capped rather than left to the core count because
#: importing this module to run pure-Python chess costs a worker most of a
#: gigabyte, and the stage it shortens is seconds long.
_PREPARE_WORKER_LIMIT = 8

#: Derived games per unit of work handed to a worker. Large enough that
#: pickling a result is not most of the job, small enough that the last worker
#: to finish does not hold up the arm.
_PREPARE_CHUNK_GAMES = 64

#: The bands in descending order of what they cover, so the first floor a
#: position's largest win clears names it.
_BAND_FLOORS: tuple[tuple[str, int], ...] = tuple(
    sorted(MATERIAL_GAIN_BAND_FLOORS.items(), key=lambda item: -item[1])
)

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


class NoveltyBenchmarkConfig(CheckpointSelection, PoolGenerationPin):
    """Code-owned schema for ``anthro eval novelty``."""

    pool: Path
    view: ViewConfig = ViewConfig(name="novelty")
    loader: EvaluationLoaderConfig = EvaluationLoaderConfig()
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
class _Opportunity:
    """One scored material-gain opportunity, with the game it came from."""

    game_id: int
    band: str
    success: bool
    policy_mass: float
    #: Never absent here: a band names a set only where a capture wins, so the
    #: set has a member to rank.
    best_rank: int


@dataclass(frozen=True)
class BandReading:
    """One win-size band's reading on one arm."""

    band: str
    opportunities: int
    selected_rate: float
    policy_mass: float
    mean_best_rank: float

    def as_record(self) -> dict[str, object]:
        return {
            "opportunities": self.opportunities,
            "selected_rate": self.selected_rate,
            "policy_mass": self.policy_mass,
            "mean_best_rank": self.mean_best_rank,
        }


@dataclass(frozen=True)
class ArmReading:
    """Everything one dose measured, before it is joined to the control."""

    dose: float
    games: tuple[DerivedGame, ...]
    bands: Mapping[str, BandReading]
    opportunities: tuple[_Opportunity, ...]
    positions: tuple[PositionPolicy, ...]
    scored_positions: int
    #: Scored positions per game phase. Kept because truncation moves the mix
    #: hard, from 43% opening on the control to 68% at full dose, so a reading
    #: that surprises is read against what it was taken over.
    phases: Mapping[str, int]

    @property
    def legality(self) -> _Legality:
        """Return legality over every position this arm scored."""

        return self.paired_legality(None)

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

    def paired_legality(self, keys: frozenset[PositionKey] | None) -> _Legality:
        """Return legality over the positions in ``keys``, or over all of them.

        A perturbed arm ends where the human's reply stopped being legal, so it
        scores a subset of the control's positions, and comparing its mean
        against the control's mean over everything reports that composition
        difference as a novelty effect. That artifact is large enough to invert
        the reading, making legality look *better* under perturbation.
        """

        kept = (
            self.positions
            if keys is None
            else [
                position
                for position in self.positions
                if (position.game_id, position.ply_index) in keys
            ]
        )
        if not kept:
            return _Legality(mask_penalty=0.0, positions=0)
        return _Legality(
            mask_penalty=fmean(item.mask_penalty for item in kept),
            positions=len(kept),
        )

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
            "phases": dict(sorted(self.phases.items())),
            "material_gain_bands": {
                band: reading.as_record()
                for band, reading in sorted(self.bands.items())
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
    envelopes: tuple[ResultEnvelope, ...] = ()
    recorded_paths: tuple[Path, ...] = ()
    detail_paths: tuple[Path, ...] = ()

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
        for row in sorted(rows, key=lambda item: row_game_id(item))
        if (derived := _derive_game(row, dose=dose, config=config)) is not None
    )


def benchmark_novelty(
    resolved_config: ResolvedConfig[NoveltyBenchmarkConfig],
    *,
    run_root: Path | None = None,
    recording: ResultRecording,
) -> NoveltyBenchmarkResult:
    """Measure one checkpoint's dose response and optionally record it."""

    config = resolved_config.value
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
    # Spawn rather than fork: the runner above has initialized CUDA, and a
    # forked worker inherits both that context and whatever locks its threads
    # were holding. One pool for the whole sweep, because spawning re-imports
    # this module and paying that per arm would cost more than it saves.
    with ProcessPoolExecutor(
        max_workers=_prepare_workers(),
        mp_context=get_context("spawn"),
    ) as executor:
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
                        executor=executor,
                    )
                )
            except (CheckpointEvaluationError, ValueError) as error:
                if isinstance(error, NoveltyBenchmarkError):
                    raise
                raise NoveltyBenchmarkError(str(error)) from error

    component = projection_content_digest(source_rows, MOVE_PREDICTION_PROJECTION)
    checkpoint = checkpoint_reference(runner, label=config.checkpoint_label)
    data = pool_dataset_reference(
        pool,
        selection,
        component,
        error=NoveltyBenchmarkError,
    )

    result = NoveltyBenchmarkResult(
        checkpoint=checkpoint,
        dataset=data,
        view=selection,
        leakage=leakage,
        arms=tuple(arms),
    )
    control = result.control

    recorder = recording.measuring(
        checkpoint,
        kind=NOVELTY_KIND,
        benchmark=NOVELTY_BENCHMARK,
    )
    for arm in result.arms:
        execution = _execution_record(runner, config.perturbation, arm.dose)
        workload = execution.workload_component()
        dose_slug = f"{arm.dose:.4f}".replace(".", "-")
        recorder.disperse(
            _arm_dispersions(
                arm,
                config.noise,
                component=component,
                workload=workload,
            )
        )
        recorder.add(
            _arm_measurements(arm, control, component, workload),
            payload=partial(_arm_payload, arm, per_position=config.detail.per_position),
            description=(
                "Per-arm derivation provenance, phase counts, and "
                "material-gain band readings for one novelty dose."
            ),
            slug=f"dose-{dose_slug}",
            data=data,
            execution=execution,
        )
    return result


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

    game_id = row_game_id(row)
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
    would trust. The remaining per-ply clock column is never read here, so the
    pool read leaves it behind rather than truncating it for nobody.
    """

    updated = dict(row)
    plies = len(actions)
    updated[NormalizedColumn.ACTION_IDS.value] = list(actions)
    clocks = updated[NormalizedColumn.CLOCK_REMAINING_DELTA_MS.value]
    updated[NormalizedColumn.CLOCK_REMAINING_DELTA_MS.value] = list(clocks)[:plies]
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
    executor: Executor,
) -> ArmReading:
    """Derive one arm and score the player decisions inside its window."""

    games = derive_arm(source_rows, dose=dose, config=config.perturbation)
    if not games:
        raise NoveltyBenchmarkError(
            f"the novelty arm at dose {dose} derived no measurable position; "
            "the view's games are shorter than the configured onset"
        )
    rows = [game.row for game in games]
    encodings, subsets = _prepare_arm(games, executor)
    inputs = build_scoring_inputs(
        rows,
        split=split,
        batch_size=config.loader.batch_size,
        length_bucket_width=config.loader.length_bucket_width,
        identity_sha256=rows_identity_sha256(
            rows,
            context=(selection.as_record(), config.perturbation.recipe.value, dose),
        ),
        encodings=encodings,
    )
    measured: set[PositionKey] = {
        (game.game_id, ply_index) for game in games for ply_index in game.measured_plies
    }

    positions: list[PositionPolicy] = []
    band_scores: list[ActionSetPolicy] = []
    for batch in _batches(inputs, runner):
        active = active_batch(runner.action_logits(batch), batch)
        positions.extend(
            item
            for item in score_positions(active)
            if (item.game_id, item.ply_index) in measured
        )
        # No window filter here, unlike the scored positions above: the scorer
        # reads a subset only where one was named, and the subsets were named
        # over the window.
        band_scores.extend(score_action_sets(active, subsets))
    if not positions:
        raise NoveltyBenchmarkError(
            f"the novelty arm at dose {dose} scored no position inside its window"
        )

    opportunities = _opportunities(band_scores, subsets)
    return ArmReading(
        dose=dose,
        games=games,
        bands=_reduce_bands(opportunities),
        opportunities=opportunities,
        positions=tuple(positions),
        scored_positions=len(positions),
        phases=Counter(
            str(inputs.slices[(item.game_id, item.ply_index)].phase)
            for item in positions
        ),
    )


def _prepare_games(
    games: Sequence[DerivedGame],
) -> tuple[
    dict[int, tuple[PlyEncoding, ...]], dict[PositionKey, dict[str, frozenset[int]]]
]:
    """Encode one chunk of derived games and name its material-gain sets.

    Both halves are pure functions of the derived game and together they are
    most of a reading's wall clock, so they are done in one pass: a worker that
    already holds the encoding is the cheapest place to rebuild the board.

    The name a set carries is the band, so the scorer groups by difficulty for
    free and a position lands in exactly one. The set holds the captures that
    win the most, not every capture that wins something: mass over a set counts
    its members, and a random opponent leaves more of them standing, which
    reintroduces inside a band the mix effect the bands exist to remove.

    This resolves the one predicate the dose reading keeps rather than calling
    the shared matcher, which would also push every legal move for mate, probe
    a null move for a threat, and answer three questions a perturbed position
    has no sample for.

    Module level rather than a closure because it is the unit of work a process
    pool sends to a worker.
    """

    encodings: dict[int, tuple[PlyEncoding, ...]] = {}
    sets: dict[PositionKey, dict[str, frozenset[int]]] = {}
    for game in games:
        encoded = encode_game(encoding_input(game.row))
        encodings[game.game_id] = tuple(encoded)
        measured = frozenset(game.measured_plies)
        for ply in encoded:
            if ply.ply_index not in measured:
                continue
            board = board_from_encoding(ply.board)
            # The encoding's own legal actions rather than a second generation:
            # rebuilding them is the most expensive thing encoding a ply does.
            winning = material_winning_moves(
                board,
                tuple(
                    decode_move(action)
                    for action in ply.enabled_actions()
                    if not is_terminal_action(action)
                ),
            )
            if not winning:
                continue
            best = max(gain for _, gain in winning)
            sets[(ply.game_id, ply.ply_index)] = {
                _gain_band(best): frozenset(
                    encode_move(move) for move, gain in winning if gain == best
                )
            }
    return encodings, sets


def _prepare_workers() -> int:
    """Return how many workers prepare an arm on this machine."""

    return max(1, min(_PREPARE_WORKER_LIMIT, os.process_cpu_count() or 1))


def _prepare_arm(
    games: Sequence[DerivedGame],
    executor: Executor,
) -> tuple[
    dict[int, tuple[PlyEncoding, ...]], dict[PositionKey, dict[str, frozenset[int]]]
]:
    """Prepare every derived game, across ``executor`` where there are enough.

    Chunks are consumed in order, so a run on one core and a run on thirty
    assemble the same inputs.
    """

    chunks = [
        games[start : start + _PREPARE_CHUNK_GAMES]
        for start in range(0, len(games), _PREPARE_CHUNK_GAMES)
    ]
    mapper = map if len(chunks) < 2 else executor.map
    encodings: dict[int, tuple[PlyEncoding, ...]] = {}
    sets: dict[PositionKey, dict[str, frozenset[int]]] = {}
    for chunk_encodings, chunk_sets in mapper(_prepare_games, chunks):
        encodings.update(chunk_encodings)
        sets.update(chunk_sets)
    return encodings, sets


def _gain_band(gain: int) -> str:
    """Return which band a position's largest material win falls in."""

    return next(band for band, floor in _BAND_FLOORS if gain >= floor)


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

    mask_penalty: float
    positions: int


def _opportunities(
    scored: Sequence[ActionSetPolicy],
    subsets: Mapping[PositionKey, Mapping[str, frozenset[int]]],
) -> tuple[_Opportunity, ...]:
    """Return one record per scored opportunity, keeping the game it came from.

    Kept per game rather than summed away, because a sampling floor resamples
    games and cannot be derived from an arm's totals.
    """

    records: list[_Opportunity] = []
    for item in scored:
        assert item.best_rank is not None  # a band names a set only where one wins
        records.append(
            _Opportunity(
                game_id=item.game_id,
                band=item.name,
                success=item.selected_action_id
                in subsets[(item.game_id, item.ply_index)][item.name],
                policy_mass=item.raw_probability_mass,
                best_rank=item.best_rank,
            )
        )
    return tuple(records)


def _reduce_bands(opportunities: Sequence[_Opportunity]) -> dict[str, BandReading]:
    """Reduce the scored opportunities to one reading per win-size band."""

    grouped: dict[str, list[_Opportunity]] = defaultdict(list)
    for item in opportunities:
        grouped[item.band].append(item)
    return {
        band: BandReading(
            band=band,
            opportunities=len(items),
            selected_rate=sum(1.0 for item in items if item.success) / len(items),
            policy_mass=sum(item.policy_mass for item in items) / len(items),
            mean_best_rank=sum(item.best_rank for item in items) / len(items),
        )
        for band, items in grouped.items()
    }


def _arm_measurements(
    arm: ArmReading,
    control: ArmReading,
    component: DataComponent,
    workload: WorkloadComponent,
) -> tuple[Measurement, ...]:
    """Return one arm's committed measurements."""

    overall = arm.legality
    values: list[Measurement] = [
        _measure(
            NOVELTY_MASK_PENALTY,
            overall.mask_penalty,
            component,
            workload,
            overall.positions,
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

    # Paired on position: the control is read over the plies this arm actually
    # reached, never over everything it had. A perturbed arm ends where the
    # human's reply stopped being legal, so its survivors are the positions the
    # perturbation disturbed least.
    if not arm.is_control:
        reference = control.paired_legality(arm.measured_keys)
        values.append(
            _measure(
                NOVELTY_MASK_PENALTY_DELTA,
                overall.mask_penalty - reference.mask_penalty,
                component,
                workload,
                overall.positions,
            )
        )

    for band, reading in sorted(arm.bands.items()):
        values.append(
            _measure(
                NOVELTY_MATERIAL_GAIN_POLICY_MASS[band],
                reading.policy_mass,
                component,
                workload,
                reading.opportunities,
            )
        )
        values.append(
            _measure(
                NOVELTY_MATERIAL_GAIN_SELECTED_RATE[band],
                reading.selected_rate,
                component,
                workload,
                reading.opportunities,
            )
        )
        values.append(
            _measure(
                NOVELTY_MATERIAL_GAIN_OPPORTUNITY_SHARE[band],
                _ratio(reading.opportunities, arm.scored_positions),
                component,
                workload,
                arm.scored_positions,
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
    """Return a share, with an absent reference reported as zero.

    An arm that scored nothing leaves the share undefined rather than
    infinite, and the counts recorded beside it say which happened.
    """

    if reference == 0.0:
        return 0.0
    return value / reference


def _arm_game_totals(arm: ArmReading) -> tuple[GameTotals, ...]:
    """Return each derived game's contribution to this arm's own metrics.

    The band readings are here as well as the legality one, because the view
    is sized on a band and a value with no floor cannot carry a claim that a
    difference survived the draw.
    """

    by_game: dict[int, list[PositionPolicy]] = defaultdict(list)
    for position in arm.positions:
        by_game[position.game_id].append(position)
    bands: dict[int, dict[str, list[_Opportunity]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for opportunity in arm.opportunities:
        bands[opportunity.game_id][opportunity.band].append(opportunity)

    totals: list[GameTotals] = []
    for game_id, positions in sorted(by_game.items()):
        metrics = {
            NOVELTY_MASK_PENALTY.identifier: MetricTotal(
                total=sum(item.mask_penalty for item in positions),
                positions=len(positions),
            ),
        }
        for band, items in bands[game_id].items():
            metrics[NOVELTY_MATERIAL_GAIN_POLICY_MASS[band].identifier] = MetricTotal(
                total=sum(item.policy_mass for item in items),
                positions=len(items),
            )
            metrics[NOVELTY_MATERIAL_GAIN_SELECTED_RATE[band].identifier] = MetricTotal(
                total=sum(1.0 for item in items if item.success),
                positions=len(items),
            )
        totals.append(GameTotals(game_id=game_id, metrics=metrics))
    return tuple(totals)


def _arm_dispersions(
    arm: ArmReading,
    config: NoiseConfig,
    *,
    component: DataComponent,
    workload: WorkloadComponent,
) -> dict[str, MetricDispersion]:
    """Bootstrap this arm's own data-sampling spread over its derived games."""

    if not config.enabled:
        return {}
    try:
        return sampling_dispersions(
            merge_game_totals(_arm_game_totals(arm)),
            component=component,
            config=config,
            source=(
                f"bootstrap over {len(arm.games)} derived game(s) at novelty "
                f"dose {arm.dose}"
            ),
            workload=workload,
        )
    except NoiseCharacterizationError as error:
        logger.warning(
            "Skipping the novelty spread at dose %s: %s",
            arm.dose,
            error,
        )
        return {}


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
    pool = load_pool(
        config.pool,
        expected_game_ids_sha256=config.expected_pool_game_ids_sha256,
    )
    selection = apply_view(pool.games, config.view)
    if not selection.game_ids:
        raise NoveltyBenchmarkError(
            f"view {config.view.name!r} selected no games from the pool"
        )
    rows = pool_rows(
        pool,
        selection.game_ids,
        SCORED_COLUMNS,
        error=NoveltyBenchmarkError,
    )
    return pool, selection, rows


def _pool_split(pool: FrozenPool) -> SplitName:
    record = pool.manifest.get("pool")
    split = record.get("split") if isinstance(record, Mapping) else None
    if split not in SPLIT_NAMES:
        raise NoveltyBenchmarkError(
            f"evaluation pool manifest names an unknown split: {split!r}"
        )
    return cast(SplitName, split)


def _arm_payload(arm: ArmReading, *, per_position: bool) -> dict[str, object]:
    """Return one arm's bulk record for the detail tier."""

    payload: dict[str, object] = dict(arm.as_record())
    if per_position:
        payload["positions"] = [position.as_record() for position in arm.positions]
    return payload


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
    "BandReading",
    "benchmark_novelty",
    "derive_arm",
]
