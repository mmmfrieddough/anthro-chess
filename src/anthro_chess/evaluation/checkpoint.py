"""The offline checkpoint evaluation runner.

This is the canonical end-of-run reading: one compatible checkpoint, scored
over a deterministic view of the frozen test pool, written into the committed
results store with the provenance needed to recompute its own series
fingerprints.

It is a library first and a command second. In-training evaluation at declared
cadences calls the same entry point over a smaller view, so an early preview
and the canonical reading stay the same measurement at two precisions rather
than two implementations that have to be reconciled.

The order of operations matters. The leakage check runs before any scoring, so
a checkpoint that trained on these games fails loudly instead of producing a
plausible number nobody re-examines.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import StrictBool
from torch import Tensor

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import (
    DataLoadingError,
    SequenceDataLoader,
)
from anthro_chess.data.schema import (
    SPLIT_NAMES,
    NormalizedColumn,
    SplitName,
    row_game_id,
)
from anthro_chess.evaluation.adjudication import (
    AdjudicationReport,
    action_sets,
    build_adjudication_report,
    merge_game_totals,
)
from anthro_chess.evaluation.aggregation import SliceTable
from anthro_chess.evaluation.dependency import (
    DEGRADATION_METRICS,
    Conditioning,
    ConditioningKind,
    DependencyError,
    DependencyTestConfig,
    DependencyTestResult,
    MaturityContext,
    PositionContext,
    PositionKey,
    TrajectorySignal,
    build_dependency_result,
)
from anthro_chess.evaluation.leakage import LeakageCheck, LeakageError, check_leakage
from anthro_chess.evaluation.noise import NoiseConfig, sampling_dispersions
from anthro_chess.evaluation.opening_frequency import (
    OpeningFrequency,
    OpeningFrequencyError,
    OpeningTailReading,
    count_opening_families,
    read_opening_tail,
)
from anthro_chess.evaluation.policy import (
    POLICY_SCORING_VERSION,
    ActionSetPolicy,
    PositionPolicy,
    active_batch,
    legal_policy_log_probabilities,
    policy_divergence,
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
    ResultRecorder,
    ResultRecording,
    checkpoint_reference,
    pool_dataset_reference,
)
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
    DatasetReference,
    Measurement,
    MetricDispersion,
    ResultEnvelope,
    measurement,
    projection_content_digest,
)
from anthro_chess.evaluation.results.metrics import (
    DEPENDENCY_RATING_ANCHOR_POLICY_DIVERGENCE,
    DEPENDENCY_RATING_ANCHOR_TOP1_AGREEMENT,
    DEPENDENCY_RATING_CROSS_CONDITIONING_MATCH_RATE,
    DEPENDENCY_RATING_WITHIN_GAME_RESPONSE,
    MOVE_PREDICTION_PROJECTION,
)
from anthro_chess.evaluation.results.noise import NoiseCharacterizationError
from anthro_chess.evaluation.scoring import (
    SCORED_COLUMNS,
    EvaluationLoaderConfig,
    ScoringInputs,
    aggregate_positions,
    build_scoring_inputs,
    per_game_totals,
    rows_identity_sha256,
    slice_measurements,
)
from anthro_chess.evaluation.selection import CheckpointSelection
from anthro_chess.evaluation.slices import SLICE_SCHEME_VERSION
from anthro_chess.evaluation.views import (
    DualSelection,
    ViewConfig,
    ViewSelection,
    apply_dual_view,
)
from anthro_chess.inference import CheckpointModelRunner
from anthro_chess.inference.runner import ModelRunnerError
from anthro_chess.models import MoveModelBatch, OptionalTensor

#: Version 2 carries each series' own dispersion where version 1 carried the
#: characterization the run recorded beside the reading.
CHECKPOINT_EVALUATION_VERSION = 2

HELD_OUT_KIND = "held-out-prediction"
DEPENDENCY_KIND = "rating-dependency"
ADJUDICATION_KIND = "adjudicated-decisions"
HELD_OUT_BENCHMARK = BenchmarkReference(name="held-out-prediction", version=1)
DEPENDENCY_BENCHMARK = BenchmarkReference(name="rating-dependency", version=1)
ADJUDICATION_BENCHMARK = BenchmarkReference(name="adjudicated-decisions", version=1)
#: One invocation produces three readings, and its cost belongs to none of
#: them: the scoring pass, the dependency treatments, and the adjudication all
#: happen once. The cost record names the whole evaluation instead.
CHECKPOINT_COST_BENCHMARK = BenchmarkReference(
    name="checkpoint-evaluation",
    version=CHECKPOINT_EVALUATION_VERSION,
)

logger = logging.getLogger(__name__)

_TRUE_CONDITIONING = Conditioning(name="true", kind=ConditioningKind.TRUE)


def _constant_conditioning(rating: int) -> Conditioning:
    """Return the treatment that shows every rated position one fixed rating.

    One place builds it, because the anchor comparison and the
    cross-conditioning table now share the passes it describes.
    """

    return Conditioning(
        name=f"constant-{rating}",
        kind=ConditioningKind.CONSTANT,
        rating=rating,
    )


class CheckpointEvaluationError(ValueError):
    """Raised when a checkpoint cannot be evaluated over a frozen pool."""


class LeakageConfig(ConfigModel):
    """Where to find the corpus a checkpoint trained on, when it moved."""

    training_normalized: Path | None = None


class DetailConfig(ConfigModel):
    """What the machine-local detail tier keeps beside a committed summary."""

    #: Per-position records are what decision decomposition and rollout
    #: comparisons need, and they are large. They stay opt-in rather than
    #: growing every routine evaluation by an order of magnitude.
    per_position: StrictBool = False


class OpeningConfig(ConfigModel):
    """Whether the reading places each opening family on a frequency axis."""

    #: Counting costs a replay of every training game's opening, so it scales
    #: with the training corpus rather than with the pool being scored. The
    #: whole opening reading hangs off it, because a per-family table without
    #: the frequency axis cannot answer the question it exists for.
    training_frequency: StrictBool = False


class CheckpointEvaluationConfig(CheckpointSelection, PoolGenerationPin):
    """Code-owned schema for ``anthro eval run``."""

    pool: Path
    view: ViewConfig = ViewConfig(name="canonical")
    loader: EvaluationLoaderConfig = EvaluationLoaderConfig()
    dependency: DependencyTestConfig = DependencyTestConfig()
    leakage: LeakageConfig = LeakageConfig()
    detail: DetailConfig = DetailConfig()
    noise: NoiseConfig = NoiseConfig()
    openings: OpeningConfig = OpeningConfig()


@dataclass(frozen=True)
class CheckpointEvaluationResult:
    """Everything one evaluation measured, plus where it was written."""

    checkpoint: CheckpointReference
    dataset: DatasetReference
    view: ViewSelection
    leakage: LeakageCheck
    slices: SliceTable
    adjudication: AdjudicationReport | None
    dependency: DependencyTestResult | None
    dispersions: Mapping[str, MetricDispersion]
    opening_frequency: OpeningFrequency | None = None
    opening_tail: OpeningTailReading | None = None
    envelopes: tuple[ResultEnvelope, ...] = ()
    recorded_paths: tuple[Path, ...] = ()
    detail_paths: tuple[Path, ...] = ()

    def as_record(self) -> dict[str, object]:
        """Return the full structured result, detail tier included."""

        return {
            "version": CHECKPOINT_EVALUATION_VERSION,
            "policy_scoring_version": POLICY_SCORING_VERSION,
            "slice_scheme_version": SLICE_SCHEME_VERSION,
            "checkpoint": self.checkpoint.model_dump(mode="json"),
            "dataset": self.dataset.model_dump(mode="json"),
            "view": self.view.as_record(),
            "leakage": self.leakage.as_record(),
            "slices": self.slices.as_record(),
            "adjudication": (
                self.adjudication.as_record() if self.adjudication is not None else None
            ),
            "dependency": (
                self.dependency.as_record() if self.dependency is not None else None
            ),
            "dispersions": {
                fingerprint: dispersion.model_dump(mode="json")
                for fingerprint, dispersion in sorted(self.dispersions.items())
            },
            "opening_frequency": (
                None
                if self.opening_frequency is None
                else self.opening_frequency.as_record()
            ),
            "opening_tail": (
                None if self.opening_tail is None else self.opening_tail.as_record()
            ),
            "recorded": [str(path) for path in self.recorded_paths],
        }


@dataclass(frozen=True)
class _EvaluationInputs:
    """The frozen games one evaluation scores, and everything derived once."""

    pool: FrozenPool
    dual: DualSelection
    scoring: ScoringInputs

    @property
    def selection(self) -> ViewSelection:
        """Return the view a scored game count and a log line describe."""

        return self.dual.current


class _ScoringSession:
    """Repeated deterministic passes over one view under varied conditioning."""

    def __init__(
        self,
        runner: CheckpointModelRunner,
        inputs: ScoringInputs,
        *,
        shuffle_seed: str,
    ) -> None:
        self._runner = runner
        self._inputs = inputs
        self._shuffled_ratings = _shuffled_ratings(inputs.contexts, shuffle_seed)

    def score(self, conditioning: Conditioning) -> tuple[PositionPolicy, ...]:
        """Score every position in the view under one conditioning treatment."""

        positions: list[PositionPolicy] = []
        for batch in self._batches():
            conditioned = self._condition(batch, conditioning)
            active = active_batch(self._runner.action_logits(conditioned), conditioned)
            positions.extend(score_positions(active))
        if not positions:
            raise CheckpointEvaluationError(
                "the configured view selected no positions to score"
            )
        return tuple(positions)

    def score_primary(
        self,
    ) -> tuple[tuple[PositionPolicy, ...], tuple[ActionSetPolicy, ...]]:
        """Score ordinary quantities and predicate action sets in one pass."""

        positions: list[PositionPolicy] = []
        adjudicated: list[ActionSetPolicy] = []
        subsets = action_sets(self._inputs)
        for batch in self._batches():
            active = active_batch(self._runner.action_logits(batch), batch)
            positions.extend(score_positions(active))
            adjudicated.extend(score_action_sets(active, subsets))
        if not positions:
            raise CheckpointEvaluationError(
                "the configured view selected no positions to score"
            )
        return tuple(positions), tuple(adjudicated)

    def trajectory(
        self,
        *,
        anchor_low: int,
        anchor_high: int,
    ) -> tuple[
        dict[PositionKey, TrajectorySignal],
        dict[int, tuple[PositionPolicy, ...]],
    ]:
        """Compare each position's policy at two anchor conditioning ratings.

        All three policies a signal needs are computed for one batch at a
        time. Retaining the true-conditioning policy from the primary pass
        would save a forward pass and cost a distribution per position held
        for the whole run, which is gigabytes over a full pool.

        The two anchors are the true-conditioning pass' own rows under other
        ratings, so the alignment that pass built is carried to them rather
        than rebuilt twice.

        The anchors' ordinary scores come back beside the signals, because
        both anchors are fixed-conditioning passes the cross-conditioning
        table wants anyway. That retention is a handful of scalars per
        position rather than a distribution, so it does not pay the cost the
        paragraph above declines — and without it these two conditionings run
        a second time.
        """

        signals: dict[PositionKey, TrajectorySignal] = {}
        anchors = (anchor_low, anchor_high)
        scored: dict[int, list[PositionPolicy]] = {rating: [] for rating in anchors}
        for batch in self._batches():
            true_batch = self._condition(batch, _TRUE_CONDITIONING)
            active = active_batch(self._runner.action_logits(true_batch), true_batch)
            true = legal_policy_log_probabilities(active)
            policies = []
            for rating in anchors:
                conditioned = self._condition(batch, _constant_conditioning(rating))
                rescored = active.rescored(
                    self._runner.action_logits(conditioned),
                    conditioned,
                )
                policies.append(legal_policy_log_probabilities(rescored))
                scored[rating].extend(score_positions(rescored))
            low, high = policies
            for offset, key in enumerate(
                zip(active.game_ids, active.ply_indices, strict=True)
            ):
                signals[key] = _trajectory_signal(
                    legal_actions=self._inputs.plies[key].enabled_actions(),
                    target_action_id=self._inputs.plies[key].target_action_id,
                    true=true[offset],
                    low=low[offset],
                    high=high[offset],
                )
        return signals, {
            rating: tuple(positions) for rating, positions in scored.items()
        }

    def _batches(self) -> Iterator[MoveModelBatch]:
        loader = SequenceDataLoader(self._inputs.dataset, self._inputs.loader_config)
        for sequence_batch in loader:
            yield MoveModelBatch.from_sequence_batch(
                sequence_batch,
                device=self._runner.device,
            )

    def _condition(
        self,
        batch: MoveModelBatch,
        conditioning: Conditioning,
    ) -> MoveModelBatch:
        rating = batch.inputs.target_rating
        if conditioning.kind is ConditioningKind.TRUE:
            return batch
        if conditioning.kind is ConditioningKind.ABSENT:
            replacement = OptionalTensor(
                values=torch.zeros_like(rating.values),
                present=torch.zeros_like(rating.present),
            )
        elif conditioning.kind is ConditioningKind.CONSTANT:
            replacement = OptionalTensor(
                values=torch.where(
                    rating.present,
                    torch.full_like(rating.values, int(conditioning.rating or 0)),
                    torch.zeros_like(rating.values),
                ),
                present=rating.present,
            )
        else:
            replacement = OptionalTensor(
                values=torch.as_tensor(
                    self._shuffled_values(batch),
                    dtype=rating.values.dtype,
                    device=rating.values.device,
                ),
                present=rating.present,
            )
        return replace(
            batch,
            inputs=replace(batch.inputs, target_rating=replacement),
        )

    def _shuffled_values(self, batch: MoveModelBatch) -> list[list[int]]:
        game_ids = batch.game_ids.detach().cpu().tolist()
        ply_indices = batch.ply_indices.detach().cpu().tolist()
        present = batch.inputs.target_rating.present.detach().cpu().tolist()
        return [
            [
                (
                    self._shuffled_ratings.get((int(game_id), int(ply_index)), 0)
                    if is_present
                    else 0
                )
                for game_id, ply_index, is_present in zip(
                    game_row, ply_row, present_row, strict=True
                )
            ]
            for game_row, ply_row, present_row in zip(
                game_ids, ply_indices, present, strict=True
            )
        ]


def evaluate_checkpoint(
    resolved_config: ResolvedConfig[CheckpointEvaluationConfig],
    *,
    run_root: Path | None = None,
    recording: ResultRecording,
) -> CheckpointEvaluationResult:
    """Evaluate one checkpoint over a frozen pool and record the result.

    A recording opened without a store computes everything and records nothing,
    which is what an exploratory reading wants: the committed tier should hold
    results somebody meant to keep.
    """

    config = resolved_config.value
    try:
        inputs = _load_inputs(config)
        runner = CheckpointModelRunner.load(config.model, run_root=run_root)
    except (
        DataLoadingError,
        EvaluationPoolError,
        ModelRunnerError,
        OSError,
        ValueError,
    ) as error:
        if isinstance(error, CheckpointEvaluationError):
            raise
        raise CheckpointEvaluationError(str(error)) from error

    leakage = check_leakage(
        inputs.pool,
        runner.metadata,
        training_normalized=config.leakage.training_normalized,
    )

    # Before the scoring pass rather than after it, for the reason the leakage
    # check runs where it does: this reads the same corpus and can fail on it,
    # and a failure that arrives after the passes have run discards them.
    frequency = _count_training_frequency(config, leakage)

    session = _ScoringSession(
        runner,
        inputs.scoring,
        shuffle_seed=config.dependency.shuffle_seed,
    )
    logger.info(
        "Scoring %s game(s) from pool view %r",
        inputs.selection.selected_games,
        inputs.selection.name,
    )
    positions, action_set_scores = session.score_primary()
    passes = (
        _score_conditionings(config, session, runner)
        if config.dependency.enabled
        else None
    )

    checkpoint = checkpoint_reference(runner, label=config.checkpoint_label)
    recorder = recording.measuring(
        checkpoint,
        kind=HELD_OUT_KIND,
        benchmark=HELD_OUT_BENCHMARK,
    )
    reported = [
        _report_view(
            selection,
            config=config,
            inputs=inputs,
            checkpoint=checkpoint,
            leakage=leakage,
            positions=positions,
            action_set_scores=action_set_scores,
            passes=passes,
            frequency=frequency,
            recorder=recorder,
        )
        for selection in inputs.dual.reported
    ]
    return reported[0]


def _load_inputs(config: CheckpointEvaluationConfig) -> _EvaluationInputs:
    pool = load_pool(
        config.pool,
        expected_game_ids_sha256=config.expected_pool_game_ids_sha256,
    )
    dual = apply_dual_view(pool.games, config.view, pool.core)
    if not dual.current.game_ids:
        raise CheckpointEvaluationError(
            f"view {config.view.name!r} selected no games from the pool"
        )

    # The union of both views, scored once. Where a core is designated the two
    # views overlap heavily and diverge only as the pool grows past the core.
    scored = dual.scored_game_ids
    rows = [
        _truncate(row, dual.current.prefix_plies)
        for row in pool_rows(
            pool,
            scored,
            SCORED_COLUMNS,
            error=CheckpointEvaluationError,
        )
    ]
    scoring = build_scoring_inputs(
        rows,
        split=_pool_split(pool),
        batch_size=config.loader.batch_size,
        length_bucket_width=config.loader.length_bucket_width,
        identity_sha256=rows_identity_sha256(rows, context=dual.current.as_record()),
    )
    return _EvaluationInputs(pool=pool, dual=dual, scoring=scoring)


def _estimate_dispersions(
    config: CheckpointEvaluationConfig,
    inputs: _EvaluationInputs,
    positions: Sequence[PositionPolicy],
    adjudication: AdjudicationReport | None,
    dependency: DependencyTestResult | None,
    component: DataComponent,
    *,
    opening_frequency: OpeningFrequency | None,
) -> dict[str, MetricDispersion]:
    """Estimate this reading's own data-sampling spread from the same pass.

    The estimate costs one resampling of numbers already computed, so it is on
    by default. A reading with no spread beside it can only report that a number
    moved.
    """

    if not config.noise.enabled:
        return {}
    try:
        adjudication_totals = (
            () if adjudication is None else adjudication.per_game_totals()
        )
        dependency_totals = () if dependency is None else dependency.per_game_totals
        return sampling_dispersions(
            merge_game_totals(
                per_game_totals(
                    positions,
                    inputs.scoring,
                    opening_frequency=opening_frequency,
                ),
                adjudication_totals,
                dependency_totals,
            ),
            component=component,
            config=config.noise,
            source=(
                f"bootstrap over {inputs.selection.selected_games} game(s) of "
                f"pool view {inputs.selection.name!r}"
            ),
        )
    except NoiseCharacterizationError as error:
        # A view too small to resample is a fact about the view, not a failed
        # evaluation. The reading still stands; it simply has no spread.
        logger.warning("Skipping the data-sampling estimate: %s", error)
        return {}


def _count_training_frequency(
    config: CheckpointEvaluationConfig,
    leakage: LeakageCheck,
) -> OpeningFrequency | None:
    """Count how often the training selection saw each opening family.

    The corpus and split come from the leakage check rather than being resolved
    a second time, so the frequency axis is counted over exactly the games that
    check proved this checkpoint trained on.

    That corpus has to be the one the pool was drawn from, or a committed tier
    would mean a share of games no fingerprint records; `docs/evaluation.md`
    argues it. What settles the question is the training manifest the checkpoint
    recorded, so a ``leakage.training_normalized`` override still has to name a
    copy of that corpus rather than a subset of it.
    """

    if not config.openings.training_frequency:
        return None
    if not leakage.same_source_corpus:
        raise CheckpointEvaluationError(
            "the opening-frequency axis needs the checkpoint's training corpus "
            "and this pool's source corpus to be the same one; they are not, "
            "so a family's training share says nothing about the games scored"
        )
    logger.info(
        "Classifying the %s split of the training corpus by opening family",
        leakage.training_split,
    )
    try:
        return count_opening_families(
            [Path(path) for path in leakage.training_normalized_paths],
            leakage.training_split,
        )
    except OpeningFrequencyError as error:
        raise CheckpointEvaluationError(str(error)) from error


def _report_view(
    selection: ViewSelection,
    *,
    config: CheckpointEvaluationConfig,
    inputs: _EvaluationInputs,
    checkpoint: CheckpointReference,
    leakage: LeakageCheck,
    positions: Sequence[PositionPolicy],
    action_set_scores: Sequence[ActionSetPolicy],
    passes: _ConditioningPasses | None,
    frequency: OpeningFrequency | None,
    recorder: ResultRecorder,
) -> CheckpointEvaluationResult:
    """Aggregate one scoring pass under one view, and record what it measured.

    Every quantity here is derived from games already scored, so the second
    view costs aggregation rather than another pass over the pool.
    """

    game_ids = frozenset(selection.game_ids)
    scored = _within(positions, game_ids)
    slices = aggregate_positions(scored, inputs.scoring, opening_frequency=frequency)
    opening_tail = None if frequency is None else read_opening_tail(slices, frequency)
    adjudication = build_adjudication_report(
        [item for item in action_set_scores if item.game_id in game_ids],
        inputs.scoring,
        game_ids=game_ids,
    )
    dependency = (
        None
        if passes is None
        else _dependency_for(config, inputs, passes, positions, game_ids)
    )
    component = projection_content_digest(
        [row for row in inputs.scoring.rows if row_game_id(row) in game_ids],
        MOVE_PREDICTION_PROJECTION,
    )
    data = pool_dataset_reference(
        inputs.pool,
        selection,
        component,
        error=CheckpointEvaluationError,
    )
    dispersions = _estimate_dispersions(
        config,
        inputs,
        scored,
        adjudication,
        dependency,
        component,
        opening_frequency=frequency,
    )
    recorder.disperse(dispersions)
    result = CheckpointEvaluationResult(
        checkpoint=checkpoint,
        dataset=data,
        view=selection,
        leakage=leakage,
        slices=slices,
        adjudication=adjudication,
        dependency=dependency,
        dispersions=dispersions,
        opening_frequency=frequency,
        opening_tail=opening_tail,
    )
    recorder.add(
        slice_measurements(slices, component),
        payload=lambda: {
            **result.as_record(),
            "positions": (
                [position.as_record() for position in scored]
                if config.detail.per_position
                else None
            ),
        },
        description="Slice tables and view provenance for one evaluation.",
        slug=selection.name,
        data=data,
    )
    if adjudication is not None:
        recorder.add(
            adjudication.measurements(component),
            kind=ADJUDICATION_KIND,
            benchmark=ADJUDICATION_BENCHMARK,
            payload=adjudication.as_record,
            description=(
                "Per-predicate human and model rates with rating-band "
                "drill-down and opportunity counts."
            ),
            slug=selection.name,
            data=data,
        )
    if dependency is not None:
        recorder.add(
            _dependency_measurements(dependency, component),
            kind=DEPENDENCY_KIND,
            benchmark=DEPENDENCY_BENCHMARK,
            payload=dependency.as_record,
            description="Cross-conditioning and within-game dependency tables.",
            slug=selection.name,
            data=data,
        )
    return result


@dataclass(frozen=True)
class _ConditioningPasses:
    """Every conditioning pass one dependency reading is assembled from.

    Scored once over the union of the reported views and aggregated per view,
    because a second pass here is nine more passes over the pool.
    """

    corrupted: Mapping[str, tuple[Conditioning, Sequence[PositionPolicy]]]
    conditioned: Mapping[int, Sequence[PositionPolicy]]
    trajectory: Mapping[PositionKey, TrajectorySignal]
    maturity: MaturityContext


def _score_conditionings(
    config: CheckpointEvaluationConfig,
    session: _ScoringSession,
    runner: CheckpointModelRunner,
) -> _ConditioningPasses:
    settings = config.dependency
    values = settings.conditioning_values()
    corrupted: dict[str, tuple[Conditioning, Sequence[PositionPolicy]]] = {}
    for conditioning in (
        Conditioning(name="shuffled", kind=ConditioningKind.SHUFFLED),
        Conditioning(
            name="constant",
            kind=ConditioningKind.CONSTANT,
            rating=settings.constant_rating,
        ),
        Conditioning(name="absent", kind=ConditioningKind.ABSENT),
    ):
        logger.info("Scoring under %s rating conditioning", conditioning.name)
        corrupted[conditioning.name] = (conditioning, session.score(conditioning))

    logger.info("Comparing anchor policies at ratings %s and %s", values[0], values[-1])
    trajectory, conditioned = session.trajectory(
        anchor_low=values[0], anchor_high=values[-1]
    )

    # The anchor comparison above already scored the outermost two
    # conditionings, so only the values it did not cover need a pass here.
    for value in values:
        if value in conditioned:
            continue
        logger.info("Scoring under a fixed conditioning rating of %s", value)
        conditioned[value] = session.score(_constant_conditioning(value))
    return _ConditioningPasses(
        corrupted=corrupted,
        conditioned=conditioned,
        trajectory=trajectory,
        maturity=MaturityContext(
            step=runner.global_step,
            processed_positions=runner.processed_positions,
        ),
    )


def _dependency_for(
    config: CheckpointEvaluationConfig,
    inputs: _EvaluationInputs,
    passes: _ConditioningPasses,
    positions: Sequence[PositionPolicy],
    game_ids: frozenset[int],
) -> DependencyTestResult:
    """Assemble the dependency reading over one view's share of the passes."""

    try:
        return build_dependency_result(
            config=config.dependency,
            contexts=inputs.scoring.contexts,
            true_positions=_within(positions, game_ids),
            corrupted_positions={
                name: (conditioning, _within(scored, game_ids))
                for name, (conditioning, scored) in passes.corrupted.items()
            },
            conditioned_positions={
                rating: _within(scored, game_ids)
                for rating, scored in passes.conditioned.items()
            },
            trajectory=passes.trajectory,
            maturity=passes.maturity,
        )
    except DependencyError as error:
        raise CheckpointEvaluationError(str(error)) from error


def _within(
    positions: Sequence[PositionPolicy],
    game_ids: frozenset[int],
) -> tuple[PositionPolicy, ...]:
    return tuple(position for position in positions if position.game_id in game_ids)


def _dependency_measurements(
    dependency: DependencyTestResult,
    component: DataComponent,
) -> tuple[Measurement, ...]:
    values: list[Measurement] = []
    for kind, definition in DEGRADATION_METRICS.items():
        result = dependency.corruption(kind)
        if result is None:
            continue
        values.append(
            measurement(
                definition.identifier,
                result.degradation,
                data=component,
                sample_size=result.position_count,
            )
        )
    values.append(
        measurement(
            DEPENDENCY_RATING_ANCHOR_POLICY_DIVERGENCE.identifier,
            dependency.anchor_divergence,
            data=component,
            sample_size=dependency.rated_position_count,
        )
    )
    values.append(
        measurement(
            DEPENDENCY_RATING_ANCHOR_TOP1_AGREEMENT.identifier,
            dependency.anchor_agreement_rate,
            data=component,
            sample_size=dependency.rated_position_count,
        )
    )
    match_rate = dependency.cross_conditioning.match_rate
    if match_rate is not None:
        values.append(
            measurement(
                DEPENDENCY_RATING_CROSS_CONDITIONING_MATCH_RATE.identifier,
                match_rate,
                data=component,
                sample_size=len(dependency.cross_conditioning.compared_bands),
            )
        )
    response = dependency.within_game.response
    if response is not None:
        values.append(
            measurement(
                DEPENDENCY_RATING_WITHIN_GAME_RESPONSE.identifier,
                response,
                data=component,
                sample_size=dependency.within_game.positions_with_prefix,
            )
        )
    return tuple(values)


def _pool_split(pool: FrozenPool) -> SplitName:
    record = pool.manifest.get("pool")
    split = record.get("split") if isinstance(record, Mapping) else None
    if split not in SPLIT_NAMES:
        raise CheckpointEvaluationError(
            f"evaluation pool manifest names an unknown split: {split!r}"
        )
    return cast(SplitName, split)


def _truncate(row: Mapping[str, Any], prefix_plies: int | None) -> dict[str, Any]:
    """Project one pool game onto the prefix a view selected.

    A prefix view scores fewer plies of the same games, so the truncation has
    to reach the rows the content digest is computed over. Otherwise a prefix
    reading and a full reading would claim the same series.
    """

    updated = dict(row)
    if prefix_plies is None:
        return updated
    for column in (
        NormalizedColumn.ACTION_IDS,
        NormalizedColumn.CLOCK_REMAINING_DELTA_MS,
    ):
        updated[column.value] = list(updated[column.value])[:prefix_plies]
    updated[NormalizedColumn.PLY_COUNT.value] = len(
        updated[NormalizedColumn.ACTION_IDS.value]
    )
    return updated


def _shuffled_ratings(
    contexts: Mapping[PositionKey, PositionContext],
    seed: str,
) -> dict[PositionKey, int]:
    """Deal every present rating to a different position, deterministically."""

    keys = sorted(
        key for key, context in contexts.items() if context.rating is not None
    )
    values = [contexts[key].rating for key in keys]
    random.Random(seed).shuffle(values)
    return {
        key: int(value)
        for key, value in zip(keys, values, strict=True)
        if value is not None
    }


def _trajectory_signal(
    *,
    legal_actions: Sequence[int],
    target_action_id: int,
    true: Tensor,
    low: Tensor,
    high: Tensor,
) -> TrajectorySignal:
    target_offset = legal_actions.index(target_action_id)
    return TrajectorySignal(
        strength_signal=float(high[target_offset].item() - low[target_offset].item()),
        alignment=policy_divergence(true, low) - policy_divergence(true, high),
        anchor_divergence=policy_divergence(low, high),
        anchor_agreement=int(torch.argmax(low).item())
        == int(torch.argmax(high).item()),
    )


__all__ = [
    "CHECKPOINT_EVALUATION_VERSION",
    "DEPENDENCY_KIND",
    "HELD_OUT_KIND",
    "CheckpointEvaluationConfig",
    "CheckpointEvaluationError",
    "CheckpointEvaluationResult",
    "DetailConfig",
    "EvaluationLoaderConfig",
    "LeakageConfig",
    "LeakageError",
    "evaluate_checkpoint",
]
