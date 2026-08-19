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
from array import array
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import chess
import torch
from pydantic import StrictBool
from torch import Tensor

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import (
    DataLoadingError,
    SequenceLoaderConfig,
    collate_sequences,
    length_bucketed_batches,
)
from anthro_chess.data.schema import (
    NormalizedColumn,
    SplitName,
)
from anthro_chess.evaluation.adjudication import (
    AdjudicationAccumulator,
    AdjudicationReport,
    action_sets,
    merge_game_totals,
)
from anthro_chess.evaluation.aggregation import SliceAggregator, SliceTable
from anthro_chess.evaluation.dependency import (
    DEGRADATION_METRICS,
    Conditioning,
    ConditioningKind,
    DependencyColumnBuilder,
    DependencyError,
    DependencyTestConfig,
    DependencyTestResult,
    MaturityContext,
    PositionKey,
    TrajectorySignal,
    reduce_dependency_columns,
)
from anthro_chess.evaluation.leakage import LeakageCheck, LeakageError, check_leakage
from anthro_chess.evaluation.noise import (
    GameTotals,
    NoiseConfig,
    sampling_dispersions,
)
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
    ActiveBatch,
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
    PoolProjection,
    load_pool,
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
    ScoringError,
    ScoringInputs,
    accumulate_positions,
    build_scoring_inputs,
    per_game_totals,
    rows_identity_sha256,
    slice_measurements,
)
from anthro_chess.evaluation.selection import CheckpointSelection
from anthro_chess.evaluation.slices import SLICE_SCHEME_VERSION
from anthro_chess.evaluation.views import ViewConfig, ViewSelection, apply_view
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
class _PoolReading:
    """The pool, the view over it, and the batches one reading will score.

    Planned before anything is encoded. The batch a game lands in follows from
    how many actions it holds, which the pool can answer columnar, so the
    reading knows every batch it will score while holding none of them.
    """

    pool: FrozenPool
    selection: ViewSelection
    projection: PoolProjection
    split: SplitName
    loader: SequenceLoaderConfig
    prefix_plies: int | None
    #: Selected game ids in ascending order, which is the order
    #: :func:`build_scoring_inputs` puts a whole view in.
    game_ids: tuple[int, ...]
    #: How many timesteps each selected game encodes to, after any prefix.
    encoded_plies: Mapping[int, int]
    batches: tuple[tuple[int, ...], ...]
    #: The view record each batch's loader identity is taken against. Held
    #: rather than rebuilt, because deriving it digests every selected game id.
    identity_context: dict[str, object]

    def rows(self, game_ids: Sequence[int]) -> tuple[dict[str, Any], ...]:
        """Return one batch's games, projected onto the view's prefix."""

        return tuple(
            _truncate(row, self.prefix_plies) for row in self.projection.rows(game_ids)
        )

    def projected_rows(self) -> Iterator[dict[str, Any]]:
        """Yield every selected game's projected row, one batch at a time."""

        for game_ids in self.batches:
            yield from self.rows(game_ids)

    def inputs(self, game_ids: Sequence[int]) -> ScoringInputs:
        """Encode one batch's games, and nothing else.

        Length is what the plan bucketed on, so a disagreement means this is
        not the batch the loader would have built, and the forward pass is not
        reproducible across batch shapes.
        """

        rows = self.rows(game_ids)
        encoded = build_scoring_inputs(
            rows,
            split=self.split,
            batch_size=self.loader.batch_size,
            length_bucket_width=self.loader.length_bucket_width,
            identity_sha256=rows_identity_sha256(rows, context=self.identity_context),
        )
        for example in encoded.dataset:
            planned = self.encoded_plies[example.game_id]
            if len(example.plies) != planned:
                raise CheckpointEvaluationError(
                    f"game {example.game_id} encoded to {len(example.plies)} "
                    f"timestep(s) where the batch plan counted {planned}"
                )
        return encoded


class _ShuffledRatings:
    """The dependency test's rating permutation, dealt over the whole view.

    Every rated position is dealt another rated position's rating, so the
    treatment shows the model a wrong value drawn from the distribution it was
    trained on rather than one from nowhere. That makes it a property of the
    view rather than of a batch, and it has to exist before the first batch is
    scored.

    Derived from the pool's columns rather than from encodings: which rating a
    decision carries follows from whose turn it is, and whose turn it is
    follows from the ply and the position the game opened in.
    """

    def __init__(self, reading: _PoolReading, seed: str) -> None:
        projection = reading.projection
        by_game = dict(
            zip(
                projection.game_ids(),
                zip(
                    projection.column(NormalizedColumn.WHITE_NORMALIZED_RATING.value),
                    projection.column(NormalizedColumn.BLACK_NORMALIZED_RATING.value),
                    projection.column(NormalizedColumn.INITIAL_POSITION.value),
                    strict=True,
                ),
                strict=True,
            )
        )
        openings: dict[str, bool] = {}
        spans: dict[int, tuple[int, int]] = {}
        values: array[int] = array("i")
        # Where each rated position sits, so the deal below can be compacted
        # rather than carrying an entry for positions it does not cover.
        rated: array[int] = array("q")
        for game_id in reading.game_ids:
            white, black, opening = by_game[game_id]
            white_opens = openings.get(opening)
            if white_opens is None:
                white_opens = chess.Board(opening).turn == chess.WHITE
                openings[opening] = white_opens
            plies = reading.encoded_plies[game_id]
            spans[game_id] = (len(values), plies)
            for ply in range(plies):
                rating = white if white_opens == (ply % 2 == 0) else black
                if rating is not None:
                    rated.append(len(values))
                values.append(int(rating or 0))

        dealt = array("i", (values[index] for index in rated))
        random.Random(seed).shuffle(dealt)
        for index, rating in zip(rated, dealt, strict=True):
            values[index] = rating
        self._spans = spans
        self._values = values

    def value(self, game_id: int, ply_index: int) -> int:
        """Return the rating one position is dealt, zero where it has none."""

        span = self._spans.get(game_id)
        if span is None:
            return 0
        offset, plies = span
        if not 0 <= ply_index < plies:
            return 0
        return int(self._values[offset + ply_index])


@dataclass(frozen=True)
class _BatchScores:
    """Everything one batch of games contributes to a reading.

    The conditioning maps are empty where the dependency family is off.
    """

    positions: tuple[PositionPolicy, ...]
    action_sets: tuple[ActionSetPolicy, ...]
    corrupted: Mapping[str, tuple[Conditioning, tuple[PositionPolicy, ...]]]
    conditioned: Mapping[int, tuple[PositionPolicy, ...]]
    trajectory: Mapping[PositionKey, TrajectorySignal]


class _BatchSession:
    """Every conditioning treatment, applied to one batch before the next.

    A position's scores exist together only while its batch does.
    """

    def __init__(
        self,
        runner: CheckpointModelRunner,
        config: DependencyTestConfig,
        shuffled: _ShuffledRatings | None,
    ) -> None:
        self._runner = runner
        self._config = config
        self._shuffled = shuffled

    def score(self, inputs: ScoringInputs) -> _BatchScores:
        """Score one batch under every treatment the configuration asks for."""

        batch = self._batch(inputs)
        active = active_batch(self._runner.action_logits(batch), batch)
        positions = score_positions(active)
        if not positions:
            raise CheckpointEvaluationError(
                "the configured view selected no positions to score"
            )
        adjudicated = score_action_sets(active, action_sets(inputs))
        if self._shuffled is None:
            return _BatchScores(positions, adjudicated, {}, {}, {})

        corrupted: dict[str, tuple[Conditioning, tuple[PositionPolicy, ...]]] = {}
        conditioned: dict[int, tuple[PositionPolicy, ...]] = {}
        trajectory: dict[PositionKey, TrajectorySignal] = {}
        for conditioning in (
            Conditioning(name="shuffled", kind=ConditioningKind.SHUFFLED),
            Conditioning(
                name="constant",
                kind=ConditioningKind.CONSTANT,
                rating=self._config.constant_rating,
            ),
            Conditioning(name="absent", kind=ConditioningKind.ABSENT),
        ):
            corrupted[conditioning.name] = (
                conditioning,
                self._conditioned(batch, conditioning),
            )

        values = self._config.conditioning_values()
        self._trajectory(
            batch, active, inputs, conditioned, trajectory, values[0], values[-1]
        )
        for value in values:
            if value in conditioned:
                continue
            conditioned[value] = self._conditioned(batch, _constant_conditioning(value))
        return _BatchScores(positions, adjudicated, corrupted, conditioned, trajectory)

    def _trajectory(
        self,
        batch: MoveModelBatch,
        active: ActiveBatch,
        inputs: ScoringInputs,
        conditioned: dict[int, tuple[PositionPolicy, ...]],
        trajectory: dict[PositionKey, TrajectorySignal],
        anchor_low: int,
        anchor_high: int,
    ) -> None:
        """Compare each position's policy at two anchor conditioning ratings.

        All three policies a signal needs are computed for one batch, which is
        also the only span over which the distributions themselves exist: a
        policy is a value per legal action, and retaining one per position is
        gigabytes over a pool.

        The anchors' ordinary scores come back beside the signals, because both
        anchors are fixed-conditioning passes the cross-conditioning table
        wants anyway. That retention is a handful of scalars per position, and
        without it these two conditionings run a second time.
        """

        true = legal_policy_log_probabilities(active)
        policies = []
        for rating in (anchor_low, anchor_high):
            anchored = self._condition(batch, _constant_conditioning(rating))
            rescored = active.rescored(
                self._runner.action_logits(anchored),
                anchored,
            )
            policies.append(legal_policy_log_probabilities(rescored))
            conditioned[rating] = score_positions(rescored)
        low, high = policies
        for offset, key in enumerate(
            zip(active.game_ids, active.ply_indices, strict=True)
        ):
            trajectory[key] = _trajectory_signal(
                legal_actions=inputs.plies[key].enabled_actions(),
                target_action_id=inputs.plies[key].target_action_id,
                true=true[offset],
                low=low[offset],
                high=high[offset],
            )

    def _conditioned(
        self,
        batch: MoveModelBatch,
        conditioning: Conditioning,
    ) -> tuple[PositionPolicy, ...]:
        conditioned = self._condition(batch, conditioning)
        active = active_batch(self._runner.action_logits(conditioned), conditioned)
        return score_positions(active)

    def _batch(self, inputs: ScoringInputs) -> MoveModelBatch:
        return MoveModelBatch.from_sequence_batch(
            collate_sequences(inputs.dataset),
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
        assert self._shuffled is not None
        game_ids = batch.game_ids.detach().cpu().tolist()
        ply_indices = batch.ply_indices.detach().cpu().tolist()
        present = batch.inputs.target_rating.present.detach().cpu().tolist()
        return [
            [
                (
                    self._shuffled.value(int(game_id), int(ply_index))
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


@dataclass(frozen=True)
class _Measured:
    """What one streamed pass over a view measured."""

    slices: SliceTable
    adjudication: AdjudicationReport | None
    dependency: DependencyTestResult | None
    game_totals: tuple[GameTotals, ...]
    positions: tuple[PositionPolicy, ...] | None


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
        reading = _open_reading(config)
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
        reading.pool,
        runner.metadata,
        training_normalized=config.leakage.training_normalized,
    )

    # Before the scoring pass rather than after it, for the reason the leakage
    # check runs where it does: this reads the same corpus and can fail on it,
    # and a failure that arrives after the passes have run discards them.
    frequency = _count_training_frequency(config, leakage)

    logger.info(
        "Scoring %s game(s) from pool view %r in %s batch(es)",
        reading.selection.selected_games,
        reading.selection.name,
        len(reading.batches),
    )
    try:
        measured = _score(config, reading, runner, frequency)
    except (DataLoadingError, DependencyError, ScoringError, ValueError) as error:
        if isinstance(error, CheckpointEvaluationError):
            raise
        raise CheckpointEvaluationError(str(error)) from error
    opening_tail = (
        None if frequency is None else read_opening_tail(measured.slices, frequency)
    )

    component = projection_content_digest(
        reading.projected_rows(),
        MOVE_PREDICTION_PROJECTION,
    )
    checkpoint = checkpoint_reference(runner, label=config.checkpoint_label)
    data = pool_dataset_reference(
        reading.pool,
        reading.selection,
        component,
        error=CheckpointEvaluationError,
    )

    recorder = recording.measuring(
        checkpoint,
        kind=HELD_OUT_KIND,
        benchmark=HELD_OUT_BENCHMARK,
    )
    dispersions = _estimate_dispersions(config, reading, measured, component)
    recorder.disperse(dispersions)
    result = CheckpointEvaluationResult(
        checkpoint=checkpoint,
        dataset=data,
        view=reading.selection,
        leakage=leakage,
        slices=measured.slices,
        adjudication=measured.adjudication,
        dependency=measured.dependency,
        dispersions=dispersions,
        opening_frequency=frequency,
        opening_tail=opening_tail,
    )
    recorder.add(
        slice_measurements(measured.slices, component),
        payload=lambda: {
            **result.as_record(),
            "positions": (
                None
                if measured.positions is None
                else [position.as_record() for position in measured.positions]
            ),
        },
        description="Slice tables and view provenance for one evaluation.",
        data=data,
    )
    if measured.adjudication is not None:
        recorder.add(
            measured.adjudication.measurements(component),
            kind=ADJUDICATION_KIND,
            benchmark=ADJUDICATION_BENCHMARK,
            payload=measured.adjudication.as_record,
            description=(
                "Per-predicate human and model rates with rating-band "
                "drill-down and opportunity counts."
            ),
            data=data,
        )
    if measured.dependency is not None:
        recorder.add(
            _dependency_measurements(measured.dependency, component),
            kind=DEPENDENCY_KIND,
            benchmark=DEPENDENCY_BENCHMARK,
            payload=measured.dependency.as_record,
            description="Cross-conditioning and within-game dependency tables.",
            data=data,
        )
    return result


def _score(
    config: CheckpointEvaluationConfig,
    reading: _PoolReading,
    runner: CheckpointModelRunner,
    frequency: OpeningFrequency | None,
) -> _Measured:
    """Score every batch of the view, keeping only what outlives its batch."""

    dependency = config.dependency.enabled
    session = _BatchSession(
        runner,
        config.dependency,
        _ShuffledRatings(reading, config.dependency.shuffle_seed)
        if dependency
        else None,
    )
    aggregator = SliceAggregator()
    adjudication = AdjudicationAccumulator(retain_positions=config.detail.per_position)
    columns = DependencyColumnBuilder() if dependency else None
    resampled = config.noise.enabled
    totals: list[GameTotals] = []
    retained: list[PositionPolicy] | None = [] if config.detail.per_position else None

    for game_ids in reading.batches:
        inputs = reading.inputs(game_ids)
        scores = session.score(inputs)
        accumulate_positions(
            aggregator,
            scores.positions,
            inputs,
            opening_frequency=frequency,
        )
        adjudication.add(scores.action_sets, inputs)
        if resampled:
            totals.extend(
                per_game_totals(scores.positions, inputs, opening_frequency=frequency)
            )
        if columns is not None:
            columns.add(
                inputs.contexts,
                scores.positions,
                scores.corrupted,
                scores.conditioned,
                scores.trajectory,
            )
        if retained is not None:
            retained.extend(scores.positions)

    report = adjudication.report()
    result = None
    if columns is not None:
        result = reduce_dependency_columns(
            config=config.dependency,
            columns=columns.build(),
            maturity=MaturityContext(
                step=runner.global_step,
                processed_positions=runner.processed_positions,
            ),
        )
    return _Measured(
        slices=aggregator.compute(),
        adjudication=report,
        dependency=result,
        game_totals=(
            merge_game_totals(
                tuple(totals),
                () if report is None else report.per_game_totals,
                () if result is None else result.per_game_totals,
            )
            if resampled
            else ()
        ),
        positions=None if retained is None else tuple(retained),
    )


def _open_reading(config: CheckpointEvaluationConfig) -> _PoolReading:
    """Resolve the pool, the view, and the batches a reading will score."""

    pool = load_pool(
        config.pool,
        expected_game_ids_sha256=config.expected_pool_game_ids_sha256,
    )
    selection = apply_view(pool.games, config.view)
    if not selection.game_ids:
        raise CheckpointEvaluationError(
            f"view {config.view.name!r} selected no games from the pool"
        )
    projection = PoolProjection(
        pool,
        SCORED_COLUMNS,
        error=CheckpointEvaluationError,
    )
    actions = projection.encoded_ply_counts()
    missing = tuple(game_id for game_id in selection.game_ids if game_id not in actions)
    if missing:
        raise CheckpointEvaluationError(
            "the evaluation pool does not contain every selected game"
        )
    game_ids = tuple(sorted(selection.game_ids))
    prefix = selection.prefix_plies
    encoded = {
        game_id: (actions[game_id] if prefix is None else min(actions[game_id], prefix))
        for game_id in game_ids
    }
    loader = SequenceLoaderConfig(
        split=pool.split,
        batch_size=config.loader.batch_size,
        length_bucket_width=config.loader.length_bucket_width,
        chunk_length=None,
        shuffle=False,
        drop_last=False,
    )
    plan = length_bucketed_batches(
        [encoded[game_id] for game_id in game_ids],
        loader,
    )
    return _PoolReading(
        pool=pool,
        selection=selection,
        projection=projection,
        split=loader.split,
        loader=loader,
        prefix_plies=prefix,
        game_ids=game_ids,
        encoded_plies=encoded,
        batches=tuple(tuple(game_ids[index] for index in batch) for batch in plan),
        identity_context=selection.as_record(),
    )


def _estimate_dispersions(
    config: CheckpointEvaluationConfig,
    reading: _PoolReading,
    measured: _Measured,
    component: DataComponent,
) -> dict[str, MetricDispersion]:
    """Estimate this reading's own data-sampling spread from the same pass.

    The estimate costs one resampling of numbers already computed, so it is on
    by default. A reading with no spread beside it can only report that a number
    moved.
    """

    if not config.noise.enabled:
        return {}
    try:
        return sampling_dispersions(
            measured.game_totals,
            component=component,
            config=config.noise,
            source=(
                f"bootstrap over {reading.selection.selected_games} game(s) of "
                f"pool view {reading.selection.name!r}"
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
