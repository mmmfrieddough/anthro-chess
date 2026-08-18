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
from typing import Any, cast

import chess
import torch
from pydantic import StrictBool
from torch import Tensor

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import (
    DataLoadingError,
    SequenceDataLoader,
    SequenceLoaderConfig,
    length_bucketed_batches,
)
from anthro_chess.data.schema import (
    SPLIT_NAMES,
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
    DependencyColumns,
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

    @property
    def scored_positions(self) -> int:
        """Return how many decisions one pass over this view covers."""

        return sum(self.encoded_plies.values())

    def rows(self, game_ids: Sequence[int]) -> tuple[dict[str, Any], ...]:
        """Return one batch's games, projected onto the view's prefix."""

        return tuple(
            _truncate(row, self.prefix_plies) for row in self.projection.rows(game_ids)
        )

    def projected_rows(self) -> Iterator[dict[str, Any]]:
        """Yield every selected game's projected row, one batch at a time.

        The content digest reads the games rather than the scores, and it reads
        each one once. Streaming it keeps the reading's own rule: a batch of
        rows exists while it is being consumed and not afterwards.
        """

        for game_ids in self.batches:
            yield from self.rows(game_ids)

    def inputs(self, game_ids: Sequence[int]) -> ScoringInputs:
        """Encode one batch's games, and nothing else."""

        rows = self.rows(game_ids)
        return build_scoring_inputs(
            rows,
            split=self.split,
            batch_size=self.loader.batch_size,
            length_bucket_width=self.loader.length_bucket_width,
            identity_sha256=rows_identity_sha256(
                rows, context=self.selection.as_record()
            ),
        )


class _ShuffledRatings:
    """The dependency test's rating permutation, dealt over the whole view.

    Every rated position is dealt another rated position's rating, so the
    treatment shows the model a wrong value drawn from the distribution it was
    trained on rather than one from nowhere. That makes it a property of the
    view rather than of a batch, and it has to exist before the first batch is
    scored.

    Derived from the pool's columns rather than from encodings: which rating a
    decision carries follows from whose turn it is, and whose turn it is
    follows from the ply and the position the game opened in. Encoding the pool
    to find that out is what the streaming reading exists to avoid.
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
        offsets: dict[int, int] = {}
        ratings: list[int | None] = []
        for game_id in reading.game_ids:
            white, black, opening = by_game[game_id]
            white_opens = openings.get(opening)
            if white_opens is None:
                white_opens = chess.Board(opening).turn == chess.WHITE
                openings[opening] = white_opens
            offsets[game_id] = len(ratings)
            ratings.extend(
                white if white_opens == (ply % 2 == 0) else black
                for ply in range(reading.encoded_plies[game_id])
            )

        rated = [index for index, rating in enumerate(ratings) if rating is not None]
        dealt = [ratings[index] for index in rated]
        random.Random(seed).shuffle(dealt)
        self._offsets = offsets
        self._values = array("i", bytes(4 * len(ratings)))
        for index, rating in zip(rated, dealt, strict=True):
            self._values[index] = int(rating or 0)

    def value(self, game_id: int, ply_index: int) -> int:
        """Return the rating one position is dealt, zero where it has none."""

        offset = self._offsets.get(game_id)
        if offset is None:
            return 0
        index = offset + ply_index
        return int(self._values[index]) if index < len(self._values) else 0


@dataclass(frozen=True)
class _BatchScores:
    """Everything one batch of games contributes to a reading."""

    positions: tuple[PositionPolicy, ...]
    action_sets: tuple[ActionSetPolicy, ...]
    corrupted: dict[str, tuple[Conditioning, tuple[PositionPolicy, ...]]]
    conditioned: dict[int, tuple[PositionPolicy, ...]]
    trajectory: dict[PositionKey, TrajectorySignal]


class _BatchSession:
    """Every conditioning treatment, applied to one batch before the next.

    The passes are nested inside the traversal rather than wrapped around it.
    Running them as separate sweeps of the pool is what forced eight complete
    per-position score sets to exist at once, which was the largest thing a
    canonical reading held; here a position's eight scores live together for
    one batch, are compared, and are dropped.
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

    def score(self, inputs: ScoringInputs, *, dependency: bool) -> _BatchScores:
        """Score one batch under every treatment the configuration asks for."""

        batch = self._batch(inputs)
        active = active_batch(self._runner.action_logits(batch), batch)
        positions = score_positions(active)
        if not positions:
            raise CheckpointEvaluationError(
                "the configured view selected no positions to score"
            )
        scores = _BatchScores(
            positions=positions,
            action_sets=score_action_sets(active, action_sets(inputs)),
            corrupted={},
            conditioned={},
            trajectory={},
        )
        if not dependency:
            return scores

        for conditioning in (
            Conditioning(name="shuffled", kind=ConditioningKind.SHUFFLED),
            Conditioning(
                name="constant",
                kind=ConditioningKind.CONSTANT,
                rating=self._config.constant_rating,
            ),
            Conditioning(name="absent", kind=ConditioningKind.ABSENT),
        ):
            scores.corrupted[conditioning.name] = (
                conditioning,
                self._conditioned(batch, conditioning),
            )

        values = self._config.conditioning_values()
        self._trajectory(batch, inputs, scores, values[0], values[-1])
        for value in values:
            if value in scores.conditioned:
                continue
            scores.conditioned[value] = self._conditioned(
                batch, _constant_conditioning(value)
            )
        return scores

    def _trajectory(
        self,
        batch: MoveModelBatch,
        inputs: ScoringInputs,
        scores: _BatchScores,
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

        true_batch = self._condition(batch, _TRUE_CONDITIONING)
        active = active_batch(self._runner.action_logits(true_batch), true_batch)
        true = legal_policy_log_probabilities(active)
        policies = []
        for rating in (anchor_low, anchor_high):
            conditioned = self._condition(batch, _constant_conditioning(rating))
            rescored = active.rescored(
                self._runner.action_logits(conditioned),
                conditioned,
            )
            policies.append(legal_policy_log_probabilities(rescored))
            scores.conditioned[rating] = score_positions(rescored)
        low, high = policies
        for offset, key in enumerate(
            zip(active.game_ids, active.ply_indices, strict=True)
        ):
            scores.trajectory[key] = _trajectory_signal(
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
        loader = SequenceDataLoader(inputs.dataset, inputs.loader_config)
        batches = list(loader)
        if len(batches) != 1:
            raise CheckpointEvaluationError(
                f"a planned batch encoded to {len(batches)} loader batch(es); "
                "the batch plan and the loader disagree about batching"
            )
        return MoveModelBatch.from_sequence_batch(
            batches[0],
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
        if self._shuffled is None:
            raise CheckpointEvaluationError(
                "the shuffled conditioning treatment needs the view's dealt ratings"
            )
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


class _DependencyColumnBuilder:
    """Fills the dependency family's columns one batch at a time.

    The one thing a checkpoint reading cannot finish as it goes. Every other
    reduction here is a running total; the within-game split takes a median of
    each rating slice, so the values have to survive the pass that produced
    them. They survive as typed columns rather than as scored objects, which is
    the difference between a gigabyte and a hundred of them.
    """

    def __init__(self) -> None:
        # Unsigned: a game id is a 64-bit hash, and a signed column wraps
        # every id past the signed maximum onto one that matches no game.
        self._game_ids: array[int] = array("Q")
        self._ply_indices: array[int] = array("i")
        self._colors = _Interner()
        self._bands = _Interner()
        self._rated = bytearray()
        self._true: array[float] = array("d")
        self._corrupted: dict[str, tuple[Conditioning, array[float]]] = {}
        self._conditioned: dict[int, array[float]] = {}
        self._has_signal = bytearray()
        self._strength: array[float] = array("d")
        self._alignment: array[float] = array("d")
        self._divergence: array[float] = array("d")
        self._agreement = bytearray()

    def add(self, inputs: ScoringInputs, scores: _BatchScores) -> None:
        """Append one batch's scored positions to every column."""

        for position in scores.positions:
            key = (position.game_id, position.ply_index)
            context = inputs.contexts[key]
            self._game_ids.append(position.game_id)
            self._ply_indices.append(position.ply_index)
            self._colors.append(context.color)
            self._bands.append(context.rating_band)
            self._rated.append(context.rating is not None)
            self._true.append(position.move_nll)
            signal = scores.trajectory.get(key)
            self._has_signal.append(signal is not None)
            self._strength.append(0.0 if signal is None else signal.strength_signal)
            self._alignment.append(0.0 if signal is None else signal.alignment)
            self._divergence.append(0.0 if signal is None else signal.anchor_divergence)
            self._agreement.append(False if signal is None else signal.anchor_agreement)

        for name, (conditioning, positions) in scores.corrupted.items():
            self._extend(
                self._corrupted.setdefault(name, (conditioning, array("d")))[1],
                positions,
                scores.positions,
                f"conditioning pass {name!r}",
            )
        for rating, positions in scores.conditioned.items():
            self._extend(
                self._conditioned.setdefault(rating, array("d")),
                positions,
                scores.positions,
                f"conditioning rating {rating}",
            )

    def build(self) -> DependencyColumns:
        """Return the filled columns every dependency reduction reads."""

        return DependencyColumns(
            game_ids=self._game_ids,
            ply_indices=self._ply_indices,
            color_names=self._colors.names(),
            colors=self._colors.offsets,
            band_names=self._bands.names(),
            bands=self._bands.offsets,
            rated=[bool(value) for value in self._rated],
            true_move_nll=self._true,
            corrupted={
                name: (conditioning, values)
                for name, (conditioning, values) in self._corrupted.items()
            },
            conditioned=dict(self._conditioned),
            has_signal=[bool(value) for value in self._has_signal],
            strength_signal=self._strength,
            alignment=self._alignment,
            anchor_divergence=self._divergence,
            anchor_agreement=[bool(value) for value in self._agreement],
        )

    def _extend(
        self,
        column: array[float],
        positions: Sequence[PositionPolicy],
        reference: Sequence[PositionPolicy],
        what: str,
    ) -> None:
        # Positional rather than keyed: every treatment scores one batch
        # through the same enabled-row mask, so row *i* is the same decision in
        # all of them. The width check is what says that still holds.
        if len(positions) != len(reference):
            raise CheckpointEvaluationError(
                f"{what} scored {len(positions)} of {len(reference)} position(s) "
                "in one batch"
            )
        column.extend(position.move_nll for position in positions)


class _Interner:
    """A column of repeated names, kept as offsets into the names themselves."""

    def __init__(self) -> None:
        self.offsets: list[int] = []
        self._names: dict[str, int] = {}

    def append(self, name: str | None) -> None:
        """Append one value, recording an absent one as a negative offset."""

        if name is None:
            self.offsets.append(-1)
            return
        offset = self._names.get(name)
        if offset is None:
            offset = len(self._names)
            self._names[name] = offset
        self.offsets.append(offset)

    def names(self) -> tuple[str, ...]:
        """Return the distinct names, in the order they were first seen."""

        return tuple(self._names)


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
    measured = _score(config, reading, runner, frequency)
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
    columns = _DependencyColumnBuilder() if dependency else None
    totals: list[GameTotals] = []
    retained: list[PositionPolicy] | None = [] if config.detail.per_position else None

    for game_ids in reading.batches:
        inputs = reading.inputs(game_ids)
        scores = session.score(inputs, dependency=dependency)
        accumulate_positions(
            aggregator,
            scores.positions,
            inputs,
            opening_frequency=frequency,
        )
        adjudication.add(scores.action_sets, inputs)
        totals.extend(
            per_game_totals(scores.positions, inputs, opening_frequency=frequency)
        )
        if columns is not None:
            columns.add(inputs, scores)
        if retained is not None:
            retained.extend(scores.positions)

    report = adjudication.report()
    result = None
    if columns is not None:
        try:
            result = reduce_dependency_columns(
                config=config.dependency,
                columns=columns.build(),
                maturity=MaturityContext(
                    step=runner.global_step,
                    processed_positions=runner.processed_positions,
                ),
            )
        except DependencyError as error:
            raise CheckpointEvaluationError(str(error)) from error
    return _Measured(
        slices=aggregator.compute(),
        adjudication=report,
        dependency=result,
        game_totals=merge_game_totals(
            tuple(totals),
            () if report is None else report.per_game_totals,
            () if result is None else result.per_game_totals,
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
        split=_pool_split(pool),
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
