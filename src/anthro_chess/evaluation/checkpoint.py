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
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import Field, StrictBool
from torch import Tensor

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import (
    DataLoadingError,
    GameEncodingInput,
    PlyEncoding,
    SequenceDataLoader,
    SequenceDataset,
    SequenceExample,
    SequenceLoaderConfig,
    encode_game,
)
from anthro_chess.data.artifacts import read_normalized_rows
from anthro_chess.data.schema import (
    SCHEMA_VERSION,
    SPLIT_NAMES,
    NormalizedColumn,
    SplitName,
)
from anthro_chess.evaluation.aggregation import (
    PHASE_DIMENSION,
    RATING_DIMENSION,
    RULE_CASE_DIMENSION,
    SliceAggregator,
    SliceTable,
)
from anthro_chess.evaluation.dependency import (
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
from anthro_chess.evaluation.policy import (
    POLICY_SCORING_VERSION,
    PositionPolicy,
    legal_policy_log_probabilities,
    policy_divergence,
    score_positions,
)
from anthro_chess.evaluation.pool import (
    EvaluationPoolError,
    FrozenPool,
    load_pool,
)
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
    DatasetReference,
    DetailReference,
    DetailStore,
    Measurement,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    ResultsStoreError,
    build_result,
    configuration_reference,
    dataset_reference,
    measurement,
    projection_content_digest,
)
from anthro_chess.evaluation.results.metrics import (
    DEPENDENCY_RATING_ABSENT_DEGRADATION,
    DEPENDENCY_RATING_ANCHOR_POLICY_DIVERGENCE,
    DEPENDENCY_RATING_ANCHOR_TOP1_AGREEMENT,
    DEPENDENCY_RATING_CONSTANT_DEGRADATION,
    DEPENDENCY_RATING_CROSS_CONDITIONING_MATCH_RATE,
    DEPENDENCY_RATING_SHUFFLED_DEGRADATION,
    DEPENDENCY_RATING_WITHIN_GAME_RESPONSE,
    HELD_OUT_LEGAL_MOVE_LOSS,
    HELD_OUT_MOVE_LOSS,
    HELD_OUT_MOVE_LOSS_BY_PHASE,
    HELD_OUT_MOVE_LOSS_BY_RATING_BAND,
    HELD_OUT_TOP_K_ACCURACY,
    HELD_OUT_UNIFORM_OVER_LEGAL_MOVE_LOSS,
    LEGALITY_LEGAL_MARGIN,
    LEGALITY_LEGAL_MASS,
    LEGALITY_LIFT,
    LEGALITY_MASK_PENALTY,
    LEGALITY_MASK_PENALTY_BY_PHASE,
    LEGALITY_MASK_PENALTY_BY_RULE_CASE,
    LEGALITY_TOP1_ILLEGAL_RATE,
    LEGALITY_TOP_ILLEGAL_FRACTION,
    MOVE_PREDICTION_PROJECTION,
    MetricDefinition,
)
from anthro_chess.evaluation.slices import (
    DEFAULT_RATING_BANDS,
    SLICE_SCHEME_VERSION,
    PositionCharacteristic,
    PositionSlices,
    ply_characteristics,
    position_slices,
)
from anthro_chess.evaluation.views import ViewConfig, ViewSelection, apply_view
from anthro_chess.inference import CheckpointModelRunner, ModelRunnerConfig
from anthro_chess.inference.runner import ModelRunnerError
from anthro_chess.models import MoveModelBatch, OptionalTensor

CHECKPOINT_EVALUATION_VERSION = 1

HELD_OUT_KIND = "held-out-prediction"
DEPENDENCY_KIND = "rating-dependency"
HELD_OUT_BENCHMARK = BenchmarkReference(name="held-out-prediction", version=1)
DEPENDENCY_BENCHMARK = BenchmarkReference(name="rating-dependency", version=1)

logger = logging.getLogger(__name__)

_TRUE_CONDITIONING = Conditioning(name="true", kind=ConditioningKind.TRUE)


class CheckpointEvaluationError(ValueError):
    """Raised when a checkpoint cannot be evaluated over a frozen pool."""


class EvaluationLoaderConfig(ConfigModel):
    """Batching for evaluation, which never shuffles and never drops a game."""

    batch_size: int = Field(default=8, ge=1)
    length_bucket_width: int | None = Field(default=32, ge=1)


class LeakageConfig(ConfigModel):
    """Where to find the corpus a checkpoint trained on, when it moved."""

    training_normalized: Path | None = None


class DetailConfig(ConfigModel):
    """What the machine-local detail tier keeps beside a committed summary."""

    #: Per-position records are what decision decomposition and rollout
    #: comparisons need, and they are large. They stay opt-in rather than
    #: growing every routine evaluation by an order of magnitude.
    per_position: StrictBool = False


class CheckpointEvaluationConfig(ConfigModel):
    """Code-owned schema for ``anthro eval run``."""

    pool: Path
    view: ViewConfig = ViewConfig(name="canonical")
    model: ModelRunnerConfig = ModelRunnerConfig()
    loader: EvaluationLoaderConfig = EvaluationLoaderConfig()
    checkpoint_label: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    dependency: DependencyTestConfig = DependencyTestConfig()
    leakage: LeakageConfig = LeakageConfig()
    detail: DetailConfig = DetailConfig()


@dataclass(frozen=True)
class CheckpointEvaluationResult:
    """Everything one evaluation measured, plus where it was written."""

    checkpoint: CheckpointReference
    dataset: DatasetReference
    view: ViewSelection
    leakage: LeakageCheck
    slices: SliceTable
    dependency: DependencyTestResult | None
    envelopes: tuple[ResultEnvelope, ...]
    recorded_paths: tuple[Path, ...]
    detail_paths: tuple[Path, ...]

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
            "dependency": (
                self.dependency.as_record() if self.dependency is not None else None
            ),
            "recorded": [str(path) for path in self.recorded_paths],
        }


@dataclass(frozen=True)
class _EvaluationInputs:
    """The frozen games one evaluation scores, and everything derived once."""

    pool: FrozenPool
    selection: ViewSelection
    rows: tuple[dict[str, Any], ...]
    dataset: SequenceDataset
    loader_config: SequenceLoaderConfig
    plies: Mapping[PositionKey, PlyEncoding]
    slices: Mapping[PositionKey, PositionSlices]
    characteristics: Mapping[PositionKey, frozenset[PositionCharacteristic]]
    contexts: Mapping[PositionKey, PositionContext]


class _ScoringSession:
    """Repeated deterministic passes over one view under varied conditioning."""

    def __init__(
        self,
        runner: CheckpointModelRunner,
        inputs: _EvaluationInputs,
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
            positions.extend(
                score_positions(self._runner.action_logits(conditioned), conditioned)
            )
        if not positions:
            raise CheckpointEvaluationError(
                "the configured view selected no positions to score"
            )
        return tuple(positions)

    def trajectory(
        self,
        *,
        anchor_low: int,
        anchor_high: int,
    ) -> dict[PositionKey, TrajectorySignal]:
        """Compare each position's policy at two anchor conditioning ratings.

        All three policies a signal needs are computed for one batch at a
        time. Retaining the true-conditioning policy from the primary pass
        would save a forward pass and cost a distribution per position held
        for the whole run, which is gigabytes over a full pool.
        """

        signals: dict[PositionKey, TrajectorySignal] = {}
        treatments = (
            _TRUE_CONDITIONING,
            Conditioning(
                name=f"constant-{anchor_low}",
                kind=ConditioningKind.CONSTANT,
                rating=anchor_low,
            ),
            Conditioning(
                name=f"constant-{anchor_high}",
                kind=ConditioningKind.CONSTANT,
                rating=anchor_high,
            ),
        )
        for batch in self._batches():
            keys = _batch_keys(batch)
            policies = []
            for conditioning in treatments:
                conditioned = self._condition(batch, conditioning)
                policies.append(
                    legal_policy_log_probabilities(
                        self._runner.action_logits(conditioned),
                        conditioned,
                    )
                )
            true, low, high = policies
            for offset, key in enumerate(keys):
                signals[key] = _trajectory_signal(
                    legal_actions=self._inputs.plies[key].legal_action_ids,
                    target_action_id=self._inputs.plies[key].target_action_id,
                    true=true[offset],
                    low=low[offset],
                    high=high[offset],
                )
        return signals

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
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> CheckpointEvaluationResult:
    """Evaluate one checkpoint over a frozen pool and record the result.

    Passing no ``store`` computes everything and records nothing, which is
    what an exploratory reading wants: the committed tier should hold results
    somebody meant to keep.
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

    session = _ScoringSession(
        runner,
        inputs,
        shuffle_seed=config.dependency.shuffle_seed,
    )
    logger.info(
        "Scoring %s game(s) from pool view %r",
        inputs.selection.selected_games,
        inputs.selection.name,
    )
    positions = session.score(_TRUE_CONDITIONING)
    slices = _aggregate(positions, inputs)
    dependency = (
        _run_dependency_tests(config, session, inputs, positions, runner)
        if config.dependency.enabled
        else None
    )

    component = projection_content_digest(inputs.rows, MOVE_PREDICTION_PROJECTION)
    checkpoint = _checkpoint_reference(config, runner)
    data = _dataset_reference(inputs, component)
    recorded_at = datetime.now(tz=UTC)

    envelopes: list[ResultEnvelope] = []
    detail_paths: list[Path] = []
    configuration = configuration_reference(
        resolved_config.as_record(),
        source=resolved_config.provenance.source,
        overrides=resolved_config.provenance.overrides,
    )
    result = CheckpointEvaluationResult(
        checkpoint=checkpoint,
        dataset=data,
        view=inputs.selection,
        leakage=leakage,
        slices=slices,
        dependency=dependency,
        envelopes=(),
        recorded_paths=(),
        detail_paths=(),
    )

    try:
        held_out_detail = _write_detail(
            detail,
            kind=HELD_OUT_KIND,
            checkpoint=checkpoint,
            recorded_at=recorded_at,
            payload={
                **result.as_record(),
                "positions": (
                    [position.as_record() for position in positions]
                    if config.detail.per_position
                    else None
                ),
            },
            description="Slice tables and view provenance for one evaluation.",
            paths=detail_paths,
        )
        envelopes.append(
            build_result(
                kind=HELD_OUT_KIND,
                benchmark=HELD_OUT_BENCHMARK,
                checkpoint=checkpoint,
                configuration=configuration,
                data=data,
                measurements=_held_out_measurements(slices, component),
                detail=held_out_detail,
                recorded_at=recorded_at,
            )
        )
        if dependency is not None:
            dependency_detail = _write_detail(
                detail,
                kind=DEPENDENCY_KIND,
                checkpoint=checkpoint,
                recorded_at=recorded_at,
                payload=dependency.as_record(),
                description="Cross-conditioning and within-game dependency tables.",
                paths=detail_paths,
            )
            envelopes.append(
                build_result(
                    kind=DEPENDENCY_KIND,
                    benchmark=DEPENDENCY_BENCHMARK,
                    checkpoint=checkpoint,
                    configuration=configuration,
                    data=data,
                    measurements=_dependency_measurements(dependency, component),
                    detail=dependency_detail,
                    recorded_at=recorded_at,
                )
            )
    except (ResultRecordError, ResultsStoreError) as error:
        raise CheckpointEvaluationError(str(error)) from error

    recorded_paths: list[Path] = []
    if store is not None:
        try:
            recorded_paths = [store.append(envelope) for envelope in envelopes]
        except (ResultRecordError, ResultsStoreError) as error:
            raise CheckpointEvaluationError(str(error)) from error

    return replace(
        result,
        envelopes=tuple(envelopes),
        recorded_paths=tuple(recorded_paths),
        detail_paths=tuple(detail_paths),
    )


def _load_inputs(config: CheckpointEvaluationConfig) -> _EvaluationInputs:
    pool = load_pool(config.pool)
    selection = apply_view(pool.games, config.view)
    if not selection.game_ids:
        raise CheckpointEvaluationError(
            f"view {config.view.name!r} selected no games from the pool"
        )

    wanted = set(selection.game_ids)
    rows = [
        _truncate(row, selection.prefix_plies)
        for row in read_normalized_rows(pool.games_path)
        if int(row[NormalizedColumn.GAME_ID]) in wanted
    ]
    if len(rows) != len(wanted):
        raise CheckpointEvaluationError(
            "the evaluation pool does not contain every selected game"
        )
    rows.sort(key=lambda row: int(row[NormalizedColumn.GAME_ID]))

    examples: list[SequenceExample] = []
    plies: dict[PositionKey, PlyEncoding] = {}
    slices: dict[PositionKey, PositionSlices] = {}
    characteristics: dict[PositionKey, frozenset[PositionCharacteristic]] = {}
    contexts: dict[PositionKey, PositionContext] = {}
    for row in rows:
        encoded = encode_game(_encoding_input(row))
        examples.append(
            SequenceExample(
                shard_index=0,
                game_id=int(row[NormalizedColumn.GAME_ID]),
                start_ply=encoded[0].ply_index,
                plies=encoded,
            )
        )
        for ply in encoded:
            key = (ply.game_id, ply.ply_index)
            plies[key] = ply
            derived = position_slices(ply, DEFAULT_RATING_BANDS)
            slices[key] = derived
            characteristics[key] = ply_characteristics(ply)
            contexts[key] = PositionContext(
                game_id=ply.game_id,
                ply_index=ply.ply_index,
                color=str(derived.color),
                rating=ply.target_rating,
                rating_band=derived.rating_band,
            )

    split = _pool_split(pool)
    dataset = SequenceDataset(
        examples,
        identity_sha256=_view_identity(selection, rows),
        split=split,
        chunk_length=None,
    )
    loader_config = SequenceLoaderConfig(
        split=split,
        batch_size=config.loader.batch_size,
        length_bucket_width=config.loader.length_bucket_width,
        chunk_length=None,
        shuffle=False,
        drop_last=False,
    )
    return _EvaluationInputs(
        pool=pool,
        selection=selection,
        rows=tuple(rows),
        dataset=dataset,
        loader_config=loader_config,
        plies=plies,
        slices=slices,
        characteristics=characteristics,
        contexts=contexts,
    )


def _aggregate(
    positions: Sequence[PositionPolicy],
    inputs: _EvaluationInputs,
) -> SliceTable:
    aggregator = SliceAggregator()
    for position in positions:
        key = (position.game_id, position.ply_index)
        aggregator.add(position, inputs.slices[key], inputs.characteristics[key])
    return aggregator.compute()


def _run_dependency_tests(
    config: CheckpointEvaluationConfig,
    session: _ScoringSession,
    inputs: _EvaluationInputs,
    positions: Sequence[PositionPolicy],
    runner: CheckpointModelRunner,
) -> DependencyTestResult:
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

    conditioned: dict[int, Sequence[PositionPolicy]] = {}
    for value in values:
        logger.info("Scoring under a fixed conditioning rating of %s", value)
        conditioned[value] = session.score(
            Conditioning(
                name=f"constant-{value}",
                kind=ConditioningKind.CONSTANT,
                rating=value,
            )
        )

    logger.info("Comparing anchor policies at ratings %s and %s", values[0], values[-1])
    trajectory = session.trajectory(anchor_low=values[0], anchor_high=values[-1])
    try:
        return build_dependency_result(
            config=settings,
            contexts=inputs.contexts,
            true_positions=positions,
            corrupted_positions=corrupted,
            conditioned_positions=conditioned,
            trajectory=trajectory,
            maturity=_maturity(runner),
        )
    except DependencyError as error:
        raise CheckpointEvaluationError(str(error)) from error


def _held_out_measurements(
    slices: SliceTable,
    component: DataComponent,
) -> tuple[Measurement, ...]:
    overall = slices.overall
    values: list[Measurement] = [
        measurement(
            HELD_OUT_MOVE_LOSS.identifier,
            overall.move_loss,
            data=component,
            sample_size=overall.position_count,
        ),
        measurement(
            HELD_OUT_LEGAL_MOVE_LOSS.identifier,
            overall.legal_move_loss,
            data=component,
            sample_size=overall.position_count,
        ),
        measurement(
            HELD_OUT_UNIFORM_OVER_LEGAL_MOVE_LOSS.identifier,
            overall.uniform_over_legal_move_loss,
            data=component,
            sample_size=overall.position_count,
        ),
        measurement(
            LEGALITY_MASK_PENALTY.identifier,
            overall.mask_penalty,
            data=component,
            sample_size=overall.position_count,
        ),
        measurement(
            LEGALITY_LEGAL_MASS.identifier,
            overall.legal_mass,
            data=component,
            sample_size=overall.position_count,
        ),
        measurement(
            LEGALITY_TOP1_ILLEGAL_RATE.identifier,
            overall.top1_illegal_rate,
            data=component,
            sample_size=overall.position_count,
        ),
        measurement(
            LEGALITY_TOP_ILLEGAL_FRACTION.identifier,
            overall.top_illegal_fraction,
            data=component,
            sample_size=overall.position_count,
        ),
        measurement(
            LEGALITY_LEGAL_MARGIN.identifier,
            overall.legal_margin,
            data=component,
            sample_size=overall.position_count,
        ),
        measurement(
            LEGALITY_LIFT.identifier,
            overall.legality_lift,
            data=component,
            sample_size=overall.position_count,
        ),
    ]
    for cutoff, definition in HELD_OUT_TOP_K_ACCURACY.items():
        values.append(
            measurement(
                definition.identifier,
                overall.accuracy(cutoff),
                data=component,
                sample_size=overall.position_count,
            )
        )

    sliced: tuple[tuple[str, Mapping[str, MetricDefinition], str], ...] = (
        (PHASE_DIMENSION, HELD_OUT_MOVE_LOSS_BY_PHASE, "move_loss"),
        (PHASE_DIMENSION, LEGALITY_MASK_PENALTY_BY_PHASE, "mask_penalty"),
        (RATING_DIMENSION, HELD_OUT_MOVE_LOSS_BY_RATING_BAND, "move_loss"),
        (RULE_CASE_DIMENSION, LEGALITY_MASK_PENALTY_BY_RULE_CASE, "mask_penalty"),
    )
    for dimension, definitions, attribute in sliced:
        for name, definition in definitions.items():
            summary = slices.slice_summary(dimension, name)
            if summary is None:
                continue
            values.append(
                measurement(
                    definition.identifier,
                    float(getattr(summary, attribute)),
                    data=component,
                    sample_size=summary.position_count,
                )
            )
    return tuple(values)


def _dependency_measurements(
    dependency: DependencyTestResult,
    component: DataComponent,
) -> tuple[Measurement, ...]:
    values: list[Measurement] = []
    degradations = (
        (ConditioningKind.SHUFFLED, DEPENDENCY_RATING_SHUFFLED_DEGRADATION),
        (ConditioningKind.CONSTANT, DEPENDENCY_RATING_CONSTANT_DEGRADATION),
        (ConditioningKind.ABSENT, DEPENDENCY_RATING_ABSENT_DEGRADATION),
    )
    for kind, definition in degradations:
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


def _write_detail(
    detail: DetailStore | None,
    *,
    kind: str,
    checkpoint: CheckpointReference,
    recorded_at: datetime,
    payload: Mapping[str, object],
    description: str,
    paths: list[Path],
) -> DetailReference | None:
    if detail is None:
        return None
    stamp = recorded_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    relative = Path(kind) / checkpoint.label / f"{stamp}.json"
    reference = detail.write(relative, dict(payload), description=description)
    paths.append(detail.root / relative)
    return reference


def _checkpoint_reference(
    config: CheckpointEvaluationConfig,
    runner: CheckpointModelRunner,
) -> CheckpointReference:
    run_id = runner.selection.run_path.name
    label = config.checkpoint_label or _default_label(run_id, runner.global_step)
    return CheckpointReference(
        label=label,
        step=runner.global_step,
        run_id=run_id,
        parameter_sha256=runner.parameter_sha256(),
    )


def _default_label(run_id: str, global_step: int) -> str:
    slug = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in run_id.lower()
    ).strip("-")
    prefix = slug or "run"
    if not prefix[0].isalnum():
        prefix = f"run-{prefix}"
    return f"{prefix}-step-{global_step:08d}"


def _dataset_reference(
    inputs: _EvaluationInputs,
    component: DataComponent,
) -> DatasetReference:
    pool = inputs.pool.manifest.get("pool")
    if not isinstance(pool, Mapping):
        raise CheckpointEvaluationError("evaluation pool manifest has no pool identity")
    record = inputs.selection.as_record()
    return dataset_reference(
        pool_id=str(pool["id"]),
        pool_version=int(pool["version"]),
        view=inputs.selection.name,
        selected_games=inputs.selection.selected_games,
        game_ids_sha256=str(record["game_ids_sha256"]),
        components=[component],
    )


def _maturity(runner: CheckpointModelRunner) -> MaturityContext:
    optimization = runner.run_record.get("optimization")
    processed = None
    if isinstance(optimization, Mapping):
        value = optimization.get("processed_positions")
        processed = int(value) if isinstance(value, int) else None
    return MaturityContext(step=runner.global_step, processed_positions=processed)


def _pool_split(pool: FrozenPool) -> SplitName:
    record = pool.manifest.get("pool")
    split = record.get("split") if isinstance(record, Mapping) else None
    if split not in SPLIT_NAMES:
        raise CheckpointEvaluationError(
            f"evaluation pool manifest names an unknown split: {split!r}"
        )
    return cast(SplitName, split)


def _view_identity(
    selection: ViewSelection,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    digest = sha256()
    digest.update(str(selection.as_record()).encode())
    for row in rows:
        digest.update(f"\n{row[NormalizedColumn.GAME_ID]}".encode())
    return digest.hexdigest()


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
        NormalizedColumn.CLOCK_REMAINING_MS,
        NormalizedColumn.CLOCK_STATUS,
        NormalizedColumn.CLOCK_PRECISION_MS,
    ):
        values = updated.get(column.value)
        if values is not None:
            updated[column.value] = list(values)[:prefix_plies]
    updated[NormalizedColumn.PLY_COUNT.value] = len(
        updated[NormalizedColumn.ACTION_IDS.value]
    )
    return updated


def _encoding_input(row: Mapping[str, Any]) -> GameEncodingInput:
    if row[NormalizedColumn.SCHEMA_VERSION] != SCHEMA_VERSION:
        raise CheckpointEvaluationError(
            f"evaluation pool uses normalized schema version "
            f"{row[NormalizedColumn.SCHEMA_VERSION]}; expected {SCHEMA_VERSION}"
        )
    return GameEncodingInput(
        game_id=int(row[NormalizedColumn.GAME_ID]),
        ruleset=str(row[NormalizedColumn.RULESET]),
        initial_position=str(row[NormalizedColumn.INITIAL_POSITION]),
        action_ids=tuple(row[NormalizedColumn.ACTION_IDS]),
        white_normalized_rating=row[NormalizedColumn.WHITE_NORMALIZED_RATING],
        black_normalized_rating=row[NormalizedColumn.BLACK_NORMALIZED_RATING],
        time_initial_ms=row[NormalizedColumn.TIME_INITIAL_MS],
        time_increment_ms=row[NormalizedColumn.TIME_INCREMENT_MS],
        clock_remaining_ms=tuple(row[NormalizedColumn.CLOCK_REMAINING_MS]),
    )


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


def _batch_keys(batch: MoveModelBatch) -> tuple[PositionKey, ...]:
    active = (
        torch.nonzero(batch.action_loss_mask, as_tuple=False).detach().cpu().tolist()
    )
    return tuple(
        (
            int(batch.game_ids[batch_index, sequence_index].item()),
            int(batch.ply_indices[batch_index, sequence_index].item()),
        )
        for batch_index, sequence_index in active
    )


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
