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
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import Field, StrictBool
from torch import Tensor

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import (
    DataLoadingError,
    SequenceDataLoader,
)
from anthro_chess.data.artifacts import read_normalized_rows
from anthro_chess.data.schema import (
    SPLIT_NAMES,
    NormalizedColumn,
    SplitName,
)
from anthro_chess.evaluation.adjudication import (
    AdjudicationReport,
    action_sets,
    build_adjudication_report,
    merge_game_totals,
)
from anthro_chess.evaluation.aggregation import SliceTable
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
from anthro_chess.evaluation.noise import NoiseConfig, characterize_sampling_noise
from anthro_chess.evaluation.policy import (
    POLICY_SCORING_VERSION,
    ActionSetPolicy,
    PositionPolicy,
    legal_policy_log_probabilities,
    policy_divergence,
    score_action_sets,
    score_positions,
)
from anthro_chess.evaluation.pool import (
    EvaluationPoolError,
    FrozenPool,
    load_pool,
)
from anthro_chess.evaluation.results import (
    PAIRED_CONTRIBUTIONS_KEY,
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
    default_checkpoint_label,
    measurement,
    paired_contributions,
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
    MOVE_PREDICTION_PROJECTION,
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
    per_game_totals,
    rows_identity_sha256,
    slice_measurements,
)
from anthro_chess.evaluation.slices import SLICE_SCHEME_VERSION
from anthro_chess.evaluation.views import ViewConfig, ViewSelection, apply_view
from anthro_chess.inference import CheckpointModelRunner, ModelRunnerConfig
from anthro_chess.inference.runner import ModelRunnerError
from anthro_chess.models import MoveModelBatch, OptionalTensor

CHECKPOINT_EVALUATION_VERSION = 1

HELD_OUT_KIND = "held-out-prediction"
DEPENDENCY_KIND = "rating-dependency"
ADJUDICATION_KIND = "adjudicated-decisions"
HELD_OUT_BENCHMARK = BenchmarkReference(name="held-out-prediction", version=1)
DEPENDENCY_BENCHMARK = BenchmarkReference(name="rating-dependency", version=1)
ADJUDICATION_BENCHMARK = BenchmarkReference(name="adjudicated-decisions", version=1)

logger = logging.getLogger(__name__)

_TRUE_CONDITIONING = Conditioning(name="true", kind=ConditioningKind.TRUE)


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
    noise: NoiseConfig = NoiseConfig()


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
    noise: NoiseCharacterization | None
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
            "adjudication": (
                self.adjudication.as_record() if self.adjudication is not None else None
            ),
            "dependency": (
                self.dependency.as_record() if self.dependency is not None else None
            ),
            "noise": self.noise.as_record() if self.noise is not None else None,
            "recorded": [str(path) for path in self.recorded_paths],
        }


@dataclass(frozen=True)
class _EvaluationInputs:
    """The frozen games one evaluation scores, and everything derived once."""

    pool: FrozenPool
    selection: ViewSelection
    scoring: ScoringInputs


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
            positions.extend(
                score_positions(self._runner.action_logits(conditioned), conditioned)
            )
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
            logits = self._runner.action_logits(batch)
            positions.extend(score_positions(logits, batch))
            adjudicated.extend(score_action_sets(logits, batch, subsets))
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
        inputs.scoring,
        shuffle_seed=config.dependency.shuffle_seed,
    )
    logger.info(
        "Scoring %s game(s) from pool view %r",
        inputs.selection.selected_games,
        inputs.selection.name,
    )
    positions, action_set_scores = session.score_primary()
    slices = aggregate_positions(positions, inputs.scoring)
    adjudication = build_adjudication_report(action_set_scores, inputs.scoring)
    dependency = (
        _run_dependency_tests(config, session, inputs, positions, runner)
        if config.dependency.enabled
        else None
    )

    component = projection_content_digest(
        inputs.scoring.rows,
        MOVE_PREDICTION_PROJECTION,
    )
    checkpoint = _checkpoint_reference(config, runner)
    data = _dataset_reference(inputs, component)
    recorded_at = datetime.now(tz=UTC)
    noise = _characterize_noise(
        config,
        inputs,
        positions,
        adjudication,
        component,
        recorded_at=recorded_at,
    )

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
        adjudication=adjudication,
        dependency=dependency,
        noise=noise,
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
                measurements=slice_measurements(slices, component),
                detail=held_out_detail,
                recorded_at=recorded_at,
            )
        )
        if adjudication is not None:
            adjudication_detail = _write_detail(
                detail,
                kind=ADJUDICATION_KIND,
                checkpoint=checkpoint,
                recorded_at=recorded_at,
                payload=adjudication.as_record(),
                description=(
                    "Per-predicate human and model rates with rating-band "
                    "drill-down and opportunity counts."
                ),
                paths=detail_paths,
            )
            envelopes.append(
                build_result(
                    kind=ADJUDICATION_KIND,
                    benchmark=ADJUDICATION_BENCHMARK,
                    checkpoint=checkpoint,
                    configuration=configuration,
                    data=data,
                    measurements=adjudication.measurements(component),
                    detail=adjudication_detail,
                    recorded_at=recorded_at,
                )
            )
        if dependency is not None:
            dependency_payload = dependency.as_record()
            contributions = _dependency_contributions(dependency, config.noise)
            if contributions is not None:
                dependency_payload[PAIRED_CONTRIBUTIONS_KEY] = contributions
            dependency_detail = _write_detail(
                detail,
                kind=DEPENDENCY_KIND,
                checkpoint=checkpoint,
                recorded_at=recorded_at,
                payload=dependency_payload,
                description=(
                    "Cross-conditioning and within-game dependency tables, and "
                    "the aligned per-game values a later paired floor needs."
                ),
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
            if noise is not None:
                recorded_paths.append(store.append_characterization(noise))
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
    scoring = build_scoring_inputs(
        rows,
        split=_pool_split(pool),
        batch_size=config.loader.batch_size,
        length_bucket_width=config.loader.length_bucket_width,
        identity_sha256=rows_identity_sha256(rows, context=selection.as_record()),
    )
    return _EvaluationInputs(pool=pool, selection=selection, scoring=scoring)


def _characterize_noise(
    config: CheckpointEvaluationConfig,
    inputs: _EvaluationInputs,
    positions: Sequence[PositionPolicy],
    adjudication: AdjudicationReport | None,
    component: DataComponent,
    *,
    recorded_at: datetime,
) -> NoiseCharacterization | None:
    """Estimate this reading's own data-sampling noise from the same pass.

    The floor costs one resampling of numbers already computed, so it is on by
    default. A reading with no floor beside it can only report that a number
    moved.
    """

    if not config.noise.enabled:
        return None
    try:
        adjudication_totals = (
            () if adjudication is None else adjudication.per_game_totals()
        )
        return characterize_sampling_noise(
            merge_game_totals(
                per_game_totals(positions, inputs.scoring),
                adjudication_totals,
            ),
            component=component,
            config=config.noise,
            source=(
                f"bootstrap over {inputs.selection.selected_games} game(s) of "
                f"pool view {inputs.selection.name!r}"
            ),
            recorded_at=recorded_at,
        )
    except NoiseCharacterizationError as error:
        # A view too small to resample is a fact about the view, not a failed
        # evaluation. The reading still stands; it simply has no floor.
        logger.warning("Skipping data-sampling noise characterization: %s", error)
        return None


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
            contexts=inputs.scoring.contexts,
            true_positions=positions,
            corrupted_positions=corrupted,
            conditioned_positions=conditioned,
            trajectory=trajectory,
            maturity=_maturity(runner),
        )
    except DependencyError as error:
        raise CheckpointEvaluationError(str(error)) from error


#: Which metric each conditioning treatment's degradation is reported as. One
#: table rather than two, because the reading and the floor beside it have to
#: name the same quantity, and a treatment added to only one of them is the
#: missing-floor gap this family already had once.
_DEGRADATION_METRICS = {
    ConditioningKind.SHUFFLED: DEPENDENCY_RATING_SHUFFLED_DEGRADATION,
    ConditioningKind.CONSTANT: DEPENDENCY_RATING_CONSTANT_DEGRADATION,
    ConditioningKind.ABSENT: DEPENDENCY_RATING_ABSENT_DEGRADATION,
}


def _dependency_measurements(
    dependency: DependencyTestResult,
    component: DataComponent,
) -> tuple[Measurement, ...]:
    values: list[Measurement] = []
    for kind, definition in _DEGRADATION_METRICS.items():
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


def _dependency_contributions(
    dependency: DependencyTestResult,
    config: NoiseConfig,
) -> dict[str, object] | None:
    """Retain the aligned per-game values a later paired floor bootstraps.

    The dependency family scores one frozen view repeatedly under different
    conditioning, so its uncertainty is uncertainty in the *paired* delta
    between two checkpoints on the same games. That floor belongs to the
    comparison and cannot be attached to either reading alone, which is why it
    is retained here rather than characterized into the committed tier.
    """

    if not config.enabled:
        return None
    contributions = dependency.contributions
    if len(contributions) < 2:
        return None
    metrics = {
        definition.identifier: [
            contribution.degradations[kind] for contribution in contributions
        ]
        for kind, definition in _DEGRADATION_METRICS.items()
        if kind in contributions[0].degradations
    }
    metrics[DEPENDENCY_RATING_ANCHOR_POLICY_DIVERGENCE.identifier] = [
        contribution.anchor_divergence for contribution in contributions
    ]
    metrics[DEPENDENCY_RATING_ANCHOR_TOP1_AGREEMENT.identifier] = [
        contribution.anchor_agreement for contribution in contributions
    ]
    return paired_contributions(
        unit="pool-game",
        unit_ids=[str(contribution.game_id) for contribution in contributions],
        metrics=metrics,
        weights=[float(contribution.positions) for contribution in contributions],
        resamples=config.resamples,
        seed=config.seed,
        coverage=config.coverage,
        confidence=config.confidence,
    ).as_record()


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
    """Return the identity of every enabled position, in one device read."""

    # Game ids stay in their own unsigned dtype; see ``_active_batch``.
    game_ids = batch.game_ids.detach().cpu().tolist()
    enabled, ply_indices = (
        torch.stack(
            (
                batch.action_loss_mask.to(dtype=torch.long),
                batch.ply_indices.to(dtype=torch.long),
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    return tuple(
        (
            game_ids[batch_index][sequence_index],
            ply_indices[batch_index][sequence_index],
        )
        for batch_index, row in enumerate(enabled)
        for sequence_index, active in enumerate(row)
        if active
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
