"""Evaluation at declared cadences during a training run.

A run that only reports at the end says nothing while it is still cheap to
stop. This module makes the same measurements available mid-run, under three
rules that decide whether the readings mean anything.

Cadence is a schedule, not a class of benchmark. Each entry declares when it
runs, which registered metrics it computes, and which view it computes them
over. There is no tier taxonomy: an early reading and the canonical one are
the same metric at two data sizes, so they differ in the view rather than in
kind.

Previews run against the ``validation`` split and never against the held-out
test pool. Running them against the pool would let early readings influence
when a run stops and which checkpoint is kept, which is precisely the
selection pressure the held-out partition exists to prevent. Both splits are
uniform hash assignments over the same corpus, so a validation reading is an
unbiased estimate of the same population quantity, and nothing is lost.

A preview view may subsample and may not filter. That is enforced structurally
here: a preview view has no filter fields to set. Subsampling keeps a preview
an estimate of the canonical quantity with wider error bars; filtering would
make it a different measurement needing a documented conversion.

The corpus a preview reads is narrowed by one thing, before any view sees it:
the marked-account snapshot its validation selection names. That one says which
games count as human play rather than which of them this run trains on, so
applying it is what keeps a preview an estimate of the pool it previews.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from pydantic import Field, StrictBool
from torch import nn

from anthro_chess.config import ConfigModel
from anthro_chess.data import DataLoadingError, SequenceDataConfig, SequenceDataLoader
from anthro_chess.data.accounts import marks_a_player
from anthro_chess.data.artifacts import (
    normalized_shard_paths,
    read_normalized_rows,
)
from anthro_chess.data.schema import (
    NormalizedColumn,
    SplitName,
    row_game_id,
)
from anthro_chess.evaluation.aggregation import SliceTable
from anthro_chess.evaluation.policy import PositionPolicy, active_batch, score_positions
from anthro_chess.evaluation.pool import pool_game
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    ConfigurationReference,
    DataComponent,
    DatasetReference,
    Measurement,
    MetricCost,
    MetricRegistryError,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    ResultsStoreError,
    build_result,
    dataset_reference,
    metric_definition,
    projection_content_digest,
)
from anthro_chess.evaluation.results.metrics import MOVE_PREDICTION_PROJECTION
from anthro_chess.evaluation.scoring import (
    EvaluationLoaderConfig,
    ScoringInputs,
    aggregate_positions,
    build_scoring_inputs,
    rows_identity_sha256,
    slice_measurements,
    slice_metric_identifiers,
)
from anthro_chess.evaluation.views import ViewConfig, ViewSelection, apply_view
from anthro_chess.models import MoveModelBatch
from anthro_chess.training.health import STEP_HEALTH_METRICS, StepHealth

PREVIEW_KIND = "held-out-preview"
HEALTH_KIND = "training-health"
PREVIEW_BENCHMARK = BenchmarkReference(name="held-out-preview", version=1)
HEALTH_BENCHMARK = BenchmarkReference(name="training-health", version=1)

#: Scored positions a schedule may spend per optimizer step, amortized over a
#: cadence's interval. The budget is counted in positions rather than seconds
#: so the same schedule resolves identically on every machine, and it exists to
#: reject an unaffordable pairing rather than to predict a runtime.
DEFAULT_POSITION_BUDGET_PER_STEP = 512

logger = logging.getLogger(__name__)


class CadenceError(ValueError):
    """Raised when a declared cadence cannot run as configured."""


class PreviewViewConfig(ConfigModel):
    """An explicitly sized uniform subsample of the validation split.

    There is deliberately nothing here to filter with. A view that kept only
    short games would run faster and measure a different quantity, and it
    would still be reported as an estimate of the canonical reading.
    """

    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    #: Declared explicitly rather than resolved from a compute or time budget,
    #: which would make one cadence resolve differently on two machines.
    maximum_games: int = Field(ge=1)
    seed: str = Field(default="anthro-training-preview-v1", min_length=1)


class CadenceConfig(ConfigModel):
    """One declared schedule entry."""

    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    every_steps: int = Field(ge=1)
    metrics: tuple[str, ...] = Field(min_length=1)
    view: PreviewViewConfig | None = None


class TrainingEvaluationConfig(ConfigModel):
    """In-training evaluation for one run."""

    #: Identifies the evaluation inputs a preview reads. Previews measure the
    #: validation split of the run's own corpus rather than a frozen pool, and
    #: their series identity comes from the scored content digest regardless.
    dataset_id: str = Field(
        default="validation-split",
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    dataset_version: int = Field(default=1, ge=1)
    position_budget_per_step: int = Field(
        default=DEFAULT_POSITION_BUDGET_PER_STEP,
        ge=1,
    )
    loader: EvaluationLoaderConfig = EvaluationLoaderConfig()
    cadences: tuple[CadenceConfig, ...] = ()
    #: Whether readings are written to the results store. Turning this off
    #: keeps an exploratory run out of committed history.
    record: StrictBool = True


@dataclass(frozen=True)
class PreparedCadence:
    """One resolved cadence entry, with its view already materialized."""

    config: CadenceConfig
    slice_metrics: tuple[str, ...]
    health_metrics: tuple[str, ...]
    inputs: ScoringInputs | None
    component: DataComponent | None
    dataset: DatasetReference | None
    selection: ViewSelection | None
    view_passes: int
    positions_per_step: float

    def due(self, global_step: int) -> bool:
        """Return whether this entry runs at one optimizer step."""

        return global_step % self.config.every_steps == 0

    def as_record(self) -> dict[str, object]:
        """Return the resolved schedule entry for the run artifact."""

        return {
            "name": self.config.name,
            "every_steps": self.config.every_steps,
            "metrics": list(self.config.metrics),
            "view": None if self.selection is None else self.selection.as_record(),
            "scored_positions": (
                0 if self.inputs is None else self.inputs.position_count
            ),
            "view_passes": self.view_passes,
            "positions_per_step": self.positions_per_step,
        }


@dataclass(frozen=True)
class CadenceReading:
    """What one cadence firing measured, and where it was recorded."""

    cadence: str
    global_step: int
    seconds: float
    slices: SliceTable | None
    health: StepHealth | None
    envelopes: tuple[ResultEnvelope, ...]
    recorded_paths: tuple[Path, ...]

    def as_record(self) -> dict[str, object]:
        """Return the metrics-stream record for one reading."""

        return {
            "record": "evaluation",
            "global_step": self.global_step,
            "cadence": self.cadence,
            "seconds": self.seconds,
            "measurements": {
                item.metric: item.value
                for envelope in self.envelopes
                for item in envelope.measurements
            },
            "recorded": [str(path) for path in self.recorded_paths],
        }


class CadenceSchedule:
    """The resolved schedule a run follows, plus how it records readings."""

    def __init__(
        self,
        entries: Sequence[PreparedCadence],
        *,
        configuration: ConfigurationReference | None = None,
        store: ResultsStore | None = None,
    ) -> None:
        self._entries = tuple(entries)
        self._configuration = configuration
        self._store = store

    def __bool__(self) -> bool:
        return bool(self._entries)

    @property
    def entries(self) -> tuple[PreparedCadence, ...]:
        """Return every resolved entry."""

        return self._entries

    def due(self, global_step: int) -> tuple[PreparedCadence, ...]:
        """Return the entries that run at one optimizer step."""

        return tuple(entry for entry in self._entries if entry.due(global_step))

    def run(
        self,
        entry: PreparedCadence,
        model: nn.Module,
        *,
        device: torch.device,
        global_step: int,
        checkpoint: CheckpointReference,
        health: StepHealth | None,
    ) -> CadenceReading:
        """Measure one cadence entry and record it as its own results."""

        started = time.perf_counter()
        recorded_at = datetime.now(tz=UTC)
        envelopes: list[ResultEnvelope] = []
        slices: SliceTable | None = None

        if entry.slice_metrics:
            assert entry.inputs is not None
            assert entry.component is not None
            assert entry.dataset is not None
            logger.info(
                "Running cadence %r over %s preview position(s) at step %s",
                entry.config.name,
                entry.inputs.position_count,
                global_step,
            )
            slices = score_preview(model, entry.inputs, device=device)
            envelopes.append(
                self._envelope(
                    kind=PREVIEW_KIND,
                    benchmark=PREVIEW_BENCHMARK,
                    checkpoint=checkpoint,
                    data=entry.dataset,
                    measurements=slice_measurements(
                        slices,
                        entry.component,
                        metrics=entry.slice_metrics,
                    ),
                    recorded_at=recorded_at,
                )
            )

        if entry.health_metrics and health is not None:
            measurements = health.measurements(entry.health_metrics)
            if measurements:
                envelopes.append(
                    self._envelope(
                        kind=HEALTH_KIND,
                        benchmark=HEALTH_BENCHMARK,
                        checkpoint=checkpoint,
                        data=None,
                        measurements=measurements,
                        recorded_at=recorded_at,
                    )
                )

        recorded: list[Path] = []
        if self._store is not None:
            try:
                recorded = [self._store.append(envelope) for envelope in envelopes]
            except (ResultRecordError, ResultsStoreError) as error:
                raise CadenceError(str(error)) from error

        return CadenceReading(
            cadence=entry.config.name,
            global_step=global_step,
            seconds=time.perf_counter() - started,
            slices=slices,
            health=health if entry.health_metrics else None,
            envelopes=tuple(envelopes),
            recorded_paths=tuple(recorded),
        )

    def as_record(self) -> list[dict[str, object]]:
        """Return the resolved schedule for the run artifact."""

        return [entry.as_record() for entry in self._entries]

    def _envelope(
        self,
        *,
        kind: str,
        benchmark: BenchmarkReference,
        checkpoint: CheckpointReference,
        data: DatasetReference | None,
        measurements: Sequence[Measurement],
        recorded_at: datetime,
    ) -> ResultEnvelope:
        try:
            return build_result(
                kind=kind,
                benchmark=benchmark,
                checkpoint=checkpoint,
                configuration=self._configuration,
                data=data,
                measurements=measurements,
                recorded_at=recorded_at,
            )
        except ResultRecordError as error:
            raise CadenceError(str(error)) from error


def score_preview(
    model: nn.Module,
    inputs: ScoringInputs,
    *,
    device: torch.device,
) -> SliceTable:
    """Score a preview view with the live model and aggregate every slice."""

    positions: list[PositionPolicy] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for sequence_batch in SequenceDataLoader(
                inputs.dataset,
                inputs.loader_config,
            ):
                batch = MoveModelBatch.from_sequence_batch(
                    sequence_batch,
                    device=device,
                )
                positions.extend(score_positions(active_batch(model(batch), batch)))
    finally:
        model.train(was_training)
    if not positions:
        raise CadenceError("the configured preview view scored no positions")
    return aggregate_positions(positions, inputs)


def prepare_schedule(
    config: TrainingEvaluationConfig,
    validation: SequenceDataConfig | None,
    *,
    configuration: ConfigurationReference | None = None,
    store: ResultsStore | None = None,
    marked_digests: frozenset[int] | None = None,
) -> CadenceSchedule:
    """Resolve every declared cadence before the first optimizer step.

    Everything that can fail is resolved here: unknown metrics, a metric that
    needs machinery a training run does not have, a missing preview view, and
    an unaffordable pairing of view size against interval. A run that cannot
    honor its schedule should say so before it spends an hour training.
    """

    if not config.cadences:
        return CadenceSchedule((), configuration=configuration, store=store)

    names = [entry.name for entry in config.cadences]
    if len(set(names)) != len(names):
        raise CadenceError("each cadence entry needs a distinct name")

    supported = slice_metric_identifiers()
    cache: dict[tuple[str, int, str], _PreparedView] = {}
    entries: list[PreparedCadence] = []
    for entry in config.cadences:
        slice_metrics, health_metrics, passes = _resolve_metrics(entry, supported)
        prepared_view: _PreparedView | None = None
        if slice_metrics:
            if entry.view is None:
                raise CadenceError(
                    f"cadence {entry.name!r} computes {slice_metrics[0]!r} over "
                    "held-out data and must declare a view"
                )
            key = (entry.view.name, entry.view.maximum_games, entry.view.seed)
            if key not in cache:
                cache[key] = _prepare_view(
                    config, entry.view, validation, marked_digests
                )
            prepared_view = cache[key]

        positions = 0 if prepared_view is None else prepared_view.inputs.position_count
        positions_per_step = positions * passes / entry.every_steps
        if positions_per_step > config.position_budget_per_step:
            raise CadenceError(
                f"cadence {entry.name!r} scores {positions} position(s) "
                f"{passes} time(s) every {entry.every_steps} step(s), which is "
                f"{positions_per_step:.1f} position(s) per optimizer step "
                f"against a budget of {config.position_budget_per_step}. Widen "
                "the interval, shrink the view, or raise the budget "
                "deliberately."
            )
        entries.append(
            PreparedCadence(
                config=entry,
                slice_metrics=slice_metrics,
                health_metrics=health_metrics,
                inputs=None if prepared_view is None else prepared_view.inputs,
                component=None if prepared_view is None else prepared_view.component,
                dataset=None if prepared_view is None else prepared_view.dataset,
                selection=None if prepared_view is None else prepared_view.selection,
                view_passes=passes,
                positions_per_step=positions_per_step,
            )
        )
        logger.info(
            "Cadence %r runs every %s step(s) over %s position(s)",
            entry.name,
            entry.every_steps,
            positions,
        )
    return CadenceSchedule(entries, configuration=configuration, store=store)


@dataclass(frozen=True)
class _PreparedView:
    """One materialized preview view, shared by every cadence naming it."""

    selection: ViewSelection
    inputs: ScoringInputs
    component: DataComponent
    dataset: DatasetReference


def _resolve_metrics(
    entry: CadenceConfig,
    supported: frozenset[str],
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    """Split one entry's metrics by what computes them, and price the pairing."""

    if len(set(entry.metrics)) != len(entry.metrics):
        raise CadenceError(f"cadence {entry.name!r} names a metric twice")
    slice_metrics: list[str] = []
    health_metrics: list[str] = []
    passes = 0
    for identifier in entry.metrics:
        try:
            definition = metric_definition(identifier)
        except MetricRegistryError as error:
            raise CadenceError(f"cadence {entry.name!r}: {error}") from error
        if definition.cost is MetricCost.GENERATED:
            raise CadenceError(
                f"cadence {entry.name!r} names {identifier!r}, which needs "
                "generated games rather than a pass over stored positions and "
                "cannot run at a training cadence"
            )
        if definition.cost is MetricCost.MEASURED_EXECUTION:
            raise CadenceError(
                f"cadence {entry.name!r} names {identifier!r}, which measures "
                "execution time. Taken beside a training step it would report "
                "contention with that step rather than a property of the "
                "checkpoint, so it runs as its own benchmark"
            )
        if identifier in STEP_HEALTH_METRICS:
            health_metrics.append(identifier)
            continue
        if identifier not in supported:
            raise CadenceError(
                f"cadence {entry.name!r} names {identifier!r}, which no "
                "in-training measurement computes. Declare a metric the "
                "preview or the per-step health monitor reports."
            )
        cost = definition.cost.view_passes
        if cost is None:  # pragma: no cover - generated costs are rejected above
            raise CadenceError(
                f"cadence {entry.name!r} names {identifier!r}, whose cost "
                "cannot be expressed as passes over a view"
            )
        slice_metrics.append(identifier)
        passes = max(passes, cost)
    return tuple(slice_metrics), tuple(health_metrics), passes


def _prepare_view(
    config: TrainingEvaluationConfig,
    view: PreviewViewConfig,
    validation: SequenceDataConfig | None,
    marked_digests: frozenset[int] | None,
) -> _PreparedView:
    if validation is None:
        raise CadenceError(
            "in-training previews read the validation split, so a run with a "
            "preview cadence needs a validation data selection"
        )
    split = validation.loader.split
    if split == "test":
        raise CadenceError(
            "in-training previews must never read the held-out test split"
        )

    rows = _validation_rows(validation, split, marked_digests)
    selection = apply_view(
        [pool_game(row) for row in rows],
        ViewConfig(
            name=view.name,
            maximum_games=view.maximum_games,
            seed=view.seed,
        ),
    )
    if not selection.game_ids:
        raise CadenceError(f"preview view {view.name!r} selected no games")

    wanted = set(selection.game_ids)
    selected = [row for row in rows if row_game_id(row) in wanted]
    inputs = build_scoring_inputs(
        selected,
        split=split,
        batch_size=config.loader.batch_size,
        length_bucket_width=config.loader.length_bucket_width,
        identity_sha256=rows_identity_sha256(
            selected,
            context=selection.as_record(),
        ),
    )
    component = projection_content_digest(inputs.rows, MOVE_PREDICTION_PROJECTION)
    return _PreparedView(
        selection=selection,
        inputs=inputs,
        component=component,
        dataset=dataset_reference(
            pool_id=config.dataset_id,
            pool_version=config.dataset_version,
            view=selection.name,
            selected_games=selection.selected_games,
            game_ids_sha256=str(selection.as_record()["game_ids_sha256"]),
            components=[component],
        ),
    )


def _validation_rows(
    validation: SequenceDataConfig,
    split: SplitName,
    marked_digests: frozenset[int] | None,
) -> list[dict[str, Any]]:
    """Read the split a preview scores, less the accounts its corpus rejects.

    A preview otherwise applies no selection filter, which is what keeps it an
    unbiased estimate of the canonical reading rather than a different
    measurement. The marked-account rejection is the exception because it is a
    statement about which games count as human play rather than a narrowing of
    what this run trains on: leaving it out here would have a preview estimate
    a population the pool it previews does not hold.
    """

    try:
        paths = normalized_shard_paths(validation.normalized)
        rows = [
            dict(row)
            for path in paths
            for row in read_normalized_rows(path)
            if row[NormalizedColumn.SPLIT] == split
            and not marks_a_player(row, marked_digests)
        ]
    except (DataLoadingError, OSError, json.JSONDecodeError) as error:
        raise CadenceError(str(error)) from error
    if not rows:
        raise CadenceError(
            f"the validation selection contains no games in the {split} split"
            + (" that its marked-account snapshot leaves" if marked_digests else "")
        )
    return rows


__all__ = [
    "DEFAULT_POSITION_BUDGET_PER_STEP",
    "HEALTH_KIND",
    "PREVIEW_KIND",
    "CadenceConfig",
    "CadenceError",
    "CadenceReading",
    "CadenceSchedule",
    "PreparedCadence",
    "PreviewViewConfig",
    "TrainingEvaluationConfig",
    "prepare_schedule",
    "score_preview",
]
