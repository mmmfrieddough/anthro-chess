"""Executable device-aware training over the ordinary package boundaries."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from anthro_chess.chess import action_vocabulary_identity
from anthro_chess.config import ResolvedConfig
from anthro_chess.data import (
    STREAMING_LOADER_NAME,
    DataLoadingError,
    SequenceBatchSource,
    SequenceDataConfig,
    SequenceDataLoader,
    StreamingSequenceDataLoader,
    build_sharded_index,
    encoding_identity,
    maximum_position_bound,
)
from anthro_chess.data.artifacts import (
    ShardIdentity,
    normalized_shard_paths,
    validate_manifest_compatibility,
    validate_manifest_outputs,
)
from anthro_chess.evaluation import MoveValidationMetrics, evaluate_move_model
from anthro_chess.evaluation.results import (
    CheckpointReference,
    ConfigurationReference,
    DetailReference,
    DetailStore,
    ExecutionRecord,
    ResultsStore,
    configuration_reference,
    default_checkpoint_label,
)
from anthro_chess.models import CausalMoveModel, MoveModelBatch
from anthro_chess.provenance import code_provenance
from anthro_chess.training.cadence import (
    CadenceError,
    CadenceReading,
    CadenceSchedule,
    prepare_schedule,
)
from anthro_chess.training.checkpoints import (
    CheckpointError,
    checkpoint_path,
    clear_latest_checkpoint,
    latest_checkpoint_path,
    load_training_checkpoint,
    parameter_sha256,
    restore_rng_state,
    save_training_checkpoint,
)
from anthro_chess.training.config import TrainingConfig
from anthro_chess.training.devices import (
    STRICT_DETERMINISM_BACKENDS,
    DeviceCapabilities,
    DeviceError,
    hardware_record,
    resolve_training_device,
)
from anthro_chess.training.efficiency import (
    DeferredStepTotals,
    TrainingEfficiencyError,
    TrainingEfficiencyMonitor,
    TrainingEfficiencySummary,
    coordinate_record,
    driver_allocated_memory_bytes,
    record_efficiency,
    write_efficiency_detail,
)
from anthro_chess.training.efficiency import (
    execution_record as efficiency_execution_record,
)
from anthro_chess.training.health import StepHealth, StepHealthMonitor
from anthro_chess.training.losses import masked_action_cross_entropy
from anthro_chess.training.tensorboard import (
    TENSORBOARD_DIRECTORY,
    TrainingTensorBoard,
)

RUN_ARTIFACT_VERSION = 6
logger = logging.getLogger(__name__)


class TrainingError(ValueError):
    """Raised when a configured training run cannot execute safely."""


@dataclass(frozen=True)
class TrainingResult:
    """Paths and summary values produced by one training run."""

    run_path: Path
    metrics_path: Path
    checkpoint_path: Path
    steps: int
    initial_parameter_sha256: str
    final_parameter_sha256: str
    validation: MoveValidationMetrics | None
    readings: tuple[CadenceReading, ...] = ()
    efficiency: TrainingEfficiencySummary | None = None
    efficiency_paths: tuple[Path, ...] = ()
    efficiency_detail_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _DataSelection:
    loader: SequenceBatchSource
    provenance: dict[str, object]


@dataclass(frozen=True)
class _OptimizationResult:
    processed_positions: int
    checkpoint_path: Path
    readings: tuple[CadenceReading, ...]
    instrumentation_seconds: float
    efficiency_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _EfficiencyRecorder:
    """How a run writes what it cost, at a cadence and at the end.

    An intermediate reading exists so a budget report has more than one point
    per run. Without it, quality against processed positions would be a curve
    of one point per training run, which answers nothing about where in a run
    the returns fell off.
    """

    monitor: TrainingEfficiencyMonitor
    execution: ExecutionRecord
    configuration: ConfigurationReference | None
    store: ResultsStore | None
    record_at_cadence: bool

    def record(
        self,
        *,
        checkpoint: CheckpointReference,
        processed_positions: int,
        detail: DetailReference | None = None,
    ) -> tuple[Path, ...]:
        """Append one efficiency reading for the parameters named."""

        summary = self.monitor.summary(processed_positions=processed_positions)
        try:
            _, paths = record_efficiency(
                summary,
                checkpoint=checkpoint,
                execution=self.execution,
                store=self.store,
                configuration=self.configuration,
                detail=detail,
            )
        except TrainingEfficiencyError as error:
            raise TrainingError(str(error)) from error
        return paths


def run_training(
    resolved_config: ResolvedConfig[TrainingConfig],
    *,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> TrainingResult:
    """Run a bounded optimization on the resolved device and write provenance.

    Passing no ``store`` runs any declared cadence and records nothing, which
    is what an exploratory run wants: committed history should hold readings
    somebody meant to keep.
    """

    run_started = time.perf_counter()
    config = resolved_config.value
    device = _training_device(config)
    efficiency_monitor = TrainingEfficiencyMonitor(
        config.efficiency,
        device=device,
        started=run_started,
    )
    logger.info(
        "Starting training on %s for %s optimizer step(s)",
        device.type,
        config.steps,
    )
    try:
        # Training reads targets and masks; validation scores policies against
        # each position's legal actions. Only the second needs them, so only
        # the second reconstructs them — which is most of what decoding a game
        # costs, before any of what packing and pickling one costs.
        train = _load_data_selection(
            config.train,
            legal_actions=False,
            maximum_context_plies=config.model.maximum_context_plies,
        )
        validation = (
            _load_data_selection(
                config.validation,
                legal_actions=True,
                maximum_context_plies=config.model.maximum_context_plies,
            )
            if config.validation is not None
            else None
        )
    except (DataLoadingError, OSError, ValueError, json.JSONDecodeError) as error:
        raise TrainingError(str(error)) from error

    output_directory = config.output_directory
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise TrainingError(
            f"cannot create training output directory {output_directory}: {error}"
        ) from error
    metrics_path = output_directory / "metrics.jsonl"
    run_path = output_directory / "run.json"

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "mps":
        torch.mps.manual_seed(config.seed)
    elif device.type == "cuda":  # pragma: no cover - no CUDA in the CPU suite
        torch.cuda.manual_seed_all(config.seed)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(config.determinism == "strict")
    try:
        model = CausalMoveModel(config.model).to(
            device=device,
            dtype=_training_dtype(config),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            fused=_fused_optimizer(device),
        )
        code_record = code_provenance().as_record()
        data_record = {
            "train": train.provenance,
            "validation": validation.provenance if validation is not None else None,
        }
        model_identity = model.identity()
        execution_record = _execution_record(config, device)
        compatibility = _compatibility_record(
            config,
            data=data_record,
            model=model_identity,
        )
        checkpoint_metadata = {
            "resolved_config": resolved_config.as_record(),
            "code": code_record,
            "data": data_record,
            "model": model_identity,
            "action_vocabulary": action_vocabulary_identity(),
            "encoding": encoding_identity(),
            "execution": execution_record,
        }
        resumed_from: Path | None = None
        starting_step = 0
        processed_positions = 0
        if config.resume_from is not None:
            resumed_from = (
                latest_checkpoint_path(output_directory)
                if config.resume_from == "latest"
                else config.resume_from
            )
            checkpoint = load_training_checkpoint(resumed_from)
            logger.info("Resuming training from %s", resumed_from)
            _validate_checkpoint_compatibility(
                checkpoint["compatibility"],
                compatibility,
            )
            _validate_checkpoint_execution(
                checkpoint["metadata"],
                execution_record,
            )
            starting_step = checkpoint["global_step"]
            if starting_step >= config.steps:
                raise TrainingError(
                    f"checkpoint global step {starting_step} must be below "
                    f"configured target steps {config.steps}"
                )
            try:
                model.load_state_dict(checkpoint["model_state"])
            except RuntimeError as error:
                raise CheckpointError(
                    f"checkpoint model state cannot be restored on "
                    f"{device.type}: {error}"
                ) from error
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            except (RuntimeError, ValueError) as error:
                raise CheckpointError(
                    f"checkpoint optimizer state cannot be restored on "
                    f"{device.type}: {error}"
                ) from error
            if checkpoint["scheduler_state"] is not None:
                raise CheckpointError(
                    "checkpoint contains scheduler state but the current "
                    "training configuration has no scheduler"
                )
            if checkpoint["scaler_state"] is not None:
                raise CheckpointError(
                    "checkpoint contains scaler state but float32 training "
                    "does not use a gradient scaler"
                )
            train.loader.load_state(checkpoint["loader_state"])
            counters = checkpoint["counters"]
            if set(counters) != {"processed_positions"}:
                raise CheckpointError("checkpoint counters are incomplete or unknown")
            processed_positions = counters["processed_positions"]
            if type(processed_positions) is not int or processed_positions < 0:
                raise CheckpointError(
                    "checkpoint processed_positions must be nonnegative"
                )
            restore_rng_state(
                checkpoint["rng_state"],
                device=device,
                fallback_seed=config.seed,
            )

        configuration = configuration_reference(
            resolved_config.as_record(),
            source=resolved_config.provenance.source,
            overrides=resolved_config.provenance.overrides,
        )
        schedule = prepare_schedule(
            config.evaluation,
            config.validation,
            configuration=configuration,
            store=store if config.evaluation.record else None,
        )
        efficiency_recorder = _EfficiencyRecorder(
            monitor=efficiency_monitor,
            execution=efficiency_execution_record(
                coordinate_record(
                    dataset_sha256=str(train.provenance["dataset_sha256"]),
                    loader_configuration_sha256=str(
                        train.provenance["loader_configuration_sha256"]
                    ),
                    model_identity=model_identity,
                    batch_size=train.loader.config.batch_size,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    determinism=config.determinism,
                    profile_phases=config.profile_phases,
                ),
                device=device,
                precision=config.precision,
            ),
            configuration=configuration,
            store=store if config.efficiency.record else None,
            record_at_cadence=config.efficiency.record_at_cadence,
        )

        initial_parameter_sha256 = parameter_sha256(model)
        if resumed_from is None:
            clear_latest_checkpoint(output_directory)
        _prepare_metrics(metrics_path, through_step=starting_step)
        efficiency_monitor.begin_optimization(starting_step=starting_step)
        optimization = _optimize(
            model,
            optimizer,
            train.loader,
            steps=config.steps,
            starting_step=starting_step,
            processed_positions=processed_positions,
            log_every_steps=config.log_every_steps,
            checkpoint_every_steps=config.checkpoint_every_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            device=device,
            autocast=_autocast(config, device),
            profile_phases=config.profile_phases,
            metrics_path=metrics_path,
            output_directory=output_directory,
            compatibility=compatibility,
            checkpoint_metadata=checkpoint_metadata,
            schedule=schedule,
            run_id=output_directory.name,
            efficiency=efficiency_recorder,
        )
        _synchronize_device(device)
        final_parameter_sha256 = parameter_sha256(model)
        if final_parameter_sha256 == initial_parameter_sha256:
            raise TrainingError("optimizer completed without changing model parameters")

        if validation is not None:
            logger.info("Running validation")
            validation_started = time.perf_counter()
            validation_metrics = evaluate_move_model(
                model,
                validation.loader,
                device=device,
            )
            efficiency_monitor.charge_validation(
                time.perf_counter() - validation_started
            )
        else:
            validation_metrics = None

        final_checkpoint = CheckpointReference(
            label=default_checkpoint_label(output_directory.name, config.steps),
            step=config.steps,
            run_id=output_directory.name,
            parameter_sha256=final_parameter_sha256,
        )
        efficiency_summary = efficiency_monitor.summary(
            processed_positions=optimization.processed_positions,
        )
        efficiency_detail_paths: list[Path] = []
        recorded_at = datetime.now(tz=UTC)
        efficiency_detail = write_efficiency_detail(
            detail,
            checkpoint=final_checkpoint,
            recorded_at=recorded_at,
            payload=efficiency_summary.as_record(),
            paths=efficiency_detail_paths,
        )
        efficiency_paths = (
            *optimization.efficiency_paths,
            *efficiency_recorder.record(
                checkpoint=final_checkpoint,
                processed_positions=optimization.processed_positions,
                detail=efficiency_detail,
            ),
        )
        run_record = {
            "version": RUN_ARTIFACT_VERSION,
            "resolved_config": resolved_config.as_record(),
            "seed": config.seed,
            "code": code_record,
            "data": data_record,
            "model": model_identity,
            "action_vocabulary": action_vocabulary_identity(),
            "encoding": encoding_identity(),
            "execution": execution_record,
            "hardware": hardware_record(device),
            "optimization": {
                "optimizer": "Adam",
                "starting_step": starting_step,
                "completed_steps": config.steps,
                "processed_positions": optimization.processed_positions,
                "resumed_from": (
                    str(resumed_from.resolve()) if resumed_from is not None else None
                ),
                "checkpoint": str(optimization.checkpoint_path.resolve()),
                "initial_parameter_sha256": initial_parameter_sha256,
                "final_parameter_sha256": final_parameter_sha256,
            },
            "validation": (
                validation_metrics.as_record()
                if validation_metrics is not None
                else None
            ),
            "evaluation": {
                "cadences": schedule.as_record(),
                "readings": [
                    {
                        "cadence": reading.cadence,
                        "global_step": reading.global_step,
                        "seconds": reading.seconds,
                        "recorded": [str(path) for path in reading.recorded_paths],
                    }
                    for reading in optimization.readings
                ],
                "instrumentation_seconds": optimization.instrumentation_seconds,
            },
            "efficiency": {
                **efficiency_summary.as_record(),
                "coordinates": efficiency_recorder.execution.coordinates,
                "recorded": [str(path) for path in efficiency_paths],
                "detail": [str(path) for path in efficiency_detail_paths],
            },
        }
        run_path.write_text(
            json.dumps(run_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except torch.OutOfMemoryError as error:  # pragma: no cover - needs a real device
        raise TrainingError(
            _out_of_memory_message(
                config,
                device,
                train.loader.config.batch_size,
                error,
            )
        ) from error
    except (
        CadenceError,
        CheckpointError,
        DataLoadingError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        if isinstance(error, TrainingError):
            raise
        raise TrainingError(f"training failed: {error}") from error
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        # A shard-backed loader holds worker processes and an open shard; both
        # outlive the run unless the run releases them, including the run that
        # is failing.
        train.loader.close()
        if validation is not None:
            validation.loader.close()

    logger.info("Completed training and wrote run, metrics, and checkpoint artifacts")
    return TrainingResult(
        run_path=run_path,
        metrics_path=metrics_path,
        checkpoint_path=optimization.checkpoint_path,
        steps=config.steps,
        initial_parameter_sha256=initial_parameter_sha256,
        final_parameter_sha256=final_parameter_sha256,
        validation=validation_metrics,
        readings=optimization.readings,
        efficiency=efficiency_summary,
        efficiency_paths=efficiency_paths,
        efficiency_detail_paths=tuple(efficiency_detail_paths),
    )


def _optimize(
    model: CausalMoveModel,
    optimizer: torch.optim.Optimizer,
    loader: SequenceBatchSource,
    *,
    steps: int,
    starting_step: int,
    processed_positions: int,
    log_every_steps: int,
    checkpoint_every_steps: int,
    gradient_accumulation_steps: int,
    device: torch.device,
    autocast: AbstractContextManager[None],
    profile_phases: bool,
    metrics_path: Path,
    output_directory: Path,
    compatibility: Mapping[str, object],
    checkpoint_metadata: Mapping[str, object],
    schedule: CadenceSchedule,
    run_id: str,
    efficiency: _EfficiencyRecorder,
) -> _OptimizationResult:
    model.train()
    monitor = efficiency.monitor
    saved_checkpoint: Path | None = None
    data_seconds = 0.0
    transfer_seconds = 0.0
    compute_seconds = 0.0
    optimizer_seconds = 0.0
    health_monitor = StepHealthMonitor(model.parameters())
    charged_instrumentation = 0.0
    readings: list[CadenceReading] = []
    efficiency_paths: list[Path] = []
    totals = DeferredStepTotals(device)
    interval_start_step = starting_step + 1
    interval_positions = 0
    purge_step = starting_step + 1 if starting_step > 0 else None
    with (
        TrainingTensorBoard(
            output_directory / TENSORBOARD_DIRECTORY,
            purge_step=purge_step,
        ) as tensorboard,
        metrics_path.open("a", encoding="utf-8") as metrics_file,
    ):
        for global_step in range(starting_step + 1, steps + 1):
            # Everything that decides how a step is instrumented is resolved
            # before it runs. A probe step has to know it is one before its
            # first micro-batch, and a step that draws its arm afterwards would
            # already have skipped the synchronization being measured.
            due = schedule.due(global_step)
            reported = (
                global_step % log_every_steps == 0 or global_step == steps or bool(due)
            )
            saving = global_step % checkpoint_every_steps == 0 or global_step == steps
            draining = reported or saving
            if not monitor.interval_open:
                monitor.begin_interval(global_step)
            # The whole interval shares an arm. A step that drew its own would
            # skip the synchronization being measured before the draw, and its
            # neighbours would absorb the device work it deferred.
            probe = monitor.interval_probing
            in_window = monitor.interval_in_window

            monitor.begin_step()
            totals.begin_step()
            optimizer.zero_grad()
            for _ in range(gradient_accumulation_steps):
                data_started = time.perf_counter()
                try:
                    sequence_batch = next(loader)
                except StopIteration:
                    loader.start_epoch(loader.state().epoch + 1)
                    try:
                        sequence_batch = next(loader)
                    except StopIteration as error:
                        raise TrainingError(
                            "training data produced no batches; "
                            "check drop_last and batch size"
                        ) from error
                data_seconds += time.perf_counter() - data_started

                if profile_phases:
                    _synchronize_device(device)
                transfer_started = time.perf_counter()
                batch = MoveModelBatch.from_sequence_batch(
                    sequence_batch,
                    device=device,
                )
                if profile_phases:
                    _synchronize_device(device)
                transfer_seconds += time.perf_counter() - transfer_started

                compute_started = time.perf_counter()
                with autocast:
                    loss = masked_action_cross_entropy(
                        model(batch),
                        batch.action_targets,
                        batch.action_loss_mask,
                    )
                # Backward stays outside the autocast scope. Each operation's
                # backward already runs in the dtype its forward chose, and
                # nesting it would autocast a second time.
                (loss / gradient_accumulation_steps).backward()
                if profile_phases:
                    _synchronize_device(device)
                compute_seconds += time.perf_counter() - compute_started
                totals.observe(
                    loss,
                    batch.action_loss_mask,
                    window=in_window,
                    probe=probe,
                )
                if probe:
                    totals.synchronize()

            health_monitor.observe_gradients()
            if reported:
                health_monitor.snapshot_parameters()

            optimizer_started = time.perf_counter()
            optimizer.step()
            if profile_phases:
                _synchronize_device(device)
            optimizer_seconds += time.perf_counter() - optimizer_started
            health_monitor.observe_update()
            totals.end_step()
            # Both drains stay inside the timed span. They are where the
            # interval's queued device work is finally paid for, and timing an
            # interval without them would report the enqueue rather than the
            # work. Their cost is charged to instrumentation, not to training.
            health: StepHealth | None = (
                health_monitor.drain(global_step) if reported else None
            )
            charged_instrumentation = _charge_instrumentation(
                monitor,
                health_monitor,
                charged_instrumentation,
            )
            padded_positions = totals.padded_positions
            drained = totals.drain() if draining else None
            monitor.end_step()

            average_loss: float | None = None
            if drained is not None:
                if not drained.finite:
                    raise TrainingError(
                        "move loss is not finite between global steps "
                        f"{interval_start_step} and {global_step}; the deferred "
                        "loss read-back reports the interval rather than the "
                        "exact step"
                    )
                interval_positions = drained.active_positions
                processed_positions += drained.active_positions
                monitor.close_interval(drained, padded_positions=padded_positions)
                average_loss = drained.final_step_loss_sum / gradient_accumulation_steps
                interval_start_step = global_step + 1

            epoch = loader.state().epoch
            if reported:
                assert average_loss is not None  # a reported step always drains
                report_started = time.perf_counter()
                learning_rate = float(optimizer.param_groups[0]["lr"])
                summary = monitor.summary(processed_positions=processed_positions)
                record: dict[str, object] = {
                    "record": "step",
                    "global_step": global_step,
                    "epoch": epoch,
                    "move_loss": average_loss,
                    "learning_rate": learning_rate,
                    # The logging interval's active positions, not one step's.
                    # A per-step count would need a device read-back on every
                    # step, which is the cost the deferred totals exist to
                    # avoid, so the field says which quantity it carries.
                    "interval_active_positions": interval_positions,
                    "processed_positions": processed_positions,
                    # Steady state, so it is null until the window opens. The
                    # figure that included warmup answered a different question
                    # and drifted for the whole run.
                    "positions_per_second": summary.active_positions_per_second,
                    "elapsed_seconds": summary.run_seconds,
                    "training_seconds": summary.training_seconds,
                    "evaluation_seconds": summary.evaluation_seconds,
                    "checkpoint_seconds": summary.checkpoint_seconds,
                    "data_seconds": data_seconds if profile_phases else None,
                    "transfer_seconds": (transfer_seconds if profile_phases else None),
                    # Forward and backward only. The optimizer's own work is
                    # reported beside it rather than inside it, because the two
                    # scale with different things and a merged figure cannot
                    # say which of them a slow step is spending its time in.
                    "compute_seconds": compute_seconds if profile_phases else None,
                    "optimizer_seconds": (
                        optimizer_seconds if profile_phases else None
                    ),
                    "peak_sampled_allocated_memory_bytes": (
                        monitor.peak_allocated_memory_bytes
                    ),
                    "peak_sampled_driver_memory_bytes": (
                        monitor.peak_driver_memory_bytes
                    ),
                    "training_health": (
                        health.as_record() if health is not None else None
                    ),
                    "health_instrumentation_seconds": (
                        health_monitor.instrumentation_seconds
                    ),
                }
                metrics_file.write(
                    json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                )
                metrics_file.flush()
                tensorboard.write_step(
                    global_step=global_step,
                    move_loss=average_loss,
                    learning_rate=learning_rate,
                    health=health,
                )
                logger.info(
                    f"step={global_step} move_loss={average_loss:.6f} "
                    f"lr={learning_rate:.6g} positions={processed_positions} "
                    f"positions_per_second="
                    f"{_render_throughput(summary.active_positions_per_second)}"
                )
                monitor.charge(
                    time.perf_counter() - report_started,
                    kind="instrumentation",
                )
            if due:
                # One reference for every entry firing here: an in-training
                # preview and the later canonical reading of these parameters
                # have to agree on which checkpoint they describe.
                measured = CheckpointReference(
                    label=default_checkpoint_label(run_id, global_step),
                    step=global_step,
                    run_id=run_id,
                    parameter_sha256=parameter_sha256(model),
                )
                for entry in due:
                    reading = schedule.run(
                        entry,
                        model,
                        device=device,
                        global_step=global_step,
                        checkpoint=measured,
                        health=health,
                    )
                    readings.append(reading)
                    monitor.charge(reading.seconds, kind="evaluation")
                    metrics_file.write(
                        json.dumps(
                            reading.as_record(),
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    metrics_file.flush()
                    tensorboard.write_evaluation(reading)
                if efficiency.record_at_cadence:
                    # The budget point a quality-versus-time report joins to
                    # the preview reading just taken, under the same label.
                    efficiency_paths.extend(
                        efficiency.record(
                            checkpoint=measured,
                            processed_positions=processed_positions,
                        )
                    )
            if saving:
                checkpoint_started = time.perf_counter()
                saved_checkpoint = checkpoint_path(output_directory, global_step)
                save_training_checkpoint(
                    saved_checkpoint,
                    global_step=global_step,
                    counters={"processed_positions": processed_positions},
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=None,
                    scaler_state=None,
                    loader_state=loader.state().as_record(),
                    compatibility=compatibility,
                    metadata=checkpoint_metadata,
                    device=device,
                )
                monitor.charge(
                    time.perf_counter() - checkpoint_started,
                    kind="checkpoint",
                )
                logger.info("Saved checkpoint at optimizer step %s", global_step)
    if saved_checkpoint is None:
        raise TrainingError("training completed without saving a checkpoint")
    return _OptimizationResult(
        processed_positions=processed_positions,
        checkpoint_path=saved_checkpoint,
        readings=tuple(readings),
        instrumentation_seconds=health_monitor.instrumentation_seconds,
        efficiency_paths=tuple(efficiency_paths),
    )


def _charge_instrumentation(
    monitor: TrainingEfficiencyMonitor,
    health_monitor: StepHealthMonitor,
    charged: float,
) -> float:
    """Charge the health monitor's new instrumentation time to this step.

    The monitor reports a running total, so the step owns the increment. Left
    uncharged, a run that measures its own gradients would report itself as
    slower at training than one that does not.
    """

    total = health_monitor.instrumentation_seconds
    monitor.charge_step_overhead(max(total - charged, 0.0), kind="instrumentation")
    return total


def _out_of_memory_message(  # pragma: no cover - needs a real device to raise
    config: TrainingConfig,
    device: torch.device,
    batch_size: int,
    error: BaseException,
) -> str:
    """Explain an exhausted accelerator in terms of the dials that fix it.

    Torch's own message reports allocator arithmetic, which establishes that
    the run did not fit but not what to change. Both halves of the effective
    batch are named because they trade against each other: accumulation buys
    the same effective batch back out of a smaller resident one.
    """

    accumulation = config.gradient_accumulation_steps
    reserved = driver_allocated_memory_bytes(device)
    held = (
        "an unmeasurable amount"
        if reserved is None
        else f"{reserved / 2**30:.2f} GiB reserved"
    )
    return (
        f"training ran out of memory on {device.type} with batch {batch_size} "
        f"by {accumulation} accumulation step(s), an effective batch of "
        f"{batch_size * accumulation}, while holding {held}: {error}. Lower "
        f"the loader batch_size, raising gradient_accumulation_steps by the "
        f"same factor to keep the effective batch unchanged"
    )


def _render_throughput(value: float | None) -> str:
    """Render steady-state throughput, which is absent during warmup."""

    return "warmup" if value is None else f"{value:.2f}"


def _synchronize_device(device: torch.device) -> None:
    if device.type == "mps":  # pragma: no cover - no MPS in the CPU suite
        torch.mps.synchronize()
    elif device.type == "cuda":  # pragma: no cover - no CUDA in the CPU suite
        torch.cuda.synchronize(device)


def _prepare_metrics(path: Path, *, through_step: int) -> None:
    """Keep only records committed by the selected restart boundary."""

    records: list[str] = []
    if through_step > 0 and path.is_file():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TrainingError(
                    f"metrics record {line_number} is not valid JSON"
                ) from error
            global_step = record.get("global_step")
            if type(global_step) is not int:
                raise TrainingError(
                    f"metrics record {line_number} has no integer global_step"
                )
            if global_step <= through_step:
                records.append(json.dumps(record, sort_keys=True, allow_nan=False))
    path.write_text(
        "".join(f"{record}\n" for record in records),
        encoding="utf-8",
    )


def _compatibility_record(
    config: TrainingConfig,
    *,
    data: Mapping[str, object],
    model: Mapping[str, object],
) -> dict[str, object]:
    training_config = config.model_dump(
        mode="json",
        exclude={
            "checkpoint_every_steps",
            "device",
            "efficiency",
            "evaluation",
            "log_every_steps",
            "model",
            "output_directory",
            "profile_phases",
            "resume_from",
            "steps",
            "train",
            "validation",
        },
    )
    return {
        "training_config": training_config,
        "data": {
            "train": _data_compatibility(data["train"]),
            "validation": (
                _data_compatibility(data["validation"])
                if data["validation"] is not None
                else None
            ),
        },
        "model": dict(model),
        "action_vocabulary": action_vocabulary_identity(),
        "encoding": encoding_identity(),
    }


def _training_device(
    config: TrainingConfig,
    *,
    capabilities: DeviceCapabilities | None = None,
) -> torch.device:
    try:
        device = resolve_training_device(
            config.device,
            capabilities=capabilities,
        )
    except DeviceError as error:
        raise TrainingError(str(error)) from error
    if (
        config.determinism == "strict"
        and device.type not in STRICT_DETERMINISM_BACKENDS
    ):
        raise TrainingError(
            f"strict determinism is not supported for the current "
            f"{device.type.upper()} training path; select determinism='relaxed'"
        )
    return device


def _training_dtype(config: TrainingConfig) -> torch.dtype:
    """Return the dtype parameters and optimizer state are held in.

    Mixed precision does not change this. Master weights stay float32 and only
    the forward pass is autocast, so a checkpoint written under either setting
    holds the same tensors and loads on either.
    """

    if config.precision in ("float32", "bfloat16-mixed"):
        return torch.float32
    raise TrainingError(f"unsupported training precision: {config.precision}")


def _autocast_dtype(config: TrainingConfig) -> torch.dtype | None:
    """Return the dtype the forward pass is autocast to, if any."""

    return torch.bfloat16 if config.precision == "bfloat16-mixed" else None


def _autocast(
    config: TrainingConfig, device: torch.device
) -> AbstractContextManager[None]:
    """Return the autocast scope one micro-batch's forward pass runs under."""

    dtype = _autocast_dtype(config)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _fused_optimizer(device: torch.device) -> bool:
    """Return whether the optimizer update runs as one fused kernel.

    Derived from the backend rather than configured, because there is one right
    answer per backend and a dial with a single correct setting is a dial
    nobody should have to find. Adam over this model's parameter list is
    otherwise dozens of small elementwise launches, and on CUDA the launches
    cost more than the arithmetic: fusing them was worth 17.7% of a whole
    training step, which is larger than every precision and transfer option
    measured for this milestone put together.

    Elementwise and deterministic either way, so this stays compatible with the
    strict correctness path. It does reassociate some floating-point work, so
    the two forms are not bit-identical to each other — which is already true
    of any change of backend, and is why neither is a compatibility identity.
    """

    return device.type == "cuda"


#: The execution settings a continuation has to match. Each one decides what
#: the optimizer does to the weights rather than only where it happened, so a
#: checkpoint written under one value is not a checkpoint another value can
#: continue. Adding to this set changes what a stored checkpoint means, which
#: is `CHECKPOINT_VERSION`'s job.
_EXECUTION_COMPATIBILITY_KEYS = ("precision", "parameter_dtype", "determinism")

#: The execution settings recorded to describe the run that wrote a checkpoint.
#: A continuation is free to differ in every one of them, so adding one is not
#: a breaking change and a checkpoint written before one existed still resumes.
_EXECUTION_PROVENANCE_KEYS = (
    "device",
    "backend",
    "gradient_accumulation_steps",
    "fused_optimizer",
    "phase_profiling",
)


def _execution_record(
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, object]:
    """Record how one run executed, in the two roles resume distinguishes.

    Every key belongs to exactly one of `_EXECUTION_COMPATIBILITY_KEYS` and
    `_EXECUTION_PROVENANCE_KEYS`, which is what lets a later version record
    more about a run without invalidating the checkpoints already on disk.
    """

    return {
        "device": config.device,
        "backend": device.type,
        "precision": config.precision,
        "parameter_dtype": str(_training_dtype(config)).removeprefix("torch."),
        "determinism": config.determinism,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "fused_optimizer": _fused_optimizer(device),
        "phase_profiling": config.profile_phases,
    }


def _validate_checkpoint_execution(
    metadata: object,
    current: Mapping[str, object],
) -> None:
    """Reject a continuation the checkpoint's execution settings cannot support.

    Only the compatibility set is compared. The rest is provenance that this
    run never reads, including `backend`, so a checkpoint holding a key this
    version does not know, or missing one it added later, still resumes.
    """

    if not isinstance(metadata, Mapping):
        raise CheckpointError("checkpoint metadata is not a mapping")
    execution = metadata.get("execution")
    if not isinstance(execution, Mapping):
        raise CheckpointError("checkpoint has no execution metadata")
    for key in _EXECUTION_COMPATIBILITY_KEYS:
        if key not in execution:
            raise CheckpointError(f"checkpoint execution metadata has no {key}")
        if execution[key] != current[key]:
            raise CheckpointError(
                f"checkpoint execution {key} is incompatible with the current "
                f"run: checkpoint {execution[key]!r}, current {current[key]!r}"
            )


def _data_compatibility(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TrainingError("data provenance is not a mapping")
    keys = (
        "manifest_sha256",
        "dataset_sha256",
        "loader_configuration_sha256",
    )
    if any(key not in value for key in keys):
        raise TrainingError("data provenance is missing compatibility identities")
    return {key: value[key] for key in keys}


#: How each compatibility identity is named in a resume failure. An identity
#: absent from here is still compared, under its own key, so adding one to
#: `_compatibility_record` cannot leave it silently unchecked.
_COMPATIBILITY_LABELS = {
    "training_config": "training configuration",
    "data": "data",
    "model": "model",
    "action_vocabulary": "action vocabulary",
    "encoding": "model-facing encoding",
}


def _validate_checkpoint_compatibility(
    saved: object,
    current: Mapping[str, object],
) -> None:
    """Reject a resume whose code-owned identities differ from this run's.

    Unlike execution provenance, every identity here is compared, and one the
    checkpoint does not carry is an unknown rather than a match. The running
    code decides what has to agree, so an identity it does not know about is
    left alone; retiring or adding one is a `CHECKPOINT_VERSION` change.
    """

    if not isinstance(saved, Mapping):
        raise CheckpointError("checkpoint compatibility metadata is not a mapping")
    for key, value in current.items():
        label = _COMPATIBILITY_LABELS.get(key, key)
        if key not in saved:
            raise CheckpointError(f"checkpoint has no {label} compatibility identity")
        if saved[key] != value:
            raise CheckpointError(
                f"checkpoint {label} is incompatible with the current run"
            )


def _load_data_selection(
    config: SequenceDataConfig,
    *,
    legal_actions: bool,
    maximum_context_plies: int,
) -> _DataSelection:
    paths = normalized_shard_paths(config.normalized)
    manifest_path = config.manifest
    if not manifest_path.is_file():
        raise DataLoadingError(f"data manifest does not exist: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise DataLoadingError("data manifest must contain a JSON object")
    validate_manifest_compatibility(manifest, manifest_path)
    # Ahead of the output check, which hashes every shard end to end. Both
    # refuse the run, and this one reads two integers, so it is the cheaper of
    # the two to fail on.
    _reject_uncoverable_corpus(manifest, config, maximum_context_plies)
    shards = validate_manifest_outputs(manifest, manifest_path, paths)
    manifest_sha256 = sha256(manifest_bytes).hexdigest()
    loader = _open_loader(
        config,
        shards,
        manifest_sha256=manifest_sha256,
        legal_actions=legal_actions,
    )
    return _DataSelection(
        loader=loader,
        provenance={
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha256,
            "manifest": manifest,
            "normalized_paths": [str(path.resolve()) for path in paths],
            # Which loader read the corpus, because the two order an epoch
            # differently and a run's curve is not comparable across them.
            "loader": (
                STREAMING_LOADER_NAME if config.streaming is not None else "eager"
            ),
            "dataset_sha256": loader.identity_sha256,
            "loader_configuration_sha256": loader.configuration_sha256,
            # The resolved selection, not only the requested one: two runs
            # differing only in what they trained on are told apart here, and
            # the digest is what lets a later run confirm it reproduced the
            # same games rather than merely the same configuration.
            "selection": loader.resolution.as_record(),
        },
    )


def _open_loader(
    config: SequenceDataConfig,
    shards: Sequence[ShardIdentity],
    *,
    manifest_sha256: str,
    legal_actions: bool,
) -> SequenceBatchSource:
    """Open the loader this selection declared, eager or shard-backed."""

    if config.streaming is None:
        return SequenceDataLoader.from_parquet(
            [shard.path for shard in shards],
            config.loader,
            legal_actions=legal_actions,
        )
    index = build_sharded_index(
        shards,
        split=config.loader.split,
        selection=config.loader.selection,
        chunk_length=config.loader.chunk_length,
        manifest_sha256=manifest_sha256,
    )
    return StreamingSequenceDataLoader(
        index,
        config.loader,
        config.streaming,
        legal_actions=legal_actions,
    )


def _reject_uncoverable_corpus(
    manifest: dict[str, Any],
    config: SequenceDataConfig,
    maximum_context_plies: int,
) -> None:
    """Refuse a corpus the model could not encode, before a shard is read.

    The model refuses a batch reaching past the context it declares, and it
    does so from inside the micro-batch loop, which has no handler. Left there,
    one long game kills a run at whichever step it happened to land on.

    Preparation already recorded the longest game, so nothing is scanned. That
    number is corpus-wide rather than per split or per selection, which fails
    closed: this can refuse a corpus whose long games the selection would never
    have reached, and it never admits one that it would. It counts moves only,
    and the encoding appends one terminal action to a game a player ended on
    their own turn, so the longest encoded sequence is one past it.
    """

    manifest_path = config.manifest
    chunk_length = config.loader.chunk_length
    games = manifest.get("games")
    plies = games.get("plies") if isinstance(games, Mapping) else None
    longest = plies.get("maximum_per_game") if isinstance(plies, Mapping) else None
    if type(longest) is not int or longest < 1:
        raise DataLoadingError(f"{manifest_path} records no positive longest game")
    encoded = longest + 1
    bound = maximum_position_bound(encoded, chunk_length)
    if bound <= maximum_context_plies:
        return
    chunked = "" if chunk_length is None else f", chunked at {chunk_length},"
    raise DataLoadingError(
        f"{manifest_path} records a longest game of {longest} plies, which "
        f"encodes to at most {encoded}{chunked} and reaches ply index "
        f"{bound - 1}, past the {maximum_context_plies} plies the model "
        f"declares as its context"
    )
