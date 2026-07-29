"""Executable device-aware training over the ordinary package boundaries."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from anthro_chess.chess import action_vocabulary_identity
from anthro_chess.config import ResolvedConfig
from anthro_chess.data import (
    DataLoadingError,
    SequenceDataConfig,
    SequenceDataLoader,
    encoding_identity,
)
from anthro_chess.data.artifacts import (
    normalized_shard_paths,
    validate_manifest_compatibility,
    validate_manifest_outputs,
)
from anthro_chess.evaluation import MoveValidationMetrics, evaluate_move_model
from anthro_chess.evaluation.results import (
    CheckpointReference,
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
    DeviceCapabilities,
    DeviceError,
    resolve_training_device,
)
from anthro_chess.training.health import StepHealth, StepHealthMonitor
from anthro_chess.training.losses import masked_action_cross_entropy
from anthro_chess.training.tensorboard import (
    TENSORBOARD_DIRECTORY,
    TrainingTensorBoard,
)

RUN_ARTIFACT_VERSION = 4
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


@dataclass(frozen=True)
class _DataSelection:
    loader: SequenceDataLoader
    provenance: dict[str, object]


@dataclass(frozen=True)
class _OptimizationResult:
    processed_positions: int
    checkpoint_path: Path
    readings: tuple[CadenceReading, ...]
    instrumentation_seconds: float


def run_training(
    resolved_config: ResolvedConfig[TrainingConfig],
    *,
    store: ResultsStore | None = None,
) -> TrainingResult:
    """Run a bounded optimization on the resolved device and write provenance.

    Passing no ``store`` runs any declared cadence and records nothing, which
    is what an exploratory run wants: committed history should hold readings
    somebody meant to keep.
    """

    config = resolved_config.value
    device = _training_device(config)
    logger.info(
        "Starting training on %s for %s optimizer step(s)",
        device.type,
        config.steps,
    )
    try:
        train = _load_data_selection(config.train)
        validation = (
            _load_data_selection(config.validation)
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

        schedule = prepare_schedule(
            config.evaluation,
            config.validation,
            configuration=configuration_reference(
                resolved_config.as_record(),
                source=resolved_config.provenance.source,
                overrides=resolved_config.provenance.overrides,
            ),
            store=store if config.evaluation.record else None,
        )

        initial_parameter_sha256 = parameter_sha256(model)
        if resumed_from is None:
            clear_latest_checkpoint(output_directory)
        _prepare_metrics(metrics_path, through_step=starting_step)
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
            profile_phases=config.profile_phases,
            metrics_path=metrics_path,
            output_directory=output_directory,
            compatibility=compatibility,
            checkpoint_metadata=checkpoint_metadata,
            schedule=schedule,
            run_id=output_directory.name,
        )
        _synchronize_device(device)
        final_parameter_sha256 = parameter_sha256(model)
        if final_parameter_sha256 == initial_parameter_sha256:
            raise TrainingError("optimizer completed without changing model parameters")

        if validation is not None:
            logger.info("Running validation")
            validation_metrics = evaluate_move_model(
                model,
                validation.loader,
                device=device,
            )
        else:
            validation_metrics = None
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
        }
        run_path.write_text(
            json.dumps(run_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
    )


def _optimize(
    model: CausalMoveModel,
    optimizer: torch.optim.Optimizer,
    loader: SequenceDataLoader,
    *,
    steps: int,
    starting_step: int,
    processed_positions: int,
    log_every_steps: int,
    checkpoint_every_steps: int,
    gradient_accumulation_steps: int,
    device: torch.device,
    profile_phases: bool,
    metrics_path: Path,
    output_directory: Path,
    compatibility: Mapping[str, object],
    checkpoint_metadata: Mapping[str, object],
    schedule: CadenceSchedule,
    run_id: str,
) -> _OptimizationResult:
    model.train()
    start_time = time.perf_counter()
    saved_checkpoint: Path | None = None
    measured_positions = 0
    data_seconds = 0.0
    transfer_seconds = 0.0
    compute_seconds = 0.0
    health_monitor = StepHealthMonitor(model.parameters())
    readings: list[CadenceReading] = []
    evaluation_seconds = 0.0
    peak_sampled_allocated_memory_bytes = _allocated_memory_bytes(device)
    peak_sampled_driver_memory_bytes = _driver_allocated_memory_bytes(device)
    purge_step = starting_step + 1 if starting_step > 0 else None
    with (
        TrainingTensorBoard(
            output_directory / TENSORBOARD_DIRECTORY,
            purge_step=purge_step,
        ) as tensorboard,
        metrics_path.open("a", encoding="utf-8") as metrics_file,
    ):
        for global_step in range(starting_step + 1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            step_positions = 0
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
                loss = masked_action_cross_entropy(
                    model(batch),
                    batch.action_targets,
                    batch.action_loss_mask,
                )
                if not torch.isfinite(loss):
                    raise TrainingError(
                        f"move loss is not finite at global step {global_step}"
                    )
                (loss / gradient_accumulation_steps).backward()
                if profile_phases:
                    _synchronize_device(device)
                compute_seconds += time.perf_counter() - compute_started
                positions = int(batch.action_loss_mask.sum().item())
                step_positions += positions
                accumulated_loss += float(loss.detach().item())

            due = schedule.due(global_step)
            reported = (
                global_step % log_every_steps == 0 or global_step == steps or bool(due)
            )
            health_monitor.observe_gradients()
            if reported:
                health_monitor.snapshot_parameters()

            compute_started = time.perf_counter()
            optimizer.step()
            if profile_phases:
                _synchronize_device(device)
            compute_seconds += time.perf_counter() - compute_started
            health_monitor.observe_update()
            processed_positions += step_positions
            measured_positions += step_positions
            peak_sampled_allocated_memory_bytes = _maximum_optional(
                peak_sampled_allocated_memory_bytes,
                _allocated_memory_bytes(device),
            )
            peak_sampled_driver_memory_bytes = _maximum_optional(
                peak_sampled_driver_memory_bytes,
                _driver_allocated_memory_bytes(device),
            )
            epoch = loader.state().epoch

            health: StepHealth | None = None
            if reported:
                _synchronize_device(device)
                health = health_monitor.drain(global_step)
                elapsed = max(time.perf_counter() - start_time, 1e-12)
                average_loss = accumulated_loss / gradient_accumulation_steps
                learning_rate = float(optimizer.param_groups[0]["lr"])
                # Throughput has to describe training, so time spent inside a
                # cadence reading comes out of the denominator. Leaving it in
                # would report a run as several times slower for no reason
                # other than that it measured itself on the way past.
                training_seconds = max(elapsed - evaluation_seconds, 1e-12)
                positions_per_second = measured_positions / training_seconds
                record: dict[str, object] = {
                    "record": "step",
                    "global_step": global_step,
                    "epoch": epoch,
                    "move_loss": average_loss,
                    "learning_rate": learning_rate,
                    "batch_positions": step_positions,
                    "processed_positions": processed_positions,
                    "positions_per_second": positions_per_second,
                    "elapsed_seconds": elapsed,
                    "evaluation_seconds": evaluation_seconds,
                    "data_seconds": data_seconds if profile_phases else None,
                    "transfer_seconds": (transfer_seconds if profile_phases else None),
                    "compute_seconds": compute_seconds if profile_phases else None,
                    "peak_sampled_allocated_memory_bytes": (
                        peak_sampled_allocated_memory_bytes
                    ),
                    "peak_sampled_driver_memory_bytes": (
                        peak_sampled_driver_memory_bytes
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
                    f"positions_per_second={positions_per_second:.2f}"
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
                    evaluation_seconds += reading.seconds
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
            if global_step % checkpoint_every_steps == 0 or global_step == steps:
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
                logger.info("Saved checkpoint at optimizer step %s", global_step)
    if saved_checkpoint is None:
        raise TrainingError("training completed without saving a checkpoint")
    return _OptimizationResult(
        processed_positions=processed_positions,
        checkpoint_path=saved_checkpoint,
        readings=tuple(readings),
        instrumentation_seconds=health_monitor.instrumentation_seconds,
    )


def _synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _allocated_memory_bytes(device: torch.device) -> int | None:
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    return None


def _driver_allocated_memory_bytes(device: torch.device) -> int | None:
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    return None


def _maximum_optional(current: int | None, observed: int | None) -> int | None:
    if current is None:
        return observed
    if observed is None:
        return current
    return max(current, observed)


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
    if device.type == "mps" and config.determinism == "strict":
        raise TrainingError(
            "strict determinism is not supported for the current MPS "
            "training path; select determinism='relaxed'"
        )
    return device


def _training_dtype(config: TrainingConfig) -> torch.dtype:
    if config.precision == "float32":
        return torch.float32
    raise TrainingError(f"unsupported training precision: {config.precision}")


def _execution_record(
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, object]:
    return {
        "device": config.device,
        "backend": device.type,
        "precision": config.precision,
        "parameter_dtype": str(_training_dtype(config)).removeprefix("torch."),
        "determinism": config.determinism,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "phase_profiling": config.profile_phases,
    }


def _validate_checkpoint_execution(
    metadata: object,
    current: Mapping[str, object],
) -> None:
    if not isinstance(metadata, Mapping):
        raise CheckpointError("checkpoint metadata is not a mapping")
    execution = metadata.get("execution")
    if not isinstance(execution, Mapping):
        raise CheckpointError("checkpoint has no execution metadata")
    required = {
        "device",
        "backend",
        "precision",
        "parameter_dtype",
        "determinism",
        "gradient_accumulation_steps",
        "phase_profiling",
    }
    if set(execution) != required:
        raise CheckpointError("checkpoint execution metadata is incomplete or unknown")
    if execution["backend"] not in {"cpu", "mps"}:
        raise CheckpointError(
            f"checkpoint execution backend is unsupported: {execution['backend']}"
        )
    for key in ("precision", "parameter_dtype", "determinism"):
        if execution[key] != current[key]:
            raise CheckpointError(
                f"checkpoint execution {key} is incompatible with the current run"
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


def _validate_checkpoint_compatibility(
    saved: object,
    current: Mapping[str, object],
) -> None:
    if not isinstance(saved, Mapping):
        raise CheckpointError("checkpoint compatibility metadata is not a mapping")
    if set(saved) != set(current):
        raise CheckpointError(
            "checkpoint compatibility metadata is incomplete or unknown"
        )
    labels = {
        "training_config": "training configuration",
        "data": "data",
        "model": "model",
        "action_vocabulary": "action vocabulary",
        "encoding": "model-facing encoding",
    }
    for key, label in labels.items():
        if saved[key] != current[key]:
            raise CheckpointError(
                f"checkpoint {label} is incompatible with the current run"
            )


def _load_data_selection(config: SequenceDataConfig) -> _DataSelection:
    paths = normalized_shard_paths(config.normalized)
    manifest_path = config.manifest
    if not manifest_path.is_file():
        raise DataLoadingError(f"data manifest does not exist: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise DataLoadingError("data manifest must contain a JSON object")
    _validate_manifest(manifest, manifest_path, paths)
    loader = SequenceDataLoader.from_parquet(paths, config.loader)
    return _DataSelection(
        loader=loader,
        provenance={
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_bytes).hexdigest(),
            "manifest": manifest,
            "normalized_paths": [str(path.resolve()) for path in paths],
            "dataset_sha256": loader.dataset.identity_sha256,
            "loader_configuration_sha256": loader.configuration_sha256,
        },
    )


def _validate_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    paths: tuple[Path, ...],
) -> None:
    validate_manifest_compatibility(manifest, manifest_path)
    validate_manifest_outputs(manifest, manifest_path, paths)
