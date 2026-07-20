"""Executable CPU training over the ordinary data, model, and loss boundaries."""

from __future__ import annotations

import json
import random
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from anthro_chess import __version__
from anthro_chess.chess import action_vocabulary_identity
from anthro_chess.config import ResolvedConfig
from anthro_chess.data import (
    DataLoadingError,
    SequenceDataConfig,
    SequenceDataLoader,
    encoding_identity,
)
from anthro_chess.data.schema import PREPROCESSING_VERSION, SCHEMA_VERSION
from anthro_chess.evaluation import MoveValidationMetrics, evaluate_move_model
from anthro_chess.models import CausalMoveModel, MoveModelBatch
from anthro_chess.training.checkpoints import (
    CheckpointError,
    checkpoint_path,
    clear_latest_checkpoint,
    latest_checkpoint_path,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from anthro_chess.training.config import TrainingConfig
from anthro_chess.training.losses import masked_action_cross_entropy

RUN_ARTIFACT_VERSION = 2


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


@dataclass(frozen=True)
class _DataSelection:
    loader: SequenceDataLoader
    provenance: dict[str, object]


@dataclass(frozen=True)
class _OptimizationResult:
    processed_positions: int
    checkpoint_path: Path


def run_training(
    resolved_config: ResolvedConfig[TrainingConfig],
) -> TrainingResult:
    """Run bounded CPU or MPS optimization and write its provenance."""

    config = resolved_config.value
    device = _training_device(config)
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
        code_record = {
            "package_version": __version__,
            "git_revision": _git_revision(),
        }
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

        initial_parameter_sha256 = _parameter_sha256(model)
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
            device=device,
            metrics_path=metrics_path,
            output_directory=output_directory,
            compatibility=compatibility,
            checkpoint_metadata=checkpoint_metadata,
        )
        final_parameter_sha256 = _parameter_sha256(model)
        if final_parameter_sha256 == initial_parameter_sha256:
            raise TrainingError("optimizer completed without changing model parameters")

        validation_metrics = (
            evaluate_move_model(
                model,
                validation.loader,
                device=device,
            )
            if validation is not None
            else None
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
        }
        run_path.write_text(
            json.dumps(run_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
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

    return TrainingResult(
        run_path=run_path,
        metrics_path=metrics_path,
        checkpoint_path=optimization.checkpoint_path,
        steps=config.steps,
        initial_parameter_sha256=initial_parameter_sha256,
        final_parameter_sha256=final_parameter_sha256,
        validation=validation_metrics,
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
    device: torch.device,
    metrics_path: Path,
    output_directory: Path,
    compatibility: Mapping[str, object],
    checkpoint_metadata: Mapping[str, object],
) -> _OptimizationResult:
    model.train()
    start_time = time.perf_counter()
    saved_checkpoint: Path | None = None
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for global_step in range(starting_step + 1, steps + 1):
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

            batch = MoveModelBatch.from_sequence_batch(
                sequence_batch,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = masked_action_cross_entropy(
                model(batch),
                batch.action_targets,
                batch.action_loss_mask,
            )
            if not torch.isfinite(loss):
                raise TrainingError(
                    f"move loss is not finite at global step {global_step}"
                )
            loss.backward()
            optimizer.step()
            positions = int(batch.action_loss_mask.sum().item())
            processed_positions += positions
            epoch = loader.state().epoch

            if global_step % log_every_steps == 0 or global_step == steps:
                elapsed = max(time.perf_counter() - start_time, 1e-12)
                record = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "move_loss": float(loss.detach().item()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "batch_positions": positions,
                    "processed_positions": processed_positions,
                    "positions_per_second": processed_positions / elapsed,
                }
                metrics_file.write(
                    json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                )
                metrics_file.flush()
                print(
                    "step={global_step} move_loss={move_loss:.6f} "
                    "lr={learning_rate:.6g} positions={processed_positions} "
                    "positions_per_second={positions_per_second:.2f}".format(**record)
                )
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
    if saved_checkpoint is None:
        raise TrainingError("training completed without saving a checkpoint")
    return _OptimizationResult(
        processed_positions=processed_positions,
        checkpoint_path=saved_checkpoint,
    )


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
            "log_every_steps",
            "model",
            "output_directory",
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


def _training_device(config: TrainingConfig) -> torch.device:
    device = torch.device(config.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        availability = (
            "the installed Torch build has no MPS support"
            if not torch.backends.mps.is_built()
            else "MPS is not available on this host"
        )
        raise TrainingError(f"training device mps was requested but {availability}")
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
        "device": str(device),
        "backend": device.type,
        "precision": config.precision,
        "parameter_dtype": str(_training_dtype(config)).removeprefix("torch."),
        "determinism": config.determinism,
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
    paths = _normalized_paths(config.normalized)
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


def _normalized_paths(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    if path.is_dir():
        paths = tuple(sorted(path.glob("games*.parquet")))
        if paths:
            return paths
        raise DataLoadingError(
            f"normalized data directory has no games*.parquet files: {path}"
        )
    raise DataLoadingError(f"normalized data path does not exist: {path}")


def _validate_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    paths: tuple[Path, ...],
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DataLoadingError(
            f"{manifest_path} uses normalized schema version "
            f"{manifest.get('schema_version')}; expected {SCHEMA_VERSION}"
        )
    if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise DataLoadingError(
            f"{manifest_path} uses preprocessing version "
            f"{manifest.get('preprocessing_version')}; "
            f"expected {PREPROCESSING_VERSION}"
        )
    if manifest.get("action_vocabulary") != action_vocabulary_identity():
        raise DataLoadingError(
            f"{manifest_path} uses an incompatible action vocabulary"
        )

    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise DataLoadingError(f"{manifest_path} has no output record")
    shards = output.get("shards")
    if not isinstance(shards, list):
        raise DataLoadingError(f"{manifest_path} has no output shard records")
    expected: dict[Path, str] = {}
    artifact_root = manifest_path.parent.parent
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise DataLoadingError(f"{manifest_path} has an invalid output shard")
        relative_path = shard.get("path")
        digest = shard.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(digest, str):
            raise DataLoadingError(f"{manifest_path} has an invalid output shard")
        expected[(artifact_root / relative_path).resolve()] = digest

    configured = {path.resolve() for path in paths}
    if configured != set(expected):
        raise DataLoadingError(
            "configured normalized paths do not match the data manifest outputs"
        )
    for path in paths:
        observed = _file_sha256(path)
        if observed != expected[path.resolve()]:
            raise DataLoadingError(f"normalized data checksum mismatch: {path}")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_sha256(model: CausalMoveModel) -> str:
    digest = sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        value = parameter.detach().cpu().contiguous()
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None
