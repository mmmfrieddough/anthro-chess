"""Executable CPU training over the ordinary data, model, and loss boundaries."""

from __future__ import annotations

import json
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
from anthro_chess.training.config import TrainingConfig
from anthro_chess.training.losses import masked_action_cross_entropy

RUN_ARTIFACT_VERSION = 1


class TrainingError(ValueError):
    """Raised when a configured training run cannot execute safely."""


@dataclass(frozen=True)
class TrainingResult:
    """Paths and summary values produced by one training run."""

    run_path: Path
    metrics_path: Path
    steps: int
    initial_parameter_sha256: str
    final_parameter_sha256: str
    validation: MoveValidationMetrics | None


@dataclass(frozen=True)
class _DataSelection:
    loader: SequenceDataLoader
    provenance: dict[str, object]


def run_training(
    resolved_config: ResolvedConfig[TrainingConfig],
) -> TrainingResult:
    """Run a bounded deterministic CPU optimization and write its provenance."""

    config = resolved_config.value
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

    torch.manual_seed(config.seed)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = CausalMoveModel(config.model).to(config.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
        )
        initial_parameter_sha256 = _parameter_sha256(model)
        _optimize(
            model,
            optimizer,
            train.loader,
            steps=config.steps,
            log_every_steps=config.log_every_steps,
            device=config.device,
            metrics_path=metrics_path,
        )
        final_parameter_sha256 = _parameter_sha256(model)
        if final_parameter_sha256 == initial_parameter_sha256:
            raise TrainingError("optimizer completed without changing model parameters")

        validation_metrics = (
            evaluate_move_model(
                model,
                validation.loader,
                device=config.device,
            )
            if validation is not None
            else None
        )
        run_record = {
            "version": RUN_ARTIFACT_VERSION,
            "resolved_config": resolved_config.as_record(),
            "seed": config.seed,
            "code": {
                "package_version": __version__,
                "git_revision": _git_revision(),
            },
            "data": {
                "train": train.provenance,
                "validation": (
                    validation.provenance if validation is not None else None
                ),
            },
            "model": model.identity(),
            "action_vocabulary": action_vocabulary_identity(),
            "encoding": encoding_identity(),
            "optimization": {
                "optimizer": "Adam",
                "completed_steps": config.steps,
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
    except (OSError, RuntimeError, ValueError) as error:
        if isinstance(error, TrainingError):
            raise
        raise TrainingError(f"training failed: {error}") from error
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)

    return TrainingResult(
        run_path=run_path,
        metrics_path=metrics_path,
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
    log_every_steps: int,
    device: str,
    metrics_path: Path,
) -> None:
    epoch = 0
    model.train()
    start_time = time.perf_counter()
    processed_positions = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for global_step in range(1, steps + 1):
            try:
                sequence_batch = next(loader)
            except StopIteration:
                epoch += 1
                loader.start_epoch(epoch)
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
