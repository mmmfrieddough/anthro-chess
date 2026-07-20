"""Versioned, step-keyed persistence for restartable training."""

from __future__ import annotations

import json
import os
import pickle
import random
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

CHECKPOINT_VERSION = 1
_CHECKPOINT_DIRECTORY = "checkpoints"
_CHECKPOINT_PATTERN = re.compile(r"^step-(\d{8})\.pt$")
_LATEST_RECORD = "latest.json"
_REQUIRED_KEYS = {
    "version",
    "global_step",
    "counters",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "scaler_state",
    "loader_state",
    "rng_state",
    "compatibility",
    "metadata",
}


class CheckpointError(ValueError):
    """Raised when a training checkpoint cannot be saved or restored safely."""


def checkpoint_path(output_directory: Path, global_step: int) -> Path:
    """Return the canonical path for one optimizer-step checkpoint."""

    if type(global_step) is not int or global_step < 1:
        raise CheckpointError("checkpoint global step must be a positive integer")
    return output_directory / _CHECKPOINT_DIRECTORY / f"step-{global_step:08d}.pt"


def latest_checkpoint_path(output_directory: Path) -> Path:
    """Resolve the last atomically completed checkpoint for a run directory."""

    directory = output_directory / _CHECKPOINT_DIRECTORY
    latest = directory / _LATEST_RECORD
    try:
        record = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(
            f"cannot resolve latest checkpoint from {latest}: {error}"
        ) from error
    if not isinstance(record, dict) or set(record) != {"global_step", "path"}:
        raise CheckpointError(f"latest checkpoint record is invalid: {latest}")
    global_step = record["global_step"]
    name = record["path"]
    if (
        type(global_step) is not int
        or not isinstance(name, str)
        or _CHECKPOINT_PATTERN.fullmatch(name) is None
        or name != f"step-{global_step:08d}.pt"
    ):
        raise CheckpointError(f"latest checkpoint record is invalid: {latest}")
    path = directory / name
    if not path.is_file():
        raise CheckpointError(f"latest checkpoint does not exist: {path}")
    return path


def clear_latest_checkpoint(output_directory: Path) -> None:
    """Invalidate a prior run's latest pointer before a fresh optimization."""

    try:
        (output_directory / _CHECKPOINT_DIRECTORY / _LATEST_RECORD).unlink(
            missing_ok=True
        )
    except OSError as error:
        raise CheckpointError(
            f"cannot reset latest checkpoint in {output_directory}: {error}"
        ) from error


def save_training_checkpoint(
    path: Path,
    *,
    global_step: int,
    counters: Mapping[str, int],
    model_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    loader_state: Mapping[str, object],
    compatibility: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    """Atomically save all state needed by the current CPU training path."""

    if type(global_step) is not int or global_step < 1:
        raise CheckpointError("checkpoint global step must be a positive integer")
    payload = {
        "version": CHECKPOINT_VERSION,
        "global_step": global_step,
        "counters": dict(counters),
        "model_state": dict(model_state),
        "optimizer_state": dict(optimizer_state),
        "scheduler_state": None,
        "scaler_state": None,
        "loader_state": dict(loader_state),
        "rng_state": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
        },
        "compatibility": dict(compatibility),
        "metadata": dict(metadata),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, temporary)
        os.replace(temporary, path)
        latest = path.parent / _LATEST_RECORD
        latest_temporary = latest.with_name(f".{latest.name}.tmp")
        latest_temporary.write_text(
            json.dumps(
                {"global_step": global_step, "path": path.name},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(latest_temporary, latest)
    except (OSError, RuntimeError) as error:
        try:
            temporary.unlink(missing_ok=True)
            (path.parent / f".{_LATEST_RECORD}.tmp").unlink(missing_ok=True)
        except OSError:
            pass
        raise CheckpointError(f"cannot save checkpoint {path}: {error}") from error


def load_training_checkpoint(path: Path) -> dict[str, Any]:
    """Load and structurally validate a checkpoint without executing pickle code."""

    if not path.is_file():
        raise CheckpointError(f"checkpoint does not exist: {path}")
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise CheckpointError(f"cannot load checkpoint {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise CheckpointError(f"checkpoint {path} does not contain a state record")
    payload = cast(dict[str, Any], loaded)
    if set(payload) != _REQUIRED_KEYS:
        raise CheckpointError(
            f"checkpoint {path} has incomplete or unknown top-level fields"
        )
    if payload["version"] != CHECKPOINT_VERSION:
        raise CheckpointError(f"unsupported checkpoint version: {payload['version']}")
    global_step = payload["global_step"]
    if type(global_step) is not int or global_step < 1:
        raise CheckpointError("checkpoint global step must be a positive integer")
    for key in (
        "counters",
        "model_state",
        "optimizer_state",
        "loader_state",
        "rng_state",
        "compatibility",
        "metadata",
    ):
        if not isinstance(payload[key], Mapping):
            raise CheckpointError(f"checkpoint {key} must be a mapping")
    if payload["scheduler_state"] is not None:
        raise CheckpointError(
            "checkpoint scheduler state is unsupported by this training runner"
        )
    if payload["scaler_state"] is not None:
        raise CheckpointError(
            "checkpoint scaler state is unsupported by this training runner"
        )
    return payload


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore the Python and CPU Torch random-number-generator states."""

    if set(state) != {"python", "torch_cpu"}:
        raise CheckpointError("checkpoint RNG state is incomplete or unknown")
    python_state = state["python"]
    torch_state = state["torch_cpu"]
    if not isinstance(python_state, tuple):
        raise CheckpointError("checkpoint Python RNG state is invalid")
    if not isinstance(torch_state, torch.Tensor):
        raise CheckpointError("checkpoint Torch RNG state is invalid")
    try:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
    except (TypeError, ValueError, RuntimeError) as error:
        raise CheckpointError(f"checkpoint RNG state is invalid: {error}") from error
