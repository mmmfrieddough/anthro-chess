"""Resolution of complete retained runs and their selected checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from anthro_chess.inference.config import LATEST_CHECKPOINT, ModelRunnerConfig
from anthro_chess.machine import ROOT_CONTENTS, RUN_ROOT_VARIABLE
from anthro_chess.training.checkpoints import CheckpointError, latest_checkpoint_path

MODEL_SELECTION_VERSION = 1
MODEL_SELECTION_FILE = "selected-model.json"


class ModelSelectionError(ValueError):
    """Raised when an inference artifact selection cannot be resolved safely."""


@dataclass(frozen=True)
class ResolvedModelSelection:
    """Absolute retained-run and checkpoint paths used by a model runner."""

    run_path: Path
    run_record_path: Path
    checkpoint_path: Path
    source: Literal["explicit-checkpoint", "explicit-run", "default-selection"]

    def as_record(self) -> dict[str, str]:
        """Return resolved paths suitable for runtime provenance."""

        return {
            "source": self.source,
            "run_path": str(self.run_path),
            "run_record_path": str(self.run_record_path),
            "checkpoint_path": str(self.checkpoint_path),
        }


def resolve_model_selection(
    config: ModelRunnerConfig,
    *,
    run_root: Path | None = None,
) -> ResolvedModelSelection:
    """Resolve explicit inputs or the intentional machine-local selection."""

    if config.checkpoint_path is not None:
        checkpoint_path = config.checkpoint_path.expanduser().resolve()
        if checkpoint_path.parent.name != "checkpoints":
            raise ModelSelectionError(
                "explicit checkpoint must be retained beneath a run's "
                "checkpoints directory"
            )
        return _resolved(
            checkpoint_path.parent.parent,
            checkpoint_path,
            source="explicit-checkpoint",
        )

    if config.run_path is not None:
        run_path = _resolve_explicit_run(config.run_path, run_root)
        return _resolved(
            run_path,
            _checkpoint_in_run(run_path, config.checkpoint),
            source="explicit-run",
        )

    root = _required_root(run_root)
    record_path = root / MODEL_SELECTION_FILE
    record = _load_selection_record(record_path)
    run_path = _resolve_beneath(root, Path(record["run"]), label="selected run")
    return _resolved(
        run_path,
        _checkpoint_in_run(run_path, record["checkpoint"]),
        source="default-selection",
    )


def write_model_selection(
    run_root: Path,
    *,
    run: str,
    checkpoint: str = LATEST_CHECKPOINT,
) -> Path:
    """Atomically maintain the intentional default model selection record."""

    root = run_root.expanduser().resolve()
    run_path = _resolve_beneath(root, Path(run), label="selected run")
    _validate_checkpoint_name(checkpoint)
    _resolved(
        run_path,
        _checkpoint_in_run(run_path, checkpoint),
        source="default-selection",
    )
    record_path = root / MODEL_SELECTION_FILE
    temporary = record_path.with_name(f".{record_path.name}.tmp")
    try:
        root.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(
                {
                    "version": MODEL_SELECTION_VERSION,
                    "run": run,
                    "checkpoint": checkpoint,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(record_path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise ModelSelectionError(
            f"cannot write model selection {record_path}: {error}"
        ) from error
    return record_path


def _load_selection_record(path: Path) -> dict[str, str]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelSelectionError(
            f"cannot resolve default model selection from {path}: {error}"
        ) from error
    if not isinstance(loaded, dict) or set(loaded) != {
        "version",
        "run",
        "checkpoint",
    }:
        raise ModelSelectionError(f"model selection record is invalid: {path}")
    if (
        type(loaded["version"]) is not int
        or loaded["version"] != MODEL_SELECTION_VERSION
    ):
        raise ModelSelectionError(
            f"unsupported model selection version: {loaded['version']}"
        )
    run = loaded["run"]
    checkpoint = loaded["checkpoint"]
    if not isinstance(run, str) or not run or Path(run).is_absolute():
        raise ModelSelectionError("selected run must be a nonempty relative path")
    if not isinstance(checkpoint, str):
        raise ModelSelectionError("selected checkpoint must be a string")
    _validate_checkpoint_name(checkpoint)
    return {"run": run, "checkpoint": checkpoint}


def _resolve_explicit_run(path: Path, run_root: Path | None) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return _resolve_beneath(
        _required_root(run_root),
        expanded,
        label="configured run",
    )


def _required_root(run_root: Path | None) -> Path:
    """Return the run root, or fail naming the configuration that is missing.

    Every caller resolves this root from the environment, so an unset variable
    presents here as an artifact that could not be found. Naming the variable
    is what separates a misconfigured machine from one that genuinely holds no
    runs; the two are otherwise identical from inside a checkout.
    """

    if run_root is None:
        raise ModelSelectionError(
            "a relative or default model selection needs a run root: pass an "
            f"explicit checkpoint path, or set {RUN_ROOT_VARIABLE} to the "
            f"directory holding {ROOT_CONTENTS[RUN_ROOT_VARIABLE]}"
        )
    return run_root.expanduser().resolve()


def _resolve_beneath(root: Path, relative: Path, *, label: str) -> Path:
    if relative.is_absolute():
        raise ModelSelectionError(f"{label} must be relative to the run root")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ModelSelectionError(f"{label} escapes the run root") from error
    return resolved


def _checkpoint_in_run(run_path: Path, name: str) -> Path:
    _validate_checkpoint_name(name)
    if name == LATEST_CHECKPOINT:
        try:
            return latest_checkpoint_path(run_path).resolve()
        except CheckpointError as error:
            raise ModelSelectionError(str(error)) from error
    return (run_path / "checkpoints" / name).resolve()


def _validate_checkpoint_name(name: str) -> None:
    if name == LATEST_CHECKPOINT:
        return
    try:
        ModelRunnerConfig(checkpoint=name)
    except ValueError as error:
        raise ModelSelectionError(str(error)) from error


def _resolved(
    run_path: Path,
    checkpoint_path: Path,
    *,
    source: Literal["explicit-checkpoint", "explicit-run", "default-selection"],
) -> ResolvedModelSelection:
    run_path = run_path.resolve()
    run_record_path = run_path / "run.json"
    if not run_path.is_dir():
        raise ModelSelectionError(f"selected run does not exist: {run_path}")
    if not run_record_path.is_file():
        raise ModelSelectionError(f"selected run has no run record: {run_record_path}")
    if not checkpoint_path.is_file():
        raise ModelSelectionError(
            f"selected checkpoint does not exist: {checkpoint_path}"
        )
    try:
        checkpoint_path.relative_to(run_path / "checkpoints")
    except ValueError as error:
        raise ModelSelectionError(
            "selected checkpoint is outside the retained run"
        ) from error
    return ResolvedModelSelection(
        run_path=run_path,
        run_record_path=run_record_path,
        checkpoint_path=checkpoint_path,
        source=source,
    )
