"""Which of a run's checkpoints outlive the save after them, and removing the rest."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthro_chess.inference.config import ModelRunnerConfig
from anthro_chess.inference.selection import (
    ModelSelectionError,
    resolve_model_selection,
)
from anthro_chess.training.checkpoints import (
    CheckpointError,
    checkpoint_step,
    latest_checkpoint_path,
)


@dataclass(frozen=True)
class RetentionPolicy:
    """The checkpoints a run keeps once a later one supersedes them."""

    final_step: int
    #: Unset keeps every checkpoint the run saves.
    keep_every_steps: int | None = None
    keep_steps: frozenset[int] = frozenset()

    @classmethod
    def declared_by(cls, config: Mapping[str, Any]) -> RetentionPolicy:
        """Read the policy a run's resolved training configuration declares.

        From the raw record rather than through the configuration schema, so a
        run recorded before these settings existed reads as keeping everything
        instead of failing to validate.
        """

        return cls(
            final_step=config["steps"],
            keep_every_steps=config.get("keep_every_steps"),
            keep_steps=frozenset(config.get("keep_steps", ())),
        )

    def requires(self, global_step: int) -> bool:
        """Return whether the run has to save at this step whatever its cadence."""

        return (
            global_step == self.final_step
            or global_step in self.keep_steps
            or (
                self.keep_every_steps is not None
                and global_step % self.keep_every_steps == 0
            )
        )

    def retains(self, global_step: int) -> bool:
        """Return whether a checkpoint saved at this step outlives the next save."""

        return self.keep_every_steps is None or self.requires(global_step)


def protected_checkpoints(run_root: Path) -> frozenset[Path]:
    """Return the checkpoints something outside their own run still reads.

    A branch's record names the trunk checkpoint it resumed from, and the default
    model selection can pin a checkpoint by name. A run that resumed from its
    own checkpoint protects nothing, because it has moved past it. Only runs
    beneath ``run_root`` are seen.
    """

    protected: set[Path] = set()
    for record_path in run_root.glob("*/run.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            resumed_from = record["optimization"]["resumed_from"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        if not isinstance(resumed_from, str):
            continue
        source = Path(resumed_from).resolve()
        if source.parent.parent != record_path.parent.resolve():
            protected.add(source)
    try:
        selected = resolve_model_selection(ModelRunnerConfig(), run_root=run_root)
    except ModelSelectionError:
        pass
    else:
        protected.add(selected.checkpoint_path.resolve())
    return frozenset(protected)


def remove_checkpoint(path: Path, *, protected: Collection[Path]) -> bool:
    """Delete one superseded checkpoint unless another reader still needs it."""

    if path.resolve() in protected:
        return False
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        raise CheckpointError(f"cannot remove checkpoint {path}: {error}") from error
    return True


@dataclass(frozen=True)
class RunPruning:
    """What pruning one run to its policy removes, and what it holds back."""

    run_path: Path
    removable: tuple[Path, ...] = ()
    #: Checkpoints the policy would drop that another reader still needs.
    held: tuple[Path, ...] = ()
    #: Why the run was left alone entirely, when it was.
    skipped: str | None = None

    @property
    def removable_bytes(self) -> int:
        """Return the disk space removing every listed checkpoint frees."""

        return sum(path.stat().st_size for path in self.removable)


def plan_pruning(
    run_path: Path,
    *,
    protected: Collection[Path],
    keep_every_steps: int | None = None,
    keep_steps: Collection[int] = (),
) -> RunPruning:
    """Work out which of a finished or paused run's checkpoints its policy drops.

    ``keep_every_steps`` applies only where the run's record declares no
    cadence of its own, and ``keep_steps`` adds to the steps it declares. Only
    checkpoints below the one ``latest.json`` names are candidates: a live run
    writes above it, and a save is not complete until the pointer moves.
    """

    try:
        record = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
        declared = RetentionPolicy.declared_by(record["resolved_config"]["config"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return RunPruning(run_path, skipped="no readable run record")
    try:
        latest = latest_checkpoint_path(run_path)
    except CheckpointError:
        return RunPruning(run_path, skipped="no latest checkpoint")
    latest_step = checkpoint_step(latest)
    assert latest_step is not None  # the latest pointer only names valid steps
    policy = RetentionPolicy(
        final_step=declared.final_step,
        keep_every_steps=(
            declared.keep_every_steps
            if declared.keep_every_steps is not None
            else keep_every_steps
        ),
        keep_steps=declared.keep_steps | frozenset(keep_steps),
    )
    removable: list[Path] = []
    held: list[Path] = []
    for path in sorted(latest.parent.glob("step-*.pt")):
        step = checkpoint_step(path)
        if step is None or step >= latest_step or policy.retains(step):
            continue
        (held if path.resolve() in protected else removable).append(path)
    return RunPruning(run_path, removable=tuple(removable), held=tuple(held))
