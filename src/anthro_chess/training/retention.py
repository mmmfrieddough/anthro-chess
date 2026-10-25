"""Which of a run's checkpoints outlive the save after them, and removing the rest."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

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

#: Beside a run's checkpoints, one note per run branched from them.
BRANCHES_DIRECTORY = "branches"


@dataclass(frozen=True)
class RetentionPolicy:
    """The checkpoints a run keeps once a later one supersedes them."""

    final_step: int
    #: Unset keeps every checkpoint the run saves.
    keep_every_steps: int | None = None
    keep_steps: frozenset[int] = frozenset()

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


def branch_point(run_path: Path, resumed_from: Path | None) -> Path | None:
    """Return the checkpoint in another run that a run's weights continue from.

    A run resuming its own checkpoint keeps whatever its record already names,
    so pausing a branch does not cut it loose from its trunk.
    """

    if resumed_from is None:
        return None
    source = resumed_from.resolve()
    if source.parent.parent != run_path.resolve():
        return source
    return _recorded_branch_point(run_path / "run.json")


def _recorded_branch_point(record_path: Path) -> Path | None:
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        optimization = record["optimization"]
        # Records written before `branched_from` existed name only the
        # checkpoint their last invocation resumed from.
        recorded = optimization.get("branched_from", optimization["resumed_from"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None
    if not isinstance(recorded, str):
        return None
    source = Path(recorded).resolve()
    if source.parent.parent == record_path.parent.resolve():
        return None
    return source


def record_branch(run_path: Path, source: Path) -> None:
    """Note beside a trunk that the run at ``run_path`` continues from ``source``.

    The note is named for the branch and lives in the trunk's own directory,
    where ``run_protected_checkpoints`` reads it.
    """

    # A source outside a run's checkpoints directory is never pruned.
    if source.parent.name != "checkpoints":
        return
    directory = source.parent.parent / BRANCHES_DIRECTORY
    name = run_path.resolve().name
    note = directory / name
    temporary = directory / f".{name}.tmp"
    try:
        directory.mkdir(exist_ok=True)
        temporary.write_text(f"{source.resolve()}\n", encoding="utf-8")
        temporary.replace(note)
    except OSError as error:
        raise CheckpointError(
            f"cannot record the branch point {source} beside its run: {error}"
        ) from error


def run_protected_checkpoints(run_path: Path) -> frozenset[Path]:
    """Return the checkpoints of one run that something outside it still reads.

    Only the branches noted beside the run are seen, which is every branch the
    runner started.
    """

    protected: set[Path] = set()
    try:
        notes = list((run_path / BRANCHES_DIRECTORY).iterdir())
    except FileNotFoundError:
        notes = []
    except OSError as error:
        raise CheckpointError(
            f"cannot read the branches of run {run_path}: {error}"
        ) from error
    for note in notes:
        # A note `record_branch` is still writing.
        if note.name.startswith("."):
            continue
        try:
            protected.add(Path(note.read_text(encoding="utf-8").strip()))
        except OSError as error:
            raise CheckpointError(
                f"cannot read the branch point {note}: {error}"
            ) from error
    if (selected := _selected_checkpoint(run_path.parent)) is not None:
        protected.add(selected)
    return frozenset(protected)


def protected_checkpoints(run_root: Path) -> frozenset[Path]:
    """Return the checkpoints something outside their own run still reads.

    A branch's record names the trunk checkpoint it continues from, and the
    default model selection can pin a checkpoint by name. Only runs beneath
    ``run_root`` are seen.
    """

    protected = {
        source
        for record_path in run_root.glob("*/run.json")
        if (source := _recorded_branch_point(record_path)) is not None
    }
    if (selected := _selected_checkpoint(run_root)) is not None:
        protected.add(selected)
    return frozenset(protected)


def _selected_checkpoint(run_root: Path) -> Path | None:
    try:
        selected = resolve_model_selection(ModelRunnerConfig(), run_root=run_root)
    except ModelSelectionError:
        return None
    return selected.checkpoint_path.resolve()


def remove_checkpoint(path: Path, *, protected: Collection[Path]) -> None:
    """Delete one superseded checkpoint unless another reader still needs it."""

    if path.resolve() in protected:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        raise CheckpointError(f"cannot remove checkpoint {path}: {error}") from error


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
        # Raw rather than through the configuration schema, which a run
        # recorded under an older one need not satisfy.
        config = record["resolved_config"]["config"]
        declared_every = config.get("keep_every_steps")
        policy = RetentionPolicy(
            final_step=config["steps"],
            keep_every_steps=(
                declared_every if declared_every is not None else keep_every_steps
            ),
            keep_steps=frozenset(config.get("keep_steps", ())) | frozenset(keep_steps),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return RunPruning(run_path, skipped="no readable run record")
    try:
        latest = latest_checkpoint_path(run_path)
    except CheckpointError:
        return RunPruning(run_path, skipped="no latest checkpoint")
    latest_step = checkpoint_step(latest)
    assert latest_step is not None  # the latest pointer only names valid steps
    removable: list[Path] = []
    held: list[Path] = []
    for path in sorted(latest.parent.glob("step-*.pt")):
        step = checkpoint_step(path)
        if step is None or step >= latest_step or policy.retains(step):
            continue
        (held if path.resolve() in protected else removable).append(path)
    return RunPruning(run_path, removable=tuple(removable), held=tuple(held))
