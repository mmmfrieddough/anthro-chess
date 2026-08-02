"""What this machine is configured to hold, and where it says so.

Corpora and training runs are far too large for a worktree, so they live
outside every checkout beneath a matched pair of environment variables. That
makes the checkout silent about what the machine has: unset, checked-in
relative paths resolve inside the working directory, which is right for a fresh
clone and indistinguishable from a machine that genuinely holds nothing.

Two things follow, and this module owns both. A command that needs a root and
cannot resolve one names the variable and what it would have to contain, rather
than failing as though the artifact were merely absent. And one inventory
reports what the configured roots actually hold, so the question a session
would otherwise answer by searching the repository is answered where the
answer lives.

The variable names are here rather than at their call sites because inference,
training, evaluation, and the UCI process all resolve the same roots, and a
name repeated as a literal in each of them is a name that can drift.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT_VARIABLE = "ANTHRO_CHESS_DATA_ROOT"
RUN_ROOT_VARIABLE = "ANTHRO_CHESS_RUN_ROOT"
RESULTS_ROOT_VARIABLE = "ANTHRO_CHESS_RESULTS_ROOT"
RESULT_DETAIL_ROOT_VARIABLE = "ANTHRO_CHESS_RESULT_DETAIL_ROOT"

#: What each root holds, phrased to complete "must be set to the directory
#: holding ...". A failure that names a variable without saying what belongs in
#: it leaves the reader knowing they are misconfigured and not how to fix it.
ROOT_CONTENTS: Mapping[str, str] = {
    DATA_ROOT_VARIABLE: "corpora, frozen evaluation pools, and puzzle records",
    RUN_ROOT_VARIABLE: "retained training runs and the default model selection",
    RESULTS_ROOT_VARIABLE: "the committed benchmark results store",
    RESULT_DETAIL_ROOT_VARIABLE: "machine-local benchmark detail payloads",
}

#: What happens to each root when it is unset. The pair is genuinely optional,
#: so an unset root is only sometimes a defect and the report has to say which.
ROOT_FALLBACKS: Mapping[str, str] = {
    DATA_ROOT_VARIABLE: "configured relative paths resolve in the working directory",
    RUN_ROOT_VARIABLE: "configured relative paths resolve in the working directory",
    RESULTS_ROOT_VARIABLE: "the committed store resolves as ./results",
    RESULT_DETAIL_ROOT_VARIABLE: (
        f"detail resolves beneath {RUN_ROOT_VARIABLE}, when that is set"
    ),
}

#: The two halves of one setup. Evaluation reads a checkpoint from one and a
#: pool from the other, so a machine holding only one can run neither training
#: nor evaluation end to end.
ARTIFACT_ROOT_PAIR = (DATA_ROOT_VARIABLE, RUN_ROOT_VARIABLE)

#: What the report cannot answer without the optional model dependencies. This
#: is a property of the installation rather than a misconfiguration, so it is
#: reported apart from the problems and does not fail the command.
MODEL_EXTRA_NOTE = (
    "checkpoint pointers and the default model selection were not resolved "
    "because the optional model dependencies are not installed"
)


def optional_root(name: str) -> Path | None:
    """Return a configured machine root, or ``None`` when it is unset."""

    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def required_root(name: str, *, alternative: str) -> Path:
    """Return a configured machine root, or fail naming what is missing.

    ``alternative`` states what the caller could have passed instead, so the
    message covers both ways out of the failure rather than only the
    environment one.
    """

    root = optional_root(name)
    if root is not None:
        return root
    from anthro_chess.config import ConfigError

    raise ConfigError(
        f"{alternative}, or {name} must be set to the directory holding "
        f"{ROOT_CONTENTS[name]}"
    )


@dataclass(frozen=True)
class RootStatus:
    """One machine root, whether it is configured, and whether it is there."""

    variable: str
    path: Path | None
    exists: bool
    contents: str
    fallback: str

    @property
    def configured(self) -> bool:
        """Whether the variable is set at all."""

        return self.path is not None

    def as_record(self) -> dict[str, object]:
        """Return the status as a JSON-serializable record."""

        return {
            "variable": self.variable,
            "path": None if self.path is None else str(self.path),
            "configured": self.configured,
            "exists": self.exists,
            "contents": self.contents,
            "fallback": self.fallback,
        }


@dataclass(frozen=True)
class RetainedRun:
    """One retained training run beneath the run root."""

    name: str
    path: Path
    has_run_record: bool
    checkpoints: int
    latest_checkpoint: str | None

    def as_record(self) -> dict[str, object]:
        """Return the run as a JSON-serializable record."""

        return {
            "name": self.name,
            "path": str(self.path),
            "has_run_record": self.has_run_record,
            "checkpoints": self.checkpoints,
            "latest_checkpoint": self.latest_checkpoint,
        }


@dataclass(frozen=True)
class DataArtifact:
    """One artifact directory beneath the data root, classified by its markers."""

    name: str
    path: Path
    kind: str

    def as_record(self) -> dict[str, object]:
        """Return the artifact as a JSON-serializable record."""

        return {"name": self.name, "path": str(self.path), "kind": self.kind}


@dataclass(frozen=True)
class DefaultModelSelection:
    """How the machine-local default model selection resolves right now."""

    record_path: Path | None
    resolved: dict[str, str] | None
    error: str | None

    def as_record(self) -> dict[str, object]:
        """Return the selection as a JSON-serializable record."""

        return {
            "record_path": None if self.record_path is None else str(self.record_path),
            "resolved": self.resolved,
            "error": self.error,
        }


@dataclass(frozen=True)
class MachineReport:
    """Everything the configured roots say this machine holds."""

    roots: tuple[RootStatus, ...]
    runs: tuple[RetainedRun, ...]
    artifacts: tuple[DataArtifact, ...]
    selection: DefaultModelSelection
    problems: tuple[str, ...]
    unavailable: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        """Return the whole report as a JSON-serializable record."""

        return {
            "roots": [root.as_record() for root in self.roots],
            "runs": [run.as_record() for run in self.runs],
            "artifacts": [artifact.as_record() for artifact in self.artifacts],
            "selection": self.selection.as_record(),
            "problems": list(self.problems),
            "unavailable": list(self.unavailable),
        }


def inspect_machine() -> MachineReport:
    """Report the configured roots and what they hold.

    Nothing here is decoded. A corpus is recognized by the manifest a
    preparation run wrote beside it, not by reading its games, so the report
    stays cheap enough to run before every other command.
    """

    roots = tuple(
        _root_status(variable)
        for variable in (
            DATA_ROOT_VARIABLE,
            RUN_ROOT_VARIABLE,
            RESULTS_ROOT_VARIABLE,
            RESULT_DETAIL_ROOT_VARIABLE,
        )
    )
    run_root = optional_root(RUN_ROOT_VARIABLE)
    data_root = optional_root(DATA_ROOT_VARIABLE)
    model_extra = _model_extra_installed()
    return MachineReport(
        roots=roots,
        runs=_retained_runs(run_root, resolve_latest=model_extra),
        artifacts=_data_artifacts(data_root),
        selection=_default_selection(run_root, resolvable=model_extra),
        problems=_problems(roots),
        unavailable=() if model_extra else (MODEL_EXTRA_NOTE,),
    )


def _model_extra_installed() -> bool:
    """Return whether the model extra this report reads through is installed.

    Checkpoint pointers and the default selection resolve through the packages
    that extra brings in. A base installation can still say which roots are
    configured and what directories are beneath them, and that partial answer
    is worth far more than a traceback from the one command a broken
    environment is most likely to reach for.
    """

    try:
        import anthro_chess.inference.selection  # noqa: F401
    except ImportError:
        return False
    return True


def _root_status(variable: str) -> RootStatus:
    path = optional_root(variable)
    return RootStatus(
        variable=variable,
        path=path,
        exists=path is not None and path.is_dir(),
        contents=ROOT_CONTENTS[variable],
        fallback=ROOT_FALLBACKS[variable],
    )


def _problems(roots: tuple[RootStatus, ...]) -> tuple[str, ...]:
    """Name the configurations that will read as an empty machine.

    Only the artifact pair is checked for half-configuration. The results and
    detail roots have their own documented defaults, so leaving either unset is
    an ordinary setup rather than a mismatch.
    """

    by_variable = {root.variable: root for root in roots}
    problems: list[str] = []
    configured = [
        variable for variable in ARTIFACT_ROOT_PAIR if by_variable[variable].configured
    ]
    if len(configured) == 1:
        missing = next(
            variable for variable in ARTIFACT_ROOT_PAIR if variable not in configured
        )
        problems.append(
            f"{configured[0]} is set but {missing} is not. They are two halves "
            f"of one setup, so this machine cannot find its "
            f"{ROOT_CONTENTS[missing]}."
        )
    for root in roots:
        if root.configured and not root.exists:
            problems.append(
                f"{root.variable} is set to {root.path}, which is not a directory."
            )
    return tuple(problems)


def _retained_runs(
    run_root: Path | None,
    *,
    resolve_latest: bool,
) -> tuple[RetainedRun, ...]:
    if run_root is None or not run_root.is_dir():
        return ()

    runs: list[RetainedRun] = []
    for path in sorted(_directories(run_root)):
        checkpoints = sorted((path / "checkpoints").glob("step-*.pt"))
        has_run_record = (path / "run.json").is_file()
        if not has_run_record and not checkpoints:
            continue
        latest = _latest_checkpoint(path) if resolve_latest else None
        runs.append(
            RetainedRun(
                name=path.name,
                path=path,
                has_run_record=has_run_record,
                checkpoints=len(checkpoints),
                latest_checkpoint=latest,
            )
        )
    return tuple(runs)


def _latest_checkpoint(run_path: Path) -> str | None:
    """Return the run's latest checkpoint name, resolved the way a runner does."""

    from anthro_chess.training.checkpoints import (
        CheckpointError,
        latest_checkpoint_path,
    )

    try:
        return latest_checkpoint_path(run_path).name
    except CheckpointError:
        return None


def _data_artifacts(data_root: Path | None) -> tuple[DataArtifact, ...]:
    if data_root is None or not data_root.is_dir():
        return ()
    from anthro_chess.evaluation.pool import (
        POOL_GAMES_FILE_NAME,
        POOL_MANIFEST_FILE_NAME,
    )
    from anthro_chess.evaluation.puzzles.dataset import (
        PUZZLE_FILE_NAME,
        PUZZLE_METADATA_FILE_NAME,
    )

    artifacts: list[DataArtifact] = []
    for path in sorted(_directories(data_root)):
        if (path / "manifests" / "manifest.json").is_file():
            kind = "corpus"
        elif (path / POOL_MANIFEST_FILE_NAME).is_file() and (
            path / POOL_GAMES_FILE_NAME
        ).is_file():
            kind = "evaluation-pool"
        elif (path / PUZZLE_METADATA_FILE_NAME).is_file() and (
            path / PUZZLE_FILE_NAME
        ).is_file():
            kind = "puzzle-set"
        else:
            kind = "other"
        artifacts.append(DataArtifact(name=path.name, path=path, kind=kind))
    return tuple(artifacts)


def _default_selection(
    run_root: Path | None,
    *,
    resolvable: bool,
) -> DefaultModelSelection:
    """Resolve the default selection exactly as a runtime command would.

    Through the real resolver rather than by reading the record, because a
    report that agreed with the record but disagreed with the runtime would be
    worse than no report at all.
    """

    if not resolvable:
        # Neither resolved nor failed: the reason is a property of the
        # installation, and it is reported once, beside the other things this
        # report could not answer.
        return DefaultModelSelection(record_path=None, resolved=None, error=None)

    from anthro_chess.inference.config import ModelRunnerConfig
    from anthro_chess.inference.selection import (
        MODEL_SELECTION_FILE,
        ModelSelectionError,
        resolve_model_selection,
    )

    record_path = None if run_root is None else run_root / MODEL_SELECTION_FILE
    try:
        selection = resolve_model_selection(ModelRunnerConfig(), run_root=run_root)
    except ModelSelectionError as error:
        return DefaultModelSelection(
            record_path=record_path,
            resolved=None,
            error=str(error),
        )
    return DefaultModelSelection(
        record_path=record_path,
        resolved=selection.as_record(),
        error=None,
    )


def _directories(root: Path) -> list[Path]:
    try:
        return [path for path in root.iterdir() if path.is_dir()]
    except OSError:
        return []
