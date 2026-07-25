from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "anthro-uci-gui"
TARGET_TOOL = REPOSITORY_ROOT / "scripts" / "anthro-gui-target"

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="the GUI launcher is a POSIX shell entry point",
)


def test_launcher_without_a_pointer_serves_its_own_checkout(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "main", "main-engine")
    state = tmp_path / "state"

    result = _run(checkout, state, ["--config-check"])

    assert result.returncode == 0
    # Only the engine writes to standard output.
    assert _served(result) == "main-engine"
    assert "arg --config-check" in result.stdout
    # The shared configuration and log verbosity reach the engine.
    assert "arg --config" in result.stdout
    assert "arg DEBUG" in result.stdout
    assert "no pointer" in result.stderr


def test_pointer_redirects_the_gui_to_another_checkout(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "main", "main-engine")
    worktree = _checkout(tmp_path / "issue-99", "worktree-engine")
    state = tmp_path / "state"

    assert _point(checkout, state, worktree).returncode == 0
    assert _point(checkout, state, None).stdout == f"{worktree}\n"

    result = _run(checkout, state, [])

    assert result.returncode == 0
    assert _served(result) == "worktree-engine"
    assert str(worktree) in result.stderr


def test_clearing_the_pointer_restores_the_default_checkout(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "main", "main-engine")
    worktree = _checkout(tmp_path / "issue-99", "worktree-engine")
    state = tmp_path / "state"
    _point(checkout, state, worktree)

    cleared = _point(checkout, state, None, clear=True)

    assert cleared.returncode == 0
    assert _served(_run(checkout, state, [])) == "main-engine"


def test_removed_target_fails_loudly_instead_of_starting(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "main", "main-engine")
    worktree = _checkout(tmp_path / "issue-99", "worktree-engine")
    state = tmp_path / "state"
    _point(checkout, state, worktree)
    # Routine housekeeping removes merged worktrees.
    shutil.rmtree(worktree)

    result = _run(checkout, state, [])

    assert result.returncode == 1
    assert result.stdout == ""
    assert str(worktree) in result.stderr
    assert "--clear" in result.stderr


def test_target_without_an_initialized_environment_is_rejected(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "main", "main-engine")
    bare = tmp_path / "uninitialized"
    bare.mkdir()
    state = tmp_path / "state"

    pointed = _point(checkout, state, bare)
    assert pointed.returncode == 1
    assert "uv sync" in pointed.stderr

    # A pointer written by hand still cannot start a nonexistent engine.
    pointer = state / "gui-target"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(f"{bare}\n", encoding="utf-8")
    result = _run(checkout, state, [])

    assert result.returncode == 1
    assert result.stdout == ""
    assert "no runnable engine" in result.stderr


def test_missing_configuration_fails_before_the_engine_starts(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "main", "main-engine")
    state = tmp_path / "state"

    result = _run(checkout, state, [], config=False)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "no engine configuration" in result.stderr


def test_empty_pointer_reports_the_cause(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "main", "main-engine")
    state = tmp_path / "state"
    (state).mkdir(parents=True, exist_ok=True)
    (state / "gui-target").write_text("\n", encoding="utf-8")

    result = _run(checkout, state, [])

    assert result.returncode == 1
    assert result.stdout == ""
    assert "empty" in result.stderr


def _checkout(path: Path, engine_name: str) -> Path:
    """Build a checkout whose engine reports its identity and arguments."""

    scripts = path / "scripts"
    scripts.mkdir(parents=True)
    for tool in (LAUNCHER, TARGET_TOOL):
        copied = scripts / tool.name
        shutil.copy2(tool, copied)
        copied.chmod(0o755)

    engine = path / ".venv" / "bin" / "anthro-uci"
    engine.parent.mkdir(parents=True)
    engine.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' '{engine_name}'\n"
        'for argument in "$@"; do printf \'arg %s\\n\' "$argument"; done\n',
        encoding="utf-8",
    )
    engine.chmod(0o755)
    return path


def _served(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout.splitlines()[0]


def _environment(state: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["ANTHRO_CHESS_GUI_ROOT"] = str(state)
    environment.pop("ANTHRO_CHESS_GUI_TARGET", None)
    environment.pop("ANTHRO_CHESS_GUI_CONFIG", None)
    return environment


def _run(
    checkout: Path,
    state: Path,
    arguments: list[str],
    *,
    config: bool = True,
) -> subprocess.CompletedProcess[str]:
    if config:
        state.mkdir(parents=True, exist_ok=True)
        (state / "gui.toml").write_text("[model]\n", encoding="utf-8")
    return subprocess.run(
        [str(checkout / "scripts" / "anthro-uci-gui"), *arguments],
        capture_output=True,
        text=True,
        env=_environment(state),
        timeout=30,
        check=False,
    )


def _point(
    checkout: Path,
    state: Path,
    target: Path | None,
    *,
    clear: bool = False,
) -> subprocess.CompletedProcess[str]:
    arguments = ["--clear"] if clear else ([str(target)] if target else [])
    return subprocess.run(
        [str(checkout / "scripts" / "anthro-gui-target"), *arguments],
        capture_output=True,
        text=True,
        env=_environment(state),
        timeout=30,
        check=False,
    )
