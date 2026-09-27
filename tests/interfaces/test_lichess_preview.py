from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "lichess-preview.py"
STATE = Path("anthro-chess") / "lichess-preview" / "preview.json"

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="the preview manages a POSIX process group",
)


def test_start_without_machine_values_points_at_the_setup(tmp_path: Path) -> None:
    result = _run(tmp_path, ["start"])

    assert result.returncode == 1
    assert "LICHESS_BOT_TOKEN is not set" in result.stderr
    assert "docs/playable-uci.md" in result.stderr
    assert "Traceback" not in result.stderr


def test_start_is_refused_while_another_preview_runs(
    tmp_path: Path, live_pid: int
) -> None:
    _record(tmp_path, live_pid, "/elsewhere/issue-99")

    result = _run(tmp_path, ["start"])

    assert result.returncode == 1
    assert "already running for /elsewhere/issue-99" in result.stderr
    assert (tmp_path / "state" / STATE).is_file()


def test_a_dead_preview_does_not_block_a_new_one(tmp_path: Path) -> None:
    _record(tmp_path, _dead_pid(), "/elsewhere/issue-99")

    result = _run(tmp_path, ["start"])

    # Past the running-preview check, stopped by the next prerequisite.
    assert "LICHESS_BOT_TOKEN is not set" in result.stderr
    assert not (tmp_path / "state" / STATE).exists()


def test_env_file_supplies_what_the_environment_lacks(tmp_path: Path) -> None:
    config = tmp_path / "config" / "anthro-chess"
    config.mkdir(parents=True)
    (config / "lichess-preview.env").write_text("LICHESS_BOT_TOKEN=from-file\n")

    result = _run(tmp_path, ["start"])

    assert "ANTHRO_CHESS_PREVIEW_OPPONENT is not set" in result.stderr


def test_a_checkpoint_needs_its_run(tmp_path: Path) -> None:
    result = _run(tmp_path, ["start", "--checkpoint", "step-1.pt"])

    assert result.returncode == 2
    assert "--checkpoint names a file within --run" in result.stderr


@pytest.fixture
def live_pid() -> Iterator[int]:
    process = subprocess.Popen(["sleep", "60"])
    yield process.pid
    process.kill()
    process.wait()


def _dead_pid() -> int:
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


def _record(root: Path, pid: int, checkout: str) -> None:
    path = root / "state" / STATE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"pid": pid, "checkout": checkout}))


def _run(root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["XDG_CONFIG_HOME"] = str(root / "config")
    environment["XDG_STATE_HOME"] = str(root / "state")
    environment.pop("LICHESS_BOT_TOKEN", None)
    environment.pop("ANTHRO_CHESS_PREVIEW_OPPONENT", None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )
