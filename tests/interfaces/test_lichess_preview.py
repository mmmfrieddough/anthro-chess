from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

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


def test_a_recorded_pid_that_is_not_lichess_bot_is_ignored(
    tmp_path: Path,
) -> None:
    _record(tmp_path, os.getpid(), "/elsewhere/issue-99")

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


def test_forced_stop_ends_the_preview_without_asking_lichess(
    tmp_path: Path, live_pid: int
) -> None:
    _record(tmp_path, live_pid, "/elsewhere/issue-99")

    result = _run(tmp_path, ["stop", "--force"])

    assert result.returncode == 0
    assert "Stopped the preview of /elsewhere/issue-99" in result.stdout
    assert not (tmp_path / "state" / STATE).exists()


def test_stop_refuses_while_a_game_is_in_progress(
    tmp_path: Path, live_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview = _load(tmp_path, monkeypatch)
    _record(tmp_path, live_pid, "/elsewhere/issue-99")
    monkeypatch.setattr(preview, "current_games", lambda account: ["abcd1234"])

    with pytest.raises(preview.PreviewError, match="abcd1234"):
        preview.stop(argparse.Namespace(force=False))
    assert (tmp_path / "state" / STATE).is_file()


@pytest.mark.parametrize(
    ("title", "scopes", "message"),
    [("BOT", "challenge:read", "bot:play"), (None, "bot:play", "not a BOT account")],
)
def test_the_token_must_belong_to_a_bot_able_to_play(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    title: str | None,
    scopes: str,
    message: str,
) -> None:
    preview = _load(tmp_path, monkeypatch)
    account = {"username": "anthro-dev", "title": title}
    monkeypatch.setattr(
        preview,
        "lichess_get",
        lambda path, token: (account, {"X-OAuth-Scopes": scopes}),
    )

    with pytest.raises(preview.PreviewError, match=message):
        preview.bot_account("token")


@pytest.fixture
def live_pid() -> Iterator[int]:
    # Named like lichess-bot, in its own session, as a started preview is.
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "lichess-bot.py"],
        start_new_session=True,
    )
    yield process.pid
    process.kill()
    process.wait()


def _load(root: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(root / "state"))
    spec = importlib.util.spec_from_file_location("lichess_preview", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(root: Path, pid: int, checkout: str) -> None:
    path = root / "state" / STATE
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"pid": pid, "checkout": checkout, "account": "anthro-dev"})
    )


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
