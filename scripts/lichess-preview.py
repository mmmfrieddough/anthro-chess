#!/usr/bin/env python3
"""Serve this checkout's engine as a Lichess bot, so a change can be played.

This is development tooling, not part of the runtime package. Run it from the
checkout to preview, through that checkout's environment:

    uv run scripts/lichess-preview.py start [--run RUN] [--checkpoint FILE]
                                            [--rating ELO] [--temperature T]
    uv run scripts/lichess-preview.py status
    uv run scripts/lichess-preview.py stop [--force]

The bot account, its token, and the one Lichess username allowed to challenge
it belong to the machine. They are read from the environment, falling back to
an env file under the user's config directory. `docs/playable-uci.md` covers
the one-time setup.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# lichess-bot is an application rather than a package, so it is fetched into
# machine state. Pinned because the generated configuration is written against
# this revision's schema.
LICHESS_BOT_REPOSITORY = "https://github.com/lichess-bot-devs/lichess-bot.git"
LICHESS_BOT_COMMIT = "df7e730de58cc3ef2f1415a0dc2eeda842d39167"

TOKEN_VARIABLE = "LICHESS_BOT_TOKEN"
OPPONENT_VARIABLE = "ANTHRO_CHESS_PREVIEW_OPPONENT"
LICHESS = "https://lichess.org"
SETUP_HINT = "See 'Preview A Change On Lichess' in docs/playable-uci.md."
STARTUP_SECONDS = 120
STOP_SECONDS = 30


class PreviewError(RuntimeError):
    """A prerequisite is missing or the preview cannot do what was asked."""


def _xdg_home(variable: str, fallback: Path) -> Path:
    value = os.environ.get(variable, "").strip()
    return Path(value).expanduser() if value else fallback


ENV_FILE = (
    _xdg_home("XDG_CONFIG_HOME", Path.home() / ".config")
    / "anthro-chess"
    / "lichess-preview.env"
)
STATE_ROOT = (
    _xdg_home("XDG_STATE_HOME", Path.home() / ".local" / "state")
    / "anthro-chess"
    / "lichess-preview"
)
STATE_FILE = STATE_ROOT / "preview.json"
LOCK_FILE = STATE_ROOT / "preview.lock"
LOG_FILE = STATE_ROOT / "lichess-bot.log"
CONSOLE_FILE = STATE_ROOT / "console.log"


def machine_value(name: str) -> str:
    """Read a machine value from the environment, then from the env file."""

    value = os.environ.get(name, "").strip()
    if value:
        return value
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text().splitlines():
            key, separator, candidate = line.strip().partition("=")
            if separator and key.strip() == name and candidate.strip():
                return candidate.strip().strip("'\"")
    raise PreviewError(
        f"{name} is not set in the environment or in {ENV_FILE}. {SETUP_HINT}"
    )


def lichess_get(path: str, token: str | None = None) -> tuple[Any, dict[str, str]]:
    request = urllib.request.Request(
        LICHESS + path, headers={"Accept": "application/json"}
    )
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response), dict(response.headers)
    except urllib.error.HTTPError as error:
        if error.code == 401:
            raise PreviewError(
                f"Lichess rejected {TOKEN_VARIABLE}. {SETUP_HINT}"
            ) from error
        raise PreviewError(f"Lichess answered {error.code} for {path}") from error
    except urllib.error.URLError as error:
        raise PreviewError(f"Lichess is unreachable: {error.reason}") from error


def bot_account(token: str) -> str:
    account, headers = lichess_get("/api/account", token)
    name: str = account["username"]
    scopes = {scope.strip() for scope in headers.get("X-OAuth-Scopes", "").split(",")}
    if "bot:play" not in scopes:
        raise PreviewError(
            f"The token for {name} lacks the bot:play scope. {SETUP_HINT}"
        )
    if account.get("title") != "BOT":
        raise PreviewError(f"{name} is not a BOT account yet. {SETUP_HINT}")
    return name


def current_games(account: str) -> list[str]:
    status, _ = lichess_get(f"/api/users/status?ids={account}&withGameIds=true")
    return [
        f"{LICHESS}/{entry['playingId']}" for entry in status if entry.get("playingId")
    ]


def take_lock() -> int | None:
    """Return a descriptor holding the preview lock, or None if a bot holds it.

    The running bot inherits the descriptor, so the lock is held for exactly as
    long as that process lives, whatever happens to the command that started it.
    """

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


def running_preview() -> dict[str, Any] | None:
    """Return the recorded preview if its bot still holds the lock.

    A record left behind by a bot that has exited is deleted.
    """

    descriptor = take_lock()
    if descriptor is not None:
        os.close(descriptor)
        STATE_FILE.unlink(missing_ok=True)
        return None
    record: dict[str, Any] = json.loads(STATE_FILE.read_text())
    return record


def ensure_lichess_bot() -> Path:
    home = STATE_ROOT / "lichess-bot" / LICHESS_BOT_COMMIT
    ready = home / ".ready"
    if ready.is_file():
        return home
    if home.exists():
        shutil.rmtree(home)
    home.mkdir(parents=True)
    print(f"Fetching lichess-bot {LICHESS_BOT_COMMIT[:12]} into {home}", flush=True)
    for command in (
        ["git", "init", "--quiet"],
        [
            "git",
            "fetch",
            "--quiet",
            "--depth",
            "1",
            LICHESS_BOT_REPOSITORY,
            LICHESS_BOT_COMMIT,
        ],
        ["git", "checkout", "--quiet", "FETCH_HEAD"],
        ["uv", "venv", "--quiet", ".venv"],
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            ".venv/bin/python",
            "-r",
            "requirements.txt",
        ],
    ):
        subprocess.run(command, cwd=home, check=True)
    ready.touch()
    return home


def bot_configuration(
    engine: Path, checkpoint: Path, opponent: str, uci_options: dict[str, Any]
) -> dict[str, Any]:
    engine_section: dict[str, Any] = {
        "dir": str(engine.parent),
        "name": engine.name,
        "protocol": "uci",
        # lichess-bot passes each of these to the engine as a --key=value argument.
        "engine_options": {
            "set": f'model.checkpoint_path="{checkpoint}"',
            # DEBUG logs every decision with its seed, which is what lets a
            # previewed game be replayed and analyzed afterwards.
            "log-level": "DEBUG",
        },
    }
    if uci_options:
        engine_section["uci_options"] = uci_options
    # Everything absent here keeps lichess-bot's own default, which already
    # leaves out pondering, opening books, tablebases, chat, and bot opponents.
    return {
        "url": LICHESS + "/",
        "engine": engine_section,
        "challenge": {
            "allow_list": [opponent],
            "variants": ["standard"],
            "time_controls": [
                "bullet",
                "blitz",
                "rapid",
                "classical",
                "correspondence",
            ],
            "modes": ["casual"],
        },
    }


def start(arguments: argparse.Namespace) -> None:
    lock = take_lock()
    if lock is None:
        # The lock is taken before the record is written, so a preview that is
        # still starting has no record yet.
        checkout = (
            json.loads(STATE_FILE.read_text())["checkout"]
            if STATE_FILE.is_file()
            else "another checkout, still starting"
        )
        raise PreviewError(
            f"A preview is already running for {checkout}. "
            "Stop it first with: uv run scripts/lichess-preview.py stop"
        )
    token = machine_value(TOKEN_VARIABLE)
    opponent = machine_value(OPPONENT_VARIABLE)
    checkout = Path(__file__).resolve().parent.parent
    engine = checkout / ".venv" / "bin" / "anthro-uci"
    if not engine.is_file():
        raise PreviewError(f"{engine} does not exist. Run 'uv sync' in {checkout}.")

    # Imported here because they load torch, which status and stop never need.
    from anthro_chess.inference.config import LATEST_CHECKPOINT, ModelRunnerConfig
    from anthro_chess.inference.selection import resolve_model_selection
    from anthro_chess.interfaces.config import UCI_TEMPERATURE_SCALE
    from anthro_chess.machine import RUN_ROOT_VARIABLE, optional_root

    try:
        selection = resolve_model_selection(
            ModelRunnerConfig(
                run_path=arguments.run,
                checkpoint=arguments.checkpoint or LATEST_CHECKPOINT,
            ),
            run_root=optional_root(RUN_ROOT_VARIABLE),
        )
    except ValueError as error:
        raise PreviewError(f"No checkpoint to serve: {error}") from error
    account = bot_account(token)
    home = ensure_lichess_bot()

    uci_options: dict[str, Any] = {}
    if arguments.rating is not None:
        uci_options |= {"UCI_LimitStrength": True, "UCI_Elo": arguments.rating}
    if arguments.temperature is not None:
        uci_options["Anthro Temperature"] = round(
            arguments.temperature * UCI_TEMPERATURE_SCALE
        )

    # JSON is valid YAML, which is what lichess-bot reads.
    bot_config = STATE_ROOT / "config.yml"
    bot_config.write_text(
        json.dumps(
            bot_configuration(engine, selection.checkpoint_path, opponent, uci_options),
            indent=2,
        )
    )
    LOG_FILE.unlink(missing_ok=True)
    process = subprocess.Popen(
        [
            str(home / ".venv/bin/python"),
            "lichess-bot.py",
            "--config",
            str(bot_config),
            "--logfile",
            str(LOG_FILE),
            "--disable_auto_logging",
        ],
        cwd=home,
        env={**os.environ, TOKEN_VARIABLE: token},
        stdin=subprocess.DEVNULL,
        stdout=CONSOLE_FILE.open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        pass_fds=[lock],
    )
    record = {
        "pid": process.pid,
        "checkout": str(checkout),
        "checkpoint": str(selection.checkpoint_path),
        "account": account,
        "opponent": opponent,
        "uci_options": uci_options,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    STATE_FILE.write_text(json.dumps(record, indent=2))

    deadline = time.monotonic() + STARTUP_SECONDS
    while not LOG_FILE.is_file() or "awaiting challenges" not in LOG_FILE.read_text():
        if process.poll() is not None or time.monotonic() > deadline:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            STATE_FILE.unlink(missing_ok=True)
            tail = "\n".join(CONSOLE_FILE.read_text().splitlines()[-20:])
            raise PreviewError(
                f"lichess-bot did not come online. Last console lines:\n{tail}"
            )
        time.sleep(1)
    describe(record)
    print(f"Challenge it (casual only): {LICHESS}/?user={account}#friend")


def describe(record: dict[str, Any]) -> None:
    checkout = Path(record["checkout"])
    print(f"Account:    {record['account']}, challengeable by {record['opponent']}")
    print(
        f"Checkout:   {checkout}{'' if checkout.is_dir() else '  (no longer exists)'}"
    )
    print(f"Checkpoint: {record['checkpoint']}")
    print(f"Options:    {record['uci_options'] or 'engine defaults'}")
    print(f"Started:    {record['started_at']}  (pid {record['pid']}, log {LOG_FILE})")


def status(arguments: argparse.Namespace) -> None:
    record = running_preview()
    if record is None:
        print("No preview is running.")
        return
    describe(record)
    games = current_games(record["account"])
    print("Game:       " + (", ".join(games) or "none in progress"))


def stop(arguments: argparse.Namespace) -> None:
    record = running_preview()
    if record is None:
        print("No preview is running.")
        return
    games = [] if arguments.force else current_games(record["account"])
    if games:
        raise PreviewError(
            f"A game is in progress ({', '.join(games)}). "
            "Stopping now abandons it; pass --force to stop anyway."
        )
    os.killpg(record["pid"], signal.SIGINT)
    deadline = time.monotonic() + STOP_SECONDS
    while (lock := take_lock()) is None:
        if time.monotonic() > deadline:
            os.killpg(record["pid"], signal.SIGKILL)
            deadline = float("inf")
        time.sleep(0.2)
    os.close(lock)
    STATE_FILE.unlink(missing_ok=True)
    print(f"Stopped the preview of {record['checkout']}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    start_parser = commands.add_parser(
        "start", help="Bring the bot online serving this checkout."
    )
    start_parser.add_argument(
        "--run", type=Path, help="Retained run, relative to ANTHRO_CHESS_RUN_ROOT."
    )
    start_parser.add_argument("--checkpoint", help="Checkpoint file within the run.")
    start_parser.add_argument(
        "--rating", type=int, help="Target rating; strength limiting on."
    )
    start_parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    start_parser.set_defaults(handler=start)
    commands.add_parser("status", help="Report what the bot is serving.").set_defaults(
        handler=status
    )
    stop_parser = commands.add_parser("stop", help="Take the bot offline.")
    stop_parser.add_argument("--force", action="store_true", help="Stop even mid-game.")
    stop_parser.set_defaults(handler=stop)
    arguments = parser.parse_args()
    if arguments.command == "start" and arguments.checkpoint and not arguments.run:
        parser.error("--checkpoint names a file within --run")
    try:
        arguments.handler(arguments)
    except PreviewError as error:
        print(f"lichess-preview: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
