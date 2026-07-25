from __future__ import annotations

import copy
import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import pytest
import torch
from pydantic import ValidationError

from anthro_chess.application_logging import (
    LOG_LEVEL_NAMES,
    configure_application_logging,
    parse_log_level,
)
from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    action_vocabulary_identity,
    encode_move,
)
from anthro_chess.data import DecisionContext, encoding_identity
from anthro_chess.interfaces.config import UCI_MAX_RATING, UciConfig
from anthro_chess.interfaces.uci import UciEngine
from anthro_chess.models import CausalMoveModel, MoveModelConfig
from anthro_chess.runtime import RuntimeConfig
from anthro_chess.runtime import session as session_module
from anthro_chess.training.checkpoints import save_training_checkpoint


@dataclass
class StubRunner:
    logits: torch.Tensor

    def predict(self, _context: DecisionContext) -> torch.Tensor:
        return self.logits.clone()


class FailingRunner:
    def predict(self, _context: DecisionContext) -> torch.Tensor:
        raise RuntimeError("inference exploded")


@pytest.mark.parametrize("level", LOG_LEVEL_NAMES)
def test_stdout_contains_only_protocol_traffic_at_every_log_level(level: str) -> None:
    output = io.StringIO()
    diagnostics = io.StringIO()
    configure_application_logging(level=level, stream=diagnostics)
    engine = UciEngine(
        lambda: StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        UciConfig(),
        normal_log_level=parse_log_level(level),
    )

    assert (
        engine.run(
            io.StringIO("uci\ndebug on\ndebug off\nquit\n"),
            output,
            diagnostics,
        )
        == 0
    )

    assert output.getvalue().endswith("uciok\n")
    assert all(
        line.startswith(("id ", "option ", "uciok"))
        for line in output.getvalue().splitlines()
    )


def test_protocol_transcript_orders_handshake_and_returns_legal_move() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("g1f3"))] = 10.0
    engine = UciEngine(
        lambda: StubRunner(logits),
        UciConfig(runtime=RuntimeConfig(temperature=0.0)),
    )

    output, error = _run(
        engine,
        "\n".join(
            (
                "uci",
                "isready",
                "position startpos moves e2e4 e7e5",
                "go wtime 1000 btime 1000 depth 4",
                "stop",
                "quit",
            )
        ),
    )

    lines = output.splitlines()
    assert lines[0].startswith("id name Anthro Chess ")
    assert lines[1] == "id author Anthro Chess contributors"
    assert lines[2:6] == [
        "option name UCI_LimitStrength type check default false",
        "option name UCI_Elo type spin default 1500 min 400 max 2500",
        "option name Anthro Temperature type spin default 0 min 0 max 300",
        "option name Anthro Seed type spin default -1 min -1 max 2147483647",
    ]
    assert lines[6:] == ["uciok", "readyok", "bestmove g1f3"]
    assert error == ""


def test_option_changes_keep_rating_and_temperature_independent() -> None:
    engine = UciEngine(
        lambda: StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        UciConfig(),
    )

    _run(engine, "setoption name Anthro Temperature value 25\n")
    assert engine.runtime_config == RuntimeConfig(
        target_rating=UCI_MAX_RATING,
        temperature=0.25,
    )

    _run(engine, "setoption name UCI_Elo value 1800\n")
    assert engine.runtime_config.target_rating == UCI_MAX_RATING
    assert engine.runtime_config.temperature == 0.25

    _run(engine, "setoption name UCI_LimitStrength value true\n")
    assert engine.runtime_config.target_rating == 1800

    _run(engine, "setoption name UCI_Elo value 1200\n")
    assert engine.runtime_config.target_rating == 1200

    _run(engine, "setoption name UCI_LimitStrength value false\n")
    assert engine.runtime_config.target_rating == UCI_MAX_RATING
    assert engine.runtime_config.temperature == 0.25


def test_invalid_position_and_options_do_not_mutate_current_state() -> None:
    engine = UciEngine(
        lambda: StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        UciConfig(),
    )
    _run(engine, "position startpos moves e2e4 e7e5\n")
    expected = engine.board

    output, error = _run(
        engine,
        "\n".join(
            (
                "position startpos moves e2e5",
                "setoption name UCI_Elo value 399",
                "quit",
            )
        ),
    )

    assert output == ""
    assert "illegal position move at ply 0" in error
    assert "UCI_Elo must be between 400 and 2500" in error
    assert engine.board == expected
    assert engine.runtime_config.target_rating == UCI_MAX_RATING


def test_new_game_reset_fen_terminal_and_clean_exit() -> None:
    engine = UciEngine(
        lambda: StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        UciConfig(),
    )

    output, error = _run(
        engine,
        "\n".join(
            (
                "position startpos moves e2e4",
                "ucinewgame",
                "position fen 7k/5Q2/7K/8/8/8/8/8 b - - 0 1",
                "go",
                "quit",
            )
        ),
    )

    assert output == "bestmove 0000\n"
    assert error == ""
    assert engine.board.is_stalemate()


def test_model_initialization_is_deferred_to_isready_and_stdout_is_redirected() -> None:
    calls = 0

    def load_runner() -> StubRunner:
        nonlocal calls
        calls += 1
        print("model loader diagnostic")
        return StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE))

    engine = UciEngine(load_runner, UciConfig())

    handshake, handshake_error = _run(engine, "uci\n")
    assert calls == 0
    assert handshake.endswith("uciok\n")
    assert handshake_error == ""

    ready, ready_error = _run(engine, "isready\nisready\n")
    assert calls == 1
    assert ready == "readyok\nreadyok\n"
    assert ready_error == "model loader diagnostic\n"


def test_debug_diagnostics_stay_off_protocol_stdout() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("g1f3"))] = 10.0
    engine = UciEngine(
        lambda: StubRunner(logits),
        UciConfig(runtime=RuntimeConfig(temperature=0.0)),
    )

    output, error = _run(
        engine,
        "\n".join(
            (
                "unknown-prefix debug on",
                "ignored-token isready",
                "position startpos ignored moves e2e4 nonsense e7e5",
                "go unsupported-field 12",
                "quit",
            )
        ),
    )

    assert output == "readyok\nbestmove g1f3\n"
    assert "Received UCI command" in error
    assert all(
        line.startswith(("readyok", "bestmove ")) for line in output.splitlines()
    )


def test_malformed_commands_do_not_crash_or_mutate_valid_state() -> None:
    engine = UciEngine(
        lambda: StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        UciConfig(),
    )
    _run(engine, "position startpos moves e2e4 e7e5\n")
    expected = engine.board

    output, error = _run(
        engine,
        "\n".join(
            (
                "setoption value 100",
                "position fen malformed",
                "debug perhaps",
                "completely unknown input",
                "isready",
                "quit",
            )
        ),
    )

    assert output == "readyok\n"
    assert "setoption requires 'name'" in error
    assert "position fen requires all six FEN fields" in error
    assert "debug expects 'on' or 'off'" in error
    assert engine.board == expected


def test_malformed_position_move_is_rejected_without_shifting_history() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("g1f3"))] = 10.0
    engine = UciEngine(
        lambda: StubRunner(logits),
        UciConfig(runtime=RuntimeConfig(temperature=0.0)),
    )
    _run(engine, "position startpos moves e2e4 e7e5\n")
    expected = engine.board

    # Dropping the bad token would leave a history of e2e4 e7e5 and answer from
    # a position the GUI never asked about.
    output, error = _run(engine, "position startpos moves e2e4 not-a-move e7e5\n")

    assert output == ""
    assert "malformed position move at ply 1: not-a-move" in error
    assert engine.board == expected

    answered, _ = _run(engine, "isready\ngo\nquit\n")
    assert _bestmoves(answered) == ["g1f3"]


def test_go_infinite_withholds_bestmove_until_stop() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("e2e4"))] = 10.0
    engine = UciEngine(
        lambda: StubRunner(logits),
        UciConfig(runtime=RuntimeConfig(temperature=0.0)),
    )

    output, _ = _run(engine, "isready\nposition startpos\ngo infinite\n")
    assert _bestmoves(output) == []

    output, _ = _run(engine, "stop\nquit\n")
    assert _bestmoves(output) == ["e2e4"]


def test_stop_without_an_infinite_search_stays_silent() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("e2e4"))] = 10.0
    engine = UciEngine(
        lambda: StubRunner(logits),
        UciConfig(runtime=RuntimeConfig(temperature=0.0)),
    )

    # An ordinary go already answered, and a second bestmove would be unpaired.
    output, _ = _run(engine, "isready\nposition startpos\ngo\nstop\nstop\nquit\n")

    assert _bestmoves(output) == ["e2e4"]


def test_new_game_discards_an_unstopped_infinite_response() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("e2e4"))] = 10.0
    engine = UciEngine(
        lambda: StubRunner(logits),
        UciConfig(runtime=RuntimeConfig(temperature=0.0)),
    )

    output, _ = _run(
        engine,
        "isready\nposition startpos\ngo infinite\nucinewgame\nstop\nquit\n",
    )

    assert _bestmoves(output) == []


def test_terminal_position_under_infinite_search_also_waits_for_stop() -> None:
    engine = UciEngine(
        lambda: StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        UciConfig(runtime=RuntimeConfig(temperature=0.0)),
    )
    checkmate = "position fen 7k/5Q2/7K/8/8/8/8/8 b - - 0 1"

    output, _ = _run(engine, f"isready\n{checkmate}\ngo infinite\n")
    assert _bestmoves(output) == []

    output, _ = _run(engine, "stop\nquit\n")
    assert _bestmoves(output) == ["0000"]


def test_inference_failure_reports_critical_error_and_exits_without_bestmove() -> None:
    engine = UciEngine(lambda: FailingRunner(), UciConfig())
    output = io.StringIO()
    error = io.StringIO()
    configure_application_logging(level="WARNING", stream=error)

    return_code = engine.run(
        io.StringIO("isready\nposition startpos\ngo\nisready\nquit\n"),
        output,
        error,
    )

    assert return_code == 1
    assert output.getvalue() == (
        "readyok\ninfo string CRITICAL ERROR: move generation failed\n"
    )
    assert "bestmove" not in output.getvalue()
    assert "move generation failed: inference exploded" in error.getvalue()


def test_uci_config_rejects_unrepresentable_or_resigning_runtime() -> None:
    with pytest.raises(ValidationError, match="between 400 and 2500"):
        UciConfig(runtime=RuntimeConfig(target_rating=300))
    with pytest.raises(ValidationError, match="increments of 0.01"):
        UciConfig(runtime=RuntimeConfig(temperature=0.755))
    with pytest.raises(ValidationError, match="does not support resignation"):
        UciConfig(runtime=RuntimeConfig(resignation_enabled=True))
    with pytest.raises(ValidationError, match="UCI seed must be between"):
        UciConfig(runtime=RuntimeConfig(seed=2**31))
    assert UciConfig(runtime=RuntimeConfig(seed=None)).runtime.seed is None


def test_installed_console_script_loads_checkpoint_and_keeps_stdout_clean(
    tmp_path: Path,
) -> None:
    checkpoint = _write_run(tmp_path / "run")
    config = tmp_path / "uci.toml"
    config.write_text(
        "\n".join(
            (
                "[model]",
                f'checkpoint_path = "{checkpoint}"',
                'device = "cpu"',
                "",
                "[runtime]",
                "target_rating = 1500",
                "temperature = 0.0",
                "seed = 0",
                "resignation_enabled = false",
            )
        ),
        encoding="utf-8",
    )
    executable = Path(sys.executable).with_name("anthro-uci")
    environment = os.environ.copy()
    environment["ANTHRO_CHESS_LOG_ROOT"] = str(tmp_path / "logs")

    completed = subprocess.run(
        [str(executable), "--config", str(config)],
        input="\n".join(
            (
                "uci",
                "isready",
                "position startpos",
                "go",
                "position fen 7k/5Q2/7K/8/8/8/8/8 b - - 0 1",
                "go",
                "quit",
                "",
            )
        ),
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert (tmp_path / "logs/uci.log").is_file()
    lines = completed.stdout.splitlines()
    assert lines[0].startswith("id name Anthro Chess ")
    assert "uciok" in lines
    assert "readyok" in lines
    bestmoves = [
        line.removeprefix("bestmove ") for line in lines if line.startswith("bestmove ")
    ]
    assert len(bestmoves) == 2
    assert chess.Move.from_uci(bestmoves[0]) in chess.Board().legal_moves
    assert bestmoves[1] == "0000"
    assert all(
        line.startswith(("id ", "option ", "uciok", "readyok", "bestmove "))
        for line in lines
    )


@pytest.mark.integration
def test_python_chess_client_plays_terminal_position_and_starts_new_game(
    tmp_path: Path,
) -> None:
    checkpoint = _write_run(tmp_path / "run")
    executable = Path(sys.executable).with_name("anthro-uci")
    environment = os.environ.copy()
    environment["ANTHRO_CHESS_LOG_ROOT"] = str(tmp_path / "logs")
    engine = chess.engine.SimpleEngine.popen_uci(
        [
            str(executable),
            "--set",
            f'model.checkpoint_path="{checkpoint}"',
            "--set",
            'model.device="cpu"',
        ],
        timeout=30.0,
        cwd=tmp_path,
        env=environment,
    )

    try:
        engine.configure(
            {
                "UCI_LimitStrength": True,
                "UCI_Elo": 1300,
                "Anthro Temperature": 0,
            }
        )
        first_game = object()
        board = chess.Board()
        for _ in range(4):
            result = engine.play(
                board,
                chess.engine.Limit(depth=1),
                game=first_game,
            )
            assert result.move is not None
            assert result.move in board.legal_moves
            board.push(result.move)
            assert not board.is_game_over()
            opponent_move = min(board.legal_moves, key=lambda move: move.uci())
            board.push(opponent_move)

        terminal = chess.Board()
        for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
            terminal.push_uci(move)
        assert terminal.is_checkmate()
        terminal_result = engine.play(
            terminal,
            chess.engine.Limit(depth=1),
            game=first_game,
        )
        assert terminal_result.move == chess.Move.null()

        fresh_board = chess.Board()
        fresh_result = engine.play(
            fresh_board,
            chess.engine.Limit(depth=1),
            game=object(),
        )
        assert fresh_result.move is not None
        assert fresh_result.move in fresh_board.legal_moves
    finally:
        engine.quit()

    assert (tmp_path / "logs/uci.log").is_file()


def test_console_script_reports_loading_failure_only_in_log_file(
    tmp_path: Path,
) -> None:
    executable = Path(sys.executable).with_name("anthro-uci")
    log_path = tmp_path / "logs/uci.log"
    completed = subprocess.run(
        [
            str(executable),
            "--log-file",
            str(log_path),
            "--set",
            f'model.checkpoint_path="{tmp_path / "missing/checkpoints/model.pt"}"',
        ],
        input="uci\nisready\n",
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert completed.returncode == 2
    assert completed.stdout.endswith("uciok\n")
    assert all(
        line.startswith(("id ", "option ", "uciok"))
        for line in completed.stdout.splitlines()
    )
    assert completed.stderr == ""
    assert "model initialization failed" in log_path.read_text(encoding="utf-8")


def test_console_script_falls_back_to_stderr_without_contaminating_stdout(
    tmp_path: Path,
) -> None:
    executable = Path(sys.executable).with_name("anthro-uci")
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("occupied", encoding="utf-8")

    completed = subprocess.run(
        [
            str(executable),
            "--log-file",
            str(blocking_file / "uci.log"),
        ],
        input="uci\nquit\n",
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert completed.returncode == 0
    assert completed.stdout.endswith("uciok\n")
    assert "using standard error" in completed.stderr
    assert all(
        line.startswith(("id ", "option ", "uciok"))
        for line in completed.stdout.splitlines()
    )


def test_debug_logs_exclude_raw_commands_and_complete_game_history() -> None:
    engine = UciEngine(
        lambda: StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        UciConfig(),
    )
    sensitive = "private-user-secret-corpus-record"

    output, error = _run(
        engine,
        "\n".join(
            (
                "debug on",
                sensitive,
                "position startpos moves e2e4 e7e5 g1f3",
                "quit",
            )
        ),
    )

    assert output == ""
    assert "Received UCI command" in error
    assert sensitive not in error
    assert "e2e4 e7e5 g1f3" not in error


def test_debug_logs_record_option_names_and_values() -> None:
    engine = UciEngine(
        lambda: StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        UciConfig(),
    )

    # A reported session is only reproducible if the settings that produced it
    # are recoverable, so every applied option is recorded by name and value.
    _, error = _run(
        engine,
        "\n".join(
            (
                "debug on",
                "setoption name UCI_LimitStrength value true",
                "setoption name UCI_Elo value 1234",
                "setoption name Anthro Temperature value 75",
                "setoption name Unsupported Option value 9",
                "quit",
            )
        ),
    )

    assert "UCI_LimitStrength with value true" in error
    assert "UCI_Elo with value 1234" in error
    assert "Anthro Temperature with value 75" in error
    # Options the engine does not implement are recorded rather than hidden.
    assert "Unsupported Option with value 9" in error


def test_position_updates_reuse_the_runner_and_reproduce_a_seeded_game() -> None:
    logits = _sampling_logits()
    error = io.StringIO()
    configure_application_logging(level="WARNING", stream=error)

    first_engine, first_loads = _counting_engine(logits, seed=321)
    first = _play_engine_game(first_engine, error, plies=4)
    second_engine, second_loads = _counting_engine(logits, seed=321)
    second = _play_engine_game(second_engine, error, plies=4)

    assert len(first) == 4
    assert first == second
    # Many position updates and go calls reuse one loaded runner per process.
    assert first_loads() == 1
    assert second_loads() == 1


def test_new_game_draws_a_fresh_stream_without_reloading_the_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = _sampling_logits()
    monkeypatch.setattr(session_module, "_draw_fresh_seed", iter([900, 901]).__next__)
    error = io.StringIO()
    configure_application_logging(level="WARNING", stream=error)

    fresh, loads = _counting_engine(logits, seed=None)
    game_one = _play_engine_game(fresh, error, plies=1)
    _drive(fresh, "ucinewgame\n", error)
    game_two = _play_engine_game(fresh, error, plies=1)

    # A fresh game establishes a new stream but never reloads the model.
    assert loads() == 1
    reference_one, _ = _counting_engine(logits, seed=900)
    reference_two, _ = _counting_engine(logits, seed=901)
    assert game_one == _play_engine_game(reference_one, error, plies=1)
    assert game_two == _play_engine_game(reference_two, error, plies=1)


def test_seed_option_selects_reproducible_then_fresh_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = _sampling_logits()
    monkeypatch.setattr(session_module, "_draw_fresh_seed", iter([555]).__next__)
    error = io.StringIO()
    configure_application_logging(level="WARNING", stream=error)

    engine, _ = _counting_engine(logits, seed=None)
    _drive(engine, "setoption name Anthro Seed value 42\nisready\n", error)
    explicit = _play_engine_game(engine, error, plies=1)

    reference_42, _ = _counting_engine(logits, seed=42)
    assert explicit == _play_engine_game(reference_42, error, plies=1)

    _drive(engine, "ucinewgame\nsetoption name Anthro Seed value -1\n", error)
    fresh = _play_engine_game(engine, error, plies=1)

    reference_555, _ = _counting_engine(logits, seed=555)
    assert fresh == _play_engine_game(reference_555, error, plies=1)


def test_console_script_reproduces_seeded_games_and_varies_fresh_streams(
    tmp_path: Path,
) -> None:
    checkpoint = _write_run(tmp_path / "run")
    executable = Path(sys.executable).with_name("anthro-uci")
    transcript = "\n".join(
        (
            "uci",
            "isready",
            "position startpos",
            "go",
            "position startpos moves e2e4 e7e5",
            "go",
            "ucinewgame",
            "position startpos",
            "go",
            "quit",
            "",
        )
    )

    def run(config_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["ANTHRO_CHESS_LOG_ROOT"] = str(tmp_path / "logs")
        return subprocess.run(
            [str(executable), "--config", str(config_path), *extra],
            input=transcript,
            check=False,
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=environment,
        )

    fixed = _uci_config(tmp_path / "fixed.toml", checkpoint, seed="seed = 1234")
    first = run(fixed)
    second = run(fixed)
    assert first.returncode == 0, first.stderr
    assert _bestmoves(first.stdout) == _bestmoves(second.stdout)
    assert len(_bestmoves(first.stdout)) == 3
    assert chess.Move.from_uci(_bestmoves(first.stdout)[0]) in chess.Board().legal_moves

    fresh = _uci_config(tmp_path / "fresh.toml", checkpoint, seed="")
    log_path = tmp_path / "fresh.log"
    third = run(fresh, "--log-level", "DEBUG", "--log-file", str(log_path))
    assert third.returncode == 0, third.stderr
    seeds = re.findall(r"resolved seed (\d+)", log_path.read_text(encoding="utf-8"))
    # Each new game establishes an independent fresh stream.
    assert len(set(seeds)) >= 2


def _run(engine: UciEngine, transcript: str) -> tuple[str, str]:
    output = io.StringIO()
    error = io.StringIO()
    configure_application_logging(level="WARNING", stream=error)
    assert engine.run(io.StringIO(transcript), output, error) == 0
    return output.getvalue(), error.getvalue()


def _sampling_logits() -> torch.Tensor:
    logits = torch.arange(ACTION_VOCABULARY_SIZE, dtype=torch.float32)
    return logits / logits.max()


def _counting_engine(
    logits: torch.Tensor, *, seed: int | None
) -> tuple[UciEngine, Callable[[], int]]:
    calls = 0

    def load_runner() -> StubRunner:
        nonlocal calls
        calls += 1
        return StubRunner(logits)

    engine = UciEngine(
        load_runner,
        UciConfig(runtime=RuntimeConfig(temperature=1.0, seed=seed)),
    )
    return engine, lambda: calls


def _drive(engine: UciEngine, transcript: str, error: io.StringIO) -> str:
    output = io.StringIO()
    assert engine.run(io.StringIO(transcript), output, error) == 0
    return output.getvalue()


def _play_engine_game(
    engine: UciEngine, error: io.StringIO, *, plies: int
) -> list[str]:
    board = chess.Board()
    moves: list[chess.Move] = []
    bestmoves: list[str] = []
    for _ in range(plies):
        command = "position startpos"
        if moves:
            command += " moves " + " ".join(move.uci() for move in moves)
        output = _drive(engine, command + "\ngo\n", error)
        best = output.splitlines()[-1].removeprefix("bestmove ")
        bestmoves.append(best)
        if best == "0000":
            break
        engine_move = chess.Move.from_uci(best)
        board.push(engine_move)
        moves.append(engine_move)
        if board.is_game_over():
            break
        reply = min(board.legal_moves, key=lambda move: move.uci())
        board.push(reply)
        moves.append(reply)
        if board.is_game_over():
            break
    return bestmoves


def _bestmoves(stdout: str) -> list[str]:
    return [
        line.removeprefix("bestmove ")
        for line in stdout.splitlines()
        if line.startswith("bestmove ")
    ]


def _uci_config(path: Path, checkpoint: Path, *, seed: str) -> Path:
    lines = [
        "[model]",
        f'checkpoint_path = "{checkpoint}"',
        'device = "cpu"',
        "",
        "[runtime]",
        "target_rating = 1500",
        "temperature = 1.0",
    ]
    if seed:
        lines.append(seed)
    lines.append("resignation_enabled = false")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_run(path: Path) -> Path:
    torch.manual_seed(7)
    path.mkdir(parents=True)
    model_config = MoveModelConfig(
        piece_embedding_dim=2,
        action_embedding_dim=2,
        model_dim=4,
        attention_heads=1,
        transformer_layers=1,
        feedforward_dim=8,
        dropout=0.0,
    )
    model = CausalMoveModel(model_config)
    model_identity = model.identity()
    resolved_config = {
        "config": {"model": model_config.model_dump(mode="json")},
        "provenance": {"source": None, "overrides": []},
    }
    execution = {
        "device": "cpu",
        "backend": "cpu",
        "precision": "float32",
        "parameter_dtype": "float32",
        "determinism": "strict",
        "gradient_accumulation_steps": 1,
        "phase_profiling": False,
    }
    metadata = {
        "resolved_config": copy.deepcopy(resolved_config),
        "code": {"package_version": "test", "git_revision": "test"},
        "data": {},
        "model": copy.deepcopy(model_identity),
        "action_vocabulary": action_vocabulary_identity(),
        "encoding": encoding_identity(),
        "execution": copy.deepcopy(execution),
    }
    compatibility = {
        "training_config": {},
        "data": {},
        "model": copy.deepcopy(model_identity),
        "action_vocabulary": action_vocabulary_identity(),
        "encoding": encoding_identity(),
    }
    checkpoint = path / "checkpoints" / "step-00000001.pt"
    save_training_checkpoint(
        checkpoint,
        global_step=1,
        counters={"processed_positions": 1},
        model_state=model.state_dict(),
        optimizer_state={},
        scheduler_state=None,
        scaler_state=None,
        loader_state={},
        compatibility=compatibility,
        metadata=metadata,
        device="cpu",
    )
    (path / "run.json").write_text(
        json.dumps(
            {
                "version": 3,
                "resolved_config": resolved_config,
                "model": model_identity,
                "action_vocabulary": action_vocabulary_identity(),
                "encoding": encoding_identity(),
                "execution": execution,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return checkpoint
