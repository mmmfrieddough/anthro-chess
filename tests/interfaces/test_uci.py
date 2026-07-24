from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import chess
import pytest
import torch
from pydantic import ValidationError

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
from anthro_chess.training.checkpoints import save_training_checkpoint


@dataclass
class StubRunner:
    logits: torch.Tensor

    def predict(self, _context: DecisionContext) -> torch.Tensor:
        return self.logits.clone()


class FailingRunner:
    def predict(self, _context: DecisionContext) -> torch.Tensor:
        raise RuntimeError("inference exploded")


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
    assert lines[2:5] == [
        "option name UCI_LimitStrength type check default false",
        "option name UCI_Elo type spin default 1500 min 400 max 2500",
        "option name Anthro Temperature type spin default 0 min 0 max 300",
    ]
    assert lines[5:] == ["uciok", "readyok", "bestmove g1f3"]
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


def test_unknown_tokens_are_ignored_and_recognized_commands_still_run() -> None:
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
    assert error == ""


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


def test_inference_failure_returns_null_move_and_process_stays_responsive() -> None:
    engine = UciEngine(lambda: FailingRunner(), UciConfig())

    output, error = _run(engine, "isready\nposition startpos\ngo\nisready\nquit\n")

    assert output == "readyok\nbestmove 0000\nreadyok\n"
    assert "move generation failed: inference exploded" in error


def test_uci_config_rejects_unrepresentable_or_resigning_runtime() -> None:
    with pytest.raises(ValidationError, match="between 400 and 2500"):
        UciConfig(runtime=RuntimeConfig(target_rating=300))
    with pytest.raises(ValidationError, match="increments of 0.01"):
        UciConfig(runtime=RuntimeConfig(temperature=0.755))
    with pytest.raises(ValidationError, match="does not support resignation"):
        UciConfig(runtime=RuntimeConfig(resignation_enabled=True))


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
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
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


def test_console_script_reports_loading_failure_only_on_stderr(tmp_path: Path) -> None:
    executable = Path(sys.executable).with_name("anthro-uci")
    completed = subprocess.run(
        [
            str(executable),
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
    assert "anthro-uci:" in completed.stderr


def _run(engine: UciEngine, transcript: str) -> tuple[str, str]:
    output = io.StringIO()
    error = io.StringIO()
    assert engine.run(io.StringIO(transcript), output, error) == 0
    return output.getvalue(), error.getvalue()


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
