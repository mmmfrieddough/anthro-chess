from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import chess
import pytest
import torch
from pydantic import ValidationError

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    RESIGNATION_ACTION_ID,
    encode_move,
)
from anthro_chess.config import load_config
from anthro_chess.data import DecisionContext
from anthro_chess.runtime import (
    ActionSelectionError,
    GameSession,
    MoveAction,
    ResignationAction,
    RuntimeConfig,
    SessionStateError,
)


@dataclass
class StubRunner:
    logits: torch.Tensor
    contexts: list[DecisionContext] = field(default_factory=list)

    def predict(self, context: DecisionContext) -> torch.Tensor:
        self.contexts.append(context)
        return self.logits.clone()


def test_white_session_builds_full_context_and_applies_only_a_legal_move() -> None:
    runner = StubRunner(_ranked_logits("e2e4", illegal="e2e5"))
    session = GameSession(
        runner,
        controlled_color=chess.WHITE,
        config=RuntimeConfig(target_rating=1725, temperature=0.0),
    )

    action = session.choose_action()

    assert isinstance(action, MoveAction)
    assert action.action_id == encode_move(chess.Move.from_uci("e2e4"))
    assert action.move == chess.Move.from_uci("e2e4")
    assert session.move_history == (action.move,)
    assert runner.contexts[0].target_rating == 1725
    assert len(runner.contexts[0].plies) == 1
    assert all(ply.time_initial_ms is None for ply in runner.contexts[0].plies)


def test_black_session_observes_both_players_without_an_opponent_rating() -> None:
    runner = StubRunner(_ranked_logits("c7c5"))
    session = GameSession(
        runner,
        controlled_color=chess.BLACK,
        config=RuntimeConfig(target_rating=1400, temperature=0.0),
    )
    white_move = chess.Move.from_uci("e2e4")
    session.apply_move(white_move)

    action = session.choose_action()

    assert isinstance(action, MoveAction)
    assert action.move == chess.Move.from_uci("c7c5")
    assert session.move_history == (white_move, action.move)
    context = runner.contexts[0]
    assert context.target_rating == 1400
    assert len(context.plies) == 2
    assert all(not hasattr(ply, "target_rating") for ply in context.plies)


def test_greedy_mask_excludes_illegal_moves_and_disabled_resignation() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("e2e5"))] = 100.0
    logits[RESIGNATION_ACTION_ID] = 90.0
    logits[encode_move(chess.Move.from_uci("g1f3"))] = 10.0
    session = GameSession(
        StubRunner(logits),
        controlled_color=chess.WHITE,
        config=RuntimeConfig(temperature=0.0),
    )

    action = session.choose_action()

    assert isinstance(action, MoveAction)
    assert action.move == chess.Move.from_uci("g1f3")
    assert action.move in chess.Board().legal_moves


def test_enabled_resignation_is_preserved_and_ends_the_session() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[RESIGNATION_ACTION_ID] = 10.0
    session = GameSession(
        StubRunner(logits),
        controlled_color=chess.WHITE,
        config=RuntimeConfig(temperature=0.0, resignation_enabled=True),
    )

    action = session.choose_action()

    assert action == ResignationAction()
    assert session.resigned_by == chess.WHITE
    assert session.is_terminal
    assert session.move_history == ()
    with pytest.raises(SessionStateError, match="terminal"):
        session.choose_action()


def test_seeded_sampling_is_repeatable_and_reset_restarts_the_stream() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    first = GameSession(
        StubRunner(logits),
        controlled_color=chess.WHITE,
        config=RuntimeConfig(temperature=1.0, seed=91),
    )
    second = GameSession(
        StubRunner(logits),
        controlled_color=chess.WHITE,
        config=RuntimeConfig(temperature=1.0, seed=91),
    )

    first_action = first.choose_action()
    second_action = second.choose_action()
    first.reset()
    repeated_action = first.choose_action()

    assert first_action == second_action == repeated_action


def test_reset_validates_history_and_defensively_owns_board_state() -> None:
    session = GameSession(
        StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        controlled_color=chess.BLACK,
    )
    move = chess.Move.from_uci("d2d4")
    session.reset(moves=(move,))
    exposed = session.board
    exposed.push(chess.Move.from_uci("d7d5"))

    assert session.move_history == (move,)
    with pytest.raises(SessionStateError, match="illegal at ply 0"):
        session.reset(moves=(chess.Move.from_uci("e2e5"),))


def test_turn_terminal_and_observed_move_failures_are_deliberate() -> None:
    runner = StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE))
    black = GameSession(runner, controlled_color=chess.BLACK)
    with pytest.raises(SessionStateError, match="not black's turn"):
        black.choose_action()
    with pytest.raises(SessionStateError, match="illegal move"):
        black.apply_move(chess.Move.from_uci("e2e5"))

    terminal = GameSession(runner, controlled_color=chess.BLACK)
    terminal.reset(
        moves=tuple(
            chess.Move.from_uci(move) for move in ("f2f3", "e7e5", "g2g4", "d8h4")
        )
    )
    assert terminal.is_terminal
    with pytest.raises(SessionStateError, match="terminal"):
        terminal.choose_action()
    with pytest.raises(SessionStateError, match="terminal"):
        terminal.apply_move(chess.Move.null())


@pytest.mark.parametrize(
    ("logits", "message"),
    [
        (torch.zeros(ACTION_VOCABULARY_SIZE - 1), "invalid action-logit shape"),
        (
            torch.full((ACTION_VOCABULARY_SIZE,), float("nan")),
            "non-finite action logits",
        ),
    ],
)
def test_malformed_model_outputs_fail(logits: torch.Tensor, message: str) -> None:
    session = GameSession(
        StubRunner(logits),
        controlled_color=chess.WHITE,
        config=RuntimeConfig(temperature=0.0),
    )

    with pytest.raises(ActionSelectionError, match=message):
        session.choose_action()
    assert session.move_history == ()


def test_runtime_config_uses_shared_loading_and_enforces_control_bounds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.toml"
    path.write_text(
        "target_rating = 1600\ntemperature = 0.75\nseed = 17\n",
        encoding="utf-8",
    )

    resolved = load_config(RuntimeConfig, path=path)

    assert resolved.value == RuntimeConfig(
        target_rating=1600,
        temperature=0.75,
        seed=17,
    )
    assert resolved.provenance.source == str(path.resolve())
    with pytest.raises(ValidationError):
        RuntimeConfig(temperature=-0.01)
    with pytest.raises(ValidationError):
        RuntimeConfig(temperature=float("inf"))
    with pytest.raises(ValidationError):
        RuntimeConfig(seed=-1)


def _ranked_logits(best: str, *, illegal: str | None = None) -> torch.Tensor:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci(best))] = 10.0
    if illegal is not None:
        logits[encode_move(chess.Move.from_uci(illegal))] = 100.0
    return logits
