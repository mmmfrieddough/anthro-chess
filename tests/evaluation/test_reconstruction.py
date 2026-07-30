from __future__ import annotations

import io
import json
from dataclasses import dataclass

import chess
import pytest
import torch

from anthro_chess.application_logging import configure_application_logging
from anthro_chess.chess import ACTION_VOCABULARY_SIZE, encode_move
from anthro_chess.data import DecisionContext
from anthro_chess.evaluation.decisions import (
    score_played_decisions,
    summarize_decisions,
)
from anthro_chess.evaluation.reconstruction import (
    ReconstructionError,
    reconstruct_uci_games,
)
from anthro_chess.interfaces.config import UciConfig
from anthro_chess.interfaces.uci import UCI_GAME_EVENT_SCHEMA, UciEngine
from anthro_chess.runtime import RuntimeConfig

NONSTANDARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"


@dataclass
class StubRunner:
    logits: torch.Tensor

    def predict(self, _context: DecisionContext) -> torch.Tensor:
        return self.logits.clone()


@dataclass
class TrajectoryRunner:
    """A policy that depends on how much history it was given.

    Enough to show that a re-scored decision was scored from its own prefix
    rather than from whatever position happened to be current.
    """

    def predict(self, context: DecisionContext) -> torch.Tensor:
        generator = torch.Generator().manual_seed(len(context.plies))
        return torch.randn(ACTION_VOCABULARY_SIZE, generator=generator)


def _play(transcript: str, *, preferred: str = "g1f3", seed: int = 123) -> str:
    """Drive one real UCI session and return its diagnostic log."""

    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci(preferred))] = 10.0
    engine = UciEngine(
        lambda: StubRunner(logits),
        UciConfig(runtime=RuntimeConfig(seed=seed, temperature=0.0)),
    )
    error = io.StringIO()
    configure_application_logging(level="WARNING", stream=error)
    assert engine.run(io.StringIO(transcript), io.StringIO(), error) == 0
    return error.getvalue()


def _event(**fields: object) -> str:
    payload = {
        "schema": UCI_GAME_EVENT_SCHEMA,
        "session_id": "abc",
        "game_index": 0,
        **fields,
    }
    return f"DEBUG UCI game event {json.dumps(payload, sort_keys=True)}"


def _runtime(**fields: object) -> dict[str, object]:
    return {
        "target_rating": 1500,
        "temperature": 1.0,
        "resignation_enabled": False,
        "configured_seed": None,
        "resolved_seed": 4,
        **fields,
    }


def test_a_played_session_reconstructs_its_own_decisions() -> None:
    log = _play(
        "\n".join(
            (
                "debug on",
                "position startpos moves e2e4 e7e5",
                "go",
                "position startpos moves e2e4 e7e5 g1f3 b8c6",
                "go",
                "quit",
            )
        )
    )

    games = reconstruct_uci_games(log.splitlines())

    assert len(games) == 1
    game = games[0]
    assert game.initial_fen == chess.STARTING_FEN
    assert [move.uci() for move in game.moves[:4]] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert [decision.ply_index for decision in game.decisions] == [2, 4]
    assert [len(decision.history) for decision in game.decisions] == [2, 4]
    assert game.decisions[0].action_id == encode_move(chess.Move.from_uci("g1f3"))
    assert encode_move(game.moves[4]) == game.decisions[1].action_id
    assert game.decisions[0].config.temperature == 0.0
    assert game.decisions[0].config.seed == 123
    assert game.game_id.endswith(":0")


def test_a_takeback_leaves_the_history_the_engine_actually_decided_from() -> None:
    """The replaced position is the one in force, so the decision carries it."""

    log = _play(
        "\n".join(
            (
                "debug on",
                "position startpos moves e2e4 e7e5 g1f3 b8c6",
                "position startpos moves e2e4 e7e5",
                "go",
                "quit",
            )
        )
    )

    game = reconstruct_uci_games(log.splitlines())[0]

    assert [decision.ply_index for decision in game.decisions] == [2]
    assert [move.uci() for move in game.decisions[0].history] == ["e2e4", "e7e5"]


def test_a_new_game_boundary_starts_a_separate_reconstruction() -> None:
    log = _play(
        "\n".join(
            (
                "debug on",
                "position startpos moves e2e4",
                "go",
                "ucinewgame",
                f"position fen {NONSTANDARD_FEN}",
                "go",
                "quit",
            )
        ),
        preferred="e7e5",
    )

    games = reconstruct_uci_games(log.splitlines())

    assert [game.game_index for game in games] == [0, 1]
    assert games[0].initial_fen == chess.STARTING_FEN
    assert games[1].initial_fen == NONSTANDARD_FEN
    assert [len(game.decisions) for game in games] == [1, 1]
    assert {game.game_id for game in games} == {
        f"{games[0].session_id}:0",
        f"{games[0].session_id}:1",
    }


def test_lines_without_game_events_are_ignored() -> None:
    log = _play("\n".join(("debug on", "position startpos moves e2e4", "go", "quit")))
    noisy = "\n".join(("INFO Model runtime initialized", log, "INFO quitting"))

    assert len(reconstruct_uci_games(noisy.splitlines())) == 1


def test_an_unknown_event_schema_is_an_error_rather_than_a_skip() -> None:
    line = 'DEBUG UCI game event {"schema": "anthro-uci-game-event-v9"}'

    with pytest.raises(ReconstructionError, match="this build reads"):
        reconstruct_uci_games([line])


def test_an_unreadable_event_payload_names_its_line() -> None:
    with pytest.raises(ReconstructionError, match="line 2"):
        reconstruct_uci_games(["nothing here", "DEBUG UCI game event {oops"])


def test_a_log_whose_replay_disagrees_with_its_recorded_position_is_rejected() -> None:
    """A truncated or interleaved log fails instead of being quietly wrong."""

    lines = [
        _event(
            event="position",
            initial_fen=chess.STARTING_FEN,
            moves=["e2e4"],
            position_fen=chess.STARTING_FEN,
            transition="validated",
        )
    ]

    with pytest.raises(ReconstructionError, match="replays to"):
        reconstruct_uci_games(lines)


def test_a_decision_that_is_illegal_in_its_reconstructed_position_is_rejected() -> None:
    board = chess.Board()
    lines = [
        _event(
            event="decision",
            action_id=encode_move(chess.Move.from_uci("e7e5")),
            move="e7e5",
            observed_plies=0,
            position_fen=board.fen(),
            runtime=_runtime(),
        )
    ]

    with pytest.raises(ReconstructionError, match="illegal"):
        reconstruct_uci_games(lines)


def test_a_decision_disagreeing_about_observed_plies_is_rejected() -> None:
    board = chess.Board()
    lines = [
        _event(
            event="decision",
            action_id=encode_move(chess.Move.from_uci("e2e4")),
            move="e2e4",
            observed_plies=7,
            position_fen=board.fen(),
            runtime=_runtime(),
        )
    ]

    with pytest.raises(ReconstructionError, match="observed plies"):
        reconstruct_uci_games(lines)


def test_a_decision_event_without_runtime_settings_is_rejected() -> None:
    board = chess.Board()
    lines = [
        _event(
            event="decision",
            action_id=encode_move(chess.Move.from_uci("e2e4")),
            move="e2e4",
            observed_plies=0,
            position_fen=board.fen(),
        )
    ]

    with pytest.raises(ReconstructionError, match="no runtime settings"):
        reconstruct_uci_games(lines)


def test_a_reconstructed_session_decomposes_through_the_shared_layer() -> None:
    """The whole point: a manually played game analyzed like a rollout."""

    log = _play(
        "\n".join(
            (
                "debug on",
                "position startpos moves e2e4 e7e5",
                "go",
                "position startpos moves e2e4 e7e5 g1f3 b8c6",
                "go",
                "quit",
            )
        )
    )
    decisions = tuple(
        decision
        for game in reconstruct_uci_games(log.splitlines())
        for decision in game.decisions
    )

    decomposition = summarize_decisions(
        score_played_decisions(TrajectoryRunner(), decisions)
    )

    assert decomposition.overall.decisions == 2
    assert decomposition.reference_cell().setting is not None
    assert 1.0 <= decomposition.overall.selected_rank <= 40.0
    assert [sample.ply_index for sample in decomposition.samples] == [2, 4]
