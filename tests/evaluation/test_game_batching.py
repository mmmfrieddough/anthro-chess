"""Playing several generated games at once without changing what is measured.

Concurrency is an optimization, so almost every test here compares a batched
run against the sequential run of the same suite. What varies is scheduling;
what must not vary is any game.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import chess
import pytest
import torch

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    RESIGNATION_ACTION_ID,
    encode_move,
)
from anthro_chess.data import DecisionContext, DecisionHistory
from anthro_chess.evaluation.games import (
    DecisionRequest,
    ExternalEnginePlayer,
    GamePlayer,
    GameRecord,
    GameTermination,
    GenerationConfig,
    ModelPlayer,
    PlayerError,
    RandomPlayer,
    StartPosition,
    generate_games,
    standard_positions,
)
from anthro_chess.runtime import ActionModelRunner, RuntimeConfig


@dataclass
class RecordingRunner:
    """A stand-in policy that reports how many decisions it was asked for.

    The same logits come back whether one context or ten were asked about, so
    any difference between a batched and a sequential suite is scheduling
    rather than a different model.
    """

    batch_sizes: list[int] = field(default_factory=list)

    def predict(self, context: DecisionContext) -> torch.Tensor:
        self.batch_sizes.append(1)
        return _policy(context)

    def predict_batch(
        self,
        contexts: tuple[DecisionContext, ...],
    ) -> tuple[torch.Tensor, ...]:
        self.batch_sizes.append(len(contexts))
        return tuple(_policy(context) for context in contexts)


@dataclass
class SequentialOnlyRunner:
    """A stand-in policy with no batched entry point at all."""

    calls: int = 0

    def predict(self, context: DecisionContext) -> torch.Tensor:
        self.calls += 1
        return _policy(context)


@dataclass
class DeadlineRunner:
    """A policy that resigns once its own game reaches a chosen length.

    Games in one wave otherwise run to the same ply limit and end together,
    which is the case that would hide a scheduler that mishandles a finished
    game. Keying the deadline on the board the opening move produced gives each
    game in a wave its own ending ply.
    """

    deadlines: dict[bytes, int]
    batch_sizes: list[int] = field(default_factory=list)

    def predict(self, context: DecisionContext) -> torch.Tensor:
        self.batch_sizes.append(1)
        return self._logits(context)

    def predict_batch(
        self,
        contexts: tuple[DecisionContext, ...],
    ) -> tuple[torch.Tensor, ...]:
        self.batch_sizes.append(len(contexts))
        return tuple(self._logits(context) for context in contexts)

    def _logits(self, context: DecisionContext) -> torch.Tensor:
        opening = context.plies[1].board.piece_ids
        if len(context.plies) >= self.deadlines[opening]:
            resigning = torch.zeros(ACTION_VOCABULARY_SIZE)
            resigning[RESIGNATION_ACTION_ID] = 10.0
            return resigning
        return _policy(context)


@dataclass
class FakeEngine:
    """A deterministic external opponent that needs no process."""

    new_games: int = 0
    closed: bool = False

    def new_game(self) -> None:
        self.new_games += 1

    def play(self, board: chess.Board) -> chess.Move:
        return min(board.legal_moves, key=encode_move)

    def close(self) -> None:
        self.closed = True


def _policy(context: DecisionContext) -> torch.Tensor:
    """Return one reproducible distribution for a trajectory.

    Resignation is pushed below every move so a game ends where a test says it
    does rather than wherever the random draw happened to favor it.
    """

    board = context.plies[-1].board
    key = hash((len(context.plies), board.piece_ids, board.side_to_move))
    generator = torch.Generator().manual_seed(key % (2**31))
    logits = torch.randn(ACTION_VOCABULARY_SIZE, generator=generator)
    logits[RESIGNATION_ACTION_ID] = -100.0
    return logits


def _model(
    runner: ActionModelRunner,
    *,
    label: str,
    resignation: bool = False,
) -> ModelPlayer:
    return ModelPlayer(
        runner,
        label=label,
        config=RuntimeConfig(
            target_rating=1500,
            temperature=0.0 if resignation else 1.0,
            resignation_enabled=resignation,
        ),
    )


def _play(
    first: GamePlayer,
    second: GamePlayer,
    *,
    concurrency: int,
    positions: tuple[StartPosition, ...] | None = None,
    maximum_generated_plies: int = 6,
) -> tuple[GameRecord, ...]:
    return tuple(
        generate_games(
            first,
            second,
            positions or standard_positions(4),
            config=GenerationConfig(
                seed=23,
                swap_colors=False,
                maximum_generated_plies=maximum_generated_plies,
                concurrency=concurrency,
            ),
        )
    )


def _written(records: tuple[GameRecord, ...]) -> str:
    """Return the suite exactly as the detail tier would store it."""

    return "\n".join(
        json.dumps(record.as_record(), sort_keys=True) for record in records
    )


def _openings(count: int) -> tuple[StartPosition, ...]:
    """Return roots that differ by their first move, one per game."""

    moves = ("e2e4", "d2d4", "c2c4", "g1f3")
    return tuple(
        StartPosition(prefix_action_ids=(encode_move(chess.Move.from_uci(move)),))
        for move in moves[:count]
    )


def test_a_batched_wave_plays_the_same_games_as_the_sequential_path() -> None:
    sequential = _play(
        _model(RecordingRunner(), label="white"),
        _model(RecordingRunner(), label="black"),
        concurrency=1,
    )
    batched = _play(
        _model(RecordingRunner(), label="white"),
        _model(RecordingRunner(), label="black"),
        concurrency=4,
    )

    assert _written(batched) == _written(sequential)


def test_a_wave_asks_one_configuration_for_every_pending_game_at_once() -> None:
    runner = RecordingRunner()
    player = _model(runner, label="self-play")

    _play(player, player, concurrency=4)

    assert runner.batch_sizes[0] == 4
    assert 1 not in runner.batch_sizes


def test_each_player_configuration_is_asked_for_its_own_seats_only() -> None:
    white = RecordingRunner()
    black = RecordingRunner()

    _play(
        _model(white, label="white"),
        _model(black, label="black"),
        concurrency=4,
        positions=_openings(4),
    )

    # Every game starts one ply into the opening, so black moves first in all
    # four and the two configurations are never mixed into one pass.
    assert set(white.batch_sizes) == {4}
    assert set(black.batch_sizes) == {4}


def _board_after(move: str) -> bytes:
    """Return the encoded board one opening move produces, as a deadline key."""

    history = DecisionHistory(moves=(chess.Move.from_uci(move),))
    return history.context(target_rating=None).plies[-1].board.piece_ids


def test_games_that_end_mid_wave_leave_the_rest_of_the_wave_alone() -> None:
    deadlines = {
        _board_after(move): plies
        for move, plies in (("e2e4", 3), ("d2d4", 5), ("c2c4", 9), ("g1f3", 9))
    }
    batched_runner = DeadlineRunner(deadlines)
    batched_player = _model(batched_runner, label="deadline", resignation=True)
    sequential_player = _model(
        DeadlineRunner(deadlines),
        label="deadline",
        resignation=True,
    )

    sequential = _play(
        sequential_player,
        sequential_player,
        concurrency=1,
        positions=_openings(4),
    )
    batched = _play(
        batched_player,
        batched_player,
        concurrency=4,
        positions=_openings(4),
    )

    assert _written(batched) == _written(sequential)
    assert [record.generated_plies for record in batched] == [2, 4, 6, 6]
    assert {record.outcome.termination for record in batched} == {
        GameTermination.RESIGNATION,
        GameTermination.PLY_LIMIT,
    }
    # The wave shrinks as games finish instead of asking about ended games.
    assert batched_runner.batch_sizes == [4, 4, 3, 3, 2, 2]


def test_a_runner_without_a_batched_path_still_plays_a_wave() -> None:
    runner = SequentialOnlyRunner()

    batched = _play(
        _model(runner, label="white"),
        _model(SequentialOnlyRunner(), label="black"),
        concurrency=4,
    )
    sequential = _play(
        _model(SequentialOnlyRunner(), label="white"),
        _model(SequentialOnlyRunner(), label="black"),
        concurrency=1,
    )

    assert _written(batched) == _written(sequential)
    assert runner.calls > 0


def test_an_external_engine_seat_keeps_the_suite_at_one_game_at_a_time() -> None:
    runner = RecordingRunner()
    engine = FakeEngine()

    records = _play(
        _model(runner, label="model"),
        ExternalEnginePlayer(engine, label="engine"),
        concurrency=4,
    )

    assert set(runner.batch_sizes) == {1}
    assert engine.new_games == len(records)


def test_a_wave_of_random_seats_needs_no_model_at_all() -> None:
    first = RandomPlayer(label="left")
    second = RandomPlayer(label="right")

    batched = _play(first, second, concurrency=4)
    sequential = _play(first, second, concurrency=1)

    assert _written(batched) == _written(sequential)


def test_a_model_player_refuses_to_resolve_a_seat_that_is_not_its_own() -> None:
    player = _model(RecordingRunner(), label="model")
    foreign = RandomPlayer().seat(seed=1)
    request = _request()

    with pytest.raises(PlayerError, match="its own seats"):
        player.decide_batch(((foreign, request),))


def test_a_runner_returning_the_wrong_number_of_decisions_fails_loudly() -> None:
    class ShortRunner(RecordingRunner):
        def predict_batch(
            self,
            contexts: tuple[DecisionContext, ...],
        ) -> tuple[torch.Tensor, ...]:
            return super().predict_batch(contexts)[:-1]

    player = _model(ShortRunner(), label="model")
    seats = (player.seat(seed=1), player.seat(seed=2))
    request = _request()

    with pytest.raises(PlayerError, match="wrong number of decisions"):
        player.decide_batch(tuple((seat, request) for seat in seats))


def _request() -> DecisionRequest:
    return DecisionRequest(
        board=chess.Board(),
        initial_position=chess.STARTING_FEN,
        ply_index=0,
    )
