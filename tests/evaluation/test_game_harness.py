from __future__ import annotations

from dataclasses import dataclass, field

import chess
import pytest
import torch

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    DRAW_CLAIM_ACTION_ID,
    RESIGNATION_ACTION_ID,
    encode_move,
)
from anthro_chess.data import DecisionContext
from anthro_chess.evaluation.games import (
    ExternalEnginePlayer,
    GamePlayer,
    GameRecord,
    GameTermination,
    GenerationConfig,
    GenerationError,
    ModelPlayer,
    PlayerError,
    RandomPlayer,
    StartPosition,
    generate_games,
    prefix_positions,
    standard_positions,
)
from anthro_chess.evaluation.results import CheckpointReference
from anthro_chess.runtime import ActionModelRunner, RuntimeConfig

#: Position after 1. f3 e5 2. g4, where Qh4 is mate.
FOOLS_MATE_FEN = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
#: Position after 1. f3 e5 2. g4 Qh4#, which is already over.
FINISHED_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
#: Eight reversible plies returning the starting position for a third time, so
#: the seat to move can claim a draw without announcing anything.
_REPEATED_POSITION_PREFIX = tuple(
    encode_move(chess.Move.from_uci(uci))
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8")
)


@dataclass
class TrajectoryRunner:
    """A stand-in policy that depends only on trajectory length and rating.

    Two seats configured with different ratings therefore see different
    distributions, which is what makes a mixed-configuration pairing
    observable without a trained checkpoint.
    """

    calls: int = 0

    def predict(self, context: DecisionContext) -> torch.Tensor:
        self.calls += 1
        generator = torch.Generator().manual_seed(
            len(context.plies) * 1000 + (context.target_rating or 0)
        )
        return torch.randn(ACTION_VOCABULARY_SIZE, generator=generator)


@dataclass
class PreferenceRunner:
    """A stand-in policy that ranks one action above every other."""

    preferred_action_id: int

    def predict(self, context: DecisionContext) -> torch.Tensor:
        logits = torch.zeros(ACTION_VOCABULARY_SIZE)
        logits[self.preferred_action_id] = 10.0
        return logits


@dataclass
class FakeEngine:
    """A deterministic external opponent that needs no process.

    Plays the lowest-numbered legal action, which is reproducible and has
    nothing to do with chess strength: the point of these tests is process
    handling, not play quality.
    """

    new_games: int = 0
    closed: bool = False
    played: list[str] = field(default_factory=list)
    forced_move: chess.Move | None = None

    def new_game(self) -> None:
        self.new_games += 1

    def play(self, board: chess.Board) -> chess.Move:
        if self.forced_move is not None:
            return self.forced_move
        move = min(board.legal_moves, key=encode_move)
        self.played.append(move.uci())
        return move

    def close(self) -> None:
        self.closed = True


def _model(
    *,
    rating: int | None = 1500,
    temperature: float = 1.0,
    label: str = "model",
    runner: ActionModelRunner | None = None,
    resignation: bool = False,
    draw_claim: bool = False,
) -> ModelPlayer:
    return ModelPlayer(
        runner or TrajectoryRunner(),
        label=label,
        config=RuntimeConfig(
            target_rating=rating,
            temperature=temperature,
            resignation_enabled=resignation,
            draw_claim_enabled=draw_claim,
        ),
    )


def _play(
    first: GamePlayer,
    second: GamePlayer,
    *,
    positions: tuple[StartPosition, ...] | None = None,
    **config: object,
) -> tuple[GameRecord, ...]:
    settings: dict[str, object] = {
        "maximum_generated_plies": 6,
        "swap_colors": False,
    }
    settings.update(config)
    return tuple(
        generate_games(
            first,
            second,
            positions or standard_positions(),
            config=GenerationConfig(**settings),  # type: ignore[arg-type]
        )
    )


def test_one_seed_reproduces_a_suite_exactly() -> None:
    first = _play(_model(), _model(label="other"), seed=17)
    again = _play(_model(), _model(label="other"), seed=17)
    different = _play(_model(), _model(label="other"), seed=18)

    assert [record.game_id for record in first] == [record.game_id for record in again]
    assert [record.action_ids for record in first] != [
        record.action_ids for record in different
    ]


def test_a_game_records_the_seed_that_reproduces_it_on_its_own() -> None:
    (record,) = _play(_model(), _model(label="other"), seed=5)

    assert record.seed > 0
    assert record.white.seed is not None
    assert record.black.seed is not None
    assert record.white.seed != record.black.seed


def test_both_color_assignments_are_played_without_special_casing() -> None:
    records = _play(
        _model(label="alpha"),
        _model(label="beta"),
        swap_colors=True,
    )

    assert len(records) == 2
    assert [(record.white.label, record.black.label) for record in records] == [
        ("alpha", "beta"),
        ("beta", "alpha"),
    ]


def test_a_mixed_rating_pairing_gives_each_seat_its_own_configuration() -> None:
    records = _play(
        _model(rating=1200, label="low"),
        _model(rating=2200, label="high"),
    )

    (record,) = records
    assert record.white.configuration["target_rating"] == 1200
    assert record.black.configuration["target_rating"] == 2200
    assert "seed" not in record.white.configuration


def test_a_checkpoint_identity_travels_with_the_seat() -> None:
    player = ModelPlayer(
        TrajectoryRunner(),
        label="step-100",
        checkpoint=CheckpointReference(label="step-100", step=100),
    )

    (record,) = _play(player, _model(label="other"))

    assert record.white.checkpoint is not None
    assert record.white.checkpoint.step == 100


def test_a_frozen_prefix_is_continued_rather_than_replayed_as_decisions() -> None:
    prefix = tuple(
        encode_move(chess.Move.from_uci(uci))
        for uci in ("e2e4", "e7e5", "g1f3", "b8c6")
    )
    positions = prefix_positions([(77, prefix)], plies=4)

    (record,) = _play(_model(), _model(label="other"), positions=positions)

    assert record.prefix_plies == 4
    assert record.action_ids[:4] == prefix
    assert record.generated_plies == 6
    assert record.source_game_id == 77
    assert record.position_label == "prefix-4"
    assert [decision.ply_index for decision in record.decisions] == list(range(4, 10))


def test_a_prefix_longer_than_its_source_game_is_dropped() -> None:
    short = (encode_move(chess.Move.from_uci("e2e4")),)

    with pytest.raises(GenerationError, match="long enough"):
        prefix_positions([(1, short)], plies=4)


def test_a_game_ends_when_the_rules_end_it() -> None:
    mate = encode_move(chess.Move.from_uci("d8h4"))
    black = _model(runner=PreferenceRunner(mate), temperature=0.0, label="mater")

    (record,) = _play(
        _model(label="white"),
        black,
        positions=(StartPosition(initial_position=FOOLS_MATE_FEN),),
    )

    assert record.outcome.termination is GameTermination.CHECKMATE
    assert record.outcome.result == "0-1"
    assert not record.outcome.adjudicated
    assert record.generated_plies == 1


def test_a_position_that_is_already_over_yields_a_game_with_no_decisions() -> None:
    (record,) = _play(
        _model(),
        _model(label="other"),
        positions=(StartPosition(initial_position=FINISHED_FEN),),
    )

    assert record.generated_plies == 0
    assert record.outcome.termination is GameTermination.CHECKMATE


def test_an_unfinished_game_is_adjudicated_at_the_ply_limit() -> None:
    (record,) = _play(_model(), _model(label="other"), maximum_generated_plies=4)

    assert record.generated_plies == 4
    assert record.outcome.termination is GameTermination.PLY_LIMIT
    assert record.outcome.result == "*"
    assert record.outcome.adjudicated


def test_a_resignation_ends_the_game_and_decides_it() -> None:
    resigner = _model(
        runner=PreferenceRunner(RESIGNATION_ACTION_ID),
        temperature=0.0,
        resignation=True,
        label="resigner",
    )

    (record,) = _play(resigner, _model(label="other"))

    assert record.action_ids == (RESIGNATION_ACTION_ID,)
    assert record.outcome.termination is GameTermination.RESIGNATION
    assert record.outcome.result == "0-1"


def test_a_seat_claim_ends_the_game_as_the_rule_that_allowed_it() -> None:
    """A claimed draw is the same ending however it was settled.

    What changes is who settled it: the seat chose the claim, so the record
    must not report the harness adjudicating on its behalf.
    """

    claimer = _model(
        runner=PreferenceRunner(DRAW_CLAIM_ACTION_ID),
        temperature=0.0,
        draw_claim=True,
        label="claimer",
    )

    (record,) = _play(
        claimer,
        _model(label="other"),
        positions=(StartPosition(prefix_action_ids=_REPEATED_POSITION_PREFIX),),
    )

    assert record.action_ids[-1] == DRAW_CLAIM_ACTION_ID
    assert record.generated_plies == 1
    assert record.outcome.termination is GameTermination.THREEFOLD_REPETITION
    assert record.outcome.result == "1/2-1/2"
    assert not record.outcome.adjudicated


def test_a_claimable_position_is_played_on_when_no_seat_claims_it() -> None:
    records = _play(
        _model(),
        _model(label="other"),
        positions=(StartPosition(prefix_action_ids=_REPEATED_POSITION_PREFIX),),
        maximum_generated_plies=2,
    )

    (record,) = records
    assert record.outcome.termination is GameTermination.PLY_LIMIT
    assert DRAW_CLAIM_ACTION_ID not in record.action_ids


def test_a_prefix_cannot_carry_a_terminal_action() -> None:
    position = StartPosition(prefix_action_ids=(DRAW_CLAIM_ACTION_ID,))

    with pytest.raises(GenerationError, match="terminal action"):
        position.board()


def test_each_decision_carries_the_policy_the_runtime_sampled_from() -> None:
    (record,) = _play(_model(), _model(label="other"))

    for decision in record.decisions:
        assert decision.policy is not None
        assert decision.policy.enabled_action_count >= 1
        assert decision.policy.selected_rank >= 1
        assert (
            decision.policy.selected_probability
            <= decision.policy.preferred_probability
        )


def test_a_random_opponent_is_reproducible_and_reports_no_preference() -> None:
    first = _play(RandomPlayer(), RandomPlayer(label="second"), seed=3)
    again = _play(RandomPlayer(), RandomPlayer(label="second"), seed=3)

    assert first[0].action_ids == again[0].action_ids
    assert all(decision.policy is None for decision in first[0].decisions)
    assert first[0].white.kind == "random"


def test_a_model_can_be_paired_with_a_random_opponent() -> None:
    (record,) = _play(_model(label="model"), RandomPlayer())

    assert record.white.kind == "model"
    assert record.black.kind == "random"
    assert record.generated_plies == 6


def test_an_external_engine_plays_a_seat_and_marks_each_game() -> None:
    engine = FakeEngine()
    player = ExternalEnginePlayer(engine, label="fake", configuration={"depth": 1})

    records = _play(_model(label="model"), player, swap_colors=True)

    assert len(records) == 2
    assert engine.new_games == 2
    assert not engine.closed
    assert records[0].black.kind == "external-engine"
    assert records[0].black.seed is None
    assert records[0].black.configuration == {"depth": 1}
    assert all(
        decision.policy is None
        for record in records
        for decision in record.decisions
        if record.seat(decision.slot).kind == "external-engine"
    )

    player.close()
    assert engine.closed


def test_an_external_engine_returning_an_illegal_move_fails_loudly() -> None:
    engine = FakeEngine(forced_move=chess.Move.from_uci("a1a8"))
    player = ExternalEnginePlayer(engine, label="broken")

    with pytest.raises(PlayerError, match="illegal move"):
        _play(player, _model(label="model"))


def test_a_suite_needs_at_least_one_position() -> None:
    with pytest.raises(GenerationError, match="at least one position"):
        tuple(generate_games(_model(), _model(label="other"), ()))
    with pytest.raises(GenerationError, match="at least one position"):
        standard_positions(0)


def test_a_prefix_containing_an_illegal_move_is_refused() -> None:
    prefix = (
        encode_move(chess.Move.from_uci("e2e4")),
        encode_move(chess.Move.from_uci("e2e4")),
    )

    with pytest.raises(GenerationError, match="illegal"):
        _play(
            _model(),
            _model(label="other"),
            positions=(StartPosition(prefix_action_ids=prefix),),
        )
