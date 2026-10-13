"""The parts of the played move a move-component head is trained to name."""

from __future__ import annotations

import dataclasses

import chess
import pytest
import torch
from tiny_models import tiny_model_config

from anthro_chess.chess import RESIGNATION_ACTION_ID, encode_move
from anthro_chess.data import (
    GameEncodingInput,
    SequenceExample,
    collate_sequences,
    encode_game,
)
from anthro_chess.models import MoveModel, MoveModelBatch
from anthro_chess.models.move_model import (
    IGNORED_COMPONENT,
    MOVE_COMPONENT_GROUPS,
    model_identity,
)

#: White takes on d5, then en passant on e6, and black recaptures; a knight
#: move closes it so a piece type other than the pawn is read.
_MOVES = ("e2e4", "d7d5", "e4d5", "e7e5", "d5e6", "f7e6", "g1f3")


def _batch() -> MoveModelBatch:
    plies = encode_game(
        GameEncodingInput(
            game_id=100,
            ruleset="standard",
            initial_position=chess.STARTING_FEN,
            action_ids=tuple(encode_move(chess.Move.from_uci(m)) for m in _MOVES),
            white_normalized_rating=1500,
            black_normalized_rating=1500,
            time_initial_ms=None,
            time_increment_ms=None,
            clock_remaining_ms=(None,) * len(_MOVES),
        )
    )
    example = SequenceExample(shard_index=0, game_id=100, start_ply=0, plies=plies)
    return MoveModelBatch.from_sequence_batch(collate_sequences((example,)))


def _model(weight: float = 1.0) -> MoveModel:
    torch.manual_seed(5)
    return MoveModel(tiny_model_config(move_component_weight=weight))


def test_each_decision_names_its_piece_capture_and_squares_from_the_movers_side() -> (
    None
):
    head = _model().move_component_head
    assert head is not None

    targets = head.targets(_batch())

    pawn, knight = chess.PAWN - 1, chess.KNIGHT - 1
    # Black's squares are mirrored, so its d7 reads as d2.
    assert targets[0].tolist() == [
        [pawn, 0, chess.E2, chess.E4],
        [pawn, 0, chess.D2, chess.D4],
        [pawn, chess.PAWN, chess.E4, chess.D5],
        [pawn, 0, chess.E2, chess.E4],
        [pawn, chess.PAWN, chess.D5, chess.E6],
        [pawn, chess.PAWN, chess.F2, chess.E3],
        [knight, 0, chess.G1, chess.F3],
    ]


def test_a_terminal_action_or_a_disabled_decision_has_no_components() -> None:
    head = _model().move_component_head
    assert head is not None
    batch = _batch()
    actions = batch.action_targets.clone()
    actions[0, -1] = RESIGNATION_ACTION_ID
    enabled = batch.action_loss_mask.clone()
    enabled[0, 0] = False
    batch = dataclasses.replace(batch, action_targets=actions, action_loss_mask=enabled)

    targets = head.targets(batch)

    assert (targets[0, 0] == IGNORED_COMPONENT).all()
    assert (targets[0, -1] == IGNORED_COMPONENT).all()
    assert (targets[0, 1:-1] != IGNORED_COMPONENT).all()


def test_the_component_pass_scores_moves_exactly_as_the_ordinary_one() -> None:
    model = _model().eval()
    batch = _batch()

    with torch.no_grad():
        logits, components = model(batch, components=True)
        ordinary = model(batch)

    torch.testing.assert_close(logits, ordinary, rtol=0.0, atol=0.0)
    assert components.shape == (*ordinary.shape[:2], sum(MOVE_COMPONENT_GROUPS))


def test_a_model_without_the_head_builds_none_and_refuses_the_component_pass() -> None:
    model = _model(weight=0.0)

    assert model.move_component_head is None
    with pytest.raises(ValueError, match="no move-component head"):
        model(_batch(), components=True)


def test_a_model_without_the_head_keeps_the_identity_recorded_before_it() -> None:
    without = model_identity(tiny_model_config())
    with_head = model_identity(tiny_model_config(move_component_weight=0.5))

    assert "move_component_weight" not in without["config"]  # type: ignore[operator]
    assert with_head["config"]["move_component_weight"] == 0.5  # type: ignore[index]
