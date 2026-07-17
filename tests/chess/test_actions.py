import chess
import pytest

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    MOVE_ACTION_COUNT,
    RESIGNATION_ACTION_ID,
    action_vocabulary_identity,
    decode_move,
    encode_move,
    legal_action_ids,
)


@pytest.mark.parametrize(
    "move",
    [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e1g1"),
        chess.Move.from_uci("e5d6"),
        chess.Move.from_uci("a7a8q"),
        chess.Move.from_uci("a7b8n"),
    ],
)
def test_move_actions_round_trip(move: chess.Move) -> None:
    action_id = encode_move(move)

    assert decode_move(action_id) == move
    assert 0 <= action_id < 65536


def test_resignation_has_an_explicit_non_move_id() -> None:
    assert RESIGNATION_ACTION_ID == MOVE_ACTION_COUNT
    assert ACTION_VOCABULARY_SIZE == MOVE_ACTION_COUNT + 1
    with pytest.raises(ValueError):
        decode_move(RESIGNATION_ACTION_ID)


def test_legal_ids_come_directly_from_python_chess() -> None:
    board = chess.Board()
    move_ids = legal_action_ids(board)

    assert len(move_ids) == 20
    assert {decode_move(action_id) for action_id in move_ids} == set(board.legal_moves)
    assert legal_action_ids(board, include_resignation=True) == (
        *move_ids,
        RESIGNATION_ACTION_ID,
    )


def test_vocabulary_identity_is_stable_and_serializable() -> None:
    assert action_vocabulary_identity() == {
        "name": "anthro-standard-actions",
        "version": 1,
        "size": 1969,
        "sha256": "f95e6069227ad773de35c12f9601d89b622da0539d7793ff88232aff368a48d6",
    }


@pytest.mark.parametrize("action_id", [-1, True, RESIGNATION_ACTION_ID, 65536])
def test_rejects_invalid_move_action_ids(action_id: int) -> None:
    with pytest.raises(ValueError):
        decode_move(action_id)


def test_rejects_move_outside_the_standard_vocabulary() -> None:
    with pytest.raises(ValueError):
        encode_move(chess.Move.null())
