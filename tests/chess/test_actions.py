import pytest

from anthro_chess.chess import (
    RESIGN,
    STANDARD_ACTION_CODEC,
    InvalidActionError,
    Move,
    Position,
    Promotion,
)


@pytest.mark.parametrize(
    "move",
    [
        Move.from_uci("e2e4"),
        Move.from_uci("e1g1"),
        Move.from_uci("e5d6"),
        Move(48, 56, Promotion.QUEEN),
        Move(48, 57, Promotion.KNIGHT),
    ],
)
def test_move_actions_round_trip(move: Move) -> None:
    action_id = STANDARD_ACTION_CODEC.encode(move)

    assert STANDARD_ACTION_CODEC.decode(action_id) == move
    assert 0 <= action_id < 65536


def test_resignation_is_explicit_and_not_a_board_move() -> None:
    assert STANDARD_ACTION_CODEC.decode(STANDARD_ACTION_CODEC.resignation_id) is RESIGN
    assert (
        STANDARD_ACTION_CODEC.resignation_id
        >= STANDARD_ACTION_CODEC.move_vocabulary_size
    )


def test_legal_ids_share_the_same_codec_and_gate_resignation() -> None:
    position = Position.initial()
    move_ids = STANDARD_ACTION_CODEC.legal_action_ids(position)
    action_ids = STANDARD_ACTION_CODEC.legal_action_ids(
        position, include_resignation=True
    )

    assert len(move_ids) == 20
    assert tuple(STANDARD_ACTION_CODEC.decode(value) for value in move_ids) == tuple(
        sorted(
            position.legal_moves, key=lambda move: STANDARD_ACTION_CODEC.encode(move)
        )
    )
    assert action_ids == (*move_ids, STANDARD_ACTION_CODEC.resignation_id)


def test_vocabulary_identity_is_stable_and_serializable() -> None:
    assert STANDARD_ACTION_CODEC.identity.as_record() == {
        "name": "anthro-standard-actions",
        "version": 1,
        "size": 1969,
        "sha256": "f95e6069227ad773de35c12f9601d89b622da0539d7793ff88232aff368a48d6",
    }


@pytest.mark.parametrize("action_id", [-1, True, 65536])
def test_rejects_invalid_action_ids(action_id: int) -> None:
    with pytest.raises(InvalidActionError):
        STANDARD_ACTION_CODEC.decode(action_id)
