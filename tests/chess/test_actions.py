import chess
import pytest

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    DRAW_CLAIM_ACTION_ID,
    MOVE_ACTION_COUNT,
    RESIGNATION_ACTION_ID,
    TERMINAL_ACTION_IDS,
    action_vocabulary_identity,
    decode_move,
    draw_claim_available,
    encode_move,
    is_terminal_action,
    legal_action_ids,
)


def _play(board: chess.Board, *moves: str) -> chess.Board:
    for move in moves:
        board.push(chess.Move.from_uci(move))
    return board


def _shuffle_knights(board: chess.Board, rounds: int) -> chess.Board:
    """Return the board after ``rounds`` reversible knight round trips."""

    for _ in range(rounds):
        _play(board, "g1f3", "g8f6", "f3g1", "f6g8")
    return board


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


def test_terminal_actions_have_explicit_non_move_ids() -> None:
    assert RESIGNATION_ACTION_ID == MOVE_ACTION_COUNT
    assert DRAW_CLAIM_ACTION_ID == MOVE_ACTION_COUNT + 1
    assert TERMINAL_ACTION_IDS == (RESIGNATION_ACTION_ID, DRAW_CLAIM_ACTION_ID)
    assert ACTION_VOCABULARY_SIZE == MOVE_ACTION_COUNT + 2
    for action_id in TERMINAL_ACTION_IDS:
        assert is_terminal_action(action_id)
        with pytest.raises(ValueError):
            decode_move(action_id)
    assert not is_terminal_action(MOVE_ACTION_COUNT - 1)


def test_legal_ids_come_directly_from_python_chess() -> None:
    board = chess.Board()
    move_ids = legal_action_ids(board)

    assert len(move_ids) == 20
    assert {decode_move(action_id) for action_id in move_ids} == set(board.legal_moves)
    assert legal_action_ids(board, include_resignation=True) == (
        *move_ids,
        RESIGNATION_ACTION_ID,
    )


def test_draw_claim_needs_the_rules_not_only_the_policy() -> None:
    board = chess.Board()

    assert not draw_claim_available(board)
    assert legal_action_ids(board, include_draw_claim=True) == legal_action_ids(board)


def test_draw_claim_appears_once_a_position_repeats_three_times() -> None:
    board = _shuffle_knights(chess.Board(), 1)
    move_ids = legal_action_ids(board)

    # Two occurrences so far: the starting position and this one.
    assert not draw_claim_available(board)

    _shuffle_knights(board, 1)

    assert draw_claim_available(board)
    assert legal_action_ids(board, include_draw_claim=True) == (
        *move_ids,
        DRAW_CLAIM_ACTION_ID,
    )
    assert legal_action_ids(
        board,
        include_resignation=True,
        include_draw_claim=True,
    ) == (*move_ids, RESIGNATION_ACTION_ID, DRAW_CLAIM_ACTION_ID)


def test_draw_claim_is_unavailable_one_ply_before_the_third_repetition() -> None:
    """A claim only an announced move would reach is not this action.

    One ply early the mover can still claim under the rules, by announcing the
    move that repeats the position a third time. ``python-chess`` reports that
    as claimable; the draw-claim action carries no move, so it is not offered
    until the position itself has repeated.
    """

    board = _shuffle_knights(chess.Board(), 1)
    _play(board, "g1f3", "g8f6", "f3g1")

    assert board.can_claim_threefold_repetition()
    assert not draw_claim_available(board)
    assert DRAW_CLAIM_ACTION_ID not in legal_action_ids(board, include_draw_claim=True)

    _play(board, "f6g8")

    assert draw_claim_available(board)


def test_draw_claim_appears_on_a_full_fifty_move_clock() -> None:
    board = chess.Board("8/8/8/4k3/8/8/4K3/7R w - - 99 60")

    assert not draw_claim_available(board)

    board.push(chess.Move.from_uci("h1h2"))

    assert board.halfmove_clock == 100
    assert draw_claim_available(board)
    assert DRAW_CLAIM_ACTION_ID in legal_action_ids(board, include_draw_claim=True)


def test_draw_claim_is_absent_where_the_game_already_ended() -> None:
    checkmate = chess.Board("7k/5QK1/8/8/8/8/8/8 b - - 100 60")

    assert checkmate.is_checkmate()
    assert not draw_claim_available(checkmate)


def test_vocabulary_identity_is_stable_and_serializable() -> None:
    assert action_vocabulary_identity() == {
        "name": "anthro-standard-actions",
        "version": 2,
        "size": 1970,
        "sha256": "4335db3058aad53dc1a4a873a21cc48e2db4d3e6aff003887260e831d3f677ab",
    }


@pytest.mark.parametrize(
    "action_id",
    [-1, True, RESIGNATION_ACTION_ID, DRAW_CLAIM_ACTION_ID, 65536],
)
def test_rejects_invalid_move_action_ids(action_id: int) -> None:
    with pytest.raises(ValueError):
        decode_move(action_id)


def test_rejects_move_outside_the_standard_vocabulary() -> None:
    with pytest.raises(ValueError):
        encode_move(chess.Move.null())
