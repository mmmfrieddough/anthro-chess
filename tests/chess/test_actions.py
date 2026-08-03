import chess
import pytest

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    DRAW_CLAIM_ACTION_ID,
    MOVE_ACTION_COUNT,
    RESIGNATION_ACTION_ID,
    TERMINAL_ACTION_IDS,
    action_is_legal,
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


@pytest.mark.parametrize(
    ("name", "fen", "moves"),
    [
        ("opening", chess.STARTING_FEN, ()),
        # Both sides castling both ways, which is where the two answers most
        # nearly diverged: a castle can be spelled as the king taking its own
        # rook, and only one of those spellings is a legal action id.
        (
            "castling both ways",
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            (),
        ),
        (
            "en passant available",
            chess.STARTING_FEN,
            ("e2e4", "a7a6", "e4e5", "d7d5"),
        ),
        ("promotions", "8/P6k/8/8/8/8/6K1/8 w - - 0 1", ()),
        (
            "in check",
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
            (),
        ),
        ("checkmate", "7k/5QK1/8/8/8/8/8/8 b - - 100 60", ()),
        ("stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 60", ()),
        ("full fifty-move clock", "8/8/8/4k3/8/8/4K3/6R1 b - - 100 60", ()),
        # The claim a position has repeated three times, which no single
        # position carries: it is the move stack that makes it available.
        (
            "third repetition",
            chess.STARTING_FEN,
            ("g1f3", "g8f6", "f3g1", "f6g8") * 2,
        ),
        # The last id in the vocabulary is legal here, so a check that dropped
        # its lower bound and wrapped around to it is caught rather than
        # answering correctly by accident.
        ("king on the last vocabulary square", "7k/8/8/8/8/8/8/K7 b - - 0 1", ()),
    ],
)
def test_one_candidate_answers_exactly_what_the_whole_set_answers(
    name: str,
    fen: str,
    moves: tuple[str, ...],
) -> None:
    """The cheap check is what the encoding trusts, so it must not drift."""

    board = _play(chess.Board(fen), *moves)
    for include_resignation in (False, True):
        for include_draw_claim in (False, True):
            enabled = frozenset(
                legal_action_ids(
                    board,
                    include_resignation=include_resignation,
                    include_draw_claim=include_draw_claim,
                )
            )
            for action_id in range(-1, ACTION_VOCABULARY_SIZE + 1):
                assert action_is_legal(
                    board,
                    action_id,
                    include_resignation=include_resignation,
                    include_draw_claim=include_draw_claim,
                ) is (action_id in enabled), (
                    name,
                    action_id,
                    include_resignation,
                    include_draw_claim,
                )


def test_a_castle_spelled_as_taking_the_rook_is_not_a_legal_action() -> None:
    """``python-chess`` accepts both spellings; only one is in the vocabulary."""

    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    king_side = chess.Move.from_uci("e1g1")
    takes_rook = chess.Move.from_uci("e1h1")

    assert board.is_legal(king_side) and board.is_legal(takes_rook)
    assert encode_move(king_side) in legal_action_ids(board)
    assert encode_move(takes_rook) not in legal_action_ids(board)
    assert action_is_legal(board, encode_move(king_side))
    assert not action_is_legal(board, encode_move(takes_rook))


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
