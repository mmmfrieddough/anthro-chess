import chess
import pytest

from anthro_chess.chess import decode_move, encode_move, legal_action_ids


def test_replays_exact_positions_before_each_ply() -> None:
    board = chess.Board()
    positions_before: list[chess.Board] = []

    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5"):
        positions_before.append(board.copy(stack=False))
        board.push_san(san)

    assert positions_before[0].fen() == chess.STARTING_FEN
    assert positions_before[1].turn == chess.BLACK
    assert positions_before[1].ep_square == chess.E3
    assert positions_before[1].piece_at(chess.E4) == chess.Piece(
        chess.PAWN, chess.WHITE
    )
    assert board.turn == chess.BLACK


def test_check_and_illegal_moves_use_python_chess_directly() -> None:
    board = chess.Board()
    for san in ("e4", "f6", "Qh5+"):
        board.push_san(san)

    assert board.is_check()
    with pytest.raises(chess.IllegalMoveError):
        board.push_uci("a7a6")
    with pytest.raises(ValueError):
        chess.Board("not a fen")


def test_castling_move_round_trips_through_the_action_codec() -> None:
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"):
        board.push_san(san)

    castle = board.parse_san("O-O")
    assert castle == chess.Move.from_uci("e1g1")
    assert decode_move(encode_move(castle)) == castle

    board.push(castle)
    assert board.piece_at(chess.G1) == chess.Piece(chess.KING, chess.WHITE)
    assert board.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)
    assert not board.has_castling_rights(chess.WHITE)


def test_en_passant_move_is_legal_and_encodable() -> None:
    board = chess.Board()
    for san in ("e4", "a6", "e5", "d5"):
        board.push_san(san)

    en_passant = board.parse_san("exd6")
    assert en_passant == chess.Move.from_uci("e5d6")
    assert encode_move(en_passant) in legal_action_ids(board)

    board.push(en_passant)
    assert board.piece_at(chess.D5) is None
    assert board.piece_at(chess.D6) == chess.Piece(chess.PAWN, chess.WHITE)


@pytest.mark.parametrize("san", ["a8=Q+", "a8=N"])
def test_promotion_moves_round_trip_through_the_action_codec(san: str) -> None:
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    promotion = board.parse_san(san)

    assert promotion.promotion in {
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
    }
    assert decode_move(encode_move(promotion)) == promotion
