import pytest

from anthro_chess.chess import (
    CastlingRights,
    Color,
    IllegalMoveError,
    InvalidMoveError,
    InvalidPositionError,
    Move,
    Piece,
    PieceType,
    Position,
    Promotion,
    replay_moves,
    replay_san,
)


def test_replays_positions_before_each_ply_without_mutating_them() -> None:
    game = replay_san(("e4", "e5", "Nf3", "Nc6", "Bb5"))

    assert len(game.positions_before) == 5
    assert game.positions_before[0].fen == Position.initial().fen
    assert game.positions_before[1].turn is Color.BLACK
    assert game.positions_before[1].fen.split()[3] == "e3"
    assert game.positions_before[1].en_passant_square == 20
    assert game.positions_before[1].pieces[28] == Piece(Color.WHITE, PieceType.PAWN)
    assert game.positions_before[1].halfmove_clock == 0
    assert game.positions_before[1].fullmove_number == 1
    assert game.moves[0].uci == "e2e4"
    assert game.final_position.turn is Color.BLACK


def test_generates_legal_moves_and_rejects_illegal_history_with_ply() -> None:
    initial = Position.initial()

    assert len(initial.legal_moves) == 20
    with pytest.raises(IllegalMoveError, match="ply 1"):
        replay_moves((Move.from_uci("e2e4"), Move.from_uci("e2e3")))


def test_tracks_check_and_rejects_moves_that_do_not_answer_it() -> None:
    position = replay_san(("e4", "f6", "Qh5+")).final_position

    assert position.is_check
    with pytest.raises(IllegalMoveError, match="illegal move"):
        position.apply(Move.from_uci("a7a6"))


def test_castling_is_the_king_move_and_updates_both_pieces() -> None:
    position = replay_san(("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6")).final_position
    castle = position.parse_san("O-O")

    assert castle.uci == "e1g1"
    castled = position.apply(castle)
    assert (
        castled.fen.split()[0]
        == "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1"
    )
    assert castled.castling_rights == CastlingRights(
        white_kingside=False,
        white_queenside=False,
        black_kingside=True,
        black_queenside=True,
    )


def test_en_passant_is_legal_only_while_the_target_exists() -> None:
    position = replay_san(("e4", "a6", "e5", "d5")).final_position

    assert position.parse_san("exd6").uci == "e5d6"
    after_en_passant = position.apply(Move.from_uci("e5d6"))
    assert (
        after_en_passant.fen.split()[0]
        == "rnbqkbnr/1pp1pppp/p2P4/8/8/8/PPPP1PPP/RNBQKBNR"
    )


@pytest.mark.parametrize(
    ("san", "promotion"),
    [("a8=Q+", Promotion.QUEEN), ("a8=N", Promotion.KNIGHT)],
)
def test_parses_and_applies_promotions(san: str, promotion: Promotion) -> None:
    position = Position.from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1")
    move = position.parse_san(san)

    assert move.promotion is promotion
    assert position.apply(move).turn is Color.BLACK


def test_invalid_position_and_move_text_fail_at_the_boundary() -> None:
    with pytest.raises(InvalidPositionError):
        Position.from_fen("not a fen")
    with pytest.raises(InvalidMoveError):
        Move.from_uci("e2e9")
