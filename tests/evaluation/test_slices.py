"""Tests for derived evaluation slices."""

from collections.abc import Callable

import chess
import pytest

from anthro_chess.chess import decode_move
from anthro_chess.data import GameEncodingInput, encode_game
from anthro_chess.evaluation import (
    GamePhase,
    PlayerColor,
    PositionCharacteristic,
    board_characteristics,
    board_from_encoding,
    board_phase,
    board_piece_ids,
    game_phase,
    legal_move_count_bucket,
    ply_characteristics,
    position_slices,
    rating_band_name,
)


def _phase(fen: str) -> GamePhase:
    return board_phase(chess.Board(fen))


def test_opening_becomes_endgame_when_material_leaves_regardless_of_move_number() -> (
    None
):
    """Material decides before move number so an early queen trade is not an opening."""

    assert _phase(chess.STARTING_FEN) is GamePhase.OPENING
    assert _phase("4k3/8/8/8/8/8/4P3/4K3 w - - 0 4") is GamePhase.ENDGAME


def test_untouched_material_past_the_opening_window_is_a_middlegame() -> None:
    crowded = "3q1rk1/pb1nbppp/1p2pn2/2p5/2PP4/1PN1PN2/PB2BPPP/2RQ1RK1 w - - 0 12"
    later = "3q1rk1/pb1nbppp/1p2pn2/2p5/2PP4/1PN1PN2/PB2BPPP/2RQ1RK1 w - - 0 20"

    assert _phase(crowded) is GamePhase.OPENING
    assert _phase(later) is GamePhase.MIDDLEGAME


def test_endgame_boundary_falls_between_thirteen_and_fourteen_points() -> None:
    two_rooks_and_a_knight = "4k3/8/8/8/8/8/8/RN2K2R w - - 0 30"
    queen_and_rook = "4k3/8/8/8/8/8/8/R2QK3 w - - 0 30"

    assert _phase(two_rooks_and_a_knight) is GamePhase.ENDGAME
    assert _phase(queen_and_rook) is GamePhase.MIDDLEGAME


def test_legal_move_count_buckets_follow_the_documented_intervals() -> None:
    assert legal_move_count_bucket(1) == "1_to_10"
    assert legal_move_count_bucket(10) == "1_to_10"
    assert legal_move_count_bucket(11) == "11_to_25"
    assert legal_move_count_bucket(25) == "11_to_25"
    assert legal_move_count_bucket(26) == "26_plus"
    assert legal_move_count_bucket(200) == "26_plus"

    with pytest.raises(ValueError, match="positive integer"):
        legal_move_count_bucket(0)


def test_rating_bands_distinguish_absent_ratings_from_low_ones() -> None:
    assert rating_band_name(None) is None
    assert rating_band_name(0) == "under_1200"
    assert rating_band_name(1199) == "under_1200"
    assert rating_band_name(1200) == "1200_to_1599"
    assert rating_band_name(3000) == "2000_plus"

    with pytest.raises(ValueError, match="nonnegative integer"):
        rating_band_name(-1)


def test_piece_ids_round_trip_through_the_encoding_contract() -> None:
    board = chess.Board()

    assert (
        game_phase(board_piece_ids(board), board.fullmove_number) is GamePhase.OPENING
    )


def test_position_slices_label_every_ply_of_an_encoded_game(
    action_ids: Callable[[tuple[str, ...]], tuple[int, ...]],
) -> None:
    game = GameEncodingInput(
        game_id=1,
        ruleset="standard",
        initial_position=chess.STARTING_FEN,
        action_ids=action_ids(("e2e4", "e7e5", "g1f3")),
        white_normalized_rating=1500,
        black_normalized_rating=900,
        time_initial_ms=None,
        time_increment_ms=None,
        clock_remaining_ms=(None, None, None),
    )

    slices = [position_slices(ply) for ply in encode_game(game)]

    assert [item.color for item in slices] == [
        PlayerColor.WHITE,
        PlayerColor.BLACK,
        PlayerColor.WHITE,
    ]
    assert [item.rating_band for item in slices] == [
        "1200_to_1599",
        "under_1200",
        "1200_to_1599",
    ]
    assert all(item.phase is GamePhase.OPENING for item in slices)
    assert slices[0].legal_move_count == 20
    assert slices[0].legal_move_count_bucket == "11_to_25"
    assert slices[0].as_record()["legal_move_count"] == 20


def test_slices_report_absent_ratings_without_inventing_a_band(
    action_ids: Callable[[tuple[str, ...]], tuple[int, ...]],
) -> None:
    game = GameEncodingInput(
        game_id=2,
        ruleset="standard",
        initial_position=chess.STARTING_FEN,
        action_ids=action_ids(("d2d4",)),
        white_normalized_rating=None,
        black_normalized_rating=None,
        time_initial_ms=None,
        time_increment_ms=None,
        clock_remaining_ms=(None,),
    )

    (slices,) = [position_slices(ply) for ply in encode_game(game)]

    assert slices.rating_band is None
    assert slices.as_record()["rating_band"] is None


def test_rule_case_characteristics_are_derived_from_exact_logic() -> None:
    """These isolate the rare cases that vanish from a pool-wide average."""

    cases = {
        "R5k1/5p1p/6p1/8/8/8/8/6K1 b - - 0 1": {
            PositionCharacteristic.CHECK,
            PositionCharacteristic.ONLY_MOVE,
        },
        "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1": {
            PositionCharacteristic.CHECK,
            PositionCharacteristic.CHECKMATE,
            PositionCharacteristic.TERMINAL,
        },
        "7k/8/6Q1/8/8/8/8/7K b - - 0 1": {
            PositionCharacteristic.STALEMATE,
            PositionCharacteristic.TERMINAL,
        },
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3": {
            PositionCharacteristic.EN_PASSANT,
            PositionCharacteristic.CASTLING_RIGHTS,
        },
        "3r4/4P3/8/8/8/8/8/K6k w - - 0 1": {PositionCharacteristic.PROMOTION},
        "r1bqkbnr/ppp2ppp/2np4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4": {
            PositionCharacteristic.PIN,
            PositionCharacteristic.CASTLING_RIGHTS,
        },
        "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 9": {
            PositionCharacteristic.CASTLING_RIGHTS,
            PositionCharacteristic.CASTLING_AVAILABLE,
        },
    }

    for fen, expected in cases.items():
        assert board_characteristics(chess.Board(fen)) == expected, fen


def test_castling_rights_are_distinguished_from_an_available_castle() -> None:
    blocked = "r3k2r/pppq1ppp/2npbn2/2b1p3/4P3/3PBN2/PPPQ1PPP/RN2KB1R w KQkq - 6 9"

    observed = board_characteristics(chess.Board(blocked))

    assert PositionCharacteristic.CASTLING_RIGHTS in observed
    assert PositionCharacteristic.CASTLING_AVAILABLE not in observed


def test_an_advertised_en_passant_square_is_not_an_available_capture() -> None:
    """The FEN offers a target square that no pawn can actually capture on."""

    fen = "rnbqkbnr/ppp2ppp/8/3pp3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 3"

    assert PositionCharacteristic.EN_PASSANT not in board_characteristics(
        chess.Board(fen)
    )


def test_encoded_plies_round_trip_into_exact_boards(
    action_ids: Callable[[tuple[str, ...]], tuple[int, ...]],
) -> None:
    """Rule-case slicing needs an exact board back from a compact encoding."""

    game = GameEncodingInput(
        game_id=3,
        ruleset="standard",
        initial_position=chess.STARTING_FEN,
        action_ids=action_ids(("e2e4", "e7e5", "g1f3", "b8c6", "f1b5")),
        white_normalized_rating=None,
        black_normalized_rating=None,
        time_initial_ms=None,
        time_increment_ms=None,
        clock_remaining_ms=(None,) * 5,
    )
    plies = encode_game(game)

    board = chess.Board()
    for ply in plies:
        rebuilt = board_from_encoding(ply.board)
        assert rebuilt.board_fen() == board.board_fen()
        assert rebuilt.turn == board.turn
        assert rebuilt.castling_rights == board.castling_rights
        assert rebuilt.ep_square == board.ep_square
        assert rebuilt.fullmove_number == board.fullmove_number
        assert sorted(move.uci() for move in rebuilt.legal_moves) == sorted(
            move.uci() for move in board.legal_moves
        )
        board.push(decode_move(ply.target_action_id))


def test_ply_characteristics_match_the_board_they_encode(
    action_ids: Callable[[tuple[str, ...]], tuple[int, ...]],
) -> None:
    game = GameEncodingInput(
        game_id=4,
        ruleset="standard",
        initial_position=chess.STARTING_FEN,
        action_ids=action_ids(("f2f3", "e7e5", "g2g4", "d8h4")),
        white_normalized_rating=None,
        black_normalized_rating=None,
        time_initial_ms=None,
        time_increment_ms=None,
        clock_remaining_ms=(None,) * 4,
    )

    observed = [ply_characteristics(ply) for ply in encode_game(game)]

    assert all(PositionCharacteristic.CHECK not in item for item in observed)
    assert all(PositionCharacteristic.CASTLING_RIGHTS in item for item in observed)


def test_rebuilding_rejects_a_piece_id_outside_the_contract() -> None:
    from anthro_chess.data import BoardEncoding

    board = BoardEncoding(
        piece_ids=(99,) + (0,) * 63,
        side_to_move=0,
        castling_rights=0,
        en_passant_square=None,
        halfmove_clock=0,
        fullmove_number=1,
    )

    with pytest.raises(ValueError, match="outside the encoding contract"):
        board_from_encoding(board)
