"""Tests for derived evaluation slices."""

from collections.abc import Callable

import chess
import pytest

from anthro_chess.data import GameEncodingInput, encode_game
from anthro_chess.evaluation import (
    GamePhase,
    PlayerColor,
    board_phase,
    board_piece_ids,
    game_phase,
    legal_move_count_bucket,
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
