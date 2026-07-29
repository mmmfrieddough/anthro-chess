"""Termination derivation over replayed final positions."""

from __future__ import annotations

from collections.abc import Sequence

import chess
import pytest

from anthro_chess.data import TerminationCategory, derive_termination
from anthro_chess.data.termination import DerivedTermination

#: A bare-kings-and-knights position both sides can shuffle in without
#: capturing, so repetition counts rise without any other rule intervening.
SHUFFLE_POSITION = "4k1n1/8/8/8/8/8/8/4K1N1 w - - 0 1"
SHUFFLE_CYCLE = ("g1f3", "g8f6", "f3g1", "f6g8")


def _board(fen: str = chess.STARTING_FEN, moves: Sequence[str] = ()) -> chess.Board:
    """Return the position ``moves`` reach from ``fen``, keeping the move stack."""

    board = chess.Board(fen)
    for move in moves:
        board.push(chess.Move.from_uci(move))
    return board


def _shuffled(cycles: int) -> chess.Board:
    return _board(SHUFFLE_POSITION, SHUFFLE_CYCLE * cycles)


def _derive(
    board: chess.Board,
    *,
    result: str,
    source_termination: str | None = "normal",
    clock_remaining_ms: Sequence[int | None] = (),
    time_initial_ms: int | None = 300_000,
    abandonment_clock_share: float = 0.3,
) -> DerivedTermination:
    return derive_termination(
        result=result,
        source_termination=source_termination,
        final_board=board,
        clock_remaining_ms=clock_remaining_ms,
        time_initial_ms=time_initial_ms,
        abandonment_clock_share=abandonment_clock_share,
    )


def test_checkmate_is_read_from_the_position_not_the_source_field() -> None:
    board = _board(moves=("f2f3", "e7e5", "g2g4", "d8h4"))

    derived = _derive(board, result="0-1")

    assert derived.category is TerminationCategory.CHECKMATE
    assert derived.by_side_to_move is None


def test_a_decided_normal_game_without_mate_is_a_resignation() -> None:
    # Three plies leave Black to move, so Black is both the loser and the
    # side to move: the resignation has a decision point to attach to.
    board = _board(moves=("e2e4", "e7e5", "g1f3"))

    derived = _derive(board, result="1-0")

    assert derived.category is TerminationCategory.RESIGNATION
    assert derived.by_side_to_move is True


def test_a_resignation_made_on_the_opponent_clock_is_not_attributable() -> None:
    # Two plies leave White to move while Black is the losing player.
    board = _board(moves=("e2e4", "e7e5"))

    derived = _derive(board, result="1-0")

    assert derived.category is TerminationCategory.RESIGNATION
    assert derived.by_side_to_move is False


def test_stalemate_is_derived_from_exact_logic() -> None:
    board = chess.Board()
    for move in "e3 a5 Qh5 Ra6 Qxa5 h5 Qxc7 Rah6 h4 f6 Qxd7+ Kf7 Qxb7 Qd3".split():
        board.push_san(move)
    for move in "Qxb8 Qh7 Qxc8 Kg6 Qe6".split():
        board.push_san(move)

    derived = _derive(board, result="1/2-1/2")

    assert derived.category is TerminationCategory.STALEMATE
    assert derived.by_side_to_move is None


def test_insufficient_material_is_derived_from_exact_logic() -> None:
    board = _board("4k3/8/8/8/8/8/8/4K1N1 b - - 0 1")

    derived = _derive(board, result="1/2-1/2")

    assert derived.category is TerminationCategory.INSUFFICIENT_MATERIAL


def test_a_claimable_threefold_draw_is_separated_from_an_automatic_fivefold() -> None:
    claimable = _derive(_shuffled(2), result="1/2-1/2")
    automatic = _derive(_shuffled(4), result="1/2-1/2")

    assert claimable.category is TerminationCategory.THREEFOLD_REPETITION
    assert automatic.category is TerminationCategory.FIVEFOLD_REPETITION


def test_a_claimed_draw_is_attributed_to_the_side_to_move() -> None:
    derived = _derive(_shuffled(2), result="1/2-1/2")

    assert derived.by_side_to_move is True


def test_an_automatic_draw_is_attributed_to_no_player() -> None:
    derived = _derive(_shuffled(4), result="1/2-1/2")

    assert derived.by_side_to_move is None


def test_the_move_rule_separates_a_claim_from_an_automatic_draw() -> None:
    claimable = _derive(_board("4k3/8/8/8/8/8/8/R3K3 w - - 100 200"), result="1/2-1/2")
    automatic = _derive(_board("4k3/8/8/8/8/8/8/R3K3 w - - 150 200"), result="1/2-1/2")

    assert claimable.category is TerminationCategory.FIFTY_MOVES
    assert claimable.by_side_to_move is True
    assert automatic.category is TerminationCategory.SEVENTYFIVE_MOVES
    assert automatic.by_side_to_move is None


def test_a_draw_with_no_rule_ending_or_available_claim_is_an_agreement() -> None:
    derived = _derive(_board(moves=("e2e4", "e7e5")), result="1/2-1/2")

    assert derived.category is TerminationCategory.DRAW_AGREEMENT
    assert derived.by_side_to_move is None


@pytest.mark.parametrize(
    ("remaining_ms", "expected"),
    [
        (2_000, TerminationCategory.CLOCK_EXPIRY),
        (90_000, TerminationCategory.CLOCK_EXPIRY),
        (120_000, TerminationCategory.ABANDONMENT),
        (280_000, TerminationCategory.ABANDONMENT),
    ],
)
def test_the_losing_clock_share_separates_abandonment_from_a_flag_fall(
    remaining_ms: int,
    expected: TerminationCategory,
) -> None:
    # Black is the loser and moved second, so their last clock reading is the
    # final entry in the trace.
    derived = _derive(
        _board(moves=("e2e4", "e7e5")),
        result="1-0",
        source_termination="time_forfeit",
        clock_remaining_ms=(295_000, remaining_ms),
    )

    assert derived.category is expected
    assert derived.losing_clock_share == pytest.approx(remaining_ms / 300_000)


def test_the_abandonment_threshold_is_the_configured_one() -> None:
    permissive = _derive(
        _board(moves=("e2e4", "e7e5")),
        result="1-0",
        source_termination="time_forfeit",
        clock_remaining_ms=(295_000, 30_000),
        abandonment_clock_share=0.05,
    )

    assert permissive.category is TerminationCategory.ABANDONMENT


def test_the_losing_player_clock_is_read_at_their_own_last_move() -> None:
    # White is the loser here, so the odd-indexed Black readings must be
    # ignored even though the last of them sits at the end of the trace.
    derived = _derive(
        _board(moves=("e2e4", "e7e5", "g1f3")),
        result="0-1",
        source_termination="time_forfeit",
        clock_remaining_ms=(299_000, 1_000, 250_000),
    )

    assert derived.category is TerminationCategory.ABANDONMENT
    assert derived.losing_clock_share == pytest.approx(250_000 / 300_000)


@pytest.mark.parametrize(
    ("clock_remaining_ms", "time_initial_ms"),
    [
        ((295_000, None), 300_000),
        ((295_000, 120_000), None),
        ((), 300_000),
    ],
)
def test_a_time_forfeit_without_clock_evidence_stays_a_clock_expiry(
    clock_remaining_ms: Sequence[int | None],
    time_initial_ms: int | None,
) -> None:
    derived = _derive(
        _board(moves=("e2e4", "e7e5")),
        result="1-0",
        source_termination="time_forfeit",
        clock_remaining_ms=clock_remaining_ms,
        time_initial_ms=time_initial_ms,
    )

    assert derived.category is TerminationCategory.CLOCK_EXPIRY
    assert derived.losing_clock_share is None


def test_a_drawn_time_forfeit_is_a_clock_expiry() -> None:
    derived = _derive(
        _board(moves=("e2e4", "e7e5")),
        result="1/2-1/2",
        source_termination="time_forfeit",
        clock_remaining_ms=(295_000, 1_000),
    )

    assert derived.category is TerminationCategory.CLOCK_EXPIRY


def test_a_source_reported_abandonment_keeps_its_own_category() -> None:
    derived = _derive(
        _board(moves=("e2e4", "e7e5")),
        result="1-0",
        source_termination="abandoned",
    )

    assert derived.category is TerminationCategory.ABANDONMENT


def test_a_missing_termination_field_leaves_an_unfinished_game_unknown() -> None:
    derived = _derive(
        _board(moves=("e2e4", "e7e5")), result="1-0", source_termination=None
    )

    assert derived.category is TerminationCategory.UNKNOWN
    assert derived.by_side_to_move is None


def test_a_missing_termination_field_still_yields_to_the_final_position() -> None:
    board = _board(moves=("f2f3", "e7e5", "g2g4", "d8h4"))

    derived = _derive(board, result="0-1", source_termination=None)

    assert derived.category is TerminationCategory.CHECKMATE


@pytest.mark.parametrize(
    "source_termination",
    ["rules_infraction", "adjudication", "unterminated", "death"],
)
def test_a_termination_the_derivation_cannot_read_is_unknown(
    source_termination: str,
) -> None:
    derived = _derive(
        _board(moves=("e2e4", "e7e5")),
        result="1-0",
        source_termination=source_termination,
    )

    assert derived.category is TerminationCategory.UNKNOWN


def test_an_undecided_result_is_unknown() -> None:
    derived = _derive(_board(moves=("e2e4", "e7e5")), result="*")

    assert derived.category is TerminationCategory.UNKNOWN


def test_a_result_contradicting_the_final_position_is_not_resolved() -> None:
    # The position is checkmate for Black, so a drawn result means the source
    # disagrees with itself and neither side of it is trusted.
    board = _board(moves=("f2f3", "e7e5", "g2g4", "d8h4"))

    derived = _derive(board, result="1/2-1/2")

    assert derived.category is TerminationCategory.UNKNOWN


def test_derivation_is_deterministic_for_one_game() -> None:
    boards = [_shuffled(2) for _ in range(2)]

    first, second = (_derive(board, result="1/2-1/2") for board in boards)

    assert first == second
