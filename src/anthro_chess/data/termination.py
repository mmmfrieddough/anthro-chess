"""Derive how a game ended from what a PGN reports plus exact chess logic.

Sources report endings inconsistently and incompletely, so the source
``Termination`` field alone cannot separate a resignation from a checkmate or
an abandonment from a flag fall. Replaying to the final position recovers most
of that: the position itself proves a rule ending, and a decisive game that
reached no rule ending under a normal termination can only have been resigned.

The derivation is defined over the PGN vocabulary rather than any one
platform's status values, so it stays source-agnostic per
``0004-source-agnostic-normalized-data.md``. See
``0017-derived-termination-and-terminal-actions.md`` for the measurements
behind it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import chess

_DRAW_RESULT = "1/2-1/2"
_LOSER_COLORS: dict[str, chess.Color] = {"1-0": chess.BLACK, "0-1": chess.WHITE}

#: Normalized PGN ``Termination`` values this derivation understands. Anything
#: else, including adjudication and unterminated games, cannot be classified
#: from what the source reports and becomes an explicit unknown.
_NORMAL_TERMINATION = "normal"
_TIME_FORFEIT_TERMINATION = "time_forfeit"
_ABANDONED_TERMINATION = "abandoned"


class TerminationCategory(StrEnum):
    """How a game ended, derived rather than taken from source text.

    The rule endings share their names with the generated-game vocabulary in
    ``anthro_chess.evaluation.games`` so a human reference mix and a model's
    own mix are counted over the same terms. The remaining categories describe
    endings only a human platform produces: a resignation, a clock running
    out, a player walking away, and a draw the players simply agreed.
    """

    CHECKMATE = "checkmate"
    RESIGNATION = "resignation"
    CLOCK_EXPIRY = "clock_expiry"
    ABANDONMENT = "abandonment"
    STALEMATE = "stalemate"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    SEVENTYFIVE_MOVES = "seventyfive_moves"
    FIVEFOLD_REPETITION = "fivefold_repetition"
    FIFTY_MOVES = "fifty_moves"
    THREEFOLD_REPETITION = "threefold_repetition"
    DRAW_AGREEMENT = "draw_agreement"
    UNKNOWN = "unknown"


TERMINATION_CATEGORIES = tuple(category.value for category in TerminationCategory)

_RULE_CATEGORIES = {
    chess.Termination.CHECKMATE: TerminationCategory.CHECKMATE,
    chess.Termination.STALEMATE: TerminationCategory.STALEMATE,
    chess.Termination.INSUFFICIENT_MATERIAL: TerminationCategory.INSUFFICIENT_MATERIAL,
    chess.Termination.SEVENTYFIVE_MOVES: TerminationCategory.SEVENTYFIVE_MOVES,
    chess.Termination.FIVEFOLD_REPETITION: TerminationCategory.FIVEFOLD_REPETITION,
    chess.Termination.FIFTY_MOVES: TerminationCategory.FIFTY_MOVES,
    chess.Termination.THREEFOLD_REPETITION: TerminationCategory.THREEFOLD_REPETITION,
}


@dataclass(frozen=True)
class DerivedTermination:
    """One game's derived ending.

    ``by_side_to_move`` answers whether the player whose decision ended the
    game held the move at the final position, and is ``None`` for endings no
    player decided. Emitting a terminal action depends on that distinction: a
    resignation made on the opponent's clock has no decision point to attach
    to.

    ``losing_clock_share`` is the losing player's remaining time as a share of
    their initial time at their last move, populated only when the abandonment
    test could actually run. It lets preparation report how much of the
    time-forfeit population the threshold was able to judge.
    """

    category: TerminationCategory
    by_side_to_move: bool | None = None
    losing_clock_share: float | None = None


_UNKNOWN = DerivedTermination(TerminationCategory.UNKNOWN)


def derive_termination(
    *,
    result: str,
    source_termination: str | None,
    final_board: chess.Board,
    clock_remaining_ms: Sequence[int | None],
    time_initial_ms: int | None,
    abandonment_clock_share: float,
) -> DerivedTermination:
    """Classify how one replayed game ended.

    ``final_board`` must be the position after the last mainline move, carrying
    the full move stack so repetition and move-rule claims can be evaluated.
    ``clock_remaining_ms`` is the per-ply clock trace in mainline order.

    Exact chess logic wins wherever the final position is already terminal,
    because the position is proof and the source text is not. A position that
    is terminal in a way the reported result contradicts means the source is
    internally inconsistent, which is reported as unknown rather than resolved
    in either direction.
    """

    outcome = final_board.outcome()
    if outcome is not None:
        if outcome.result() != result:
            return _UNKNOWN
        return DerivedTermination(
            _RULE_CATEGORIES.get(outcome.termination, TerminationCategory.UNKNOWN)
        )

    if source_termination == _ABANDONED_TERMINATION:
        return DerivedTermination(TerminationCategory.ABANDONMENT)
    if source_termination == _TIME_FORFEIT_TERMINATION:
        return _derive_time_forfeit(
            result=result,
            clock_remaining_ms=clock_remaining_ms,
            time_initial_ms=time_initial_ms,
            abandonment_clock_share=abandonment_clock_share,
        )
    if source_termination == _NORMAL_TERMINATION:
        return _derive_normal(result=result, final_board=final_board)
    return _UNKNOWN


def _derive_normal(*, result: str, final_board: chess.Board) -> DerivedTermination:
    """Classify a game the source reported as ending through normal play."""

    loser = _LOSER_COLORS.get(result)
    if loser is not None:
        # The final position is not checkmate, so a decided game under a normal
        # termination can only have been resigned.
        return DerivedTermination(
            TerminationCategory.RESIGNATION,
            by_side_to_move=final_board.turn == loser,
        )
    if result != _DRAW_RESULT:
        return _UNKNOWN

    claimed = final_board.outcome(claim_draw=True)
    if claimed is None:
        return DerivedTermination(TerminationCategory.DRAW_AGREEMENT)
    category = _RULE_CATEGORIES.get(claimed.termination)
    if category is None:  # pragma: no cover - the claim vocabulary is closed
        return _UNKNOWN
    # A draw claim belongs to the player having the move, so a claimed draw is
    # always attributable to the side to move.
    return DerivedTermination(category, by_side_to_move=True)


def _derive_time_forfeit(
    *,
    result: str,
    clock_remaining_ms: Sequence[int | None],
    time_initial_ms: int | None,
    abandonment_clock_share: float,
) -> DerivedTermination:
    """Separate a player walking away from a genuine flag fall.

    Only the clock trace distinguishes the two, and a source that exports no
    clocks leaves the ending as the clock expiry the source itself reported
    rather than promoting it to an unknown.
    """

    share = _losing_clock_share(
        result=result,
        clock_remaining_ms=clock_remaining_ms,
        time_initial_ms=time_initial_ms,
    )
    if share is None:
        return DerivedTermination(TerminationCategory.CLOCK_EXPIRY)
    category = (
        TerminationCategory.ABANDONMENT
        if share > abandonment_clock_share
        else TerminationCategory.CLOCK_EXPIRY
    )
    return DerivedTermination(category, losing_clock_share=share)


def _losing_clock_share(
    *,
    result: str,
    clock_remaining_ms: Sequence[int | None],
    time_initial_ms: int | None,
) -> float | None:
    """Return the losing player's remaining time share at their last move."""

    loser = _LOSER_COLORS.get(result)
    if loser is None or time_initial_ms is None or time_initial_ms <= 0:
        return None
    index = _last_ply_index(loser, len(clock_remaining_ms))
    if index is None:
        return None
    remaining = clock_remaining_ms[index]
    if remaining is None:
        return None
    return remaining / time_initial_ms


def _last_ply_index(color: chess.Color, ply_count: int) -> int | None:
    """Return the index of ``color``'s last ply in a standard-start game.

    Preparation rejects non-standard initial positions, so White always owns
    the even indices.
    """

    parity = 0 if color == chess.WHITE else 1
    last = ply_count - 1
    if last % 2 != parity:
        last -= 1
    return last if last >= 0 else None
