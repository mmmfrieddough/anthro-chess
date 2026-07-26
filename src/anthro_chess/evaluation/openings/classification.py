"""Game-level opening classification derived in the evaluation view layer.

Labels are derived, never stored in normalized artifacts, so a book or
granularity change never regenerates the corpus. The API is source-agnostic:
frozen held-out games and generated rollout games both classify through it.

Grouping by the literal first moves would be derivable with no book at all, but
it splits transpositions into different buckets and fragments broad openings
across many prefixes while narrow ones keep their mass in one, so the statement
a rollout distribution is trying to make cannot be made from it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import chess

from anthro_chess.chess import MOVE_ACTION_COUNT, decode_move
from anthro_chess.evaluation.openings.book import OpeningBook, load_book

OPENING_CLASSIFICATION_VERSION = 1

#: Label used at every level for a game the book does not name. Games that
#: match nothing are reported explicitly rather than forced into a nearest
#: family, so an unnamed line stays visible as an unnamed line.
UNCLASSIFIED = "unclassified"

#: Characters that separate one naming level from the next in book names.
_LEVEL_SEPARATORS = ":,"


class OpeningClassificationError(ValueError):
    """Raised when a game cannot be replayed for classification."""


class OpeningLevel(StrEnum):
    """Naming granularity a benchmark can aggregate at.

    One classification pass emits all three, so a benchmark picks the level it
    needs instead of the book choosing for it: broad distribution comparisons
    want families, a regression in one line wants the deepest name.
    """

    FAMILY = "family"
    VARIATION = "variation"
    LINE = "line"


@dataclass(frozen=True)
class OpeningLabel:
    """One game's opening label at every granularity level."""

    family: str
    variation: str
    line: str
    eco: str | None
    matched_ply: int

    @property
    def classified(self) -> bool:
        """Return whether the book named any position the game reached."""

        return self.matched_ply > 0

    def label(self, level: OpeningLevel) -> str:
        """Return this game's label at one granularity level."""

        if level is OpeningLevel.FAMILY:
            return self.family
        if level is OpeningLevel.VARIATION:
            return self.variation
        return self.line

    def as_record(self) -> dict[str, object]:
        """Return the stable per-game record stored in benchmark artifacts."""

        return {
            "version": OPENING_CLASSIFICATION_VERSION,
            "classified": self.classified,
            "eco": self.eco,
            "matched_ply": self.matched_ply,
            OpeningLevel.FAMILY.value: self.family,
            OpeningLevel.VARIATION.value: self.variation,
            OpeningLevel.LINE.value: self.line,
        }


UNCLASSIFIED_LABEL = OpeningLabel(
    family=UNCLASSIFIED,
    variation=UNCLASSIFIED,
    line=UNCLASSIFIED,
    eco=None,
    matched_ply=0,
)


def classify_moves(
    moves: Iterable[chess.Move],
    *,
    initial_position: str | None = None,
    book: OpeningBook | None = None,
) -> OpeningLabel:
    """Classify one game by the deepest book position it reaches.

    Matching is on positions, so a game that transposes into a line by an
    unusual move order lands in the same family as one that plays the main
    order. Taking the deepest match is the forward-scan equivalent of walking
    backward from the end of book depth until a named position appears.
    """

    resolved = load_book() if book is None else book
    board = _starting_board(initial_position)
    deepest: tuple[int, str, str] | None = None
    for ply, move in enumerate(moves, start=1):
        if ply > resolved.maximum_ply:
            break
        if not board.is_legal(move):
            raise OpeningClassificationError(
                f"move {ply} ({move.uci()}) is illegal in the position it is "
                "played from; the moves and the initial position disagree"
            )
        board.push(move)
        entry = resolved.entry_for(board.epd())
        if entry is not None:
            deepest = (ply, entry.name, entry.eco)

    if deepest is None:
        return UNCLASSIFIED_LABEL
    ply, name, eco = deepest
    family, variation, line = opening_levels(name)
    return OpeningLabel(
        family=family,
        variation=variation,
        line=line,
        eco=eco or None,
        matched_ply=ply,
    )


def classify_action_ids(
    action_ids: Iterable[int],
    *,
    initial_position: str | None = None,
    book: OpeningBook | None = None,
) -> OpeningLabel:
    """Classify a game recorded as action ids rather than as moves.

    Normalized games and generated rollouts both carry action ids. A
    non-move action such as resignation ends the game, so the scan stops there
    instead of failing.
    """

    return classify_moves(
        _decoded_moves(action_ids),
        initial_position=initial_position,
        book=book,
    )


def opening_distribution(
    labels: Iterable[OpeningLabel],
    level: OpeningLevel = OpeningLevel.FAMILY,
) -> dict[str, int]:
    """Count games per label at one level, including the unclassified ones."""

    counts: dict[str, int] = {}
    for label in labels:
        name = label.label(level)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def opening_levels(name: str) -> tuple[str, str, str]:
    """Return the family, variation, and line forms of one book name.

    Book names nest from broad to specific and separate the levels with a
    colon or a comma, so the levels are the name truncated at the first and
    second separator rather than a second table to keep in step with the book.
    """

    cuts = [index for index, char in enumerate(name) if char in _LEVEL_SEPARATORS]
    family = name[: cuts[0]].strip() if cuts else name
    variation = name[: cuts[1]].strip() if len(cuts) > 1 else name
    return family, variation, name


def _decoded_moves(action_ids: Iterable[int]) -> Iterable[chess.Move]:
    for action_id in action_ids:
        if action_id >= MOVE_ACTION_COUNT:
            return
        yield decode_move(action_id)


def _starting_board(initial_position: str | None) -> chess.Board:
    if initial_position is None:
        return chess.Board()
    try:
        return chess.Board(initial_position)
    except ValueError as error:
        raise OpeningClassificationError(
            f"cannot replay a game from {initial_position!r}: {error}"
        ) from error
