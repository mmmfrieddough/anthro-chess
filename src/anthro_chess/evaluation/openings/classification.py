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

import chess

from anthro_chess.chess import MOVE_ACTION_COUNT, decode_move
from anthro_chess.evaluation.openings.book import (
    OpeningBook,
    OpeningContinuation,
    OpeningEntry,
    load_book,
)
from anthro_chess.evaluation.openings.names import (
    UNCLASSIFIED,
    OpeningLevel,
    opening_levels,
)

#: Bumped when the per-game record changes shape. Version 2 adds the book-depth
#: and waypoint fields the repertoire and depth readings are computed from.
OPENING_CLASSIFICATION_VERSION = 2


class OpeningClassificationError(ValueError):
    """Raised when a game cannot be replayed for classification."""


@dataclass(frozen=True)
class OpeningLabel:
    """One game's opening label at every granularity level, and its depth.

    Two readings come out of one classification pass and they answer different
    questions. The labels say *which* opening was chosen, which is preference.
    The depth fields say how far into catalogued theory the game stayed in the
    line it chose, which is knowledge.
    """

    family: str
    variation: str
    line: str
    eco: str | None
    #: Ply of the game at which the deepest named position was reached. Game
    #: coordinates, so a continuation from a human prefix counts the prefix.
    matched_ply: int
    #: The matched entry's own depth in **book** coordinates. Deliberately not
    #: the game ply: the same position is the same distance into theory however
    #: many moves an unusual order took to reach it, and a prefix continuation
    #: has no shared origin with the book at all.
    book_ply: int = 0
    #: What the book still offered from the matched position, absent when the
    #: game matched nothing.
    continuation: OpeningContinuation | None = None

    @property
    def classified(self) -> bool:
        """Return whether the book named any position the game reached."""

        return self.matched_ply > 0

    @property
    def available_ply(self) -> int:
        """Return the deepest theory reachable onward from the match."""

        return 0 if self.continuation is None else self.continuation.deepest_ply

    @property
    def consumed_fraction(self) -> float:
        """Return the share of the theory still available that was played.

        Raw depth conflates choosing a well-analyzed line with knowing it: a
        model playing a line the book follows for thirty plies and stopping at
        six looks the same as one playing an offbeat line to its end. This
        separates them, and reads zero for a game the book never named.
        """

        available = self.available_ply
        return self.book_ply / available if available else 0.0

    def label(self, level: OpeningLevel) -> str:
        """Return this game's label at one granularity level."""

        if level is OpeningLevel.FAMILY:
            return self.family
        if level is OpeningLevel.VARIATION:
            return self.variation
        return self.line

    def waypoint(self, level: OpeningLevel = OpeningLevel.FAMILY) -> bool:
        """Return whether the game stopped on a waypoint rather than a choice.

        A waypoint is a position several labels still pass through, so a game
        that stops on one left the book before committing to anything more
        specific. An unnamed game is not a waypoint: it is off book entirely,
        which is a statement about what it played rather than about how far it
        got.
        """

        return self.continuation is not None and self.continuation.waypoint(level)

    def destination(self, level: OpeningLevel = OpeningLevel.FAMILY) -> str | None:
        """Return the chosen label, or ``None`` when the game only got partway.

        This is what a repertoire distribution counts. Keeping a waypoint label
        in it would let a depth effect leak into a distribution that is
        supposed to measure choice, so the waypoint rate is reported as its own
        scalar instead.
        """

        return None if self.waypoint(level) else self.label(level)

    def as_record(self) -> dict[str, object]:
        """Return the stable per-game record stored in benchmark artifacts."""

        return {
            "version": OPENING_CLASSIFICATION_VERSION,
            "classified": self.classified,
            "eco": self.eco,
            "matched_ply": self.matched_ply,
            "book_ply": self.book_ply,
            "available_ply": self.available_ply,
            "consumed_fraction": self.consumed_fraction,
            "waypoint": {level.value: self.waypoint(level) for level in OpeningLevel},
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
    plies: int | None = None,
) -> OpeningLabel:
    """Classify one game by the deepest book position it reaches.

    Matching is on positions, so a game that transposes into a line by an
    unusual move order lands in the same family as one that plays the main
    order. Taking the deepest match is the forward-scan equivalent of walking
    backward from the end of book depth until a named position appears.

    ``plies`` truncates the scan, which is what the divergence-versus-depth
    diagnostic recomputes the same distance at. Truncation only ever removes
    matches, so a truncated label is the label the game had committed to by
    that point rather than a different classification of the whole game.
    """

    resolved = load_book() if book is None else book
    limit = resolved.maximum_ply if plies is None else min(plies, resolved.maximum_ply)
    board = _starting_board(initial_position)
    deepest: tuple[int, str] | None = None
    for ply, move in enumerate(moves, start=1):
        if ply > limit:
            break
        if not board.is_legal(move):
            raise OpeningClassificationError(
                f"move {ply} ({move.uci()}) is illegal in the position it is "
                "played from; the moves and the initial position disagree"
            )
        board.push(move)
        epd = board.epd()
        if resolved.entry_for(epd) is not None:
            deepest = (ply, epd)

    if deepest is None:
        return UNCLASSIFIED_LABEL
    ply, epd = deepest
    return _label(resolved.positions[epd], epd, ply, resolved)


def _label(
    entry: OpeningEntry,
    epd: str,
    matched_ply: int,
    book: OpeningBook,
) -> OpeningLabel:
    """Build one game's label from the deepest entry it reached."""

    family, variation, line = opening_levels(entry.name)
    return OpeningLabel(
        family=family,
        variation=variation,
        line=line,
        eco=entry.eco or None,
        matched_ply=matched_ply,
        book_ply=entry.ply,
        continuation=book.continuation_for(epd),
    )


def classify_progression(
    moves: Iterable[chess.Move],
    *,
    initial_position: str | None = None,
    book: OpeningBook | None = None,
    plies: int | None = None,
) -> tuple[OpeningLabel, ...]:
    """Return the label the game carried after each ply, up to ``plies``.

    One replay produces every truncation, which is what makes a diagnostic that
    recomputes a distance at every book ply affordable: classifying each depth
    separately would replay the same opening once per depth.

    The result always has ``plies`` entries, padded with the last label when the
    game is shorter, so a caller indexing by depth gets the label the game
    still had rather than a shorter list to reason about.
    """

    resolved = load_book() if book is None else book
    limit = resolved.maximum_ply if plies is None else min(plies, resolved.maximum_ply)
    board = _starting_board(initial_position)
    labels: list[OpeningLabel] = []
    current = UNCLASSIFIED_LABEL
    for ply, move in enumerate(moves, start=1):
        if ply > limit:
            break
        if not board.is_legal(move):
            raise OpeningClassificationError(
                f"move {ply} ({move.uci()}) is illegal in the position it is "
                "played from; the moves and the initial position disagree"
            )
        board.push(move)
        epd = board.epd()
        entry = resolved.entry_for(epd)
        if entry is not None:
            current = _label(entry, epd, ply, resolved)
        labels.append(current)
    labels.extend([current] * (limit - len(labels)))
    return tuple(labels)


def classify_action_ids(
    action_ids: Iterable[int],
    *,
    initial_position: str | None = None,
    book: OpeningBook | None = None,
    plies: int | None = None,
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
        plies=plies,
    )


def progression_for_action_ids(
    action_ids: Iterable[int],
    *,
    initial_position: str | None = None,
    book: OpeningBook | None = None,
    plies: int | None = None,
) -> tuple[OpeningLabel, ...]:
    """Return the per-ply label progression of a game recorded as action ids."""

    return classify_progression(
        _decoded_moves(action_ids),
        initial_position=initial_position,
        book=book,
        plies=plies,
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


def repertoire_distribution(
    labels: Iterable[OpeningLabel],
    level: OpeningLevel = OpeningLevel.FAMILY,
) -> dict[str, int]:
    """Count games per **destination** label, leaving waypoints out.

    The repertoire is which openings were chosen. A game that stopped on a
    waypoint chose nothing yet, so counting it here would put a depth effect
    into a distribution that is supposed to measure preference; its share is
    reported as the waypoint rate instead.
    """

    counts: dict[str, int] = {}
    for label in labels:
        name = label.destination(level)
        if name is None:
            continue
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


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
