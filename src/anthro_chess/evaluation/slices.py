"""Derived position slices shared by offline and rollout evaluation.

Slices are recomputed from exact state rather than stored in normalized
artifacts. A benchmark that needs a slice this module cannot derive is a
signal that the underlying field belongs in the normalized schema, not that
this layer should grow a private cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import chess

from anthro_chess.data import BOARD_SQUARE_COUNT, BoardEncoding, PlyEncoding

SLICE_SCHEME_VERSION = 1


@dataclass(frozen=True)
class RatingBand:
    """One half-open normalized-rating interval used for validation slices."""

    name: str
    minimum_rating: int
    maximum_rating: int | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("rating band name must not be empty")
        if type(self.minimum_rating) is not int or self.minimum_rating < 0:
            raise ValueError("rating band minimum must be a nonnegative integer")
        if self.maximum_rating is not None and (
            type(self.maximum_rating) is not int
            or self.maximum_rating <= self.minimum_rating
        ):
            raise ValueError("rating band maximum must be greater than its minimum")


DEFAULT_RATING_BANDS = (
    RatingBand("under_1200", 0, 1200),
    RatingBand("1200_to_1599", 1200, 1600),
    RatingBand("1600_to_1999", 1600, 2000),
    RatingBand("2000_plus", 2000, None),
)


class GamePhase(StrEnum):
    """Coarse phase label derived from material and move number."""

    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"


class PlayerColor(StrEnum):
    """Color of the player choosing the action at a position."""

    WHITE = "white"
    BLACK = "black"


#: Half-open legal-move-count intervals. Legality metrics vary strongly with
#: how many moves are available, so they are reported per bucket.
LEGAL_MOVE_COUNT_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1_to_10", 1, 11),
    ("11_to_25", 11, 26),
    ("26_plus", 26, None),
)

#: Total non-pawn material across both sides at or below this value marks the
#: endgame. A queen is nine points, a rook five, a minor piece three, so a
#: queen and rook together (fourteen) still count as a middlegame while a
#: queen and minor, or two rooks and a minor, do not.
_ENDGAME_MATERIAL_VALUE = 13

#: Last full move still treated as the opening when material is untouched.
_OPENING_LAST_FULLMOVE = 12

#: Piece ids follow the encoding contract: 0 empty, 1-6 white pawn-to-king,
#: 7-12 black pawn-to-king. Pawns and kings contribute no phase material.
_PIECE_ID_VALUES = (0, 0, 3, 3, 5, 9, 0, 0, 3, 3, 5, 9, 0)


@dataclass(frozen=True)
class PositionSlices:
    """The derived slice labels for one evaluated position."""

    phase: GamePhase
    color: PlayerColor
    legal_move_count: int
    legal_move_count_bucket: str
    rating_band: str | None

    def as_record(self) -> dict[str, object]:
        """Return a stable JSON-serializable slice record."""

        return {
            "phase": str(self.phase),
            "color": str(self.color),
            "legal_move_count": self.legal_move_count,
            "legal_move_count_bucket": self.legal_move_count_bucket,
            "rating_band": self.rating_band,
        }


def game_phase(piece_ids: Sequence[int], fullmove_number: int) -> GamePhase:
    """Return the phase implied by remaining material and move number.

    Material decides first so a long, quiet opening still becomes an endgame
    once the pieces come off, and an early queen trade is not mislabeled as
    an opening.
    """

    if len(piece_ids) != BOARD_SQUARE_COUNT:
        raise ValueError("piece ids must cover every board square")
    if type(fullmove_number) is not int or fullmove_number < 1:
        raise ValueError("fullmove number must be a positive integer")

    material = 0
    for piece_id in piece_ids:
        if type(piece_id) is not int or not 0 <= piece_id < len(_PIECE_ID_VALUES):
            raise ValueError(f"piece id is outside the encoding contract: {piece_id}")
        material += _PIECE_ID_VALUES[piece_id]

    if material <= _ENDGAME_MATERIAL_VALUE:
        return GamePhase.ENDGAME
    if fullmove_number <= _OPENING_LAST_FULLMOVE:
        return GamePhase.OPENING
    return GamePhase.MIDDLEGAME


def board_phase(board: chess.Board) -> GamePhase:
    """Return the phase of an exact board, for generated-game evaluation."""

    return game_phase(board_piece_ids(board), board.fullmove_number)


def board_piece_ids(board: chess.Board) -> tuple[int, ...]:
    """Return encoding-contract piece ids for an exact board."""

    piece_ids = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            piece_ids.append(0)
        else:
            offset = 0 if piece.color == chess.WHITE else 6
            piece_ids.append(piece.piece_type + offset)
    return tuple(piece_ids)


def legal_move_count_bucket(legal_move_count: int) -> str:
    """Return the bucket name for a position's legal-move count."""

    if type(legal_move_count) is not int or legal_move_count < 1:
        raise ValueError("legal move count must be a positive integer")
    for name, minimum, maximum in LEGAL_MOVE_COUNT_BUCKETS:
        if legal_move_count >= minimum and (
            maximum is None or legal_move_count < maximum
        ):
            return name
    raise ValueError(
        f"legal move count is outside configured buckets: {legal_move_count}"
    )


def rating_band_name(
    rating: int | None,
    rating_bands: Sequence[RatingBand] = DEFAULT_RATING_BANDS,
) -> str | None:
    """Return the band name for a rating, or ``None`` when it is absent."""

    if rating is None:
        return None
    if type(rating) is not int or rating < 0:
        raise ValueError("player rating must be a nonnegative integer")
    for band in rating_bands:
        if rating >= band.minimum_rating and (
            band.maximum_rating is None or rating < band.maximum_rating
        ):
            return band.name
    raise ValueError(f"player rating is outside configured bands: {rating}")


def board_color(board: BoardEncoding) -> PlayerColor:
    """Return the color choosing the action at an encoded position."""

    if board.side_to_move == 0:
        return PlayerColor.WHITE
    if board.side_to_move == 1:
        return PlayerColor.BLACK
    raise ValueError(
        f"side to move is outside the encoding contract: {board.side_to_move}"
    )


def position_slices(
    ply: PlyEncoding,
    rating_bands: Sequence[RatingBand] = DEFAULT_RATING_BANDS,
) -> PositionSlices:
    """Return every derived slice label for one encoded ply."""

    return PositionSlices(
        phase=game_phase(ply.board.piece_ids, ply.board.fullmove_number),
        color=board_color(ply.board),
        legal_move_count=len(ply.legal_action_ids),
        legal_move_count_bucket=legal_move_count_bucket(len(ply.legal_action_ids)),
        rating_band=rating_band_name(ply.target_rating, rating_bands),
    )


def _validate_rating_bands(
    rating_bands: Sequence[RatingBand],
) -> tuple[RatingBand, ...]:
    bands = tuple(rating_bands)
    if not bands:
        raise ValueError("validation requires at least one rating band")
    if len({band.name for band in bands}) != len(bands):
        raise ValueError("rating band names must be unique")
    if bands[0].minimum_rating != 0:
        raise ValueError("rating bands must start at zero")
    for previous, current in zip(bands, bands[1:], strict=False):
        if previous.maximum_rating != current.minimum_rating:
            raise ValueError("rating bands must be contiguous and non-overlapping")
    if bands[-1].maximum_rating is not None:
        raise ValueError("the final rating band must have no upper bound")
    return bands


def _rating_band_index(
    rating: int,
    rating_bands: tuple[RatingBand, ...],
) -> int:
    for index, band in enumerate(rating_bands):
        if rating >= band.minimum_rating and (
            band.maximum_rating is None or rating < band.maximum_rating
        ):
            return index
    raise ValueError(f"player rating is outside configured bands: {rating}")
