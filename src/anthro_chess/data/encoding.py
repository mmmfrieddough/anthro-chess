"""Versioned, source-agnostic per-ply model-facing encodings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import chess

from anthro_chess.chess import (
    action_vocabulary_identity,
    decode_move,
    legal_action_ids,
)

ENCODING_NAME = "anthro-per-ply"
ENCODING_VERSION = 1
BOARD_SQUARE_COUNT = 64

FieldStatus = Literal["present", "unavailable", "rejected"]
_VALID_STATUSES = frozenset[FieldStatus]({"present", "unavailable", "rejected"})

_ENCODING_SCHEMA = {
    "identity": {
        "game_id": "nonnegative normalized game identifier",
        "ply_index": "zero-based integer",
    },
    "board": {
        "piece_ids": (
            "64 a1-to-h8 integers: 0 empty, 1-6 white pawn-to-king, "
            "7-12 black pawn-to-king"
        ),
        "side_to_move": "0 white, 1 black",
        "castling_rights": (
            "bit mask: 1 white kingside, 2 white queenside, "
            "4 black kingside, 8 black queenside"
        ),
        "en_passant_square": "null or python-chess square index",
        "halfmove_clock": "nonnegative integer",
        "fullmove_number": "positive integer",
    },
    "trajectory": {
        "previous_action_id": "null on the first ply, otherwise action id",
        "target_action_id": "action id",
        "legal_action_ids": "sorted action ids before the target move",
    },
    "context": {
        "optional_integer": (
            "object with nonnegative integer value when status is present; "
            "null value when status is unavailable or rejected"
        ),
        "player_rating": "optional normalized rating for side to move",
        "opponent_rating": "optional normalized rating for opposing side",
        "time_initial_ms": "optional static time control",
        "time_increment_ms": "optional static increment",
        "player_clock_ms": "optional pre-move clock for side to move",
        "opponent_clock_ms": "optional pre-move clock for opposing side",
        "target_clock_after_move_ms": (
            "optional observed post-move clock for side to move"
        ),
    },
}
ENCODING_SCHEMA_SHA256 = sha256(
    json.dumps(_ENCODING_SCHEMA, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class EncodingError(ValueError):
    """Raised when a game cannot be converted into aligned per-ply encodings."""


@dataclass(frozen=True)
class OptionalInteger:
    """An optional integer whose missingness is explicit."""

    value: int | None
    status: FieldStatus

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"unknown field status: {self.status}")
        if self.status == "present":
            if type(self.value) is not int or self.value < 0:
                raise ValueError("present optional integers must be nonnegative")
        elif self.value is not None:
            raise ValueError("unavailable or rejected optional integers need no value")

    def as_record(self) -> dict[str, object]:
        """Return a JSON-serializable value and missingness record."""

        return {"value": self.value, "status": self.status}


@dataclass(frozen=True)
class GameEncodingInput:
    """Source-agnostic normalized game fields needed for per-ply encoding."""

    game_id: int
    ruleset: str
    initial_position: str
    action_ids: tuple[int, ...]
    white_normalized_rating: OptionalInteger
    black_normalized_rating: OptionalInteger
    time_initial_ms: OptionalInteger
    time_increment_ms: OptionalInteger
    clock_remaining_ms: tuple[OptionalInteger, ...]

    def __post_init__(self) -> None:
        if type(self.game_id) is not int or self.game_id < 0:
            raise ValueError("game_id must be a nonnegative integer")
        if self.ruleset != "standard":
            raise ValueError("the first per-ply encoding supports standard chess only")
        if not self.action_ids:
            raise ValueError("a game encoding input needs at least one action")
        if len(self.clock_remaining_ms) != len(self.action_ids):
            raise ValueError("clock observations must align one-to-one with actions")


@dataclass(frozen=True)
class BoardEncoding:
    """Compact exact standard-chess state before one target move."""

    piece_ids: tuple[int, ...]
    side_to_move: int
    castling_rights: int
    en_passant_square: int | None
    halfmove_clock: int
    fullmove_number: int

    def as_record(self) -> dict[str, object]:
        """Return a JSON-serializable board record."""

        return {
            "piece_ids": list(self.piece_ids),
            "side_to_move": self.side_to_move,
            "castling_rights": self.castling_rights,
            "en_passant_square": self.en_passant_square,
            "halfmove_clock": self.halfmove_clock,
            "fullmove_number": self.fullmove_number,
        }


@dataclass(frozen=True)
class PlyEncoding:
    """One aligned model timestep and its supervised action target."""

    game_id: int
    ply_index: int
    board: BoardEncoding
    previous_action_id: int | None
    target_action_id: int
    legal_action_ids: tuple[int, ...]
    player_rating: OptionalInteger
    opponent_rating: OptionalInteger
    time_initial_ms: OptionalInteger
    time_increment_ms: OptionalInteger
    player_clock_ms: OptionalInteger
    opponent_clock_ms: OptionalInteger
    target_clock_after_move_ms: OptionalInteger

    def as_record(self) -> dict[str, object]:
        """Return the stable JSON-serializable representation."""

        return {
            "game_id": self.game_id,
            "ply_index": self.ply_index,
            "board": self.board.as_record(),
            "previous_action_id": self.previous_action_id,
            "target_action_id": self.target_action_id,
            "legal_action_ids": list(self.legal_action_ids),
            "player_rating": self.player_rating.as_record(),
            "opponent_rating": self.opponent_rating.as_record(),
            "time_initial_ms": self.time_initial_ms.as_record(),
            "time_increment_ms": self.time_increment_ms.as_record(),
            "player_clock_ms": self.player_clock_ms.as_record(),
            "opponent_clock_ms": self.opponent_clock_ms.as_record(),
            "target_clock_after_move_ms": (self.target_clock_after_move_ms.as_record()),
        }


def encoding_identity() -> dict[str, object]:
    """Return the compatibility identity for manifests and model artifacts."""

    return {
        "name": ENCODING_NAME,
        "version": ENCODING_VERSION,
        "schema_sha256": ENCODING_SCHEMA_SHA256,
        "board_square_count": BOARD_SQUARE_COUNT,
        "action_vocabulary": action_vocabulary_identity(),
    }


def encode_game(game: GameEncodingInput) -> tuple[PlyEncoding, ...]:
    """Convert one normalized game into exact, aligned per-ply examples."""

    try:
        board = chess.Board(game.initial_position)
    except ValueError as error:
        raise EncodingError(f"invalid initial position: {error}") from error

    clocks_by_color = {
        chess.WHITE: game.time_initial_ms,
        chess.BLACK: game.time_initial_ms,
    }
    encodings: list[PlyEncoding] = []
    previous_action_id: int | None = None

    for ply_index, target_action_id in enumerate(game.action_ids):
        try:
            target_move = decode_move(target_action_id)
        except ValueError as error:
            raise EncodingError(
                f"action at ply {ply_index} is not a board move: {target_action_id}"
            ) from error

        legal_ids = legal_action_ids(board)
        if target_action_id not in legal_ids:
            raise EncodingError(
                f"action at ply {ply_index} is illegal in the reconstructed position"
            )

        player_color = board.turn
        player_rating, opponent_rating = _ratings_for_color(game, player_color)
        target_clock = game.clock_remaining_ms[ply_index]
        encodings.append(
            PlyEncoding(
                game_id=game.game_id,
                ply_index=ply_index,
                board=_encode_board(board),
                previous_action_id=previous_action_id,
                target_action_id=target_action_id,
                legal_action_ids=legal_ids,
                player_rating=player_rating,
                opponent_rating=opponent_rating,
                time_initial_ms=game.time_initial_ms,
                time_increment_ms=game.time_increment_ms,
                player_clock_ms=clocks_by_color[player_color],
                opponent_clock_ms=clocks_by_color[not player_color],
                target_clock_after_move_ms=target_clock,
            )
        )
        clocks_by_color[player_color] = target_clock
        board.push(target_move)
        previous_action_id = target_action_id

    return tuple(encodings)


def _encode_board(board: chess.Board) -> BoardEncoding:
    piece_ids = tuple(_piece_id(board.piece_at(square)) for square in chess.SQUARES)
    castling_rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_rights |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_rights |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_rights |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_rights |= 8
    return BoardEncoding(
        piece_ids=piece_ids,
        side_to_move=0 if board.turn == chess.WHITE else 1,
        castling_rights=castling_rights,
        en_passant_square=board.ep_square,
        halfmove_clock=board.halfmove_clock,
        fullmove_number=board.fullmove_number,
    )


def _piece_id(piece: chess.Piece | None) -> int:
    if piece is None:
        return 0
    return piece.piece_type + (0 if piece.color == chess.WHITE else 6)


def _ratings_for_color(
    game: GameEncodingInput, color: chess.Color
) -> tuple[OptionalInteger, OptionalInteger]:
    if color == chess.WHITE:
        return game.white_normalized_rating, game.black_normalized_rating
    return game.black_normalized_rating, game.white_normalized_rating
