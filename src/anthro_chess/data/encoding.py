"""Versioned, source-agnostic per-ply model-facing encodings."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

import chess

from anthro_chess.chess import (
    action_vocabulary_identity,
    decode_move,
    encode_move,
    legal_action_ids,
)

ENCODING_NAME = "anthro-per-ply"
ENCODING_VERSION = 3
BOARD_SQUARE_COUNT = 64

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
        "nullable_integer": "nonnegative integer when present, otherwise null",
        "target_rating": (
            "optional normalized rating for the player choosing the target action; "
            "kept outside target-free historical inputs"
        ),
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
class GameEncodingInput:
    """Source-agnostic normalized game fields needed for per-ply encoding."""

    game_id: int
    ruleset: str
    initial_position: str
    action_ids: tuple[int, ...]
    white_normalized_rating: int | None
    black_normalized_rating: int | None
    time_initial_ms: int | None
    time_increment_ms: int | None
    clock_remaining_ms: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if type(self.game_id) is not int or self.game_id < 0:
            raise ValueError("game_id must be a nonnegative integer")
        if self.ruleset != "standard":
            raise ValueError("the first per-ply encoding supports standard chess only")
        if not self.action_ids:
            raise ValueError("a game encoding input needs at least one action")
        if len(self.clock_remaining_ms) != len(self.action_ids):
            raise ValueError("clock observations must align one-to-one with actions")
        optional_values = {
            "white_normalized_rating": self.white_normalized_rating,
            "black_normalized_rating": self.black_normalized_rating,
            "time_initial_ms": self.time_initial_ms,
            "time_increment_ms": self.time_increment_ms,
        }
        for name, value in optional_values.items():
            _validate_optional_nonnegative_integer(name, value)
        for ply_index, value in enumerate(self.clock_remaining_ms):
            _validate_optional_nonnegative_integer(
                f"clock_remaining_ms[{ply_index}]", value
            )


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
class PlyContext:
    """One target-free model-input timestep."""

    ply_index: int
    board: BoardEncoding
    previous_action_id: int | None
    time_initial_ms: int | None
    time_increment_ms: int | None
    player_clock_ms: int | None
    opponent_clock_ms: int | None

    def as_record(self) -> dict[str, object]:
        """Return the stable JSON-serializable input representation."""

        return {
            "ply_index": self.ply_index,
            "board": self.board.as_record(),
            "previous_action_id": self.previous_action_id,
            "time_initial_ms": self.time_initial_ms,
            "time_increment_ms": self.time_increment_ms,
            "player_clock_ms": self.player_clock_ms,
            "opponent_clock_ms": self.opponent_clock_ms,
        }


@dataclass(frozen=True)
class PlyEncoding(PlyContext):
    """One aligned model-input timestep and its supervised action target."""

    game_id: int
    target_action_id: int
    legal_action_ids: tuple[int, ...]
    target_rating: int | None
    target_clock_after_move_ms: int | None

    def context(self) -> PlyContext:
        """Return the target-free input fields for train/inference comparison."""

        return PlyContext(
            ply_index=self.ply_index,
            board=self.board,
            previous_action_id=self.previous_action_id,
            time_initial_ms=self.time_initial_ms,
            time_increment_ms=self.time_increment_ms,
            player_clock_ms=self.player_clock_ms,
            opponent_clock_ms=self.opponent_clock_ms,
        )

    def as_record(self) -> dict[str, object]:
        """Return the stable JSON-serializable representation."""

        return {
            "game_id": self.game_id,
            "ply_index": self.ply_index,
            "board": self.board.as_record(),
            "previous_action_id": self.previous_action_id,
            "target_action_id": self.target_action_id,
            "legal_action_ids": list(self.legal_action_ids),
            "target_rating": self.target_rating,
            "time_initial_ms": self.time_initial_ms,
            "time_increment_ms": self.time_increment_ms,
            "player_clock_ms": self.player_clock_ms,
            "opponent_clock_ms": self.opponent_clock_ms,
            "target_clock_after_move_ms": self.target_clock_after_move_ms,
        }


@dataclass(frozen=True)
class DecisionContext:
    """A complete target-free trajectory ending at the current decision."""

    target_rating: int | None
    plies: tuple[PlyContext, ...]

    def __post_init__(self) -> None:
        if not self.plies:
            raise ValueError("a decision context needs at least one timestep")


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
        target_clock = game.clock_remaining_ms[ply_index]
        context = _context_for_position(
            ply_index=ply_index,
            board=board,
            previous_action_id=previous_action_id,
            time_initial_ms=game.time_initial_ms,
            time_increment_ms=game.time_increment_ms,
            player_clock_ms=clocks_by_color[player_color],
            opponent_clock_ms=clocks_by_color[not player_color],
        )
        encodings.append(
            PlyEncoding(
                game_id=game.game_id,
                ply_index=context.ply_index,
                board=context.board,
                previous_action_id=context.previous_action_id,
                time_initial_ms=context.time_initial_ms,
                time_increment_ms=context.time_increment_ms,
                player_clock_ms=context.player_clock_ms,
                opponent_clock_ms=context.opponent_clock_ms,
                target_action_id=target_action_id,
                legal_action_ids=legal_ids,
                target_rating=_rating_for_color(game, player_color),
                target_clock_after_move_ms=target_clock,
            )
        )
        clocks_by_color[player_color] = target_clock
        board.push(target_move)
        previous_action_id = target_action_id

    return tuple(encodings)


def build_decision_context(
    board: chess.Board,
    move_history: Sequence[chess.Move],
    *,
    target_rating: int | None,
) -> DecisionContext:
    """Build full target-free model context from one exact canonical board."""

    _validate_optional_nonnegative_integer("target_rating", target_rating)
    history = tuple(move_history)
    if tuple(board.move_stack) != history:
        raise EncodingError("move history does not match the canonical board stack")

    replay = board.root()
    plies: list[PlyContext] = []
    previous_action_id: int | None = None
    for ply_index in range(len(history) + 1):
        plies.append(
            _context_for_position(
                ply_index=ply_index,
                board=replay,
                previous_action_id=previous_action_id,
                time_initial_ms=None,
                time_increment_ms=None,
                player_clock_ms=None,
                opponent_clock_ms=None,
            )
        )
        if ply_index == len(history):
            break
        move = history[ply_index]
        if move not in replay.legal_moves:
            raise EncodingError(f"move history is illegal at ply {ply_index}")
        previous_action_id = _encode_observed_move(move, ply_index)
        replay.push(move)

    if replay.fen() != board.fen():
        raise EncodingError("move history does not reconstruct the canonical board")
    return DecisionContext(target_rating, tuple(plies))


def _context_for_position(
    *,
    ply_index: int,
    board: chess.Board,
    previous_action_id: int | None,
    time_initial_ms: int | None,
    time_increment_ms: int | None,
    player_clock_ms: int | None,
    opponent_clock_ms: int | None,
) -> PlyContext:
    return PlyContext(
        ply_index=ply_index,
        board=_encode_board(board),
        previous_action_id=previous_action_id,
        time_initial_ms=time_initial_ms,
        time_increment_ms=time_increment_ms,
        player_clock_ms=player_clock_ms,
        opponent_clock_ms=opponent_clock_ms,
    )


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


def _rating_for_color(game: GameEncodingInput, color: chess.Color) -> int | None:
    return (
        game.white_normalized_rating
        if color == chess.WHITE
        else game.black_normalized_rating
    )


def _encode_observed_move(move: chess.Move, ply_index: int) -> int:
    try:
        return encode_move(move)
    except ValueError as error:
        raise EncodingError(
            f"move history at ply {ply_index} is outside the action vocabulary"
        ) from error


def _validate_optional_nonnegative_integer(name: str, value: int | None) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer or None")
