"""Stable adapters around the project's exact chess-rules implementation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import chess


class ChessError(ValueError):
    """Base error for invalid chess input at the Anthro boundary."""


class InvalidPositionError(ChessError):
    """Raised when a position cannot represent a valid standard chess board."""


class InvalidMoveError(ChessError):
    """Raised when move text is malformed or ambiguous."""


class IllegalMoveError(ChessError):
    """Raised when a well-formed move is not legal in its position."""


class Color(StrEnum):
    """Side to move without exposing a third-party boolean convention."""

    WHITE = "white"
    BLACK = "black"


class Promotion(StrEnum):
    """Promotion piece encoded in a standard move action."""

    KNIGHT = "n"
    BISHOP = "b"
    ROOK = "r"
    QUEEN = "q"


class PieceType(StrEnum):
    """Piece kind in an exact board snapshot."""

    PAWN = "pawn"
    KNIGHT = "knight"
    BISHOP = "bishop"
    ROOK = "rook"
    QUEEN = "queen"
    KING = "king"


_PROMOTION_TO_CHESS = {
    Promotion.KNIGHT: chess.KNIGHT,
    Promotion.BISHOP: chess.BISHOP,
    Promotion.ROOK: chess.ROOK,
    Promotion.QUEEN: chess.QUEEN,
}
_PROMOTION_FROM_CHESS = {value: key for key, value in _PROMOTION_TO_CHESS.items()}
_PIECE_TYPE_FROM_CHESS = {
    chess.PAWN: PieceType.PAWN,
    chess.KNIGHT: PieceType.KNIGHT,
    chess.BISHOP: PieceType.BISHOP,
    chess.ROOK: PieceType.ROOK,
    chess.QUEEN: PieceType.QUEEN,
    chess.KING: PieceType.KING,
}


@dataclass(frozen=True, slots=True)
class Piece:
    """A colored piece in an exact board snapshot."""

    color: Color
    piece_type: PieceType


@dataclass(frozen=True, slots=True)
class CastlingRights:
    """Standard castling rights retained by the position."""

    white_kingside: bool
    white_queenside: bool
    black_kingside: bool
    black_queenside: bool


@dataclass(frozen=True, order=True, slots=True)
class Move:
    """A protocol-independent board move using square indexes and promotion."""

    from_square: int
    to_square: int
    promotion: Promotion | None = None

    def __post_init__(self) -> None:
        for field_name, square in (
            ("from_square", self.from_square),
            ("to_square", self.to_square),
        ):
            if type(square) is not int or not 0 <= square < 64:
                raise InvalidMoveError(f"{field_name} must be an integer from 0 to 63")
        if self.from_square == self.to_square:
            raise InvalidMoveError("a move must change squares")
        if self.promotion is not None and not isinstance(self.promotion, Promotion):
            raise InvalidMoveError("promotion must be a Promotion value or None")

    @classmethod
    def from_uci(cls, value: str) -> Move:
        """Parse standard coordinate notation at an outside-interface boundary."""

        try:
            return cls._from_chess(chess.Move.from_uci(value))
        except (chess.InvalidMoveError, ValueError) as error:
            raise InvalidMoveError(f"invalid move text: {value!r}") from error

    @property
    def uci(self) -> str:
        """Render coordinate notation for storage, debugging, or protocols."""

        return self._to_chess().uci()

    @classmethod
    def _from_chess(cls, move: chess.Move) -> Move:
        promotion = (
            _PROMOTION_FROM_CHESS[move.promotion]
            if move.promotion is not None
            else None
        )
        return cls(move.from_square, move.to_square, promotion)

    def _to_chess(self) -> chess.Move:
        promotion = (
            _PROMOTION_TO_CHESS[self.promotion] if self.promotion is not None else None
        )
        return chess.Move(self.from_square, self.to_square, promotion=promotion)


class Position:
    """An immutable view of exact standard-chess state and rule bookkeeping."""

    __slots__ = ("_board",)

    def __init__(self, fen: str = chess.STARTING_FEN) -> None:
        self._board = self._validated_board(fen)

    @classmethod
    def initial(cls) -> Position:
        """Return the standard initial position."""

        return cls()

    @classmethod
    def from_fen(cls, fen: str) -> Position:
        """Build a position from a complete, valid standard-chess FEN."""

        return cls(fen)

    @staticmethod
    def _validated_board(fen: str) -> chess.Board:
        try:
            board = chess.Board(fen)
        except ValueError as error:
            raise InvalidPositionError(f"invalid FEN: {error}") from error
        if not board.is_valid():
            raise InvalidPositionError(
                f"invalid standard-chess position (status={int(board.status())})"
            )
        return board

    @classmethod
    def _from_board(cls, board: chess.Board) -> Position:
        position = cls.__new__(cls)
        position._board = board.copy(stack=True)
        return position

    @property
    def fen(self) -> str:
        """Return a complete FEN, preserving the exact en-passant target."""

        return self._board.fen(en_passant="fen")

    @property
    def turn(self) -> Color:
        """Return the side to move."""

        return Color.WHITE if self._board.turn == chess.WHITE else Color.BLACK

    @property
    def is_check(self) -> bool:
        """Whether the side to move is in check."""

        return self._board.is_check()

    @property
    def is_game_over(self) -> bool:
        """Whether standard automatic game-ending rules have ended the game."""

        return self._board.is_game_over()

    @property
    def pieces(self) -> tuple[Piece | None, ...]:
        """Return the board contents indexed from a1 through h8."""

        pieces: list[Piece | None] = []
        for square in chess.SQUARES:
            piece = self._board.piece_at(square)
            pieces.append(
                None
                if piece is None
                else Piece(
                    Color.WHITE if piece.color == chess.WHITE else Color.BLACK,
                    _PIECE_TYPE_FROM_CHESS[piece.piece_type],
                )
            )
        return tuple(pieces)

    @property
    def castling_rights(self) -> CastlingRights:
        """Return exact standard kingside and queenside castling rights."""

        return CastlingRights(
            white_kingside=self._board.has_kingside_castling_rights(chess.WHITE),
            white_queenside=self._board.has_queenside_castling_rights(chess.WHITE),
            black_kingside=self._board.has_kingside_castling_rights(chess.BLACK),
            black_queenside=self._board.has_queenside_castling_rights(chess.BLACK),
        )

    @property
    def en_passant_square(self) -> int | None:
        """Return the exact en-passant target square, when one exists."""

        return self._board.ep_square

    @property
    def halfmove_clock(self) -> int:
        """Return the halfmove clock used by draw rules."""

        return self._board.halfmove_clock

    @property
    def fullmove_number(self) -> int:
        """Return the one-based fullmove number."""

        return self._board.fullmove_number

    @property
    def legal_moves(self) -> tuple[Move, ...]:
        """Return all legal moves in deterministic coordinate order."""

        return tuple(
            sorted(
                (Move._from_chess(move) for move in self._board.legal_moves),
                key=lambda move: move.uci,
            )
        )

    def parse_uci(self, value: str) -> Move:
        """Parse and validate a coordinate move for this position."""

        try:
            return Move._from_chess(self._board.parse_uci(value))
        except chess.IllegalMoveError as error:
            raise IllegalMoveError(f"illegal move {value!r} for {self.fen}") from error
        except (chess.InvalidMoveError, ValueError) as error:
            raise InvalidMoveError(f"invalid move text: {value!r}") from error

    def parse_san(self, value: str) -> Move:
        """Parse and validate standard algebraic notation for this position."""

        try:
            return Move._from_chess(self._board.parse_san(value))
        except chess.IllegalMoveError as error:
            raise IllegalMoveError(f"illegal SAN {value!r} for {self.fen}") from error
        except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError) as error:
            raise InvalidMoveError(f"invalid SAN: {value!r}") from error

    def apply(self, move: Move) -> Position:
        """Return the exact position after a legal move without mutating this one."""

        chess_move = move._to_chess()
        if chess_move not in self._board.legal_moves:
            raise IllegalMoveError(f"illegal move {move.uci!r} for {self.fen}")
        board = self._board.copy(stack=True)
        board.push(chess_move)
        return Position._from_board(board)


@dataclass(frozen=True, slots=True)
class ReplayedGame:
    """Exact positions before every move plus the resulting final position."""

    positions_before: tuple[Position, ...]
    moves: tuple[Move, ...]
    final_position: Position


def replay_moves(
    moves: Iterable[Move], *, initial_position: Position | None = None
) -> ReplayedGame:
    """Replay validated moves and retain the exact position before every ply."""

    position = initial_position or Position.initial()
    positions: list[Position] = []
    replayed_moves: list[Move] = []
    for ply_index, move in enumerate(moves):
        positions.append(position)
        try:
            position = position.apply(move)
        except IllegalMoveError as error:
            raise IllegalMoveError(
                f"illegal move at ply {ply_index}: {error}"
            ) from error
        replayed_moves.append(move)
    return ReplayedGame(tuple(positions), tuple(replayed_moves), position)


def replay_san(
    moves: Iterable[str], *, initial_position: Position | None = None
) -> ReplayedGame:
    """Parse a SAN mainline while reconstructing exact positions before each ply."""

    position = initial_position or Position.initial()
    positions: list[Position] = []
    replayed_moves: list[Move] = []
    for ply_index, san in enumerate(moves):
        positions.append(position)
        try:
            move = position.parse_san(san)
            position = position.apply(move)
        except ChessError as error:
            raise type(error)(f"invalid move at ply {ply_index}: {error}") from error
        replayed_moves.append(move)
    return ReplayedGame(tuple(positions), tuple(replayed_moves), position)
