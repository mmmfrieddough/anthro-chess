"""Fixed rule-sensitive position suites for legality diagnostics.

These positions answer whether the code is correct, not whether the model is
good. They are hand-authored rather than sampled, so declared characteristics
are verified against exact chess logic when the suite loads: a mislabeled
position fails loudly instead of quietly weakening the suite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Any

import chess

from anthro_chess.chess import legal_action_ids
from anthro_chess.evaluation.slices import (
    GamePhase,
    PlayerColor,
    board_phase,
    legal_move_count_bucket,
)

POSITION_SUITE_PACKAGE = "anthro_chess.evaluation.position_suites"
DEFAULT_POSITION_SUITE = "tricky-rules-v1"


class PositionSuiteError(ValueError):
    """Raised when a position suite is malformed or mislabeled."""


class PositionCharacteristic(StrEnum):
    """Rule-sensitive properties a suite position can declare."""

    CHECK = "check"
    PIN = "pin"
    CASTLING_RIGHTS = "castling_rights"
    CASTLING_AVAILABLE = "castling_available"
    EN_PASSANT = "en_passant"
    PROMOTION = "promotion"
    ONLY_MOVE = "only_move"
    TERMINAL = "terminal"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"


@dataclass(frozen=True)
class SuitePosition:
    """One labeled position plus the facts derived from exact chess logic."""

    id: str
    fen: str
    characteristics: frozenset[PositionCharacteristic]
    notes: str
    color: PlayerColor
    phase: GamePhase
    legal_action_ids: tuple[int, ...]

    @property
    def legal_move_count(self) -> int:
        """Return how many legal moves the side to move has."""

        return len(self.legal_action_ids)

    @property
    def is_terminal(self) -> bool:
        """Return whether the position has no legal continuation."""

        return not self.legal_action_ids

    def as_record(self) -> dict[str, object]:
        """Return a stable JSON-serializable position record."""

        return {
            "id": self.id,
            "fen": self.fen,
            "characteristics": sorted(str(item) for item in self.characteristics),
            "color": str(self.color),
            "phase": str(self.phase),
            "legal_move_count": self.legal_move_count,
            "legal_move_count_bucket": (
                legal_move_count_bucket(self.legal_move_count)
                if not self.is_terminal
                else None
            ),
        }


@dataclass(frozen=True)
class PositionSuite:
    """A loaded, verified fixed position suite."""

    suite_id: str
    version: int
    description: str
    positions: tuple[SuitePosition, ...]

    def identity(self) -> dict[str, object]:
        """Return the identity recorded alongside diagnostics that use it."""

        return {
            "suite_id": self.suite_id,
            "version": self.version,
            "positions": len(self.positions),
        }

    def with_characteristic(
        self, characteristic: PositionCharacteristic
    ) -> tuple[SuitePosition, ...]:
        """Return the positions declaring one characteristic."""

        return tuple(
            position
            for position in self.positions
            if characteristic in position.characteristics
        )

    def scorable_positions(self) -> tuple[SuitePosition, ...]:
        """Return positions a model can be asked to move in.

        Terminal positions are part of the suite because runtime code must
        handle them, but they have no action to predict.
        """

        return tuple(
            position for position in self.positions if not position.is_terminal
        )


def load_position_suite(
    name: str = DEFAULT_POSITION_SUITE,
    *,
    path: str | Path | None = None,
) -> PositionSuite:
    """Load and verify a packaged suite, or an explicit suite file."""

    if path is not None:
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PositionSuiteError(
                f"cannot load position suite {source}: {error}"
            ) from error
    else:
        resource = resources.files(POSITION_SUITE_PACKAGE) / f"{name}.json"
        try:
            raw = json.loads(resource.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PositionSuiteError(
                f"cannot load packaged position suite {name}: {error}"
            ) from error

    if not isinstance(raw, dict):
        raise PositionSuiteError("position suite must be a JSON object")
    suite_id = raw.get("suite_id")
    version = raw.get("version")
    description = raw.get("description", "")
    entries = raw.get("positions")
    if not isinstance(suite_id, str) or not suite_id:
        raise PositionSuiteError("position suite needs a non-empty suite_id")
    if type(version) is not int or version < 1:
        raise PositionSuiteError("position suite needs a positive integer version")
    if not isinstance(entries, list) or not entries:
        raise PositionSuiteError("position suite needs at least one position")

    positions = tuple(_position(entry) for entry in entries)
    identifiers = [position.id for position in positions]
    if len(set(identifiers)) != len(identifiers):
        raise PositionSuiteError("position suite ids must be unique")
    return PositionSuite(
        suite_id=suite_id,
        version=version,
        description=str(description),
        positions=positions,
    )


def _position(entry: Any) -> SuitePosition:
    if not isinstance(entry, dict):
        raise PositionSuiteError("each suite position must be a JSON object")
    identifier = entry.get("id")
    fen = entry.get("fen")
    declared = entry.get("characteristics", [])
    notes = entry.get("notes", "")
    if not isinstance(identifier, str) or not identifier:
        raise PositionSuiteError("each suite position needs a non-empty id")
    if not isinstance(fen, str) or not fen:
        raise PositionSuiteError(f"suite position {identifier} needs a FEN")
    if not isinstance(declared, list):
        raise PositionSuiteError(
            f"suite position {identifier} characteristics must be a list"
        )

    try:
        board = chess.Board(fen)
    except ValueError as error:
        raise PositionSuiteError(
            f"suite position {identifier} has an invalid FEN: {error}"
        ) from error
    if not board.is_valid():
        raise PositionSuiteError(f"suite position {identifier} is not a legal position")

    try:
        characteristics = frozenset(PositionCharacteristic(item) for item in declared)
    except ValueError as error:
        raise PositionSuiteError(
            f"suite position {identifier} declares an unknown characteristic: {error}"
        ) from error

    observed = _observed_characteristics(board)
    wrong = sorted(str(item) for item in characteristics - observed)
    if wrong:
        raise PositionSuiteError(
            f"suite position {identifier} declares {', '.join(wrong)} "
            "but exact chess logic disagrees"
        )

    return SuitePosition(
        id=identifier,
        fen=fen,
        characteristics=characteristics,
        notes=str(notes),
        color=PlayerColor.WHITE if board.turn == chess.WHITE else PlayerColor.BLACK,
        phase=board_phase(board),
        legal_action_ids=legal_action_ids(board),
    )


def _observed_characteristics(
    board: chess.Board,
) -> frozenset[PositionCharacteristic]:
    """Return the characteristics exact chess logic finds in a position."""

    legal_moves = list(board.legal_moves)
    observed: set[PositionCharacteristic] = set()
    if board.is_check():
        observed.add(PositionCharacteristic.CHECK)
    if len(legal_moves) == 1:
        observed.add(PositionCharacteristic.ONLY_MOVE)
    if not legal_moves:
        observed.add(PositionCharacteristic.TERMINAL)
    if board.is_checkmate():
        observed.add(PositionCharacteristic.CHECKMATE)
    if board.is_stalemate():
        observed.add(PositionCharacteristic.STALEMATE)
    if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
        observed.add(PositionCharacteristic.CASTLING_RIGHTS)
    if any(board.is_castling(move) for move in legal_moves):
        observed.add(PositionCharacteristic.CASTLING_AVAILABLE)
    if any(board.is_en_passant(move) for move in legal_moves):
        observed.add(PositionCharacteristic.EN_PASSANT)
    if any(move.promotion is not None for move in legal_moves):
        observed.add(PositionCharacteristic.PROMOTION)
    if any(
        board.is_pinned(board.turn, square)
        for square in chess.SQUARES
        if (piece := board.piece_at(square)) is not None and piece.color == board.turn
    ):
        observed.add(PositionCharacteristic.PIN)
    return frozenset(observed)
