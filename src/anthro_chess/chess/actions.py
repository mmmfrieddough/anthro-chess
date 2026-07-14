"""The shared, versioned action vocabulary for standard chess."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import TypeAlias

import chess

from anthro_chess.chess.core import Move, Position


class InvalidActionError(ValueError):
    """Raised when an action or action id is outside the shared vocabulary."""


class NonMoveAction(StrEnum):
    """Game actions that are not legal board moves."""

    RESIGN = "resign"


Action: TypeAlias = Move | NonMoveAction
RESIGN = NonMoveAction.RESIGN


@dataclass(frozen=True, slots=True)
class VocabularyIdentity:
    """Serializable compatibility identity for artifacts and checkpoints."""

    name: str
    version: int
    size: int
    sha256: str

    def as_record(self) -> dict[str, str | int]:
        """Return a JSON-compatible identity record."""

        return {
            "name": self.name,
            "version": self.version,
            "size": self.size,
            "sha256": self.sha256,
        }


def _move_tokens() -> tuple[str, ...]:
    tokens: set[str] = set()
    for from_square in chess.SQUARES:
        from_file = chess.square_file(from_square)
        from_rank = chess.square_rank(from_square)
        for to_square in chess.SQUARES:
            to_file = chess.square_file(to_square)
            to_rank = chess.square_rank(to_square)
            file_distance = abs(to_file - from_file)
            rank_distance = abs(to_rank - from_rank)
            if not file_distance and not rank_distance:
                continue
            is_sliding = (
                file_distance == 0
                or rank_distance == 0
                or file_distance == rank_distance
            )
            is_knight = sorted((file_distance, rank_distance)) == [1, 2]
            if is_sliding or is_knight:
                tokens.add(chess.Move(from_square, to_square).uci())

    for from_rank, to_rank in ((6, 7), (1, 0)):
        for from_file in range(8):
            from_square = chess.square(from_file, from_rank)
            for to_file in range(max(0, from_file - 1), min(7, from_file + 1) + 1):
                to_square = chess.square(to_file, to_rank)
                for promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                    tokens.add(
                        chess.Move(from_square, to_square, promotion=promotion).uci()
                    )
    return tuple(sorted(tokens))


_TOKENS = (*_move_tokens(), RESIGN.value)
_TOKEN_TO_ID = {token: action_id for action_id, token in enumerate(_TOKENS)}
_VOCABULARY_BYTES = "\n".join(_TOKENS).encode("ascii")


class StandardActionCodec:
    """Bidirectional codec shared by data, models, evaluation, and runtime."""

    @property
    def identity(self) -> VocabularyIdentity:
        """Return the exact compatibility identity for this vocabulary."""

        return VocabularyIdentity(
            name="anthro-standard-actions",
            version=1,
            size=len(_TOKENS),
            sha256=sha256(_VOCABULARY_BYTES).hexdigest(),
        )

    @property
    def move_vocabulary_size(self) -> int:
        """Number of board-move entries, excluding resignation."""

        return len(_TOKENS) - 1

    @property
    def resignation_id(self) -> int:
        """Action id for resignation."""

        return _TOKEN_TO_ID[RESIGN.value]

    def encode(self, action: Action) -> int:
        """Encode a board move or resignation as a stable compact integer."""

        token = action.uci if isinstance(action, Move) else action.value
        try:
            return _TOKEN_TO_ID[token]
        except KeyError as error:
            raise InvalidActionError(
                f"action is outside the standard vocabulary: {token}"
            ) from error

    def decode(self, action_id: int) -> Action:
        """Decode a stable integer into a board move or resignation."""

        if type(action_id) is not int or not 0 <= action_id < len(_TOKENS):
            raise InvalidActionError(
                f"action id must be an integer from 0 to {len(_TOKENS) - 1}"
            )
        token = _TOKENS[action_id]
        return RESIGN if token == RESIGN.value else Move.from_uci(token)

    def legal_action_ids(
        self, position: Position, *, include_resignation: bool = False
    ) -> tuple[int, ...]:
        """Return sorted ids enabled by exact chess rules and runtime policy."""

        action_ids = [self.encode(move) for move in position.legal_moves]
        if include_resignation:
            action_ids.append(self.resignation_id)
        return tuple(sorted(action_ids))


STANDARD_ACTION_CODEC = StandardActionCodec()
