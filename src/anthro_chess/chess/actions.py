"""Stable model action ids for standard ``python-chess`` moves."""

from __future__ import annotations

from hashlib import sha256

import chess

ACTION_VOCABULARY_NAME = "anthro-standard-actions"
ACTION_VOCABULARY_VERSION = 1


def _standard_moves() -> tuple[chess.Move, ...]:
    moves: set[chess.Move] = set()
    for from_square in chess.SQUARES:
        from_file = chess.square_file(from_square)
        from_rank = chess.square_rank(from_square)
        for to_square in chess.SQUARES:
            to_file = chess.square_file(to_square)
            to_rank = chess.square_rank(to_square)
            file_distance = abs(to_file - from_file)
            rank_distance = abs(to_rank - from_rank)
            is_sliding = (
                file_distance == 0
                or rank_distance == 0
                or file_distance == rank_distance
            )
            is_knight = sorted((file_distance, rank_distance)) == [1, 2]
            if from_square != to_square and (is_sliding or is_knight):
                moves.add(chess.Move(from_square, to_square))

    for from_rank, to_rank in ((6, 7), (1, 0)):
        for from_file in range(8):
            from_square = chess.square(from_file, from_rank)
            for to_file in range(max(0, from_file - 1), min(7, from_file + 1) + 1):
                to_square = chess.square(to_file, to_rank)
                for promotion in chess.PIECE_TYPES[1:5]:
                    moves.add(chess.Move(from_square, to_square, promotion=promotion))

    return tuple(sorted(moves, key=lambda move: move.uci()))


_MOVES = _standard_moves()
_MOVE_TO_ACTION_ID = {move: action_id for action_id, move in enumerate(_MOVES)}

MOVE_ACTION_COUNT = len(_MOVES)
RESIGNATION_ACTION_ID = MOVE_ACTION_COUNT
ACTION_VOCABULARY_SIZE = MOVE_ACTION_COUNT + 1
ACTION_VOCABULARY_SHA256 = sha256(
    "\n".join((*[move.uci() for move in _MOVES], "resign")).encode("ascii")
).hexdigest()


def action_vocabulary_identity() -> dict[str, str | int]:
    """Return the serializable identity stored with data and model artifacts."""

    return {
        "name": ACTION_VOCABULARY_NAME,
        "version": ACTION_VOCABULARY_VERSION,
        "size": ACTION_VOCABULARY_SIZE,
        "sha256": ACTION_VOCABULARY_SHA256,
    }


def encode_move(move: chess.Move) -> int:
    """Return the stable action id for a standard-chess move."""

    try:
        return _MOVE_TO_ACTION_ID[move]
    except KeyError as error:
        raise ValueError(
            f"move is outside the standard action vocabulary: {move.uci()}"
        ) from error


def decode_move(action_id: int) -> chess.Move:
    """Return the standard-chess move for a non-resignation action id."""

    if type(action_id) is not int or not 0 <= action_id < MOVE_ACTION_COUNT:
        raise ValueError(
            f"move action id must be an integer from 0 to {MOVE_ACTION_COUNT - 1}"
        )
    return _MOVES[action_id]


def legal_action_ids(
    board: chess.Board, *, include_resignation: bool = False
) -> tuple[int, ...]:
    """Return the action ids enabled by the board and runtime policy."""

    action_ids = sorted(encode_move(move) for move in board.legal_moves)
    if include_resignation:
        action_ids.append(RESIGNATION_ACTION_ID)
    return tuple(action_ids)
