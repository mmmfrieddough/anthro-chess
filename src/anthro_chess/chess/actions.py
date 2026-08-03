"""Stable model action ids for standard ``python-chess`` moves."""

from __future__ import annotations

from hashlib import sha256

import chess

ACTION_VOCABULARY_NAME = "anthro-standard-actions"
ACTION_VOCABULARY_VERSION = 2


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
DRAW_CLAIM_ACTION_ID = MOVE_ACTION_COUNT + 1
#: The actions that end a game instead of changing the position, in ascending
#: id order. They sit above every move id, so a terminal action is exactly an
#: action id at or past :data:`MOVE_ACTION_COUNT`.
TERMINAL_ACTION_IDS = (RESIGNATION_ACTION_ID, DRAW_CLAIM_ACTION_ID)
ACTION_VOCABULARY_SIZE = MOVE_ACTION_COUNT + len(TERMINAL_ACTION_IDS)
ACTION_VOCABULARY_SHA256 = sha256(
    "\n".join((*[move.uci() for move in _MOVES], "resign", "claim-draw")).encode(
        "ascii"
    )
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


def is_terminal_action(action_id: int) -> bool:
    """Return whether an action ends the game rather than moving a piece."""

    return action_id >= MOVE_ACTION_COUNT


def draw_claim_available(board: chess.Board) -> bool:
    """Return whether the player to move may claim a draw right now.

    Deliberately narrower than ``python-chess``'s ``can_claim_draw``, which
    also reports a claim that becomes available only after announcing a
    particular move. The draw-claim action carries no move, so the condition it
    can honestly represent is the one the current position already satisfies:
    the position has repeated three times, or the fifty-move clock is full and
    the game is not already over by another rule.
    """

    return board.is_repetition(3) or board.is_fifty_moves()


def legal_action_ids(
    board: chess.Board,
    *,
    include_resignation: bool = False,
    include_draw_claim: bool = False,
) -> tuple[int, ...]:
    """Return the action ids enabled by the board and runtime policy.

    Resignation is available in any position, so the caller's policy alone
    decides it. A draw claim is offered only when the caller enables it *and*
    exact chess logic says the claim exists; nothing here lets a policy enable
    a claim the rules do not support.
    """

    action_ids = sorted(encode_move(move) for move in board.legal_moves)
    if include_resignation:
        action_ids.append(RESIGNATION_ACTION_ID)
    if include_draw_claim and draw_claim_available(board):
        action_ids.append(DRAW_CLAIM_ACTION_ID)
    return tuple(action_ids)


def action_is_legal(
    board: chess.Board,
    action_id: int,
    *,
    include_resignation: bool = False,
    include_draw_claim: bool = False,
) -> bool:
    """Return whether one action id is enabled by the board and runtime policy.

    Answers exactly what membership in :func:`legal_action_ids` answers, for a
    caller holding a single candidate rather than needing the whole set. That
    caller generates one square's moves instead of every move on the board and
    encoding each one, which is most of what building the set costs.

    Deliberately not ``board.is_legal``, which is the cheaper test and the wrong
    one: it also accepts a castle written as the king taking its own rook. The
    generator never yields that spelling, so the set never carries it, and a
    membership test that did would accept a target no legal mask enables.
    """

    if action_id == RESIGNATION_ACTION_ID:
        return include_resignation
    if action_id == DRAW_CLAIM_ACTION_ID:
        return include_draw_claim and draw_claim_available(board)
    if not 0 <= action_id < MOVE_ACTION_COUNT:
        return False
    move = _MOVES[action_id]
    return move in board.generate_legal_moves(chess.BB_SQUARES[move.from_square])
