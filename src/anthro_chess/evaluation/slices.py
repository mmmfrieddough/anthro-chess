"""Derived position slices shared by offline and rollout evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import chess

from anthro_chess.chess import is_terminal_action
from anthro_chess.data import (
    BOARD_SQUARE_COUNT,
    BoardEncoding,
    PlyEncoding,
    Speed,
    speed_from_clock_ms,
)

SLICE_SCHEME_VERSION = 2


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


class PositionCharacteristic(StrEnum):
    """Rule-sensitive properties a position can hold.

    These isolate cases that are rare enough to vanish from an average over
    held-out positions. Slicing real games by them keeps the measurement on
    the true distribution instead of on hand-picked examples.
    """

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


class PredicateClass(StrEnum):
    """How strongly exact chess logic supports a position predicate."""

    DECIDABLE = "decidable"
    HEURISTIC = "heuristic"


class PositionPredicate(StrEnum):
    """Forward-looking decisions shared by adjudication and novelty evaluation."""

    MATE_AVAILABLE = "mate_available"
    MATE_THREATENED = "mate_threatened"
    STALEMATE_AVAILABLE = "stalemate_available"
    ONLY_MOVE = "only_move"
    MATERIAL_GAIN = "material_gain"


@dataclass(frozen=True)
class PredicateDefinition:
    """Stable metadata for one derived position predicate."""

    predicate: PositionPredicate
    classification: PredicateClass
    summary: str


@dataclass(frozen=True)
class PredicateMatch:
    """One predicate realized at a position and the actions that handle it."""

    predicate: PositionPredicate
    successful_action_ids: frozenset[int]


PREDICATE_REGISTRY: Mapping[PositionPredicate, PredicateDefinition] = {
    PositionPredicate.MATE_AVAILABLE: PredicateDefinition(
        predicate=PositionPredicate.MATE_AVAILABLE,
        classification=PredicateClass.DECIDABLE,
        summary="The side to move can checkmate immediately.",
    ),
    PositionPredicate.MATE_THREATENED: PredicateDefinition(
        predicate=PositionPredicate.MATE_THREATENED,
        classification=PredicateClass.DECIDABLE,
        summary="Passing would allow an immediate mate; successful moves remove it.",
    ),
    PositionPredicate.STALEMATE_AVAILABLE: PredicateDefinition(
        predicate=PositionPredicate.STALEMATE_AVAILABLE,
        classification=PredicateClass.DECIDABLE,
        summary="The side to move can end the game by stalemate immediately.",
    ),
    PositionPredicate.ONLY_MOVE: PredicateDefinition(
        predicate=PositionPredicate.ONLY_MOVE,
        classification=PredicateClass.DECIDABLE,
        summary="The side to move has exactly one legal move.",
    ),
    PositionPredicate.MATERIAL_GAIN: PredicateDefinition(
        predicate=PositionPredicate.MATERIAL_GAIN,
        classification=PredicateClass.HEURISTIC,
        summary=(
            "A capture wins material through the full exchange on its square; "
            "successful moves are those captures."
        ),
    ),
}

#: Material threshold a capture's exchange sequence has to clear to count as
#: winning. One pawn is the smallest gain worth calling a gain.
MATERIAL_GAIN_THRESHOLD = 1

#: What the king is worth when choosing which attacker recaptures. Priced far
#: above every other piece so it recaptures last, which is the conventional way
#: to keep it out of a swap-off it would never enter. This is an ordering
#: device rather than a valuation, which is why it does not belong in
#: ``MATERIAL_VALUES``: a balance that priced the king would be nonsense.
_EXCHANGE_KING_VALUE = 1000


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
    speed: Speed | None

    def as_record(self) -> dict[str, object]:
        """Return a stable JSON-serializable slice record."""

        return {
            "phase": str(self.phase),
            "color": str(self.color),
            "legal_move_count": self.legal_move_count,
            "legal_move_count_bucket": self.legal_move_count_bucket,
            "rating_band": self.rating_band,
            "speed": None if self.speed is None else str(self.speed),
        }


@dataclass(frozen=True)
class PositionLabels:
    """Everything exact chess logic says about one evaluated position.

    The two halves are derived together because they are the same work: the
    characteristics read the predicates, and both read the position's legal
    moves. Splitting them made each caller generate that list again.
    """

    predicates: Mapping[PositionPredicate, PredicateMatch]
    characteristics: frozenset[PositionCharacteristic]


def position_labels(board: chess.Board) -> PositionLabels:
    """Return one position's rule-sensitive labels from a single reading."""

    legal_moves = tuple(board.legal_moves)
    predicates = match_position_predicates(board, legal_moves=legal_moves)
    return PositionLabels(
        predicates=predicates,
        characteristics=board_characteristics(
            board,
            predicates=predicates,
            legal_moves=legal_moves,
        ),
    )


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


def board_from_encoding(board: BoardEncoding) -> chess.Board:
    """Rebuild an exact board from a compact per-ply encoding."""

    if len(board.piece_ids) != BOARD_SQUARE_COUNT:
        raise ValueError("piece ids must cover every board square")

    exact = chess.Board(None)
    for square, piece_id in zip(chess.SQUARES, board.piece_ids, strict=True):
        if piece_id == 0:
            continue
        if not 1 <= piece_id <= 12:
            raise ValueError(f"piece id is outside the encoding contract: {piece_id}")
        color = chess.WHITE if piece_id <= 6 else chess.BLACK
        piece_type = piece_id if piece_id <= 6 else piece_id - 6
        exact.set_piece_at(square, chess.Piece(piece_type, color))

    exact.turn = chess.WHITE if board.side_to_move == 0 else chess.BLACK
    rights = ""
    for bit, symbol in ((1, "K"), (2, "Q"), (4, "k"), (8, "q")):
        if board.castling_rights & bit:
            rights += symbol
    exact.set_castling_fen(rights or "-")
    exact.ep_square = board.en_passant_square
    exact.halfmove_clock = board.halfmove_clock
    exact.fullmove_number = board.fullmove_number
    return exact


def board_characteristics(
    board: chess.Board,
    *,
    predicates: Mapping[PositionPredicate, PredicateMatch] | None = None,
    legal_moves: Sequence[chess.Move] | None = None,
) -> frozenset[PositionCharacteristic]:
    """Return the rule-sensitive properties exact chess logic finds.

    This is the expensive slice, so it is applied to the positions a benchmark
    actually scores rather than computed for every ply of a pool. A caller that
    has already generated the position's legal moves passes them rather than
    paying for a second generation of the same list.
    """

    moves = tuple(board.legal_moves) if legal_moves is None else legal_moves
    observed: set[PositionCharacteristic] = set()
    if board.is_check():
        observed.add(PositionCharacteristic.CHECK)
    resolved = (
        match_position_predicates(board, legal_moves=moves)
        if predicates is None
        else predicates
    )
    if PositionPredicate.ONLY_MOVE in resolved:
        observed.add(PositionCharacteristic.ONLY_MOVE)
    if not moves:
        # Checkmate and stalemate both need a position with no legal move, so
        # asking which one this is belongs behind that test rather than in
        # front of every position that has one.
        observed.add(PositionCharacteristic.TERMINAL)
        if board.is_checkmate():
            observed.add(PositionCharacteristic.CHECKMATE)
        elif board.is_stalemate():
            observed.add(PositionCharacteristic.STALEMATE)
    if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
        observed.add(PositionCharacteristic.CASTLING_RIGHTS)
    if any(board.is_castling(move) for move in moves):
        observed.add(PositionCharacteristic.CASTLING_AVAILABLE)
    if any(board.is_en_passant(move) for move in moves):
        observed.add(PositionCharacteristic.EN_PASSANT)
    if any(move.promotion is not None for move in moves):
        observed.add(PositionCharacteristic.PROMOTION)
    if any(
        board.is_pinned(board.turn, square)
        for square in chess.SQUARES
        if (piece := board.piece_at(square)) is not None and piece.color == board.turn
    ):
        observed.add(PositionCharacteristic.PIN)
    return frozenset(observed)


def match_position_predicates(
    board: chess.Board,
    *,
    legal_moves: Sequence[chess.Move] | None = None,
) -> Mapping[PositionPredicate, PredicateMatch]:
    """Return every forward-looking predicate exact chess logic resolves.

    Threat predicates use the conventional null-move question: what could the
    opponent do immediately if the side to move passed? A successful action is
    then one that removes every such immediate reply. Null moves are used only
    to derive a label; they are never exposed as model actions.

    Material gain is the one heuristic predicate here, and it resolves the
    exchange rather than counting the captured piece. Plain counting would
    admit every capture of a defended piece, which is not a gain at all; the
    exchange sequence is still deterministic and identically applied to both
    sides, which is what a human-referenced predicate needs. It is not a claim
    that the capture is objectively best.
    """

    moves = tuple(board.legal_moves) if legal_moves is None else tuple(legal_moves)
    if not moves:
        return {}

    action_ids = {move: _move_action_id(move) for move in moves}
    matches: dict[PositionPredicate, PredicateMatch] = {}

    if len(moves) == 1:
        matches[PositionPredicate.ONLY_MOVE] = PredicateMatch(
            predicate=PositionPredicate.ONLY_MOVE,
            successful_action_ids=frozenset(action_ids.values()),
        )

    mates, stalemates = _terminal_moves(board, moves)
    if mates:
        matches[PositionPredicate.MATE_AVAILABLE] = PredicateMatch(
            predicate=PositionPredicate.MATE_AVAILABLE,
            successful_action_ids=frozenset(action_ids[move] for move in mates),
        )
    if stalemates:
        matches[PositionPredicate.STALEMATE_AVAILABLE] = PredicateMatch(
            predicate=PositionPredicate.STALEMATE_AVAILABLE,
            successful_action_ids=frozenset(action_ids[move] for move in stalemates),
        )

    winning = material_winning_moves(board, moves)
    if winning:
        matches[PositionPredicate.MATERIAL_GAIN] = PredicateMatch(
            predicate=PositionPredicate.MATERIAL_GAIN,
            successful_action_ids=frozenset(action_ids[move] for move, _ in winning),
        )

    if board.is_check():
        return matches

    passed = board.copy(stack=True)
    passed.push(chess.Move.null())
    if _has_terminal_reply(passed, checkmate=True):
        safe = _moves_avoiding_reply(board, moves, checkmate=True)
        matches[PositionPredicate.MATE_THREATENED] = PredicateMatch(
            predicate=PositionPredicate.MATE_THREATENED,
            successful_action_ids=frozenset(action_ids[move] for move in safe),
        )
    return matches


def _terminal_moves(
    board: chess.Board,
    legal_moves: Sequence[chess.Move],
) -> tuple[tuple[chess.Move, ...], tuple[chess.Move, ...]]:
    """Return which of ``legal_moves`` deliver mate and which force stalemate.

    Both outcomes need a reply position with no legal move at all, so that one
    question is asked first and exact chess logic is only asked to tell the two
    apart where the answer can be either.
    """

    mates: list[chess.Move] = []
    stalemates: list[chess.Move] = []
    for move in legal_moves:
        board.push(move)
        if not any(board.generate_legal_moves()):
            if board.is_checkmate():
                mates.append(move)
            elif board.is_stalemate():
                stalemates.append(move)
        board.pop()
    return tuple(mates), tuple(stalemates)


def _has_terminal_reply(board: chess.Board, *, checkmate: bool) -> bool:
    """Return whether any legal move from ``board`` ends the game that way.

    Every caller asks whether such a reply exists rather than which moves are
    it, so the first one found settles the question and the rest of the scan
    never happens.
    """

    for move in board.legal_moves:
        board.push(move)
        ended = board.is_checkmate() if checkmate else board.is_stalemate()
        board.pop()
        if ended:
            return True
    return False


def _moves_avoiding_reply(
    board: chess.Board,
    legal_moves: Sequence[chess.Move],
    *,
    checkmate: bool,
) -> tuple[chess.Move, ...]:
    safe: list[chess.Move] = []
    for move in legal_moves:
        board.push(move)
        threatened = _has_terminal_reply(board, checkmate=checkmate)
        board.pop()
        if not threatened:
            safe.append(move)
    return tuple(safe)


def material_winning_moves(
    board: chess.Board,
    legal_moves: Sequence[chess.Move],
) -> tuple[tuple[chess.Move, int], ...]:
    """Return the captures whose exchange sequence nets material, and what each nets.

    Only captures are considered. A quiet move that wins material needs an
    evaluation function to recognize, which is the dependency this project
    keeps out of benchmark time.

    The amount comes back with the move because resolving an exchange is most
    of what matching this predicate costs, and a caller that reads how large
    the win is would otherwise resolve every one of them a second time.
    """

    winning = (
        (move, _exchange_gain(board, move))
        for move in legal_moves
        if board.is_capture(move)
    )
    return tuple(
        (move, gain) for move, gain in winning if gain >= MATERIAL_GAIN_THRESHOLD
    )


def _exchange_value(piece_type: int | None) -> int:
    """Return one piece's worth inside an exchange resolution.

    Reads the shared material table so a pawn is worth the same here as in a
    balance, with the king's ordering price applied on top. An empty square
    resolves to a pawn, which only arises for an en-passant capture whose
    target square holds nothing.
    """

    if piece_type is None:
        return MATERIAL_VALUES[chess.PAWN]
    if piece_type == chess.KING:
        return _EXCHANGE_KING_VALUE
    return MATERIAL_VALUES[piece_type]


def _exchange_gain(board: chess.Board, move: chess.Move) -> int:
    """Return the material one capture nets once the exchange is played out.

    The exchange is resolved through exact legal move generation rather than
    through a bitboard approximation, so pins, discovered attacks, and the
    king's inability to recapture into check are handled by the chess layer
    instead of by a second implementation of the rules. It costs a handful of
    pushes per capture, which is why predicates belong to the positions a
    benchmark actually scores.
    """

    captured = (
        MATERIAL_VALUES[chess.PAWN]
        if board.is_en_passant(move)
        else _exchange_value(board.piece_type_at(move.to_square))
    )
    board.push(move)
    try:
        at_risk = _exchange_value(board.piece_type_at(move.to_square))
        return captured - _continue_exchange(board, move.to_square, at_risk)
    finally:
        board.pop()


def _continue_exchange(board: chess.Board, square: chess.Square, at_risk: int) -> int:
    """Return what the side to move wins by recapturing on ``square``.

    Standing pat is always available, so a side never continues into a loss.
    The least valuable attacker recaptures, with the move's own notation
    breaking ties so the resolution is deterministic.
    """

    captures = [
        move
        for move in board.legal_moves
        if move.to_square == square and board.is_capture(move)
    ]
    if not captures:
        return 0
    move = min(
        captures,
        key=lambda candidate: (
            _exchange_value(board.piece_type_at(candidate.from_square)),
            candidate.uci(),
        ),
    )
    board.push(move)
    try:
        next_at_risk = _exchange_value(board.piece_type_at(square))
        return max(0, at_risk - _continue_exchange(board, square, next_at_risk))
    finally:
        board.pop()


#: Conventional pawn values for the material proxy. Deliberately the textbook
#: numbers rather than tuned ones: the proxy's job is to be identical on both
#: sides of a human-referenced comparison, and a tuned table would make it a
#: position evaluation this project has declined to own.
MATERIAL_VALUES: Mapping[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_balance(board: chess.Board, color: chess.Color) -> int:
    """Return ``color``'s material advantage in pawns, negative when behind.

    One definition, shared by every reading that needs a dependency-free
    position-quality signal: a reading that compares a model against humans is
    only meaningful when the same arithmetic ran on both sides.
    """

    own = sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in MATERIAL_VALUES.items()
    )
    opponent = sum(
        len(board.pieces(piece_type, not color)) * value
        for piece_type, value in MATERIAL_VALUES.items()
    )
    return own - opponent


def _move_action_id(move: chess.Move) -> int:
    # Import locally so the slice module remains usable by lightweight result
    # readers without pulling the action codec into module initialization.
    from anthro_chess.chess import encode_move

    return encode_move(move)


def ply_characteristics(ply: PlyEncoding) -> frozenset[PositionCharacteristic]:
    """Return the rule-sensitive properties of one encoded ply."""

    return board_characteristics(board_from_encoding(ply.board))


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
    """Return every derived slice label for one encoded ply.

    The move count is the position's branching factor, so the terminal actions
    an encoding also enables are left out: they are available in every position
    and would shift every bucket by a constant.

    Speed comes from the game's control rather than the clock left at the ply,
    so every decision in a blitz game counts toward blitz, endgame included.
    """

    legal_moves = sum(
        1 for action_id in ply.enabled_actions() if not is_terminal_action(action_id)
    )
    return PositionSlices(
        phase=game_phase(ply.board.piece_ids, ply.board.fullmove_number),
        color=board_color(ply.board),
        legal_move_count=legal_moves,
        legal_move_count_bucket=legal_move_count_bucket(legal_moves),
        rating_band=rating_band_name(ply.target_rating, rating_bands),
        speed=speed_from_clock_ms(ply.time_initial_ms, ply.time_increment_ms),
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
