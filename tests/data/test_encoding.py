import json
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import chess
import pytest

from anthro_chess.chess import (
    DRAW_CLAIM_ACTION_ID,
    RESIGNATION_ACTION_ID,
    decode_move,
    encode_move,
)
from anthro_chess.data import (
    BOARD_SQUARE_COUNT,
    DecisionColumn,
    DecisionHistory,
    EncodingError,
    GameEncodingInput,
    build_decision_context,
    en_passant_token,
    encode_game,
    encoding_identity,
)
from anthro_chess.data import encoding as encoding_module

PRESENT_60_SECONDS = 60_000


def test_encodes_exact_positions_and_legal_targets() -> None:
    game = _game(("e2e4", "e7e5", "g1f3"))

    plies = encode_game(game)

    assert len(plies) == 3
    first, second, third = plies
    assert first.ply_index == 0
    assert first.board.side_to_move == 0
    assert first.board.fullmove_number == 1
    assert first.board.castling_rights == 15
    assert len(first.board.piece_ids) == BOARD_SQUARE_COUNT
    assert first.board.piece_ids[chess.E2] == chess.PAWN
    assert first.target_action_id in first.enabled_actions()

    assert second.board.side_to_move == 1
    assert second.board.en_passant_square == chess.E3
    assert second.board.piece_ids[chess.E4] == chess.PAWN

    assert third.board.fullmove_number == 2
    assert third.board.piece_ids[chess.E5] == chess.PAWN + 6


@pytest.mark.parametrize(
    ("initial_position", "moves"),
    (
        # A trailing move on each, because a ply carries the position before
        # its own move: without one the last interesting position is encoded by
        # nothing and compared against nothing.
        (
            chess.STARTING_FEN,
            ("e2e4", "e7e6", "g1f3", "g8f6", "f1e2", "f8e7", "e1g1", "e8g8", "d2d4"),
        ),
        (chess.STARTING_FEN, ("e2e4", "a7a6", "e4e5", "d7d5", "e5d6", "c7d6")),
        ("8/P6k/8/8/8/8/6K1/8 w - - 0 1", ("a7a8q", "h7h6")),
    ),
)
def test_every_encoded_board_says_what_asking_each_square_would_have_said(
    initial_position: str,
    moves: tuple[str, ...],
) -> None:
    """The board bytes are read off bitboards; this reads them the other way.

    Castling moves two pieces, an en-passant capture removes a pawn from a
    square the move never named, and a promotion puts a piece on the board that
    was not on it before. Those are where a derivation over piece bitboards and
    one over squares in turn could part.
    """

    game = replace(_game(moves), initial_position=initial_position)

    board = chess.Board(initial_position)
    for ply in encode_game(game, legal_actions=False):
        assert ply.board.piece_ids == bytes(
            _piece_id_by_square(board.piece_at(square)) for square in chess.SQUARES
        )
        board.push(chess.Move.from_uci(moves[ply.ply_index]))


def test_aligns_decision_ratings_and_pre_move_clocks_without_fake_values() -> None:
    game = GameEncodingInput(
        game_id=42,
        ruleset="standard",
        initial_position=chess.STARTING_FEN,
        action_ids=_action_ids(("e2e4", "e7e5", "g1f3")),
        white_normalized_rating=1350,
        black_normalized_rating=None,
        time_initial_ms=PRESENT_60_SECONDS,
        time_increment_ms=0,
        clock_remaining_ms=(58_000, None, 55_000),
    )

    first, second, third = encode_game(game)

    assert first.target_rating == 1350
    assert second.target_rating is None
    assert third.target_rating == 1350
    assert first.player_clock_ms == PRESENT_60_SECONDS
    assert first.target_clock_after_move_ms == 58_000

    assert second.player_clock_ms == PRESENT_60_SECONDS
    assert second.opponent_clock_ms == 58_000

    assert third.player_clock_ms == 58_000
    assert third.opponent_clock_ms is None
    assert third.time_increment_ms == 0


def test_untimed_game_keeps_timing_explicitly_unavailable() -> None:
    plies = encode_game(
        _game(
            ("e2e4", "e7e5"),
            time_initial=None,
            clocks=(None, None),
        )
    )

    for ply in plies:
        assert ply.time_initial_ms is None
        assert ply.time_increment_ms is None
        assert ply.player_clock_ms is None
        assert ply.opponent_clock_ms is None
        assert ply.target_clock_after_move_ms is None


def test_identity_and_records_are_stable_and_json_serializable() -> None:
    identity = encoding_identity()
    records = [ply.as_record() for ply in encode_game(_game(("e2e4",)))]

    assert identity == {
        "name": "anthro-per-ply",
        "version": 5,
        "schema_sha256": (
            "dfc86ad727571972294e1582c2a9cb4b47e80eb150a98988bcea833a062f80d4"
        ),
        "board_square_count": 64,
        "action_vocabulary": {
            "name": "anthro-standard-actions",
            "version": 2,
            "size": 1970,
            "sha256": (
                "4335db3058aad53dc1a4a873a21cc48e2db4d3e6aff003887260e831d3f677ab"
            ),
        },
    }
    assert json.loads(json.dumps(records)) == records


def test_rejects_invalid_optional_values() -> None:
    game = _game(("e2e4",))
    with pytest.raises(ValueError, match="white_normalized_rating must be"):
        replace(game, white_normalized_rating=-1)
    with pytest.raises(ValueError, match="black_normalized_rating must be"):
        replace(game, black_normalized_rating=True)
    with pytest.raises(ValueError, match="time_initial_ms must be"):
        replace(game, time_initial_ms=-1)
    with pytest.raises(ValueError, match="time_increment_ms must be"):
        replace(game, time_increment_ms=True)


def test_rejects_illegal_or_misaligned_game_sequences() -> None:
    illegal = _game(("e2e4", "e2e3"))
    with pytest.raises(EncodingError, match="ply 1 is illegal"):
        encode_game(illegal)

    with pytest.raises(ValueError, match="align one-to-one"):
        GameEncodingInput(
            game_id=1,
            ruleset="standard",
            initial_position=chess.STARTING_FEN,
            action_ids=_action_ids(("e2e4",)),
            white_normalized_rating=None,
            black_normalized_rating=None,
            time_initial_ms=None,
            time_increment_ms=None,
            clock_remaining_ms=(),
        )


def test_rejects_invalid_initial_positions_and_misplaced_terminal_actions() -> None:
    invalid_position = replace(_game(("e2e4",)), initial_position="not a position")
    with pytest.raises(EncodingError, match="invalid initial position"):
        encode_game(invalid_position)

    mid_game_terminal = replace(
        _game(("e2e4", "e7e5")),
        action_ids=(RESIGNATION_ACTION_ID, *_action_ids(("e2e4",))),
    )
    with pytest.raises(EncodingError, match="terminal action at ply 0"):
        encode_game(mid_game_terminal)


def test_every_step_enables_the_terminal_actions_the_rules_allow() -> None:
    """The enabled set is what the player could choose, not only what they did."""

    plies = encode_game(_game(("e2e4", "e7e5")))

    for ply in plies:
        assert RESIGNATION_ACTION_ID in ply.enabled_actions()
        assert DRAW_CLAIM_ACTION_ID not in ply.enabled_actions()
        assert ply.target_action_id in ply.enabled_actions()


def test_a_trailing_resignation_is_a_step_that_moves_nothing() -> None:
    moves = ("e2e4", "e7e5", "g1f3")
    game = replace(
        _game(moves),
        action_ids=(*_action_ids(moves), RESIGNATION_ACTION_ID),
        clock_remaining_ms=(59_000, 59_000, 59_000, None),
    )

    plies = encode_game(game)

    assert len(plies) == len(moves) + 1
    resignation = plies[-1]
    assert resignation.ply_index == len(moves)
    assert resignation.target_action_id == RESIGNATION_ACTION_ID
    assert RESIGNATION_ACTION_ID in resignation.enabled_actions()
    assert resignation.target_clock_after_move_ms is None
    # The board is the one the last move left, read by the player who resigned.
    assert resignation.board.piece_ids[chess.F3] == chess.KNIGHT
    assert resignation.board.side_to_move == 1


def test_a_trailing_draw_claim_is_enabled_only_where_the_rules_allow_it() -> None:
    shuffle = ("g1f3", "g8f6", "f3g1", "f6g8") * 2
    claimable = replace(
        _game(shuffle),
        action_ids=(*_action_ids(shuffle), DRAW_CLAIM_ACTION_ID),
        clock_remaining_ms=(*(59_000 for _ in shuffle), None),
    )

    plies = encode_game(claimable)

    claim = plies[-1]
    assert claim.target_action_id == DRAW_CLAIM_ACTION_ID
    assert DRAW_CLAIM_ACTION_ID in claim.enabled_actions()
    # The claim becomes available exactly where the position repeats a third
    # time, which is the final step and no earlier one.
    assert [
        index
        for index, ply in enumerate(plies)
        if DRAW_CLAIM_ACTION_ID in ply.enabled_actions()
    ] == [len(shuffle)]

    unclaimable = replace(
        _game(("e2e4", "e7e5")),
        action_ids=(*_action_ids(("e2e4", "e7e5")), DRAW_CLAIM_ACTION_ID),
        clock_remaining_ms=(59_000, 59_000, None),
    )
    with pytest.raises(EncodingError, match="ply 2 is illegal"):
        encode_game(unclaimable)


def test_an_encoding_asked_for_no_legal_actions_changes_nothing_else() -> None:
    """Skipping the set has to be an omission, not a different encoding."""

    moves = ("g1f3", "g8f6", "f3g1", "f6g8") * 2
    game = replace(
        _game(moves),
        action_ids=(*_action_ids(moves), DRAW_CLAIM_ACTION_ID),
        clock_remaining_ms=(*(59_000 for _ in moves), None),
    )

    scoring = encode_game(game)
    training = encode_game(game, legal_actions=False)

    assert [replace(ply, legal_action_ids=None) for ply in scoring] == list(training)
    for absent in (training[0].enabled_actions, training[0].as_record):
        with pytest.raises(EncodingError, match="without its legal actions"):
            absent()


def test_an_action_the_position_does_not_allow_is_refused_without_the_set() -> None:
    """The target check has to survive losing the set it used to read."""

    illegal_move = _game(("e2e4", "e2e3"))
    with pytest.raises(EncodingError, match="ply 1 is illegal"):
        encode_game(illegal_move, legal_actions=False)

    unavailable_claim = replace(
        _game(("e2e4", "e7e5")),
        action_ids=(*_action_ids(("e2e4", "e7e5")), DRAW_CLAIM_ACTION_ID),
        clock_remaining_ms=(59_000, 59_000, None),
    )
    with pytest.raises(EncodingError, match="ply 2 is illegal"):
        encode_game(unavailable_claim, legal_actions=False)

    outside_vocabulary = replace(
        _game(("e2e4",)),
        action_ids=(DRAW_CLAIM_ACTION_ID + 1,),
        clock_remaining_ms=(None,),
    )
    with pytest.raises(EncodingError, match="ply 0 is illegal"):
        encode_game(outside_vocabulary, legal_actions=False)


def test_game_input_rejects_unsupported_or_empty_sequences() -> None:
    with pytest.raises(ValueError, match="standard chess only"):
        replace(_game(("e2e4",)), ruleset="chess960")
    with pytest.raises(ValueError, match="at least one action"):
        replace(_game(("e2e4",)), action_ids=(), clock_remaining_ms=())


def test_game_input_rejects_invalid_clock_values() -> None:
    with pytest.raises(ValueError, match=r"clock_remaining_ms\[0\] must be"):
        replace(_game(("e2e4",)), clock_remaining_ms=(-1,))


def test_target_free_decision_context_matches_training_history() -> None:
    moves = tuple(chess.Move.from_uci(text) for text in ("e2e4", "e7e5", "g1f3"))
    board = chess.Board()
    for move in moves:
        board.push(move)

    decision = build_decision_context(
        board,
        moves,
        target_rating=1725,
    )
    training = encode_game(_game(("e2e4", "e7e5", "g1f3")))

    assert len(decision.plies) == len(training) + 1
    for expected, actual in zip(training, decision.plies, strict=False):
        context = expected.context()
        assert actual.board == context.board
        assert actual.ply_index == context.ply_index
    assert decision.plies[-1].board.side_to_move == 1
    assert decision.plies[-1].board.piece_ids[chess.F3] == chess.KNIGHT
    assert decision.target_rating == 1725
    assert all("target_rating" not in ply.as_record() for ply in decision.plies)


def test_target_free_context_preserves_missing_rating_and_rejects_mismatch() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    board.push(move)

    context = build_decision_context(
        board,
        (move,),
        target_rating=None,
    )
    assert context.target_rating is None

    with pytest.raises(EncodingError, match="does not match"):
        build_decision_context(
            board,
            (),
            target_rating=None,
        )


def test_reused_prefix_encodes_exactly_what_a_full_rebuild_would() -> None:
    """Every update path must be indistinguishable from encoding from scratch."""

    history = DecisionHistory()
    updates = (
        (chess.STARTING_FEN, ("e2e4",), 0),
        (chess.STARTING_FEN, ("e2e4", "e7e5"), 1),
        (chess.STARTING_FEN, ("e2e4", "e7e5", "g1f3", "b8c6"), 2),
        # A takeback and a divergence both keep the moves they still share.
        (chess.STARTING_FEN, ("e2e4", "e7e5"), 2),
        (chess.STARTING_FEN, ("e2e4", "c7c5"), 1),
        # An unchanged history reuses everything and re-encodes nothing.
        (chess.STARTING_FEN, ("e2e4", "c7c5"), 2),
        # A different root shares nothing with the game it replaces.
        ("7k/5Q2/7K/8/8/8/8/8 b - - 0 1", (), 0),
    )

    for initial_fen, texts, expected_reuse in updates:
        moves = tuple(chess.Move.from_uci(text) for text in texts)

        reused = history.synchronize(initial_fen=initial_fen, moves=moves)

        assert reused == expected_reuse
        assert history.moves == moves
        assert history.initial_fen == initial_fen
        assert history.context(target_rating=1500) == build_decision_context(
            history.board,
            moves,
            target_rating=1500,
        )


def test_the_column_form_describes_the_same_timesteps_as_the_plies() -> None:
    """The two forms of a decision context are written apart and must agree."""

    # Reaches a capture, both castling rights, and an en-passant square, so
    # every column has something other than its zero value to carry.
    history = DecisionHistory(
        moves=tuple(
            chess.Move.from_uci(text)
            for text in ("e2e4", "d7d5", "e4d5", "e7e5", "g1f3", "f8e7", "f1e2")
        )
    )
    context = history.context(target_rating=1500)
    columns = context.columns
    stride = len(DecisionColumn)

    assert columns.length == len(context.plies)
    assert len(columns.piece_ids) == columns.length * BOARD_SQUARE_COUNT
    values = array("q")
    values.frombytes(columns.values)
    assert len(values) == columns.length * stride

    for index, ply in enumerate(context.plies):
        board = ply.board
        row = values[index * stride : (index + 1) * stride]
        squares = columns.piece_ids[
            index * BOARD_SQUARE_COUNT : (index + 1) * BOARD_SQUARE_COUNT
        ]
        assert squares == board.piece_ids
        assert row[DecisionColumn.PLY_INDEX] == ply.ply_index
        assert row[DecisionColumn.SIDE_TO_MOVE] == board.side_to_move
        assert row[DecisionColumn.CASTLING_RIGHTS] == board.castling_rights
        assert row[DecisionColumn.EN_PASSANT_TOKEN] == en_passant_token(
            board.en_passant_square
        )
        assert row[DecisionColumn.HALFMOVE_CLOCK] == board.halfmove_clock
        assert row[DecisionColumn.FULLMOVE_NUMBER] == board.fullmove_number
        assert row[DecisionColumn.REPETITION_COUNT] == board.repetition_count

    # The en-passant square is what a shared zero would stand in for, so the
    # case the test was built around is checked to have occurred.
    assert any(ply.board.en_passant_square is not None for ply in context.plies)


def test_a_nullable_input_travels_as_the_row_that_names_its_absence() -> None:
    """Absence is a row of the same table, so nothing reassembles it downstream.

    Square a1 is index 0, so the column would otherwise put a real value and a
    missing one at the same index and need a presence flag beside it to be read
    apart.
    """

    assert en_passant_token(None) == 0
    assert en_passant_token(chess.A1) == 1
    assert en_passant_token(chess.H8) == BOARD_SQUARE_COUNT


def test_a_repeated_position_counts_toward_the_claim() -> None:
    """A model blind to repetition cannot decide when to claim a draw.

    Shuffling both knights out and back returns the game to its own starting
    position, so the third occurrence is the one where the rules first allow a
    threefold claim and the encoding has to say so.
    """

    shuffle = ("g1f3", "g8f6", "f3g1", "f6g8")

    plies = encode_game(_game(shuffle * 2 + ("e2e4",)), legal_actions=False)

    counts = [ply.board.repetition_count for ply in plies]
    assert counts == [0, 0, 0, 0, 1, 1, 1, 1, 2]
    board = chess.Board()
    for ply in plies[:-1]:
        board.push(decode_move(ply.target_action_id))
    assert board.can_claim_threefold_repetition()


def test_a_takeback_forgets_the_repetitions_it_unwound() -> None:
    """A history rewound past a repeat is a history that has not repeated yet."""

    shuffle = [chess.Move.from_uci(move) for move in ("g1f3", "g8f6", "f3g1", "f6g8")]
    history = DecisionHistory(moves=shuffle * 2)
    assert history.context(target_rating=None).plies[-1].board.repetition_count == 2

    history.synchronize(moves=shuffle[:2])

    counts = [
        ply.board.repetition_count for ply in history.context(target_rating=None).plies
    ]
    assert counts == [0, 0, 0]


def test_a_rule_counter_larger_than_play_produces_carries_through() -> None:
    """A root FEN a caller supplied bounds nothing, so neither may the columns."""

    history = DecisionHistory(initial_fen="8/8/8/8/8/8/6k1/K7 w - - 100000 40000")
    context = history.context(target_rating=None)

    ply = context.plies[0]
    assert ply.board.halfmove_clock == 100_000
    assert ply.board.fullmove_number == 40_000
    values = array("q")
    values.frombytes(context.columns.values)
    assert values[DecisionColumn.HALFMOVE_CLOCK] == 100_000
    assert values[DecisionColumn.FULLMOVE_NUMBER] == 40_000


def test_only_the_plies_past_the_divergence_point_are_encoded() -> None:
    """The reuse count has to describe encoding actually skipped, not intent."""

    history = DecisionHistory(
        moves=tuple(chess.Move.from_uci(text) for text in ("e2e4", "e7e5", "g1f3"))
    )
    appended = tuple(
        chess.Move.from_uci(text) for text in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5")
    )
    divergent = tuple(
        chess.Move.from_uci(text) for text in ("e2e4", "e7e5", "d2d4", "e5d4")
    )

    with _counted_encodings() as counter:
        history.synchronize(moves=appended)
    assert counter.encodings == 2

    with _counted_encodings() as counter:
        history.synchronize(moves=divergent)
    assert counter.encodings == 2

    with _counted_encodings() as counter:
        history.push(chess.Move.from_uci("g1f3"))
    assert counter.encodings == 1


def test_a_rejected_history_leaves_the_encoded_prefix_untouched() -> None:
    """Validation failures must not leave a half-applied game behind."""

    moves = tuple(chess.Move.from_uci(text) for text in ("e2e4", "e7e5"))
    history = DecisionHistory(moves=moves)
    before = history.context(target_rating=None)

    illegal_append = (*moves, chess.Move.from_uci("g1f3"), chess.Move.from_uci("a1a8"))
    with pytest.raises(EncodingError, match="illegal at ply 3"):
        history.synchronize(moves=illegal_append)
    assert history.moves == moves
    assert history.context(target_rating=None) == before

    with pytest.raises(EncodingError, match="illegal at ply 1"):
        history.synchronize(moves=(chess.Move.from_uci("d2d4"), *moves))
    assert history.moves == moves
    assert history.context(target_rating=None) == before

    with pytest.raises(EncodingError, match="illegal at ply 2"):
        history.push(chess.Move.from_uci("a1a8"))
    assert history.moves == moves
    assert history.context(target_rating=None) == before

    with pytest.raises(EncodingError, match="invalid initial position"):
        history.synchronize(initial_fen="not a position", moves=())
    assert history.initial_fen == chess.STARTING_FEN
    assert history.context(target_rating=None) == before

    with pytest.raises(TypeError, match="move at ply 0 must be"):
        history.synchronize(moves=("e2e4",))  # type: ignore[arg-type]
    assert history.moves == moves


class _EncodingCounter:
    """Count board encodings performed inside one block."""

    def __init__(self) -> None:
        self.encodings = 0


@contextmanager
def _counted_encodings() -> Iterator[_EncodingCounter]:
    counter = _EncodingCounter()
    original = encoding_module._context_for_position

    def counted(**kwargs: object) -> object:
        counter.encodings += 1
        return original(**kwargs)  # type: ignore[arg-type]

    encoding_module._context_for_position = counted  # type: ignore[assignment]
    try:
        yield counter
    finally:
        encoding_module._context_for_position = original


def _game(
    moves: tuple[str, ...],
    *,
    time_initial: int | None = PRESENT_60_SECONDS,
    clocks: tuple[int | None, ...] | None = None,
) -> GameEncodingInput:
    return GameEncodingInput(
        game_id=7,
        ruleset="standard",
        initial_position=chess.STARTING_FEN,
        action_ids=_action_ids(moves),
        white_normalized_rating=None,
        black_normalized_rating=None,
        time_initial_ms=time_initial,
        time_increment_ms=1_000 if time_initial is not None else None,
        clock_remaining_ms=(
            clocks if clocks is not None else tuple(59_000 for _ in moves)
        ),
    )


def _action_ids(moves: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(encode_move(chess.Move.from_uci(move)) for move in moves)


def _piece_id_by_square(piece: chess.Piece | None) -> int:
    if piece is None:
        return 0
    return piece.piece_type + (0 if piece.color == chess.WHITE else 6)
