import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import chess
import pytest

from anthro_chess.chess import RESIGNATION_ACTION_ID, encode_move
from anthro_chess.data import (
    BOARD_SQUARE_COUNT,
    DecisionHistory,
    EncodingError,
    GameEncodingInput,
    build_decision_context,
    encode_game,
    encoding_identity,
)
from anthro_chess.data import encoding as encoding_module

PRESENT_60_SECONDS = 60_000


def test_encodes_exact_positions_previous_moves_and_legal_targets() -> None:
    game = _game(("e2e4", "e7e5", "g1f3"))

    plies = encode_game(game)

    assert len(plies) == 3
    first, second, third = plies
    assert first.ply_index == 0
    assert first.previous_action_id is None
    assert first.board.side_to_move == 0
    assert first.board.fullmove_number == 1
    assert first.board.castling_rights == 15
    assert len(first.board.piece_ids) == BOARD_SQUARE_COUNT
    assert first.board.piece_ids[chess.E2] == chess.PAWN
    assert first.target_action_id in first.legal_action_ids

    assert second.previous_action_id == first.target_action_id
    assert second.board.side_to_move == 1
    assert second.board.en_passant_square == chess.E3
    assert second.board.piece_ids[chess.E4] == chess.PAWN

    assert third.previous_action_id == second.target_action_id
    assert third.board.fullmove_number == 2
    assert third.board.piece_ids[chess.E5] == chess.PAWN + 6


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
        "version": 3,
        "schema_sha256": (
            "a8170a23da62322ba50886c1b2d32b7a5a2c0e4d6dee2629641a823f59cc033e"
        ),
        "board_square_count": 64,
        "action_vocabulary": {
            "name": "anthro-standard-actions",
            "version": 1,
            "size": 1969,
            "sha256": (
                "f95e6069227ad773de35c12f9601d89b622da0539d7793ff88232aff368a48d6"
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


def test_rejects_invalid_initial_positions_and_non_move_actions() -> None:
    invalid_position = replace(_game(("e2e4",)), initial_position="not a position")
    with pytest.raises(EncodingError, match="invalid initial position"):
        encode_game(invalid_position)

    non_move = replace(
        _game(("e2e4",)),
        action_ids=(RESIGNATION_ACTION_ID,),
    )
    with pytest.raises(EncodingError, match="is not a board move"):
        encode_game(non_move)


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
        assert actual.previous_action_id == context.previous_action_id
    assert decision.plies[-1].board.side_to_move == 1
    assert decision.plies[-1].previous_action_id == training[-1].target_action_id
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
