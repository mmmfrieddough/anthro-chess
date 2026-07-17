import json
from dataclasses import replace

import chess
import pytest

from anthro_chess.chess import RESIGNATION_ACTION_ID, encode_move
from anthro_chess.data import (
    BOARD_SQUARE_COUNT,
    EncodingError,
    GameEncodingInput,
    OptionalInteger,
    encode_game,
    encoding_identity,
)

PRESENT_60_SECONDS = OptionalInteger(60_000, "present")
UNAVAILABLE = OptionalInteger(None, "unavailable")


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


def test_aligns_ratings_color_and_pre_move_clocks_without_fake_values() -> None:
    game = GameEncodingInput(
        game_id=42,
        ruleset="standard",
        initial_position=chess.STARTING_FEN,
        action_ids=_action_ids(("e2e4", "e7e5", "g1f3")),
        white_normalized_rating=OptionalInteger(1350, "present"),
        black_normalized_rating=OptionalInteger(None, "rejected"),
        time_initial_ms=PRESENT_60_SECONDS,
        time_increment_ms=OptionalInteger(0, "present"),
        clock_remaining_ms=(
            OptionalInteger(58_000, "present"),
            OptionalInteger(None, "rejected"),
            OptionalInteger(55_000, "present"),
        ),
    )

    first, second, third = encode_game(game)

    assert first.player_rating == OptionalInteger(1350, "present")
    assert first.opponent_rating == OptionalInteger(None, "rejected")
    assert first.player_clock_ms == PRESENT_60_SECONDS
    assert first.target_clock_after_move_ms == OptionalInteger(58_000, "present")

    assert second.player_rating == OptionalInteger(None, "rejected")
    assert second.opponent_rating == OptionalInteger(1350, "present")
    assert second.player_clock_ms == PRESENT_60_SECONDS
    assert second.opponent_clock_ms == OptionalInteger(58_000, "present")

    assert third.player_clock_ms == OptionalInteger(58_000, "present")
    assert third.opponent_clock_ms == OptionalInteger(None, "rejected")
    assert third.time_increment_ms == OptionalInteger(0, "present")


def test_untimed_game_keeps_timing_explicitly_unavailable() -> None:
    plies = encode_game(
        _game(
            ("e2e4", "e7e5"),
            time_initial=UNAVAILABLE,
            clocks=(UNAVAILABLE, UNAVAILABLE),
        )
    )

    for ply in plies:
        assert ply.time_initial_ms == UNAVAILABLE
        assert ply.time_increment_ms == UNAVAILABLE
        assert ply.player_clock_ms == UNAVAILABLE
        assert ply.opponent_clock_ms == UNAVAILABLE
        assert ply.target_clock_after_move_ms == UNAVAILABLE


def test_identity_and_records_are_stable_and_json_serializable() -> None:
    identity = encoding_identity()
    records = [ply.as_record() for ply in encode_game(_game(("e2e4",)))]

    assert identity == {
        "name": "anthro-per-ply",
        "version": 1,
        "schema_sha256": (
            "3a0f9de9dde89886fb6b394ddaea8e82a98b53990935433139428b41cedd1d08"
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


@pytest.mark.parametrize(
    "value",
    [
        OptionalInteger(None, "unavailable"),
        OptionalInteger(None, "rejected"),
        OptionalInteger(0, "present"),
    ],
)
def test_optional_integer_accepts_explicit_valid_states(value: OptionalInteger) -> None:
    assert value.as_record()["status"] in {"present", "unavailable", "rejected"}


@pytest.mark.parametrize(
    ("value", "status"),
    [
        (None, "present"),
        (-1, "present"),
        (0, "unavailable"),
        (0, "rejected"),
    ],
)
def test_optional_integer_rejects_fake_or_misaligned_values(
    value: int | None, status: str
) -> None:
    with pytest.raises(ValueError):
        OptionalInteger(value, status)  # type: ignore[arg-type]


def test_optional_integer_rejects_an_unknown_status() -> None:
    with pytest.raises(ValueError, match="unknown field status"):
        OptionalInteger(None, "unknown")  # type: ignore[arg-type]


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
            white_normalized_rating=UNAVAILABLE,
            black_normalized_rating=UNAVAILABLE,
            time_initial_ms=UNAVAILABLE,
            time_increment_ms=UNAVAILABLE,
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


def _game(
    moves: tuple[str, ...],
    *,
    time_initial: OptionalInteger = PRESENT_60_SECONDS,
    clocks: tuple[OptionalInteger, ...] | None = None,
) -> GameEncodingInput:
    return GameEncodingInput(
        game_id=7,
        ruleset="standard",
        initial_position=chess.STARTING_FEN,
        action_ids=_action_ids(moves),
        white_normalized_rating=UNAVAILABLE,
        black_normalized_rating=UNAVAILABLE,
        time_initial_ms=time_initial,
        time_increment_ms=(
            OptionalInteger(1_000, "present")
            if time_initial.status == "present"
            else UNAVAILABLE
        ),
        clock_remaining_ms=clocks
        or tuple(OptionalInteger(59_000, "present") for _ in moves),
    )


def _action_ids(moves: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(encode_move(chess.Move.from_uci(move)) for move in moves)
