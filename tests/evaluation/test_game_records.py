from __future__ import annotations

import chess
import pytest
from pydantic import ValidationError

from anthro_chess.chess import RESIGNATION_ACTION_ID, encode_move
from anthro_chess.evaluation.games import (
    GAME_RECORD_VERSION,
    DecisionPolicy,
    DecisionRecord,
    GameOutcome,
    GameRecord,
    GameRecordError,
    GameTermination,
    SeatRecord,
    build_game_record,
    parse_game_records,
    read_game_records,
    termination_from_outcome,
    write_game_records,
)
from anthro_chess.evaluation.results import DetailStore
from anthro_chess.runtime import SelectionPolicy


def _seat(label: str = "model-a") -> SeatRecord:
    return SeatRecord(kind="model", label=label, seed=7, configuration={"t": 1.0})


def _actions(*uci: str) -> tuple[int, ...]:
    return tuple(encode_move(chess.Move.from_uci(value)) for value in uci)


def _decisions(action_ids: tuple[int, ...], *, start: int = 0) -> list[DecisionRecord]:
    return [
        DecisionRecord(
            ply_index=start + offset,
            slot="white" if (start + offset) % 2 == 0 else "black",
            action_id=action_id,
        )
        for offset, action_id in enumerate(action_ids)
    ]


def _record(**overrides: object) -> GameRecord:
    action_ids = _actions("e2e4", "e7e5", "g1f3")
    defaults: dict[str, object] = {
        "initial_position": chess.STARTING_FEN,
        "prefix_plies": 0,
        "action_ids": action_ids,
        "white": _seat(),
        "black": _seat("model-b"),
        "seed": 11,
        "decisions": _decisions(action_ids),
        "outcome": GameOutcome(
            result="*",
            termination=GameTermination.PLY_LIMIT,
            adjudicated=True,
        ),
    }
    defaults.update(overrides)
    return build_game_record(**defaults)  # type: ignore[arg-type]


def test_a_record_derives_its_identity_from_its_own_content() -> None:
    first = _record()
    second = _record()

    assert first.game_id == second.game_id
    assert first.record_version == GAME_RECORD_VERSION
    assert _record(seed=12).game_id != first.game_id


def test_a_record_reconstructs_its_final_position_from_the_moves() -> None:
    record = _record()

    board = record.board()

    assert board.fen() == (
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    )
    assert record.ply_count == 3
    assert record.generated_plies == 3


def test_a_prefix_leaves_only_the_continuation_as_decisions() -> None:
    action_ids = _actions("e2e4", "e7e5", "g1f3", "b8c6")

    record = _record(
        action_ids=action_ids,
        prefix_plies=2,
        decisions=_decisions(action_ids[2:], start=2),
        source_game_id=4242,
        position_label="prefix-2",
    )

    assert record.ply_count == 4
    assert record.generated_plies == 2
    assert record.source_game_id == 4242


def test_a_record_rejects_a_decision_that_disagrees_with_the_history() -> None:
    action_ids = _actions("e2e4", "e7e5")
    decisions = _decisions(action_ids)
    decisions[1] = decisions[1].model_copy(
        update={"action_id": encode_move(chess.Move.from_uci("d7d5"))}
    )

    with pytest.raises(ValidationError, match="disagrees with the recorded action"):
        _record(action_ids=action_ids, decisions=decisions)


def test_a_record_rejects_a_decision_attributed_to_the_wrong_seat() -> None:
    action_ids = _actions("e2e4", "e7e5")
    decisions = _decisions(action_ids)
    decisions[1] = decisions[1].model_copy(update={"slot": "white"})

    with pytest.raises(ValidationError, match="names the wrong seat"):
        _record(action_ids=action_ids, decisions=decisions)


def test_a_record_rejects_an_illegal_move_sequence() -> None:
    action_ids = _actions("e2e4", "e7e5", "e4e5")

    with pytest.raises(ValidationError, match="is illegal in the position"):
        _record(action_ids=action_ids, decisions=_decisions(action_ids))


def test_a_record_rooted_at_a_non_standard_position_stays_valid() -> None:
    fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
    action_ids = _actions("d8h4")

    record = _record(
        initial_position=fen,
        action_ids=action_ids,
        decisions=[DecisionRecord(ply_index=0, slot="black", action_id=action_ids[0])],
        outcome=GameOutcome(
            result="0-1",
            termination=GameTermination.CHECKMATE,
            adjudicated=False,
        ),
    )

    assert record.board().is_checkmate()
    assert record.decisions[0].slot == "black"


def test_a_resignation_is_recorded_as_the_final_action_only() -> None:
    action_ids = (*_actions("e2e4", "e7e5"), RESIGNATION_ACTION_ID)

    record = _record(
        action_ids=action_ids,
        decisions=_decisions(action_ids),
        outcome=GameOutcome(
            result="1-0",
            termination=GameTermination.RESIGNATION,
            adjudicated=False,
        ),
    )

    assert record.ply_count == 2
    assert record.action_ids[-1] == RESIGNATION_ACTION_ID

    with pytest.raises(ValidationError, match="final action"):
        _record(
            action_ids=(RESIGNATION_ACTION_ID, *_actions("e2e4")),
            decisions=_decisions((RESIGNATION_ACTION_ID, *_actions("e2e4"))),
        )


def test_an_unfinished_game_is_only_valid_as_an_adjudicated_ply_limit() -> None:
    with pytest.raises(ValidationError, match="must report a decided result"):
        GameOutcome(
            result="*",
            termination=GameTermination.CHECKMATE,
            adjudicated=False,
        )
    with pytest.raises(ValidationError, match="always adjudicated"):
        GameOutcome(
            result="*",
            termination=GameTermination.PLY_LIMIT,
            adjudicated=False,
        )


def test_a_claimed_draw_is_distinguished_from_an_automatic_one() -> None:
    assert GameTermination.THREEFOLD_REPETITION.claimed
    assert GameTermination.FIFTY_MOVES.claimed
    assert not GameTermination.FIVEFOLD_REPETITION.claimed
    assert not GameTermination.CHECKMATE.claimed


def test_terminations_map_from_the_chess_layer() -> None:
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push(chess.Move.from_uci(uci))
    outcome = board.outcome()

    assert outcome is not None
    assert termination_from_outcome(outcome) is GameTermination.CHECKMATE


def test_a_decision_policy_carries_the_runtime_selection_unchanged() -> None:
    policy = DecisionPolicy.from_selection(
        SelectionPolicy(
            enabled_action_count=20,
            selected_probability=0.25,
            selected_rank=2,
            preferred_action_id=5,
            preferred_probability=0.5,
        )
    )

    assert policy.selected_rank == 2
    assert policy.preferred_probability == 0.5

    with pytest.raises(ValidationError, match="likelier than the preferred one"):
        DecisionPolicy(
            enabled_action_count=20,
            selected_probability=0.75,
            selected_rank=1,
            preferred_action_id=5,
            preferred_probability=0.5,
        )


def test_games_round_trip_through_the_machine_local_detail_tier(
    tmp_path: object,
) -> None:
    store = DetailStore(tmp_path)  # type: ignore[arg-type]
    records = (_record(), _record(seed=99))

    reference = write_game_records(
        store,
        "rollout/games.json",
        records,
        description="fixture games",
    )
    loaded = read_game_records(store, reference)

    assert loaded == records
    assert reference.description == "fixture games"


def test_a_payload_from_a_future_record_version_is_rejected() -> None:
    with pytest.raises(GameRecordError, match="this build understands"):
        parse_game_records({"version": GAME_RECORD_VERSION + 1, "games": []})
    with pytest.raises(GameRecordError, match="must be an object"):
        parse_game_records([])
    with pytest.raises(GameRecordError, match="list of games"):
        parse_game_records({"version": GAME_RECORD_VERSION})
