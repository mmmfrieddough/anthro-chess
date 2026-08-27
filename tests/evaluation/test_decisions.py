from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chess
import pytest
import torch

from anthro_chess.chess import ACTION_VOCABULARY_SIZE, encode_move
from anthro_chess.data import DecisionContext
from anthro_chess.evaluation.decisions import (
    DECISION_DECOMPOSITION_VERSION,
    DecisionDecompositionError,
    DecisionSample,
    DecisionSet,
    DecisionSetting,
    PlayedDecision,
    collect_decisions,
    decompose_game_records,
    score_played_decisions,
    summarize_decisions,
)
from anthro_chess.evaluation.games import (
    GAME_RECORD_VERSION,
    DecisionPolicy,
    DecisionRecord,
    GameOutcome,
    GameRecord,
    GameTermination,
    SeatRecord,
    build_game_record,
)
from anthro_chess.evaluation.results import DataComponent
from anthro_chess.evaluation.results.metrics import (
    MOVE_PREDICTION_PROJECTION,
)
from anthro_chess.runtime import RuntimeConfig

PLAYED = ("e2e4", "e7e5", "g1f3")


@dataclass
class RankedRunner:
    """A policy that prefers one action and orders the rest by action id.

    Deterministic and independent of position, so an expected rank and an
    expected probability can be written down rather than approximated.
    """

    preferred_action_id: int
    calls: int = 0

    def predict(self, _context: DecisionContext) -> torch.Tensor:
        self.calls += 1
        logits = torch.zeros(ACTION_VOCABULARY_SIZE)
        logits[self.preferred_action_id] = 5.0
        return logits


def _actions(*uci: str) -> tuple[int, ...]:
    return tuple(encode_move(chess.Move.from_uci(value)) for value in uci)


def _seat(
    *,
    label: str = "model-a",
    target_rating: int | None = 1500,
    temperature: float = 1.0,
    kind: str = "model",
    configured: bool = True,
) -> SeatRecord:
    configuration = (
        {
            "target_rating": target_rating,
            "temperature": temperature,
            "resignation_enabled": False,
        }
        if configured
        else {}
    )
    return SeatRecord(
        kind=kind,
        label=label,
        seed=7,
        configuration=configuration,
    )


def _policy(
    *,
    selected_probability: float,
    selected_rank: int,
    preferred_action_id: int,
    preferred_probability: float,
    enabled_action_count: int = 20,
) -> DecisionPolicy:
    return DecisionPolicy(
        enabled_action_count=enabled_action_count,
        selected_probability=selected_probability,
        selected_rank=selected_rank,
        preferred_action_id=preferred_action_id,
        preferred_probability=preferred_probability,
    )


def _record(
    *,
    policies: tuple[DecisionPolicy | None, ...],
    white: SeatRecord | None = None,
    black: SeatRecord | None = None,
    seed: int = 11,
) -> GameRecord:
    action_ids = _actions(*PLAYED)
    decisions = [
        DecisionRecord(
            ply_index=index,
            slot="white" if index % 2 == 0 else "black",
            action_id=action_id,
            policy=policy,
        )
        for index, (action_id, policy) in enumerate(
            zip(action_ids, policies, strict=True)
        )
    ]
    return build_game_record(
        initial_position=chess.STARTING_FEN,
        prefix_plies=0,
        action_ids=action_ids,
        white=white or _seat(),
        black=black or _seat(label="model-b"),
        seed=seed,
        decisions=decisions,
        outcome=GameOutcome(
            result="*",
            termination=GameTermination.PLY_LIMIT,
            adjudicated=True,
        ),
    )


def _sample(
    *,
    selected_rank: int,
    selected_probability: float,
    preferred_probability: float,
    setting: DecisionSetting | None = None,
    ply_index: int = 0,
) -> DecisionSample:
    return DecisionSample(
        game_id="game",
        ply_index=ply_index,
        slot="white",
        action_id=100,
        setting=setting or DecisionSetting(target_rating=1500, temperature=1.0),
        enabled_action_count=20,
        selected_probability=selected_probability,
        selected_rank=selected_rank,
        preferred_action_id=100 if selected_rank == 1 else 101,
        preferred_probability=preferred_probability,
    )


def _data_component() -> DataComponent:
    return DataComponent(
        projection=MOVE_PREDICTION_PROJECTION.name,
        projection_version=MOVE_PREDICTION_PROJECTION.version,
        content_sha256="a" * 64,
        games=3,
    )


def test_collected_decisions_carry_the_dials_their_seat_recorded() -> None:
    record = _record(
        policies=(
            _policy(
                selected_probability=0.5,
                selected_rank=1,
                preferred_action_id=_actions("e2e4")[0],
                preferred_probability=0.5,
            ),
            _policy(
                selected_probability=0.1,
                selected_rank=3,
                preferred_action_id=999,
                preferred_probability=0.4,
            ),
            _policy(
                selected_probability=0.2,
                selected_rank=2,
                preferred_action_id=999,
                preferred_probability=0.3,
            ),
        ),
        white=_seat(target_rating=1200, temperature=0.5),
        black=_seat(label="model-b", target_rating=2000, temperature=1.5),
    )

    collected = collect_decisions([record])

    assert [sample.setting for sample in collected.samples] == [
        DecisionSetting(target_rating=1200, temperature=0.5),
        DecisionSetting(target_rating=2000, temperature=1.5),
        DecisionSetting(target_rating=1200, temperature=0.5),
    ]
    assert [sample.ply_index for sample in collected.samples] == [0, 1, 2]
    assert [sample.slot for sample in collected.samples] == ["white", "black", "white"]
    assert collected.unscored_decisions == 0


def test_a_seat_with_no_policy_is_counted_rather_than_dropped() -> None:
    """A random or engine opponent reports no distribution to decompose."""

    record = _record(
        policies=(
            _policy(
                selected_probability=0.5,
                selected_rank=1,
                preferred_action_id=_actions("e2e4")[0],
                preferred_probability=0.5,
            ),
            None,
            _policy(
                selected_probability=0.2,
                selected_rank=2,
                preferred_action_id=999,
                preferred_probability=0.3,
            ),
        ),
        black=_seat(label="random", kind="random", configured=False),
    )

    collected = collect_decisions([record])
    decomposition = summarize_decisions(collected)

    assert len(collected.samples) == 2
    assert collected.unscored_decisions == 1
    assert decomposition.unscored_decisions == 1
    assert decomposition.overall.decisions == 2


def test_a_tie_at_the_top_counts_as_following_the_policy() -> None:
    """Equal logits are no preference, so a draw between them overrode nothing."""

    tied = _sample(
        selected_rank=1,
        selected_probability=0.25,
        preferred_probability=0.25,
    )

    assert tied.followed_policy
    assert tied.policy_regret == 0.0


def test_aggregation_reports_both_error_classes_and_their_regret() -> None:
    samples = (
        _sample(selected_rank=1, selected_probability=0.6, preferred_probability=0.6),
        _sample(selected_rank=2, selected_probability=0.3, preferred_probability=0.4),
        _sample(selected_rank=7, selected_probability=0.02, preferred_probability=0.52),
        _sample(selected_rank=1, selected_probability=0.4, preferred_probability=0.4),
    )

    cell = summarize_decisions(DecisionSet(samples=samples)).overall

    assert cell.decisions == 4
    assert cell.followed_policy == 2
    assert cell.departures == 2
    assert cell.preferred_selection_rate == 0.5
    assert cell.policy_regret == pytest.approx((0.0 + 0.1 + 0.5 + 0.0) / 4)
    assert cell.departure_policy_regret == pytest.approx((0.1 + 0.5) / 2)
    assert cell.maximum_policy_regret == pytest.approx(0.5)
    assert cell.selected_rank == pytest.approx((1 + 2 + 7 + 1) / 4)
    assert cell.preferred_probability == pytest.approx((0.6 + 0.4 + 0.52 + 0.4) / 4)
    assert cell.selected_probability == pytest.approx((0.6 + 0.3 + 0.02 + 0.4) / 4)


def test_the_pooled_regret_hides_what_the_departure_regret_shows() -> None:
    """Two passes agree pooled and disagree on what sampling actually cost."""

    near_ties = tuple(
        _sample(
            selected_rank=2,
            selected_probability=0.30,
            preferred_probability=0.35,
            ply_index=index,
        )
        for index in range(10)
    )
    one_disaster = (
        _sample(selected_rank=9, selected_probability=0.0, preferred_probability=0.5),
        *(
            _sample(
                selected_rank=1,
                selected_probability=0.5,
                preferred_probability=0.5,
                ply_index=index,
            )
            for index in range(9)
        ),
    )

    ties = summarize_decisions(DecisionSet(samples=near_ties)).overall
    disaster = summarize_decisions(DecisionSet(samples=one_disaster)).overall

    assert ties.policy_regret == pytest.approx(disaster.policy_regret)
    assert ties.departure_policy_regret == pytest.approx(0.05)
    assert disaster.departure_policy_regret == pytest.approx(0.5)


def test_cells_split_by_setting_in_a_stable_order() -> None:
    cold = DecisionSetting(target_rating=1200, temperature=0.0)
    warm = DecisionSetting(target_rating=1200, temperature=1.0)
    strong = DecisionSetting(target_rating=2000, temperature=1.0)
    unset = DecisionSetting(target_rating=None, temperature=1.0)
    samples = tuple(
        _sample(
            selected_rank=1,
            selected_probability=0.5,
            preferred_probability=0.5,
            setting=setting,
            ply_index=index,
        )
        for index, setting in enumerate((strong, unset, warm, cold))
    )

    decomposition = summarize_decisions(DecisionSet(samples=samples))

    assert decomposition.settings == (cold, warm, strong, unset)
    assert decomposition.cell(warm).decisions == 1
    assert decomposition.overall.setting is None
    assert decomposition.overall.decisions == 4


def test_a_pass_that_varied_the_dials_has_no_single_reference_cell() -> None:
    """Its pooled reading would move with grid composition, not with the model."""

    samples = tuple(
        _sample(
            selected_rank=1,
            selected_probability=0.5,
            preferred_probability=0.5,
            setting=DecisionSetting(target_rating=1500, temperature=temperature),
            ply_index=index,
        )
        for index, temperature in enumerate((0.5, 1.0))
    )

    decomposition = summarize_decisions(DecisionSet(samples=samples))

    with pytest.raises(DecisionDecompositionError, match="no single reference cell"):
        decomposition.reference_cell()


def test_one_setting_reports_a_reference_cell() -> None:
    setting = DecisionSetting(target_rating=1500, temperature=1.0)
    samples = (
        _sample(
            selected_rank=1,
            selected_probability=0.5,
            preferred_probability=0.5,
            setting=setting,
        ),
    )

    decomposition = summarize_decisions(DecisionSet(samples=samples))

    assert decomposition.reference_cell().setting == setting


def test_a_decomposition_needs_at_least_one_classifiable_decision() -> None:
    with pytest.raises(DecisionDecompositionError, match="at least one"):
        summarize_decisions(DecisionSet(samples=()))


def test_the_record_is_deterministic_and_carries_every_decision() -> None:
    samples = (
        _sample(selected_rank=1, selected_probability=0.6, preferred_probability=0.6),
        _sample(
            selected_rank=3,
            selected_probability=0.1,
            preferred_probability=0.4,
            ply_index=1,
        ),
    )

    record = summarize_decisions(DecisionSet(samples=samples)).as_record()

    assert record["version"] == DECISION_DECOMPOSITION_VERSION
    assert json.dumps(record, sort_keys=True) == json.dumps(
        summarize_decisions(DecisionSet(samples=samples)).as_record(),
        sort_keys=True,
    )
    assert [entry["policy_regret"] for entry in record["samples"]] == [
        pytest.approx(0.0),
        pytest.approx(0.3),
    ]
    assert [entry["selected_rank"] for entry in record["samples"]] == [1, 3]


def test_a_stored_payload_decomposes_without_loading_a_checkpoint(
    tmp_path: Path,
) -> None:
    record = _record(
        policies=(
            _policy(
                selected_probability=0.5,
                selected_rank=1,
                preferred_action_id=_actions("e2e4")[0],
                preferred_probability=0.5,
            ),
            _policy(
                selected_probability=0.1,
                selected_rank=5,
                preferred_action_id=999,
                preferred_probability=0.6,
            ),
            _policy(
                selected_probability=0.3,
                selected_rank=1,
                preferred_action_id=_actions("g1f3")[0],
                preferred_probability=0.3,
            ),
        )
    )
    path = tmp_path / "games.json"
    path.write_text(
        json.dumps(
            {"version": GAME_RECORD_VERSION, "games": [record.as_record()]},
        ),
        encoding="utf-8",
    )

    decomposition = decompose_game_records(path)

    assert decomposition.overall.decisions == 3
    assert decomposition.overall.departures == 1
    assert decomposition.overall.departure_policy_regret == pytest.approx(0.5)


def test_an_unreadable_payload_reports_where_it_failed(tmp_path: Path) -> None:
    path = tmp_path / "games.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(DecisionDecompositionError, match="cannot read game records"):
        decompose_game_records(path)


def test_played_decisions_are_rescored_through_the_selection_path() -> None:
    preferred = encode_move(chess.Move.from_uci("g1f3"))
    runner = RankedRunner(preferred_action_id=preferred)
    played = PlayedDecision(
        game_id="session:0",
        ply_index=0,
        initial_fen=chess.STARTING_FEN,
        history=(),
        action_id=encode_move(chess.Move.from_uci("e2e4")),
        config=RuntimeConfig(target_rating=1500, temperature=0.7, seed=3),
    )

    collected = score_played_decisions(runner, [played])
    sample = collected.samples[0]

    assert sample.slot == "white"
    assert sample.preferred_action_id == preferred
    assert sample.selected_rank == 2
    assert sample.enabled_action_count == 20
    assert sample.setting == DecisionSetting(target_rating=1500, temperature=0.7)
    assert sample.policy_regret == pytest.approx(
        sample.preferred_probability - sample.selected_probability
    )
    assert runner.calls == 1


def test_rescoring_a_whole_game_reuses_one_session_per_setting() -> None:
    runner = RankedRunner(preferred_action_id=encode_move(chess.Move.from_uci("g1f3")))
    config = RuntimeConfig(target_rating=1500, temperature=1.0, seed=5)
    board = chess.Board()
    decisions = []
    for ply_index, uci in enumerate(PLAYED):
        move = chess.Move.from_uci(uci)
        decisions.append(
            PlayedDecision(
                game_id="session:0",
                ply_index=ply_index,
                initial_fen=chess.STARTING_FEN,
                history=tuple(board.move_stack),
                action_id=encode_move(move),
                config=config,
            )
        )
        board.push(move)

    collected = score_played_decisions(runner, decisions)

    assert [sample.slot for sample in collected.samples] == [
        "white",
        "black",
        "white",
    ]
    assert [sample.ply_index for sample in collected.samples] == [0, 1, 2]
    assert collected.samples[2].selected_rank == 1
    assert collected.unscored_decisions == 0


def test_rescoring_rejects_an_action_the_position_never_enabled() -> None:
    runner = RankedRunner(preferred_action_id=encode_move(chess.Move.from_uci("g1f3")))
    played = PlayedDecision(
        game_id="session:0",
        ply_index=0,
        initial_fen=chess.STARTING_FEN,
        history=(),
        action_id=encode_move(chess.Move.from_uci("e7e5")),
        config=RuntimeConfig(seed=1),
    )

    with pytest.raises(DecisionDecompositionError, match="ply 0"):
        score_played_decisions(runner, [played])
