from __future__ import annotations

from collections.abc import Sequence

import chess
import pytest

from anthro_chess.chess import draw_claim_available, encode_move
from anthro_chess.evaluation.games import (
    GAME_ANALYSIS_VERSION,
    DecisionRecord,
    GameOutcome,
    GameRecord,
    GameTermination,
    SeatRecord,
    analyze_game,
    analyze_games,
    build_game_record,
    summarize_games,
)
from anthro_chess.evaluation.openings import UNCLASSIFIED, OpeningLevel

#: A shuffle that returns both sides to the starting position twice, so the
#: third occurrence makes a threefold draw claimable at a known ply.
SHUFFLE = (
    "g1f3",
    "g8f6",
    "f3g1",
    "f6g8",
    "g1f3",
    "g8f6",
    "f3g1",
    "f6g8",
)


def _record(
    moves: Sequence[str],
    *,
    result: str = "*",
    termination: GameTermination = GameTermination.PLY_LIMIT,
    adjudicated: bool = True,
    prefix_plies: int = 0,
    initial_position: str = chess.STARTING_FEN,
    seed: int = 3,
) -> GameRecord:
    action_ids = tuple(encode_move(chess.Move.from_uci(uci)) for uci in moves)
    root = chess.Board(initial_position)
    return build_game_record(
        initial_position=initial_position,
        prefix_plies=prefix_plies,
        action_ids=action_ids,
        white=SeatRecord(kind="model", label="white-seat", seed=1),
        black=SeatRecord(kind="model", label="black-seat", seed=2),
        seed=seed,
        decisions=[
            DecisionRecord(
                ply_index=index,
                slot="white" if (index % 2 == 0) == root.turn else "black",
                action_id=action_id,
            )
            for index, action_id in enumerate(action_ids)
            if index >= prefix_plies
        ],
        outcome=GameOutcome(
            result=result,  # type: ignore[arg-type]
            termination=termination,
            adjudicated=adjudicated,
        ),
    )


def test_features_come_from_the_record_without_replaying_a_model() -> None:
    record = _record(("e2e4", "c7c5", "g1f3"))

    features = analyze_game(record)

    assert features.game_id == record.game_id
    assert features.ply_count == 3
    assert features.generated_plies == 3
    assert features.termination is GameTermination.PLY_LIMIT
    assert features.distinct_move_fraction == 1.0


def test_openings_are_named_by_family_from_the_owned_book() -> None:
    sicilian = analyze_game(_record(("e2e4", "c7c5", "g1f3", "d7d6")))

    assert sicilian.opening.family == "Sicilian Defense"
    assert sicilian.opening.classified


def test_a_position_the_book_does_not_name_is_reported_as_unclassified() -> None:
    features = analyze_game(
        _record(
            ("e1e2", "a7a6", "e2e3"),
            initial_position="4k3/p7/8/8/8/8/P7/4K3 w - - 0 1",
        )
    )

    assert features.opening.family == UNCLASSIFIED
    assert not features.opening.classified


def test_a_shuffled_game_reports_when_recurrence_started_and_how_deep_it_went() -> None:
    features = analyze_game(_record(SHUFFLE))

    repetition = features.repetition
    assert repetition.repeated
    assert repetition.first_repetition_ply == 3
    assert repetition.threefold_claimable
    assert repetition.threefold_ply == 7
    assert repetition.maximum_occurrences == 3
    assert repetition.repeated_ply_fraction == pytest.approx(0.625)
    # Every ply from the first recurrence onward is itself a recurrence, which
    # is what staying inside a cycle looks like.
    assert repetition.cycle_ply_fraction == pytest.approx(1.0)
    assert features.distinct_move_fraction == pytest.approx(0.5)


def test_a_game_that_never_repeats_reports_no_recurrence() -> None:
    features = analyze_game(_record(("e2e4", "e7e5", "g1f3", "b8c6")))

    assert not features.repetition.repeated
    assert features.repetition.first_repetition_ply is None
    assert features.repetition.threefold_ply is None
    assert features.repetition.maximum_occurrences == 1
    assert features.repetition.repeated_ply_fraction == 0.0
    assert features.repetition.cycle_ply_fraction == 0.0


def test_a_prefix_continuation_separates_played_plies_from_generated_ones() -> None:
    record = _record(("e2e4", "c7c5", "g1f3", "d7d6"), prefix_plies=2)

    features = analyze_game(record)

    assert features.ply_count == 4
    assert features.generated_plies == 2
    assert features.opening.family == "Sicilian Defense"


def test_a_summary_aggregates_the_shapes_a_rollout_benchmark_reports() -> None:
    records = (
        _record(
            ("e2e4", "c7c5", "g1f3", "d7d6"),
            result="1-0",
            termination=GameTermination.RESIGNATION,
            adjudicated=False,
        ),
        _record(("e2e4", "e7e5", "g1f3", "b8c6")),
        _record(SHUFFLE),
    )

    summary = summarize_games(analyze_games(records))

    assert summary.version == GAME_ANALYSIS_VERSION
    assert summary.games == 3
    assert summary.mean_ply_count == pytest.approx((4 + 4 + 8) / 3)
    assert summary.result_counts == {"*": 2, "1-0": 1}
    assert summary.termination_counts == {"ply_limit": 2, "resignation": 1}
    assert summary.adjudicated_games == 2
    assert summary.opening_counts["Sicilian Defense"] == 1
    assert summary.repeated_games == 1
    assert summary.threefold_claimable_games == 1
    # Averaged over the one game that repeated, not diluted by the two that
    # never did.
    assert summary.mean_cycle_ply_fraction == pytest.approx(1.0)
    assert summary.mean_repeated_ply_fraction == pytest.approx(0.625 / 3)
    assert summary.distinct_game_fraction == 1.0
    assert summary.opening_book["name"]
    assert summary.opening_level is OpeningLevel.FAMILY


def test_a_summary_reports_identical_trajectories_as_collapsed_diversity() -> None:
    record = _record(("e2e4", "c7c5"))

    summary = summarize_games(analyze_games((record, record, record)))

    assert summary.games == 3
    assert summary.distinct_game_fraction == pytest.approx(1 / 3)


def test_replicates_that_played_the_same_game_collapse_diversity() -> None:
    """Diversity is a property of the trajectory, not of the record.

    A record's identity is derived from the whole record, seeds included, so
    replicates that played the identical game carry different ids. Counting ids
    would report a suite collapsed to one trajectory as fully diverse, which is
    precisely the failure this metric exists to catch.
    """

    moves = ("e2e4", "c7c5")
    records = tuple(_record(moves, seed=seed) for seed in (1, 2, 3))

    summary = summarize_games(analyze_games(records))

    assert len({record.game_id for record in records}) == 3
    assert summary.distinct_game_fraction == pytest.approx(1 / 3)


def test_the_same_moves_from_a_different_root_is_a_different_trajectory() -> None:
    """The root belongs to the trajectory, not only the moves played from it.

    A prefix arm continues many different positions, and two continuations that
    happen to play the same moves from different openings are two games. Keying
    on the move list alone would merge them.
    """

    moves = ("e2e4", "c7c5")
    # The standard position, and the standard position a black pawn short. The
    # same two moves are legal in both, so only the root differs.
    handicap = "rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    features = analyze_games(
        (
            _record(moves),
            _record(moves, initial_position=handicap),
        )
    )

    assert len({feature.trajectory_sha256 for feature in features}) == 2


def test_a_summary_reports_when_recurrence_started_across_games() -> None:
    """Averaged over the games that repeated, so it says when, not how often."""

    records = (
        _record(SHUFFLE),
        _record(("e2e4", "e7e5", "g1f3", "b8c6")),
    )

    summary = summarize_games(analyze_games(records))

    assert summary.repeated_games == 1
    assert summary.mean_first_repetition_ply == pytest.approx(3.0)


def test_a_summary_with_no_recurrence_reports_no_first_repetition_ply() -> None:
    summary = summarize_games(analyze_games((_record(("e2e4", "c7c5")),)))

    assert summary.repeated_games == 0
    assert summary.mean_first_repetition_ply == 0.0


def test_a_summary_can_aggregate_at_a_deeper_naming_level() -> None:
    records = (_record(("e2e4", "c7c5", "g1f3", "d7d6")),)

    summary = summarize_games(analyze_games(records), level=OpeningLevel.LINE)

    assert summary.opening_level is OpeningLevel.LINE
    assert all("Sicilian" in name for name in summary.opening_counts)


def test_a_summary_needs_at_least_one_game() -> None:
    with pytest.raises(ValueError, match="at least one game"):
        summarize_games(())


def test_a_summary_record_is_json_ready() -> None:
    summary = summarize_games(analyze_games((_record(("e2e4", "c7c5")),)))

    record = summary.as_record()

    assert record["games"] == 1
    assert record["opening_level"] == "family"
    assert isinstance(record["opening_counts"], dict)


def test_the_final_material_is_signed_from_one_seat() -> None:
    """Two readings judge different players, so neither side may be assumed."""

    trajectory = analyze_game(
        _record(("e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"))
    ).trajectory

    assert trajectory.final_material_balance == 1
    assert trajectory.final_turn_white is False
    assert trajectory.material_advantage(white=True) == pytest.approx(1.0)
    assert trajectory.material_advantage(white=False) == pytest.approx(-1.0)


def test_a_claim_the_rules_never_offered_is_not_available() -> None:
    """The non-termination guardrail's denominator, and it must be able to be empty."""

    assert analyze_game(_record(("e2e4", "e7e5"))).trajectory.claim_ever_available is (
        False
    )


def test_a_threefold_makes_a_claim_available() -> None:
    """The counter that reports the repetition is the one that reports the claim."""

    trajectory = analyze_game(_record(SHUFFLE)).trajectory

    assert trajectory.repetition.threefold_ply == 7
    assert trajectory.claim_ever_available is True


def test_a_full_fifty_move_clock_makes_a_claim_available() -> None:
    """A claim is not only a repetition, and the counter cannot see this half."""

    # Ninety-nine half-moves already made, so one more fills the clock.
    position = "7k/8/8/8/8/8/R7/6K1 w - - 99 80"
    trajectory = analyze_game(_record(("a2a3",), initial_position=position)).trajectory

    assert trajectory.repetition.threefold_ply is None
    assert trajectory.claim_ever_available is True


@pytest.mark.parametrize(
    "moves",
    (
        pytest.param(("e2e4", "e7e5"), id="neither"),
        pytest.param(SHUFFLE, id="threefold"),
        pytest.param(("g1f3", "g8f6", "f3g1"), id="repeated-once"),
    ),
)
def test_claim_availability_agrees_with_the_rule_the_runtime_offers_it_by(
    moves: Sequence[str],
) -> None:
    """One definition decides the denominator and whether a seat was offered one.

    The walk reads claim availability off the repetition counter rather than
    through `draw_claim_available`, so what keeps the two one rule is that they
    are checked against each other.
    """

    board = chess.Board()
    reached = [draw_claim_available(board)]
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
        reached.append(draw_claim_available(board))

    assert analyze_game(_record(moves)).trajectory.claim_ever_available is any(reached)
