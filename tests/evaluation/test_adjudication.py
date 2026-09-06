"""Rare forced outcomes are read against humans, not perfect play."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from anthro_chess.evaluation.adjudication import AdjudicationAccumulator, action_sets
from anthro_chess.evaluation.policy import ActionSetPolicy
from anthro_chess.evaluation.results import DataComponent
from anthro_chess.evaluation.scoring import build_scoring_inputs
from anthro_chess.evaluation.slices import PositionPredicate

FORCED_FEN = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"


def test_adjudication_reports_human_model_and_rating_band_rates(
    normalized_row: Callable[..., dict[str, Any]],
    fixture_game_id: Callable[[int], int],
) -> None:
    row = normalized_row(
        41,
        split="test",
        rating=1500,
        initial_position=FORCED_FEN,
        moves=("f7f8",),
    )
    inputs = build_scoring_inputs(
        [row],
        split="test",
        batch_size=1,
        length_bucket_width=None,
        identity_sha256="a" * 64,
    )
    key = (fixture_game_id(41), 0)
    target = inputs.plies[key].target_action_id
    scores = tuple(
        ActionSetPolicy(
            game_id=fixture_game_id(41),
            ply_index=0,
            name=predicate.value,
            selected_action_id=target,
            raw_probability_mass=(
                0.75 if predicate is PositionPredicate.MATE_AVAILABLE else 0.1
            ),
            best_rank=1 if predicate is PositionPredicate.MATE_AVAILABLE else 4,
        )
        for predicate in inputs.labels(key).predicates
    )

    accumulator = AdjudicationAccumulator(retain_positions=True)
    accumulator.add(scores, inputs)
    report = accumulator.report()

    assert report is not None
    mate = report.predicates[PositionPredicate.MATE_AVAILABLE]
    assert mate.overall.games == 1
    assert mate.overall.opportunities == 1
    assert mate.overall.effective_sample_size == pytest.approx(1.0)
    assert mate.overall.human_rate == pytest.approx(1.0)
    assert mate.overall.selected_rate == pytest.approx(1.0)
    assert mate.overall.policy_mass == pytest.approx(0.75)
    assert mate.overall.human_gap == pytest.approx(0.0)
    assert mate.rating_bands["1200_to_1599"] == mate.overall
    assert mate.mean_best_rank == pytest.approx(1.0)
    assert mate.rankable_opportunities == 1

    totals = report.per_game_totals
    assert len(totals) == 1
    assert totals[0].metrics[
        "adjudicated.mate_available_human_rate"
    ].total == pytest.approx(1.0)


def test_no_realized_predicate_is_explicitly_unavailable(
    normalized_row: Callable[..., dict[str, Any]],
) -> None:
    row = normalized_row(42, split="test", rating=1500, moves=("e2e4",))
    inputs = build_scoring_inputs(
        [row],
        split="test",
        batch_size=1,
        length_bucket_width=None,
        identity_sha256="b" * 64,
    )

    accumulator = AdjudicationAccumulator()
    accumulator.add((), inputs)
    assert accumulator.report() is None


def test_action_sets_narrow_to_the_positions_a_reading_keeps(
    normalized_row: Callable[..., dict[str, Any]],
    fixture_game_id: Callable[[int], int],
) -> None:
    """A benchmark scoring a window inside longer games asks for the window.

    Resolving the predicates of a position whose score is discarded is that
    position's whole cost, so the subsets a scorer consumes are built over the
    keys it will keep rather than over the view.
    """

    rows = [
        normalized_row(
            game_id,
            split="test",
            rating=1500,
            initial_position=FORCED_FEN,
            moves=("f7f8",),
        )
        for game_id in (43, 44)
    ]
    inputs = build_scoring_inputs(
        rows,
        split="test",
        batch_size=1,
        length_bucket_width=None,
        identity_sha256="e" * 64,
    )
    wide = action_sets(inputs)

    assert set(wide) == {(fixture_game_id(43), 0), (fixture_game_id(44), 0)}
    narrowed = (fixture_game_id(44), 0)
    assert action_sets(inputs, [narrowed]) == {narrowed: wide[narrowed]}


def test_the_human_gap_is_reported_per_rating_band(
    normalized_row: Callable[..., dict[str, Any]],
    fixture_game_id: Callable[[int], int],
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    """The band drill-down is what says whether the dial delivers a player."""

    rows = [
        normalized_row(
            index,
            split="test",
            rating=rating,
            initial_position=FORCED_FEN,
            moves=("f7f8",),
        )
        for index, rating in ((43, 1500), (44, 2200))
    ]
    inputs = build_scoring_inputs(
        rows,
        split="test",
        batch_size=2,
        length_bucket_width=None,
        identity_sha256="c" * 64,
    )

    scores: list[ActionSetPolicy] = []
    for index, converts in ((43, True), (44, False)):
        key = (fixture_game_id(index), 0)
        matches = inputs.labels(key).predicates
        target = inputs.plies[key].target_action_id
        off_target = (
            max(matches[PositionPredicate.MATE_AVAILABLE].successful_action_ids) + 1
        )
        scores.extend(
            ActionSetPolicy(
                game_id=fixture_game_id(index),
                ply_index=0,
                name=predicate.value,
                selected_action_id=target if converts else off_target,
                raw_probability_mass=0.75,
                best_rank=1,
            )
            for predicate in matches
        )

    accumulator = AdjudicationAccumulator()
    accumulator.add(tuple(scores), inputs)
    report = accumulator.report()

    assert report is not None
    values = {
        item.metric: item
        for item in report.measurements(move_prediction_component(rows))
    }
    matched = values["adjudicated.mate_available_human_gap_1200_to_1599"]
    missed = values["adjudicated.mate_available_human_gap_2000_plus"]
    assert matched.value == pytest.approx(0.0)
    assert matched.sample_size == 1
    assert missed.value == pytest.approx(-1.0)
    assert missed.sample_size == 1
    assert values["adjudicated.mate_available_human_gap"].value == pytest.approx(-0.5)
    assert "adjudicated.mate_available_human_gap_under_1200" not in values

    floored = {
        metric
        for game in report.per_game_totals
        for metric in game.metrics
        if metric.startswith("adjudicated.mate_available_human_gap_")
    }
    assert floored == {
        "adjudicated.mate_available_human_gap_1200_to_1599",
        "adjudicated.mate_available_human_gap_2000_plus",
    }
