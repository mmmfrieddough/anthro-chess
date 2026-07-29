"""Rare forced outcomes are read against humans, not perfect play."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from anthro_chess.evaluation.adjudication import build_adjudication_report
from anthro_chess.evaluation.policy import ActionSetPolicy
from anthro_chess.evaluation.scoring import build_scoring_inputs
from anthro_chess.evaluation.slices import PositionPredicate

FORCED_FEN = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"


def test_adjudication_reports_human_model_and_rating_band_rates(
    normalized_row: Callable[..., dict[str, Any]],
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
    key = (41, 0)
    target = inputs.plies[key].target_action_id
    scores = tuple(
        ActionSetPolicy(
            game_id=41,
            ply_index=0,
            name=predicate.value,
            selected_action_id=target,
            raw_probability_mass=(
                0.75 if predicate is PositionPredicate.MATE_AVAILABLE else 0.1
            ),
        )
        for predicate in inputs.predicates[key]
    )

    report = build_adjudication_report(scores, inputs)

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

    totals = report.per_game_totals()
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

    assert build_adjudication_report((), inputs) is None
