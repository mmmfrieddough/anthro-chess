"""Tests for assembling rating dependency tests from scored passes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest

from anthro_chess.evaluation.dependency import (
    STRONGER_PREFIX_GROUP,
    WEAKER_PREFIX_GROUP,
    Conditioning,
    ConditioningKind,
    DependencyError,
    DependencyTestConfig,
    DependencyTestResult,
    MaturityContext,
    PositionContext,
    PositionKey,
    TrajectorySignal,
    build_dependency_result,
)
from anthro_chess.evaluation.policy import PositionPolicy

CONDITIONING_VALUES = (1000, 2200)


def test_corruption_degradation_uses_only_positions_with_a_true_rating() -> None:
    contexts = {
        (1, 0): _context(1, 0, rating=1500, band="1200_to_1599"),
        (1, 1): _context(1, 1, rating=None, band=None),
    }
    true = [_policy(1, 0, 2.0), _policy(1, 1, 9.0)]
    corrupted = {
        "shuffled": (
            Conditioning(name="shuffled", kind=ConditioningKind.SHUFFLED),
            [_policy(1, 0, 2.5), _policy(1, 1, 1.0)],
        ),
    }

    result = _build(contexts, true, corrupted)

    shuffled = result.corruption(ConditioningKind.SHUFFLED)
    assert result.rated_position_count == 1
    assert result.true_move_loss == pytest.approx(2.0)
    assert shuffled is not None
    assert shuffled.position_count == 1
    assert shuffled.degradation == pytest.approx(0.5)


def test_cross_conditioning_matches_when_each_slice_prefers_its_own_value() -> None:
    contexts: dict[PositionKey, PositionContext] = {}
    true: list[PositionPolicy] = []
    low_scores: list[PositionPolicy] = []
    high_scores: list[PositionPolicy] = []
    for index in range(4):
        contexts[(1, index)] = _context(1, index, rating=1000, band="under_1200")
        contexts[(2, index)] = _context(2, index, rating=2200, band="2000_plus")
        true.extend([_policy(1, index, 2.0), _policy(2, index, 2.0)])
        low_scores.extend([_policy(1, index, 1.0), _policy(2, index, 4.0)])
        high_scores.extend([_policy(1, index, 3.0), _policy(2, index, 1.5)])

    result = _build(
        contexts,
        true,
        {},
        conditioned={1000: low_scores, 2200: high_scores},
    )

    cross = result.cross_conditioning
    assert cross.match_rate == pytest.approx(1.0)
    assert set(cross.compared_bands) == {"under_1200", "2000_plus"}
    assert cross.excluded_bands == ()
    assert len(cross.cells) == 4
    assert {
        (cell.rating_band, cell.conditioning_rating): cell.move_loss
        for cell in cross.cells
    }[("under_1200", 1000)] == pytest.approx(1.0)


def test_cross_conditioning_reports_thin_slices_without_scoring_them() -> None:
    contexts = {(1, 0): _context(1, 0, rating=1000, band="under_1200")}
    true = [_policy(1, 0, 2.0)]

    result = _build(
        contexts,
        true,
        {},
        conditioned={1000: [_policy(1, 0, 1.0)], 2200: [_policy(1, 0, 3.0)]},
        minimum_slice_positions=2,
    )

    cross = result.cross_conditioning
    assert cross.compared_bands == ()
    assert cross.excluded_bands == ("under_1200",)
    assert cross.match_rate is None
    assert len(cross.cells) == 2


def test_within_game_response_compares_prefix_strength_halves() -> None:
    contexts: dict[PositionKey, PositionContext] = {}
    trajectory: dict[PositionKey, TrajectorySignal] = {}
    true: list[PositionPolicy] = []
    for ply_index in range(8):
        contexts[(1, ply_index)] = _context(
            1,
            ply_index,
            rating=1500,
            band="1200_to_1599",
            color="white" if ply_index % 2 == 0 else "black",
        )
        true.append(_policy(1, ply_index, 2.0))
        # White's play looks progressively stronger; Black's looks weaker.
        strength = 1.0 if ply_index % 2 == 0 else -1.0
        trajectory[(1, ply_index)] = TrajectorySignal(
            strength_signal=strength,
            alignment=0.4 if ply_index % 2 == 0 else 0.1,
            anchor_divergence=0.2,
            anchor_agreement=ply_index % 4 == 0,
        )

    result = _build(
        contexts,
        true,
        {},
        trajectory=trajectory,
        minimum_slice_positions=1,
        minimum_prefix_decisions=1,
    )

    within = result.within_game
    groups = {(group.rating_band, group.group): group for group in within.groups}
    assert within.compared_bands == ("1200_to_1599",)
    assert within.anchor_low_rating == 1000
    assert within.anchor_high_rating == 2200
    assert groups[("1200_to_1599", STRONGER_PREFIX_GROUP)].mean_prefix_strength > 0
    assert groups[("1200_to_1599", WEAKER_PREFIX_GROUP)].mean_prefix_strength < 0
    assert within.response == pytest.approx(0.3)
    assert result.anchor_divergence == pytest.approx(0.2)
    assert result.anchor_agreement_rate == pytest.approx(0.25)


def test_result_carries_maturity_and_states_no_verdict() -> None:
    contexts = {(1, 0): _context(1, 0, rating=1500, band="1200_to_1599")}

    record = _build(contexts, [_policy(1, 0, 2.0)], {}).as_record()

    assert record["maturity"] == {"step": 8000, "processed_positions": 640_000}
    assert "interpretation" in record
    assert not {"passed", "failed", "verdict"} & set(record)
    json.dumps(record, sort_keys=True, allow_nan=False)


def test_a_missing_conditioning_pass_fails_clearly() -> None:
    contexts = {(1, 0): _context(1, 0, rating=1500, band="1200_to_1599")}

    with pytest.raises(DependencyError, match="missing a scoring pass"):
        _build(contexts, [_policy(1, 0, 2.0)], {}, conditioned={1000: []})


def test_a_conditioning_grid_needs_two_distinct_ratings() -> None:
    """A rating named twice is one column, not two."""

    with pytest.raises(ValueError, match="two distinct conditioning ratings"):
        DependencyTestConfig(cross_conditioning_ratings=(1500, 1500))


def test_dependency_tests_need_a_rated_position() -> None:
    contexts = {(1, 0): _context(1, 0, rating=None, band=None)}

    with pytest.raises(DependencyError, match="present rating"):
        _build(contexts, [_policy(1, 0, 2.0)], {})


def _build(
    contexts: Mapping[PositionKey, PositionContext],
    true: Sequence[PositionPolicy],
    corrupted: Mapping[str, tuple[Conditioning, Sequence[PositionPolicy]]],
    *,
    conditioned: Mapping[int, Sequence[PositionPolicy]] | None = None,
    trajectory: Mapping[PositionKey, TrajectorySignal] | None = None,
    minimum_slice_positions: int = 1,
    minimum_prefix_decisions: int = 1,
) -> DependencyTestResult:
    resolved = (
        {value: list(true) for value in CONDITIONING_VALUES}
        if conditioned is None
        else conditioned
    )
    signals = trajectory or {
        key: TrajectorySignal(
            strength_signal=0.0,
            alignment=0.0,
            anchor_divergence=0.2,
            anchor_agreement=True,
        )
        for key in contexts
    }
    return build_dependency_result(
        config=DependencyTestConfig(
            cross_conditioning_ratings=CONDITIONING_VALUES,
            minimum_slice_positions=minimum_slice_positions,
            minimum_prefix_decisions=minimum_prefix_decisions,
        ),
        contexts=contexts,
        true_positions=true,
        corrupted_positions=corrupted,
        conditioned_positions=resolved,
        trajectory=signals,
        maturity=MaturityContext(step=8000, processed_positions=640_000),
    )


def _context(
    game_id: int,
    ply_index: int,
    *,
    rating: int | None,
    band: str | None,
    color: str = "white",
) -> PositionContext:
    return PositionContext(
        game_id=game_id,
        ply_index=ply_index,
        color=color,
        rating=rating,
        rating_band=band,
    )


def _policy(game_id: int, ply_index: int, move_nll: float) -> PositionPolicy:
    return PositionPolicy(
        game_id=game_id,
        ply_index=ply_index,
        target_action_id=11,
        legal_action_count=20,
        conditioned_rating=1500,
        move_nll=move_nll,
        legal_move_nll=move_nll - 0.1,
        uniform_over_legal_move_nll=3.0,
        mask_penalty=0.25,
        legal_mass=0.8,
        legality_lift=1.5,
        legal_margin=0.5,
        top1_illegal=False,
        top_illegal_fraction=0.2,
        target_rank=1,
    )
