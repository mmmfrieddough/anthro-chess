"""Tests for assembling rating dependency tests from scored passes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest

from anthro_chess.evaluation.dependency import (
    DEGRADATION_METRICS,
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
from anthro_chess.evaluation.results.metrics import (
    DEPENDENCY_RATING_ANCHOR_POLICY_DIVERGENCE,
    DEPENDENCY_RATING_ANCHOR_TOP1_AGREEMENT,
)

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


def test_per_game_totals_recover_the_reported_degradation() -> None:
    """A game's share carries its own weight, not an equal one.

    The two games hold different numbers of rated positions, which is the case
    an unweighted mean over games gets wrong. Recovering the reported value
    from the retained totals is what says the dispersion describes the quantity
    printed beside it.
    """

    contexts = {
        (1, 0): _context(1, 0, rating=1500, band="1200_to_1599"),
        (2, 0): _context(2, 0, rating=1500, band="1200_to_1599"),
        (2, 1): _context(2, 1, rating=1500, band="1200_to_1599"),
        (2, 2): _context(2, 2, rating=1500, band="1200_to_1599"),
    }
    true = [
        _policy(1, 0, 2.0),
        _policy(2, 0, 2.0),
        _policy(2, 1, 3.0),
        _policy(2, 2, 4.0),
    ]
    corrupted = {
        "absent": (
            Conditioning(name="absent", kind=ConditioningKind.ABSENT),
            [
                _policy(1, 0, 5.0),
                _policy(2, 0, 2.1),
                _policy(2, 1, 3.2),
                _policy(2, 2, 4.3),
            ],
        ),
    }

    result = _build(contexts, true, corrupted)

    degradation = result.corruption(ConditioningKind.ABSENT)
    absent = DEGRADATION_METRICS[ConditioningKind.ABSENT].identifier
    shares = {item.game_id: item.metrics[absent] for item in result.per_game_totals}
    assert degradation is not None
    assert set(shares) == {1, 2}
    assert shares[1].positions == 1
    assert shares[2].positions == 3
    assert shares[1].total == pytest.approx(3.0)
    assert shares[2].total == pytest.approx(0.6)
    weighted = sum(item.total for item in shares.values()) / sum(
        item.positions for item in shares.values()
    )
    assert weighted == pytest.approx(degradation.degradation)
    assert weighted != pytest.approx(
        (3.0 + 0.2) / 2,
    ), "an unweighted mean over games would not be the reported degradation"


def test_the_anchor_quantities_are_weighted_by_the_positions_that_carry_them() -> None:
    """A rated position with no anchor signal is absent from the anchors only.

    The degradations average over every rated position and the anchors average
    over the ones a signal was computed for, so each metric carries the count
    behind its own total rather than sharing one.
    """

    contexts = {
        (1, 0): _context(1, 0, rating=1500, band="1200_to_1599"),
        (2, 0): _context(2, 0, rating=1500, band="1200_to_1599"),
    }
    trajectory = {
        (1, 0): TrajectorySignal(
            strength_signal=0.0,
            alignment=0.0,
            anchor_divergence=0.4,
            anchor_agreement=True,
        ),
    }
    corrupted = {
        "shuffled": (
            Conditioning(name="shuffled", kind=ConditioningKind.SHUFFLED),
            [_policy(1, 0, 2.5), _policy(2, 0, 2.5)],
        ),
    }

    result = _build(
        contexts,
        [_policy(1, 0, 2.0), _policy(2, 0, 2.0)],
        corrupted,
        trajectory=trajectory,
    )

    shuffled = DEGRADATION_METRICS[ConditioningKind.SHUFFLED].identifier
    divergence = DEPENDENCY_RATING_ANCHOR_POLICY_DIVERGENCE.identifier
    agreement = DEPENDENCY_RATING_ANCHOR_TOP1_AGREEMENT.identifier
    games = {item.game_id: item.metrics for item in result.per_game_totals}
    assert games[2][shuffled].positions == 1
    assert games[2][divergence].positions == 0
    assert games[1][divergence].positions == 1
    assert games[1][divergence].total == pytest.approx(0.4)
    assert games[1][agreement].total == pytest.approx(1.0)
    recovered = sum(metrics[divergence].total for metrics in games.values()) / sum(
        metrics[divergence].positions for metrics in games.values()
    )
    assert recovered == pytest.approx(result.anchor_divergence)


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
