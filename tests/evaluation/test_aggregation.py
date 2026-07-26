"""Tests for slice aggregation over scored held-out positions."""

from __future__ import annotations

import json

import pytest

from anthro_chess.evaluation.aggregation import (
    COLOR_DIMENSION,
    LEGAL_MOVE_COUNT_DIMENSION,
    PHASE_DIMENSION,
    RATING_DIMENSION,
    REPORTED_CHARACTERISTICS,
    RULE_CASE_DIMENSION,
    UNRATED_SLICE,
    SliceAggregator,
    summarize,
)
from anthro_chess.evaluation.policy import PositionPolicy
from anthro_chess.evaluation.results.metrics import (
    PHASE_SLICE_NAMES,
    RATING_BAND_SLICE_NAMES,
    RULE_CASE_SLICE_NAMES,
    UNRATED_SLICE_NAME,
)
from anthro_chess.evaluation.slices import (
    DEFAULT_RATING_BANDS,
    GamePhase,
    PlayerColor,
    PositionCharacteristic,
    PositionSlices,
)


def test_slices_cover_every_dimension_and_omit_empty_ones() -> None:
    aggregator = SliceAggregator()
    aggregator.add(
        _policy(move_nll=1.0, mask_penalty=0.5),
        _slices(phase=GamePhase.OPENING, rating_band="1200_to_1599"),
        (PositionCharacteristic.CHECK,),
    )
    aggregator.add(
        _policy(ply_index=1, move_nll=3.0, mask_penalty=1.5),
        _slices(
            phase=GamePhase.ENDGAME,
            color=PlayerColor.BLACK,
            rating_band=None,
            legal_move_count=30,
            bucket="26_plus",
        ),
        (),
    )

    table = aggregator.compute()

    assert table.overall.position_count == 2
    assert table.overall.move_loss == pytest.approx(2.0)
    assert table.overall.mask_penalty == pytest.approx(1.0)
    opening = table.slice_summary(PHASE_DIMENSION, str(GamePhase.OPENING))
    assert opening is not None
    assert opening.mask_penalty == pytest.approx(0.5)
    assert table.slice_summary(PHASE_DIMENSION, str(GamePhase.MIDDLEGAME)) is None
    assert table.slice_summary(COLOR_DIMENSION, str(PlayerColor.BLACK)) is not None
    assert table.slice_summary(RATING_DIMENSION, UNRATED_SLICE) is not None
    assert table.slice_summary(LEGAL_MOVE_COUNT_DIMENSION, "26_plus") is not None
    check = table.slice_summary(RULE_CASE_DIMENSION, str(PositionCharacteristic.CHECK))
    assert check is not None
    assert check.position_count == 1
    assert (
        table.slice_summary(RULE_CASE_DIMENSION, str(PositionCharacteristic.EN_PASSANT))
        is None
    )
    json.dumps(table.as_record(), sort_keys=True, allow_nan=False)


def test_top_k_accuracy_follows_the_target_rank() -> None:
    summary = summarize(
        [
            _policy(ply_index=0, target_rank=1),
            _policy(ply_index=1, target_rank=3),
            _policy(ply_index=2, target_rank=6),
            _policy(ply_index=3, target_rank=4),
        ]
    )

    assert summary is not None
    assert summary.accuracy(1) == pytest.approx(0.25)
    assert summary.accuracy(3) == pytest.approx(0.5)
    assert summary.accuracy(5) == pytest.approx(0.75)
    with pytest.raises(KeyError, match="top-2"):
        summary.accuracy(2)


def test_a_position_joins_every_rule_case_it_realizes() -> None:
    aggregator = SliceAggregator()
    aggregator.add(
        _policy(),
        _slices(),
        (
            PositionCharacteristic.CHECK,
            PositionCharacteristic.PIN,
            PositionCharacteristic.TERMINAL,
        ),
    )

    table = aggregator.compute()

    reported = set(table.dimensions[RULE_CASE_DIMENSION])
    assert reported == {
        str(PositionCharacteristic.CHECK),
        str(PositionCharacteristic.PIN),
    }


def test_an_empty_evaluation_is_rejected_rather_than_reported_as_zero() -> None:
    assert summarize([]) is None
    with pytest.raises(ValueError, match="at least one scored position"):
        SliceAggregator().compute()


def test_registered_slice_names_match_the_slice_layer() -> None:
    assert PHASE_SLICE_NAMES == tuple(str(phase) for phase in GamePhase)
    assert RATING_BAND_SLICE_NAMES == tuple(band.name for band in DEFAULT_RATING_BANDS)
    assert RULE_CASE_SLICE_NAMES == tuple(
        sorted(str(case) for case in REPORTED_CHARACTERISTICS)
    )
    assert UNRATED_SLICE_NAME == UNRATED_SLICE


def _policy(
    *,
    game_id: int = 1,
    ply_index: int = 0,
    move_nll: float = 2.0,
    mask_penalty: float = 0.25,
    target_rank: int = 1,
) -> PositionPolicy:
    return PositionPolicy(
        game_id=game_id,
        ply_index=ply_index,
        target_action_id=7,
        legal_action_count=20,
        conditioned_rating=1500,
        move_nll=move_nll,
        legal_move_nll=move_nll - 0.1,
        uniform_over_legal_move_nll=3.0,
        mask_penalty=mask_penalty,
        legal_mass=0.8,
        legality_lift=1.5,
        legal_margin=0.5,
        top1_illegal=False,
        top_illegal_fraction=0.2,
        target_rank=target_rank,
    )


def _slices(
    *,
    phase: GamePhase = GamePhase.MIDDLEGAME,
    color: PlayerColor = PlayerColor.WHITE,
    rating_band: str | None = "1200_to_1599",
    legal_move_count: int = 20,
    bucket: str = "11_to_25",
) -> PositionSlices:
    return PositionSlices(
        phase=phase,
        color=color,
        legal_move_count=legal_move_count,
        legal_move_count_bucket=bucket,
        rating_band=rating_band,
    )
