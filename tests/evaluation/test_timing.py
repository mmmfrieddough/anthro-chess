"""The move-time readings' arithmetic, on decisions built by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from anthro_chess.evaluation.results import DataComponent
from anthro_chess.evaluation.timing import (
    MARGINAL_BIN_WIDTH,
    TimingAccumulator,
    TimingColumns,
)

_COMPONENT = DataComponent(
    projection="move_timing",
    projection_version=1,
    content_sha256="0" * 64,
    games=1,
)


def _columns(
    human_ms: list[int],
    sampled_ms: list[int],
    *,
    speed: str = "blitz",
    clock: str = "clock_30s_plus",
) -> TimingColumns:
    count = len(human_ms)
    return TimingColumns(
        human_ms=np.array(human_ms, dtype=np.int64),
        sampled_ms=np.array(sampled_ms, dtype=np.int64),
        loss=np.full(count, 2.0),
        slices=((speed, clock),) * count,
        cells=(("1200_to_1599", speed, clock, "middlegame"),) * count,
    )


def _readings(accumulator: TimingAccumulator) -> dict[str, float]:
    return {item.metric: item.value for item in accumulator.measurements(_COMPONENT)}


def test_a_sample_matching_the_humans_is_no_distance_from_them() -> None:
    accumulator = TimingAccumulator()
    accumulator.add(_columns([800, 2_000, 9_000], [9_000, 800, 2_000]))

    readings = _readings(accumulator)

    assert readings["timing.move_time_distance"] == pytest.approx(0.0)
    assert readings["timing.move_time_loss"] == pytest.approx(2.0)


def test_the_distance_is_how_far_the_logged_times_moved() -> None:
    human = [1_000, 3_000, 20_000]
    accumulator = TimingAccumulator()
    accumulator.add(
        _columns(human, [round(math.expm1(math.log1p(ms) + 1.0)) for ms in human])
    )

    assert _readings(accumulator)["timing.move_time_distance"] == pytest.approx(
        1.0, abs=0.02
    )


def test_the_marginal_spreads_a_bins_share_over_the_times_it_holds() -> None:
    accumulator = TimingAccumulator()
    accumulator.add(_columns([2_000] * 4, [2_000] * 4))
    logged_bin = math.floor(math.log1p(2_000) / MARGINAL_BIN_WIDTH)
    lower, upper = (
        math.ceil(math.expm1(edge * MARGINAL_BIN_WIDTH) / 10)
        for edge in (logged_bin, logged_bin + 1)
    )

    marginal = _readings(accumulator)["timing.marginal_move_time_loss"]

    assert marginal == pytest.approx(math.log(upper - lower))


def test_each_reading_is_reported_overall_and_in_its_speed_and_clock() -> None:
    accumulator = TimingAccumulator()
    accumulator.add(_columns([1_000], [1_000], speed="bullet", clock="clock_under_10s"))
    accumulator.add(_columns([5_000], [5_000]))

    readings = _readings(accumulator)

    assert {
        "timing.move_time_loss",
        "timing.move_time_loss_bullet",
        "timing.move_time_loss_blitz",
        "timing.move_time_loss_clock_under_10s",
        "timing.move_time_loss_clock_30s_plus",
    } <= set(readings)
    assert "timing.move_time_loss_rapid" not in readings
    record = accumulator.as_record()
    slices = record["slices"]
    assert isinstance(slices, dict)
    assert slices["all"]["moves"] == 2
