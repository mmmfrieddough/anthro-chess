from __future__ import annotations

import math

import pytest

from anthro_chess.training.schedule import LearningRateSchedule, resolve_schedule


def _schedule(
    *,
    steps: int = 1000,
    warmup_sequences: int = 800,
    cooldown_fraction: float = 0.2,
    sequences_per_step: int = 8,
    peak: float = 0.01,
) -> LearningRateSchedule:
    return resolve_schedule(
        peak=peak,
        steps=steps,
        warmup_sequences=warmup_sequences,
        cooldown_fraction=cooldown_fraction,
        sequences_per_step=sequences_per_step,
    )


def test_warmup_converts_the_declared_data_through_batch_and_accumulation() -> None:
    schedule = _schedule(warmup_sequences=800, sequences_per_step=8)
    accumulated = _schedule(warmup_sequences=800, sequences_per_step=32)

    assert schedule.warmup_steps == 100
    assert accumulated.warmup_steps == 25


def test_a_partial_final_warmup_step_still_belongs_to_warmup() -> None:
    schedule = _schedule(steps=2000, warmup_sequences=801, sequences_per_step=8)

    assert schedule.warmup_steps == 101


def test_the_curve_holds_the_peak_between_warmup_and_cooldown() -> None:
    schedule = _schedule()

    assert schedule.rate_at(1) == pytest.approx(0.01 / 100)
    assert schedule.rate_at(99) == pytest.approx(0.01 * 0.99)
    # Warmup ends at the peak rather than one step short of it.
    assert schedule.rate_at(100) == pytest.approx(0.01)
    assert schedule.rate_at(101) == pytest.approx(0.01)
    assert schedule.rate_at(800) == pytest.approx(0.01)


def test_the_cooldown_starts_at_the_peak_and_ends_just_above_zero() -> None:
    schedule = _schedule()

    assert schedule.cooldown_steps == 200
    assert schedule.rate_at(801) == pytest.approx(0.01)
    assert schedule.rate_at(802) == pytest.approx(0.01 * (1 - math.sqrt(1 / 200)))
    assert schedule.rate_at(900) == pytest.approx(0.01 * (1 - math.sqrt(99 / 200)))
    final = schedule.rate_at(1000)
    assert final == pytest.approx(0.01 * (1 - math.sqrt(199 / 200)))
    # Zero is reached at the horizon, so the last step still trains.
    assert 0.0 < final < 0.01 * 0.05


def test_the_curve_never_rises_once_it_has_reached_the_peak() -> None:
    schedule = _schedule()
    rates = [schedule.rate_at(step) for step in range(1, 1001)]

    assert rates == sorted(rates[:100]) + sorted(rates[100:], reverse=True)


def test_a_declared_run_without_warmup_or_cooldown_holds_one_rate() -> None:
    schedule = _schedule(warmup_sequences=0, cooldown_fraction=0.0)

    assert schedule.warmup_steps == 0
    assert schedule.cooldown_steps == 0
    assert [schedule.rate_at(step) for step in (1, 500, 1000)] == [0.01] * 3


def test_a_branch_cools_at_its_own_horizon_rather_than_the_trunk_s() -> None:
    trunk = _schedule(steps=2000)
    branch = _schedule(steps=1200)

    # One step sits in the trunk's constant stretch and inside the branch's
    # cooldown, which is the whole of what branching a horizon is.
    assert trunk.rate_at(1100) == pytest.approx(0.01)
    assert branch.rate_at(1100) < 0.01 * 0.5
    # The prefix the two share is the reason a branch is not a second run.
    assert branch.warmup_steps == trunk.warmup_steps


def test_a_warmup_past_the_range_the_rule_holds_over_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the range"):
        _schedule(steps=1000, warmup_sequences=1600, sequences_per_step=8)


def test_a_cooldown_that_rounds_away_on_a_short_horizon_is_refused() -> None:
    """A declared cooldown that decays nothing is the silent reshape the
    fraction was chosen to prevent, so it fails instead of holding the peak.
    """

    with pytest.raises(ValueError, match="decays nothing"):
        _schedule(steps=10, warmup_sequences=0, cooldown_fraction=0.04)


def test_a_warmup_and_cooldown_that_do_not_fit_the_horizon_are_refused() -> None:
    with pytest.raises(ValueError, match="do not fit"):
        _schedule(steps=1000, warmup_sequences=800, cooldown_fraction=0.95)
