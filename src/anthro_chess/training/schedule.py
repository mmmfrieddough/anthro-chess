"""The learning-rate curve one run follows, recomputed from the step it is on.

A run warms up, holds the peak rate through a constant trunk, and cools to zero
on a square-root curve over the final fraction of its horizon. The trunk does
not depend on the horizon, so a shorter or longer run is a cooldown branched off
a trunk checkpoint rather than a run from initialization;
`docs/decisions/0067-a-horizon-is-a-branch-not-a-restart.md` owns why the family
is this one.

Nothing here is state. A rate follows from the step and from the horizon the
running configuration declares, which is what makes a branch cool at its own
boundary instead of at the trunk's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: The share of a run its warmup may occupy before the rule that set the warmup
#: stops describing anything. Past it a run spends enough of itself warming up
#: that its readings are partly about the warmup, which is one of the confounds
#: the scaling program exists to avoid. A short branch is where that bites, so a
#: configuration past this is refused rather than quietly reshaped.
MAXIMUM_WARMUP_FRACTION = 0.1


@dataclass(frozen=True)
class LearningRateSchedule:
    """The rate each optimizer step of one run is taken at."""

    peak: float
    warmup_steps: int
    cooldown_steps: int
    steps: int

    def rate_at(self, global_step: int) -> float:
        """Return the rate for a one-based optimizer step of this run."""

        if global_step <= self.warmup_steps:
            return self.peak * global_step / self.warmup_steps
        trunk_steps = self.steps - self.cooldown_steps
        if global_step <= trunk_steps:
            return self.peak
        # The curve reaches zero at the horizon rather than at the last step, so
        # no step of a run is spent at a rate that cannot move the weights.
        cooled = (global_step - trunk_steps - 1) / self.cooldown_steps
        return self.peak * (1.0 - math.sqrt(cooled))


def resolve_schedule(
    *,
    peak: float,
    steps: int,
    warmup_sequences: int,
    cooldown_fraction: float,
    sequences_per_step: int,
) -> LearningRateSchedule:
    """Convert one run's declared schedule into the step counts it applies.

    Warmup arrives as a quantity of training data, so a branch and the trunk it
    resumes warm up over the same prefix whatever horizon either declares. The
    conversion is exact before a run starts because the loader delivers whole
    sequences at a rate the batch and accumulation fix.
    """

    warmup_steps = -(-warmup_sequences // sequences_per_step)
    cooldown_steps = round(cooldown_fraction * steps)
    if warmup_steps > MAXIMUM_WARMUP_FRACTION * steps:
        raise ValueError(
            f"warmup of {warmup_sequences} sequence(s) is {warmup_steps} step(s) "
            f"at {sequences_per_step} sequence(s) per step, more than "
            f"{MAXIMUM_WARMUP_FRACTION:.0%} of the {steps}-step horizon and "
            f"outside the range the warmup rule holds over"
        )
    if warmup_steps + cooldown_steps > steps:
        raise ValueError(
            f"warmup of {warmup_steps} step(s) and cooldown of "
            f"{cooldown_steps} step(s) do not fit in a {steps}-step horizon"
        )
    return LearningRateSchedule(
        peak=peak,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
        steps=steps,
    )


__all__ = [
    "MAXIMUM_WARMUP_FRACTION",
    "LearningRateSchedule",
    "resolve_schedule",
]
