# 0102: The Second-Moment Span Stays At The Vehicle's

Date: 2026-10-20

## Status

Accepted. Answers `#552`. Settles the span
`0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md` left open, at
the vehicle's scale.

## Context

At width 32 a shorter second-moment span read 1.0% better than the vehicle's on
training loss, and the span's optimum moved from 33 to 100 optimizer steps when
the horizon halved. `0087` held the vehicle's 0.999 because nothing there
identified a span to carry. What it could not say is whether any shorter span
helps at the vehicle, which is the only comparison a candidate is accepted on.

## What Was Measured

Two arms of the ablation vehicle at seed 17, each changing
`second_moment_decay` alone, at the vehicle's rate of 3e-3, read on the
checkpoint suite against the six `#488` replicates. 0.99 and 0.97 are the two
values that beat 0.999 at width 32 and 800 positions per parameter.

| metric | replicates | 0.99 (100 steps) | 0.97 (33 steps) |
| --- | ---: | ---: | ---: |
| end-of-run training loss | 1.43249 ± 0.00138 | 1.43247 (0.0) | 1.43130 (-0.9) |
| `held_out.move_loss` | 1.42915 ± 0.00094 | 1.42843 (-0.8) | 1.42907 (-0.1) |
| `held_out.top1_accuracy` | 0.53757 ± 0.00022 | 0.53775 (+0.8) | 0.53756 (-0.1) |
| `held_out.move_loss_ultrabullet` | 1.94927 ± 0.00064 | 1.95112 (+2.9) | 1.95422 (+7.7) |
| `puzzle.sampled_first_move_solve_rate` | 0.42884 ± 0.00103 | 0.43374 (+4.7) | 0.42536 (-3.4) |

Parenthesized figures are replicate deviations. The claim written before either
arm ran was that a shorter span lowers held-out loss by more than one replicate
deviation. Neither arm does, and the shorter span moves it least.

**The one consistent movement is against the change.** Ultrabullet loss rises
with a shorter span, monotonically across the two arms, and unclocked games
follow it at +1.6 and +2.5. The puzzle solve rate moves in opposite directions
in the two arms, which a span effect cannot do.

**Training health matches the vehicle's** apart from the final update-to-weight
ratio, which is lower in both arms; gradient norms and clip rates read the same.
The departure is the optimizer doing what a shorter average does, not
instability.

## Decision

**`second_moment_decay` stays at 0.999 and `SECOND_MOMENT_POSITIONS` stays the
vehicle's span.** Nothing here is a candidate for the combined arm.

## What This Does Not Claim

**That 0.999 is optimal.** The arms ran at the rate fitted for 0.999, and the
span is coupled to the rate, so a shorter span is discarded at that rate only.
A longer span than the vehicle's was not tried at this width.

**That the width-32 reading was wrong.** It stands at its own scale; it does not
reach the vehicle's.

## References

- `#552`, which holds every reading, and `#488`
- `0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md`
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
