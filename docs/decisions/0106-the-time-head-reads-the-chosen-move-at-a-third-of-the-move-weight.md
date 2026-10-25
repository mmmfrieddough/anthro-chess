# 0106: The Time Head Reads The Chosen Move At A Third Of The Move Weight

Date: 2026-10-25

## Status

Accepted. Answers `#593`. Gives `0003-action-conditioned-timing.md` its
representation, and takes the reading
`0103-the-time-head-is-read-on-the-combined-arm.md` arranged for it.

## Context

`0003` fixed the factorization: the action first, then the time from the same
context plus that action. `docs/engine-behavior.md` asks for a sampleable
distribution rather than an average. Neither says what the distribution is,
where the head reads the trunk, or how hard its loss pulls on it.

The field's released code is the nearest prior art. Maia-3's ponder head and
Allie's time head both read the trunk's final square tokens, pooled, through a
small two-layer readout, and write one scalar the move never reaches. Neither
publishes what the head costs move prediction, and no released Maia-3
checkpoint trains it.

## Decision

**The head reads where theirs does, and adds the two things this product
asks for.** It reads the final normalized square tokens, pooled, through a
two-layer readout. Beside the pool it reads the final tokens of the two
squares the chosen move joins, which is what conditions the time on the move.
It writes a mixture of eight logistics over `log(1 + ms)`, scored as the mass
of the centisecond interval the clock recorded rather than as a density, so a
run of exact zeros cannot drive the loss without bound.

**The time a move took is read from the clocks either side of it**, with the
increment the move earned. Lichess starts neither clock until each side has
moved once, so the first two plies carry no target, and neither does a
terminal action or an untimed game.

**The move and its time are weighed equally, and `model.time_loss_weight` is
0.3.** Under Adam the head's own parameters barely see the weight. What it sets
is how far timing bends the trunk the move is read from. Weighed equally, the
criterion is the likelihood of the move and its time together, and on it 1.0 and
0.3 read within 0.004 nats of each other on one draw each, nearer than either
side's seed noise can separate. At 1.0 the move pays a cost seven replicate
deviations wide; at 0.3 neither output pays one that can be measured. That is
the reason for 0.3: it is the weight at which both outputs hold.

**The target trains with the head.** `model.time_head = true` joins the settings
`0104` and `0105` give the target run.

## What Was Measured

Three arms, each the `#495` set with `model.time_head = true` at one weight,
each at full horizon on one card, scored on the full checkpoint suite and read
against `arm-495-combined`, which differs from them by the head alone. The
claim and the bounds were posted on `#593` before each pair ran. No seed floor
covers this comparison, so the vehicle's stored seed bounds were the
yardstick: 0.0021 on held-out move loss and 0.022 on the ladder slope.

| weight | `held_out.move_loss` | delta | `timing.move_time_loss` | marginal | margin | distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none (`arm-495-combined`) | 1.40237 | | | | | |
| 1.0 | 1.40884 | +0.0065 | 6.218 | 6.610 | -0.392 | 0.032 |
| 0.3 | 1.40287 | +0.0005 | 6.228 | 6.610 | -0.382 | 0.043 |
| 0.1 | 1.40142 | -0.0009 | 6.246 | 6.610 | -0.364 | 0.034 |

**At 1.0 the head costs move prediction.** The delta is three times the bound
and about seven replicate deviations. Every rating band and every timed speed
reads worse, and top-1 accuracy falls by 0.0015. The training move loss sits
0.004 to 0.006 above the control's from step 15,000 to the end, widening through
the cooldown, so this is a lasting cost at this size rather than slower
convergence.

**At 0.3 and 0.1 it does not.** Every rating band and top-1 accuracy read inside
the evaluation floor. `legality.mask_penalty` rises by about 5e-5 at both.

**The head reads more than the clock at every weight.** The marginal is the
human move-time histogram at matched rating band, speed, clock bucket and
phase, fitted on the scored decisions themselves. At 0.3 the head beats it by 0.24
nats in ultrabullet and by 0.33 to 0.43 in every other speed and clock slice.
Sampled times follow the human quartiles in every slice: at 0.3 the sampled
median is 0.74 s with under ten seconds left and 2.5 s with more than thirty,
against human medians of 0.75 s and 2.6 s.

**The dial holds in five of six cells.** `ladder.fitted_rating_slope` at 1+0
T=0 reads 0.447 at 0.3 and 0.448 at 0.1, against 0.475, which is just past the
bound in the narrower direction. The 1.0 arm read 0.459 there. That is the cell
`0104` found the combined arm reading above the sum of its members, at T=0
only. The other cells read within the bound at both weights.

**Weighed together, 1.0 and 0.3 tie.** Per held-out decision, counting the
timing loss on the 96% of decisions that carry a time:

| weight | move | timing | together |
| --- | ---: | ---: | ---: |
| 1.0 | 1.40884 | 6.218 | 7.399 |
| 0.3 | 1.40287 | 6.228 | 7.403 |
| 0.1 | 1.40142 | 6.246 | 7.418 |

**The head reads the chosen move.** On 126k timed decisions from 2,000 held-out
games, the 0.3 head scores the human's time at 6.251 given the human's move,
6.457 given the policy's likeliest other legal move, 6.478 given the second
likeliest, and 6.580 given a random legal one, against a marginal of 6.595. Told
the wrong move, even a plausible one, it keeps under half its margin.

Throughput is unchanged: about 51k positions a second on one card, the
combined arm's rate per card.

## What This Does Not Claim

**That the cost at 1.0 holds at the target's width.** A wider trunk may have
room for both. It is not read here, and 0.3 is adopted on the vehicle's
reading alone.

**That a frozen-trunk head would be worse.** Stopping the timing gradient at the
trunk costs no move prediction by construction. 0.1 approaches it and loses 7%
of the 1.0 arm's margin; the endpoint itself is not read.

**Anything about promotion choice.** A queen and a knight promotion join the
same two squares and get the same distribution.

**Anything about the interaction with `0105`.** The arms predate query-key
normalization and the log-partition penalty, which the target also carries.

## Consequences

`#596` re-checks the target's learning rate with the timing loss on, at the
target's width. A move-loss cost there would show beside the rate, and the
weight is re-read if it does.

`#594` and `#597` sample from the distribution through the model's time head.
`MoveTimeDistribution.sample` returns the model's time to the millisecond. Only
the benchmark puts a sample on the clock's centisecond grid, to compare it with
recorded times.

The three arms are retained: `arm-593-time-head`, `arm-593-time-weight-0p3`
and `arm-593-time-weight-0p1`.

## References

- `#593`, `#495`, `#596`, `#594`, `#597`
- `0003-action-conditioned-timing.md`
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0096-the-move-head-already-names-the-parts-of-a-move.md`
- `0097-a-value-head-does-not-improve-move-prediction.md`
- `0103-the-time-head-is-read-on-the-combined-arm.md`
- `0104-the-accepted-set-composes-and-the-ladder-reads-two-speeds.md`
- `0105-the-target-trains-with-query-key-normalization-and-a-log-partition-penalty.md`
- `docs/engine-behavior.md` (Optional Timing Behavior)
- `docs/evaluation.md` (Timing Evaluation)
- `docs/research.md` (Chessformer, Allie)
