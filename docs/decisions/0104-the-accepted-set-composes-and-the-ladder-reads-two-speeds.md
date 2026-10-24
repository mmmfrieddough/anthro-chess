# 0104: The Accepted Set Composes, And The Ladder Reads Two Speeds

Date: 2026-10-21

## Status

Accepted. Answers `#495`. Pays the price
`0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md` named
for reading candidates against a frozen base, for the set
`0092-the-rating-axis-is-balanced-by-thinning-at-four.md`,
`0093-the-target-run-draws-each-batch-across-sixteen-days.md` and
`0095-the-clock-enters-beside-the-rating-and-absence-is-trained.md` accepted.
Settles the second speed
`0099-the-rating-benchmarks-declare-the-control-of-their-games.md` left to a
clock-reading model's adoption.

`0105-the-target-trains-with-query-key-normalization-and-a-log-partition-penalty.md` adds query-key normalization and a log-partition penalty to this set
for the target run.

## Context

Each candidate was read alone against the vehicle, which gives main effects and
no interaction terms. The set is three settings: `model.clock_inputs`, a
`rating_balance` of four, and `interleaved_row_groups` of sixteen. No other
candidate this milestone adopted a change.

The claim was posted on `#495` before the arm ran: held-out loss would compose,
the dial read at a declared control would not, and a shortfall against the sum
beyond about four replicate deviations would trigger a leave-one-out pass.

## Decision

**All three settings are adopted together.** `arm-495-combined` is the arm that
carries them, and its records are the first promoted to the committed store.
The time head and the stability package are read against it, per
`0103-the-time-head-is-read-on-the-combined-arm.md`.

**The checkpoint sweep reads the ladder at bullet as well as blitz.** The
bullet step shows every seat 1+0, 72% of the pool's games rated in the bullet
pool, over bullet openings, and starts its own series. It costs about six
minutes on a sweep of about an hour. The vehicle's stored seed dispersion was
regenerated with it, so its series carry a floor.

## What Was Measured

Every arm is the vehicle at seed 17 on its own loader seed, at full horizon,
scored on one sweep. Deltas are against the mean of the six `#488` replicates.
The clock's is the mean of its two seeds. "Short" is the combined delta minus
the sum of the members' deltas, in replicate standard deviations.

**Held-out move loss composes.**

| | clock | balance | span | sum | combined | short |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | -0.0261 | -0.0009 | -0.0013 | -0.0283 | -0.0268 | +1.6 |
| under 1200 | -0.0233 | -0.0045 | -0.0011 | -0.0289 | -0.0274 | +3.1 |
| 1200 to 1599 | -0.0260 | +0.0005 | -0.0013 | -0.0267 | -0.0254 | +1.6 |
| 1600 to 1999 | -0.0268 | +0.0010 | -0.0013 | -0.0271 | -0.0254 | +1.6 |
| 2000 and up | -0.0262 | -0.0063 | -0.0013 | -0.0338 | -0.0326 | +1.0 |

The combined arm reads better than the clock alone, its best member, in every
band.

**The dial does not compose at blitz.** `ladder.fitted_rating_slope` at T=0 /
0.7 / 1.0 (the balance and span arms read no clock, so their declared reading
is their untimed one):

| | clock | balance | span | sum | combined | short |
| --- | --- | --- | --- | --- | --- | --- |
| 3+0 blitz | +0.078 / +0.082 / +0.086 | +0.030 / +0.038 / +0.027 | -0.004 / +0.008 / +0.002 | +0.104 / +0.127 / +0.115 | +0.059 / +0.081 / +0.089 | -2.6 / -4.7 / -3.1 |
| 1+0 bullet | +0.141 / +0.115 / +0.108 | -0.002 / +0.026 / +0.027 | -0.010 / -0.013 / +0.002 | +0.129 / +0.128 / +0.136 | +0.184 / +0.133 / +0.128 | +4.9 / +1.1 / -0.7 |

At blitz the combined arm reads where the clock arm reads alone, and the
shortfall at T=0.7 passed the threshold. At bullet the set reads above the
clock arm at every temperature and above the sum at T=0. Untimed, an
unrecorded reading of the combined arm puts its shortfall at +0.2 / -3.2 / -2.5.

**Leave-one-out.** Each member's contribution in the stack is the combined arm
minus the arm without it. Dropping the span is `arm-497-clock-balance4`, which
`#497` already ran; `arm-495-loo-no-balance` and `arm-495-loo-no-clock` are the
other two.

| in the stack | held-out | under 1200 | 1200 to 1599 | 1600 to 1999 | 2000 and up | blitz dial | bullet dial |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| clock | -0.0259 | -0.0234 | -0.0257 | -0.0264 | -0.0264 | +0.060 / +0.076 / +0.080 | +0.187 / +0.141 / +0.125 |
| balance | +0.0005 | -0.0032 | +0.0016 | +0.0027 | -0.0048 | -0.004 / +0.006 / +0.000 | +0.011 / +0.026 / -0.001 |
| span | +0.0008 | +0.0006 | +0.0004 | +0.0009 | +0.0013 | +0.001 / +0.002 / +0.016 | +0.009 / -0.026 / -0.003 |

The member responsible for the blitz shortfall is the balance: beside the
clock, its slope gain at blitz is gone. The clock and the balance both widen
the dial, and at blitz the clock already does what the balance did. What the
balance was adopted for survives in part: the rating tails keep about three quarters of
the gain they had alone. The middle bands pay more than they did alone, +0.0016
and +0.0027 against +0.0005 and +0.0010, so overall held-out loss moves from a
gain of 0.0009 alone to a cost of 0.0005 in the stack.

**The span reads slightly worse in the stack on one draw, and this reading
cannot show what it was adopted for.** `0093` adopted it because two draws
agreed under it where a draw change alone moved held-out loss by 0.0043; it
claimed nothing about the size of a single draw's improvement. Every arm here
is draw 1. The arm without it trains at a span of one, so its held-out reading
carries that draw noise, several times the span's held-out row. How far a draw
change moves the dial was never measured, so the span's bullet row at T=0.7,
about five replicate deviations, has no such account.

## Why Nothing Is Dropped

A member is dropped where the stack shows it contributes nothing. The clock
carries almost all of the set's effect. The balance still contributes most of
its tail loss gain, the reason `0092` adopted it, and at bullet adds to the
dial; it costs the middle bands more than it did alone, which is the trade
`0092` accepted, now larger. The span's held-out row is inside the draw noise
of the arm it is read against, and its reason is a property of two draws that a
single-draw arm does not test. It costs no throughput.

The set does not underperform its best member beyond noise: held-out loss is
better than the clock alone's, the blitz dial is 1.1 replicate deviations below
it at T=0 and level above, and the bullet dial is wider.

## What This Does Not Claim

**No seed floor qualifies the combined arm.** Its final gradient norm reads
0.159 against the replicates' 0.138 ± 0.006, as the span arm's did, so
`anthro eval report` declines the floor for it. The clock's own bullet
widening, read on an arm whose health matches the vehicle's, clears the seed
floor at all three temperatures.

**That the bullet super-additivity is real.** It is one arm against a sum of
four, and it clears the threshold at T=0 only.

**Which speed the dial should be tuned on.** Nothing here says which a player
meets more.

## Consequences

The target run carries the three settings on top of whatever its own
configuration decides.

## References

- `#495`, `#497`, `#498`, `#574`, `#583`, `#488`, `#593`, `#499`
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0092-the-rating-axis-is-balanced-by-thinning-at-four.md`
- `0093-the-target-run-draws-each-batch-across-sixteen-days.md`
- `0095-the-clock-enters-beside-the-rating-and-absence-is-trained.md`
- `0099-the-rating-benchmarks-declare-the-control-of-their-games.md`
- `0103-the-time-head-is-read-on-the-combined-arm.md`
- `docs/evaluation.md` (Rating Ladder)
