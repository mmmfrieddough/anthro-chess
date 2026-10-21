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

| in the stack | held-out | under 1200 | 2000 and up | 1600 to 1999 | blitz dial | bullet dial |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| clock | -0.0259 | -0.0234 | -0.0264 | -0.0264 | +0.060 / +0.076 / +0.080 | +0.187 / +0.141 / +0.125 |
| balance | +0.0005 | -0.0032 | -0.0048 | +0.0027 | -0.004 / +0.006 / +0.000 | +0.011 / +0.026 / -0.001 |
| span | +0.0008 | +0.0006 | +0.0013 | +0.0009 | +0.001 / +0.002 / +0.016 | +0.009 / -0.026 / -0.003 |

The member responsible for the blitz shortfall is the balance: beside the
clock, its slope gain at blitz is gone. The clock and the balance both widen
the dial, and at blitz the clock already does what the balance did. What the
balance was adopted for survives in full: the rating tails gain about what
they gained alone, and the middle bands pay what they paid alone.

**The span shows nothing in the stack on one draw, and this reading cannot show
what it was adopted for.** `0093` adopted it because two draws agreed under it
where a draw change alone moved held-out loss by 0.0043; it claimed nothing
about the size of a single draw's improvement. Every arm here is draw 1. The
arm without it trains at a span of one, so its own reading carries that draw
noise, which is several times anything in the span's row.

## Why Nothing Is Dropped

A member is dropped where the stack shows it contributes nothing. The clock
carries almost all of the set's effect. The balance contributes its tail loss
gain, the reason `0092` adopted it, and at bullet adds to the dial. The span's
row is inside the draw noise of the arm it is read against, and its reason is a
property of two draws that a single-draw arm does not test; it costs no
throughput and nothing measured gets worse.

The set does not underperform its best member: held-out loss is better than
the clock alone's, the blitz dial is within a replicate deviation of it, and
the bullet dial is wider.

## What This Does Not Claim

**No seed floor qualifies the combined arm.** Its final gradient norm reads
0.159 against the replicates' 0.138 ± 0.006, as the span arm's did, so
`anthro eval report` declines the floor for it. The clock's own bullet
widening, read on an arm whose health matches the vehicle's, clears the seed
floor at all three temperatures.

**That the bullet super-additivity is real.** It is one arm against a sum of
four, and it clears the threshold at T=0 only.

**Which speed the dial should be tuned on.** The two read differently, and a
model's spacing at one says little about the other; nothing here says which a
player meets more.

## Consequences

The arm `#593` and `#499` read against is `arm-495-combined`. The target run
carries the three settings on top of whatever its own configuration decides.

A comparison that reads the dial now reads it at two speeds, and a
clock-reading arm can move them by different amounts.

## References

- `#495`, `#497`, `#498`, `#574`, `#583`, `#488`, `#593`, `#499`
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0092-the-rating-axis-is-balanced-by-thinning-at-four.md`
- `0093-the-target-run-draws-each-batch-across-sixteen-days.md`
- `0095-the-clock-enters-beside-the-rating-and-absence-is-trained.md`
- `0099-the-rating-benchmarks-declare-the-control-of-their-games.md`
- `0103-the-time-head-is-read-on-the-combined-arm.md`
- `docs/evaluation.md` (Rating Ladder)
