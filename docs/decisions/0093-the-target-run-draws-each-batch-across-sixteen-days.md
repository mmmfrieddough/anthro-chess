# 0093: The Target Run Draws Each Batch Across Sixteen Days

Date: 2026-10-07

## Status

Accepted. Answers `#574`. Refines
`0092-the-rating-axis-is-balanced-by-thinning-at-four.md`, which found that a
loader-seed change alone moves held-out loss by several times the replicates'
spread and left that noise to `#574`.

## Context

The shard-backed loader shuffles row groups, then the games inside each one, and
a batch never left its row group. In the widened corpus a row group is a whole
shard of 50,000 games from one day, so training stayed on one day's rating and
speed mix for about 200 optimizer steps at a time. Shards differ: across 30
sampled ones the mean rating has a standard deviation of 34 and the ultrabullet
share swings between 0.7% and 3.0% within a single day.

`#498` then trained the vehicle with only the loader seed changed. Held-out loss
moved by +0.0043, 4.5 times the six replicates' spread, and every rating band
moved together.

## Decision

**The target run sets `interleaved_row_groups = 16`.** Sixteen consecutive row
groups of the epoch order are shuffled together, so every batch draws from
sixteen days chosen at random across the corpus. It reaches the target through
`#495`, the combined arm, like every other accepted candidate.

**The vehicle keeps a span of one.** Moving it would change its order, and with
that its stored seed dispersion and every comparison read against it.

Sixteen was chosen rather than swept. It holds about 440 MB of row groups, and
it divides the spread of a span's mean rating by four. No reading below says a
different width would do better or worse.

## What Was Measured

Two vehicle arms at seed 17, full horizon, scored on the checkpoint suite at a
span of sixteen. Draw 1 is the vehicle's loader seed, read against the mean and
standard deviation of the six `#488` replicates that share it. Draw 2 is the
second loader seed, read against `arm-498-control-draw2`, the vehicle trained on
it. Held-out loss, with lower better:

| | replicates, draw 1 | span 16, draw 1 | vehicle, draw 2 | span 16, draw 2 |
| --- | ---: | ---: | ---: | ---: |
| `held_out.move_loss` | 1.42915 ± 0.00094 | 1.42788 | 1.43342 | 1.42822 |
| below 1200 | 1.61807 ± 0.00051 | 1.61693 | 1.62176 | 1.61764 |
| 2000 and up | 1.32683 ± 0.00118 | 1.32553 | 1.33257 | 1.32566 |
| opening | 1.51430 ± 0.00017 | 1.51323 | 1.51708 | 1.51378 |
| ultrabullet slice | 1.94927 ± 0.00064 | 1.96588 | 1.98479 | 1.98032 |

`ladder.fitted_rating_slope`:

| | replicates | span 16, draw 1 | vehicle, draw 2 | span 16, draw 2 |
| --- | ---: | ---: | ---: | ---: |
| T=0 | 0.298 ± 0.017 | 0.294 | 0.297 | 0.289 |
| T=0.7 | 0.303 ± 0.010 | 0.311 | 0.283 | 0.306 |
| T=1.0 | 0.303 ± 0.008 | 0.305 | 0.282 | 0.306 |

**The two draws stop disagreeing.** Without interleaving they differ by 0.0043
in held-out loss; with it, by 0.0003. Both interleaved arms sit at or below the
best replicate, and the slope the second draw lost at T=0.7 and T=1.0 comes
back. So most of what a loader-seed change did was which run of single days
happened to fall under the cooldown, not a property of the draw.

**Nothing measured costs.** Held-out loss improves by about one replicate
deviation on draw 1 and the dial holds inside the replicates' spread.
`ladder.temperature_strength_response` reads -305.9 and -310.7 against
-307.9 ± 4.1.

**It costs no throughput.** On the vehicle, steps 2,000 to 4,000, which
include a span boundary, run at 53,600 active positions per second at a span of
sixteen against 53,772 at one on the same card, and 54,234 against 54,390 on the
other. The two arms above ran on an earlier gather that took a slice per row
group and paid about 2.5%; the order is the same under both.

**The early loss spikes are not the cause.** Draw 2 spiked at step 4,000 without
interleaving and at step 1,000 with it. The largest interval gradient norm reads
9.6 on draw 1 against 8.5 to 13.9 across the replicates, and 36.0 on draw 2
against its control's 45.7.

## What This Does Not Claim

**No seed floor qualifies it.** The final gradient norm reads 0.155 on draw 1,
2.9 replicate deviations above the replicates, and the update-to-weight ratio
5.1 below, so `anthro eval report` declines the floor. The claim rests on two
draws agreeing where they did not before, and on nothing getting worse, not on
the size of the draw-1 improvement.

**The ultrabullet slice still moves with the draw.** It reads 1.966 and 1.980
under interleaving against the replicates' 1.949. The replicates' value looks
like a property of their one draw rather than something interleaving gives up.

**One arm per draw.** Two draws agreeing is evidence against draw noise being
large under interleaving, not a measurement of how large it is.

## Consequences

A comparison between arms that change which games are read, such as a selection
or a batch shape, carries most of the draw noise `0092` measured as long as it
runs at a span of one, which every vehicle arm does. Reading such a change at a
span of sixteen on both sides should carry much less of it.

## Alternatives Considered

**A global shuffle.** It costs a seek per example over a corpus of two billion
games, which is why the loader shuffles row groups at all.

**Re-preparing the corpus with shards that mix days.** The same effect, paid as a
rewrite of 41,763 shards and a new corpus identity, where a loader setting costs
neither.

## References

- `#574`, `#498`, `#488`, `#495`, `#54`
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0092-the-rating-axis-is-balanced-by-thinning-at-four.md`
- `docs/data.md` (the shard-backed loader)
