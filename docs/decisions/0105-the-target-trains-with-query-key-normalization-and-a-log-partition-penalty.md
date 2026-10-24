# 0105: The Target Trains With Query-Key Normalization And A Log-Partition Penalty

Date: 2026-10-24

## Status

Accepted. Answers `#499`. Adds two settings to the set
`0104-the-accepted-set-composes-and-the-ladder-reads-two-speeds.md` adopted, for
the target run only, and takes the one vehicle-scale reading
`0103-the-time-head-is-read-on-the-combined-arm.md` arranged for it.

Bears on `0088-the-horizon-has-a-ceiling-and-it-is-counted-in-steps.md`: the
turnover it found coincides with a log-partition drift that costs bfloat16
logits their precision, and a trunk with the penalty does not turn over through
step 347,200. Bears on
`0089-bounded-growth-removes-the-ceiling-and-costs-a-flat-offset.md` for the
same reason: decay is not the only setting that removes the ceiling.

## Context

A stability setting does nothing visible at the scale it is measured on and
prevents a specific failure at a larger one, so a loss comparison cannot accept
it. It is accepted here only where this project can reach the failure it
prevents, and validated by widening the learning rate rather than the model.
Neither validation touches the ablation vehicle: both members change
`training_sha256`, and `0065` is explicit that an edit to the vehicle retires
every reading taken against it.

## Decision

**Two members are adopted, each for a failure measured here.**

- **Query-key normalization**, `model.query_key_normalization = true`: each
  head's queries and keys are RMS-normalized before their dot product. It
  prevents attention-logit growth, which at width 512 collapses the first
  layer's attention and costs the run a loss spike of several nats, at the
  target's own rate.
- **A log-partition penalty**, `z_loss = 1e-4`: the squared log-partition of the
  action logits is added to the objective. It prevents the shared logit offset,
  which the cross-entropy cannot see, from drifting to magnitudes where
  bfloat16 logits quantize. Without it the offset passes -27 by step 166,656
  at width 128, and the target runs about a million steps. With it a width-128
  trunk held the offset near -2.7 and was still improving at step 347,200,
  where the same trunk without it had turned over.

**Three are not.**

- **Normalization placement.** `0101` read post-norm blocks against pre-norm
  ones: post-norm diverged at the vehicle's rate, and every rate rule here is
  fitted for pre-norm. The blocks stay pre-norm.
- **Excluding embeddings from weight decay.** No run here trains with decay
  (`0089`), so the exclusion would change nothing.
- **A deliberate initialization scale.** The failure the range test finds
  starts hundreds of steps after warmup, from query and key projections that
  have grown away from their initial values, and normalizing them removes it.
  At the target's rate warmup is clean in every arm, so nothing observable here
  is left for an initialization change to prevent.

Gradient clipping is `#493`'s, ahead of the vehicle's freeze.

## What Was Measured

### The Mechanism

Width 512 under the target's recipe (the vehicle's shape rules, `0104`'s three
settings, the rate rule's 6.89e-4 and the warmup `0090` used), at twice the
rule's rate, with a checkpoint every hundred steps. Each was read on one fixed
held-out batch. The first layer's attention, with the run's loss:

| step | loss | max attention logit | attention entropy, nats | rows above 0.99 on one square | query norm | residual RMS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 800 | 1.75 | 6.5 | 3.27 | 0% | 3.8 | 0.85 |
| 900 | 4.82 | 286.9 | 0.77 | 29% | 25.6 | 52.4 |
| 2000 | 2.02 | 671.2 | 1.17 | 13% | 16.6 | 12.8 |

Within a hundred steps the first layer's queries and keys grew sixfold, its
logits saturated the softmax, and the residual stream it writes to grew sixty
times. The geometric bias added to those logits stayed below 14 throughout, so
the growth is in the dot product. With query-key normalization the largest
attention logit in any layer stays between 4.6 and 6.4 over the same
3,000 steps, the first layer's entropy stays near 3.0, and no row saturates.

### The Range Test

The same configuration, at each rate for the first 3,000 steps of the
63,000-step schedule `0090`'s trunks declared. Worst rise is the largest
interval loss above the run's best so far after warmup; the tail is the mean
interval loss over steps 2,850 to 3,000.

| rate | arm | worst rise | gradient maximum | loss at 1,000 | tail |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1x | baseline | 2.99 | 25 | 4.74 | 1.560 |
| 1x | package | 0.07 | 15 | 1.77 | 1.542 |
| 2x | baseline | 5.55 | 50 | 3.91 | 1.696 |
| 2x | query-key normalization alone | 0.07 | 13 | 1.68 | 1.520 |
| 2x | package | 0.21 | 18 | 1.67 | 1.522 |
| 4x | baseline | 3.98 | 142 | 4.47 | 1.789 |
| 4x | package | 0.06 | 11 | 1.64 | 1.517 |
| 8x | baseline | stalled near 3.5 through warmup | 114 | 2.24 | 1.617 |
| 8x | package | 1.5, at step 200 in warmup | 204 | 2.12 | 1.578 |

**The ceiling rises from below the target's rate to between four and eight
times it.** Without the package every rate tried spikes, the target's own
included. With it 1x, 2x and 4x train without a spike, and 8x fails early in
warmup. Query-key normalization alone removes the spike at 2x, so it is the
member that does.

The tails also order by rate, best at 4x. A 3,000-step reading favours a higher
rate than a full run does (`0087`), so this says nothing about the target's
rate.

Query-key normalization costs about 2% of training throughput at width 512.

### The Log-Partition Drift

The mean log-partition of the action logits on a fixed held-out batch, and what
computing the logits in bfloat16 costs that batch's loss, on the width-128
trunks `0088` read and the decayed trunks `0089` read:

| run | step | mean log-partition | bfloat16 cost to loss |
| --- | ---: | ---: | ---: |
| 3e-3, no decay | 111,104 | +1.6 | -0.0002 |
| 3e-3, no decay | 166,656 | -26.8 | +0.0040 |
| 3e-3, no decay | 222,208 | -95.7 | +0.0273 |
| 3e-3, no decay | 277,760 | -241.6 | +0.0800 |
| 3e-3, no decay | 520,800 | -262.3 | +0.1112 |
| 1.5e-3, no decay | 534,688 | -312.2 | +0.1210 |
| 3e-3, grouped decay | 347,200 | -38.1 | -0.0010 |
| 1.5e-3, grouped decay | 534,688 | -37.0 | +0.0030 |
| width 512, 2x, no decay | 63,000 | +3.5 | -0.0002 |

The drift is per step, at either rate. Training reads the bfloat16 loss, and
the undecayed trunk's logged training loss rose by about 0.1 after step 223,000,
the size of the cost in the last column.

### The Penalty Over A Long Trunk

`h499-zloss-trunk-0p003` is `0088`'s 3e-3 trunk, the vehicle's configuration
declaring 2,222,080 steps, with `z_loss = 1e-4` and nothing else changed. It was
stopped at step 347,200, the first point at which that trunk was clearly
degraded. Pool loss is `held_out.move_loss` at view `canonical`, uncooled;
log-partition and training loss are as above, the training loss and gradient
norm averaged over the last three logged intervals.

| step | pool loss, without | pool loss, with | log-partition, without | log-partition, with | training loss, without | training loss, with | gradient norm, without | gradient norm, with |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 111,104 | 1.4415 | 1.4423 | +1.6 | -2.4 | | | | |
| 166,656 | 1.4378 | 1.4386 | -26.8 | -2.7 | 1.437 | 1.437 | 0.27 | 0.27 |
| 222,208 | 1.4372 | 1.4351 | -95.7 | -2.7 | 1.462 | 1.438 | 0.41 | 0.26 |
| 277,760 | 1.4398 | 1.4363 | -241.6 | -2.6 | 1.502 | 1.423 | 0.46 | 0.26 |
| 347,200 | 1.4672 | 1.4296 | -262.7 | -2.8 | 1.595 | 1.459 | 0.54 | 0.25 |

**With the penalty the trunk does not turn over.** The two agree to within
0.001 while the offset is small, separate as it drifts, and by step 347,200 the
trunk with the penalty is 0.038 lower on the pool and still improving, below the
best point the trunk without it reached. Its gradient norm stays flat where the
other's doubles. Top-1 accuracy there is 0.5372 against 0.5346.

### The Package On The Combined Arm

`arm-499-package` is `arm-495-combined` with both members on, at seed 17 and
full horizon, scored on one sweep. Deltas are in the six `#488` replicates'
standard deviations. The claim posted before it ran was that nothing would move
by more than about four.

| reading | `arm-495-combined` | `arm-499-package` | delta |
| --- | ---: | ---: | ---: |
| `held_out.move_loss` | 1.4024 | 1.4006 | -1.9 |
| under 1200 | 1.5907 | 1.5889 | -3.7 |
| 1200 to 1599 | 1.4509 | 1.4497 | -1.5 |
| 1600 to 1999 | 1.3616 | 1.3596 | -1.8 |
| 2000 and up | 1.2943 | 1.2919 | -2.0 |
| bullet dial slope, T=0 / 0.7 / 1 | 0.475 / 0.423 / 0.411 | 0.476 / 0.425 / 0.400 | +0.1 / +0.4 / -1.0 |
| blitz dial slope, T=0 / 0.7 / 1 | 0.357 / 0.385 / 0.393 | 0.356 / 0.374 / 0.364 | -0.1 / -1.1 / -3.4 |
| `puzzle.sampled_first_move_solve_rate` | 0.4301 | 0.4329 | +2.7 |
| final gradient norm | 0.159 | 0.144 | |
| mean log-partition | +7.2 | -2.6 | |

**Quality agrees.** Held-out loss is slightly better in every band, and no dial
reading moves past the threshold. The largest, blitz at T=1, is 3.4 deviations
lower. The final gradient norm sits nearer the replicates' 0.138 ± 0.006 than
the combined arm's does.

Two readings moved beyond their own noise and are recorded as costs rather than
as quality. Serving forward throughput fell from 51,185 to 44,013 decisions a
second, and move latency rose from 5.1 to 5.4 ms. With the rating absent, play
measured on the ladder is 93 to 150 Elo weaker on every series, and
`dependency.rating_absent_degradation` rose from 0.085 to 0.117.

## What This Does Not Claim

**That the target will not turn over.** The long trunk is one run at width 128
against one control, trained at a different code revision, and it stops at step
347,200. `0088` traced the turnover to growth that sharpens the distribution;
the trunk with the penalty shows that drift and quantization drive most of it at
this width, not that nothing else can. `#491` still reads held-out loss along
its trunk.

**That no spike can occur later in a run.** The range test reads the first
3,000 steps, where every spike it found happened.

**That the target's rate should move.** See the range test.

## Consequences

The target run carries `model.query_key_normalization = true` and
`z_loss = 1e-4` on top of `0104`'s three settings. Both change what it trains,
so `#596`'s rate bracket is taken with them on. The vehicle is unchanged and
keeps its digest; both switches are left out of a recorded identity while off.

`#569`, the batch that keeps the target under a turnover, is needed only if
`#491` sees one with these settings on.

`arm-499-package` and `h499-zloss-trunk-0p003` stay on the machine that trained
them. The range and diagnostic runs are scratch and are not retained. Their
configuration is the vehicle's file with the width-512 overrides `0090`
records, `0104`'s three settings, `steps=63000`, and the rate.

## References

- `#499`, `#495`, `#488`, `#493`, `#596`, `#491`, `#494`
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md`
- `0089-bounded-growth-removes-the-ceiling-and-costs-a-flat-offset.md`
- `0090-the-size-is-not-fitted-here-and-the-rate-is-measured-at-the-target.md`
- `0101-the-rating-conditioning-and-pre-norm-blocks-stay.md`
