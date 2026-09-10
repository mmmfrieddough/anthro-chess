# 0088: The Horizon Has A Ceiling, And It Is Counted In Steps

Date: 2026-09-09

## Status

Accepted. Answers the horizon half of the binding-resource question that step 6
of the order in `docs/scaling.md` depends on.

Qualifies `0067-a-horizon-is-a-branch-not-a-restart.md`, whose consequence that
"the final run stays extendable" holds only up to the ceiling recorded here.

Rests on `0076-the-vehicle-is-width-128-at-the-target-regime.md` for the
configuration every arm copies, and on
`0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md` for the rate rule
and for the weight-decay reading this record puts a range on.

Bears on `0071-the-target-is-the-size-the-published-ladder-flattens-at.md`: the
target's horizon is unchanged, but the step count it implies sits past the
ceiling below.

## Context

Every checkpoint this project had read was stopped by a step bound while still
improving, so the milestone did not know whether capacity or training budget was
binding. `#181` answered that question at 276k parameters and its corpus,
checkpoints and results were all subsequently discarded.

The question was re-asked at the vehicle's width, by holding size fixed and
extending the horizon under the branched schedule `0067` fixed.

## What Was Measured

One trunk per learning rate at `model_dim` 128, the vehicle's configuration with
only `run_name`, `steps`, `checkpoint_every_steps` and `resume_from` changed, all
four outside `training_sha256`. Cooled endpoints branched at seven horizons.
Scored on the frozen pool at view `canonical`, 211,475 games and 14,161,038
positions.

Training loss is unavailable as the response here: `0087` establishes it is
comparable only between arms sharing a horizon, an interval spacing and a data
order, and a horizon sweep varies all three.

### The Curve Turns Over

`held_out.move_loss`, and `held_out.top1_accuracy` beside it:

| positions per parameter | 3e-3 loss | 3e-3 top1 | 1.5e-3 loss | 1.5e-3 top1 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 1.482278 | 0.524744 | 1.501655 | 0.520258 |
| 200 | 1.454458 | 0.531538 | 1.463581 | 0.529271 |
| 400 | 1.438981 | 0.535200 | 1.443493 | 0.534179 |
| 800 | 1.428087 | 0.537821 | 1.430039 | 0.537301 |
| 1600 | **1.420795** | **0.539619** | 1.421130 | 0.539515 |
| 3200 | 1.452536 | 0.537209 | **1.415871** | **0.540838** |
| 6400 | 1.476849 | 0.538283 | 1.483562 | 0.537213 |

**There is no plateau. The curve reaches a minimum and then degrades**, on both
metrics and at both rates. Reported as a turnaround rather than as a plateau,
which is what the acceptance criteria this reading was taken against require.

### The Rate Is Not The Cause

A second trunk at half the rate ran the same ladder. The gap closes
monotonically, 1.307, 0.627, 0.314, 0.137 and 0.024 percent, and at 1600 it is
below the pool's own bootstrap dispersion of 0.00058. That confirms `0087`'s
measured horizon null inside the range this sweep uses rather than transferring
it from width 32. Both arms then degrade.

Reading a single rate would have produced a confident wrong answer: the 3e-3 arm
alone reverses at 3200 and reads as an architecture meeting its capacity wall.

### The Degradation Is Miscalibration

Uncooled trunk checkpoints, which `0067` permits comparing since both sides sit
at the same rate:

| positions per parameter | steps | move loss | top1 accuracy |
| ---: | ---: | ---: | ---: |
| 1,280 | 111,104 | 1.441493 | 0.534348 |
| 1,919 | 166,656 | 1.437835 | 0.535380 |
| 2,559 | 222,208 | **1.437198** | 0.535229 |
| 3,199 | 277,760 | 1.439773 | 0.534837 |
| 3,999 | 347,200 | 1.467248 | 0.534615 |
| 5,998 | 520,800 | 1.469417 | 0.534992 |

**Loss degrades 1.94 percent while top-1 accuracy is flat.** The model ranks
moves as well as it ever did and assigns worse probabilities to them, which is
the signature of a distribution that has become too sharp. `legality.mask_penalty`
is unaffected.

### The Mechanism Is Unbounded Parameter Growth

Parameter L2 norm on the same checkpoints, and by group at the ends:

| positions per parameter | norm | growth |
| ---: | ---: | ---: |
| 80 | 155.11 | 1.00x |
| 1,280 | 939.70 | 6.06x |
| 2,559 | 1331.98 | 8.59x |
| 5,998 | 2024.15 | **13.05x** |

| group | growth |
| --- | ---: |
| output head | **16.4x** |
| feedforward and other | 13.9x |
| attention value and output | 12.8x |
| embeddings and bias | 6.1x |
| normalization gains | 2.0x |

The output head grows most, and it is the one group whose scale sets how sharp
the predicted distribution is, which is what ties the mechanism to the symptom.
`training_health` shows no instability at any cadence: `clip_rate` is 0
throughout and gradient norms rise only from 0.21 to 0.49.

`configs/training/ablation-vehicle.toml` sets `weight_decay = 0.0`, and its
stated reason is that nothing repeats at this horizon so decay has no
overfitting to prevent. That is correct about overfitting and does not address
what decay does over a long run, which is to bound the norm growth that
scale-invariant parameters accumulate by construction.

### Growth Is Counted In Steps, Not In Positions Per Parameter

A ladder at width 32 to 6400 positions per parameter shows **no turnaround**:
1.681462, 1.658529, 1.642317, still improving. Norm growth there reaches only
3.32x, because 6400 positions per parameter is 56,880 steps at that width
against 555,520 at width 128.

At matched step counts the two widths grow alike, 2.23x against 2.20x. **Norm
growth is a per-step phenomenon and is width independent.**

### The Data Axis, Ruled Out On Arithmetic

The corpus manifest reports 2,087,063,655 games and 138,692,878,042 plies, with
1,878,353,187 games in the train split, so roughly 1.25e11 training plies. The
longest rung here is 9.11e9 positions, which is **0.073 passes**. The target's
own horizon is 0.128 passes. Nothing repeats at any horizon this project trains
at, and no run was spent establishing it.

## Decision

**A run's useful horizon is bounded, and the bound is a step count rather than a
ratio of positions to parameters.** At this configuration the loss minimum sits
near 222,000 optimizer steps and the reading is clearly degraded by 347,000.

**A horizon is therefore stated in steps wherever this ceiling is the concern.**
Positions per parameter remains the right coordinate for the regime a model is
trained in, which is what `0076` matched the vehicle on. It is the wrong
coordinate for this failure, because two runs at one ratio and different widths
differ in steps by the ratio of their parameter counts.

**The vehicle cannot detect this and is not expected to.** At 800 positions per
parameter it runs 69,466 steps, a fifth of the way to the observed minimum. The
target at the same ratio runs 1,007,941 steps, which is 14.5 times the vehicle
and nearly three times past where width 128 was visibly degraded.

## What This Gives Up, Deliberately

**The cause is evidenced but not demonstrated.** Every symptom is consistent with
unbounded norm growth and the output head leads that growth, but no arm has been
run with decay enabled at a width and step count where the degradation appears.
A width-32 arm at 1x horizon bounded growth as predicted, 3.32x against 2.67x,
and changed loss by 0.06 percent, which is inside that width's dispersion and at
a step count far short of the failure. `#562` carries what would settle
it.

**One width, two rates.** The threshold is measured at `model_dim` 128 only.
Norm growth is width independent per step; whether the loss damage appears at the
same step count at 20.6M parameters is not established, and the direction of any
size dependence is unknown.

**The suite reading is partial.** Three checkpoints were scored through
`anthro eval suite`. `termination.resignation_calibration_error` degrades
eightfold from the peak to 6400, which is consistent with the sharpness
mechanism. The rating dial is near flat, `ladder.fitted_rating_slope` 0.337 to
0.312 with order accuracy 1.0 throughout. The rating-dependency family appears to
improve, but its members are divergence measures that a sharper policy inflates
mechanically, so they are not read as improvement here. Timing rows were taken
under contention and are not usable.

## Consequences

**Extending a finished run is bounded.** `0067` makes a horizon change a branch
rather than a restart, and that remains true. What this record adds is that the
gain goes negative past the ceiling, so the option is worth exercising only up to
it.

**The allocation ladder has to place its rungs against this.** At 800 positions
per parameter the ceiling is reached near 4.55e6 parameters, which is about width
240. A ladder spanning one to two decades of size at a fixed ratio puts its upper
rungs past that and its lower rungs far below it, which biases a fitted exponent
in the direction the size term is read from. `docs/research.md` (Resolving
Discrepancies In Compute-Optimal Scaling) records that no goodness-of-fit
statistic detects this class of error.

**The weight-decay rule is stated over the wrong invariant for this failure.**
`anthro_chess.training.scaling_rules` sets the timescale as a multiple of the
horizon, so the per-step decay rate is `1 / (horizons * steps)` and weakens as a
run lengthens. The equilibrium norm depends on the per-step rate, so a longer run
is given less of exactly what bounds it. A constant coefficient, which is what
published practice holds fixed, keeps that rate invariant to the horizon.

## References

- `0067-a-horizon-is-a-branch-not-a-restart.md`
- `0071-the-target-is-the-size-the-published-ladder-flattens-at.md`
- `0076-the-vehicle-is-width-128-at-the-target-regime.md`
- `0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md`
- `docs/scaling.md`: the program, and the order this reading sits in
- `#490`: the issue this answers
