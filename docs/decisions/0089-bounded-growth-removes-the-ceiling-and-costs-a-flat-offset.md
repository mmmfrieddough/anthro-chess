# 0089: Bounded Growth Removes The Ceiling, And Costs A Flat Offset

Date: 2026-09-21

## Status

Accepted. Demonstrates the mechanism
`0088-the-horizon-has-a-ceiling-and-it-is-counted-in-steps.md` evidenced but
could not show, and replaces the weight-decay rule
`0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md` recorded as
unresolved.

Bears on `0071-the-target-is-the-size-the-published-ladder-flattens-at.md`: the
target's horizon sits past the ceiling `0088` measured, and what follows is the
condition under which that stops being a defect.

Leaves `0076-the-vehicle-is-width-128-at-the-target-regime.md` untouched on
purpose. The vehicle does not adopt this, for the reason under **The Vehicle
Does Not Adopt It** below.

## Context

`0088` found that a width-128 run reaches its loss minimum near 222,000
optimizer steps and is clearly degraded by 347,000, on held-out loss and top-1
accuracy alike, at two learning rates. It traced the symptom to unbounded
parameter growth under `weight_decay = 0.0`: the norm grew 13.05x over 520,800
steps with no saturation, the output head grew most, and the damage was
miscalibration rather than lost ranking.

Every part of that was consistent with the mechanism and none of it demonstrated
the mechanism, because no arm had been run with decay enabled at a width and a
step count where the degradation exists. The one decay arm that had been run
reached 56,880 steps at width 32, far short of the failure.

## What Was Measured

Two arms at `model_dim` 128, each copying the control `h490-trunk-0p003` and
differing from it in `weight_decay` alone. All three share a seed, a data order,
a declared horizon of 2,222,080 steps and a constant 3e-3 trunk rate, so no arm
enters its cooldown and every checkpoint compared sits at the same rate, which
is the comparison `0067` permits.

```console
uv run anthro train --config configs/training/ablation-vehicle.toml \
  --set 'run_name="h562-wd-w128-0p01"' --set steps=2222080 \
  --set checkpoint_every_steps=6944 --set weight_decay=0.01

uv run anthro train --config configs/training/ablation-vehicle.toml \
  --set 'run_name="h562-wd-w128-0p1"' --set steps=2222080 \
  --set checkpoint_every_steps=6944 --set weight_decay=0.1
```

Both arms ran to 354,144 steps. Decoupled decay shrinks a weight by
`learning_rate * weight_decay` each step, so what the two coefficients set is a
decay timescale of 33,333 and 3,333 optimizer steps. Checkpoints were scored on
the frozen pool at view `canonical`, 211,475 games and 14,161,038 positions,
which is the instrument `0088` read.

Both arms carry a different `training_sha256` from the control, correctly, since
`weight_decay` is inside the digest. **No seed floor applies to anything below.**

### The Norm Is Bounded

Parameter L2 norm, read from the checkpoints directly:

| step | control | 33,333-step | 3,333-step |
| ---: | ---: | ---: | ---: |
| 6,944 | 155.11 | 142.19 | 96.49 |
| 111,104 | 939.70 | 411.30 | 136.17 |
| 166,656 | 1155.53 | 415.07 | 134.48 |
| 222,208 | 1331.98 | 416.94 | 133.46 |
| 277,760 | 1482.31 | 417.86 | 138.56 |
| 347,200 | **1652.46** | **418.53** | **138.93** |

The control reaches 10.65x and is still climbing. Both arms are flat to within
2% from step 111,104 onward, each at an equilibrium its own timescale predicts.
The `action_head` group follows the same shape, 14.92 to 158.66 on the control
against 48.26 and 21.60 on the arms.

**Growth is bounded rather than assumed to be bounded**, which is what the
comparison below rests on.

### The Turnaround Is Removed

`held_out.move_loss`, and `held_out.top1_accuracy` beside it:

| step | control | 33,333-step | 3,333-step | control top1 | 33,333-step top1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 111,104 | 1.441493 | 1.454880 | 1.504091 | 0.534348 | 0.531124 |
| 166,656 | 1.437835 | 1.450825 | 1.513595 | **0.535380** | 0.532215 |
| 222,208 | **1.437198** | 1.450683 | 1.515635 | 0.535229 | 0.532009 |
| 277,760 | 1.439773 | 1.450313 | 1.502808 | 0.534837 | 0.532249 |
| 347,200 | 1.467248 | **1.444943** | 1.500206 | 0.534615 | **0.533473** |

The control minimises at 222,208 and reads 2.09% worse by 347,200, a rise of
0.0276 against a combined dispersion floor of 0.00084. **The 33,333-step arm has
no minimum inside the range read.** Its best reading on both metrics is its last,
at the step where the control is clearly degraded, and no reading of it is worse
than an earlier one by more than that pair's floor. The middle three steps are
flat within noise and the two ends clear it, on loss and on top-1 alike.

So the conjunction the reading was declared against holds, and it holds about the
shape rather than the level. **At 347,200 the arm leads on loss by 0.0223 and
trails on top-1 by 0.0011**, both outside the floor. That split is the mechanism
read from the other side: what the control loses past its ceiling is calibration
rather than ranking, so an arm that fixes the calibration does not thereby rank
better.

### Decay Costs A Flat Offset, And The Curves Cross

The arm is not better everywhere. It sits 0.93% above the control at 111,104 and
0.73% above it at 277,760, an offset roughly constant across the range where the
control is healthy. What changes is only that the control then fails and the arm
does not, so the two cross between 277,760 and 347,200, and by 347,200 the arm is
1.52% ahead.

**Against the control's own best reading the arm is still 0.54% behind.** So
decay removes the ceiling without, by 347,200 steps, recovering what the ceiling
cost. Whether it would with more steps is not established here: the arm was still
improving when it stopped, and nothing was run past that point.

### Too Much Decay Is Its Own Failure

The 3,333-step arm bounds growth hardest and is the worst run of the three by a
wide margin, 4.4% above the control's minimum at every step. It is also unstable:
seven training-loss spikes above 1.75 against the control's three, one of them to
4.00 at step 205,000, and an `update_to_weight_ratio` that rises through the run
to 0.00462 where the control's falls to 0.00044.

The mechanism is the one that makes decay work, taken too far. A smaller norm
means larger relative updates, so past some strength the same dial that bounds
growth starts driving the step size instead. Its `legality.mask_penalty` reads
0.003447 against the control's 0.000788, so this arm also loses capability rather
than only calibration, which the other two do not.

This is why the rule below is a bracket rather than an optimum. One decade
separates an arm that works from an arm that does not, and nothing here locates
anything inside it.

### The Stability Reading Reverses

Over matched 354,144-step windows, counted on the per-interval training-health
record rather than on the cadence:

| run | spikes above 1.75 | clipped intervals | largest interval gradient norm |
| --- | ---: | ---: | ---: |
| control | 3 | 31 | 96.77 |
| 33,333-step | 1 | **1** | **12.30** |
| 3,333-step | 7 | 7 | 44.67 |

**The decayed arm at the working strength is the most stable of the three**, not
the least. The control clips in 31 intervals and reaches a gradient norm of
96.77; the arm clips in one and reaches 12.30. Each run's single spike at step
1,000 is warmup and is common to all three.

`0088` reports `clip_rate` as 0 at every cadence, which is correct and correctly
scoped: the cadence is a short instrumented probe every 69,465 steps and cannot
see this. The denser record is what separates the three runs, and it is the
instrument any future stability claim at this horizon should use.

## Decision

**Weight decay is what bounds the horizon ceiling, and the mechanism `0088`
named is confirmed.** Bounding parameter growth removes the turnaround in both
carrying metrics at the width and step count where the turnaround exists.

**The rule is an absolute decay timescale in optimizer steps, 33,333 of them.**
`anthro_chess.training.scaling_rules` states it that way and derives the
coefficient a run takes from its own peak rate.

**The timescale-as-a-multiple-of-the-horizon form does not survive.** Its defect
is what `0088` predicted and what these arms show directly: both equilibrated by
step 111,104 against a declared horizon of 2,222,080, so what set the equilibrium
was the per-step rate and not the run length. Stated as a multiple of the
horizon, a one-horizon timescale on these runs would have been 2,222,080 steps
and would have bounded nothing at all. The longer the run, the less it would
give of exactly what the run needs more of.

**An absolute timescale also has the right behaviour at the short end**, which is
not a separate argument but the same one read backwards. A run shorter than the
timescale is barely touched, which is correct, because growth has not accumulated
far enough to cost anything there. The horizon-multiple form instead scales its
strength to the run and so acts hardest where it is least needed.

**A constant coefficient is rejected for the same reason it was considered.**
Published practice holds the coefficient fixed, and that keeps the per-step rate
invariant only where the peak rate is also fixed. Here the rate rule moves the
peak with width, so a fixed coefficient would make the timescale drift with model
size. `0088` measured norm growth as per-step and width independent, 2.23x at
width 32 against 2.20x at width 128 at matched steps, so the quantity to hold
across widths is the step timescale.

### The Vehicle Does Not Adopt It

`configs/training/ablation-vehicle.toml` keeps `weight_decay = 0.0`. This is
decided rather than deferred.

`weight_decay` is inside `training_sha256`, so changing it invalidates the stored
seed dispersion and every candidate arm read against it, which `0065` and `0076`
price at five arms plus the loss of every prior comparison. What that buys the
vehicle is nothing: it runs 69,465 steps, a fifth of the way to the observed
minimum, on the healthy side of a bound it cannot reach. A dial that does nothing
at a configuration's horizon is not worth a configuration's history.

## What This Gives Up, Deliberately

**One width.** The strength is located at `model_dim` 128 only. The argument for
carrying a step timescale across widths rests on `0088`'s width-independence
finding rather than on an arm run at another width with decay enabled.

**One decade, two points.** 33,333 steps works and 3,333 steps does not.
Nothing was run between them, so the value in the rule is the arm that was run
rather than an optimum that was found, and the digits past the first describe
the arm.

**No seed floor.** Both arms differ from the control in `training_sha256`, so no
floor applies. The deltas that carry the decision are 2.09% and 1.52%, far
outside the pool's own dispersion of 0.00058, but nothing here is qualified on
training-seed noise and the reading is not presented as though it were.

**The arm stops where the control was read, not where it stops improving.** At
347,200 the arm was still improving and remains 0.54% behind the control's best
reading. Whether a bounded run eventually recovers that is the question the
ladder needs answered and this record does not answer it.

## Consequences

**The ladder can place rungs past the ceiling.** `0088` made the upper rungs of a
one-to-two-decade ladder unusable at a fixed ratio of positions to parameters,
because their step counts sit past where the response degrades. With growth
bounded the response is monotone through 347,200 steps, so a rung there measures
capacity rather than a turnaround. What is not yet established is the offset: the
arm runs about 0.9% above an undecayed run in the healthy range, and a ladder
fitted across rungs that differ in whether decay binds would read that offset as
a size effect.

**A long run is configured from the rule rather than from the vehicle.** The
vehicle's `weight_decay = 0.0` is correct for the vehicle and wrong for anything
that runs long enough to reach the bound. `anthro scale` is what produces the
setting for a run that does.

**Stability at this horizon is read on the per-interval record.** The cadence
cannot resolve what separates these runs, and a claim of no instability that
rests on it is a claim about a probe rather than about a run.

## References

- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0067-a-horizon-is-a-branch-not-a-restart.md`
- `0076-the-vehicle-is-width-128-at-the-target-regime.md`
- `0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md`
- `0088-the-horizon-has-a-ceiling-and-it-is-counted-in-steps.md`
- `docs/scaling.md`: the program, and where this sits in its order
- `#562`: the issue this answers, and `#54`, which it unblocks
