# 0089: Bounded Growth Removes The Ceiling, And Costs A Flat Offset

Date: 2026-09-25

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
purpose. The vehicle does not adopt this, for the reason under **Decision**.

## Context

`0088` found that a width-128 run reaches its loss minimum near 222,000
optimizer steps and degrades after it, and traced that to unbounded parameter
growth under `weight_decay = 0.0`. No arm had been run with decay at a width and
step count where the failure appears, so the cause was evidenced rather than
shown.

## What Was Measured

Arms at `model_dim` 128 copying a control and differing in `weight_decay`
alone, each against a control at its own learning rate. All share a seed, a data
order and a declared horizon of 2,222,080 steps, so every checkpoint compared sits
on a constant trunk rate, which is the comparison `0067` permits:

```console
uv run anthro train --config configs/training/ablation-vehicle.toml \
  --set 'run_name="<arm>"' --set steps=2222080 \
  --set checkpoint_every_steps=6944 --set weight_decay=0.01 \
  [--set learning_rate=0.0015]
```

The first arms decayed every parameter the model owns. `#566` found that this
erased the rating conditioning and changed the optimizer to exempt
one-dimensional parameters; the arms named "grouped" below ran with that change.
Every arm differs from its control in `training_sha256`, so **no seed floor
applies**, and each arm is a single seed. Readings are on the frozen pool at view
`canonical`.

## What Was Found

### Decay Removes The Ceiling

Parameter norm, at matched steps:

| step | control | uniform decay | grouped decay |
| ---: | ---: | ---: | ---: |
| 6,944 | 155.11 | 142.19 | 142.93 |
| 111,104 | 939.70 | 411.30 | 421.76 |
| 222,208 | 1331.98 | 416.94 | 455.68 |
| 347,200 | **1652.46** | **418.53** | **500.55** |

The control reaches 10.65x and is still climbing. Both decayed arms are bounded.
Neither shows the control's turnaround: the grouped arm at 3e-3 reads 1.451922,
1.450106, 1.449200, 1.449483 and 1.444759 across steps 111,104 to 347,200, flat
within noise and then lower, where the control is 2.09% worse at 347,200 than at
its minimum. **The mechanism `0088` named is confirmed.**

### The Learning Rate Decides What It Costs

At matched step 347,200, past the ceiling:

| arm | `held_out.move_loss` | ladder error, T=1 | game-length distance, T=1 |
| --- | ---: | ---: | ---: |
| control, 3e-3 | 1.467248 | 199.58 | 2.60 |
| uniform decay, 3e-3 | 1.444943 | 213.29 | 9.74 |
| grouped decay, 3e-3 | 1.444759 | 213.31 | 6.66 |
| control, 1.5e-3 | 1.464029 | 211.57 | 21.93 |
| **grouped decay, 1.5e-3** | **1.436491** | **202.33** | 11.44 |

**At 3e-3, decay costs the rating dial and human-likeness.** Against its control
the grouped arm is worse on the reference ladder and on every generated-play
distance, by 156% on game length.

**At 1.5e-3 the same comparison reverses.** The grouped arm is better than its
control on the reference ladder and on every generated-play distance at both
temperatures read, game length by 47.8%, and it has the lowest loss of any arm
past the ceiling. Against the best reading either control ever reached, 1.431748
at 1.5e-3 and step 222,208, it is 0.33% behind, so a decayed run still does not
beat an undecayed one stopped at its peak.

**The lower rate reads better at these horizons with or without decay**: 1.431748
against 1.437198 at the undecayed peaks, and 1.436491 against 1.444759 decayed.
`0087` fitted the rate over horizons that stop short of the ceiling, where the two
rates `0088` compared were within 0.024% of each other.

### The Grouping Is Necessary And Not Sufficient

Against uniform decay at the same rate, the grouping improves every conditioning
and human-likeness reading that moves and worsens none, and leaves loss
unchanged. It does not move the reference ladder. The frozen rating embedding
was real, 0.14x of the control under uniform decay against 0.41x grouped, but it
was not what cost the dial. `#566` carries that reading.

## Decision

**Weight decay is what bounds the horizon ceiling, and any run with decay uses the
grouping `#566` introduced.** Uniform decay erases the rating conditioning.

**The best configuration measured is a peak rate of 1.5e-3 with coefficient
0.01**, a decay timescale of 66,667 optimizer steps. A run long enough to cross
the ceiling starts from that, and **reads its rate and its decay strength at its
own horizon rather than taking either from a rule fitted short of the bound.**

**The timescale is held in optimizer steps, and the horizon-multiple form does
not survive.** Every decayed arm reached its equilibrium norm by step 111,104
against a declared horizon of 2,222,080, so the per-step rate set the bound and
the run length did not. Held as a multiple of the horizon, the rate weakens as a
run lengthens, which gives least where growth is worst.

**Nothing in this repository adopts it yet.** `anthro_chess.training.scaling_rules`
keeps producing no decay, because the longest run its fitted ranges can express
is a third of the way to the ceiling. The vehicle keeps `weight_decay = 0.0`: it
runs 69,465 steps, short of the bound, and changing the setting would invalidate
its stored seed dispersion and every arm read against it for nothing it measures.

## What This Gives Up, Deliberately

**One seed per arm, one width.** The rate reversal is a single-seed reading of
two arms, and nothing here is qualified on training-seed noise.

**Generated play past the ceiling is not a stable readout.** The two undecayed
controls differ eightfold on game length at the same step, 2.60 against 21.93,
because each sits at a different point in its own degradation. The comparisons
above hold each arm against its own control for that reason, and the cross-rate
numbers are not read against each other.

**The rate and the decay strength moved together.** The 1.5e-3 arm kept the
coefficient at 0.01, which halves its per-step decay to a 66,667-step timescale
against the 3e-3 arm's 33,333. Which of the two changes produced the reversal is
not separable here. A 3,333-step arm, coefficient 0.1 at 3e-3, bounds the
strength from above only: it destabilized, with seven loss spikes, and read 4.4%
worse than its control.

**The timescale is realized only at the peak rate.** Every arm was a constant
trunk with no cooldown, so a run declaring its real horizon decays more slowly
than the stated timescale over its warmup and cooldown.

## Consequences

**The ladder can place rungs past the ceiling**, provided those rungs take
grouped decay. What it has to price is the rate: a rung long enough to need decay
is long enough that the fitted rate is no longer the right one.

## References

- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0067-a-horizon-is-a-branch-not-a-restart.md`
- `0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md`
- `0088-the-horizon-has-a-ceiling-and-it-is-counted-in-steps.md`
- `#562`, `#566` for the grouping, `#565` for the adjudicated-decisions defect
  found alongside, and `#54`, which this unblocks
