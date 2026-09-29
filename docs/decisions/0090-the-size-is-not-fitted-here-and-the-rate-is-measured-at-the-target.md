# 0090: The Size Is Not Fitted Here, And The Rate Is Measured At The Target

Date: 2026-09-28

## Status

Accepted. Replaces step 6 of the order in `docs/scaling.md`, which was a fitted
allocation ladder.

Keeps the size `0071-the-target-is-the-size-the-published-ladder-flattens-at.md`
derived, without the ladder fit that record expected to check it against.

Extends the width range of
`0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md` to the target, by
the bracket below.

Reads the turnover
`0088-the-horizon-has-a-ceiling-and-it-is-counted-in-steps.md` found as an
observation at one width rather than a bound on others.

Trains none of the rungs
`0089-bounded-growth-removes-the-ceiling-and-costs-a-flat-offset.md` placed past
that turnover, since no ladder runs.

## Context

`#54` was to train a ladder of widths across one to two decades, fit loss
against size and positions, and read the target's size and horizon off the fit.
Priced at measured throughput with its top rung at the target width, that ladder
came to about 380 GPU-hours, roughly eight days on both cards, against a target
run budgeted at fourteen days on both. A ladder earns its cost when the target is
far beyond its rungs. Here the top rung was nearly the target.

What it would have checked was already constrained. Chessformer's published
ladder on this architecture and task is flat by 23M parameters.
`docs/scaling.md` puts a sizing error within roughly 1.5x below what any reading
here resolves. The nearest widths either side, 384 and 768, sit at about half
and twice the parameters. Within the same budget 768 gets about half the
positions, and Chessformer's curve prices the step from 23M to 79M near half a
point of move matching. And the size did not rest on the serving premise
`0089-bounded-growth-removes-the-ceiling-and-costs-a-flat-offset.md` withdrew.

What the run could not start without was a learning rate. The rules were fitted
over widths 32 to 128 and refused the target's width.

## What Was Measured

Three trunks at `model_dim` 512, 20,642,630 parameters under the shape rules, at
half, one and twice the rate the size rule extrapolates to, 6.89e-4. Every other
setting is the vehicle's or the rules': batch 16,384 positions by accumulation,
warmup of 1% of the 50-position-per-parameter horizon, the square-root cooldown
over the final fifth, no decay, `bfloat16-mixed`, one seed, and the vehicle's data
order. Each trunk declares 63,000 steps, so its own cooldown is the 50 positions
per parameter endpoint. A branch resumes each at step 25,200 and cools to 31,500,
which is 25.

```console
uv run anthro train --config configs/training/ablation-vehicle.toml \
  --set 'run_name="w54-w512-lr1x"' --set steps=63000 \
  --set checkpoint_every_steps=25200 --set learning_rate=0.000689 \
  --set warmup_positions=10321920 --set model.model_dim=512 \
  --set model.attention_heads=16 --set model.feedforward_dim=1024 \
  --set model.geometric_bias_dim=128
```

Each trunk took 28.1 to 28.5 hours on one RTX 4090 at 10.1 to 10.2 thousand
positions per second, holding 9.5 GiB of the card's 24, and each branch 2.8
hours. Two trunks ran at once, one per card, at the rate one ran alone. Code at
`4563daf`. Scored on the frozen pool at view `canonical`.

### The Rule's Rate Is The Best Of Three

| rate | training tail | pool loss, 50 | top-1, 50 | pool loss, 25 | top-1, 25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5x | 1.39369 | 1.38157 | 0.54922 | 1.39806 | 0.54529 |
| **1x** | **1.39317** | **1.37930** | **0.54980** | **1.39380** | **0.54626** |
| 2x | 1.39576 | 1.38169 | 0.54910 | 1.39543 | 0.54580 |

The rule's rate is best at both horizons on the pool, and each neighbour is
worse by about four times the pool's own dispersion of 0.00058 at 50 positions per
parameter. A parabola through the three pool losses at 50 puts the vertex within
about a hundredth of an octave below the rule's rate.

The training tail, the mean of the last three logged intervals, which the three
trunks share because they share a data order, agrees on the order and not on the
shape. It puts half the rate only 0.04% behind, where the pool puts it 0.16%
behind. The pool is what the rate is selected on: it scores each arm's final
weights on one fixed set of positions, where the tail averages the last 1,500
steps of a cooldown whose weights are still moving.

Both higher rates spiked early and recovered. Twice the rate went from 1.99 to
4.30 between steps 500 and 1,000 with a gradient maximum of 51; the rule's rate
from 1.75 to 1.91 at step 1,500 with a maximum of 33. Warmup ends at step 630.
Both were back on their curves within about 1,000 steps. Half the rate did not
spike.

### The Suite Does Not Reverse It

Against twice the rate, the rule's rate is better on every held-out reading that
clears its floor, on reference-ladder error at temperature 0.7, 210 against 223,
and on the greedy dial slope, 0.401 against 0.373. Every generated-play distance
to the human reference that carries a floor is within it. It is worse by a cleared margin on two
readings: `dependency.rating_absent_degradation`, 0.043 against 0.051, and the
legality margins.

Against half the rate it is better on every held-out, legality and rating
dependency reading that clears its floor, and on puzzle solving. The dial is
within noise at every temperature.

### The Dial Widens With Size

`ladder.fitted_rating_slope` at each temperature:

| checkpoint | T=0 | T=0.7 | T=1 |
| --- | ---: | ---: | ---: |
| the vehicle, width 128 at 800 | 0.289 | 0.316 | 0.291 |
| width 128 at 3200, the best loss before this | 0.328 | 0.330 | 0.314 |
| width 512 at 50, the rule's rate | 0.401 | 0.420 | 0.375 |

Every row clears its floor against both width-128 checkpoints, ordering stays
perfect, and reference-ladder error falls at every temperature. With the 276k
model's -0.003 that is three sizes, and the slope rises across them while the
width-512 model has seen a sixteenth of the vehicle's positions per parameter.

## Decision

**The target stays at width 512 and no size ladder is fitted.**

**The rate rule holds at the target's width.** Its width range extends to 512 and
its ratio range down to 25 positions per parameter, which the arms covered at
width 512 only. The ranges are separate dials, so narrower widths now answer
below 100 too, on a horizon null measured at width 32 above it and at width 512
below it. The exponent stays where 0087 rounded it. A fit through the four rungs'
vertices, width 512's taken from the pool, returns -0.555, so the rung 14.5 times
past the largest fitted count confirms the rule rather than moving it.

**The target's full horizon is still refused.** The positions range stops at
1.2e9 and the target runs about 1.6e10, which is the extrapolation `#491`
exists to test.

## What This Gives Up, Deliberately

**No fitted size curve.** `#491` has no ladder prediction with an interval to
test against, and predicts instead from these branches and the horizon curve 0088
measured at width 128.

**Whether human-likeness saturates at a different size than strength** stays open.
The dial's slope at three sizes is the only reading here that bears on it.

**One seed per rate, and no seed floor at this width.** The neighbours clear the
pool's dispersion, which is a different draw of positions, not a different draw
of training. The vehicle's seed deviation, 0.097% on training loss, is below both
margins, but it was measured at width 128.

**A short horizon.** These arms reach 50 positions per parameter against the
target's 800, and they run 63,000 steps, far short of where 0088 saw width 128
turn over. Whether the target turns over, and where, is not read here.

## Consequences

**`anthro scale` answers at the target's width** up to the horizons the arms
reached and refuses the target's own.

**`#491` reads the target's trunk for a turnover**, and `#569` is needed only if
one appears within the target's horizon.

## References

- `0071-the-target-is-the-size-the-published-ladder-flattens-at.md`
- `0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md`
- `0088-the-horizon-has-a-ceiling-and-it-is-counted-in-steps.md`
- `0089-bounded-growth-removes-the-ceiling-and-costs-a-flat-offset.md`
- `docs/scaling.md`: the order, the flatness rule, and the turnover
- `docs/research.md` (Chessformer / Maia-3)
- `#54`, `#491`, `#569`, `#496`
