# 0090: The Size Is Not Fitted Here, And The Rate Is Measured At The Target

Date: 2026-09-28

## Status

Accepted. Replaces step 6 of the order in `docs/scaling.md`, which was a fitted
allocation ladder.

Keeps the size `0071-the-target-is-the-size-the-published-ladder-flattens-at.md`
derived, without the ladder fit that record expected to check it against, and
replaces the horizon it derived with one read from the run.

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

What it would have checked was partly constrained already. Chessformer's
published ladder on this architecture and task is flat by 23M parameters on move
matching, and `docs/scaling.md` puts a sizing error within roughly 1.5x below
what any reading here resolves. Two things argued for checking wider anyway:
`0089-bounded-growth-removes-the-ceiling-and-costs-a-flat-offset.md` made serving
cost secondary, which 0071 had weighed against width 768, and the throughput
measured at width 512 below is about 48% above the figure 0071 budgeted, which
0071 names as a reason to recompute. So widths 768 and 1024 were checked
directly, on the benchmarks the target is for rather than on loss.

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
parameter. A parabola through the three pool losses at 50 puts the vertex about
a hundredth of an octave below the rule's rate.

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
to the human reference that carries a floor is within it. It is worse by a
cleared margin on two readings: `dependency.rating_absent_degradation`, 0.043
against 0.051, and the legality margins.

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

### Wider Than The Target Buys Loss, Not The Dial

A 400-step probe at each width on one idle card, under the same settings:

| width | parameters | positions/s | peak memory | positions in the budget | per parameter | steps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 20.6M | 10,100 | 9.5 GiB | 2.4e10 | 1,180 | 1.49M |
| 768 | 47.9M | 5,500 | 14.7 GiB | 1.3e10 | 280 | 0.81M |
| 1024 | 88.5M | 3,620 | 19.8 GiB | 8.8e9 | 100 | 0.53M |

The budget is two cards for fourteen days at batch 16,384. Utilization rises
with width, from about 45% to about 69% of the 163 TFLOP/s 0071 measured, so each
parameter costs less at the wider widths and nothing in the setup is slowing
them.

On loss the three are not separable. The usual floor plus a power law in each of
parameters and positions, fitted through ten cooled pool points at widths 32, 128
and 512, puts all three within 0.2% of each other at this budget, and puts
doubling the budget at 0.1% to 0.2%. Leaving any one width out of the fit
reorders the three.

Two trunks ran at width 768, one at the rule's rate, 4.34e-4, and one at half
of it. Each declared 73,000 steps, 25 positions per parameter, and cooled over its
final fifth. The rule's arm was also branched at step 29,200 and cooled at
36,500. Each trunk took 60 hours on one card. The
rule's rate read 1.3715 on the pool against 1.3733 for half, so it holds on the
side checked; twice the rate was not run.

Against width 512 at nearly matched data, with `ladder.fitted_rating_slope` at
temperatures 0, 0.7 and 1:

| width | positions | pool loss | top-1 | dial slope |
| ---: | ---: | ---: | ---: | --- |
| 512 | 5.2e8 | 1.3938 | 0.5463 | 0.316 / 0.326 / 0.338 |
| 768 | 6.0e8 | 1.3849 | 0.5484 | 0.362 / 0.378 / 0.382 |
| 512 | 1.03e9 | 1.3793 | 0.5498 | 0.401 / 0.420 / 0.375 |
| 768 | 1.20e9 | 1.3715 | 0.5518 | 0.371 / 0.377 / 0.355 |

Width 768 is about 0.6% lower on loss at both points, about a third of which its
15% more positions would buy width 512 on its own curve. The dial does not
follow. Width 768 led at the first point by 0.04 to 0.05, part of that again its
extra data. Width 512 then gained 0.04 to 0.09 between its two points while
width 768 did not move, and at about 1e9 positions width 512 leads it by a
margin that clears the reading's floor at temperatures 0 and 0.7, with
reference-ladder error 16 points better at 0.7. The suite agrees: width 768 is
worse on generated game length at both temperatures and reads its rating less,
`dependency.rating_anchor_policy_divergence` 0.430 against 0.451.

Width 128 had already stopped on the dial: on its lower-rate arm the slope read
within 0.004 at 1.1e9 and 4.6e9 positions at temperatures 0 and 0.7, and lower
at 1. Width 512 has not reached that point at 1e9.

The suite itself failed its generated-play and ladder steps at width 768 with a
CUDA allocation error, and both completed when run alone. `#575` holds that.

## Decision

**The target stays at width 512 and no size ladder is fitted.** Wider models buy
loss and not the readings the target is for. At nearly matched data width 768 is
0.6% lower on loss; on the dial it led at 6e8 positions, stalled, and was behind
by 1e9, and it is worse on generated game length. The loss fit does not separate
1024 from either. This is a trend over two single runs, not a level difference.

**Its length is read rather than budgeted.** The budget buys about 2.4e10
positions at measured throughput, and width 512's dial was still rising at 1e9.
The run, `#494`, branches cooldowns along its trunk and stops where the dial and
generated play stop improving. If that happens well inside the budget, the
model is smaller than the budget can feed and the width reopens.

**The rate rule holds at the target's width.** Its width range extends to 512
and the exponent stays where 0087 rounded it. A fit through the four rungs'
vertices, width 512's taken from the pool, returns -0.555, so the rung 14.5 times
past the largest fitted count confirms the rule rather than moving it.

**The ratio range does not move.** The arms reached 25 and 50 positions per
parameter at width 512 only, and the ranges are separate dials: lowering the
floor would have let every narrower width answer there too, where nothing ran.

**The target itself still resolves nowhere.** At width 512 every ratio inside
the range is a horizon past the positions range, which stops at 1.2e9 against a
target run an order of magnitude longer. That is the extrapolation `#491` exists
to test, and the range it extends.

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

**The width comparison is one run per point.** The dial margins that decide it
are one and a half to two times the reading's own spread, and width 1024 was
probed for throughput, never trained.

**A short horizon.** These arms reach 50 positions per parameter against the
roughly 1,180 the target's budget buys, and they run 63,000 steps, far short of
where 0088 saw width 128 turn over. Whether the target turns over, and where, is
not read here.

## Consequences

**`anthro scale` answers between the vehicle's width and the target's**, on a
rule with a measured point at each end, and refuses the target until `#491`
extends the horizon.

**`#491` reads the target's trunk for a turnover and for where the dial stops**,
and `#569` is needed only if a turnover appears within the target's horizon.

**Width 768 has a rate on one side only.** The rule beat half its rate there, so
the rule's range stays at 512 rather than reaching a width whose bracket is
open.

## References

- `0071-the-target-is-the-size-the-published-ladder-flattens-at.md`
- `0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md`
- `0088-the-horizon-has-a-ceiling-and-it-is-counted-in-steps.md`
- `0089-bounded-growth-removes-the-ceiling-and-costs-a-flat-offset.md`
- `docs/scaling.md`: the order, the flatness rule, and the turnover
- `docs/research.md` (Chessformer / Maia-3)
- `#54`, `#491`, `#569`, `#496`, `#575`
