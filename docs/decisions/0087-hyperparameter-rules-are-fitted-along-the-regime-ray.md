# 0087: Hyperparameter Rules Are Fitted Along The Regime Ray, And Refuse Outside It

Date: 2026-09-02

## Status

Accepted. Fixes step 5 of the order in `docs/scaling.md`.

Rests on `0076-the-vehicle-is-width-128-at-the-target-regime.md`, whose rate
sweep is the top rung of the size fit here and whose recipe every arm below
copies, and on
`0071-the-target-is-the-size-the-published-ladder-flattens-at.md`, which fixes
the target these rules have to reach and the counting convention they are
expressed in.

`0067-a-horizon-is-a-branch-not-a-restart.md` owns the schedule family the
warmup and cooldown rules sit inside, and is the reason the cooldown fraction is
carried rather than fitted.

`0086-the-training-run-is-the-binding-floor-at-vehicle-scale.md` supplies the
dispersion every near-optimal band here is read against, and this record extends
that measurement to two smaller widths.

## Context

`docs/scaling.md` requires a scale-dependent setting to be recorded as the rule
that produces it rather than as the number the rule produced, because a value is
correct at one point and silently wrong everywhere else with no goodness-of-fit
statistic to say so. Until now the project had values.

`0076` swept the rate once, at one width and one horizon, and said in as many
words that it was a rung of this fit rather than the fit. The other four
settings had never been compared against anything: warmup, the decay timescale
and the second-moment decay only became configurable in `#493`, and the batch
had been chosen for throughput rather than for learning.

The order matters. `docs/scaling.md` marks the peak learning rate hard against
six of eight columns of its coupling table, so a ladder whose rungs are unequally
tuned measures the tuning rather than the model. `docs/research.md` (Resolving
Discrepancies In Compute-Optimal Scaling) records a published exponent moving
substantially once the batch, the rate and the second-moment decay were re-tuned
at each size, and records that every intermediate fit was statistically
well-behaved while it was wrong.

## What Was Measured

Every arm copies the vehicle's recipe and changes one thing. Relaxed
determinism, `bfloat16-mixed` with `high` matmul precision, compilation at the
shipped default, the widened corpus through the shard-backed loader, no
selection filters, and the constant trunk with a square-root cooldown over the
final fifth.

**The response is the mean training loss over the final three logged
intervals**, with the logging interval set to a hundred and twentieth of each
run's horizon so that the window is the same share of every arm. That is the
statistic `0076` used, and reproducing its three published figures to four
decimal places is what lets its sweep serve as this fit's top rung. Training
loss rather than validation for the reason `0076` gives: at these horizons
against a corpus of 138.7e9 plies nothing repeats, so a training position is a
held-out position.

**Sizes are the multiples of 32 the head dimension allows.** Widths 32, 64 and
128 carry the size fit and **width 96 is held out entirely**, so the validation
run is a prediction tested rather than a curve interpolated through its own
answer.

**The response is read within a family and never across one.** Two arms are
comparable on this statistic only when their logged intervals score the same
positions, which needs the same horizon, the same interval spacing, and the same
data order. Neighbouring intervals differ by two to four percent according to
what slice of the corpus they cover, several times the difference between two
rates, so arms sharing all three cancel that term and arms differing in any of
them are compared through it. Six arms of the vehicle differing only in seed
agree to 0.097%; one arm of this fit differing from its rung only in the loader
seed read 2% away from where its neighbours put it. The width-128 rung is
therefore `#487`'s three arms alone, and the two arms run at 5.5e-3 to extend it
are excluded rather than averaged in.

### The Rate Against Size

At 800 positions per parameter and batch 16384. Widths 32, 64 and 128 carry the
fit; 96 is held out.

| peak rate | 32 | 64 | 128 |
| ---: | ---: | ---: | ---: |
| 0.001 | | | 1.4235 |
| 0.0015 | | 1.5460 | |
| 0.003 | 1.7931 | 1.5280 | **1.4181** |
| 0.006 | 1.7606 | 1.5215 | |
| 0.01 | | | 1.4253 |
| 0.012 | **1.7440** | **1.5210** | |
| 0.015 | | 1.5350 | |
| 0.018 | | 1.6386 | |
| 0.024 | 1.7901 | diverged | |
| 0.048 | diverged | | |

**The curve is not symmetric, and that is the most useful thing in the table.**
Halving the rate below the optimum costs almost nothing; at width 64 the arms
climb 1.5210, 1.5350, 1.6386 and then diverge across a two-fold span. A rule
that has to be wrong should be wrong low.

The usable ceiling also falls with width: 2.4e-2 trains at width 32 and diverges
at 64.

### The Rate Against The Horizon

At width 32 and batch 16384.

| peak rate | 100 | 400 | 800 |
| ---: | ---: | ---: | ---: |
| 0.003 | | 1.8481 | 1.7931 |
| 0.006 | 2.1254 | 1.7857 | 1.7606 |
| 0.012 | **2.0502** | 1.7555 | **1.7440** |
| 0.024 | 2.2196 | **1.7510** | 1.7901 |
| 0.036 | | 1.9910 | |
| 0.048 | 3.2057 | 1.8566 | diverged |

### What Each Rung Puts The Optimum At

A vertex is the median over every bracket of its rung rather than the one
through the nearest neighbours. Far above the optimum the curve is steep and
erratic, and a parabola with one flank a fraction of a percent above the minimum
and the other thirteen percent above is mis-specified rather than mis-measured.
The spread across those brackets is what says so; propagating loss noise through
a single fit cannot see it, and reported ±0.001 on the rung that most needed the
warning.

| rung | optimum | near-optimal band | tolerance |
| --- | ---: | --- | ---: |
| width 32, 800 pos/param | 1.02e-2 | [6.9e-3, 1.4e-2] | 0.70% |
| width 64, 800 pos/param | 6.4e-3 | [5.5e-3, 1.2e-2] | 0.09% |
| width 128, 800 pos/param | 2.9e-3 | [2.2e-3, 3.8e-3] | 0.10% |
| width 32, 100 pos/param | 1.05e-2 | [9.6e-3, 1.3e-2] | 0.59% |
| width 32, 400 pos/param | 1.4e-2 | [1.0e-2, 2.5e-2] | 0.69% |

A band is where the measured curve stays within one run-to-run standard
deviation of its minimum, walked outward along the arms rather than read off the
parabola, because the parabola is symmetric in log rate and the curve is not.

### Dispersion Of The Response

| width | arms | mean | seed dispersion | one seed, two arms |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 4 | 1.7350 | 0.699% | 0.090% |
| 128 | 6 | 1.4325 | 0.097% | 0.029% |

**The spread grows sharply as the model shrinks**, seven-fold from the vehicle to
width 32 on the seed term and three-fold on the nondeterminism term. Two numbers
rather than one because they answer different questions. The seed dispersion is
what a later session choosing a rate would meet, so it sets the band. Every arm
of one rate grid shares a seed, so the nondeterminism term is what limits how
well that grid locates its own vertex, and it is what propagates into the
exponents. A single tolerance would either widen the bands dishonestly or claim
a vertex precision the arms do not support.

### The Held-Out Rung

Width 96 was not used in any fit. Its middle arm was configured by the resolver
itself rather than by a re-derivation of it, and the two flanking arms differ
from it in the rate and in nothing else.

| arm | rate | training loss | pool loss | top-1 accuracy | top-5 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.4x | 1.575e-3 | 1.4639 | 1.4649 | 0.528932 | 0.893828 |
| **the rule** | **3.938e-3** | **1.4598** | **1.4608** | **0.529828** | **0.894373** |
| 2.5x | 9.845e-3 | 1.4621 | 1.4625 | 0.529514 | 0.894244 |

**The rule's rate wins on training loss, and the frozen pool orders the three the
same way on every metric it reports.** The pool is 211,475 held-out games and
14,161,038 positions, and its move loss reproduces the training tail statistic to
within 0.07%, which is the first direct evidence for the substitution `0076` made
by argument: at these horizons nothing repeats, so a training position is a
held-out position.

The accuracy margins are 0.03 and 0.09 percentage points and are not claimed to
clear the benchmark's own floor. What they establish is the absence of the
failure this check was run to find, which is a rate ordering that reverses
between training loss and the benchmark the project selects on.

### The Other Three Settings

All at width 32, 800 positions per parameter, batch 16384, at the rate the rule
gives there, each arm differing from the others in its own dial alone.

| warmup, as a share of the horizon | 0.25% | 1% | 4% | 8% |
| --- | ---: | ---: | ---: | ---: |
| | **1.7353** | 1.7376 | 1.7381 | 1.7392 |

| decay timescale | 0.25x horizon | 1x | 4x | none |
| --- | ---: | ---: | ---: | ---: |
| | 1.7379 | **1.7329** | 1.7499 | 1.7402 |

| second-moment span, positions | 3.3e5 | 5.5e5 | 1.64e6 | 5.46e6 | 1.64e7 | 5.46e7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | 1.7432 | **1.7287** | 1.7352 | 1.7549 | 1.7461 | 1.7786 |

**Warmup is flat.** The response rises monotonically with length but over a total
spread of 0.38%, inside this width's 0.70% dispersion. Shorter reads very
slightly better and nothing here resolves it.

**Weight decay does not resolve either.** The best beats no decay at all by
0.42%, inside the dispersion, and the four-horizon arm reads worse than no decay,
which it cannot be. That is what a corpus repeating nothing should give: at
1.17e8 positions against 138.7e9 plies there is no overfitting for decay to
prevent.

**The second moment had a signal that did not survive a second horizon.** At 800
positions per parameter its span has an interior optimum at 33 optimizer steps
and the vehicle's value costs 1.0%, which clears the dispersion. Repeating the
sweep at 400 moved the optimum to 100 steps:

| second-moment span at 400 pos/param | 10 steps | 16.7 | 33.3 | 100 |
| --- | ---: | ---: | ---: | ---: |
| | 1.7392 | 1.7511 | 1.7451 | **1.7257** |

Adam's second moment averages over steps, so at a fixed batch a span in
positions and one in steps are the same claim; a span held as a share of the run
is a different one. Neither predicts this. A constant span wants 33 steps at both
horizons, and a constant share wants 16.7 steps at the shorter one, which is the
worst arm run there. Both curves are also non-monotone. The four arms of the
second horizon share their step count, logging interval, data order, seed and
rate, so this is not a comparability artifact.

So the span is held at the vehicle's, and the reason is that nothing measured
here identifies one to carry rather than that a better one was left on the table.

## Decision

**Every scale-dependent setting is produced by
`anthro_chess.training.scaling_rules`, which refuses a scale outside the range
its arms covered. `anthro scale` is the command that reads it.**

The peak rate is the one setting that became a rule against scale:

    learning_rate = 3.0e-3 x (parameters / 1,422,662) ^ -0.55

fitted over widths 32 to 128 at 100 to 800 positions per parameter, and holding
at batch 16384. Two figures rather than the fit's four, because every rung's
band is about a factor of two wide: at -0.55 the rule lands inside all three
bands and at -0.5 it leaves width 64's, so two is what the measurement supports
and one is not.

**The horizon term is zero, and that is a measurement rather than an omission.**
Across an eightfold change in run length the exponent came out at -0.001 with a
standard error of 0.089. Writing the point estimate down would dress a null
result as a finding.

**A parameter count is every tensor the assembled model owns, the action head
included.** `anthro_chess.models.parameter_count` is the one implementation. The
head here is a sixty-third of the model at width 32 rising to a thirty-ninth at
512, because it factors an action into two square choices rather than projecting
onto a vocabulary, so the confound `docs/research.md` puts first behaves in the
opposite direction from the language-model case it was named for.

**The other four settings are held rather than fitted, each for a stated
reason.** Batch, because its optimum did not move across the horizons measured.
Warmup and the decay timescale, because their responses do not clear the
dispersion. The second-moment span, because the better value found is one
reading at one scale against an instrument that runs at the current one.

## What This Gives Up, Deliberately

**The target is outside the range and the rules refuse there.** The target is
width 512 and the fit reaches 128, so `anthro scale --model-dim 512` fails rather
than answering. That is the intended behaviour and the ladder is what extends it.
Nothing in a fit's residuals says where it stops holding, so the boundary is
carried beside the rule and asking past it has to fail loudly.

**One rule out of five.** The issue asked for a rule against scale for each of
five settings and four of them did not earn one. Each refusal is backed by arms
rather than by omission, but a reader wanting four more exponents will not find
them, and the batch axis in particular measured two things at once: at a fixed
horizon in positions, a larger batch is also fewer optimizer steps.

**The rate is fitted on training loss.** The held-out pool agrees, ordering the
three width-96 arms the same way on every metric, and its move loss reproduces
the training statistic to 0.07%. But the accuracy margins there are 0.03 and 0.09
percentage points against a bootstrap dispersion near 0.12, so the benchmark
confirms the absence of a reversal rather than the location of the optimum.

**Every rung is one seed.** The grids hold the seed, so the vertex is limited by
the nondeterminism term rather than the seed term, which is what makes them
usable. But no rung has replicates at more than one rate, so a rung's optimum
carries no estimate of how much the seed moves it.

**The horizon reaches 800 positions per parameter and no further.** The arms that
would have carried 1600 were spent re-measuring the anchor rung instead.

**One published anchor disagrees and cannot be reconciled from here.**
`docs/research.md` (Chessformer) records a peak rate of 5e-5 held constant from
3M to 79M parameters, at a batch this project independently arrived at. At the
sizes measured here that rate is not merely low: `#487` read 2.4390 at 3e-5
against 1.4696 at the optimum. Extrapolated to their sizes this rule would give
about 3.3e-4, still some seven times higher, and it declines to extrapolate
there. Their single value spans both a 26x parameter range and a 16x range in
positions per parameter, which no rule of this shape fits, so it reads as a
default carried across a ladder rather than a measurement at each rung. The
ranges do not overlap and nothing here settles it.

## Consequences

**The ladder extends the range rather than trusting the rule past it.** Step 6 of
the order in `docs/scaling.md` runs several sizes, and each rung it adds is a
rung this fit could use. Until then the target's own settings are not something
these rules can produce.

**A candidate arm can be given its own rate.** `docs/scaling.md` says a candidate
that reads negative is discarded only where it had one, and `anthro scale` is now
what supplies it.

**The second-moment span is open rather than settled.** A shorter span measured
better at one horizon and a different one measured better at another, in a
direction neither candidate form predicts. `#552` carries both readings and what
would settle them, which is the span swept at a third horizon and a second width
with replicates, since the response was rough enough that the non-monotonicity
may itself be noise. Until then the vehicle's value is what every comparison in
this milestone is read against, and that is reason enough to keep it.

**A sweeping session can tell a diverged arm from a crashed one.** Three arms
here diverged, and the first of them stopped the run on a JSON serialization
error several intervals after the model had actually left the finite range.
`#551` is fixed alongside this work: both divergence paths now exit on their own
code.

**Two comparability conditions bind every later reading of this statistic.** Arms
compared on end-of-run training loss must share their horizon, their interval
spacing and their data order. Neither is checked by anything that runs, so a
session mixing families gets a confident wrong answer, which is what happened
here twice before the guard was written.

## References

- `0076-the-vehicle-is-width-128-at-the-target-regime.md`
- `0071-the-target-is-the-size-the-published-ladder-flattens-at.md`
- `0067-a-horizon-is-a-branch-not-a-restart.md`
- `0086-the-training-run-is-the-binding-floor-at-vehicle-scale.md`
- `docs/scaling.md`: the program, the coupling table, and the rules-not-values rule
- `#489`: the issue these rules were fitted for
