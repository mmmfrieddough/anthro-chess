# 0086: The Training Run, Not The Benchmark, Is The Binding Floor

Date: 2026-09-02

## Status

Accepted. Supplies the measurement
`0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md` created
its base for, and is stored against the identity
`0076-the-vehicle-is-width-128-at-the-target-regime.md` pins.

Rests on `0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`
for the arithmetic every floor here is built from, and on
`0026-conservative-dispersion-bounds.md` for the bound the spread is read
through. It is the successor reading to `0029-model-change-control-arm.md`,
whose figures were retired because they described a 276,002-parameter model at
8,000 steps, a fifth of the vehicle's size and an eighth of its horizon.

`docs/evaluation.md` (Regression Comparisons) states the resulting rule.

## Context

0029 measured seed variance once, at 276,002 parameters, and found it large
enough to fake a narrow result. Those figures were retired rather than quoted,
because nothing said they transferred. 0065 then created the one base whose
spread is worth buying, and 0076 froze it. What has been missing since is the
number.

Two questions were open. **Is a training run's spread large enough to matter
against the benchmark noise a reading already carries?** And **is it the seed**,
or is it the run: the vehicle takes relaxed determinism, so two arms at one seed
do not agree either.

## What Was Measured

Six arms of the vehicle at its own digest and horizon, relaxed determinism, data
draw held. Five distinct initialization seeds carry the spread; a second arm at
seed 17 carries the nondeterminism term, which is the reading 0076 substituted
for the strict agreement check that a relaxed digest cannot have. 35.57 card-hours of training and 3.34 of scoring on two RTX 4090s. Run two at
a time, that occupied the host for 19.6 hours, which is the figure a session
scheduling a re-characterization needs; the stored record sums the arms and
therefore reports card-hours.

**The two spreads are at parity; the two floors are not.** Measured dispersion
against measured dispersion, an arm's spread and a reading's own bootstrap are
the same size: a median ratio of 0.99, and the arm spread wider on about half
the cells. The floors differ by a median factor of 2.30, and almost all of that
is the replicate count rather than the measurement. Five arms put four degrees
of freedom behind the arm estimate, so decision 0026's bound sits at 2.372 times
its measured value on every cell, where a bootstrap over many games sits at a
median 1.024 times its own. The quotient of those two inflations is 2.32.

So the arm floor binds today, and what buys it down is more arms rather than a
better vehicle. The table below is the ratio of floors, and it inherits that
factor throughout. It is computed against one arm's own recorded dispersions,
seed17a; recomputed against each of the other five the median moves between 2.17
and 2.42, so the shape is not an artifact of which arm was chosen.

| family | median arm floor / benchmark floor | cells where the arm floor binds |
| --- | ---: | ---: |
| legality | 11.79 | 16/16 |
| puzzle | 7.35 | 6/6 |
| inference | 7.08 | 15/15 |
| ladder | 2.90 | 76/77 |
| adjudicated | 2.56 | 16/20 |
| held-out | 2.38 | 10/12 |
| dependency | 1.82 | 4/5 |
| generated play | 1.26 | 38/66 |
| novelty | 1.25 | 29/42 |

**Read as a control, arms of one configuration produce findings the benchmark
floor cannot reject.** The six arms form fifteen pairs. Of the roughly 261 rows
a pair qualifies, the ten distinct-seed pairs clear the benchmark floor on 26 to
50 of them, a median of 39.5; the same-seed pair, where not even an integer
differs, clears 21. Two rows across all fifteen pairs clear the arm floor as
well, which is what a floor covering 95% of same-configuration deltas should
leak. The rows concentrate in rating behavior, legality, and the adjudicated
decisions, and they are fewer distinct quantities than they are rows, since a
family's slices move together.

**The seed is the larger term, and the run is not a small one.** The same-seed
pair moved a metric by a median 0.62 of what the five distinct-seed arms did.
Those combine in quadrature, so at the median the run accounts for about two
fifths of the variance and the seed for three fifths. Reproducing an arm exactly
is not something this configuration can do, and the part of the spread the seed
does not explain is not small enough to set aside.

## Decision

**The stored floor is the total arm-to-arm spread, not the seed term extracted
from it.** What a candidate comparison faces is one training run against
another, and the two arms differ by their seed and by everything relaxed
determinism leaves free. Subtracting the nondeterminism term would produce a
narrower floor describing a comparison nobody runs.

The nondeterminism reading is stored beside the total rather than inside it, so
a later session can see how much of the width it accounts for without the floor
having been narrowed by an estimate that rests on one degree of freedom.

**A delta on the vehicle carries a claim only by clearing both floors.**
Clearing the benchmark floor alone is now known to be worth little on these
metrics: it is the condition a median 39.5 rows of a no-change comparison
satisfied, and 21 of those did not need even the seed to differ.

## What This Gives Up, Deliberately

**The nondeterminism term has one degree of freedom, and 636 cells do not
repair it.** Every cell reads the same single pair of runs, so the estimate's
error is common-mode across all of them and the median over cells is not a
median over replicates. The two-fifths figure above is what one pair happened to
show; a session wanting the split sharply pays for more replicates at one seed.
Nothing in the stored floor rests on it, which is why it is reported beside the
total rather than subtracted from it.

**The efficiency families are not measuring the seed.** An arm's inference
latency barely depends on its weights, so the spread stored for those cells is
mostly the machine drifting across the hours the six suites spanned. That is
still what a delta between two arms faces, and the floor covering it is right;
reading it as a statement about seeds is not, and the same caution applies to
those rows wherever they are counted above.

**The control reading is partly self-referential.** The arms being compared are
among the five behind the floor, so a delta between them falling inside it is
close to arithmetic, and the two rows that escaped are the useful part of that
half. The half that carries weight is the other one: the benchmark floor is
bootstrapped from each reading's own units and knew nothing of the arms, so its
rejections-that-should-not-have-been are measured rather than assumed.

**One host, one corpus, one horizon.** The floor describes the vehicle where it
was measured. `training_sha256` excludes the step budget, so a reading from a
branch at another horizon matches the key without sharing the spread, and the
comparison reports that rather than quoting the floor.

## Consequences

A vehicle comparison read on the benchmark floor alone was missing a floor of
comparable width, and on the evidence here it admitted something on the order of
15% of its rows that a rerun of the baseline would also have produced. The
language `anthro eval report` used for it has been corrected: `cleared` on the
benchmark floor is a statement about the benchmark, and the column beside it is
what sees the training run.

Re-characterizing costs about 39 card-hours, which is 20 hours of a two-card
host. A session that wants a narrower floor rather than the same one pays
more than that, because the width is set by the replicate count: eight arms
would bring the median floor ratio to about 1.7 and twenty-one to about 1.3.

## References

- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0076-the-vehicle-is-width-128-at-the-target-regime.md`
- `0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`
- `0029-model-change-control-arm.md`: the retired reading this replaces
- `docs/evaluation.md`: Regression Comparisons
