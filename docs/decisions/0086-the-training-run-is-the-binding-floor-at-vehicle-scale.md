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
whose figures were retired because they described a model three orders of
magnitude smaller.

`docs/evaluation.md` (Regression Comparisons) states the resulting rule.

## Context

0029 measured seed variance once, at 276,002 parameters, and found it large
enough to fake a narrow result. Those figures were retired rather than quoted,
because nothing said they transferred to a real size. 0065 then created the one
base whose spread is worth buying, and 0076 froze it. What has been missing
since is the number.

Two questions were open, and the second was not obviously the interesting one.
**Is seed variance large enough at vehicle scale to matter?** And **is it the
seed at all**, or is it the training run: the vehicle takes relaxed determinism,
so two arms at one seed do not agree either.

## What Was Measured

Six arms of the vehicle at its own digest and horizon, relaxed determinism, data
draw held. Five distinct initialization seeds carry the spread; a second arm at
seed 17 carries the nondeterminism term, which is the reading 0076 substituted
for the strict agreement check that a relaxed digest cannot have. 35.57 hours of
training and 3.34 of scoring on two RTX 4090s, 38.90 hours in all.

**The training run is the binding floor on most of what this project reads.** Of
the 259 metric cells carrying both a benchmark dispersion and a seed dispersion,
the seed floor is the wider one on 210, a median factor of 2.30.

| family | median seed floor / benchmark floor | cells where seed binds |
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

**Read as a control, two arms differing only in seed produce fifty findings the
benchmark floor cannot reject.** Across 865 rows of that comparison, 50 clear
the benchmark floor while sitting inside the seed floor, concentrated in
rating behavior, legality, and the adjudicated decisions. Nothing changed in
those runs but an integer.

**Most of the spread is the run rather than the seed.** The same-seed pair moved
a metric by a median 0.62 of what five distinct seeds did. Reproducing an arm
exactly is not something this configuration can do, and the part of the spread
that is not attributable to the seed is not small enough to set aside.

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
metrics: it is the condition 50 rows of a no-change comparison satisfied.

## What This Gives Up, Deliberately

**The nondeterminism term has one degree of freedom.** One pair estimates a
spread very poorly, and its bound is punishing accordingly. It is reported as a
share rather than as a quantity for that reason, and a session wanting it
sharply pays for more replicates at one seed.

**The efficiency families are not measuring the seed.** An arm's inference
latency barely depends on its weights, so the spread stored for those cells is
mostly the machine drifting across the hours the six suites spanned. That is
still what a delta between two arms faces, and the floor covering it is right;
reading it as a statement about seeds is not.

**The control reading is partly self-referential.** Two of the five arms behind
the floor are the two being compared, so a delta between them falling inside it
is close to arithmetic. The half that carries weight is the other one: the
benchmark floor is bootstrapped from each reading's own units and knew nothing
of the arms, and its 50 rejections-that-should-not-have-been are measured rather
than assumed.

**One host, one corpus, one horizon.** The floor describes the vehicle where it
was measured. `training_sha256` excludes the step budget, so a reading from a
branch at another horizon matches the key without sharing the spread, and the
comparison reports that rather than quoting the floor.

## Consequences

Every earlier vehicle comparison read on the benchmark floor alone was read
against a floor a median 2.3 times too narrow, and the language `anthro eval
report` used for it has been corrected: `cleared` on the benchmark floor is a
statement about the benchmark, and the seed column beside it is what sees the
training run.

Re-characterizing costs about 39 hours of one two-card host, which is what a
session replacing the vehicle should price. The record carries that figure so
the question does not have to be re-derived.

## References

- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0076-the-vehicle-is-width-128-at-the-target-regime.md`
- `0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`
- `0029-model-change-control-arm.md`: the retired reading this replaces
- `docs/evaluation.md`: Regression Comparisons
