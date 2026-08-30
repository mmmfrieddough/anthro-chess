# 0084: The Direction Test Is Graded, And The Pinned Pass Is Derived

Date: 2026-08-29

## Status

Accepted. Refines `0028-qualifying-the-rating-dependency-family.md`, which
settled which of this family's quantities can carry a sampling floor. This
record changes which quantities there are.

## Context

Walking the suite for `#329` read the family on three checkpoints: one at 10,000
steps, and a pair 4x apart at 138,931 and 555,727.

`dependency.rating_cross_conditioning_match_rate` read exactly 1.000 on all
three. It counts the rating bands whose best conditioning value is their own,
out of four, so it takes five values and a checkpoint that has learned the
ordering at all takes the top one. The step-10,000 arm already had. A metric
that saturates before a model is worth reading can report a regression and
never progress, and this one is the family's whole answer on direction, which
`docs/evaluation.md` calls the thing that separates a model reacting to an input
from one that learned its meaning.

The table under it was not saturated. What a position pays for being scored
outside its own band read 0.0577, 0.0930, and 0.1014 across the same three
checkpoints, a factor of 1.76 while the reported number moved by a factor of
1.00.

Separately, the reading paid eight forward passes per batch. One was
`ConditioningKind.CONSTANT` at a configured 1500, pinning every position at a
single rating. The cross-conditioning grid scores every position at each of its
four ratings, so the position-weighted mean of any of those columns is that same
degradation at that rating: +0.1405, +0.0493, +0.0358, +0.0944 on the mature
checkpoint, against +0.0389 reported for the dedicated pass at 1500.

## Decision

**The cross-conditioning comparison is reported graded as well as counted.**
`dependency.rating_cross_conditioning_penalty` is the mean, over positions whose
band the grid names a value for, of the extra move loss that position pays under
the grid's other values. The match rate stays, because a saturating metric is
exactly what a tripwire should be; it is no longer the only thing said about
direction.

The graded form is a mean over positions, so unlike the rate it has a per-game
share and carries a sampling floor. That is not a bonus, it is the point:
`0028` established that the rate structurally cannot have one, which left the
family's direction reading unqualifiable as well as saturated.

**The pinned-rating treatment is derived from the cross-conditioning table
rather than scored.** `constant_rating` is gone from the configuration and
`dependency.rating_constant_degradation` from the registry. The reading is seven
passes rather than eight, and the report shows the pinned degradation at every
grid rating instead of at one, which also names the single rating that best
explains the corpus.

**Both tables are rendered.** The cross-conditioning grid and the per-band
prefix split were computed, recorded, and shown as one scalar each. The scalars
stay as summaries and the tables print beneath them.

**The view is 8,000 games rather than 2,000.** Sized on the quantity that binds
rather than on what the reading costs. Four independent draws at 2,000 games put
the shuffled degradation's spread at 0.0031, and it moved 0.0049 across four
times the training: a clearance of 1.34, which one unlucky draw reads as no
movement at all. Four times the games halves the floor and takes that to 2.64.
Nothing this family reports asks for finer, and detecting a dead input needs a
small fraction of it.

## Why Not Redefine The Match Rate

Making the rate finer, by counting adjacent pairs or by ranking rather than
matching, keeps a count where the underlying quantity is a distance. It would
raise the ceiling and still have one, and it would still have no per-game share.
The graded form has neither problem, and the two together cost nothing extra:
both are read off passes the reading already takes.

## Why Not Keep The Pinned Pass As A Cross-Check

Two routes to one number is worth paying for where they could disagree. These
cannot: the derived value reads the same conditioned losses over the same
positions, so agreement is arithmetic rather than evidence. What the dedicated
pass bought was the freedom to pin at a rating off the grid, and no reading was
waiting on 1500 in particular.

## Consequences

The family still reports seven quantities: one is removed and one is added, so
what changes is which, not how many. The penalty's history begins here.

`dependency.rating_constant_degradation` leaves the registry, and its stored
values become unreadable rather than frozen: `anthro eval report` resolves a
metric through the registry, so `--history` on it now fails and a delta report
skips it. The records keep the numbers and nothing renders them. That is
accepted here because the whole curve it was one point of is now recorded on
every reading.

Seven passes rather than eight measured 28.4 s against 31.3 s at 2,000 games on
one idle RTX 4090, and the surviving quantities were unchanged to every reported
digit. The shipped step is 8,000 games and costs 95.1 s on that machine, so the
pass saved and the resize move the cost in opposite directions and the resize
wins.

**The resize ends every series this family carries.** A view cap selects a
different set of games, so the workload digest changes and
`series_fingerprint` returns a new value for all of them. That is the rule
`docs/evaluation.md` already states for two sizes of one reading, and it applies
here whatever else this change did.

Four places state the pass count, and a later change to the treatments moves all
four: `configs/evaluation/rating-dependency.toml`,
`configs/evaluation/checkpoint-suite.toml`, `docs/evaluation.md`, and
`tests/evaluation/test_checkpoint.py::test_the_dependency_tests_score_each_conditioning_once`,
which is the only one that fails on its own.

## References

- `#329` - the suite walkthrough this came out of
- `0028-qualifying-the-rating-dependency-family.md` - which quantities can carry a floor
- `0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md` - the estimator in force
- `docs/evaluation.md` - "Dependency Tests"
