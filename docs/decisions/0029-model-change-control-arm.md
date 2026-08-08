# 0029: A Model Change Is Read Against A Control Arm

Date: 2026-08-02

## Status

Accepted as initial design direction. Refined by
`0040-training-noise-floors-are-scoped-to-the-configuration-they-measured.md`,
which gives a stored training floor the scope this record withheld one for.

## Context

The test suite proves that a change did not break anything. Nothing in the
project said how a change showed it *improved* anything, so every change that
altered the model decided its own evidence standard, and the cheapest way to
satisfy an undefined standard is a plausible story with no reading behind it.

Two capabilities already existed and neither closed the gap. The results store
compares any two recorded checkpoints, and the noise system qualifies a delta
against a characterized floor. But the comparison a report offers by default is
against the previously recorded reading, which in this project's history means a
different corpus, a different step budget, a different machine, and every other
change that landed in between. And floors reach three of the suite's thirteen
metric families (`#218`), all produced by one estimator: the data-sampling
bootstrap inside the checkpoint evaluation runner, whose question is whether a
different draw of evaluation games would have said the same thing.

That leaves the decisive quantity unmeasured. A data-sampling floor qualifies
the measurement; nothing qualifies the training. Two models trained from
different initialization seeds differ, and the project had never measured by how
much, so "the number moved" and "the change moved the number" were
indistinguishable by construction.

### What Seed Variance Actually Measures

Measured rather than assumed, on the project's CUDA host. Five arms of one
training configuration, 8,000 steps each, identical in everything but the
initialization seed, each scored by the checkpoint evaluation runner over a
400-game view of the frozen pool.

Two arms at **one** seed are a null. No floored metric moved past four percent
of its floor, and every headline value agreed to at least four significant
figures. A separate pair under strict determinism produced identical validation
output, which is the cheap exemption's evidence.

Two arms at **different** seeds are not. Across the six pairs of four seeds, up
to 14 of the 54 floored metrics cleared their data-sampling floor, and one pair
read *better* on every one of the twelve held-out and sixteen legality metrics —
a uniformly better model produced by nothing but initialization. Spread across
the four seeds, against the data-sampling floor of the same reading: held-out
move loss 1.3x, mask penalty 1.7x, middlegame mask penalty 1.9x.

The training floor characterized from those four arms is 2.1 to 7.3x the
data-sampling floor of the same metrics. Recorded beside them, it reclassifies
every one of those seed-only deltas as noise, without any change to the
reporting machinery.

For scale, `#218` records a real improvement — one run's held-out move loss
between steps 100 and 8,000 — of 2.61, which is eight times the training floor
measured here. A large effect survives this bar comfortably. A small one is not
distinguishable from seed luck with one arm per side, and now there is a number
saying so instead of an intuition.

The absolute values belong to a 276,002-parameter model at 8,000 steps and do
not transfer. The finding that does is structural: the floor that qualifies a
checkpoint delta today is not the one that decides whether a change caused it.

The whole reading — five arms trained, five scored, one training floor
characterized — took about half an hour of one host at that scale, which is why
the process asks for the control rather than recommending it. The training is
what makes a comparison expensive at real scale, and the reading is not.

## Decision

### The Baseline Is A Control Arm

A change that decides what a training run learns is read as a delta between a
control arm trained without it and a treatment arm trained with it, identical in
configuration, corpus, seed, device, and step budget. The most recent recorded
reading is not an admissible baseline for that claim, because a delta against it
confounds the change with everything else that moved since.

The control is not per change. It is a property of a configuration and a
machine, so a run of changes tested against one base pays for it once, and the
converged baseline `#181` establishes is itself a control the work after it
reads against.

### Both Arms Are Read The Same Way

The reading is the default reduced sweep at the same checkpoint step on one
machine, and the claim — which metric moves, in which direction — is written
down before either arm runs. The expensive half of a comparison is the training,
not the reading, so there is no case for reading the arms differently or
narrowing the benchmark set to the family expected to move: a win that costs
something elsewhere is exactly what a suite is for.

Reduced rather than full, because a reduction only cuts sample counts and the
floors widen to match, so a less precise reading of both arms raises the bar
rather than lowering it. The exception is structural rather than a matter of
precision: a benchmark whose cost is a grid has no reduced form, and a claim
about a family only the full sweep reads — strength, which lives in the
ladder — has to be read there or it is not being tested at all. Requiring the
full sweep for every comparison instead would put an hours-long ladder on both
arms of every model change, and a process nobody can afford is one nobody runs,
which is the failure this record exists to prevent.

Arms are recorded into a machine-local store rather than the committed one.
Committing a candidate is a separate decision about project history, and an arm
nobody adopted would otherwise become a later report's default baseline.

### Admissibility Is Bounded By The Floors That Exist

A delta on a metric that has a floor must clear it. That is necessary and not
sufficient: today's floors are data-sampling floors, so clearing one establishes
that two models differ, not that the change is the reason.

So a claim rests on a delta far enough outside seed variance that nothing else
explains it, or on a training floor characterized for the metric it claims,
which the existing noise command already produces from arms trained at several
seeds. Like the control, that characterization belongs to a configuration rather
than to a change, so a base worth several changes pays for it once.

A characterized training floor is committed to the store like any other. It
describes the configuration its arms shared, and it now records that
configuration and is resolved only within it; decision 0040 owns that scope and
the reasoning, and until it existed such a floor was read beside its comparison
and withheld from the store.

A narrower delta is reported as not distinguished from seed variance rather than
presented as an improvement, and a family with no floor at all can show that
nothing else moved but cannot carry the claim.

This is deliberately written in terms of what the store can qualify rather than
in terms of a particular estimator. `#223` may retire the paired estimator and
reshape the floor system; the rule above survives that, because it asks whether
a floor covering the delta exists rather than how it was computed.

### A Null Result Is An Outcome

The pull request lands with the reading it got, and the issue closes having
answered its question. Arms are not re-run in the hope of a better number, and
the reading is not widened at a larger view because the first was inconclusive;
both are the same failure, and the view is sized from the effect the claim names
before the arms run rather than from the answer they gave. A re-run is
legitimate only when the first was faulty for a stated reason, and then both
readings are reported. Where a change was worth having only if it
improved the model, a null reading removes it rather than merging it disabled,
which is what `docs/design-principles.md` already says about surface that does
not win.

Without this the process inverts into its own opposite: "train and measure"
becomes "retrain until the number looks good", and a stated expectation is what
makes that visible rather than deniable.

## Consequences

A model change now costs two training runs rather than one the first time it is
tested against a given base, and less than that afterwards. That is the price of
attribution, and the alternative was not cheaper — it was a claim nobody could
check.

The rule does not make every delta decidable, and it is not meant to look like
it does. Most metric families state no resolution at all, and one arm per side
cannot resolve a small win in any of them. What changes is that this is now
visible in the pull request instead of being absorbed into a confident summary.

The cheap exemption carries much of the traffic. A change inside the trigger
paths that is meant to leave the weights alone — a loader representation, an
instrumentation path, a refactor — establishes that with two short runs at one
seed under strict determinism, which costs minutes. The efficiency work this
milestone is made of is mostly of that kind, so the expensive reading is owed
less often than the trigger's breadth suggests.

Enforcement is guidance rather than a required check, because a training run
cannot be one. The trigger is stated in `AGENTS.md`, which an agent reads during
ordinary work, and the reading lands in a pull request the maintainer is already
reading. A diff that touches the trigger paths and reports no reading is
machine-checkable in the same shape as the existing agent-guide check, and that
is deliberately held until the guidance is observed to fail.

## References

- `docs/evaluation.md`
- `docs/issue-workflow.md`
- `docs/training-and-runtime.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/decisions/0025-machine-scoped-execution-noise-floors.md`
- `docs/decisions/0026-conservative-dispersion-bounds.md`
