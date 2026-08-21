# 0067: A Horizon Is A Branch, Not A Restart

Date: 2026-08-16

## Status

Accepted. Rests on
`0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`, whose
freeze is the whole of why this is decided now: the schedule is part of the
training configuration the vehicle's digest covers, so it is a configuration edit
today and an invalidated instrument afterwards.

`docs/scaling.md` states the resulting rule and owns the program it serves.
`#493` is what makes the rule expressible in configuration and in the runner.

`0076-the-vehicle-is-width-128-at-the-target-regime.md` instantiates this
family in the frozen configuration, and inherits the qualification below that the
horizon sits outside the digest: an edit to the step budget alone leaves the
identity intact, so a test asserts the horizon separately.

## Context

Two schedule families are available, and the evidence in
`docs/research.md` (Scaling And Capacity) reports comparable final loss between
them. That is the least interesting thing about the choice, and reading it as the
whole of the comparison is how a project ends up picking on habit.

What separates them is what a horizon change costs afterwards.

**A decay shaped to a fixed horizon is invalidated by changing that horizon.**
The shape is a function of the step count, so a different step count is a
different schedule from step one. The run that already happened cannot be
extended and cannot be truncated: every question of the form *what if it trained
longer* is a fresh run from initialization.

**A constant trunk followed by a short cooldown is not.** The trunk is the same
whatever the horizon, so a horizon is chosen by branching a cooldown off the
trunk wherever the trunk has reached. The same source estimates that the
branching structure roughly halves the cost of fitting a scaling ladder, which is
not a marginal saving on a stage this project has not yet paid for.

This project's exposure to that is concentrated rather than diffuse. Milestone 5
fits a size-versus-data ladder whose data axis is exactly a set of horizons, and
`docs/planning/roadmap.md` already prices that axis as one run branched at several
horizons. Under a horizon-matched decay it is several runs, and the difference is
the largest single compute item in the stage.

The remaining question is why it has to be answered before the vehicle exists.
The vehicle is one training configuration, frozen, with its identity digest
pinned by a test and its seed dispersion stored against that digest. A schedule
added or changed after the freeze changes that digest, which is the loud failure
0065's pin exists to produce, standing in front of the quiet one it exists to
prevent.

## Decision

### The Project Trains Under A Constant Trunk With A Cooldown

**One family, for every run the program takes: a warmup, a constant trunk at the
peak rate, and a cooldown over the final fraction of the run, decaying to zero on
a square-root-shaped curve.** That shape beat the linear one it was compared
against in the published work; nothing here re-measures it.

The trunk is what the decision is for. Everything else in the family is settled by
taking the published answer, in the sense `docs/scaling.md` means by that: the
comparison exists and was run with more compute than this project will spend.

### The Cooldown Is A Fraction Of The Run

**Its length is declared as a fraction of the configured horizon, never as a step
count.** A step count survives a horizon change syntactically and not
semantically: the same number of steps is a different share of a longer run, so
the shape silently changes into one nobody chose and no error fires. A fraction
moves with the horizon by construction.

The fraction itself is a value the configuration owns, set within the range the
source supports — its cooldown benefit plateaus at a modest share of total steps,
so the useful range has a top and paying past it buys nothing.

### Warmup Is Denominated In Data, Not In Steps

**Warmup ends after a declared quantity of training data, converted to a step
count by the run's own batch and accumulation.** Two separate things force this,
and the second is the one that would be missed.

A fixed warmup step count is one of the four confounds
`docs/research.md` (Resolving Discrepancies In Compute-Optimal Scaling)
isolates, and that work's central finding is that every intermediate fit was
statistically well-behaved — the wrong answer was not detectable from the fit, only
from ablating the protocol.

The second is structural, and it rules out the obvious repair. **A warmup
expressed as a fraction of the run destroys the property the family was chosen
for.** Branches of one trunk differ in their configured horizon, so a
horizon-relative warmup gives each branch a different prefix, and two runs with
different prefixes are two runs rather than one trunk read at two points. The
warmup has to be invariant to the horizon or the branch is a fiction.

The unit is the sequences the loader delivers rather than positions, and the
reason is a property of this codebase rather than a preference. Sequences per step
follows from the declared batch and accumulation, so the boundary is computable
before a run starts; positions per step is not, because games vary in length and
batches are padded. Reading the position counter every step to find the boundary
is worse still: the runner's deferred totals exist precisely so that a step does
not synchronize with the device, and a per-step position read would reintroduce
the cost they were built to avoid. Sequences and positions differ by a factor the
corpus fixes, so a declaration in sequences is a declaration in positions for as
long as the corpus holds.

**The rule is recorded with the range it is valid over**, which `docs/scaling.md`
states. Outside it the confound above is back — a run that spends a third of
itself warming up is not measuring what it thinks — and a short branch is where
that bites, so a configuration exceeding the range is refused rather than quietly
reshaped.

No width term. Nothing found establishes how warmup should move with model width,
and inventing one would be the failure this project's own rule about rules warns
about. `#489` is where a fitted rule against scale would put one, and doing so
disturbs nothing frozen: the vehicle holds one configuration, and a later rule
decides what a new run sets rather than what the frozen one holds.

### A Branch Is A Resume, And The Configuration Already Carries One

A branch is the trunk's checkpoint at the step where the cooldown should begin,
resumed under a configuration declaring that branch's own horizon. Nothing new is
needed for that: `resume_from` exists, and both it and `steps` are outside the
identity `_compatibility_record` digests, so a branch carries the trunk's training
digest and differs from it only in where it stopped and in what it resumed from.

**A branch restores no schedule state**, which is the failure this family is most
exposed to and why `docs/training-and-runtime.md` states that a resume recomputes
the rate: a checkpoint already carries a `scheduler_state` slot, and a branch that
filled it would inherit the trunk's horizon, cool at the trunk's boundary rather
than at its own, and report nothing wrong.

That is also the check this decision owes, and it passes: the family needs a peak
rate, a horizon-independent warmup input, a cooldown fraction, and a resume that
can carry its own horizon, of which the configuration holds the first and the last
today and `#493` adds the two in the middle. A family needing something the
configuration could not carry would have been the wrong choice however it read on
loss.

## Consequences

**A mid-trunk checkpoint is not a cooled one.** Under this family the peak rate is
still applied at every step of the trunk, so a checkpoint taken there is a run
stopped at a high learning rate — the same defect as a decay that did not match
the horizon it ended on, which is one of the four confounds rather than a cheap
extra data point. A point standing for what a horizon achieved is therefore taken
at the end of a cooldown, and the ladder's data axis is a set of branches rather
than a set of steps of one trunk. What this rules out is mixing the two kinds:
comparing two uncooled checkpoints of one run is unaffected, since both sides sit
at the same rate, and an in-training preview cadence is unaffected because it
exists to show a run is alive rather than to rank anything.

**Branched and from-scratch points do not go in one fit.** The source is explicit
that a branched cooldown's loss is not identical to a from-scratch run at the same
horizon, so a curve mixing them fits the difference between two protocols along
with the effect it was reading. One or the other, stated in the reading.

**The digest does not distinguish a branch from its trunk.** That is deliberate
and follows from `docs/design-principles.md`: the horizon is a coordinate the
program measures differences across, so it cannot be part of identity. It has one
consequence worth stating before `#488` characterizes anything, because it will
otherwise be assumed away — the vehicle's seed dispersion is measured at the
vehicle's horizon, and a branch at a different horizon shares its digest without
having been shown to share its spread. A floor found by digest alone is quoted for
a reading it may not describe, so a comparison across horizons says which horizon
each side read.

**The final run stays extendable.** Training longer is a further stretch of trunk
and a new cooldown, so the horizon chosen for the target run is not a one-way
door. Under the other family it is.

**Reversing this costs the instrument, not the schedule.** Before the vehicle is
frozen the cost is an edit and any readings taken since. After the freeze the
schedule is inside `training_sha256`: the pin test fails loudly, and what has to
be rebuilt is the vehicle's seed floor — five arms at vehicle scale, per
`0029-model-change-control-arm.md` — plus every candidate arm already read against
the old digest, none of which transfer. Switching to a horizon-matched decay
additionally re-prices every later horizon question at a fresh run, which the
source puts at roughly twice the ladder's cost. That asymmetry is why the decision
is taken now rather than when the evidence would be better.

## What The Evidence Does Not Establish

The supporting work reports validation loss on models smaller than this project's
target and runs no downstream benchmarks. So the equivalence it establishes is
between the two families' *losses*, and this project decides candidate changes on
benchmark readings. **Nothing found says the two families rank candidates the same
way on anything but loss**, and this decision is taken with that stated rather
than treated as covered.

The cooldown's plateau is read off their models too. The shape and the range are
taken as prior art, not as measurements here.

What would reopen it is a benchmark result whose leading explanation is the
schedule family itself, which is a narrow condition and deliberately so. Answering
it costs two arms at vehicle scale, one per family, read on the suite rather than
on loss — worth spending only where a real result hangs on it, and not worth
spending to reassure anyone in advance.

## Alternatives Considered

**A decay matched to a fixed horizon.** The published ladder protocols use it, and
the confound work that this project's ladder is built to satisfy is written in its
terms. Rejected because its cost lands entirely on the axis Milestone 5 is about:
every horizon is a fresh run, the target run cannot be extended, and the coupling
table's hardest cell — a horizon change invalidating the schedule outright —
belongs to it rather than to the trunk. Its one genuine advantage is that a
mid-run checkpoint under a matched decay is closer to a usable point than a
mid-trunk one; that advantage is bought back here by branching, at the cost of one
short cooldown per point.

**Warmup as a multiple of the optimizer's second-moment timescale.** Attractive
because it is derived rather than fitted, which is the tier `docs/scaling.md`
prefers, and because it moves with the batch through a quantity `#493` is adding
anyway. Rejected because it collapses where the second-moment decay is set for
stability rather than as a timescale in positions: at a decay of 0.95 one
timescale is twenty steps, which is a warmup in name only, and a rule that yields
that is a rule the configuration would have to override.

**Deciding the family after the ladder says which is better.** The ladder is the
program that needs the family, so this defers the decision past the readings that
pay for it, and 0065 already settles that an instrument arrived at after the
comparisons it was meant to qualify is not an instrument.

## References

- `#486` — the decision this answers; `#493` — the surface that makes it
  expressible; `#487` and `#488` — the freeze and the floor it precedes
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md` — the
  freeze, and what a digest change costs after it
- `0029-model-change-control-arm.md` — the five arms a re-characterization costs
- `0063-the-full-sweep-decides-a-change-and-the-canonical-line-is-its-byproduct.md`
  — `training_sha256`, and the canonical line a cooled endpoint is compared on
- `docs/scaling.md` — the coupling table row this sets, and the order it belongs to
- `docs/research.md` (Scaling And Capacity) — the schedule comparison and the four
  confounds
