# 0065: A Frozen Ablation Vehicle Is The Base A Seed Floor Can Live On

Date: 2026-08-15

## Status

Accepted. Refines `0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`
in one narrow place: the training floor it removed becomes available again for a
single configuration that cannot move. Nothing else 0043 decided is reopened, and
the four kinds, six producers, two storage tiers, and fingerprint-keyed index it
collapsed stay collapsed.

Rests on `0029-model-change-control-arm.md`, which measured seed variance and
named the base this record creates, and on
`0063-the-full-sweep-decides-a-change-and-the-canonical-line-is-its-byproduct.md`,
whose training-identity digest is what freezes the vehicle by machine rather than
by convention.

`0067-a-horizon-is-a-branch-not-a-restart.md` rests on this record and settles
the learning-rate schedule the frozen configuration carries, ahead of the freeze
rather than after it. It also qualifies the exact-digest guarantee below: the
horizon is outside the digest, so a cooldown branched at a different one matches
the vehicle's floor without having been shown to share its spread.

`docs/scaling.md` states the resulting rule and owns the program the vehicle
serves.

The architecture the vehicle freezes is designated by
`0066-the-trunk-sees-the-rating-and-the-board-keeps-its-shape.md`, which landed
first for the reason this record names below: a fundamental rethink after the
freeze leaves the vehicle a base nobody wants to compare against.
`0070-one-decision-per-pass-and-history-in-the-token-depth.md` is the rest of
that rethink and lands ahead of the freeze for the same reason, and it is what
settles the layer budget the frozen configuration carries.

`0071-the-target-is-the-size-the-published-ladder-flattens-at.md` fixes the
target this record says the vehicle is derived from, and carries the throughput
measurements that price an arm at each candidate vehicle width.

`0074-the-vehicle-freezes-on-bfloat16-with-tf32-without-a-quality-reading.md`
fixes the arithmetic the freeze carries into every arm read against the vehicle,
and records that it was adopted without the seed-floored quality reading this
record's own base is what would have made possible.

`0075-a-training-batch-is-decisions-not-games.md` lands ahead of the freeze for
the reason this record names below, and fixes the shape of the batches the frozen
configuration draws.

`0076-the-vehicle-is-width-128-at-the-target-regime.md` is the
designation this record calls for. It fixes the width, horizon, selection, and
peak rate the frozen configuration carries, and pins the digest by the test this
record requires.

## Context

Three records converge on a gap none of them could close alone.

**0029 measured the quantity and named the base.** Five arms of one training
configuration, identical but for the initialization seed, showed up to 14 of 54
floored metrics clearing their data-sampling floor on seed alone, and one pair
reading better on every one of the twelve held-out and sixteen legality metrics —
a uniformly better model produced by nothing but initialization. It drew the
correct conclusion about where such a characterization belongs: "to a
configuration rather than to a change, so a base worth several changes pays for
it once." The whole reading took about half an hour of one host.

**0040 scoped stored training floors to the configuration they measured**, which
is the only scope under which they mean anything.

**0043 removed them, and was right to.** Its objection is not about the
arithmetic, which it accepts, but about the workflow: a training floor is keyed
to a training identity that includes the learning rate, the precision, and the
model, "so it survives exactly until a change is accepted: characterize on
configuration A, test B against an A arm, adopt B, and the next comparison is C
against B, which no floor describes. Re-characterizing costs five training runs.
At the step budgets this project is heading for, that is weeks per accepted
change."

What that argument establishes is narrower than it first reads. It is not that a
seed floor is unaffordable, nor that it is unwanted — 0043 explicitly keeps the
door open for "arms at several seeds, read once, for a result worth the cost."
It is that a floor keyed to an **iterative baseline** can never be current,
because the baseline is redefined by every success.

So the missing piece was never the floor. It was the thing 0029 asked for and
nothing ever built: a base worth several changes that does not move. Every
configuration this project has trained has been either a throwaway proof or a
step in the canonical line, and a step in a line is precisely what cannot carry
a floor.

Milestone 5 makes this binding rather than merely untidy. A scaling program
compares many candidate changes against one another, and its central failure
mode is attributing to a change a delta that seed luck produced — the exact
failure 0029 measured and 0043 left uncovered. Running that program on an
iterative baseline means every comparison in it is unqualified on the term most
likely to fake a narrow result.

## Decision

### The Vehicle Is An Instrument, Not A Step In The Line

**One training configuration is designated the ablation vehicle, and it is
frozen.** Candidate changes that alter what a model learns are read as an arm
against it. Adopting a change does not modify it.

That last sentence is the whole of what this record adds, and it is what makes
0043's objection stop applying. Promotions go to the canonical line, which
continues to work as 0063 describes; the vehicle is not on that line and is not
advanced by a promotion. A base that no success redefines is one whose floor is
characterized once and is still current an arbitrary number of accepted changes
later.

The vehicle is sized as a measuring instrument rather than as a product: large
enough that its readings sit in the regime the target scale will occupy, small
enough that an arm is cheap enough to spend on a question that may return
nothing. `docs/scaling.md` owns how that size is derived and why it is derived
from the target rather than chosen.

### Its Identity Is Pinned By Machine

`training_sha256` already carries model, corpus, arithmetic, encoding, and action
vocabulary, and 0063 wired it into the result envelope for exactly this class of
question. The vehicle's digest is recorded as a checked-in constant and asserted
by a test.

Freezing by convention would fail silently and in the worst possible way: an
edit to the vehicle config invalidates every comparison ever read against it,
and nothing about a later reading would look wrong. A test that fails loudly is
the only form of this rule that survives its own first violation.

### Its Seed Dispersion Is Stored Against That Digest

The floor is characterized once from arms differing only in seed, and is stored
keyed to the vehicle's digest. A comparison either finds the floor recorded for
the digest both its readings carry, or reports that it has none.

That negative check is the same shape 0063 established for the identity header,
and it is what keeps 0040's scoping concern answered without 0040's machinery: a
floor that can only be found by exact digest cannot be silently applied to a
configuration it did not measure.

### What It Qualifies, And What It Does Not

A delta read on the vehicle is qualified by the combined evaluation floor 0043
owns **and** by the vehicle's seed floor. Clearing both is what this project has
lacked, and it is a sufficient basis for a claim in a way that clearing the
evaluation floor alone is not.

Two limits are stated here because they will otherwise be assumed away.

**The floor describes the vehicle, not the arm.** It is measured on baseline
arms, and the delta it qualifies has a treatment arm whose own dispersion is
assumed to match. That assumption fails where the change itself affects training
stability, and the failure is one-directional: an unstable arm has a wider
spread than the floor allows for, so the floor reads too narrow and a noise delta
can clear it. `training_health.gradient_norm` and
`training_health.update_to_weight_ratio` are already read on every arm; a
treatment whose health readings depart from the vehicle's is the case where the
shared floor does not apply, and the comparison says so rather than quoting it.

**A vehicle-scale result is not a target-scale result.** What transfers well is
the ordering of candidates whose effect is a constant offset. What transfers
poorly is anything whose size depends on scale, and nothing here measures which
of the two a given candidate is. `docs/scaling.md` owns that distinction and
what it costs to resolve.

## What This Gives Up, Deliberately

**Every comparison is read against a baseline that goes stale.** Under an
iterative baseline, change B is tested against a model that already carries
change A, so an interaction between them is visible the moment it exists. Against
a frozen vehicle it is not: N one-at-a-time arms yield N main effects and no
interaction terms at all.

That is accepted rather than overlooked, and it is paid for rather than ignored:
a set of individually accepted changes is run as one further arm before any of
them reaches the canonical line, which is the only reading that says the set
composes. The cost is one extra arm per adopted set rather than per change.

It is also worth being plain that the iterative baseline does not actually buy
the interaction term it appears to. It confounds it instead — B tested on top of
A moved by the interaction and by B, with no way to separate them — and it
cannot carry a floor while doing so. The frozen base gives up a term that was
never cleanly measured and gains one that was never measured at all.

**The vehicle's own configuration is a guess that hardens.** It is chosen before
the ladder that would say what the target scale should be, so it may turn out to
sit at the wrong size. Changing it then costs every comparison read against it.

The alternative — deferring the vehicle until the ladder is fitted — is worse in
a way that is easy to miss: the ladder is itself a set of training comparisons,
so deferring the instrument until after the program that needs it means running
that program unqualified. A vehicle at a defensible size now beats an optimal one
after the decisions it was meant to inform.

**Half an hour becomes hours.** 0029's characterization was cheap because the
model was 276,002 parameters. A vehicle sized to be predictive of a real target
is larger, and its five arms cost proportionally more. This is still the cheapest
qualification the project can buy, and it is bought once rather than per change,
which is the entire point.

## Consequences

**0043's collapse holds.** One dispersion per reading, combined at comparison
time, remains how every evaluation floor works. This record adds one number
attached to one digest — not a kind, not a producer, not a tier, and not an
index.

**The seed exemption in the process narrows.** `docs/issue-workflow.md` currently
routes every claim through "a delta far enough outside seed variance that nothing
else explains it," because no floor could say what seed variance was. For a
comparison read on the vehicle, it now can, and the process says so.

**Comparisons read anywhere else are unchanged.** A canonical-line checkpoint
delta, a corpus-scale reading, a target-scale run: none of these carries a seed
floor, and none of them is claimed to. The negative check makes that visible in
the report rather than leaving it to be remembered.

**The vehicle is designated once and its size is a decision.** It is not derivable
from anything the repository currently holds, because it follows from the target
scale, which follows from the compute budget and the deployment envelope rather
than from a measurement.

## References

- `0029-model-change-control-arm.md` — the seed measurement, and "a base worth
  several changes pays for it once"
- `0040-training-noise-floors-are-scoped-to-the-configuration-they-measured.md` —
  superseded by 0043; its scoping concern is answered here by the digest key
- `0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md` — why a
  floor on an iterative baseline is unusable, and the door it left open
- `0063-the-full-sweep-decides-a-change-and-the-canonical-line-is-its-byproduct.md`
  — `training_sha256`, and the canonical line the vehicle is deliberately not on
- `docs/scaling.md` — the program the vehicle serves, and how its size is derived
