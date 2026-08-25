# 0063: The Full Sweep Decides A Change, And The Canonical Line Is Its Byproduct

Date: 2026-08-15

## Status

Accepted. Refines `0029-model-change-control-arm.md`, which required a freshly
trained control and ruled out every prior reading as a baseline.

Refined by `0079-one-declared-size-per-benchmark.md`, which withdraws the
reduced sweep the three-way separation below names. What survives is the
separation between a recorded sweep and a shakedown, and the rule that a claim
is decided on the sweep at its declared sizes.

The comparability, storage, and generated-workload records are read here rather
than changed. Nothing below relaxes one of them; the fingerprint rule in
particular is re-examined and kept, and the section that does so says why.

`0064-the-complete-round-robin-is-the-optimal-ladder-design.md` answers the
pressure the consequences below put on the rating ladder: its structure is not
where that cost comes off, so the seeds lever named there is the whole of what
is available rather than an interim measure.

`0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md` reads the
canonical line established here as the reason a second base can stay frozen:
promotions advance this line and leave the ablation vehicle alone, which is what
lets that vehicle carry a seed floor that stays current. The `training_sha256`
digest wired in here is what pins it.

## Context

Decision 0029 answered a real gap: nothing said how a change showed it improved
anything, so every change decided its own evidence standard. Its answer was a
control arm — a run trained without the change beside one trained with it — and
its prohibition was blunt, because the baseline it was written against was
blunt: *the most recent recorded reading*, which in this project's history meant
a different corpus, a different step budget, a different machine, and every
other change that landed in between.

That prohibition is correct about an arbitrary prior reading and over-strict
about a deliberately sequential one. The workflow this project intends is one
change at a time: benchmark the candidate, compare it against the current best,
drop it or promote it, then make the next change. Under that discipline the
prior checkpoint differs from the candidate in the change under test and the
initialization seed, and in nothing else.

Requiring a fresh control there doubles training to re-derive a run that already
exists. What that purchase buys is protection against drift. What it does not
buy is anything at all on the seed, because both arms differ by seed under
either arrangement — which is the term decision 0043 established no floor can
see, and the one most likely to carry a narrow delta.

So the expensive half of the control arm addresses a risk the workflow already
forecloses, and the cheap half addresses nothing. Meanwhile the digest that
would *verify* the workflow's discipline is already recorded on every result and
read by nothing: `training_sha256`, the checkpoint's compatibility identity,
which arrived at envelope version 5 to scope training noise floors under 0040
and was orphaned when 0043 removed them.

Two further things were undecided. Nothing said which readings constitute
project history, and nothing said which scale writes them — so the sweep that
exists to iterate quickly could append to the store that exists to be looked
back at.

## Decision

### A Control Is A Training Identity, Not A Simultaneous Run

The requirement was always *identical conditions except the change*.
Simultaneous training is one way to guarantee that; it is not the thing being
guaranteed.

**A prior checkpoint is a control when its training identity matches the
candidate's in everything but the change under test.** Where the identities
match, the comparison is a control-arm comparison and no second run is trained.
Where they do not, it is not, and no discipline asserted in a pull request makes
it one.

`training_sha256` is what settles that, and it settles it by machine rather than
by memory: model, corpus, arithmetic, encoding, and action vocabulary are all
inside the digest, and the seed deliberately is not. `#474` wires it into the
comparison, and until it lands the qualification is asserted rather than
verified — which is a weaker claim, and one a reader should be told about rather
than left to assume.

This narrows 0029's prohibition rather than removing it. The most recent
recorded reading is still not a control by default; it becomes one only when the
identity check passes.

Nothing here touches what a control arm does not buy. Both forms differ by seed,
so clearing a floor still establishes that two models differ rather than that
the change is why, exactly as 0029 and 0043 record.

### The Full Sweep Decides, The Reduced Sweep Iterates

Three activities were sharing two scales. They are separated:

- The **full sweep** is the instrument a change is judged on. Both sides of the
  comparison are read at it, and its result is what decides whether a change is
  adopted. Under `0079` that is the sweep, since it has no other size.
- A **capped reading** is a coarse view during iteration. It is never the
  evidence for a claim, and per `#475` it does not write to the committed store.
- A **shakedown** sets its own values when a benchmark lands. It records
  nothing, so it needs no permanent scale and does not get one.

This reverses the previous default, which sent an attribution claim to the
reduced sweep on the grounds that a reduction only widens floors and therefore
*"cannot let a weak claim through"*. That reasoning is sound about false
positives and incomplete about the decision being made. Under a workflow where a
comparison decides adopt-or-drop, a reading too coarse to resolve a real
improvement discards a good change as surely as a loose one admits a bad change.
Conservatism in one direction is not conservatism.

What carries over unchanged is that the scale is part of the claim rather than a
response to it. It is chosen before the arms run, and a delta inside its floor is
a null result rather than a reason to re-read at a larger view.

### The Canonical Line Is Accepted Checkpoints, And Nothing Is Run For It

History accumulates as a byproduct of the comparisons already being made. No
reading is taken for the graphs alone.

Candidate readings go to a machine-local scratch root; a checkpoint's reading
reaches the committed store when the change is accepted. That extends 0029's
existing reasoning — *an arm nobody adopted would otherwise become some later
report's baseline* — from candidate arms to every rejected reading, and it makes
the committed store the line of accepted checkpoints rather than a log of
everything attempted.

This is a convention, not machinery. `ANTHRO_CHESS_RESULTS_ROOT` already points
the store wherever it is told, and nothing needs building to use two roots.

That is true of *which* readings are promoted, and was too broad about the
default the convention ran against: with the store resolving as `results/` when
nothing said otherwise, history acquired a reading by a command being run rather
than by anyone deciding it should, and the convention had to be remembered every
time. `#475` narrows it. The default resolves beneath `ANTHRO_CHESS_RUN_ROOT`,
the way the detail tier already does, and `anthro eval promote` copies one
checkpoint's records into the committed store — a copy, so the machine keeps the
arms a later comparison reads against. Which readings are worth that stays the
maintainer's call, unchecked on purpose: a reduced reading names its own view in
the record, so the pull request being reviewed already shows what it is.

### View Size Stays In The Fingerprint

Removing it was considered, for the subset of metrics that are means over
independent units and are therefore unbiased under subsampling. It is not
adopted.

The classification is the problem. Whether a metric is unbiased under
subsampling depends on its internal estimator rather than on anything the
registry declares: a quantity averaged per position is unbiased where the same
quantity computed from pooled totals is not, and the two are indistinguishable
from the identifier, the direction, or the declared cost. Getting one wrong puts
a metric that reads high at small samples onto one line across view sizes, where
it renders as improvement — a silent false trend in the one artifact the line
exists to provide, with nothing to flag it.

Decision 0013 closed the override for the same reason and said so plainly: a
bridge is *never legitimate when what was measured changed, or when which games
were scored changed*. Both doors are shut deliberately.

**So the history line pins its view rather than tuning it.** A pool-scoring step
whose series must last takes the unbounded view, which has no size dial to
regret. The seams such a line meets are then generation cuts alone — the one
seam with anchor checkpoints re-scored across it, so a shift is attributable to
the pool rather than mistaken for a model regression. That is one seam mechanism
instead of two, and it is the one that already carries its own discipline.

### What Follows For Tuning

The split is not uniform across the suite, and the difference decides what may
be adjusted later:

| knob | reaches the fingerprint |
| --- | --- |
| seed count, games per position, concurrency | no — 0020 keeps sample counts as provenance |
| inference decision and batch counts, and both sweeps | no — headlines sit at one declared reference point |
| a pool-scoring view's game count | yes — the scored games are the data component |
| puzzles per rating | yes — the puzzles scored are the data component |
| a generated arm's opening view | yes — 0020 puts the position source, including the game-id digest, in the declared workload |

Generating more games is therefore free to tune whenever cost or precision
argues for it. Reading a different set of pool games is a once-per-generation
choice. The instinct to tune everything after measuring costs is right for the
first row and permanently destructive for the last three.

For the rating ladder specifically, games per pairing can be raised through
seeds without ending a series or through openings while ending every ladder
series, and 0039 already found the two close to interchangeable for precision.
Seeds are the lever.

## Consequences

A sequential comparison costs one training run rather than two, and the saving is
not a relaxation: the drift the second run guarded against is now checked rather
than assumed, and the check is stricter than a maintainer's recollection of what
changed. Until `#474` lands, a comparison making this claim states that it was
asserted.

The full sweep becomes the routine cost of deciding a change rather than an
occasional reading. That puts sustained pressure on whichever step dominates it,
which today is the rating ladder — `#473` is where that pressure is answered,
and the seeds lever above is what is available in the meantime.
`0064-the-complete-round-robin-is-the-optimal-ladder-design.md` has since taken
it: the ladder's pairing structure is optimal at any budget, so that lever is
not an interim measure but the only one there is.

The reduced sweep's role narrows to iteration, and its readings remain useful and
remain machine-local. `#475` did that by moving where every sweep writes rather
than by refusing the reduced one: a runtime refusal would have made the shipped
reduced sweep unrunnable, since its steps default to recording, and promotion is
the deliberate act either scale passes through.

Nothing here settles two quantities that no design decision can. The magnitude of
seed variance at a scale that matters is unmeasured — 0029's figures are
disclaimed by their own record as belonging to a 276,002-parameter model — and
the distribution of effect sizes this project's changes actually produce is
discovered by making them rather than derived in advance. Both are read off the
comparisons this decision describes, which is the only place they can come from.
