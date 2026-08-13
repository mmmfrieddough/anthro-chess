# 0043: A Delta Floor Is Combined From The Two Readings It Compares

Date: 2026-08-09

## Status

Accepted. Supersedes `0033-pairing-is-a-correctness-fix-not-a-resolution-lever.md`,
`0035-a-degraded-floor-is-annotated-rather-than-withheld.md`,
`0036-a-one-sided-floor-does-not-qualify-a-delta.md`,
`0040-training-noise-floors-are-scoped-to-the-configuration-they-measured.md`,
`0042-the-puzzle-response-is-qualified-within-its-reading.md`, and
`0025-machine-scoped-execution-noise-floors.md`, whose scoping rule has nothing
left to scope once no floor is stored. Supersedes the estimator half of
`0028-qualifying-the-rating-dependency-family.md`, whose ruling on which of that
family's quantities can be resampled at all is retained. Rests
on `0026-conservative-dispersion-bounds.md`, which is retained unchanged.
Extended by `0061-a-training-cost-reading-has-no-replicate-to-resample.md`,
which answers the obligation below — a dispersion or a stated reason — for the
training-efficiency family.

## Context

The floor system grew one kind at a time, each answering a real question. It
arrived at four kinds — evaluation, data-sampling, training, execution — six
producers, two storage tiers, a fingerprint-keyed index, bridge-aware lookup, and
scope rules that decide per kind whether a floor describes the delta in front of
it. Roughly 2,000 lines and seven decision records.

What that bought, measured against the question the project actually asks, was
less than it looks.

**The kinds are nested, not independent.** A training floor is the spread of
readings across seed-replicate models, and each of those readings carries its own
sampling noise, so the training floor already contains the data-sampling floor.
Decision 0029 measured the ratio at 2.1x to 7.3x. Since a report binds the widest
applicable floor, a training floor wins by that factor wherever one exists, and
the narrower kinds cannot change the verdict. The paired estimator — a refinement
that makes the sampling floor about 1.9x narrower still — is the clearest case: it
is arithmetic that cannot reach an answer.

**The scoping made the useful kind unusable.** A training floor is keyed to the
training identity, which includes the learning rate, the precision, and the model.
So it survives exactly until a change is accepted: characterize on configuration
A, test B against an A arm, adopt B, and the next comparison is C against B, which
no floor describes. Re-characterizing costs five training runs. At the step budgets
this project is heading for, that is weeks per accepted change, and the ordinary
workflow — adjust one thing, keep it, adjust the next — is the case it fails on.

**The complexity was not free.** The scope rules, the index, the two storage
tiers, and the four kinds are the part of this repository that has been hardest
to hold in view, and the queue reflects it.

Meanwhile the thing that qualifies a delta was always available and nearly free.
Every benchmark run already resamples the units it scored and computes a
dispersion. That number, kept, is enough.

## Decision

**One dispersion per reading, combined at comparison time.**

A benchmark scores a checkpoint and reports a value that stands on its own,
depending on nothing but that checkpoint, the benchmark logic, and the data. It
additionally reports one number per metric: the dispersion of that metric under
resampling of the units it scored.

Comparing two checkpoints combines the two dispersions the readings carry:

```
floor = coverage * sqrt(sigma_a**2 + sigma_b**2)
```

with each dispersion bounded first, by decision 0026, which is unchanged. This is
not a new quantity. The previous arithmetic was `coverage * sqrt(2) * sigma`, and
the `sqrt(2)` is `sqrt(sigma**2 + sigma**2)` under the assumption that both
readings have the same dispersion. Removing that assumption is the whole change,
and the formula reduces to the old one when it happens to hold.

There are no floor kinds for model quality, no characterization records, no
index, no scope rules, and no paired estimator. A delta is qualified by the two
readings in front of it or not at all.

Efficiency measures its dispersion differently, and stores it the same way.
Latency, throughput and training-step metrics vary with the machine rather than
with the sample, and nothing inside a reading estimates that, so the inference
benchmark measures its own by running itself again in several processes and
reports the spread beside its value. Same shape, same field, same arithmetic at
comparison time; only the estimator differs.

That removes 0025 rather than narrowing it. Scoping a floor to the machine was
how a *stored* floor avoided being applied where it did not belong, and it is
unnecessary once the dispersion is measured beside the reading it qualifies and
never applied to a second one. It also closes a gap 0025 could not: a stored
floor is scoped to the machine but not to the moment, and #161 measured a
characterization taken on a quiet machine licensing four times as many false
findings once the machine was hot — further than the dispersion bound moves
anything. Measuring in the same session is one of the shapes #161 named.

## What This Gives Up, Deliberately

**The floor ignores the covariance between two readings.** Two checkpoints scored
on one frozen pool share their draw, so the variance of their difference is
`sigma_a**2 + sigma_b**2 - 2*cov`, and dropping the covariance term reports a
width strictly wider than the truth. Decision 0033 measured that cost at about
1.9x, and decision 0042 identified the puzzle family as its extreme case, since
every checkpoint is scored on the identical set. Both are superseded here, and
both were right about the arithmetic: real improvements will read as noise more
often than a paired estimator would have let them, and the puzzle response is
where that will show first.

0042 is superseded in its routing rather than refuted in its reasoning, and it
named this path itself: having declined to attach its spreads as floors, it
records that if a cross-checkpoint qualifier is ever wanted, the cheap form is
the one to build — attach the within-reading spread through `bounded_floor`,
"which errs wide, the direction that costs findings rather than invents them".
This is that form, taken everywhere rather than for one family, with the
equal-dispersion assumption in its `sqrt(2)` removed. What 0042 estimates is
retained in full: a replicate redraws once and refits every configured rating
together, the draw is stratified and rescaled per 0039, and a quantity no redraw
moved reports no spread. Only its conclusion that such deltas stay `unknown` is
reversed.

That estimator shape is also why the paired approach could not have been kept
here in any case. A fitted rating is a nonlinear functional of the whole draw,
and pairing retains per-unit values and reduces them by a mean, so it cannot
reach the response metrics at all — as 0042 observes, the ladder refits the same
way and cannot pair either.

That is accepted. A floor that is too wide costs power; a floor that is wrong
about what it covers costs a claim. The maintainer's requirement is a bar that is
always available and always means one thing, and paying about 1.9x for it is the
price of not carrying an estimator that needs a machine-local detail tier, two
matched readings, and a rule for what to do when either is missing.

**The floor does not cover training-seed noise.** It is computed from a reading's
own units, and seed variance is a property of the training run rather than of the
benchmark, so nothing here can see it. Decision 0029 measured what that means: two
arms differing only by initialization seed cleared 14 of 54 floored metrics, and
one pair read better on every one of the twelve held-out and sixteen legality
metrics.

So a delta that clears its floor is **not** established as caused by the change.
The floor is a necessary condition and not a sufficient one: it rejects deltas
that are indistinguishable from benchmark noise, which is most of what a reader
would otherwise over-read, and it is silent on seed. Establishing that a change
caused a delta remains a deliberate act — arms at several seeds, read once, for a
result worth the cost — rather than machinery riding on every comparison. The
report says which of the two it is showing.

## Consequences

A benchmark that reports a value reports a dispersion beside it, or states why the
metric has none. That is the one obligation this design adds, and it replaces
thirteen families' worth of "no floor exists" with a uniform requirement.

Graphs and budget planning read the same stored dispersions. A band between two
adjacent points on a history plot is exact rather than nominal, and the games a
benchmark needs to reach a target resolution follows from any recent dispersion by
the inverse square root, so no separate benchmark-level constant is declared,
stored, or kept current.

`docs/evaluation.md` owns what a reading claims. `docs/issue-workflow.md` owns
what a pull request says about it.
