# 0033: Pairing Is A Correctness Fix, Not A Resolution Lever

Date: 2026-08-04

## Status

Accepted. Settles the estimator half of `#223` and supersedes its recommendation
to reconsider the checkpoint-pair machinery. Applied to reporting by
`0035-a-degraded-floor-is-annotated-rather-than-withheld.md`.

## Context

`#223` measured what the noise-floor system buys and reported it as a comparison
of two levers:

| lever | resolution bought |
| --- | ---: |
| paired checkpoint-pair estimator | 1.9x |
| view 400 -> 10,000 games | 4.6x |
| view 400 -> 50,073 games | 10.3x |

and concluded that the sophistication was aimed at the wrong axis: roughly 3,600
lines of machinery buying 1.9x where a configuration value buys 4.6x to 10.3x.

Both numbers are ratios of the same quantity — how many times narrower the floor
becomes — so they are legitimately comparable. The framing around them is not,
and the difference decides the outcome.

## The two are not the same kind of thing

**More games reduces the noise.** Draw 10,000 games instead of 400 and the metric
genuinely moves around less under resampling. The floor shrinks because the
quantity it estimates shrank. Standard error falls as `1/√n`, verified in `#223`
at 2.06x measured against 2.24x predicted.

**Pairing does not reduce noise. It stops over-reporting it.** Both checkpoints
are scored on the same sampled games, so whichever games were drawn, their
peculiarities hit both and cancel in the difference:

```
Var(A - B) = Var(A) + Var(B) - 2 * Cov(A, B)
```

That covariance is large and positive whenever the same sample is used. An
unpaired floor computes `Var(A) + Var(B)` and drops it, reporting a width about
1.9x wider than the delta's actual sampling variability. The paired estimator
measures the width that was always there.

This is the distinction between a paired and an unpaired t-test. The paired test
is not a sharper instrument applied to the same data; it is the correct test when
the data are paired, and the unpaired one answers a question nobody asked —
what the delta would look like had the two checkpoints been scored on
independent samples, which is not how they were scored.

So an unpaired floor is not a less-resolved reading. It is a reading that is
wrong in a known direction and by a known factor, and its error is conservative:
it produces false negatives, reporting real improvements as noise.

## The levers compose, and substitution is expensive

Because they act on different terms, doing both multiplies: pairing at a 10,000
game view is about 4.6 x 1.9 ≈ 8.7x, not 4.6x. Presenting them as alternatives
implies a choice that does not exist.

The 1.9x error is also multiplicative in `n` and does not diminish with more
data. `unpaired(n) = 1.9 x paired(n)` at every sample size. More games shrinks
both together and the gap stays.

An unpaired floor can still reach any absolute width by brute force, since
resolution scales as `√n` and `1.9² = 3.61`:

```
unpaired(3.61n) = paired(n)
```

But cost scales as `n`, not `√n`. Reaching parity means paying **3.61x the games
on every reading, permanently**, and arriving at a width the paired estimator
would have beaten by 1.9x at that same view. Using the fitted evaluation cost
from `#223`, `t ≈ 15 s + 0.118 s/game`, that is roughly 3x the runtime of
`anthro eval run` at a 400-game view and 3.6x at 10,000.

The two cost columns in `#223`'s table are also not the same unit. The estimator
costs 381 lines once. The view size costs suite runtime on every reading for the
life of the project.

## Decision

**Keep both.** They do different jobs and neither substitutes for the other.

- The per-reading bootstrap stays. It is cheap and it is what correctly reports
  that a model near a plateau has stopped improving.
- The paired checkpoint-pair estimator stays. It is the correct estimator for a
  delta between two checkpoints scored on one sample, which is the operation
  model iteration consists of.

Milestone 5 is the argument for the second. Every capacity variant and every
objective arm is a checkpoint-pair comparison, and improvements get smaller as
the model improves — so a floor that is systematically 1.9x too wide matters more
later, not less.

`#223`'s three structural objections are real and are now work rather than
grounds for removal:

- The estimator cannot serve `rollout`, `ladder` or `inference`, which generate
  their own games or measure the machine. There is no shared sample to pair on,
  so this is inherent rather than a coverage gap, and the families it does serve
  are the ones whose readings have been committed.
- Retained contributions live in the machine-local detail tier, so cross-machine
  pairs are impossible. Filed separately.
- The estimator falls back to unpaired silently. Filed separately.

## Consequences

Nothing is deleted, so `docs/evaluation.md` needs no reconciliation and the
triage sweep `#223` anticipated does not happen: `#190`, `#173`, `#161`, `#168`
and `#175` remain ordinary resolution work rather than candidates for
cancellation.

The live question `#223` raised is untouched by this decision and continues
separately: every floor the project has recorded is backed by 400 games out of a
50,073-game pool, and no full-view reading has ever been taken.

This record exists because the framing is the part that misleads. "Buys 1.9x"
reads as an optional sharpening, and a reader who deletes the estimator on that
basis will believe they gave up resolution when what they accepted is a known
1.9x overstatement on the comparison the project most depends on.
