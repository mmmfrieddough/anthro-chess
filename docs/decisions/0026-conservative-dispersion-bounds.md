# 0026: A Floor Is Built From A Bounded Dispersion, Not A Measured One

Date: 2026-07-31

## Status

Accepted. Refines `0025-machine-scoped-execution-noise-floors.md`.

## Context

Every noise floor this project stores reduces to one quantity: the dispersion
of a metric across replicates of a noise source, expressed as a delta at a
declared coverage. That arithmetic was `coverage * sqrt(2) * dispersion`, where
`dispersion` was the value the replicates produced.

The shakedown reading for the execution floor found the defect. Two readings of
one checkpoint, with nothing changed between them, should not produce a delta
that clears its floor more often than the coverage allows. In both configurations
measured, several did — at eight decisions per reading, the p90, p99, mean
latency and throughput deltas all cleared; at the headline thirty, p50, mean
latency and both cold-start metrics cleared. Four or five metrics out of ten is
not the one-in-twenty a 95% coverage factor promises.

The cause is not specific to timing. A measured dispersion is a point estimate
sitting in the middle of its own sampling distribution, so roughly half the time
it lands below the spread it is meant to describe. At six replicates its relative
standard error is about a third. Characterizing one checkpoint's p50 latency in
three sessions minutes apart produced dispersions of 0.14 ms, 0.29 ms and
0.72 ms; twelve readings put the truth near 0.21 ms. The first of those three
would have licensed ordinary machine noise as a finding for as long as it stood.

A floor that understates the noise is worse than a floor that is merely wide.
Wide costs sensitivity, which is visible and recoverable; narrow manufactures
findings, which is invisible and gets acted on. The two errors are not
symmetric and the arithmetic should not treat them as though they were.

## Decision

**Every floor is built from a conservative upper confidence limit on the
dispersion rather than from the dispersion itself.** The limit is the
chi-squared bound on a standard deviation for the degrees of freedom behind the
estimate. Both quantities are stored: the measured dispersion, which describes
the machine or the sample, and the bound, which is what the floor was computed
from and what qualifies a delta.

A floor is consequently a tolerance bound and declares two numbers. **Coverage**
owns the proportion of same-weights deltas covered given a known dispersion, and
keeps its existing 95% two-sided normal factor. **Confidence** owns how sure the
bound is that the dispersion is no larger than assumed, and is also 95%. They
multiply, and the resulting claim is: with 95% confidence, this floor covers 95%
of the deltas that noise alone produces.

**Degrees of freedom count independent replicates, not values.** This is the
part that decides whether the bound means anything, and it differs by estimator:

- a bootstrap floor counts the **games** resampled, not the resamples. The
  resample count is chosen for free and says only how finely an already-fixed
  spread was read;
- a paired floor counts the **matched units**, for the same reason;
- an execution floor counts the **processes**, not the readings. Two readings
  inside one process share an allocator, a warm file cache and a compiled
  kernel, so they are not independent evidence about spread. They still widen
  the dispersion, which is why they are taken;
- replicate floors count the replicate measurements, which are independent by
  construction.

**The default process count for an execution characterization rises from three
to six.** More replicates are the only lever that narrows a floor without
weakening what it claims, and three processes leave two degrees of freedom,
where the bound sits more than four times above the measured dispersion and no
floor resolves anything. Six leave five, where it sits about twice above. The
curve flattens after that: twelve processes would buy another quarter for twice
the model loads.

## Consequences

Floors are wider, and by a factor that depends on how thin the estimate behind
them is — about twice at five degrees of freedom, about a sixth more at
twenty-nine, and negligible for a bootstrap over hundreds of games. Deltas that
previously read as findings on thin evidence now read as within the floor. That
is the intended effect and not a regression.

Some series will not resolve at their current sample counts, and the widened
floor makes that legible rather than causing it. `inference.move_latency_p99_ms`
at thirty decisions is close to a max-of-thirty and already carried a floor of
17.6 ms on a 27 ms value; the bound does not change what that series can answer,
only how obvious it is that it cannot answer it. What sample count makes a p99
resolvable is a benchmark-configuration question, tracked separately.

The recorded gap between a dispersion and its bound is now actionable
information: a wide gap says a floor is wide for lack of replicates, which more
of them fix, and a narrow one says the machine really is that noisy. The
characterization commands print it.

Sizing an evaluation input from a floor errs high, because the floor it
extrapolates from carries a bound sized for the degrees of freedom available
now while a larger pool would carry more of them and a tighter bound. Erring the
other way would size a pool that turns out not to resolve the effect it was cut
for.

The bound says nothing about **non-stationarity**, which is a real and separate
limitation. It describes how well the spread within one characterization is
known; a report compares readings taken later, when the machine's thermal and
contention state has moved. No arithmetic on a characterization's own replicates
can reach that, so a floor is re-characterized when conditions plainly change
rather than treated as a constant of the hardware.

`CHARACTERIZATION_VERSION` and the paired-contribution payload version both
rise. Nothing in the committed results store held a characterization, so no
recorded reading is invalidated.

## Alternatives Considered

**Widen the coverage factor instead.** Raising 1.96 would widen every floor by a
constant, which is the wrong shape: the problem is that thin estimates are
untrustworthy and thick ones are not, and a constant cannot tell them apart. It
would over-penalize a bootstrap over a thousand games to protect a six-replicate
timing characterization.

**Use a t-quantile rather than a chi-squared bound.** Replacing the normal
coverage factor with `t` for the same degrees of freedom is the exact fix for a
different complaint. It makes the floor's coverage correct *on average across
characterizations*, which is a statement about the estimator, and it is
materially cheaper — about 1.3x at five degrees of freedom rather than 2.1x. But
it still leaves any individual floor understating the spread about half the
time, and a floor is used one at a time by a reader who has no way to know which
half they are in. The complaint here is about the individual floor, so the bound
that is conservative per characterization is the right one.

**Store only the bound.** Discarding the measured dispersion would lose the
ability to compare one characterization against another, and would hide whether
a wide floor comes from a noisy machine or a thin estimate. Both are kept.
