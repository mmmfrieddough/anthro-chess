# 0025: Execution Noise Floors Are Scoped To The Machine Their Series Is Not

Date: 2026-07-31

## Status

Accepted. Refines `0018-workload-scoped-efficiency-series.md`. Refined by
`0026-conservative-dispersion-bounds.md`.

## Context

Decision 0018 kept the machine out of efficiency series identity on purpose. A
cross-machine latency delta is interpretable rather than meaningless, and
putting the environment into the fingerprint would leave a dozen three-point
lines after a year, making long-run drift unanswerable. The environment is
recorded as coordinates and a report attributes a delta to it.

That is the right rule for comparability and the wrong shape for noise. Every
floor the project stores is keyed by the series fingerprint of the measurement
it qualifies, which is what makes a floor invalidate when the pool moves instead
of lingering as a stale constant. For an efficiency metric the fingerprint holds
the workload and nothing about the machine, so a floor characterized on a laptop
would be looked up and applied to a workstation's reading, and would describe
nothing about it. The two machines differ by a factor of three; their noise
differs too, and not by the same factor.

The noise in these metrics also cannot be estimated the way the others are. Data
sampling noise is bootstrapped from numbers a run already computed, and
evaluation and training noise are read from replicate results the store already
holds. What varies in a timing measurement is the machine, and the store holds
no replicates of that: a checkpoint is measured once per invocation, and two
invocations of the same workload on the same machine are exactly the replicate
that was never recorded.

Observed on a real MPS run of one checkpoint measured twice: p50 latency moved
+0.008 ms and throughput moved -2.5 decisions/s, both rendered as regressions
because the report had no floor to judge them against. Neither is a finding.

## Decision

An execution noise floor is characterized by repeated measurement, and it is
valid only on the machine that produced it.

### The Floor Is Measured, Not Derived

A fourth noise kind, `execution`, is characterized by running the benchmark
again rather than by resampling a result. The replicates are spread across
**several processes**, because a reading a report compares is one invocation's
and carries the model load, allocator growth, and lazy kernel compilation that
invocation paid for. Repeats **within** a process are taken as well and reported
beside the floor as a separate dispersion: they say how much of the spread the
cheap form of replication reproduces, which differs by device and is not
knowable in advance.

Cold-start metrics keep one reading per process. A reload inside a warm process
reads a cached file and an imported Torch, so including it would report a
dispersion several times narrower than two real cold starts, and a floor that
understates the noise is worse than no floor at all.

Nothing measured during a characterization is appended to the results store. The
readings are evidence about the machine rather than about the model.

### The Scope Lives On The Characterization

The characterization carries the execution record it was measured under. A
report resolves an execution floor only when that record's environment matches
**both** sides of the delta being judged; a delta spanning two machines is
covered by no characterized floor and its noise is reported as unknown.

Scoping the floor rather than the series is what keeps both properties: the
efficiency line stays continuous across a hardware change, and no machine
borrows another's noise. The environment fields the match keys on are the same
coarse ones decision 0018 attributes a delta to, so an operating system point
release does not invalidate a floor while a device or precision change does.

## Consequences

An efficiency delta can now be judged rather than only reported, and the failure
mode that motivated this — sub-percent jitter written up as a regression — is
visible in the noise column instead of in a reader's conclusion.

The cost is that a floor is per machine and does not travel. A new machine
reports unknown noise until it characterizes its own, which is honest and is
also the cheapest of the four kinds to obtain: a handful of benchmark runs
rather than a training run.

A floor also goes stale in a way a pool-scoped floor does not, since a machine's
noise changes with what else is running on it. Nothing here detects that; the
characterization records when and where it was taken, and re-characterizing is
cheap.

The floor is a point estimate of the dispersion, and at the default replicate
count that estimate is itself noisy: with six replicates its relative standard
error is about a third, and the shakedown reading saw one checkpoint's p50
dispersion characterized between 0.14 ms and 0.72 ms across sessions minutes
apart. Some same-weights deltas therefore still clear their floor. The floor
qualifying a delta needs to be a conservative bound rather than the middle of
its own sampling distribution, which is a change to the arithmetic every kind
shares and is deliberately left to its own decision.

## References

- `docs/evaluation.md`
- `docs/decisions/0018-workload-scoped-efficiency-series.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
