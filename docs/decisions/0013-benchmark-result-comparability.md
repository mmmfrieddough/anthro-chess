# 0013: Per-Series Benchmark Result Comparability

Date: 2026-07-25

## Status

Accepted, except for "Pool Generations, Core, And Current", which
`0068-a-pool-re-cut-breaks-benchmark-history-and-that-is-accepted.md`
supersedes: there is no core, a re-cut re-baselines, and containment still
binds.

## Context

Decision 0012 established how benchmark *inputs* are layered. It did not say how
benchmark *results* stay comparable to each other over the life of the project.

The evaluation goal is not only "did this change help" but "is the model better
than it was a year ago, and which parts improved." That requires a durable
result history, which in turn requires knowing which measurements sit on the
same line and which do not.

Decision 0012 resolved this at the granularity of a pool version: comparisons
are valid within a version, and a report should refuse to compare across
versions. That rule is correct and insufficient. Taken literally with a single
global version, any change anywhere invalidates the entire project's benchmark
history, including for benchmarks the change never touched. A rule that severe
either gets bypassed in practice or discards history that was never actually
compromised.

The main expected source of change is corpus growth. Preparation assigns splits
as a pure function of the split seed and a content-derived game id, so growing
the corpus never moves an existing game between splits. That property is what
makes a better answer possible: appending does not destroy the games an earlier
measurement was computed over, so the earlier measurement remains reproducible
on the subset. Removing games, changing filters so previously accepted games are
rejected, or changing the split seed do destroy it, and no future checkpoint can
ever be scored on those games again.

## Decision

Comparability is a property of a **series**, not of the project.

A series is one metric measured one way over one set of inputs. Two results
belong to the same series, and may be compared or plotted on the same line, when
their fingerprints match.

### Fingerprints Cover Realized Inputs

A fingerprint is built from what a measurement actually consumed and how it was
computed: the metric's definition version, and a digest over the content of the
games scored. It is not built from configuration text, software versions, file
layout, or command shape.

This distinction is the point. Refactors, CLI changes, configuration
restructuring, and pipeline changes that do not alter the scored content leave
the fingerprint untouched, so the series continues with no judgment call and no
manual intervention. Only a change to what was measured, or to which games it
was measured over, breaks a series.

The content digest covers the projection a benchmark consumes rather than the
whole normalized row, so adding a schema field that no existing metric reads
does not break unrelated series. Timing fields and per-ply opening metadata are
both known to be coming; without this, each would invalidate every series in the
project on arrival.

Metrics with no data dependency, such as gradient norm or weight statistics,
carry a null data component. They are structurally immune to pool changes, so at
least one metric family keeps an unbroken line across the project's lifetime.

Decision 0018 extends this to efficiency metrics, whose declared workload is a
realized input and is therefore part of series identity. It also draws the
limit: the machine an efficiency number was measured on stays *out* of
identity, because a cross-machine delta is interpretable rather than
meaningless, and fragmenting that history would cost more than it protects.

### Preview Views Subsample, Never Filter

A cheap in-training reading and an expensive canonical reading are separate
series, because their fingerprints differ. They remain interpretable together
only because a uniform hash-rank subsample of a population is an unbiased
estimator of the same quantity with wider error bars.

That property holds only for subsampling. A view that filters, for example to
short games so it runs faster, measures a different quantity and is not an
estimate of anything the canonical reading reports. Preview views may therefore
subsample and may not filter.

### Breaking Is Automatic, Rejoining Is Explicit

Fingerprint mismatch breaks a series automatically. A maintainer or agent may
record a **bridge** asserting that two fingerprints are equivalent. A bridge
records both fingerprints, the reason, and who asserted it; it is stored
alongside the results, reviewable, and reversible. Reports and charts render a
bridged seam distinctly from an unbroken line.

A bridge is legitimate only when the fingerprint moved for a reason provably
independent of the measured quantity, such as a storage format change or a
recorded configuration string that does not affect computation. It is never
legitimate when what was measured changed, or when which games were scored
changed. Automatic detection with an explicit, auditable override is what keeps
a convenience from becoming a way to launder an invalid comparison.

### Pool Generations, Core, And Current

Pool versions are **generations**, and each generation must be a superset of the
previous one. Because split assignment is stable under corpus growth, this holds
automatically for appending; it does not hold if filters change so that
previously accepted games are rejected, or if the split seed changes. Filters are
therefore part of the pool fingerprint, and a generation cut verifies containment
rather than assuming it.

Once a generation is designated as the **core**, every benchmark reports two
views: core, the intersection with that designated generation, giving one
continuous line for the rest of the project; and current, the full pool, giving
more statistical power with a line that restarts at each generation.

Current is the number to quote for how good a checkpoint is. Core is the number
to quote for whether it is better than it was a year ago. Divergence between them
for the same checkpoint is itself the signal that the core has been overfit.

The core is not designated from the first generation. It is designated once the
corpus first spans the axes the project intends to keep measuring, because a core
frozen against a narrower corpus can never measure an axis it contains no games
for. Its per-axis statistical power is fixed permanently at designation, so each
axis needs enough games at that moment, sized from measured sampling noise rather
than guessed.

### Before The Core Is Designated, Nothing Is Protected

Everything above describes how history is kept once there is history to keep.
There is none yet, and there will not be until the first generation is cut and
the core designated, which is stage 4 work in `docs/planning/roadmap.md`.

Until then the results store holds no reading any decision rests on, and the
checkpoints that exist are proof-scale. A change that breaks every series in
the project therefore costs nothing. Bumping the preprocessing version,
changing the action vocabulary, redefining a metric, and regenerating the
corpus are all ordinary changes during this period. **No work should be
deferred, resequenced, or bundled in order to avoid breaking a series, and no
issue should carry that as a reason.** The one argument that survives is
compute: an expensive corpus regeneration is worth batching so it is paid once
rather than three times, which is a cost claim about wall clock and not a
comparability claim.

This does not relax the machinery. Fingerprints are still computed, breaks
still register automatically, and reports and charts still render a seam as a
seam. The harness has to be correct before the first reading anyone trusts it
for, and the only way to arrive there is to build it as though history already
mattered. What is suspended is the caution, not the mechanism.

### Anchor Checkpoints

A small set of checkpoints is retained as anchors and re-scored whenever a
generation is cut. Their re-scored results are what give a new generation an
overlap with the previous one, so a shift at the seam is attributable to the pool
rather than mistaken for a model regression.

Retention is a policy, not a storage mechanism. Anchors stay wherever runs
already live.

## Consequences

Most changes stop invalidating history. A refactor, a new CLI flag, an unrelated
schema field, or a benchmark that changes only for its own metric no longer
touches series it did not affect.

Result artifacts must carry enough to compute a fingerprint, which means the
digest of scored content and the metric definition version travel with every
result rather than being reconstructable only from the environment that produced
it.

Changing a metric's definition ends its series and starts a new one. That is
deliberate: silently redefining a metric under a stable name is the failure this
decision exists to prevent. The cost is accumulated dead series, which is
acceptable because they remain readable and honestly labeled.

Corpus expansion becomes a constrained operation. Relaxing a filter without
raising the accepted-game bound can drop games that an earlier generation
contains, which breaks the superset property and the core with it. Expansion
configuration must preserve containment and the generation cut must verify it.

Comparing many checkpoints against a fixed core over years is real selection
pressure, more than decision 0012 anticipated when it accepted mild pressure on
the pool. The growing current view is the check: sustained divergence between
core and current is the observable symptom, available at no extra cost.

## References

- `docs/evaluation.md`
- `docs/data.md`
- `docs/decisions/0011-held-out-test-partition.md`
- `docs/decisions/0012-derived-evaluation-views.md`
- `docs/decisions/0014-evaluation-result-storage.md`
- `docs/decisions/0018-workload-scoped-efficiency-series.md`
- `docs/decisions/0068-a-pool-re-cut-breaks-benchmark-history-and-that-is-accepted.md`
