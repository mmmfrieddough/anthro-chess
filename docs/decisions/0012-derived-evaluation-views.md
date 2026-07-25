# 0012: Derived Evaluation Views Over One Frozen Pool

Date: 2026-07-25

## Status

Accepted.

## Context

The evaluation plan calls for many benchmarks with different data needs. Offline
prediction wants positions sliced by rating, phase, color, and legal-move count.
Rollout benchmarks want frozen human prefixes plus reference distributions for
game length, results, repetition, and openings. Performance benchmarks want one
fixed input measured under declared conditions. Later work adds dependency
tests, rating-response diagnostics, and human-likeness comparisons.

Building a tailored dataset per benchmark would multiply provenance, checksums,
versioning, and leakage surface by the number of benchmarks, and would make two
benchmarks silently incomparable when their inputs drifted apart.

There is also a tension between two properties the evaluation inputs need.
Tracking the corpus means the evaluation data should reflect composition
changes as the corpus widens. Staying frozen means results must remain
comparable across checkpoints over time. A single artifact cannot do both.

## Decision

Layer benchmark data as partition, pool, and views.

The **partition** decides what a game may be used for. It is owned by data
preparation and covered by decision 0011.

The **pool** is the `test` partition materialized as one versioned, checksummed
artifact with its own manifest and coverage statistics. It carries no
per-benchmark tailoring. It is a regenerable pipeline output rather than
committed data; what is checked in is the selection configuration plus, once a
pool exists, its expected identity digest.

**Views** are per-benchmark deterministic selections over the pool: filtering by
ply count, clock presence, or rating presence; projecting to prefixes;
subsampling by hash rank. Each benchmark records its resolved view spec,
including the digest of the selected game ids, in its own artifact.

Views are derivations, never new stored data. A benchmark that needs something
this layer cannot derive is a signal that the underlying field belongs in the
normalized schema.

Resolve the frozen-versus-representative tension by assigning the two properties
to different things. Representativeness is a property of the recipe: uniform,
unstratified hash-rank assignment, so pool composition tracks corpus composition
automatically. Frozen is a property of a benchmark version. When corpus
composition changes materially, regenerate, cut a new pool version, and
re-baseline; comparisons stay valid within a version.

## Consequences

Adding a benchmark costs a view spec rather than a dataset, and two benchmarks
over the same pool version are comparable by construction.

Runtime is controlled at the view layer rather than by shrinking the pool. A
benchmark that must run quickly subsamples in its own view, so the pool stays a
faithful sample of the corpus and evaluation cost does not grow with it.

Derived labels can be refined without touching stored artifacts. Opening
classification is the clearest case: computing our own labels in the view layer
means a book or granularity change never regenerates the corpus, which is why
capturing source ECO and opening headers into the schema was rejected.

Slice definitions become a shared contract. Phase boundaries, legal-move-count
buckets, and rating bands live in one place, so a change moves every benchmark
together instead of leaving each with its own drifted definition.

The cost is indirection: reading a benchmark result requires reading its view
spec to know what was measured. Recording the spec and the selected game-id
digest in every artifact keeps that answerable rather than implicit.

## References

- `docs/evaluation.md`
- `docs/data.md`
- `docs/decisions/0004-source-agnostic-normalized-data.md`
- `docs/decisions/0011-held-out-test-partition.md`
