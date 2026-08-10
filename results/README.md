# Benchmark Results

This directory is the committed **summary tier** of the benchmark results
store. Benchmarks append here; reports and comparisons are views over it.

- `records/` holds one JSON file per benchmark result: headline metric values,
  their series fingerprints, the dispersion each value's own units moved it by,
  and the provenance needed to recompute those fingerprints.
- `bridges/` holds explicit assertions that two fingerprints name the same
  series. A bridge is legitimate only when the fingerprint moved for a reason
  provably independent of the measured quantity, and revoking one is a
  reviewable diff.

`anthro eval run` is what appends here: it scores one checkpoint over a
deterministic view of the frozen pool, records the held-out prediction,
legality, and rating-dependency results for it, and bootstraps each
measurement's own sampling dispersion from the same pass. A delta between
two such readings is floored by combining the two dispersions in front of it.
Every recording benchmark also appends one `benchmark-cost` record saying what
the invocation
cost, so a claim about what a sweep can afford is a reviewable diff rather than
a comment; `docs/decisions/0031-committed-benchmark-cost.md` explains why that
belongs in this tier despite being a property of the machine.
Other benchmark commands append through the same boundary. In particular,
`anthro eval puzzles` records rating-response headlines over the owned external
puzzle set while leaving its continuous response curves, band drill-downs, and
resampled response resolution in the detail tier.

Read the numbers with ordinary file tools, or with `anthro eval report` for a
compact delta view, which annotates every delta with the floor it did or did
not clear. `anthro eval metrics` lists every metric identifier and its declared
direction of improvement. `anthro eval noise plan` reports how many games an
axis needs to resolve an effect of a given size, read off the newest reading
that measured its own spread.
`anthro eval tensorboard OUTPUT` regenerates a disposable checkpoint-history
view outside this directory, with one TensorBoard line per raw series
fingerprint.

Bulk diagnostics — per-position tables, slice breakdowns, generated games — do
not belong here. They stay in the machine-local detail tier and are referenced
from a record by path and digest.

## Pre-Core Records

The records currently here come from the two `anthro eval run` readings taken by
the first full suite reading, on steps 100 and 8000 of `training-blitz-30k-v4`,
at a reduced 400-game view rather than the canonical pool view. Each reading
appends one record per family it covers — held-out prediction, rating
dependency, and adjudicated decisions — each measurement carrying the spread
bootstrapped for it, so two readings produce six records. They exist to
demonstrate that metric movement shows up as a reviewable Git diff, which had
not been exercised with real data.

They are **pre-core**: nothing is protected before the evaluation core is
designated, so they are deleted at #90 rather than bridged or carried forward.
`#146` records what they were for.

`docs/evaluation.md` and `docs/decisions/0014-evaluation-result-storage.md`
explain the layering; `docs/decisions/0013-benchmark-result-comparability.md`
owns the fingerprint and bridge contract.
