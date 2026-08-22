# Benchmark Results

This directory is the committed **summary tier** of the benchmark results
store, and reports and comparisons are views over it.

Nothing writes here by running. A benchmark appends to a machine-local store,
and a record arrives here through `anthro eval promote --checkpoint <label>`,
whose copy is committed in the pull request adopting the change that produced
the reading. So this directory is the line of accepted checkpoints rather than a
log of everything measured. `docs/issue-workflow.md` says when a session does
that.

- `records/` holds one JSON file per benchmark result: headline metric values,
  their series fingerprints, the dispersion each value's own units moved it by,
  and the provenance needed to recompute those fingerprints.
- `bridges/` holds explicit assertions that two fingerprints name the same
  series. A bridge is legitimate only when the fingerprint moved for a reason
  provably independent of the measured quantity, and revoking one is a
  reviewable diff. A bridge about this history is recorded here by naming the
  store — `anthro eval bridge add --store results` — since the commands default
  to the machine's own store like everything else.

`anthro eval run` is what produces the records here: it scores one checkpoint
over a deterministic view of the frozen pool, records the held-out prediction,
legality, and rating-dependency results for it, and bootstraps each
measurement's own sampling dispersion from the same pass. A delta between
two such readings is floored by combining the two dispersions in front of it.
Every recording benchmark also writes one `benchmark-cost` record saying what
the invocation
cost, so a claim about what a sweep can afford is a reviewable diff rather than
a comment; `docs/decisions/0031-committed-benchmark-cost.md` explains why that
belongs in this tier despite being a property of the machine. Promotion takes a
checkpoint's records as a set, so a promoted reading brings its cost with it.
Other benchmark commands record through the same boundary. In particular,
`anthro eval puzzles` records rating-response headlines over the owned external
puzzle set while leaving its continuous response curves, band drill-downs, and
resampled response resolution in the detail tier.

Read the numbers with ordinary file tools, or with
`anthro eval report --store results` for a compact delta view, which annotates
every delta with the floor it did or did not clear. Every reading command takes
that flag, and needs it to read this directory rather than the machine's own
store. `anthro eval metrics` lists every metric identifier and its declared
direction of improvement. `anthro eval noise plan` reports how many games an
axis needs to resolve an effect of a given size, read off the newest reading
that measured its own spread.
`anthro eval tensorboard OUTPUT` regenerates a disposable checkpoint-history
view outside this directory, with one TensorBoard line per raw series
fingerprint.

Bulk diagnostics — per-position tables, slice breakdowns, generated games — do
not belong here. They stay in the machine-local detail tier and are referenced
from a record by path and digest.

This directory holds no records yet. The pre-core ones it used to describe were
deleted with the corpus generation they were read against, and nothing has been
promoted since.

`docs/evaluation.md` and `docs/decisions/0014-evaluation-result-storage.md`
explain the layering; `docs/decisions/0013-benchmark-result-comparability.md`
owns the fingerprint and bridge contract.
