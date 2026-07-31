# Benchmark Results

This directory is the committed **summary tier** of the benchmark results
store. Benchmarks append here; reports and comparisons are views over it.

- `records/` holds one JSON file per benchmark result: headline metric values,
  their series fingerprints, and the provenance needed to recompute those
  fingerprints.
- `bridges/` holds explicit assertions that two fingerprints name the same
  series. A bridge is legitimate only when the fingerprint moved for a reason
  provably independent of the measured quantity, and revoking one is a
  reviewable diff.
- `floors/` holds characterized noise floors: how far apart two measurements of
  an unchanged quantity land. A floor is keyed by the same fingerprint as the
  measurements it qualifies, so it stops applying when the series moves rather
  than lingering as a stale constant.

`anthro eval run` is what appends here: it scores one checkpoint over a
deterministic view of the frozen pool, records the held-out prediction,
legality, and rating-dependency results for it, and bootstraps the
data-sampling floors for that reading from the same pass.
Other benchmark commands append through the same boundary. In particular,
`anthro eval puzzles` records rating-response headlines over the owned external
puzzle set while leaving its continuous response curves, band drill-downs, and
source-game-aligned comparison contributions in the detail tier. Because
comparable puzzle checkpoints always score the same frozen set, `anthro eval
report` uses those contributions to bootstrap the paired delta instead of
applying an independent-input floor. Pass `--detail-root`, or configure the
ordinary detail-root environment, when reading those floors; without retained
details the paired floor is reported as unknown.

Read the numbers with ordinary file tools, or with `anthro eval report` for a
compact delta view, which annotates every delta with the floor it did or did
not clear. `anthro eval metrics` lists every metric identifier and its declared
direction of improvement. `anthro eval noise` characterizes evaluation and
training floors from recorded replicates, lists what is characterized, and
reports how many games an axis needs to resolve an effect of a given size.
`anthro eval tensorboard OUTPUT` regenerates a disposable checkpoint-history
view outside this directory, with one TensorBoard line per raw series
fingerprint.

Bulk diagnostics — per-position tables, slice breakdowns, generated games — do
not belong here. They stay in the machine-local detail tier and are referenced
from a record by path and digest.

`docs/evaluation.md` and `docs/decisions/0014-evaluation-result-storage.md`
explain the layering; `docs/decisions/0013-benchmark-result-comparability.md`
owns the fingerprint and bridge contract.
