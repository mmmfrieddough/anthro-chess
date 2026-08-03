# 0031: Benchmark Cost Is Recorded Where The Reading Is

Date: 2026-08-03

## Status

Accepted. Extends `0018-workload-scoped-efficiency-series.md`.

## Context

Nothing in the harness recorded what a benchmark cost. The suite resolved a
per-step duration, printed it, wrote it to a machine-local ledger for resume,
and discarded it. `anthro eval budget` joined *training* seconds to quality
rather than suite cost. A benchmark run on its own produced no duration at all.

So every cost figure the project acted on lived in a comment, and no diff, no
test, and no report could ever contradict one. All of them drifted. The
generated-game fix took a game from fifteen-odd seconds to well under one, and
the comments were not updated: `configs/evaluation/checkpoint-suite.toml`
priced the reduced sweep at about half an hour against a measured few minutes,
the puzzle step at about fifteen minutes, and the rating ladder at roughly
eighteen seconds a game against a measured 0.469. Those figures decided scope:
the ladder is excluded from every reduced sweep on the strength of a four-hour
number that was thirty-five times too large.

A scope argument built on a number nothing checks is not a scope argument. The
stale figures were deleted rather than corrected, because a corrected figure
drifts the same way; this is the mechanism that replaces them.

## Decision

Every benchmark that records a reading also records what the invocation cost,
in the committed summary tier, as one `benchmark.wall_clock_seconds`
measurement in its own `benchmark-cost` record.

### The Driver Records It, Not The Suite And Not The Benchmark

The timing lives in `run_benchmark`, the one path both the sweep and every
`anthro eval` subcommand invoke a benchmark through. Not in the sweep, because
a single `anthro eval puzzles` invocation is how most readings are actually
taken and a cost that only existed inside a sweep would miss them. Not in the
benchmarks, because seven copies of a clock is the duplication the driver was
extracted to end — an earlier attempt at this metric threaded a
`time.perf_counter()` and a device through all seven entry points, and was
abandoned for that reason.

The measured window is the invocation in process: the driver's first statement
to the moment the recording has assembled everything it will commit. Configuration
loading, interpreter startup, and the store append are outside it. Defining the
end there rather than at "the last measurement" is what keeps seven separately
written benchmarks timing the same thing, since each interleaves its detail-tier
writes and its bootstrap floors differently. A sweep's total is the sum of its
steps plus what the driver pays, which is small and observable in the ledger.

One consequence of the driver holding the clock rather than the benchmark: the
device a cost is attributed to is resolved from the declared model selection,
by the same public function the runner itself uses, rather than read off a
loaded runner. The driver deliberately does not load one — the inference
benchmark measures its own model load and cold start, so a pre-loaded runner
would change what it reports. A runner handed to the driver by a caller that
already loaded one wins over the declared selection, since nothing in a
configuration says where such a caller put it.

### The Committed Tier, Not The Detail Tier

Cost is machine-specific, which is an argument for the detail tier, and it is
load-bearing for scope decisions, which is an argument for the committed one.
The second wins: the failure this record exists to prevent is a stale claim
nobody contradicts, and only a committed record is contradicted by a diff. The
detail tier is machine-local, so a figure kept there would be exactly as
invisible to review as the comment it replaces.

Machine specificity is not a new problem here. Decision 0018 already keeps the
environment out of series identity and records it as coordinates a report
attributes a delta to, and decision 0025 already scopes an execution noise floor
to the machine that characterized it. A cost series inherits both.

### The Workload Is The Whole Configuration

This is where a cost metric departs from every other workload-scoped one.

Decision 0018 keeps sample counts, warmup counts, and sweep ranges out of the
workload digest, because measuring more estimates the same quantity more
precisely rather than measuring a different one. For cost the reasoning
inverts: measuring more costs more, and the cost *is* the quantity. A reduced
sweep's checkpoint reading and a full one's are two different amounts of work,
and putting them on one line would report a reduction as an improvement.

So the declared workload is a digest of the benchmark's resolved configuration,
with two normalizations:

- the **model selection and the label chosen for it are removed**, because the
  checkpoint is the coordinate a cost series varies along. Leaving it in would
  start a new line at every checkpoint and make drift unanswerable, which is
  the failure 0018 exists to prevent;
- every **path drops its machine prefix and keeps the artifact it names**,
  because artifact roots differ per machine while the artifact is the same
  work. The final component alone is not enough — several artifacts end in
  `normalized` — so this is the inverse of the rooting the commands apply, and
  a different corpus stays a different series.

The whole configuration is a blunter rule than "how much work was done", and
this is where it is blunt: a batch size or a concurrency setting changes how
fast the same work is done rather than how much of it there is, and both are in
the digest. A machine that retunes one to fit its accelerator starts its own
cost line rather than joining the shipped one. That is accepted because the
alternative is a per-schema list of which dials are sample counts, which is the
kind of second list that drifts silently against the first.

It is blunt in the other direction too, and only once. A pool contributes to
the digest as the artifact it names, not as the realized dataset identity —
`pool_id`, `pool_version`, `game_ids_sha256` — that every reading carries
beside it. So re-freezing a pool at the same path with a different or larger
game set keeps both readings on one cost line, and the shipped checkpoint
selection reads the pool whole, so the second one costs more for a reason the
series cannot see. Accepted for the same reason as the first: putting realized
data identity in would start a fresh cost line at every re-freeze, including
the ones that changed nothing about how much work there is. A frozen pool is
checksummed and immutable by construction, so re-freezing over one is a
deliberate act; the readings themselves record which pool version they scored,
which is where a reader confirms that two cost figures are comparable.

Digested rather than carried in full, which was tried first. A report labels a
series by the workload fields that differ between two groups, so the full form
would name the dial that moved — but every benchmark's cost lands in one
family, their configuration schemas share almost no fields, and the rendered
label then ran to dozens of lines of mostly absent settings before naming
anything useful. What a reader loses is which setting changed; the same
envelope's `configuration.source` and `configuration.overrides` name the file
and the overrides that produced the reading, and the configuration itself is in
Git.

## Consequences

A cost claim is now reviewable the way a metric delta is, and a benchmark that
gets slower shows up as a committed diff rather than as a comment someone
eventually re-derives by hand.

The committed tier grows by one small record per recording invocation. The
shipped reduced sweep adds six, and the full sweep seven.

**A cost reading is worth much less without an execution floor than the other
efficiency metrics are, and this is the honest limitation.** On an idle CUDA
host three reduced sweeps landed at 308.5 s, 306.9 s and 307.8 s, which is
within half a percent and is what the metric looks like at its best. The
investigation that opened this issue saw the same reading move by six times on
a host where two other sessions were running benchmarks. Nothing in the record
distinguishes the two situations, because nothing in it says the machine was
busy. Decision 0025's execution floor is the mechanism that would judge such a delta,
no floor is characterized for these workloads yet, and until one is a report
will say the noise is unknown — which is correct, and is also weaker than a
reader skimming a number will assume. Read it as a trip-wire for
order-of-magnitude drift rather than as an instrument for small deltas.

Three things this deliberately does not do. It commits no cost for a benchmark
run with `--no-record`, because a cost record is a committed reading like any
other: the envelope is assembled and returned exactly as every other reading's
is, and simply never appended. It records nothing for an invocation that
measured nothing — one that failed, or one whose every unit came back empty —
since those seconds are the cost of the failure rather than of a reading. And
it does not fold the cost reading into an execution-noise characterization,
which covers one workload: a characterization run over the inference benchmark
reads that benchmark's own envelope and leaves the cost record alone.

## References

- `docs/evaluation.md`
- `docs/decisions/0018-workload-scoped-efficiency-series.md`
- `docs/decisions/0025-machine-scoped-execution-noise-floors.md`
- `docs/decisions/0014-evaluation-result-storage.md`
