# 0030: Benchmark Cost Is Recorded Where The Reading Is

Date: 2026-08-02

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
eighteen seconds a game against a measured 0.469.

These were not decorative. The ladder is excluded from every reduced sweep on
the strength of a four-hour figure that was thirty-five times too large. The
puzzle step is declared unreducible partly on its fifteen-minute figure. The
checkpoint reading was named one of the reduced sweep's two long poles when the
rollout — never named at all — is the longest step.

A scope argument built on a number nothing checks is not a scope argument.

## Decision

Every benchmark that records a reading also records what the invocation cost,
in the committed summary tier, as one `benchmark.wall_clock_seconds`
measurement in its own `benchmark-cost` record.

### The Benchmark Records It, Not The Suite

The timing lives inside each benchmark's entry point rather than in the sweep
driver. A single `anthro eval puzzles` invocation is how most readings are
actually taken, and a cost that only exists inside a sweep would miss them. It
also keeps the suite what it is — a driver that composes benchmarks and
registers no metric of its own.

The consequence is that the measured window is the invocation in process: its
first statement to the moment it builds its own cost record, which excludes
interpreter startup and the store append. Defining the end as the cost record
rather than as "the last measurement" is what keeps seven separately written
benchmarks timing the same thing, since each interleaves its detail-tier
writes and its bootstrap floors differently. A sweep's total is the sum of its
steps plus what the driver pays, which is small and observable in the ledger.

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

So the declared workload is a digest of the benchmark's resolved
configuration, with two normalizations:

- the **model selection and the label chosen for it are removed**, because the
  checkpoint is the coordinate a cost series varies along. Leaving it in would
  start a new line at every checkpoint and make drift unanswerable, which is
  the failure 0018 exists to prevent;
- every **path drops its machine prefix and keeps the artifact it names**,
  because artifact roots differ per machine while the artifact is the same
  work. The final component alone is not enough — several artifacts end in
  `normalized` — so this is the inverse of the rooting the commands apply,
  and a different corpus stays a different series.

Digested rather than carried in full, which was tried first. A report labels a
series by the workload fields that differ between two groups, so the full form
would name the dial that moved — but every benchmark's cost lands in one
family, their configuration schemas share almost no fields, and the rendered
label then ran to dozens of lines of mostly absent settings before naming
anything useful. What a reader loses is which setting changed; the same
envelope's `configuration.source` and `configuration.overrides` name the file
and the overrides that produced the reading, and the configuration itself is
in Git.

### Prose Stops Restating Magnitudes

`configs/evaluation/checkpoint-suite.toml` no longer carries per-step cost
figures. What a step costs is recorded; what the file keeps is the ordering the
reductions are argued from, dated and attributed to the host that measured it.

## Consequences

A cost claim is now reviewable the way a metric delta is, and a benchmark that
gets slower shows up as a committed diff rather than as a comment someone
eventually re-derives by hand.

The committed tier grows by one small record per recording invocation. The
shipped reduced sweep adds six, and the full sweep seven.

**A cost reading is worth much less without an execution floor than the other
efficiency metrics are, and this is the honest limitation.** Three consecutive
reduced sweeps on one CUDA host landed within six percent of each other. A
fourth, taken while two other sessions ran benchmarks on the same machine, cost
the checkpoint reading six times what the other three did. Nothing in the record
says the machine was busy. Decision 0025's execution floor is the mechanism that
would judge such a delta, no floor is characterized for these workloads yet, and
until one is a report will say the noise is unknown — which is correct, and is
also weaker than a reader skimming a number will assume.

Two things this deliberately does not do. It does not record cost for a
benchmark run with `--no-record`, because a cost record is a committed reading
like any other. And it does not fold the cost reading into an execution-noise
characterization, which covers one workload: a characterization run over the
inference benchmark reads that benchmark's own envelope and leaves the cost
record alone.

## References

- `docs/evaluation.md`
- `docs/decisions/0018-workload-scoped-efficiency-series.md`
- `docs/decisions/0025-machine-scoped-execution-noise-floors.md`
- `docs/decisions/0014-evaluation-result-storage.md`
