# 0021: Efficiency Identity Excludes The Conditions Being Compared

Date: 2026-07-31

## Status

Accepted. Refines `0018-workload-scoped-efficiency-series.md`.

## Context

Decision 0018 established that efficiency series are scoped by **workload** and
attributed by **environment**, and described the workload as "the settings that
decide *what* was timed." That phrasing was written against inference
efficiency, where it is exactly right: a latency figure taken at ply forty and
one taken at ply eighty measure different quantities, and subtracting them is a
category error.

Training efficiency was then built by analogy, and the analogy failed. The
settings that decide the work for a training run are the model architecture,
the effective batch, and the corpus — so all three went into the workload
digest. The result was a family whose primary question was structurally
unanswerable:

```text
  metric                                 better    baseline     current       delta  change
  training.active_positions_per_second   higher        1000         600           -  unknown
      (different measurement; the declared workload changed)
```

That is a model widening from 256 to 512 and throughput dropping forty percent
— the exact number the family exists to produce — reported as `unknown`. The
long-run history view broke at the same points, giving a new series at every
model or batch change. Decision 0018 named that failure explicitly and the
implementation walked into it anyway.

The analogy failed for a specific reason worth recording. For inference, "what
decided the work" and "what would be meaningless to subtract" happen to name
the same settings, so the rule and the test agree and the rule looks
sufficient. For training they come apart: the settings that decide the work are
precisely the settings a reader most wants to subtract across. A rule that
tracked its purpose in one family stopped tracking it in the next.

## Decision

**A coordinate that a reader might want to measure the difference across
cannot be part of series identity.** Identity is reserved for changes that make
the difference *meaningless*, never for changes that make it *interesting*.

This is the test 0018 already implied — "a cross-machine delta is not
meaningless" — stated as a rule that can be checked field by field.

### Conditions Are Recorded, Not Digested

Result envelopes carry an execution record whose `workload` is digested into
series identity and whose `coordinates` are not. A benchmark puts a setting in
`coordinates` when changing it moves the number without changing what the
number means.

Training efficiency declares only its benchmark version as workload. The model
architecture, dataset digest, loader configuration, batch size, gradient
accumulation, and determinism setting are all coordinates. Inference efficiency
is unchanged: its ply depth and declared batch size genuinely do change what is
measured, so they stay in the workload.

### Conditions Confound, They Do Not Refuse

A report diffs coordinates the way it already diffs the environment, names
whichever moved, and reports the movement as `confounded`. This follows 0018's
existing posture that "confounded is a verdict, not a refusal": any reader
holding both operands can subtract them, so withholding the arithmetic protects
nobody.

A condition change confounds under either pivot. It is the confounder most
likely to pass unnoticed, because nothing about the machine or the checkpoint
label changes when a corpus is regenerated underneath a comparison.

### The Environment Pivot Pins On Conditions Where They Exist

0018 has the environment pivot pin the model by `parameter_sha256`, so a reused
label cannot sell a model change as a hardware win. Training cannot satisfy
that: the same configuration trained on two machines produces two different sets
of weights, so requiring identical parameters makes the question unaskable
rather than rigorous.

Where a benchmark declares conditions, the environment pivot pins on those
instead, and checks them exactly as strictly. The declared architecture and
corpus are what has to hold still for "did the new machine help" to mean
anything.

## Consequences

Training-efficiency history is one continuous line across model changes, batch
changes, corpus regenerations, and machine changes, annotated wherever any of
them moved. The three questions the family exists to answer — did this change
cost us speed, did an inconsequential change quietly cost us speed, are we
drifting slower over a year — are all expressible.

The cost is that a reader who ignores both the change column and the named
coordinates can misread a batch-size increase as an optimization. That is the
same trade 0018 already accepted for hardware, and the label is present in the
rendered table, in the machine-readable record, and in the movement field
automation reads.

The result envelope version is bumped for the added field.

## References

- `docs/decisions/0018-workload-scoped-efficiency-series.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/design-principles.md`, "Design A Measurement From Its Comparisons"
- `docs/evaluation.md`
