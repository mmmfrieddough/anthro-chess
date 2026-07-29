# 0018: Execution-Sensitive Efficiency Series

Date: 2026-07-28

## Status

Accepted.

## Context

Decision 0013 built series fingerprints from realized inputs: the metric's
definition version and a digest over the content of the games scored. It
explicitly excluded software versions and platform details, on the grounds that
they describe how a measurement was produced rather than what it measured. For
every metric that existed at the time, that was correct, and it is what lets a
refactor or a machine change leave the project's benchmark history intact.

Efficiency metrics break the assumption behind that exclusion. A move-latency
figure is not a property of a checkpoint. It is a property of a checkpoint on a
particular device, at a particular precision, under a particular workload. Run
the same parameters on a faster laptop and the number improves without anything
about the model changing.

Under decision 0013 as written, those two readings would carry identical
fingerprints, land on one line, and render as an optimization. The failure is
silent and it favours the wrong answer: buying a faster machine would look like
engineering progress, and the regression a slower machine masks would be
invisible. `docs/evaluation.md` already asserted that efficiency metrics are
"invalidated by a change of machine rather than a change of model"; nothing
implemented it.

## Decision

A metric declares whether it is **execution-sensitive**. One that is carries an
execution component in its fingerprint alongside the data component.

The execution component covers what decides the number: device type and name,
parameter precision, the Torch version, the platform, the host thread count on
CPU, and a digest of the declared workload. Two readings that differ in any of
these are different series.

This is an extension of decision 0013's principle rather than an exception to
it. Fingerprints cover realized inputs. For a quality metric the realized
inputs are the games scored; for an efficiency metric they are the games
scored, the machine, and the workload. Decision 0013's enumeration was a
correct reading of the only metrics that existed when it was written.

### Only A Timed Measurement May Declare It

Execution sensitivity is available only to a metric whose cost is
`measured_execution`, meaning the measurement *is* the execution. A quality
metric cannot acquire a machine dependency, because doing so would end its
series every time it moved between machines for no reason connected to what it
measures. The registry refuses the combination.

The reverse is refused too: an execution-sensitive metric with no execution
component cannot be fingerprinted at all, rather than falling back to a
machine-blind identity.

### The Workload Is Declared, Not Discovered

A benchmark digests only the settings that decide what was timed — the ply
depth a latency figure is taken at, the batch size a throughput figure is
declared for — and not its whole configuration. Warmup counts, sample counts,
sweep ranges, and output paths stay out, on the same reasoning that keeps the
scored game count out of the data component: measuring more samples estimates
the same quantity more precisely rather than measuring a different one.

### Absent Rather Than Null

A metric that is not execution-sensitive carries no execution key in its
fingerprint payload at all, rather than a null one. This keeps every series
recorded before efficiency metrics existed bit-identical, so introducing this
mechanism ends no existing series.

## Consequences

Efficiency history becomes per-machine. Comparing a checkpoint measured on a
workstation against one measured on a laptop reports incomparable, which is the
honest answer, and the provenance view names the execution difference so the
reader knows why. Getting one continuous efficiency line means measuring on one
machine, which was always the only way to get a meaningful one.

Reproducing an efficiency series requires reproducing its machine. That is a
real cost and it is the point: the alternative is a series that silently
averages over hardware.

A bridge can still rejoin two efficiency fingerprints, and the bar from decision
0013 applies unchanged. A Torch upgrade that provably did not change execution
speed is a legitimate bridge; a new machine is not.

Result envelopes gain an optional execution record, so an efficiency result
carries enough to recompute its own series identity. The envelope version is
bumped accordingly.

## References

- `docs/evaluation.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/decisions/0014-evaluation-result-storage.md`
