# 0018: Workload-Scoped, Environment-Attributed Efficiency Series

Date: 2026-07-29

## Status

Accepted. Refined by `0021-efficiency-identity-excludes-compared-conditions.md`
and `0025-machine-scoped-execution-noise-floors.md`, and extended by
`0031-committed-benchmark-cost.md`.

## Context

Decision 0013 built series fingerprints from realized inputs and explicitly
excluded software versions and platform details, on the grounds that they
describe how a measurement was produced rather than what it measured. That is
correct for every quality metric, and it is what lets a refactor or a machine
change leave the project's benchmark history intact.

Efficiency metrics raise a question 0013 never had to answer, because a latency
figure genuinely does depend on the machine. Two readings of the same
checkpoint on a laptop and a desktop differ by a factor of three, and nothing
about the model changed.

The obvious response is to put the machine into series identity, so the two
readings land on different lines and a faster desktop cannot render as an
optimization. That is wrong, for two reasons that only became clear once the
workflow was written down.

**A cross-machine delta is not meaningless.** This is the distinction that
matters. Comparing move loss across two different pools produces a number with
no interpretation; "incomparable" is simply true. Comparing latency across two
machines produces a perfectly interpretable number that is merely *attributable
to the environment rather than to the model*. Those are different failure modes,
and only the first is what fingerprints were built for.

**Fragmenting the series destroys the question the project actually cares
about.** Playability is a property of the whole shipped stack. "Have we been
creeping toward higher latency?" and "the model got bigger but a Torch upgrade
paid for it — what is the net?" are answerable only from a continuous line. Put
the environment in identity and every hardware change, Torch bump, and cloud
agent session starts a new series, leaving a dozen three-point lines after a
year. Decision 0013's own argument applies: a rule that severe "either gets
bypassed in practice or discards history that was never actually compromised."

Three legitimate comparisons exist, and a single binary same-series-or-not
mechanism can serve at most one:

1. vary the checkpoint, hold the environment fixed — did the model change cost
   us speed;
2. vary the environment, hold the model fixed — did the upgrade help;
3. vary both — what is the net effect on the thing we ship.

## Decision

Efficiency series are scoped by **workload** and attributed by **environment**.

### The Workload Is Identity

A metric declares itself execution-sensitive, and one that does carries a
workload component in its fingerprint: a digest of the settings that decide
*what* was timed, such as the ply depth a latency figure was taken at and the
batch size a throughput figure was declared for.

Change the workload and the number measures a different quantity, exactly as a
different pool does. That is the meaningless-delta case, and it is refused
outright: no delta is computed and the row is reported as incomparable.

Sample counts, warmup counts, and sweep ranges stay out of the digest, on the
same reasoning that keeps the scored game count out of the data component:
measuring more estimates the same quantity more precisely rather than measuring
a different one.

### The Environment Is Coordinates

Device, device name, precision, Torch version, coarse platform key, and CPU
thread count are recorded on every efficiency result and are **not** in the
fingerprint. They are coordinates a report attributes a delta to.

The platform key is deliberately coarse — system and machine architecture, such
as `Darwin-arm64` — while the full platform string is kept as provenance.
Keying attribution on the full string would flag every delta after an operating
system point release that changed no hardware.

### Views Declare What They Hold Fixed

Whether two numbers are safe to subtract is a question for a report, not for
storage. Putting a view-level policy into storage-level identity was the
original mistake.

Each view declares its pivot. The checkpoint pivot asks whether the model
improved, so any environment movement makes that unanswerable. The environment
pivot asks whether the environment is faster with the model pinned, so there a
model change is what would confound it. The history view varies both and
annotates the points where the environment moved.

### Confounded Is A Verdict, Not A Refusal

When the delta is real but something other than the pivot moved, the report
shows the value and reports the movement as `confounded`, together with an
attribution naming which of model, environment, and workload changed.

The judgment field carries the honesty rather than a withheld delta. Any reader
holding both operands can subtract them, so suppressing the arithmetic protects
nobody while making the record less useful to a careful reader. Automation keys
on movement, and `confounded` is not `better`.

The environment pivot pins the model by `parameter_sha256` rather than by
checkpoint label, because a reused label would quietly turn a model change into
an apparent hardware win.

## Consequences

Efficiency history is continuous across machines, so long-run drift and net
effect are answerable. The cost is that a reader who ignores both the change
column and the attribution can misread a hardware win as progress. That is
accepted: the label is present in the rendered table, in the machine-readable
record, and in the movement field automation reads.

An efficiency comparison now has a pivot, which means a report has to know
which question it is answering. That is a real addition to the reporting
surface, and it is the thing that makes all three questions expressible instead
of one.

Bridges keep their meaning from decision 0013 and are now rarely needed for
efficiency, since the common reasons a series used to break no longer break it.
A workload change remains a genuine break and is not bridgeable, because what
was measured changed.

Result envelopes carry an optional execution record holding both halves, so an
efficiency result can recompute its own fingerprint and supply its own
attribution. The envelope version is bumped accordingly.

## References

- `docs/evaluation.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/decisions/0014-evaluation-result-storage.md`
