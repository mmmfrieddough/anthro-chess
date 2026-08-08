# 0014: Evaluation Result Storage And Tooling

Date: 2026-07-25

## Status

Accepted. Refined by `0023-series-separated-tensorboard-history.md`.

## Context

Decision 0013 defines when benchmark results are comparable. It does not say
where they live. Runs currently write a run record, a metrics stream, and
checkpoints to a machine-local run root, datasets go to a machine-local data
root, nothing but configuration is committed, and publishing to an external
registry is deferred.

That leaves the durable result history with no home, and it leaves the retention
of checkpoints that later comparisons depend on to chance.

The surrounding constraints matter more than the general MLOps question. The
project has one maintainer on one machine, with a move to a separate GPU
workstation expected during the scaling milestone and little switching back and
forth. Agents do the bulk of the work, so how easily an agent can read a result
is a first-order concern rather than an afterthought.

The three things that could be stored have almost nothing in common, and the
usual framing of picking one platform obscures that.

## Decision

### Results Are Committed, In Two Tiers

The **summary tier** of benchmark results is committed to the repository. Family
and headline metrics with their fingerprints are kilobytes per checkpoint, so
even a long project history stays small.

Committing buys exactly the properties this project needs. History and
durability come from Git rather than from a service. Metric movement appears as
a reviewable diff in a pull request, which is the cheapest possible regression
alarm. Most importantly, agents read results with ordinary file tools, with no
authentication, network, or client library between them and the numbers.

The **detail tier** stays machine-local: per-position diagnostics, full slice
tables, generated game records, and anything else whose size or shape would bloat
the repository. This mirrors the split the evaluation design already draws
between headline families and deeper diagnostics.

### Checkpoints Stay Local, With A Retention Policy

Checkpoints are not committed and are not mirrored to remote storage. What is
required is a retention policy: anchor checkpoints are kept rather than cleaned
up, because they are the left edge of every long-running comparison and the
input to re-scoring at a generation cut.

Remote storage was considered and deliberately left out. Its value here is
backup, and the exposure is bounded because runs are reproducible from
configuration, seed, and a corpus regenerable from a pinned source archive
digest. Losing a checkpoint costs a re-run, not an unreconstructable result.
Treating run reproducibility as a property worth protecting is what keeps that
true.

The trigger that would revisit this is a second machine rather than a disk
failure. When training moves to the GPU workstation while evaluation and
development continue elsewhere, checkpoints have to cross machines; ordinary file
copy is a sufficient first answer for an infrequent one-way move.

### Datasets Remain Regenerable Rather Than Stored

No change. The corpus is reproducible from a pinned source archive digest plus
checked-in configuration, and the evaluation pool is already specified as a
regenerable pipeline output rather than committed data. Storing gigabytes to
avoid re-running a deterministic pipeline is the wrong trade.

### No Experiment-Tracking Platform As Source Of Truth

MLflow, Weights & Biases, Neptune, ClearML, Aim, and DVC were surveyed. None is
adopted as the source of truth, for three reasons.

Their data model does not match ours. They are run-centric, mapping a metric
name to a time series of step and value. Decision 0013 is series-centric, keyed
on an input fingerprint with explicit bridges. That logic stays ours regardless
of the backing store, so a platform would serve as a sink, and as a sink it loses
to a committed file.

Agent access is worse. Reading results through a hosted API requires
authentication, network, and a client, against ordinary file reads for a
committed store.

Cost and operational burden are real for a solo project: a hosted tracker is a
per-seat subscription, and a self-hosted tracker is a service to run and keep
running.

DVC specifically is not adopted because the reproducibility claim it would make
is already covered by pipeline determinism and the pinned source digest.

What these tools would genuinely provide is a ready-made interface, which is the
one piece this plan has to build. MLflow in local mode remains available later as
a pure viewer over the committed store, with no server and nothing to migrate if
it is dropped.

### TensorBoard For Live Training Curves, On By Default

TensorBoard is used for step-indexed training curves. It is not deprecated, it
ships with PyTorch, it needs no service beyond its own command, and overlaying a
running experiment against previous runs at the same step is its core
competency. Its known weakness is scale, degrading somewhere in the hundreds of
runs, which does not bind for this project. Trackio is the closest modern
alternative and offers a compatible-enough interface that switching later stays
cheap; Aim solves a run-volume problem this project does not have.

Every training run emits TensorBoard output by default. Following a run must not
require the maintainer to configure anything, enable a flag, or ask an agent to
turn it on; the only step is opening TensorBoard against the run root.

Event files are a derived view and never a source of truth. The metrics stream
and the committed results store remain authoritative, and event files stay
regenerable and disposable. Emitting them from the same logging point that writes
the metrics stream is preferred over a separate conversion process, which was
considered and rejected as more machinery than the drift risk justifies.

Training curves and in-training evaluation readings belong in TensorBoard.
Cross-version project history does not: TensorBoard has no notion of
comparability fingerprints and will readily overlay two lines that decision 0013
says are not comparable. That job stays with the results store and its own
reporting.

## Consequences

An evaluation run ends by writing into a committed store, so the command must
write cleanly enough to review, and concurrent runs from separate worktrees can
conflict there. Keeping the committed tier small and append-oriented limits how
often that happens.

Repository size becomes something to watch. The summary tier is small by design,
and the boundary between tiers has to be enforced rather than assumed, since the
natural pressure is to commit one more useful diagnostic.

The project owns its own reporting and charts. That is the accepted cost of a
comparability model no off-the-shelf tracker implements, partially offset by the
option to point a local viewer at the store later.

Anchors accumulate on local disk. The volume is small because anchors are a
handful per milestone rather than every interval checkpoint, but a retention
policy that is never pruned eventually needs a pruning rule.

## Notes

External pricing and tier details surveyed in July 2026 informed the checkpoint
decision and are recorded as indicative rather than authoritative. Hugging Face
offered a free private-storage tier well above the volume anchors would need,
plus object-storage buckets with content-addressable deduplication that suits
checkpoints; Cloudflare R2 was materially cheaper than S3 for this access pattern
mainly because repeated checkpoint reads incur no egress charge. These would be
the starting points if the second-machine trigger ever makes remote storage
worthwhile.

## References

- `docs/evaluation.md`
- `docs/training-and-runtime.md`
- `docs/data.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
