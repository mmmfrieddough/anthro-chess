# 0061: A Training Cost Reading Has No Replicate To Resample

Date: 2026-08-13

## Status

Accepted. Extends
`0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`, which
requires every reading to carry a dispersion or state why the metric has none,
by answering that requirement for the training-efficiency family. Extends
`0029-model-change-control-arm.md` to a change read for what it cost rather than
for what it taught, which is what a family with no floor leaves as the only
instrument.

## Context

Decision 0043 left one obligation behind: a reading reports a dispersion beside
its value, or says why it has none. Every other family answered it by
resampling. The two efficiency families could not, because a timing has no
units inside it to resample, so inference answers it by re-running the whole
benchmark in several processes and reporting the spread between them.

That answer does not transfer. `anthro eval inference` is a benchmark, so a
replicate is another invocation of it. A training-efficiency reading comes from
`TrainingEfficiencyMonitor` inside a live training run, is not in the benchmark
registry, and has no invocation to replicate; a replicate of it is a second
training run. Six of those to qualify one reading is not a trade this project
would take, and the readings were left reporting no dispersion at all — so
every training-efficiency delta read `unknown`, the verdict that tells a reader
to wait for work somebody could do.

The one thing a training reading has that an inference latency percentile does
not is **internal units**. The monitor's unit is a drained logging interval,
and a window spans many of them, so three of the eight metrics are means over
intervals and a spread across those intervals is available for free.

## Decision

**All eight `training.*` metrics declare `no_sampling_floor_reason`**, and a
report renders them `unqualifiable` rather than `unknown`. The registry owns the
per-metric wording, and the eight fall into three shapes: three means over
intervals, four figures a run produces exactly once, and the padding fraction.

**The interval spread is refused.** Intervals inside one run share a process, an
allocator, a warm device and a session, so their spread is within-run jitter,
and the quantity a delta faces is the run-to-run variation. This is the same
objection that made one reading per process the rule for inference, and #161
measured what ignoring it costs: the same quantity on the same host measured a
spread twenty times wider once the machine was hot, and a floor characterized
quiet licensed four times as many false findings. A within-run spread is that
quiet characterization, taken from inside the very run it would qualify.

Erring narrow is the direction that matters. `noise.py` is built on the
principle that a floor understating the noise licenses noise as a finding, which
is the failure floors exist to prevent, and 0043 accepts a floor about 1.9x too
wide rather than risk one that is too narrow. An interval-mean floor would be
wrong in the forbidden direction.

Three identical 60-step runs, back to back on one idle RTX 4090, measured how
wrong. Their reported `training.step_seconds` spread by 5.5e-4 s, 8.9% of the
mean, and that is the quantity a delta between two readings faces. Each run's
own eight window intervals put the spread of the same reported mean at 1.3e-4 to
2.1e-4 s. So a within-run floor lands about **three times too narrow**, and it
does so on the most favorable case available: one machine, one session, minutes
apart, nothing else running. Two real training runs are none of those, which is
what #161's factor of twenty above is about. Three replicates estimate both
spreads loosely, and the chi-squared bound exists to price exactly that — but the
sign of the error is the finding here, not its exact size.

That also answers #369, which is what any estimator here has to do. Its finding
is that a pooled dispersion under-weights the between-process term whenever one
process contributes more than one reading. A within-run interval spread is the
degenerate case: one process, every reading, and the between-process term
weighted at zero.

The synchronization probe is the one metric with an argument against this, and
it does not survive. Its two arms are interleaved through one run precisely so
that drift lands on both, which cancels within-run variation out of the
difference rather than measuring it — so its intervals are the arms rather than
replicates of the difference, and a run produces exactly one of those. The
readings above measured `training.step_seconds` rather than the probe, and no
reading here establishes what the probe's difference does between runs.

**Four figures are produced once by a run**: a total, a share of a total, and a
high-water mark have no units inside them to resample at all.

**The padding fraction has a draw, and it is pinned rather than redrawn.** The
loader's shuffle is seeded from its configuration, which the reading records as
a coordinate, so two readings of one configuration see the same batches in the
same order and nothing inside either resamples that draw. A configuration that
does move it is a coordinate difference, which a report attributes. What else
moves the fraction between two readings is which of those batches their windows
spanned, which the warmup count, the logging interval and the probe period
decide — deliberately measurement settings rather than coordinates, per
`TrainingEfficiencyConfig`. That is a different window over the same fixed
batches rather than a redraw, so a resample is not the instrument for it either.

## Consequences

**A training-efficiency delta is never floored.** `unqualifiable` says the
reading cannot produce a floor rather than that nobody has yet, which is what
stops a reader waiting on work that would cost five training runs to do.

**A training-cost claim rests on the same instrument as any other causal claim.**
Decision 0029 already requires a control arm for a change that decides what a
model learns, and a change that alters training cost is read the same way: two
arms, identical but for the change, read once. That is a deliberate act rather
than machinery riding on every comparison, which is the posture 0043 takes for
seed noise as well.

**The declaration refuses a floor from any estimator, not only a resampled one.**
That is the field's existing contract, and it is the right default here: no
estimator a reading could carry is honest. A later decision to pay for replicate
training runs would remove the declaration rather than work around it.

The `no_sampling_floor_reason` field annotates a metric rather than redefining
it, so nothing here bumps a `definition_version`, changes a fingerprint, or
invalidates a recorded reading. Committed training-efficiency results keep their
identity and start reading `unqualifiable` in reports.
