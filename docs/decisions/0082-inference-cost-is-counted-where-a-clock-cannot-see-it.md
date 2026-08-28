# 0082: Inference Cost Is Counted Where A Clock Cannot See It

Date: 2026-08-28

## Status

Accepted. Refines `0021-efficiency-identity-excludes-compared-conditions.md`.

## Context

The inference benchmark existed to answer what a checkpoint costs to play with,
and it answered that with wall clocks: batch-one latency, throughput at a
declared batch, cold start. The purpose the suite actually needs from it is
narrower and sharper. When a change to the model slows inference down, the
benchmark should say so, and say by how much, so the tradeoff can be weighed
rather than discovered later.

Measured against that purpose it was blind. Three widths of the same
architecture, spanning 1.42M to 5.21M parameters and 0.167 to 0.628 GFLOP per
decision, moved the two committed metrics by three percent, against a
run-to-run spread of two and a half percent. The reading was noise, and it was
not monotone: the middle width read slower than the largest.

The cause is the shape of the model rather than a flaw in the timing. One
decision reads a fixed set of sixty-four square tokens, and history folds into
their channels rather than into the sequence, so a forward pass is one position
per row however deep the game is. On an accelerator that is not enough work to
measure. The forward pass took the same time at batch one as at batch sixty-four,
which is to say roughly all of it was kernel-launch overhead: around 250 launches
per decision. Collapsing those into a single graph replay cut the batch-one
forward from 4.39 ms to 0.56 ms and left the separation between widths at 1.04x,
so this is a property of the work being small rather than of the launches being
many.

Two other measurements shaped what replaced it. The whole-decision throughput
figure barely separates the widths at any batch size, because most of a batched
decision is host work that does not change when the model does. And the host
reading, which the benchmark did not take at all, separates them at batch one
by 1.40x while being the quieter of the two devices from process to process.

## Decision

**The benchmark counts what a decision costs the model, and times what it costs
to play with.** Parameters, floating-point operations per decision, and peak
device memory are read from the loaded module. They carry no noise, need no
floor, and no process count buys resolution on them.

The counts do not replace the timings and the timings do not approximate the
counts. Wall clock understates an arithmetic increase whenever the device is
launch or bandwidth bound, which is where this model sits at every batch size it
serves: a 3.76x increase in operations reads as 1.92x in time even at the widest
batch measured. That gap is the answer to what a size tradeoff costs, so both
halves are reported.

**A run on an accelerator measures the host as well.** It answers whether the
engine is playable without an accelerator, which nothing else here addresses,
and it is the only single-decision wall clock that tracks the operation count.

**The serving batch is a product figure and the compute batch is an
instrument.** The first prices concurrent players. The second is wide enough
that the device is no longer launch bound, which is the only batch size at which
a wall clock separates two model sizes; nothing serves that wide.

**The declared workload names the device.** This refines 0021, which reserved
identity for changes that make a difference meaningless and left the machine a
coordinate. That rule is unchanged for the case it was written against: a delta
between two accelerators is still interpretable, still attributed rather than
refused, and the specific silicon stays a coordinate. What is new is that one
invocation now measures on two device classes deliberately, so they are two
declared conditions of one benchmark rather than one measurement taken twice.
Left as coordinates they would share a fingerprint, and a checkpoint's history
would alternate between devices on one line while a delta compared whichever
reading was written last.

**The replicate processes produce the value as well as the floor.** Repeating a
measurement inside one process reproduces it several times more closely than a
fresh process does, so where a process lands is nearly the whole of the noise
and measuring more decisions inside one buys nothing. Committing the parent's
own reading and keeping the replicates for the dispersion alone left the number
as noisy as a single process while paying for several; the committed value is
now their median.

## Consequences

The depth sweep and the batch-size sweep are gone. Latency is flat in history
depth for the architectural reason above, so one depth says what every depth
says, and the batch sweep measured five points inside one launch-bound regime.

Every series here ends. The benchmark version, the device, and both batch sizes
are declared workload, so nothing recorded before this continues. That is the
correct outcome rather than a cost to weigh: the previous headline metrics were
measuring a regime the suite does not serve in and could not see the change this
benchmark exists to catch.

A model change that alters the arithmetic is now detected exactly. One that
alters wall clock without altering arithmetic, such as a change to the kernel
count or the memory layout, is detected against the measured process spread, and
the process count is the lever on that.

Collapsing the kernel launches is a real optimization this measurement surfaced
and does not perform. It is worth roughly 7x on single-decision latency and is a
change to the serving path, not to this benchmark.
