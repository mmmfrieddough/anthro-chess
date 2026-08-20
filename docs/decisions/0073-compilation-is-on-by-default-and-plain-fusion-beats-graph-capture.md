# 0073: Compilation Is On By Default, And Plain Fusion Beats Graph Capture

## Status

Accepted.

Refines `0071-the-target-is-the-size-the-published-ladder-flattens-at.md`, whose
batch scan concluded that compilation would buy no throughput at these sizes.
The saturation finding it rests on holds; the inference about compilation does
not, and this record carries the measurement that settles it.

Rests on `0070-one-decision-per-pass-and-history-in-the-token-depth.md`, which
removed the per-ply legal-action iteration that made compilation impossible.

`0074-the-vehicle-freezes-on-bfloat16-with-tf32-without-a-quality-reading.md`
settles the precision default this record deliberately left open, and its
readings are taken with compilation on because the two compound.

## Context

Compilation was implemented, measured, and removed once already. The reason it
failed then was specific rather than general: the model iterated ragged
per-ply legal actions inside the step, Dynamo guarded on that iteration, and the
recompile budget ran out. `0070` removed the ply axis, and with it the
iteration, so the reason expired without anyone rereading the conclusion.

Two things then kept the conclusion alive longer than the reason. `0071` scanned
batch against throughput, found the memory ceiling did not bind, and inferred
from that that compilation would buy nothing. And a first re-measurement here was
taken at `float32` with `highest` matmul precision, which were the defaults then
and are not what any real run uses.

## Decision

**Compilation is on by default, in plain fusion mode, and the graph-capture
modes are not offered.**

## What Was Measured

One idle RTX 4090, the widened corpus, effective batch 16, 400 steps per point
with a 200-step steady-state window, against an eager control. Every row splits
that batch four ways except the width-256 bfloat16 one, which splits it two ways
and reads higher for that reason as well;
`0074-the-vehicle-freezes-on-bfloat16-with-tf32-without-a-quality-reading.md`
holds the same configuration at a micro-batch of 4.

| width | precision | eager | compiled | gain |
| --- | ---: | ---: | ---: | ---: |
| 256 | float32 | 8,041 | 9,334 | +16% |
| 256 | bf16 with TF32 | 15,558 | 22,621 | +45% |
| 512 | float32 | 3,521 | 3,878 | +10% |
| 512 | bf16 with TF32 | 6,906 | 9,117 | +32% |

**Compilation compounds with the precision dials rather than competing with
them.** They trade arithmetic for speed; compilation fuses the kernels a step
issues. Reduced precision makes a step faster, a faster step is more
launch-bound, and a launch-bound step is what fusion acts on. So the two
multiply, and the float32 rows above understate compilation by about a third.

The width-512 float32 figure is a correction. This record first read 3,493 there
and called the row no gain at all, from a run whose stored record carries neither
an efficiency measurement nor a checkpoint pointer;
`0074-the-vehicle-freezes-on-bfloat16-with-tf32-without-a-quality-reading.md`
re-measured it cleanly at 3,878.

**Either float32 row alone would have rejected it**, and the width-512 row would
have come closest to rejecting it. That is the same shape of error `0071` made
and the same one the original removal made: a reading taken at an operating point
nothing ships at, generalized to one that does.

## Why Graph Capture Is Not Offered

`reduce-overhead` and `max-autotune` both capture CUDA graphs. Both refuse a run
that accumulates gradients, because a captured graph writes each gradient to the
address it saw at capture and the loop frees those buffers between steps. Holding
the `.grad` tensors resident makes both run.

Made to run, both lose. At width 256 with bf16 they reach 14,391 and 14,865
against plain fusion's 18,435, and `max-autotune` spends thirteen minutes
compiling to get there. The loader buckets games by length, so a run presents
several shapes and a captured graph re-records rather than replaying, which is
the property graph capture exists to sell.

So the resident-gradient machinery buys two modes that are worse than the one
that needs none of it, and none of it is carried.

## Consequences

Compilation is an execution-compatibility key, because fusion reassociates
floating-point work and a continuation that changed it could not say which half
produced its weights. That retires every checkpoint written before it existed,
which `CHECKPOINT_VERSION` gates.

A run pays a fixed compile charge at its first step rather than at startup,
around 109 seconds at width 256. That is a loss below roughly six thousand steps,
so the smoke configurations turn compilation off and the test suite does the
same; a real arm leaves it on. `training.compiled_graphs` reports what the
compiler built, so a run that recompiles per step reads as the fixable mistake it
is rather than as a disappointing speedup.

What this does not settle is the precision default, which is the larger of the
two effects.
`0074-the-vehicle-freezes-on-bfloat16-with-tf32-without-a-quality-reading.md`
settles it. Compilation's value is stated above at both settings so that the
precision decision did not have to be retaken alongside it.
