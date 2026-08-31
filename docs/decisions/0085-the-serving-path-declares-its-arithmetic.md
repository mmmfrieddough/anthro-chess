# 0085: The Serving Path Declares Its Arithmetic

Date: 2026-08-30

## Status

Accepted.

Sits beside `0074-the-vehicle-freezes-on-bfloat16-with-tf32-without-a-quality-reading.md`,
which settled the same two dials for training and left the serving path alone.

## Context

A training run declares `precision` and `matmul_precision`, and both land in the
checkpoint's compatibility metadata. The runtime read neither. It loaded
parameters as float32, which is right and is checked, and then computed at
whatever float32 matmul precision Torch happened to default to, which was
`highest`.

So the arithmetic every served decision and every benchmark reading was produced
in was a framework default rather than a decision this project had taken. That
is the whole of the problem. The dial had no owner, and a Torch release that
moved its default would have moved every reading with it, silently, because
nothing recorded which one had run.

It matters more at the target size than at the vehicle's. Width 512 is 14.5 times
the vehicle's parameters, and where the vehicle is launch bound the target is
not: the forward pass at batch 1024 falls from 42,343 decisions per second to
11,300 as width goes from 128 to 512, so what the device does with a matmul
starts to be most of what a wide reading costs.

## What Was Measured

One untrained width-512 checkpoint and one trained width-128 checkpoint, on an
RTX 4090.

Throughput, decisions per second through `predict_batch` at batch 1024:

| setting | width 128 | width 512 |
| --- | --- | --- |
| float32, `highest` | 42,343 | 11,300 |
| float32, `high` | 51,345 | 14,558 |
| bfloat16 autocast | 64,457 | 20,211 |

What each does to a reading, taken as the novelty dose response over 1,600 games
on one checkpoint, largest shift in policy mass across its eighteen cells:

| setting | largest shift |
| --- | --- |
| float32, `high` | 0.0001 |
| bfloat16 autocast | 0.0010 |

For scale, that metric separates two checkpoints of one recipe four times apart
in training by 0.0067, and its sampling floor at the declared size is 0.0042.

And what each does to the agreement between a checkpoint served on the host and
the same checkpoint served on the accelerator, over 52 positions of one game:

| setting | largest logit difference | decisions that differ |
| --- | --- | --- |
| float32, `highest` | 0.00010 | 0 of 52 |
| float32, `high` | 0.06681 | 0 of 52 |

Only the accelerator moves. The host path computes the same logits at either
setting, to the last bit.

## Decision

**The runtime pins reduced-precision float32 matmul, keeps parameters at
float32, and records the pair on every reading that carries an execution
record.** The runner sets it around each forward pass and restores it after,
rather than once when a checkpoint loads. Torch holds the setting per process,
and a process that also trains would otherwise have the precision its own run
declared replaced by whichever model runner it happened to construct. Scoping it
also means a served decision computes the same way whatever the ambient setting
is, which a load-time pin does not guarantee once anything else moves it.

Reduced precision is chosen over full precision because full precision was never
the arithmetic behind these weights. Training's heavy matmuls run at bfloat16,
which is coarser than TF32 by a wide margin, so serving at TF32 is already more
precise than the arithmetic that produced the parameters. Serving at `highest`
buys precision the model never had, and pays for it on the only workloads where
the device is doing enough arithmetic to notice.

Bfloat16 is not adopted. It is the fastest of the three and it is what training
computes in, and both of those argue for it, but two measurements above argue
against. Its shift of 0.0010 on the tightest metric in the suite is a quarter of
that metric's own floor, spent on arithmetic rather than on anything the reading
is about. And it is slower than TF32 at batch 64, which is the width the serving
instrument and a real player use, so it pays only on the wide batches the
benchmarks run and lands its cost exactly there.

## What This Gives Up

Two backends now compute differently on purpose, and the gap between them is 660
times what it was. Nothing in the product needs them to agree: a served decision
is sampled from a policy, quality readings are taken on the accelerator, and a
0.0001 shift in an aggregate is two percent of the floor that aggregate is sized
against. But it is a real property of the system and it is the reason the
execution record now names the matmul setting rather than only the parameter
dtype. Two readings computed differently do not record the same string.

The acceptance check that a checkpoint transfers to the accelerator was written
when both backends ran identical arithmetic, and it expressed "reaches the same
decision" as logits agreeing to a relative 1e-4. That is no longer a statement
about this system, and a logit near zero has no meaningful relative agreement in
any case. It now asserts the decision itself and holds the logits to an absolute
tolerance, which still separates a working transfer from a broken one by two
orders of magnitude.
