# 0091: Ranks Share A Step, And The World Size Is Provenance

Date: 2026-10-02

## Status

Accepted.

Keeps the guarantee
`0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md` built
the training digest for: a run on two cards and a run on one at the same
configuration share a digest because they compute the same thing, and no
setting that changes what a gradient is can stay outside it.

## Context

What a gradient depends on is the loader batch, the accumulation, and the number
of ranks the step is spread across. The first two are inside `training_sha256`
and the third was in nothing, so an arm trained on two cards at an unchanged
per-rank batch would have doubled its effective batch, kept its digest, and been
read against a seed floor that never measured it. Adding the world size to the
digest closes that and makes every two-card arm permanently incomparable to a
one-card one, including where both compute the same thing.

The issue that asked for data-parallel training preferred making the loader's
batch the global one and splitting it across ranks. That cannot be done here: a
training batch is decision-shaped
(`0075-a-training-batch-is-decisions-not-games.md`), one row of a fixed number
of decisions, and a row has no second dimension to split.

## Decision

**The configuration's effective batch, loader batch times accumulation, is the
global batch, and ranks share its micro-batches.** Each of `n` ranks runs
`accumulation / n` micro-batches and gradients are averaged across ranks once,
after the last. The per-rank share is derived rather than declared, an
accumulation the ranks cannot share evenly is refused, and the world size is
execution provenance beside the accumulation it divides.

The loaders serve this by cutting the one order a single process reads into
groups of `n` batches, each rank taking its own member of a group and every
rank's cursor passing the whole group. A checkpoint's cursor is then a place in
the single-process order and resumes under any world size.

## What Was Measured

The ablation vehicle at its own configuration, 1,200 steps with a shortened
warmup, recording nothing, back to back on the two RTX 4090s of the project's
host. Compilation and bfloat16 are on, as they are for every real run.

| cards | active positions/s | step time | peak reserved per card | wall clock |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 54,156 | 0.302 s | 2.50 GB | 6:26 |
| 2 | 103,503 | 0.158 s | 2.54 GB | 3:54 |

**1.91 times one card, a scaling efficiency of 95.6%.** Both runs processed the
same 19,650,652 positions, and each card held within 2% of what one card held
alone, as a replicated model should. A gradient all-reduce over PCIe without peer-to-peer
transfer costs this model about four percent of a step, and the model is small
enough that nothing larger was expected.

Where that four percent goes is known and was not chased. The model is compiled
in place before the data-parallel wrapper sees it, so the whole backward pass is
one compiled node, every gradient arrives at once, and the reduction cannot
overlap it. Compiling the wrapper instead is the change that would buy some of it
back, and a second card that already returns 1.91 is not waiting on it.

## What This Gives Up

**A configuration with accumulation one cannot run on two cards as declared.**
Halving its loader batch to make room for accumulation two changes its loader
digest, which is the honest outcome: those are two configurations. A
configuration meant for several cards declares an accumulation they divide, and
the vehicle's sixteen divides every card count this host could hold.

**An epoch ending in a partial group drops it.** A rank cannot take a batch the
others have no partner for, so from the first epoch whose batch count the world
size does not divide, the batches a step trains on stop matching a single card's.
A vehicle arm reads under one percent of one epoch, so this is a property of
fixtures and short-corpus runs rather than of any comparison the digest serves.

**The weights are not bit-identical to a single card's.** The reduction sums
partial gradients in a different order. Nothing reads weights bit for bit across
two configurations of the same digest; the seed floor is what a comparison of
two runs rests on, and that floor already absorbs relaxed-determinism
nondeterminism of the same kind.

## Consequences

A two-card arm of the vehicle carries the vehicle's digest and is read against
its seed floor like any other arm.

`docs/training-and-runtime.md` states the rule and `docs/data.md` how the
loaders divide an epoch.
