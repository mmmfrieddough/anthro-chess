# Corpus-Scale Loading

This document records the first evidence that a training run reads the prepared
million-game selection. It is an implementation proof about the read path, not
a claim about the checkpoint it produced: the run below is 500 optimizer steps
and exists to measure startup, memory, and throughput.

`docs/data.md` owns what the two loaders are and why they order an epoch
differently. This records what each cost when measured.

## What Was Measured Against

The pinned `lichess-blitz-2017-04` selection, prepared from its checked-in
configuration: 1,000,000 accepted games in 20 shards, 198 MB normalized, one
row group per shard. Of those, 900,218 fall in the train split and 49,709 in
validation.

The host is the documented Linux CUDA machine from
`docs/planning/cuda-training-proof.md`: 32 cores, two RTX 4090s, one used.
Training resolved `auto` to CUDA.

## Why The Eager Loader Cannot Read It

Loading the whole train split of the 30k proof corpus eagerly — 26,944 games,
1,849,238 plies — took **143.9 s** and **2,515 MB**, or 94 KB and 5.3 ms per
game. A per-ply encoding is far larger than the normalized row it came from,
and all of it is retained.

Scaled to 900,218 games that is roughly **80 GB resident** and **80 minutes**
before the first optimizer step. The corpus is not readable this way on any
machine the project targets.

The cost is set by the artifact rather than by the selection, which is the part
worth stating plainly. Selecting **994 games** out of the million-game corpus
eagerly still took **45.3 s** and **1,727 MB**, because every row of every shard
is materialized before anything is filtered. A small held-out selection is not
a cheap one.

## What The Shard-Backed Loader Costs

Against the same corpus:

| phase | cost |
| --- | --- |
| verify all 20 shard digests against the manifest | 0.12 s |
| index the 900,218-game train split | 4.00 s, 826 MB |
| first batch | 0.31 s |

Nothing in the index decodes a game or hashes a shard twice. Sequence length
follows from `ply_count` and whether a terminal action was appended, and the
shard digests come from the manifest check that already ran.

Loader throughput on its own, over seven shards of this corpus at batch 16, was
**13,300 plies/s** decoding in process and **68,500 plies/s** at eight worker
processes, plateauing there against the parent's own share of the work. Worker
counts of 0, 4, 8, 12, and 16 produced byte-identical batch sequences.

## The Run

500 steps, batch 16, eight loader workers, `profile_phases` on:

```console
uv run anthro train --config configs/training/lichess-blitz-1m.toml --no-record
```

| configuration | startup | peak host resident | wall clock |
| --- | --- | --- | --- |
| as checked in, preview cadence declared | 125.4 s | 4.40 GB | 2:25.4 |
| same, no cadence | 7.6 s | 2.25 GB | 0:24.4 |

Steady state was **59,846 active positions/s** against 274.7 MB reserved on the
device. The run reported the same validation loss over the same 994 games as
the CPU run it replaced, and as the run that read validation eagerly, so the
two loaders resolve one selection and differ only in what they hold.

## Where A Step Goes, And What Workers Buy

Phase seconds over the 500 steps, sweeping the worker dial and changing nothing
else:

| workers | data | transfer | compute | steady state |
| --- | --- | --- | --- | --- |
| 4 | 4.37 s | 2.85 s | 3.35 s | 54,556 /s |
| 8 | 3.20 s | 3.06 s | 3.51 s | 59,437 /s |
| 12 | 3.29 s | 2.87 s | 3.75 s | 56,944 /s |
| 16 | 3.04 s | 3.03 s | 3.60 s | 58,476 /s |

Decoding is the largest phase at four workers and stops being so at eight,
where the three phases are within half a second of each other. Past eight the
data time does not move, because what remains is the parent's own share —
taking a batch's rows out of the columnar table and reading a packed batch back
from a worker — which no number of workers reduces. The standalone plateau at
68,500 plies/s is the same ceiling seen from the other side.

Eight is therefore what the checked-in configuration carries. On a CPU run of
the same configuration the balance is different: compute was 12.4 s against
1.35 s of data, so the loader had roughly twice its rate spare. A larger model
moves it back that way, which is worth remembering when capacity scaling
changes the shape of a step.

## What The First Table's Two Rows Say

The gap between 7.6 s and 125.4 s is the finding. Once the training loader stops
reading the whole corpus, everything else that still does becomes the startup
cost, and both remaining offenders are outside the loader.

The eager validation selection was the smaller one at about 42 s, and a
configuration fixes it: the checked-in corpus-scale configuration reads both
selections through shards.

The preview cadence's view is the larger one at about 118 s, and a configuration
does not fix it. `_prepare_view` reads normalized rows directly rather than
through either loader, so it materializes every row of every shard whatever the
loaders do. That is `#195`, tracked separately because changing what a preview
reads is a change to a measurement rather than to a read path.

Both were invisible before this work, since the training loader read the whole
corpus too and these were the smaller of identical costs.
