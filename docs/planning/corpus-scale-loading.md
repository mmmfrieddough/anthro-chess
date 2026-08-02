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

The host is Linux with 32 cores. Training resolved `auto` to CPU, because the
training device selection does not accept CUDA on this branch; that is `#56`,
not a property of the loader. Every figure below is therefore a CPU figure, and
the loader's share of a step is the part that carries over.

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

500 steps, batch 16, four loader workers, `profile_phases` on:

```console
uv run anthro train --config configs/training/lichess-blitz-1m.toml --no-record
```

| configuration | startup | peak resident | wall clock |
| --- | --- | --- | --- |
| eager validation selection, preview cadence | 171.9 s | 4.03 GB | 3:16.8 |
| shard-backed validation selection, preview cadence | 130.2 s | 3.83 GB | 2:35.6 |
| shard-backed validation selection, no cadence | 7.2 s | 1.60 GB | 0:30.1 |

All three reported the same validation loss over the same 994 games, so the two
loaders resolve one selection and differ only in what they hold.

Steady state was **33,691 active positions/s**, and the phase split over the
500 steps was **1.35 s data, 3.02 s transfer, 12.39 s compute**. Data is 8% of
the step on this host: the loader is not the bottleneck at this model size, and
the standalone figure above says it has roughly twice this rate in reserve.

## What The Third Row Says

The two rows that are not 7.2 s are the finding. Once the training loader stops
reading the whole corpus, everything else that still does becomes the startup
cost, and both remaining offenders are outside the loader.

The eager validation selection is the smaller one at about 42 s, and a
configuration fixes it: the checked-in corpus-scale configuration reads both
selections through shards.

The preview cadence's view is the larger one at about 123 s and 2.2 GB, and a
configuration does not fix it. `_prepare_view` reads normalized rows directly
rather than through either loader, so it materializes every row of every shard
whatever the loaders do. That is `#195`, tracked separately because changing
what a preview reads is a change to a measurement rather than to a read path.

Both were invisible before this work, since the training loader read the whole
corpus too and these were the smaller of identical costs.
