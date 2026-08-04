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
processes, plateauing there against what was then read as the parent's own
share of the work. That plateau was the queue depth rather than the parent, and
"The Worker Dial Was Inert" below is what measured it. Worker counts of 0, 4,
8, 12, and 16 produced byte-identical batch sequences.

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
data time does not move, and the reason recorded here — that what remains is
the parent's own share, which no number of workers reduces — was wrong. The
next section is what measured it. The standalone plateau at 68,500 plies/s is
the same ceiling seen from the other side.

On a CPU run of the same configuration the balance is different: compute was
12.4 s against 1.35 s of data, so the loader had roughly twice its rate spare.
A larger model moves it back that way, which is worth remembering when capacity
scaling changes the shape of a step.

## The Worker Dial Was Inert, And Why

`workers` bounded nothing. The loader submitted jobs only until
`prefetch_batches` of them were outstanding, so a dial documented as how far
ahead batches are built also decided how many worker processes could ever be
running. Above it a worker never received a job at all; at it, every worker sat
idle from the moment its result was ready until the consumer came back for it.

Loader throughput on its own — no model, no device, 400 batches at batch 64 on
the million-game selection, active positions per second:

| workers | prefetch | positions/s |
| ---: | ---: | ---: |
| 8 | 8 | 142,266 |
| 16 | 8 | 142,774 |
| 24 | 8 | 141,374 |
| 8 | 16 | 214,426 |
| 8 | 24 | 214,530 |
| 16 | 16 | 210,364 |
| 24 | 24 | 280,411 |
| 32 | 32 | 326,329 |
| 24 | 48 | 369,849 |

The first three rows are the flat sweep that `cuda-training-proof.md` read as
something serial in the parent. The fourth row is the **same eight workers**
with the depth doubled and nothing else changed, 51% faster than the first,
which is what says the sweep was measuring the depth rather than decode
capacity. Eight workers saturate near 214,000 however deep the queue goes past
that; only then does the pool become the dial.

The loader now keeps `workers + prefetch_batches` jobs outstanding — one per
worker so none of them waits on the consumer, and the declared depth on top of
those. **The depth is a rate, not an order**: two 200-step runs under strict
determinism, one on each side of the change, reached bit-identical parameters
across all 47 tensors and the same validation record.

### What that returned to a run

Three paired rounds at the shape the Milestone 5 baseline selected, each arm
beside its partner rather than against a pooled mean, because this host is
shared and drifts:

```console
uv run anthro train --config configs/training/lichess-blitz-1m.toml --no-record \
  --set steps=3000 --set checkpoint_every_steps=3000 \
  --set profile_phases=false \
  --set train.loader.batch_size=64 --set validation.loader.batch_size=64
```

Phase profiling off, so the figure is wall clock rather than an instrumented
split. Every other reading below is the same command with the settings it
names.

| round | before | after | paired |
| ---: | ---: | ---: | ---: |
| 1 | 139,696 /s | 219,144 /s | +56.9% |
| 2 | 137,767 /s | 217,828 /s | +58.1% |
| 3 | 129,328 /s | 170,194 /s | +31.6% |

Median **+58.1%**, at the workers and prefetch depth the configuration already
carried. Mean step time falls from 30.7 ms to 19.4 ms.

The third round is the one worth reading. Another session's evaluation sweep
was running on this host for it, and the treated arm lost far more than the
baseline did — which follows, because the arm that lost is the only one using
the cores the sweep took. **A shared host is where this change is worth least**,
and +31.6% is the floor these three rounds establish rather than an outlier to
discard. A fourth round taken later, against a quiet host and four commits
further along, read 138,894 against 219,623.

## Where The Data Phase Goes

The phase counters cannot answer this — they carry the instrument bias the
closing note below describes — so the split comes from timing the loader by
itself instead:
`build_sharded_index` and `StreamingSequenceDataLoader` from this configuration,
then `_job` and `_materialize_batch` timed apart around a `next(loader)` loop
with no model and no device in the process at all. Per batch at batch 64, over
120 batches of the million-game selection:

| part | cost | where |
| --- | ---: | --- |
| bucket assembly | 2.4 ms | parent |
| decode | 151.5 ms | worker |
| collation | 5.9 ms | worker |
| interprocess round trip | 3.2 ms | both |

A batch here is 62 games and 4,088 plies. **Decode is 93% of the 163 ms**, and
it is the part that parallelizes, which is what made the depth defect worth
this much: it was discarding the pool that exists to absorb the one term large
enough to need it. The parent's own serial share is the 2.4 ms of bucket
assembly plus its half of the round trip, or about 4 ms, so the parent runs out
near 250 batches per second — far above where the pool does.

Decode is 2.4 ms per game, and the games are whole rather than chunked, so
nothing is decoded that a batch does not use. Making a game cheaper to encode
is a different change from this one, and it is what a pool eventually runs out
of room to hide.

## Is The Step Still Host-Bound

At batch 64, with phase profiling on so the phases can be told apart, and
reading the per-step columns rather than the shares:

| workers | data | transfer | forward and backward | optimizer | data ÷ compute |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8, before | 24.75 ms | 0.96 ms | 5.06 ms | 0.28 ms | 4.90x |
| 8 | 15.65 ms | 0.89 ms | 5.41 ms | 0.28 ms | 2.89x |
| 16 | 9.47 ms | 1.03 ms | 5.20 ms | 0.28 ms | 1.82x |
| 24 | 8.49 ms | 1.38 ms | 7.60 ms | 0.39 ms | 1.12x |

The first row is the reading `#181` was filed on, reproduced at this shape:
data 4.90 times compute. **Still host-bound at eight workers, and at parity by
twenty-four.** Not device-bound — a step at 24 workers still spends about as
long waiting for a batch as computing one, and the honest form of the claim is
that waiting stopped dominating rather than that it stopped mattering.

Three more paired rounds at batch 64, both arms carrying the fix and differing
only in the pool, say the same thing in wall clock: 197,395 against 230,909,
187,828 against 236,482, and 208,981 against 323,623 — a further 17% to 55%.
The spread is the point rather than noise around a single number, because a
larger pool is worth whatever share of the host is free, and the first two
rounds ran against another session's evaluation sweep while the third did not.

### The batch is what decides whether the pool matters

The same three arms at the batch this configuration declares, and at the batch
`#276` measures:

| batch | arm | data | forward and backward | data ÷ compute | positions/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | before, 8 workers | 3.24 ms | 4.11 ms | 0.79x | 129,243 |
| 16 | 8 workers | 1.48 ms | 4.13 ms | 0.36x | 167,302 |
| 16 | 24 workers | 1.27 ms | 4.13 ms | 0.31x | 171,402 |
| 256 | before, 8 workers | 98.99 ms | 7.36 ms | 13.45x | 141,879 |
| 256 | 8 workers | 65.76 ms | 7.60 ms | 8.66x | 207,849 |
| 256 | 24 workers | 34.90 ms | 8.57 ms | 4.07x | 379,651 |

**At batch 16 the step was never data-bound**, and tripling the pool there is
worth 2.5% — inside this host's spread, so `workers` stays at eight. At batch
256 the same triple is worth 83%, because what a pool absorbs is decode and
decode scales with the batch while the device work barely does. The dial to
raise is the pool, and the thing that decides when is the batch.

The depth fix is worth having at every one of them: +29% at batch 16, +58% at
64, +46% at 256, and it is the only change here that does not depend on
choosing a number.

Over a longer run — 15,000 steps at batch 64 and 24 workers, 11,240 of them
inside the steady-state window, so the phases are measured across minutes
rather than seconds:

| phase | seconds | per step | share |
| --- | ---: | ---: | ---: |
| data | 93.61 s | 6.24 ms | 43.2% |
| transfer | 19.26 s | 1.28 ms | 8.9% |
| forward and backward | 91.50 s | 6.10 ms | 42.2% |
| optimizer | 4.97 s | 0.33 ms | 2.3% |

**Data is 1.02 times compute**, against the 4.05 the issue was filed on, and
the data figure is the biased one. The same run without the instrument reads
290,245 active positions per second. This is the shape the rest of the
milestone can be read against — but the loader is still half of it, so a
device-side setting measured here is still measured at half strength.

Startup is 131 s of that run's 428 s and is excluded from the window rather
than averaged into it, which is the only reason a reading this short is worth
quoting. Most of it is the preview cadence, below.

Read every `data` figure in this document as an upper bound. Phase profiling
synchronizes the device at each boundary, so the parent cannot overlap its wait
on the loader with device work it already queued, and `data_seconds` is
measured as exactly that wait. That is why every improvement above is quoted
from the wall-clock arms rather than from a phase table.

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
