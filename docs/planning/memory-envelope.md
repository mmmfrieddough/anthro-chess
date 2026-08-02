# The 24 GiB Memory Envelope

This document records what fits on one 24 GiB card, measured rather than
inferred. It exists because two decisions in this milestone are bounded by that
envelope and neither had it: capacity selection chooses a model size against it,
and distributed training replicates the model rather than sharding it, so the
largest trainable configuration is whatever fits on a single device.

It is a measurement, not a recommendation. Choosing a point inside the envelope
belongs to capacity selection; this says where the walls are.

`docs/planning/cuda-training-proof.md` established that width is nearly free in
time. This is the reading that says what it costs in memory, and the short
version is that it is nearly free there too — for a reason that has nothing to
do with the transformer.

The two answers a capacity or distribution decision needs are in
[The Boundary](#the-boundary). At batch 256, float32 fits a model width of 3,072
and fails at 4,096; bfloat16-mixed fits 4,096 and fails at 5,120. Everything
before that section is how those numbers were arrived at, and why the obvious
way of estimating them is wrong.

## Host

The documented Linux CUDA machine from `docs/planning/cuda-training-proof.md`,
which owns its full specification: 32 cores, two RTX 4090s of **23.5 GiB** each,
one of them used. That per-card figure is the ceiling every number here is
measured against.

## What Was Measured

Peak **reserved** device memory, which is what the runner already records and
what decides whether a configuration fits: an allocator holding cached free
blocks still holds them, and an allocation fails against the pool rather than
against the live set. Every arm is an ordinary `anthro train` run against the
pinned million-game selection through the shard-backed loader, with the
committed results store bypassed:

```console
uv run anthro train \
  --config configs/training/lichess-blitz-1m.toml \
  --set 'evaluation.cadences=[]' \
  --set 'profile_phases=false' \
  --set 'device="cuda"' \
  --set 'train.loader.shuffle=false' \
  --set 'train.loader.batch_size=<batch>' \
  --set 'steps=<batches in one planning window>' \
  --set 'model.model_dim=<width>' \
  --set 'model.feedforward_dim=<2 × width>' \
  --set 'precision="<precision>"' \
  --set 'checkpoint_every_steps=1000000' \
  --no-record
```

Normalized and manifest paths are machine-specific and supplied at run time, as
every training configuration here expects. Depth stays at the baseline two
layers and attention at four heads except where a section below says otherwise,
so the grid moves one thing at a time.

Each arm consumes exactly one planning window — the checked-in
`planning_window_examples`, 16,384 — with shuffling off. Every conclusion here
about batch saturation is conditional on that field rather than on a project
constant. That is the unit worth measuring against because a window is where the
shard-backed loader's length buckets fill and flush, so a window is the span
over which the largest padded batch of the run is decided. The step counts
follow from the batch size and are the number of batches that window produces.

The host runs other work, and a card shared with another process makes both a
fit and an out-of-memory failure say nothing about the arm that observed it.
Each arm therefore sampled, every five seconds, the device memory held by
processes outside its own tree, and any arm that saw a nonzero figure was
discarded and re-run until it had the card to itself. Every number below comes
from an uncontended arm.

## The Sequence Axis Is Not The Longest Game

Sequence length is the third axis and the one that behaves least like
intuition, because the corpus is chunked by game rather than to a fixed length.
The naive bound — batch size times the longest game — is wrong by a wide margin,
and wrong in the safe direction.

The pinned corpus holds games up to 306 plies, and the longest sequence its
train split hands the loader is 302 timesteps. That sequence never appears in a
large batch. Length bucketing groups similar lengths together, and games that
long are rare enough that their bucket never fills, so the longest sequence in
the epoch arrives as a **batch of one**. The batch that actually costs the most
is a full one at a moderate length.

Enumerating the loader's own epoch plan, which needs no game decoded because a
game's length follows from the index:

| batch | batches per epoch | worst padded batch | naive batch × 302 |
| ---: | ---: | ---: | ---: |
| 16 | 56,779 | 16 × 208 = 3,328 | 4,832 |
| 64 | 14,647 | 64 × 176 = 11,264 | 19,328 |
| 256 | 4,123 | 256 × 144 = 36,864 | 77,312 |
| 1024 | 1,556 | 990 × 128 = 126,720 | 309,248 |

At every one of those batch sizes the 302-timestep sequence arrives alone, in a
batch of one.

The gap widens with batch size, because a bucket at a long length needs more
games to fill the larger batch and the corpus has fewer of them the longer they
get. At batch 1024 the naive bound overstates the real peak by 2.4×.

Two consequences follow, and both matter to anyone sizing a run from a short
one:

**A short run understates the peak.** At batch 16 the worst batch of an epoch is
one batch in 56,779, and in the shuffled epoch a real run uses, the first batch
within 10% of it does not arrive until step 404. A hundred-step run reports a
peak that an unattended overnight run will exceed. This is why the arms below
are sized to a whole planning window rather than to a convenient step count.

One window is not the whole epoch, and the residual is worth naming rather than
waving at. The worst batch of the first window reaches 92% of the epoch's worst
at batch 16, 91% at batch 64, 100% at batch 256, and 91% at batch 1024, and the
first ten windows span a narrow band around those. Every figure below therefore
carries up to roughly a tenth of understatement against a full epoch at batch
16, 64, and 1024, and none at batch 256. One arm was then run over a whole epoch
to check that residual rather than assert it; it is measured under
[The Boundary](#the-boundary).

**Above roughly 3,072 the batch dial stops doing anything.** Within one
16,384-example planning window, a batch that large can no longer be
filled from one bucket in one window, so the worst padded batch saturates:

| requested batch | worst padded batch in a window |
| ---: | ---: |
| 1,536 | 1,429 × 112 = 160,048 |
| 2,048 | 2,048 × 96 = 196,608 |
| 3,072 | 3,072 × 80 = 245,760 |
| 4,096 | 3,156 × 80 = 252,480 |

Past that point the planning window, not the batch size, is the thing setting
the shape. Raising the batch dial further neither costs more memory nor buys
more work per step; it is the window that would have to move.

## Peak Reserved Memory

Gibibytes, at the baseline two layers and four heads. Each column heading
carries the worst padded batch that window produces, because that shape rather
than the requested batch size is what the model saw.

### float32

| width | batch 16 (16×192) | batch 64 (64×160) | batch 256 (256×144) | batch 1024 (897×128) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.32 | 1.19 | 3.73 | 10.40 |
| 128 | 0.31 | 1.25 | 3.93 | 11.07 |
| 256 | 0.39 | 1.34 | 4.49 | 13.42 |
| 512 | 0.56 | 1.54 | 6.61 | 21.11 |
| 1024 | 1.14 | 3.49 | 11.22 | **OOM** |

### bfloat16-mixed

| width | batch 16 (16×192) | batch 64 (64×160) | batch 256 (256×144) | batch 1024 (897×128) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.21 | 0.82 | 2.61 | 8.01 |
| 128 | 0.22 | 0.85 | 2.73 | 8.43 |
| 256 | 0.27 | 0.93 | 2.95 | 9.54 |
| 512 | 0.45 | 1.04 | 4.34 | 14.03 |
| 1024 | 0.90 | 2.14 | 8.64 | 17.71 |

**These figures are exactly reproducible.** Three repeats of float32 at width
256 and batch 1024 returned 14,413,725,696 bytes every time, and three repeats
at width 64 and batch 256 returned 4,005,560,320 bytes every time — identical to
the byte, not merely to the second decimal. The allocator's pool is a
deterministic function of this workload, so the boundary below is a line rather
than a band and needs no safety factor for measurement noise.

That determinism has one condition, and it is why this document says so much
about a shared host. A card held by another process at the same time produces a
*lower* figure: the same cell read 10.94 and 11.73 GiB with a multi-gibibyte
neighbour resident, against 13.42 alone. Co-tenancy makes the
allocator cache less, so a reading taken beside other work understates the
requirement, and at the boundary it turns a fit into a spurious out-of-memory
failure. One arm here first reported OOM for float32 at width 512 and batch
1024; measured alone it fits at 21.11 GiB.

## Where The Memory Actually Goes

Not into the parameters, and at the current width not into the transformer
either.

Parameters and their Adam state are close to a rounding error: 276,002
parameters at width 64 and 22,579,682 at width 1024, so 4.4 MB and 361 MB
respectively for weights, gradients, and both optimizer moments together. At
width 1024 and batch 256 that is 361 MB against a measured 11.22 GiB, or 3%.

What the table shows instead is a fixed cost plus a per-layer one. Holding
width 256 and batch 256, and varying only depth:

| layers | peak reserved |
| ---: | ---: |
| 2 | 4.49 |
| 4 | 5.64 |
| 8 | 9.00 |
| 16 | 16.24 |

That is 0.84 GiB per layer over a 2.8 GiB intercept, and the intercept is the
part that does not care about model size. It is the action head and the loss:
the head projects to the whole action vocabulary — `ACTION_VOCABULARY_SIZE` in
`anthro_chess.chess.actions`, 1,970 entries when this was measured — so the
logits are padded positions × that vocabulary, however narrow the trunk in front
of them is. A vocabulary change moves every figure below with it.

Measured on its own, at padded shapes this loader really produces:

| padded batch | one logits tensor | logits and loss, forward and backward |
| --- | ---: | ---: |
| 64 × 176 | 0.08 | 0.40 |
| 256 × 144 | 0.27 | 1.29 |
| 1024 × 128 | 0.96 | 4.60 |

The loss step alone costs roughly five times one logits tensor, and at width 64
and batch 256 it is about a third of the entire run's peak. That is why width
looks nearly free until it does not: from 64 to 256 the trunk is smaller than
the head behind it, and past 512 the trunk takes over.

It also means the axes bite differently. Width and depth buy capacity and cost
trunk activations. Batch and sequence length cost trunk activations *and* the
vocabulary-sized head, which is the larger term at the sizes this project
trains today.

Head count is nearly free until it is not:

| heads | peak reserved |
| ---: | ---: |
| 4 | 4.49 |
| 8 | 4.62 |
| 16 | 6.16 |

## The Boundary

Located by running until the card refused, not by extrapolating to where it
would.

**Width, at batch 256.** Both precisions have a measured wall. float32 fits
width 3,072 and fails at 4,096; bfloat16-mixed fits 4,096 and fails at 5,120.

| width | float32 | bfloat16-mixed |
| ---: | ---: | ---: |
| 1024 | 11.22 | 8.64 |
| 2048 | 20.56 | 16.59 |
| 3072 | 21.68 | not measured |
| 4096 | **OOM** | 22.58 |
| 5120 | not measured | **OOM** |

**Batch, at width 64.** There is no wall on this axis, because the shape
ceiling arrives before the memory one. Peak reserved stops climbing once the
planning window rather than the batch dial is setting the shape:

| batch | float32 | bfloat16-mixed |
| ---: | ---: | ---: |
| 1024 | 10.40 | 8.01 |
| 1536 | 12.77 | 9.44 |
| 2048 | 12.43 | 9.65 |
| 3072 | 12.31 | 10.52 |

Tripling the dial from 1,024 to 3,072 moves float32 by 1.9 GiB, all of it
arriving by 1,536; past that the figure declines slightly rather than rising,
because the padded shape has stopped growing in proportion to the dial. Whatever
else a larger batch buys past roughly 1,536, it is not more memory pressure, and
it is not more work per step either.

**What fits at a useful batch size.** Batch 256 is where the CUDA proof found
throughput still climbing, and it is also the batch whose worst padded shape one
planning window reproduces exactly, so both answers below are epoch-exact rather
than window-approximate:

- **float32: width 3,072** at two layers, using 21.68 GiB of 23.5.
- **bfloat16-mixed: width 4,096** at two layers, using 22.58 GiB of 23.5.

Both are the last width that fits before the next one measured fails, so each is
a wall rather than a lower bound. Neither is a recommendation: both sit above
95% of the card, and a run at either has no room for the checkpointing,
validation, or evaluation cadence a real training configuration also carries.

At batch 1024 those answers drop to width 512 for float32 and width 1,024 for
bfloat16-mixed. Depth trades against width at 0.84 GiB per layer at width 256,
so a deeper, narrower model is available inside the same envelope. Which point
in it to pick is `#54`'s question, not this document's.

### The Window Understatement, Confirmed

One arm was run over a whole epoch rather than one window, to check the residual
named earlier instead of asserting it. float32 at width 512 and batch 1024 was
chosen because it has the least headroom of any fitting cell, so if a tenth of
understatement mattered anywhere it would matter there.

| span | steps | worst padded batch | peak reserved |
| --- | ---: | ---: | ---: |
| one planning window | 28 | 897 × 128 = 114,816 | 21.11 |
| the whole epoch | 1,556 | 990 × 128 = 126,720 | 22.90 |

The epoch costs **8.5% more** than the window, against a worst padded batch
10.4% larger — so the residual is real, is close to the size predicted from the
shape plan alone, and it still fits, with 0.6 GiB to spare. Read the batch-16,
64 and 1024 rows of the grid as carrying a similar understatement; the batch-256
rows do not, because there the window reproduces the epoch's worst shape.

That 0.6 GiB is what the caution above looks like as a number: this cell
survives its own epoch and nothing else.

## What Mixed Precision Is Worth

`cuda-training-proof.md` kept `precision = "bfloat16-mixed"` on memory grounds
while measuring it 7% slower, at a width where memory was not binding, and said
this sweep is where that tradeoff either pays or does not. **It pays.**

The saving runs 19–39% across the grid and grows with the trunk, which is what
the mechanism predicts, since activations are held at half width while
parameters and optimizer state stay float32. At width 512 and batch 1024 it is
34%.

More decisively, it is the difference between a configuration existing and not
existing. float32 cannot train width 1,024 at batch 1024 at all; bfloat16-mixed
does it at 17.71 GiB with 5.8 GiB to spare. On the width axis at batch 256 it
moves the wall from 3,072 to 4,096.

None of that changes the default. The throughput cost is real, and half-width
activations are only worth buying when memory is the constraint. What is new is
that there is now a configuration the project might actually want which is
reachable only through the dial.

One incidental number fell out of that attribution and is recorded here only so
it does not have to be rediscovered. `masked_action_cross_entropy` selects with
a boolean mask, which materializes a second near-full copy of the logits and its
gradient; computing the same loss over a flattened view with `ignore_index`
avoids the copy and, measured in isolation at the three shapes above, saves
15.2%, 15.7% and 16.2% of the loss step. Against a whole run that is a few
percent — worth someone's time, not a tier of this envelope, and not changed
here.

## What This Does Not Show

- **Nothing about quality.** No arm here ran long enough to move a held-out
  metric, and none was scored against the frozen pool. Every configuration this
  document says fits is a configuration that fits, not one worth training.
- **Nothing about throughput.** These runs were sized to reach a shape, not to
  measure a rate, and they shared a host with other work. Their
  positions-per-second figures are not comparable with `cuda-training-proof.md`
  and are not reported. Memory is the only quantity here.
- **Nothing about the second card.** Every figure is one device, which is the
  point rather than a gap, for the reason the opening paragraph gives.
- **Nothing that survives a change of corpus or planning window.** The shape
  ceiling is a joint property of this corpus's length distribution, this
  bucket width, and this planning window. A corpus with a longer tail, a wider
  window, or a different bucket width moves it, and the memory figures move
  with it. The method here is what carries over, not the numbers.
- **Nothing about inference or evaluation.** Those cross their own device
  boundary and were not measured.

One measurement caveat belongs with the numbers rather than after them.
Reserved memory includes blocks the caching allocator holds but is not using,
so it overstates live tensors and includes fragmentation. That is deliberate:
an allocation fails against the pool, so the pool is what decides a fit. It
also means these figures are not a parameter-and-activation accounting and
should not be read as one.
