# CUDA Training Proof

This document records what the single-GPU CUDA training path is worth, measured
rather than assumed, and which candidate optimizations were tried and rejected.
It is an implementation proof for the shared runner on a new backend. It is not
a claim about model quality, and nothing here was appended to the committed
benchmark store: every run below used `[efficiency] record = false`, because
committing a benchmark result is a separate decision about project history.

The headline is that the accelerator helps, that it helps far less than the
hardware could, and that the reason is measurable and lives outside the model.

## Host

| | |
| --- | --- |
| Accelerator | 2 × NVIDIA GeForce RTX 4090, 23.5 GiB each, compute capability 8.9 |
| Host | Linux x86-64, 32 logical cores, Torch defaulting to 16 threads |
| Torch | 2.13.0+cu130, CUDA runtime 13.0, cuDNN 9.2.0 |

Only one card is used. Distribution is separate work and shares this as its
baseline.

## Reproducing

Every measurement below runs the ordinary `anthro train` command against the
pinned corpus, not a synthetic loop. Stand the corpus up first — the exact
selection the Milestone 5 baseline trains on:

```console
uv run anthro data acquire \
  --config configs/data/lichess-blitz-2017-04.toml

uv run anthro data prepare \
  --config configs/data/lichess-blitz-2017-04.toml \
  --set 'artifact_name="lichess-blitz-proof-30k-v4"' \
  --set filters.maximum_games=30000 \
  --set output.games_per_shard=30000
```

That yields 30,000 accepted games over 2,052,632 plies, split into 26,944
training games. The measured configuration is the checked-in baseline model —
`model_dim` 64, two transformer layers, four heads — at loader batch 16 with no
gradient accumulation, which is what the shipped baseline selection trains.

The shortest check that the device path works at all needs no corpus beyond the
checked-in sample:

```console
uv run anthro train --config configs/training/cuda-smoke.toml
```

## What The Accelerator Is Worth

Steady-state active positions per second, which excludes warmup, checkpointing,
evaluation, and the run's own instrumentation. Both devices ran the same model,
corpus, seed, and loader configuration.

| Device | Batch 16 | Batch 64 | Batch 256 |
| --- | ---: | ---: | ---: |
| CPU (16 threads) | 43,425 | 37,540 | 35,295 |
| CUDA (one 4090) | 86,799 | 142,289 | 168,107 |
| CUDA ÷ CPU | 2.00× | 3.79× | 4.76× |

At the batch the project currently trains, one 4090 is worth exactly twice the
16-thread host. That clears the bar, and it is also a disappointing figure for
this hardware, so the rest of this document is about why.

The two devices respond to batch size in opposite directions, which is the more
useful result. CUDA gains 94% going from batch 16 to 256 and has not stopped
climbing; the CPU *loses* throughput as the batch grows, because a wider batch
buys it no parallelism it did not already have and costs it cache locality.
Peak reserved device memory at batch 256 was 2.6 GiB of 23.5 GiB, so the batch
axis is nowhere near a memory ceiling.

The batch-256 columns rest on short steady-state windows — 140 CUDA steps and
15 CPU ones — because a wide batch consumes the same positions in fewer steps.
They are firm enough to establish the direction and the rough size of the gap,
and the batch-16 row is the one to quote precisely.

## Where The Time Goes

Phase profiling attributes each part of a step separately, with a device
synchronization at each boundary so the attribution is real rather than an
enqueue time. Cumulative seconds, and the same figures divided by their step
counts:

| Phase | CUDA (400 steps) | CPU (120 steps) | CUDA per step | CPU per step |
| --- | ---: | ---: | ---: | ---: |
| Input preparation | 0.748s (13.8%) | 0.248s (7.7%) | 1.9 ms | 2.1 ms |
| Batch construction and transfer | 1.904s (35.2%) | 0.567s (17.6%) | 4.8 ms | 4.7 ms |
| Forward and backward | 2.486s (46.0%) | 2.128s (66.1%) | 6.2 ms | 17.7 ms |
| Optimizer | 0.265s (4.9%) | 0.278s (8.6%) | 0.7 ms | 2.3 ms |

Read the per-step columns rather than the percentages. **Model compute is 2.9×
faster on the accelerator. Batch construction and transfer costs the same 4.8 ms
either way.** That fixed host-side cost is what collapses a 2.9× compute win
into a 2.0× end-to-end one, and it is why the card sat at roughly 11% utilization
throughout.

The reason it is the same on both devices is that the phase is not dominated by
the copy across the bus. It is dominated by building tensors out of the loader's
nested Python sequences on the host, which happens identically whether the
result then stays put or crosses to a device. The input pipeline, not the
accelerator, is the binding constraint at the current model size.

The loader has since stopped emitting those sequences. On this workload and
host, batch construction and transfer falls from 4.1 ms to 0.8 ms per step and
forward and backward becomes the largest phase, which is what the last section
of this document asked for.

The decode upstream of it then became what a step waited on, and it has since
stopped reconstructing per-ply legal actions for a training run that never
reads them. Under the shard-backed loader at eight workers, input preparation
falls from 7.3 ms to 1.6 ms per step and forward and backward is the largest
phase again.

What survived in that phase was mostly not the copy either. The tensor boundary
validated the tensors it had just built, on the device, on every micro-batch:
ten whole-column rules, about forty kernels, and one read back. It now checks
the loader's arrays instead, before they cross, because the copy only widens and
those are the same values. Batch construction and transfer falls from 1.34 ms to
0.44 ms per step, and the phase stops being the one that host contention moves —
across six counterbalanced rounds on this shared host the checked-on-device arm
ranged 0.71 ms to 1.80 ms while the checked-on-arrays arm held 0.40 ms to
1.01 ms. Forty small kernel launches are host work, and host work is what a busy
machine takes away.

Separating optimizer work from forward and backward was worth doing, and not
for the reason expected: at 4.9% of a step it looked like the last place worth
touching, and it turned out to hold the only throughput win found here.

## Optimizations Measured

Each was implemented and measured on the workload above. One was kept, one was
kept for a different reason than it was tried for, and the rest were removed:
retaining a dial that never wins costs more than the measurement it preserves,
and the readings below are what preserve it instead.

**This table has since been re-read and most of its verdicts changed**, because
the workload it describes was replaced. Read it as the record of what a
host-bound step was worth optimizing, and take the current verdicts from
[the re-read below](#the-table-re-read-against-a-device-bound-step). TF32 in
particular was removed on the strength of the reading here and now exists again
as `matmul_precision`.

| Candidate | Throughput | Verdict |
| --- | ---: | --- |
| Baseline (float32) | 86,799 | — |
| **Fused optimizer update** | **88,819** | **+2.3% — kept, derived from the backend** |
| TF32 float32 matmul | 87,011 | +0.24%, inside run-to-run noise |
| bfloat16 autocast | 80,695 | 7% slower — kept for memory, see below |
| bfloat16 autocast with TF32 | 78,790 | 9% slower |
| Page-locked staging, asynchronous copy | 37,590 and 80,111 | 8% to 57% slower over two runs |
| `torch.compile(dynamic=True)` | per-step parity | 1,098 s of compilation — rejected |
| Host prefetch thread, queue depth 2 | 17.3 ms/step vs 12.9 | 35% slower — rejected |

Run-to-run spread on one configuration is about 0.5% across three repeats,
which is the bar a claim here has to clear.

**The fused optimizer is the only throughput win.** Adam over this model's
parameter list is otherwise dozens of small elementwise kernel launches, and on
a step this short the launches cost more than the arithmetic they carry. It is
derived from the backend rather than configured, because there is one right
answer per backend and a dial with a single correct setting is one nobody
should have to find.

It is worth recording that this looked far larger in isolation. In a bare
training loop — forward, backward, step, nothing else — fusing was worth 17.7%
(12.86 ms to 10.58 ms, five interleaved repeats per arm). Through the real
runner it is worth 2.3%. The difference is the per-step health instrumentation,
which computes a gradient norm across every parameter tensor on every step and
issues its own comparable pile of small launches. **On a launch-bound step, a
run's own telemetry costs the same order as its optimizer.** That is the next
thing to look at, and it is invisible in the wall-clock accounting, because the
monitor's host time is already charged to instrumentation and excluded from the
throughput window while its device launches are not separable there.

**TF32 does nothing here** because the model is not matmul-bound. A `model_dim`
of 64 gives the tensor cores nothing to accelerate; the step is spent on kernel
launches and host work. That diagnosis was right and the inference drawn from
it — that width is what would change it — was not; see the re-read.

**`torch.compile` is not viable at this shape variety.** It reached roughly the
baseline's steady state per step, but paid 1,098 seconds of compilation for 115
steps and hit Dynamo's recompile limit of 8. The reported cause is the
Python-level iteration over ragged `legal_action_ids` in the batch validation
path — the same ragged Python structure that makes batch construction
expensive. Amortizing that compilation at the observed per-step difference
would take millions of steps.

**A prefetch thread makes it worse**, and unstably: 13.9 ms to 23.4 ms across
repeats against a 12.9 ms baseline. Building a batch is Python-object traversal
holding the interpreter lock, so a background thread contends with the training
loop instead of overlapping it. Overlapping that cost needs a worker process or
a loader that emits arrays; a thread cannot do it.

**Page-locked staging lost badly**, and the phase profile above says why.
Pinning helps when a real copy can overlap real device work. Here the expensive
part is constructing the tensor on the host, which pinning does not remove and
in fact duplicates by adding a staging copy — and there is no device work to
overlap with, because the device is idle waiting. The two runs disagreed by a
wide margin, which is itself informative: allocating page-locked buffers for the
loader's varying bucket shapes cannot reuse a cached buffer, so its cost depends
on allocation luck. It was never faster in either run.

**One suspected cost turned out not to be one.** The masked loss reads a device
tensor in boolean context on every micro-batch, to reject an empty loss mask,
which is exactly the pattern this project otherwise forbids in hot paths.
Removing it changes nothing measurable: 12.00 ms with the guard, 12.55 ms
without, against a 12.55 ms repeat of the original arm. The device is idle
waiting for the host, so a synchronization has nothing to wait for. The guard
was left in place here. It has since gone, for a reason that has nothing to do
with its cost: the loss stopped gathering its enabled rows by boolean index, and
what the guard protected against is now a non-finite loss the run already
rejects.

**The synchronization probe measured −0.11 ms per step**, meaning the arm that
synchronizes on every micro-batch was very slightly *faster* than the deferred
one. That is noise around zero, and it says the deferred read-back buys nothing
on this workload — for the opposite reason it buys nothing at four accumulation
micro-batches on MPS. There the device is busy enough that the host never gets
ahead; here the host is the bottleneck, so it never gets ahead either.

## What Was Kept

`precision = "bfloat16-mixed"` stays available and stays off by default. Its
throughput cost is real and measured, but throughput is not what it is for:

| Precision | Throughput | Peak reserved device memory | Parameter bytes |
| --- | ---: | ---: | ---: |
| float32 | 86,799 | 226.5 MB | 23.4 MB |
| bfloat16-mixed | 80,695 | 174.1 MB | 23.4 MB |

Activations are held at half width, so reserved memory falls 23% while the
master weights and optimizer state stay float32 and byte-identical. Activation
memory is what decides whether a larger model fits on a fixed card, which is the
question the next stage of this milestone asks. Trading 7% throughput for 23%
of the memory ceiling is a trade worth being able to make; making it by default,
at a model size where memory is not the constraint, is not.

`memory-envelope.md` answers that next question and the trade holds: the saving
runs 19–39% once width and batch grow, and there are configurations reachable
only with the dial on. It also revises what "activation memory" means here, by
locating the cost in the action head rather than in the transformer.

Strict determinism is also available on CUDA and is what the smoke selection
uses. Every operation this model's backward pass needs has a deterministic CUDA
implementation in the locked build, and a repeated run reaches bit-identical
gradients. That is a genuine gain over MPS, where the correctness path is
unavailable.

## The Embedding Backward, Re-Measured

A kernel-level profile taken alongside the phase profile above, and reported in
`#202` rather than here, put `embedding_dense_backward` at 0.642 ms per step and
14.3% of device busy time at `model_dim` 64 — larger than any matmul in the
model. Re-measuring it across width and batch retires it, and corrects what it
was attributed to.

These readings come from a standalone harness rather than from `anthro train`:
one real batch off the pinned corpus loader, held fixed, driven through forward,
backward, and a fused Adam step under `torch.profiler`, summing self device time
per operation over 30 steps after 10 warmup steps. Only the model is real; the
runner is not. That is deliberate, and it cuts one way: the denominator excludes
the runner's own per-step telemetry, and the gradient-norm monitor issues device
work proportional to the parameter list, so a real run's denominator is larger
at every width and larger still at the wide end. **Every share below is
therefore an upper bound.** The harness reproduces the original reading closely
enough to be read against it — 0.620 ms at batch 16 by 192 plies, against the
0.642 ms above.

### It is four tables, and not mostly the one it was blamed on

The cost was attributed to the piece table: one lookup per square, so 196,608
gradients scattering back into 13 rows. That contention is real, and it is not
the largest part of the line. At batch 16 by 192 plies and `model_dim` 64:

| Table | Rows | Indices | Calls | Device time |
| --- | ---: | ---: | ---: | ---: |
| `side`, `castling`, `en_passant` | 2, 16, 65 | 3,072 each | 3 | 0.361 ms |
| `piece` | 13 | 196,608 | 1 | 0.205 ms |
| `previous_action` | 1,971 | 3,072 | 1 | 0.055 ms |

The three rule-state tables cost 1.8× the piece table while carrying a
sixty-fourth of its indices. The last two rows are the controlled comparison:
the same 3,072 indices, and the table with 1,971 rows costs less than half of
what each 2-to-65-row table costs. **Row count, not index count, is what sets
this price.** The piece table is expensive because it is narrow, and it survives
being indexed sixty-four times as often for only 3.7× the cost.

### The share decays on width, and the tables swap places on batch

Left column is the batch this project trains today; right is the batch the
device work above argues for. Percentages are of device busy time per step.

| `model_dim` | batch 16 × 192: piece / all | batch 256 × 96: piece / all |
| ---: | ---: | ---: |
| 64 | 10.3% / 31.1% | — |
| 256 | 6.4% / 19.3% | — |
| 512 | 3.3% / 10.1% | 3.0% / 3.4% |
| 1024 | 1.5% / 4.1% | 1.2% / 1.4% |
| 2048 | — | 0.4% / 0.5% |

Width is what retires it. The operation is very nearly constant in absolute
terms — the piece table costs 0.19 ms at width 64 and 0.25 ms at 1024, because
it scales with the board encoding rather than with the model — while everything
around it grows superlinearly.

Batch does something different and worth recording: it inverts which tables
matter. Widening the batch gives the rule-state tables enough indices to fill
the device, and their combined cost falls from 0.361 ms to 0.131 ms even as
their index count grows eightfold; the piece table's cost rises roughly with its
own. At batch 256 the piece table is 88% of the line it was originally blamed
for only a third of.

Three repeats at width 1024 and batch 256 returned 1.23%, 1.24%, and 1.23%,
which is far inside the 0.5% run-to-run bar this document holds elsewhere.

### Verdict

**Not worth changing the board encoder for.** At width 1024 and batch 256,
removing the piece embedding's backward pass outright — not making it cheaper,
deleting it — would return at most 1.2% of a step, and 0.4% at width 2048. Both
are upper bounds, for the denominator reason above.

The candidate change was a fused per-square-per-piece table — one 832-row lookup
replacing `piece_embedding` and its slice of the projection. It is worth being
clear that this is **not** a simplification, because it reads like one. The
current form factors each square's 13 vectors through a shared
`piece_embedding_dim`-wide table, and at the default 8 that is a rank-8
bottleneck on a 13-way choice; removing it makes the encoder strictly more
expressive and costs 1.62× the parameters on that path at every width. What
looks like collapsing two operations into one is a capacity increase wearing a
performance argument, and `roadmap.md` puts capacity behind a demonstrated
plateau of the current architecture. It would also change the model's function,
so it costs a model identity bump and every checkpoint built against the old
one. None of that is a trade worth 1%.

The 14.3% headline was true and was never evidence the operation matters: it was
measured at the smallest width and batch this project runs, against a
denominator that both inflate. **A share measured at `model_dim` 64 is a
statement about `model_dim` 64.**

## The Table Re-Read Against A Device-Bound Step

Everything above was measured on a step where the host spent 4.8 ms building a
batch and the card sat at roughly 11% utilization. The loader work removed that
cost, so every verdict above was re-taken. **Most of them changed, and the two
that changed most changed sign.**

The arms below run through the same `anthro train` command on one CUDA device
with the eager loader. Each was implemented as a patch applied at process start
and discarded afterwards, so nothing here except the two settings named at the
end reached the codebase.

They run against a smaller selection than the table above — 8,000 games rather
than 30,000, prepared the same way:

```console
uv run anthro data prepare \
  --config configs/data/lichess-blitz-2017-04.toml \
  --set 'artifact_name="lichess-blitz-m200-8k"' \
  --set filters.maximum_games=8000 \
  --set output.games_per_shard=8000
```

The eager loader materializes the whole selection before the first step, so
that is 23 s of startup per arm instead of 85 s, and 77 arms were run. It costs
nothing the arms measure: every batch has the same shape distribution, and the
`model_dim` 64 batch 16 baseline reads 157,361 positions per second here
against 155,143–163,905 over three runs on the 30,000-game selection.

### Read this as workloads, not as widths

The original table's workload was `model_dim` 64 at batch 16. The obvious
correction is to re-read it wide. That is half the correction, and the smaller
half. Holding batch at 16 and moving only width:

| `model_dim` | data | transfer | forward and backward | optimizer | compute share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.60 ms | 0.72 ms | 4.45 ms | 0.22 ms | 63.7% |
| 256 | 1.74 ms | 1.14 ms | 4.75 ms | 0.25 ms | 60.2% |
| 512 | 1.56 ms | 0.71 ms | 4.35 ms | 0.36 ms | 62.4% |
| 1024 | 1.54 ms | 0.72 ms | 6.86 ms | 0.85 ms | 68.8% |

**Compute is flat from width 64 to 512.** Sixty-four times the matmul work in
the transformer for no additional time, because at batch 16 the step is issuing
kernels rather than doing arithmetic, and a wider kernel that is not full costs
what an empty one costs. Only at 1024 does it begin to move.

So width alone does not take this step out of launch-bound territory, and a
re-read at `model_dim` 512 and batch 16 would have repeated the original
mistake in a new place. What fills the kernels is **batch**. Three workloads
are therefore reported, and the third is the one the arms turn on:

| Workload | baseline | step | what it is |
| --- | ---: | ---: | --- |
| 64 × 16 | 157,361 pos/s | 6.93 ms | the original table's workload |
| 1024 × 16 | 136,366 pos/s | 7.99 ms | wide, still launch-bound |
| 1024 × 256 | 204,675 pos/s | 65.69 ms | wide and device-bound, 71.8% compute |

The first line is also the loader result in isolation: the same configuration
the original table measured at 86,799 positions per second now runs at 157,361,
so batch construction alone was holding back 81% of this workload.

### The arms

Median of three paired rounds, each arm compared against a baseline run beside
it rather than against a pooled mean, because this host is shared and drifts.
Baseline spread was 0.29% at 64 × 16 and 1.42% at 1024 × 256; at 1024 × 16 it
was 10%, which is why that column carries no verdict of its own.

| Candidate | Old verdict | 64 × 16 | 1024 × 16 | 1024 × 256 |
| --- | --- | ---: | ---: | ---: |
| bfloat16 autocast | 7% slower | −15.4% | −0.2% | **+70.4%** |
| bfloat16 with TF32 | 9% slower | −11.3% | — | **+73.0%** |
| TF32 float32 matmul | +0.24%, noise | −0.2% | +2.9% | **+21.1%** |
| Fused optimizer update | +2.3% | **+33.4%** | — | +1.0% |
| Page-locked staging | 8–57% slower | −3.1% | — | −8.3% |
| Host prefetch thread | 35% slower | −30.8% | — | −5.4% |
| `torch.compile(dynamic=True)` | rejected | −69.6% | — | −41.7% |

The fused optimizer is quoted the way the original table quoted it, as an
unfused baseline raised by turning it on. The other rows are the arm against
the shipped default, so a negative number is an arm that lost.

**bfloat16 is the reversal, and it is a batch effect rather than a width one.**
The middle column is what says so: at `model_dim` 1024 and batch 16 it is worth
nothing, and the same model at batch 256 is worth 70%. Half precision has an
arithmetic advantage to express only once there is arithmetic to do. It also
returns 37% of reserved memory at that workload, 10.7 GiB to 6.7 GiB, so the
memory argument the dial was originally kept for survives intact beside a
throughput argument that now points the same way.

**TF32 is the second reversal and the smaller one.** Noise at both batch-16
workloads, +21.1% at batch 256. It was removed on the strength of a reading
taken where tensor cores had nothing to accelerate. It is re-added by this
change, off by default.

**The fused optimizer inverts.** It was the only win in the original table at
+2.3%; on the same workload with the host cost removed it is worth 33.4%,
because the launches it saves are now a visible share of a step rather than
hidden behind batch construction. At batch 256 it is worth 1.0%, inside that
arm's own spread. Nothing here argues for a dial — the fused form is never
slower — but it does retire the claim that it is this backend's main prize.

**Page-locked staging is no longer catastrophic and still never wins.** The
original 8–57% loss was dominated by allocating page-locked buffers for varying
bucket shapes; against an array-emitting loader it costs 3.1% and 8.3%. Still
rejected, now for an ordinary reason rather than a pathological one.

**The prefetch thread still loses**, and the original diagnosis no longer
explains it. GIL contention over Python-object traversal was the stated cause,
and that traversal is gone; a background thread still costs 30.8% at 64 × 16.
Overlapping the loader needs a worker process, which the shard-backed loader
already is.

**`torch.compile` fails for a different reason than it used to.** It still hits
Dynamo's recompile limit of 8, but the ragged `legal_action_ids` iteration that
was blamed has been removed, and the limit is now reached through
`MoveModelBatch.position_bound`:

```
return max(self.chunk_start_plies) + self.action_targets.shape[1]
```

A Python-level `max` over a per-batch tuple. Same failure mode, next ragged
structure in line. The verdict is unchanged and the follow-up is `#275`.

### What the instrumentation costs

The synchronization probe was expected to be dead weight here. It is not.

| Workload | step | added by synchronizing per micro-batch |
| --- | ---: | ---: |
| 64 × 16 | 6.93 ms | +0.05 ms (0.8%) |
| 1024 × 16 | 7.99 ms | +0.85 ms (10.7%) |
| 1024 × 256 | 65.69 ms | +14.69 ms (22.4%) |

The original −0.11 ms was correct and was a statement about a host-bound step:
the device was idle waiting, so a synchronization had nothing to wait for. Once
the device is busy, the loader's own time overlaps device work in the deferred
arm, and synchronizing every step serializes them. **The probe is what keeps that
22.4% from being invisible**, so it stays.

Keeping it is not free, and the figure above is not what it costs. That column
is the difference between the two arms per step; what a run pays is that
difference over the steps actually spent in the slower arm, which the default
cadence puts at one interval in four:

| Workload | probe arm costs | of a run of | share |
| --- | ---: | ---: | ---: |
| 64 × 16 | 0.00 s | 9.2 s | 0.03% |
| 1024 × 256 | 1.76 s | 38.6 s | 4.6% |

So the wide run spends 4.6% of itself measuring a 22.4% effect, against the
0.6% the health monitor costs the same run. That is the honest comparison and
it is the one the keep decision rests on: the probe is the more expensive of
the two instruments this section judged, and it is kept because what it
measures moves by a factor of thirty across workloads while the health
monitor's cost does not move at all. If 4.6% is judged too much, the lever is
`synchronization_probe_every_intervals` rather than the probe's existence —
the cost falls in proportion to the share of intervals in the slower arm, and
nothing about the reading needs one interval in four rather than one in eight.
Changing that default is not done here because no reading argues for a
particular value.

The per-step health monitor was measured the same way, and the wall-clock
figure is the honest one: its host time is subtracted from the throughput
window, so an arm with the monitor disabled has nothing to subtract and reads
as slower through the window metric. Against total run time it costs 4.0% at
64 × 16 and 0.6% at 1024 × 256.

Most of that was one avoidable thing. `observe_gradients` computed a norm per
parameter tensor in a list comprehension — 47 tensors, so about 49 launches on
a step that issues roughly 665 — where `torch.nn.utils.get_total_norm` does the
same reduction through `torch._foreach_norm`. Through the real runner that is
0.80 ms to 0.46 ms of host time per step, against 0.29 ms predicted from
isolation, and 0.89% of a wide run's wall clock. The change is made here.

### What changed in the codebase

Two settings and one simplification, which is the whole of what these readings
justify:

- `matmul_precision` is re-added, `highest` by default and `high` for TF32. It
  is a declared setting that decides the arithmetic every gradient is computed
  in, so unlike the fused optimizer it is one a continuation has to match.
- `precision` keeps its default and gains a throughput argument at batch 256
  that `docs/training-and-runtime.md` previously did not have.
- The gradient and update norms go through `get_total_norm`.

Nothing was deleted. The probe earns its size at the workload this project is
moving toward, and the health monitor earns its size once its norms are cheap.

### What this re-read does not settle

**The corpus-scale loader is still the constraint, and this table cannot see
it.** These arms use the eager loader, where the corpus is resident and the
step is 71.8% compute at 1024 × 256. Under the shard-backed loader on the
million-game selection the same workload is 22.7% compute, because the parent
waits ~88 ms per batch for decoded shards, and raising workers from 8 to 24
moved that by less than 20%. Batch construction is fixed; the phase in front of
it is not, and no device-side setting in this table can be read against a step
that is waiting on data. That is `#276` rather than anything here.

**Which default is right is not answered here either.** A 70% throughput
result is a reason to re-examine the precision default at the batch capacity
selection lands on, not a reason to change it now: none of these runs was long
enough to say what bfloat16 does to the loss curve, and that comparison belongs
with `#54`.

## The Encoder Wrapper, Replaced By Explicit Blocks

`nn.TransformerEncoder` and `nn.TransformerEncoderLayer` were replaced by an
explicit block over `F.scaled_dot_product_attention(is_causal=True)`. The
reason it was proposed does not survive measurement here, and a different one
does.

**The compile argument did not reproduce.** `#203` reported the wrapper failing
`fullgraph` with `Could not guard on data-dependent expression Eq(u0, 1)`. On
this host and Torch build, the wrapper reports one graph and zero breaks, and
compiles under `fullgraph=True` — both as the model called it, `mask=` together
with `is_causal=True`, and with `mask=` alone. Whatever produced that failure is
not reproducible from this checkout, so the change does not rest on it.

**What it does rest on is backend selection.** Flash and cuDNN attention refuse
a non-null `attn_mask`, which asking `sdpa_kernel([SDPBackend.FLASH_ATTENTION])`
for each form confirms directly: the additive mask raises `No available kernel`
and `is_causal=True` is accepted. Handing attention a mask costs the two fastest
kernels, whatever else the mask is doing.

Forward, backward, and a fused Adam step on one 4090, one fixed batch held
across arms, 40 steps after 12 warmup, five paired rounds, median reported.
Worst run-to-run spread in any cell was 1.7%.

| `model_dim` × batch × plies | wrapper | explicit blocks | delta |
| --- | ---: | ---: | ---: |
| 64 × 16 × 192 | 4.190 ms | 3.561 ms | 15.0% |
| 512 × 16 × 192 | 5.617 ms | 5.271 ms | 6.2% |
| 512 × 256 × 96 | 41.373 ms | 34.684 ms | 16.2% |
| 1024 × 256 × 96 | 102.995 ms | 90.629 ms | 12.0% |

A repeat taken while the card was shared reproduced the three larger workloads
at 6.2%, 16.3%, and 12.3%. Its smallest-workload row carried 20% to 25%
run-to-run spread and is not readable, which is the same fragility the re-read
above records at batch 16.

Under `torch.compile` the gap narrows to 3.2–8.2%, because inductor fuses away
some of what the wrapper paid in Python. `reduce-overhead` behaves the same as
the default mode at every workload but the smallest, where it is worth 17.8%.
None of this is reachable by a real training step yet: `chunk_start_plies` still
specializes every compiled graph, which is `#275`.

**One finding is about the model rather than the clock.**
`nn.TransformerEncoder` builds its stack with `copy.deepcopy` of one prototype
layer, so every layer of the two-layer baseline began training identical to the
others. Explicit construction draws each block. That makes the replacement a
change to what a run learns rather than a refactor, so it was read against a
control arm; `#203` holds the reading.

## What This Does Not Show

- **Nothing about quality.** No run here was long enough to move a held-out
  metric, and none was scored against the frozen pool.
- **Nothing about evaluation cost.** Inference and the benchmarks select CUDA
  through their own device boundary, landed separately, and none of the figures
  here describe what a benchmark sweep costs on this host.
- **Nothing about the second card**, or about how this scales across both.
- **Nothing about a larger model**, with two exceptions, both since measured.
  The embedding backward was swept across width and batch above. The whole
  optimization table was re-read at width and batch in the section above, which
  is where TF32 and mixed precision stopped being noise; the expectation that
  they would start mattering with capacity was right about the direction and
  wrong about the axis, because it is batch rather than width that fills a
  kernel.

## What To Do Next

Everything above converges on one statement: **at this model size the device is
not the constraint, the host is, and every device-side option is therefore
worth almost nothing.** Seven were tried. One returned 2.3%. A 4.8 ms host-side
cost per step, unchanged between backends, sits in front of a 6.2 ms compute
step, and the card idles at roughly 11% for the difference.

That bounds what this milestone's CUDA work could deliver, and it is worth
being plain that the 2.00x figure is a ceiling reached rather than a result
optimized toward. Two things lift it, and neither is a device-side change:

- **Raise the effective batch.** Free today, no code: 3.79x at batch 64 and
  4.76x at 256, using 2.6 GiB of 23.5 GiB. What the batch should actually be is
  a question about the optimization trajectory rather than about throughput, so
  it belongs with capacity selection.
- **Stop building batches out of ragged Python structures.** This is the single
  root cause behind three separate findings here: the 4.8 ms construction cost,
  the recompile storm that made `torch.compile` unusable, and the prefetch
  thread's lock contention. A loader that hands back arrays addresses all
  three, and that is the shard-streaming work rather than anything in this
  change. The loader now does; the construction cost is what was re-measured,
  and neither `torch.compile` nor the prefetch thread was tried again.

A third and smaller one is now visible and was not before: the per-step
gradient norm costs the same order as the optimizer on a launch-bound step.
Measuring what the health monitor costs on the device, rather than only on the
host, would say whether its cadence should become a dial.

Both were done, and the section above records what they found. The batch axis
is where the device-side options went from worthless to decisive, and the
health monitor needed a cheaper reduction rather than a cadence dial.
