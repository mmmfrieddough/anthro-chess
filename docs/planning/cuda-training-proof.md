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
of this document asked for. What that change did not touch is the decode
upstream of it, and under the shard-backed loader that is now what a step waits
on.

Separating optimizer work from forward and backward was worth doing, and not
for the reason expected: at 4.9% of a step it looked like the last place worth
touching, and it turned out to hold the only throughput win found here.

## Optimizations Measured

Each was implemented and measured on the workload above. One was kept, one was
kept for a different reason than it was tried for, and the rest were removed:
retaining a dial that never wins costs more than the measurement it preserves,
and the readings below are what preserve it instead.

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
launches and host work.

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

## What This Does Not Show

- **Nothing about quality.** No run here was long enough to move a held-out
  metric, and none was scored against the frozen pool.
- **Nothing about evaluation cost.** Inference and the benchmarks select CUDA
  through their own device boundary, landed separately, and none of the figures
  here describe what a benchmark sweep costs on this host.
- **Nothing about the second card**, or about how this scales across both.
- **Nothing about a larger model.** Every conclusion above is bound to a
  `model_dim` of 64. TF32 and mixed precision are exactly the settings expected
  to start mattering as capacity grows, which is why the precision dial was
  kept; the correct move is to re-measure at the new size rather than to assume
  either result carries.

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
