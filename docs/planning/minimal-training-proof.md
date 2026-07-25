# Minimal Training Proof

This document records the first reproducible evidence that the Milestone 1
components work together. It is an implementation proof, not a claim that the
selected checkpoint is strong enough for release or that its configuration is
the final production model.

The recorded evidence below was produced under preprocessing version 2, whose
two-way split predates the held-out `test` partition. Reproducing these exact
numbers requires that preprocessing version. Under the current three-way split
the same commands still run, but split assignment differs, so the measured
values are expected to move and are not a regression. The commands, structure,
and interpretation remain accurate; only the specific numbers are pinned to the
older artifact.

## CPU Correctness Gate

The bounded CPU integration test drives the public command surface through
sample preparation, training, validation, checkpoint creation, and exact
resume:

```console
uv run pytest tests/integration/test_training_proof.py
```

The focused model tests inspect the fixed model-boundary batch, confirm padding
and nullable context alignment, prevent future-timestep leakage, and overfit
both a single timestep and a short causal sequence:

```console
uv run pytest tests/models/test_causal.py
```

These tests use the ordinary loader, tensor boundary, model, loss, runner,
validation, and checkpoint APIs. They do not maintain a separate debug
training implementation.

## Many-Game Corpus Slice

The first accelerator proof uses 10,000 accepted games from the pinned Lichess
archive and the same source-order selection implemented by the baseline data
pipeline. The smaller accepted-game bound makes the complete proof repeatable
on a laptop while retaining thousands of held-out human positions. It is a
resolved override of the canonical baseline-corpus configuration, not a second
ingestion path.

Acquire and prepare the offline proof slice directly beneath the shared data
root:

```console
uv run anthro data acquire \
  --config configs/data/lichess-blitz-2017-04.toml

uv run anthro data prepare \
  --config configs/data/lichess-blitz-2017-04.toml \
  --set 'artifact_name="lichess-blitz-proof"' \
  --set filters.maximum_games=10000 \
  --set output.games_per_shard=10000
```

After preparation, the proof needs no network access.

## MPS Baseline

The checked-in baseline starts with the current default model scale rather than
assuming the tiny smoke model is useful. It trains full-game, length-bucketed
batches with an explicit MPS device and relaxed determinism:

```console
uv run anthro train \
  --config configs/training/lichess-blitz-baseline.toml
```

The command prints raw move loss, legally masked move loss, and the
uniform-over-legal baseline after validation. Learned move preference beyond
random legal selection is demonstrated when:

```text
legal_move_loss < uniform_over_legal
```

Raw move loss and mask penalty remain important diagnostics because runtime
masking should be a guardrail rather than the only source of chess structure.
The legally masked comparison isolates move preference among the actions that
exact chess logic permits.

Resume uses the same command surface and preserves the exact loader cursor,
optimizer state, global step, and compatibility identities:

```console
uv run anthro train \
  --config configs/training/lichess-blitz-baseline.toml \
  --set 'resume_from="latest"' \
  --set steps=2100
```

## First Measured Result

The initial measured run used the checked-in model and training selection on an
Apple-silicon MPS device. Exact resolved configuration, data identities,
execution settings, checkpoints, per-step throughput, and sampled memory remain
in the generated run artifacts.

The 10,000 accepted games produced 9,509 training games and 491 frozen
validation games containing 33,866 evaluated positions. The run used 2,000
optimizer steps and processed 2,198,264 training positions. It was resumed from
checkpoints at steps 100, 1,000, and 1,300, which also exercised exact MPS
checkpoint continuation.

| Measurement | Result |
| --- | ---: |
| Raw held-out move loss | 4.29777 |
| Legally masked held-out move loss | 3.15532 |
| Uniform-over-legal move loss | 3.22612 |
| Raw-logit mask penalty | 1.14245 |
| Raw top-1 illegal rate | 31.84% |
| Peak measured throughput | 1,964.87 positions/second |
| Peak sampled MPS allocated memory | 17.79 MiB |
| Peak sampled MPS driver memory | 1.21 GiB |
| Cumulative optimizer time across resumed segments | 19.17 minutes |

The legally masked loss is lower than uniform-over-legal by 0.07080, providing
the required held-out learned-move signal. The current default model size
cleared the gate, so this proof does not increase capacity or add optimization
features. Stronger model selection belongs after the evaluation harness can
compare changes more broadly.

## Bounded Training Extension

A follow-up run kept the same model, optimizer, batch configuration, and frozen
491-game validation set while increasing the training selection to 30,000
accepted games. It processed 8,716,512 positions over 8,000 optimizer steps in
about 71.8 cumulative optimizer minutes across the initial and resumed
segments.

Held-out loss improved consistently through step 7,000. The runner validates
once, when a run exits, so this curve was assembled from separate stop-and-
resume segments rather than produced by periodic in-loop validation. Retained
checkpoints make the same curve available after the fact at whatever resolution
`checkpoint_every_steps` provides.

| Step | Raw move loss | Legally masked move loss |
| ---: | ---: | ---: |
| 2,000 | 4.24698 | 3.07584 |
| 3,000 | 3.96434 | 2.98826 |
| 4,000 | 3.80679 | 2.94297 |
| 5,000 | 3.69403 | 2.90210 |
| 6,000 | 3.65160 | 2.88657 |
| 7,000 | 3.56481 | 2.84129 |
| 8,000 | 3.57247 | 2.85548 |

Step 7,000 is the best checkpoint from this run on the frozen held-out set.
The small reversal at step 8,000 is the first plateau or variance signal, so
the run stops there rather than assuming more steps are automatically useful.
The result supports further evidence-driven tuning but does not yet distinguish
ordinary validation noise from the beginning of overfitting.

## Opening Sanity Check

As a narrow post-run diagnostic, the 2,000-step checkpoint was evaluated on the
standard starting position with equal 1600 ratings. After exact legal masking,
its leading moves were `e4` at 58.48%, `d4` at 21.31%, `Nf3` at 6.70%, and
`c4` at 4.93%. Greedy continuation began:

```text
1. e4 e5 2. Nf3 Nc6 3. d4 exd4 4. Nxd4 Nxd4 5. Qxd4
```

This confirms recognizable opening preference, but it is not a generated-game
benchmark. Complete-game coherence, sampling behavior, and rating control
remain Milestone 2 runtime and rollout questions.

## Decision-Only Rating Context Replacement

The foundational rating-context change was repeated on the same 10,000-game
corpus and frozen 491-game validation set. The selected replacement encodes
each game once, trains every valid ply, and supplies only the mover's rating to
a nonlinear decision conditioner after the rating-neutral causal transformer.
Historical timestep features contain no rating, and the model no longer needs
a controlled-color or opponent-rating input. This supersedes the earlier
paired-view proof rather than treating its incompatible checkpoint as current.

The MPS replacement run used the baseline model, loader batch, and optimizer
settings. It stopped at step 1,800 and then resumed from `latest` to step 2,000,
preserving the loader cursor, optimizer, random state, and cumulative
processed-position count. The complete run directory is retained as
`decision-conditioned-rating-proof-v3` beneath the shared run root; its
artifacts own the exact resolved paths, configuration, compatibility identities,
and execution provenance.

| Measurement | Result |
| --- | ---: |
| Processed decision positions | 2,198,264 |
| Raw held-out move loss | 4.35907 |
| Legally masked held-out move loss | 3.14189 |
| Uniform-over-legal move loss | 3.22612 |
| Raw-logit mask penalty | 1.21717 |
| Raw top-1 illegal rate | 33.75% |
| Peak measured throughput | 1,320.03 positions/second |
| Peak sampled MPS allocated memory | 20.02 MiB |
| Peak sampled MPS driver memory | 1.25 GiB |

The legally masked loss is lower than uniform-over-legal by 0.08423. This
clears the replacement learned-move gate while recovering supervision from
both sides in one transformer pass and requiring only Anthro's one target
rating at live inference.

## Held-Out Test Partition Baseline

Adding the `test` partition bumped the preprocessing version, so every corpus
prepared before it is incompatible with the current pipeline. The three-way
split assigns `test` the hash range the two-way split gave `validation`, which
means the earlier proofs validated on exactly the games the frozen evaluation
pool now holds, and their replacement validation sets were previously training
data. Both facts require a regenerated corpus and a retrained baseline before
any number can be reported against the pool.

The regenerated selection uses the same checked-in configuration and the same
bounded 30,000-game override as the earlier extension. Split assignment stays a
pure function of the split seed and game id, so the bounded training selection
and the full corpus agree about which games are held out; a build-time overlap
check confirms the pool shares no game with the corpus train split.

The retrained baseline keeps the checked-in model, optimizer, and batch
settings and runs 8,000 optimizer steps on an Apple-silicon MPS device.
Configuration, data identities, and provenance live in the generated run
artifacts.

| Measurement | Result |
| --- | ---: |
| Raw held-out move loss | 3.58922 |
| Legally masked held-out move loss | 2.83796 |
| Uniform-over-legal move loss | 3.21184 |
| Raw-logit mask penalty | 0.75126 |
| Raw top-1 illegal rate | 22.58% |

The legally masked loss is below uniform-over-legal by 0.37388. That margin is
the comparable quantity across runs, because each is measured against its own
validation set's baseline; the raw losses are not comparable to the earlier
proofs, which used a different held-out set.

Retained checkpoints supply the training curve after the fact:

| Step | Raw | Legally masked | Margin below uniform | Mask penalty |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 4.88411 | 3.25141 | -0.03958 | 1.63270 |
| 2,000 | 4.35074 | 3.12945 | 0.08238 | 1.22128 |
| 4,000 | 3.88407 | 2.96320 | 0.24864 | 0.92087 |
| 6,000 | 3.68970 | 2.88130 | 0.33054 | 0.80840 |
| 8,000 | 3.58922 | 2.83796 | 0.37388 | 0.75126 |

Two properties of that curve matter more than the endpoint. The step-1,000
checkpoint is worse than uniform-over-legal, so the retained series spans from
below a trivial baseline to clearly above it and can test whether a benchmark
ranks checkpoints at all. And every metric was still improving at step 8,000,
so this recipe was stopped by the step bound rather than by convergence.

The step-2,000 checkpoint scores within noise of the earlier decision-
conditioned proof despite training on roughly three times the games, which
suggests the model is step-limited rather than data-limited at that point.
