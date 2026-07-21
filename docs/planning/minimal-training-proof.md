# Minimal Training Proof

This document records the first reproducible evidence that the Milestone 1
components work together. It is an implementation proof, not a claim that the
selected checkpoint is strong enough for release or that its configuration is
the final production model.

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

Held-out loss improved consistently through step 7,000:

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

## Controlled-Player Context Replacement

The foundational context change for the playable proof was repeated on the
same 10,000-game corpus and frozen 491-game validation set. The replacement
encoding creates deterministic white and black trajectory views, keeps both
players' observed moves in each view, broadcasts only the controlled player's
rating and color, and applies action loss only on that player's turns. The
model and encoding compatibility versions changed, so the earlier checkpoints
remain retained evidence but cannot be loaded as though they used the new
contract.

The MPS replacement run used the baseline model and optimizer settings with a
larger sequence batch to keep its active-position budget comparable after
splitting each game into controlled-player views. It stopped at step 1,800 and
then resumed from `latest` to step 2,000, preserving the loader cursor,
optimizer, random state, and cumulative processed-position count. The complete
run directory is retained as `controlled-player-proof-v2` beneath the shared
run root; its artifacts own the exact resolved paths, configuration,
compatibility identities, and execution provenance.

| Measurement | Result |
| --- | ---: |
| Processed controlled-player positions | 2,188,528 |
| Raw held-out move loss | 4.31526 |
| Legally masked held-out move loss | 3.10874 |
| Uniform-over-legal move loss | 3.22612 |
| Raw-logit mask penalty | 1.20653 |
| Raw top-1 illegal rate | 33.60% |
| Peak measured throughput | 1,302.85 positions/second |
| Peak sampled MPS allocated memory | 21.62 MiB |
| Peak sampled MPS driver memory | 1.21 GiB |

The legally masked loss is lower than uniform-over-legal by 0.11738. This
clears the replacement learned-move gate while preserving one target rating for
the controlled player and requiring no opponent rating at live inference.
