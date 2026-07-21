# Roadmap

This document is for implementation planning. It describes how the project might
move toward the end state, not what the end state is.

The main project docs describe the intended end state:

- `docs/vision.md`
- `docs/design-principles.md`
- `docs/architecture.md`
- `docs/engine-behavior.md`
- `docs/data.md`
- `docs/training-and-runtime.md`
- `docs/interfaces.md`
- `docs/evaluation.md`

Roadmap items are allowed to change as the project develops. When the roadmap
and the end-state docs disagree, the end-state docs are authoritative unless we
explicitly decide to update them.

## Planning Principles

- Keep roadmap stages broad enough to survive implementation changes.
- Use the roadmap to organize work, not to define product requirements.
- Move detailed task lists into GitHub issues when they become actionable.
- Update the roadmap when the rough order changes.

## Milestone Shape

The roadmap should be organized around proofs of progress. Each milestone
should leave the project in a state that is more usable or more measurable than
before.

### 0. Project Setup

Make the repository ready for implementation work.

This stage should establish the package structure, entry points, test runner,
development tooling, configuration conventions, and issue-tracking workflow. It
should also keep the docs and decisions easy for future agents to follow.

Detailed setup tasks should move into GitHub issues once they are actionable.

### 1. Minimal Training Loop

Prove that the project can train a model correctly from real chess data.

This stage should include small checked-in fixtures, a bounded reproducible
many-game Lichess corpus, deterministic board-state reconstruction,
model-facing encodings, dataloading, a basic model, move-prediction loss,
training configuration, validation metrics, and practical checkpoint/resume
support.

Start with the simplest training target that proves the loop works. Timing data
should be preserved in the data pipeline where available, but Milestone 1 keeps
timing inputs and outputs out of the model and training objective. Timing
behavior is added after the move-only path is useful and measurable.

Before scaling the first run, establish the staged training-correctness
protocol in `docs/training-and-runtime.md`: inspect fixed model inputs, prove a
stripped-down deterministic path can overfit a tiny sample, exercise causal
sequence behavior, and then demonstrate held-out signal above simple
baselines. Add context and optimization features from that trusted baseline
rather than introducing several unverified changes at once.

Prefer full-game, length-bucketed batches for the first proof so training
matches full-history inference while limiting padding waste. Tune
bucket-specific batch sizes, gradient accumulation, and GPU headroom from
measured model memory and throughput rather than hardcoding a speculative
device budget in the data layer. Treat independent midgame chunks as a
scalability option that must be compared with full-history training before it
becomes a default.

The output of this milestone does not need to be a strong chess opponent or
provide the final playable interface. It does need to demonstrate learned move
structure beyond randomness made legal by masking, including held-out
performance above appropriate simple move-selection baselines. That checkpoint
should be credible input to the playable proof rather than only evidence that
the training command runs.

### 2. Playable Proof

Connect model output to an actual playable chess game.

This stage should build the basic inference runtime: exact game-state updates,
legal move generation, illegal-move masking, action sampling, configuration,
and local runtime APIs. It should also connect the runtime to a real chessboard
experience.

UCI is the preferred compatibility path for local chess GUIs. The first UCI
implementation only needs enough polish to play games reliably, keep stdout
reserved for protocol messages, and expose the core options that already exist.

The goal is a proof that the model can choose legal moves in a real game loop
and begin to play somewhat sensible chess. It is acceptable for the model to be
weak at this stage.

### 3. Evaluation Harness

Turn evaluation into a first-class project system early.

Basic validation metrics should exist during the minimal training loop, but
this stage should make evaluation coherent enough to compare model versions
without relying on subjective playtesting.

The initial harness should emphasize reusable benchmarks that can run against
future models:

- held-out move prediction metrics;
- rating-sliced validation metrics;
- illegal-move probability and legal-mask diagnostics;
- controlled training-efficiency comparisons using active positions per second,
  memory, and quality versus processed positions and wall-clock time;
- batch-one end-to-end move latency and declared-batch inference throughput,
  with cold-start time reported separately;
- generated-game rollout checks once the runtime can play;
- fixed engine-anchor matches for relative rating monotonicity;
- timing diagnostics once timing is enabled.

Human-likeness evaluation beyond simple distribution metrics belongs later in
the process. A compact human-vs-engine classifier can be useful once the model
can generate coherent games, but it should not block the basic bot or become a
separate anti-cheat project.

### 4. Scale And Improve

Use the working loop and evaluation harness to improve the bot.

This stage should expand data scale, tune sampling and weighting, improve model
capacity, calibrate rating behavior, strengthen checkpointing and reproducible
runs, improve runtime reliability, and add optional timing behavior when the
move-only path is already useful.

Corpus-scale training should replace fixture-oriented eager per-ply
materialization with bounded-memory shard-backed loading before attempting full
passes over the prepared million-game selection.

Evaluation should guide this work. New data, model changes, and runtime changes
should be judged by the same benchmark surfaces whenever possible, with deeper
diagnostics available when a regression or surprising result appears.

### 5. Late Controllability

Add learned preference controls after the model, runtime, and evaluation stack
are clearly working.

Preference controls for openings, style, timing style, player-inspired
tendencies, and other human-play concepts are important to the end state, but
they should not block the basic bot. This work should start once there is a
model whose behavior can be measured and played.

Possible late-stage preference-control work includes:

- deriving multi-label position metadata for opening families, structures,
  aggression, sacrifices, development, and other concepts;
- labeling positions or conservative position windows rather than blindly
  labeling whole games;
- using known-opening position matching and walking backward to the nearest
  useful opening or structure category;
- building matched contrast sets by rating, color, move number, and other
  relevant context;
- inspecting or steering model activations from those labeled sets;
- calibrating preference sliders so rating, legality, and position coherence
  remain intact.

Suggested order for the first learned preference controls:

1. Define a small opening-family taxonomy.
2. Build known-opening position matching and backward lookup.
3. Generate conservative per-ply opening-family labels.
4. Train or extract activation-difference steering directions for a few
   opening families.
5. Evaluate whether sliders increase the intended opening tendency without
   breaking rating calibration.
6. Add a few non-opening labels, starting with easy structural concepts.
7. Compare activation-difference steering against supervised vectors and sparse
   autoencoder features.
8. Add application sliders once the controls are stable enough to be useful.

See `docs/preference-controls.md` for the preference-control subsystem design.
See `docs/evaluation.md` for the evaluation design.
