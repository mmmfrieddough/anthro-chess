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
- Move detailed task lists into issues, pull requests, or separate planning
  notes when they become actionable.
- Update the roadmap when the rough order changes.

## Broad Stages

### 1. Foundations

Establish the repository structure, development tooling, project conventions,
and core chess/data representations.

### 2. Data And Chess Logic

Build the deterministic chess-state layer and the pipeline for turning human
games into model-ready examples.

Start with a small reproducible Lichess ingestion path, then scale to larger
normalized shards once parsing, validation, and sampling recipes are stable.
Keep filtering, weighting, and storage choices aligned with `docs/data.md`.

Begin the test and evaluation foundation here: chess-rule tests, encoding
tests, preprocessing checks, and legal-mask evaluation on held-out positions.

### 3. Model Training

Train models that predict both move choice and move timing from human-game
data. Track training-time and validation metrics early, including move loss,
optional timing loss, illegal-move mask penalty, and rating-sliced validation
metrics.

### 4. Playable Runtime

Connect the model to a runtime that can maintain game state, choose legal moves,
and optionally play with clocks. Add simulated rollout benchmarks for generated
games, timing behavior, and early rating calibration once the runtime can play
coherent games.

Once UCI interaction is stable, add fixed engine-anchor matches as secondary
rating diagnostics. These should compare Anthro rating settings against fixed
engine configurations for monotonicity and regression tracking, not as absolute
human-rating calibration.

Add a direct UCI executable around the runtime for local chess GUI
compatibility. It should run as a normal engine process, load the model itself,
advertise supported options, keep UCI stdout clean, and rely on config defaults
plus `setoption` overrides.

### 5. Iteration

Tune controls, improve data and model quality, tune optional timing behavior,
expand benchmark coverage, and keep the end-state docs aligned with what the
project becomes.

Learned preference controls for openings, style, and other human-play concepts
belong late in the process, after a model and runtime are clearly working. This
work should not block the basic bot.

Possible late-stage work includes:

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

Human-likeness evaluation beyond simple distribution metrics belongs later in
the process. A compact human-vs-engine classifier can be useful once the model
can generate coherent games, but it should not block the basic bot or become a
separate anti-cheat project.

Suggested late-stage order:

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
