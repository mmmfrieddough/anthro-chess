# Roadmap

This document is for implementation planning. It describes how the project might
move toward the end state, not what the end state is.

The main project docs describe the intended end state:

- `docs/vision.md`
- `docs/design-principles.md`
- `docs/architecture.md`
- `docs/engine-behavior.md`
- `docs/training-and-runtime.md`

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

### 3. Model Training

Train models that predict both move choice and move timing from human-game
data.

### 4. Playable Runtime

Connect the model to a runtime that can maintain game state, choose legal moves,
and optionally play with clocks.

### 5. Iteration

Tune controls, improve data and model quality, tune optional timing behavior,
and keep the end-state docs aligned with what the project becomes.
