# Agent Guide

This project is `anthro-chess`: a chess model intended to mimic human play,
optional timing, and controllable soft preferences in chess games.

Before making substantive changes, read:

1. `docs/vision.md`
2. `docs/design-principles.md`
3. `docs/architecture.md`
4. `docs/engine-behavior.md`
5. `docs/data.md`
6. `docs/training-and-runtime.md`
7. `docs/interfaces.md`
8. `docs/evaluation.md`
9. Any relevant records in `docs/decisions/`

For preference-control work, also read `docs/preference-controls.md`.

For data, training, evaluation, or preference-control work, read
`docs/research.md` when related outside work or source links are relevant.

The design docs are living documents. Treat them as the current best intent,
not as immutable requirements. If implementation work changes the project
direction, update the affected docs in the same change.

Roadmaps and staged build plans belong under `docs/planning/`. Planning docs
should describe implementation order and tradeoffs without redefining the
project's end state.

## Project Priorities

- Build a usable human-like chess opponent, not a top-strength engine.
- Use deterministic chess logic to construct board state and validate model
  outputs.
- This is a product/build project, not a research project. Prefer direct,
  practical choices over experiments whose main value is answering a research
  question.
- Make behavior controllable through explicit dials such as target rating, time
  settings when enabled, temperature, and optional preference settings.
- Human-like imperfections should emerge from training on human games, not from
  special-case mistake injection.

## Implementation Guidance

- Use exact chess logic for board reconstruction, legal move generation, and
  rule bookkeeping.
- Legal-mask model move outputs before sampling.
- Keep temperature independent from rating and time settings.
- Keep UCI and other outside protocols as runtime interfaces, not model-native
  representations.
- Keep normalized data source-agnostic, compact, and reproducible from pipeline
  scripts.
- Add tests for chess-rule changes, model-facing encodings, data preprocessing,
  and runtime behavior.
- Document major architectural choices in `docs/decisions/`.

## Current Design Posture

The preferred direction is a hybrid symbolic-neural system:

- deterministic chess logic computes board state and legal moves;
- a learned board encoder embeds the exact current board;
- a causal transformer models the game trajectory one ply at a time;
- output heads predict a valid action policy and, when timing is enabled, a
  move-time distribution.
