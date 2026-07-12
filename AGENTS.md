# Agent Guide

This project is `anthro-chess`: a chess model intended to mimic human play,
optional timing, and controllable soft preferences in chess games.

Before making substantive changes, read:

1. `docs/documentation.md`
2. `docs/vision.md`
3. `docs/design-principles.md`
4. `docs/architecture.md`
5. `docs/engine-behavior.md`
6. `docs/data.md`
7. `docs/training-and-runtime.md`
8. `docs/interfaces.md`
9. `docs/evaluation.md`
10. Any relevant records in `docs/decisions/`

For preference-control work, also read `docs/preference-controls.md`.

For data, training, evaluation, or preference-control work, read
`docs/research.md` when related outside work or source links are relevant.

When starting implementation work, infer the relevant docs from the requested
area instead of requiring the user to repeat boilerplate. Inspect the repo,
read the matching topic docs and decision records, choose a scoped first slice
when the request is broad, and keep the user informed about the approach.

Use GitHub issues as the default implementation task tracker when available.
The user should be able to provide an issue link, ask for the next appropriate
issue, or ask to refine the issue roadmap without repeating process details.
When working from an issue, inspect linked docs and decisions, keep the issue
updated with relevant findings or completion status, and keep roadmap changes
under `docs/planning/`.

The design docs are living documents. Treat them as the current best intent,
not as immutable requirements. If implementation work changes the project
direction, update the affected docs in the same change.

Roadmaps and staged build plans belong under `docs/planning/`. Planning docs
should describe implementation order and tradeoffs without redefining the
project's end state.

Docs should explain project shape, rationale, and where source-of-truth details
live. Do not duplicate exact names, thresholds, defaults, schema fields, or
config values in prose when code, schemas, or config files should own them.

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
- Document major architectural choices in `docs/decisions/` when the rationale
  has lasting value.

## Current Design Posture

The preferred direction is a hybrid symbolic-neural system:

- deterministic chess logic computes board state and legal moves;
- a learned board encoder embeds the exact current board;
- a causal transformer models the game trajectory one ply at a time;
- the action head predicts a valid action policy and, when timing is enabled,
  the time head predicts a move-time distribution conditioned on the selected
  action.
