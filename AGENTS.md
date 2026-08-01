# Agent Guide

This project is `anthro-chess`: a chess model intended to mimic human play,
optional timing, and controllable soft preferences in chess games.

Before making substantive changes, read `docs/documentation.md`, the topic docs
relevant to the requested area, and any relevant records in `docs/decisions/`.
The topic docs are:

- `docs/vision.md`
- `docs/design-principles.md`
- `docs/architecture.md`
- `docs/engine-behavior.md`
- `docs/data.md`
- `docs/training-and-runtime.md`
- `docs/interfaces.md`
- `docs/evaluation.md`

For preference-control work, also read `docs/preference-controls.md`.

For data, training, evaluation, or preference-control work, read
`docs/research.md` when related outside work or source links are relevant.

When starting implementation work, infer the relevant docs from the requested
area instead of requiring the user to repeat boilerplate. Inspect the repo,
read the matching topic docs and decision records, choose a scoped first slice
when the request is broad, and keep the user informed about the approach.

Use GitHub issues as the default implementation task tracker when available.
The user should be able to provide an issue link, ask for the next appropriate
issue, or ask to create or refine issues for a milestone without repeating
process details. When setting up milestone work, use real GitHub milestones,
tracker issues, sub-issues, issue dependencies for true blockers, and the
repo's `area:`, `type:`, `execution:`, and `verification:` labels as described
in `docs/issue-workflow.md`. When working from an issue, inspect linked docs,
decisions, parent/sub-issues, dependency state, labels, and milestone context;
keep the issue updated with relevant findings, follow-ups, and completion
status; and keep roadmap changes under `docs/planning/`.

When choosing the next issue, select only a `type: task` issue in the active
milestone that is attached to its tracker and has no open blockers. Treat any
issue missing that metadata as intake awaiting maintainer triage. An issue
labeled `execution: gpu-required` needs the GPU environment its body specifies.
One labeled `verification: gpu-required` can be implemented without that
environment and uses the documented handoff when the GPU check remains.

A request to work on an issue is authorization to carry it through to a
ready-for-review pull request that closes the issue on merge. The pull request
is the maintainer's review boundary and is not merged by the agent. Several
sessions often run against this repository at once, so the issue is claimed in
GitHub while it is being worked on.

When a change alters what a chess GUI observes, offer a real GUI check without
being asked: point the maintainer's GUI at the working checkout with
`scripts/anthro-gui-target .` and say the engine is ready to test, what to look
at, and what a good result looks like. The GUI itself is configured once and is
never reconfigured per issue. See `docs/issue-workflow.md` for when this applies
and `docs/playable-uci.md` for the mechanism.

When a change adds or alters a benchmark, take a shakedown reading on two real
checkpoints from one training run before the pull request is ready, using a
reduced view and `--no-record`, and report what was expected against what was
read. Fixtures cannot show that a benchmark measures anything. See
`docs/issue-workflow.md` for when this applies and `docs/evaluation.md` for
what the reading is and is not.

Corpora and training runs live outside the worktree, beneath
`ANTHRO_CHESS_DATA_ROOT` and `ANTHRO_CHESS_RUN_ROOT`. Check those roots before
concluding that this machine has no corpus or no checkpoints; an empty worktree
is not evidence, because unset roots and a machine with no artifacts look
identical from inside the repository.

Local setup, dependency installation, the artifact roots, and the verification
commands are in `CONTRIBUTING.md`.

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
