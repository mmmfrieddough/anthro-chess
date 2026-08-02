# Agent Guide

## Commands

Nothing this project installs is on `PATH`. Every command takes the `uv run`
prefix, including `anthro` and `anthro-uci`.

```console agent-commands
uv sync --locked
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

The suite shards across the machine's cores by default; pass `-n0` to attach a
debugger and `-m "not gpu"` to skip GPU tests. Continuous integration runs those
same commands and then `uv build`, an install of the built wheel into a clean
environment, and a smoke test of the installed package — so a green local run is
not yet a green `Required CI`.

`CONTRIBUTING.md` has first-time setup, the artifact roots, and the commands
that rebuild the data side from the pinned sources.

## Where to look

Read `docs/documentation.md`, the rows below that match the work, and the
matching records in `docs/decisions/` (0001-0028).

| Open when | File |
| --- | --- |
| unsure where something belongs, or adding a doc | `docs/documentation.md` |
| the change could widen what the product claims to do | `docs/vision.md` |
| choosing between implementation approaches | `docs/design-principles.md` |
| adding or moving a module | `docs/architecture.md` |
| the change alters what a player observes | `docs/engine-behavior.md` |
| touching acquisition, preparation, or the normalized record | `docs/data.md` |
| touching training, checkpoints, or the artifact roots | `docs/training-and-runtime.md` |
| touching UCI or any external protocol | `docs/interfaces.md` |
| adding or changing a benchmark or metric — 2300 lines, read the matching section | `docs/evaluation.md` |
| preference-control work | `docs/preference-controls.md` |
| a data, training, or evaluation choice wants prior art — never as background reading | `docs/research.md` |
| picking up, claiming, or publishing an issue | `docs/issue-workflow.md` |
| performing the GUI check | `docs/playable-uci.md` |
| sequencing work or refining a milestone | `docs/planning/roadmap.md` |
| adding a config file | `configs/README.md` |
| a benchmark writes results | `results/README.md` |

Infer the relevant rows from the requested area instead of requiring the user to
repeat them. Where a request is broad, choose a scoped first slice and say which.

## Traps

- Corpora and training runs live outside the worktree, beneath
  `ANTHRO_CHESS_DATA_ROOT` and `ANTHRO_CHESS_RUN_ROOT`. Run
  `uv run anthro machine` before concluding that this machine has no corpus or
  no checkpoints; an empty worktree is not evidence, because unset roots and a
  machine with no artifacts look identical from inside the repository.
- When a change adds or alters a benchmark, take a shakedown reading on two real
  checkpoints from one training run before the pull request is ready, using
  `--no-record` and the default reduced sweep rather than `--full`. Fixtures
  cannot show that a benchmark measures anything. `docs/evaluation.md` says what
  the reading is and is not.
- Use exact chess logic for board reconstruction, legal move generation, and
  rule bookkeeping.
- Legal-mask model move outputs before sampling.
- Keep temperature independent from rating and time settings.
- Keep UCI and other outside protocols as runtime interfaces, not model-native
  representations.
- Keep normalized data source-agnostic, compact, and reproducible from pipeline
  scripts.
- Human-like imperfections should emerge from training on human games, not from
  special-case mistake injection.
- This is a product/build project, not a research project. Prefer direct,
  practical choices over experiments whose main value is answering a research
  question.
- Do not duplicate exact names, thresholds, defaults, schema fields, or config
  values in prose when code, schemas, or config files should own them.

## Working here

- Use GitHub issues as the default implementation task tracker when available.
  When setting up milestone work, use real GitHub milestones, tracker issues,
  sub-issues, issue dependencies for true blockers, and the repo's `area:`,
  `type:`, `execution:`, and `verification:` labels as described in
  `docs/issue-workflow.md`.
- When choosing the next issue, select only a `type: task` issue in the active
  milestone that is attached to its tracker and has no open blockers. Treat any
  issue missing that metadata as intake awaiting maintainer triage. An issue
  labeled `execution: gpu-required` needs the GPU environment its body
  specifies. One labeled `verification: gpu-required` can be implemented without
  that environment and uses the documented handoff when the GPU check remains.
- A request to work on an issue is authorization to carry it through to a
  ready-for-review pull request that closes the issue on merge. The pull request
  is the maintainer's review boundary and is not merged by the agent. Several
  sessions often run against this repository at once, so the issue is claimed in
  GitHub while it is being worked on. When working from an issue, keep it
  updated with relevant findings, follow-ups, and completion status.
- When a change alters what a chess GUI observes, offer a real GUI check without
  being asked: point the maintainer's GUI at the working checkout with
  `scripts/anthro-gui-target .` and say the engine is ready to test, what to
  look at, and what a good result looks like. The GUI itself is configured once
  and is never reconfigured per issue.
- Add tests for chess-rule changes, model-facing encodings, data preprocessing,
  and runtime behavior.
- Document a major architectural choice in `docs/decisions/` when the rationale
  has lasting value. Roadmaps and staged build plans belong under
  `docs/planning/`.
- If implementation work changes the project direction, update the affected docs
  in the same change.
