# Contributing to Anthro Chess

Anthro Chess is early in development. Small, focused contributions that match
the current implementation stage are easiest to review.

## Development Setup

The project requires Python 3.11 or newer and uses
[uv](https://docs.astral.sh/uv/) for its locked environment. From a fresh clone:

```console
uv sync --locked
uv run anthro smoke
```

Use an unlocked `uv` dependency command only when intentionally changing
project dependencies, and commit the resulting `pyproject.toml` and `uv.lock`
changes together.

Optional lightweight commit hooks can be installed with:

```console
uv run pre-commit install
```

## Corpora And Training Runs

Corpora and training runs are far too large for the repository, so they live
outside every worktree and are shared across all of them through a matched pair
of environment variables:

- `ANTHRO_CHESS_DATA_ROOT` — corpora, frozen evaluation pools, puzzle records;
- `ANTHRO_CHESS_RUN_ROOT` — training runs, each holding its run record,
  metrics, and checkpoints.

Set both or neither. They are two halves of one setup: evaluation reads a
checkpoint from one and a pool from the other, so a machine with only one
configured can run neither training nor evaluation end to end.
`README.md` has the export lines and
[`docs/training-and-runtime.md`](docs/training-and-runtime.md) owns the layout
beneath them and the precedence rules.

Both are optional, and that is the part worth knowing before you draw a
conclusion from an empty directory. Unset, checked-in relative paths resolve
inside the worktree — correct for a fresh clone, and indistinguishable from a
machine that genuinely has no artifacts. The run root is the quieter of the
two: a data command fails loudly when its root is missing, while a missing run
root silently falls back to worktree-relative paths.

**So a checkout with no `artifacts/` or `runs/` directory proves nothing, and
neither does searching the repository and its worktrees.** Ask the roots
instead:

```console
env | grep ANTHRO_CHESS
ls "${ANTHRO_CHESS_DATA_ROOT:?}"
ls "${ANTHRO_CHESS_RUN_ROOT:?}"/*/checkpoints/*.pt | tail
```

This matters most before concluding that a shakedown reading cannot be taken
here; `docs/issue-workflow.md` describes when one is required.

## Quality Checks

Run the same core checks used by continuous integration before opening a pull
request:

```console
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

Continuous integration runs that last command as `uv run pytest -n auto`, which
shards the suite across the runner's cores. The same flag works locally and is
worth using for a full run, but it starts a worker process per core, so leave it
off when running a handful of tests or a debugger.

Coverage and the complete pre-commit suite are available on demand:

```console
uv run pytest --cov
uv run pre-commit run --all-files
```

The default run collects the whole suite. Markers do not remove tests from it;
they let a caller select or exclude a group explicitly, as in
`uv run pytest -m "not gpu"`. A test needing a resource the machine may not have
must carry the matching marker from `pyproject.toml` and must also skip itself
when that resource is absent. A machine without the resource therefore gets a
fast CPU-only run by default, while a machine that has it exercises those tests
too.

Add focused tests for behavior changes. Documentation-only changes do not need
artificial tests, but commands and links should be checked against the current
repository.

## Issues and Planning

Search [existing issues](https://github.com/mmmfrieddough/anthro-chess/issues)
before opening a new one. Public issues are useful for bug reports and project
ideas, but they begin as maintainer intake rather than automatically entering
the implementation queue.

Maintainers organize actionable work with milestone trackers, task labels,
sub-issues, and dependencies. If you want to implement an existing task, check
that it is unblocked and leave a short comment before investing in a large
change. Broader implementation order belongs in
[`docs/planning/roadmap.md`](docs/planning/roadmap.md); detailed task status
belongs in GitHub. The maintainer and agent mechanics are documented in the
[`issue workflow`](docs/issue-workflow.md).

## Pull Requests

Keep each pull request cohesive and reviewable. In the description:

- explain what changed and why;
- link the relevant issue, using `Closes #123` when merging should close it;
- list the checks you ran and their results;
- call out documentation changes, limitations, or follow-up work.

Update durable docs when a change alters intended behavior, architecture,
interfaces, data shape, evaluation, or source-of-truth ownership. Do not make
design docs claim that planned capabilities already exist.

Pull requests run the repository's CPU-only quality, test, build, and installed
package smoke checks. Please keep ordinary output free of secrets, generated
artifacts, unrelated formatting changes, and personal environment files.

## Project Guidance

Start with the [README](README.md) for current capabilities and the
[documentation guide](docs/documentation.md) for where design, planning, and
decision material belongs. The project's intended-use boundary applies to
contributed features as well as released artifacts.
