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

Coverage and the complete pre-commit suite are available on demand:

```console
uv run pytest --cov
uv run pre-commit run --all-files
```

The default test run is the fast CPU-only suite. Tests that require unusual
resources must use the markers defined in `pyproject.toml`, such as `slow`,
`gpu`, `network`, or `integration`, so callers can select them explicitly.

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
