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

## Platforms And Accelerators

Development happens on Linux and on Apple silicon, and the setup above is the
same on both. The locked PyTorch wheel is chosen by platform, so no extra
index, build variant, or install flag is involved:

| host | wheel the lock selects | accelerator PyTorch sees |
| --- | --- | --- |
| Linux x86-64 with an NVIDIA driver | `manylinux`, CUDA bundled | CUDA |
| macOS on Apple silicon | `macosx_14_0_arm64` | MPS |
| anything else, or no driver | that platform's build | none |

**What the project can select is still a smaller set than what PyTorch can
see.** Training, inference, and evaluation now each resolve an explicit `auto`,
`cpu`, `mps`, or `cuda` selection, so on a CUDA host all three use the GPU.
Only `auto` falls back; an explicit accelerator that is absent is an error
rather than a quiet CPU run.

The two selections are still reported apart, because they are separate lists
that have disagreed before and will again the next time a backend lands in one
path ahead of the other. An accelerator a selection does not accept produces
exactly the passing run that no accelerator at all produces, so the suite names
each selection that cannot use what is present. On a host where both accept the
card there is nothing to name:

```text
accelerators present: cuda (2 device(s))
accelerators the device selection accepts: cuda, mps
```

and while one of them was behind, as training was until CUDA training landed,
the gap said so outright:

```text
the training device selection does not accept cuda, so nothing here exercised a
training path on it
```

Which `gpu`-marked tests actually run follows from the same lists, and is a
property of the host rather than of the marker: the device-agreement check runs
on whichever accelerator is present, while the tests driving a whole training
run or a loaded checkpoint through a device selection need one that selection
accepts. Each states what the host has when it skips. The marker itself is
described under [Quality Checks](#quality-checks).

Work that is specifically about CUDA needs more than a driver. The exact
requirement belongs to the issue asking for it, because a distributed run and a
single-device run do not want the same host.

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
machine that genuinely has no artifacts.

**So a checkout with no `artifacts/` or `runs/` directory proves nothing, and
neither does searching the repository and its worktrees.** Ask the machine
instead:

```console
uv run anthro machine
```

It reports every root, the retained runs and prepared data artifacts beneath
the pair, and how the default model selection resolves. It exits nonzero when
the configuration is itself the problem — one half of the pair set without the
other, or a root pointing somewhere that is not a directory — because that
state otherwise looks exactly like a machine holding nothing.

This matters most before concluding that a shakedown reading cannot be taken
here; `docs/issue-workflow.md` describes when one is required.

A machine holding runs but no default model selection has to name a checkpoint
explicitly in every command. Record one instead:

```console
uv run anthro model select <run-directory-name>
```

### Standing One Up From The Pinned Sources

A machine whose roots are set but empty can rebuild the data side from the
checked-in configuration. Every input is pinned and every step is verified
against a recorded digest, so a rebuild elsewhere is checked rather than merely
plausible — which is what makes a corpus prepared on one machine and a pool
frozen on another the same artifact.

```console
uv run anthro data acquire --config configs/data/lichess-blitz-2017-04.toml
uv run anthro data prepare --config configs/data/lichess-blitz-2017-04.toml
uv run anthro eval freeze --config configs/evaluation/lichess-blitz-2017-04-pool.toml
uv run anthro eval prepare-puzzles --config configs/evaluation/lichess-puzzles-v1.toml
```

Three digests carry that verification, and a mismatch in each means something
different:

- `[archive] sha256` in the corpus and puzzle configurations — the downloaded
  release is the pinned one. A mismatch means the source moved, not that the
  download is corrupt in some recoverable way.
- `expected_game_ids_sha256` in the pool configuration — the frozen pool holds
  exactly the games the recorded one did. A mismatch means the corpus, the
  filters, or the split seed moved, and the benchmark needs a new pool version
  rather than a retry.
- `expected_puzzles_sha256` in the puzzle configuration — the deterministic
  selection over the pinned archive reproduced.

The network is used only to fetch those two archives, and the game archive is
by far the larger of them. Preparation, freezing, training, and evaluation all
run offline afterwards. Each archive is kept under the data root, so a later
rebuild re-verifies what is already there rather than fetching it again.

For a check that needs no network and no large download, the repository carries
a sample game that prepares, trains, and serves UCI in seconds. `README.md` has
that path.

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

Those are the commands continuous integration runs, with no extra flags on
either side. The suite shards itself across the machine's cores by default, so
a full run costs roughly a third of its serial wall time; pass `-n0` to turn
sharding off for a debugger or a handful of tests.

Sharding is why the suite pins Torch to a single thread per worker. Torch sizes
its thread pool from the core count and cannot see the other workers, so the
unpinned default oversubscribes the machine by a factor of the worker count —
which costs more wall time than sharding saves, and leaves timing assertions
measuring how contended the run was rather than what they were written to
measure.

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
