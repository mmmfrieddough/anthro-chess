# anthro-chess

[![Continuous integration](https://github.com/mmmfrieddough/anthro-chess/actions/workflows/ci.yml/badge.svg)](https://github.com/mmmfrieddough/anthro-chess/actions/workflows/ci.yml)

Anthro Chess is an early-stage project for building a controllable chess bot
that plays like a human opponent. The goal is not top engine strength. The goal
is a usable opponent with adjustable rating, optional human-like timing,
independent sampling temperature, and optional soft preferences.

## Project Status

The repository currently provides an installable Python package, a lightweight
`anthro` command, strict configuration foundations, `python-chess` integration,
stable model action ids, a reproducible PGN sample-data path, automated tests,
versioned per-ply model-facing encodings, deterministic sequence batching, and
a minimal causal action model with masked move loss, a reproducible bounded
Lichess baseline-corpus path, shared CPU/MPS training with explicit device and
determinism selection, an end-to-end minimal training proof, and a locked
development environment. Playable runtime, UCI, and packaged releases remain
planned work; validation metrics exist within the training path but the broader
evaluation harness is not yet implemented.

No trained Anthro Chess model is available yet, including through Hugging Face.

## Intended Use

Anthro Chess is intended as a standalone human-like chess opponent, sparring
partner, and training sandbox.

It is not intended to provide assistance in games against other people where
outside chess help is disallowed. Do not use Anthro Chess to cheat, evade fair
play rules, or misrepresent bot-generated moves as unaided human play.

## Quick Start

[Install uv](https://docs.astral.sh/uv/getting-started/installation/), clone the
repository, and create the locked environment:

```console
git clone https://github.com/mmmfrieddough/anthro-chess.git
cd anthro-chess
uv sync --locked
uv run anthro smoke
```

The smoke command needs no model checkpoint, dataset, GPU, or network access.
A successful run confirms that the package and command-line entry point are
installed correctly.

## Commands

The current command surface is intentionally small:

```console
uv run anthro --help
uv run anthro --version
uv run anthro smoke
uv run anthro data acquire --help
uv run anthro data prepare --help
uv run anthro train --help
```

Only implemented commands appear in `--help`. Evaluation, play, and UCI
commands will be added as those capabilities become real.

Generated datasets and runs can be shared across worktrees by setting:

```console
export ANTHRO_CHESS_DATA_ROOT="$HOME/.local/share/anthro-chess/datasets"
export ANTHRO_CHESS_RUN_ROOT="$HOME/.local/share/anthro-chess/runs"
```

The commands below then read and write those directories directly. Explicit
command path arguments and path overrides still take precedence.

## Sample Data Path

The checked-in Lichess sample can be normalized without network access:

```console
uv run anthro data prepare \
  samples/lichess/standard-export-sample.pgn \
  --config configs/data/lichess-sample.toml
```

This writes compact Parquet game records under `normalized/` and a separate
provenance manifest under `manifests/`. The command requires the data
dependencies, which the locked development environment includes; installed
packages can add them with the `data` extra.

The checked-in deterministic training selection consumes that prepared sample:

```console
uv run anthro train --config configs/training/sample-smoke.toml
```

It performs a bounded real optimizer run and validation on CPU and writes step
metrics plus a run record and optimizer-step checkpoints under the configured
run root. The artifacts preserve the resolved configuration, data manifest
and identities, model compatibility metadata, seed, code revision when
available, optimizer state, exact loader cursor, and optimizer-update evidence.
A run can continue from its latest checkpoint with strict command overrides:

```console
uv run anthro train \
  --config configs/training/sample-smoke.toml \
  --set 'resume_from="latest"' \
  --set steps=6
```

An explicit checkpoint path may be selected in configuration for a new output
directory. Resume rejects incompatible training, data, model, action, or
encoding identities before loading state. The sample selection is a correctness
smoke path, not evidence of useful chess strength.

On Apple silicon, the same full-precision path can run and resume on MPS:

```console
uv run anthro train \
  --config configs/training/sample-smoke.toml \
  --set 'device="mps"' \
  --set 'determinism="relaxed"'
```

The current MPS Transformer backward path requires relaxed determinism because
the locked Torch build does not provide a deterministic implementation for one
of its gradient operations. The selected backend, precision, and determinism
mode are retained in run and checkpoint metadata.

On Apple silicon with an MPS-enabled PyTorch build, the same runner can exercise
the real accelerator path:

```console
uv run anthro train --config configs/training/mps-smoke.toml
```

Explicit `mps` selection fails if the backend is unavailable; `auto` selects MPS
when available and otherwise CPU. Run artifacts record requested and resolved
devices, precision, determinism, accumulation, phase timings, throughput, and
sampled MPS memory. The synchronized phase profiling used by the smoke selection
adds diagnostic overhead and should be disabled for ordinary throughput runs.

## Baseline Training Corpus

The first many-game selection is pinned in
`configs/data/lichess-blitz-2017-04.toml`. Acquisition is an explicit network
operation that downloads the configured Lichess archive under the data root and
verifies its published SHA-256 digest:

Data commands use `ANTHRO_CHESS_DATA_ROOT` when their artifact directory is
omitted. They read and write the configured artifact directory directly, so
the same verified corpus can be reused across worktrees.

```console
uv run anthro data acquire \
  --config configs/data/lichess-blitz-2017-04.toml
```

Preparation then streams the compressed PGN directly into bounded normalized
shards:

```console
uv run anthro data prepare \
  --config configs/data/lichess-blitz-2017-04.toml
```

The checked-in selection owns the exact release, checksum, one Lichess rating
namespace, deterministic size bound, split recipe, and shard sizing. Raw and
normalized corpus files remain outside Git. Once acquisition finishes,
preparation and later training work need no network access.

## Minimal Training Proof

The first complete proof combines the CPU correctness gate with a measured
many-game Apple-silicon MPS run. The baseline command reports both raw move loss
and legally masked held-out move loss; the latter is compared directly with
uniform selection over exact legal actions.

See
[the minimal training proof](docs/planning/minimal-training-proof.md)
for the reproducible corpus slice, baseline configuration, resume command,
acceptance comparison, and measured evidence. The resulting checkpoint is a
development artifact, not a published model release.

When `ANTHRO_CHESS_RUN_ROOT` is set, checked-in training commands write their
complete run directories there. Keep the run record, metrics, and checkpoint
directory together so a later session can resume or select a compatible
checkpoint without copying it. See
[`docs/training-and-runtime.md`](docs/training-and-runtime.md) for the artifact
layout and boundary with future public model hosting.

## Development

Use an unlocked `uv` dependency command only when intentionally changing
`pyproject.toml` and `uv.lock`. Run the canonical local checks with:

```console
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup, test tiers,
pre-commit hooks, issue routing, and pull request expectations. Continuous
integration also builds the package and smoke-tests the installed wheel.

## Documentation

The main docs describe the intended end state rather than claiming every
capability is implemented:

- [Documentation guide](docs/documentation.md)
- [Vision](docs/vision.md)
- [Design principles](docs/design-principles.md)
- [Architecture](docs/architecture.md)
- [Engine behavior](docs/engine-behavior.md)
- [Data](docs/data.md)
- [Training and runtime](docs/training-and-runtime.md)
- [Interfaces](docs/interfaces.md)
- [Evaluation](docs/evaluation.md)
- [Preference controls](docs/preference-controls.md)

Supporting context lives in the [decision records](docs/decisions/) and
[research notes](docs/research.md). Implementation order lives separately in
the [roadmap](docs/planning/roadmap.md), while actionable work is tracked in
[GitHub issues](https://github.com/mmmfrieddough/anthro-chess/issues) using the
repository's [issue workflow](docs/issue-workflow.md).

Agents working in this repository should read [AGENTS.md](AGENTS.md) first.

## License

Copyright (C) 2026 Patrizio Spagnardi III.

Anthro Chess is free software available under the
[GNU General Public License v3.0 or later](LICENSE).
