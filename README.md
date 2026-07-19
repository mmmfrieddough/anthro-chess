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
a minimal causal action model with masked move loss, plus a locked development
environment. Broader data ingestion, the runnable model-training loop,
evaluation, playable runtime, UCI, model checkpoints, and packaged releases
remain planned work.

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
uv run anthro data prepare --help
```

Only implemented commands appear in `--help`. Training, evaluation, play, and
UCI commands will be added as those capabilities become real.

## Sample Data Path

The checked-in Lichess sample can be normalized without network access:

```console
uv run anthro data prepare \
  samples/lichess/standard-export-sample.pgn \
  artifacts/lichess-sample \
  --config configs/data/lichess-sample.toml
```

This writes compact Parquet game records under `normalized/` and a separate
provenance manifest under `manifests/`. The command requires the data
dependencies, which the locked development environment includes; installed
packages can add them with the `data` extra.

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
