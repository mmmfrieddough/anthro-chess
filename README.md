# anthro-chess

Anthro Chess is an early-stage project for building a controllable,
human-like chess model.

The intended product is a chess bot that can play competent games while
exposing dials for target rating, optional time behavior, randomness, and
optional soft preferences. The goal is not top engine strength. The goal is a
usable opponent that feels plausibly human at adjustable levels.

## Intended Use

Anthro Chess is intended as a standalone human-like chess opponent, sparring
partner, and training sandbox.

It is not intended to provide assistance in games against other people where
outside chess help is disallowed. Do not use Anthro Chess to cheat, evade fair
play rules, or misrepresent bot-generated moves as unaided human play.

## Development

Install [uv](https://docs.astral.sh/uv/) and create the locked development
environment:

```console
uv sync --locked
```

Use an unlocked `uv` dependency command only when intentionally changing
`pyproject.toml` and `uv.lock`.

Run the default verification suite:

```console
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

Coverage is available on demand with `uv run pytest --cov`. The default test
run excludes no tests yet; future slow, GPU, network, and integration tests
must use the markers configured in `pyproject.toml` so callers can select or
exclude those tiers explicitly.

Install the lightweight commit hooks with `uv run pre-commit install`, or run
them across the repository with `uv run pre-commit run --all-files`. Tests stay
outside the normal commit hook and remain part of the verification suite.

## Design Docs

The main docs describe the intended end state of the project:

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

Supporting background:

- [Decision records](docs/decisions/)
- [Related research](docs/research.md)

Implementation planning lives separately:

- [Roadmap](docs/planning/roadmap.md)

Agents working in this repo should read [AGENTS.md](AGENTS.md) first.
