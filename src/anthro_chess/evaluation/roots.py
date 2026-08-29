"""Rooting checked-in artifact paths beneath the shared machine data root.

Every shipped benchmark selection names its inputs the way the repository names
artifacts, as ``artifacts/<name>``. That path only resolves from a directory
which happens to hold an ``artifacts/`` tree, so it is rewritten beneath
``ANTHRO_CHESS_DATA_ROOT`` before the benchmark reads it.

Which fields a schema roots is declared here rather than at a call site, and
each benchmark's registry entry in :mod:`anthro_chess.evaluation.benchmarks`
names the list it is resolved through. A field left off that list is never
rewritten, and downstream nothing can tell such a path from one the caller wrote
deliberately.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import TypeVar

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.machine import DATA_ROOT_VARIABLE, optional_root

#: ``anthro eval run``: the frozen pool, plus the training corpus the leakage
#: check reads when that corpus has moved.
CHECKPOINT_ARTIFACT_FIELDS = ("pool", "leakage.training_normalized")
#: ``anthro eval novelty``: the pool every arm derives its continuations from.
NOVELTY_ARTIFACT_FIELDS = ("pool", "leakage.training_normalized")
#: ``anthro eval puzzles``: the owned puzzle artifact, and nothing else.
PUZZLE_ARTIFACT_FIELDS = ("puzzle_set",)
#: ``anthro eval rollout``: optional, since a pool-free suite records the
#: rollout scalars alone.
ROLLOUT_ARTIFACT_FIELDS = ("pool",)
#: ``anthro eval termination``: required, since the whole reading is a pass
#: over human games.
TERMINATION_ARTIFACT_FIELDS = ("pool",)
#: ``anthro eval ladder``: the pool the frozen openings are drawn from.
LADDER_ARTIFACT_FIELDS = ("openings.pool",)
#: ``anthro eval freeze``: the normalized corpus and its manifest, plus the
#: generation the cut must contain, which is a machine-local artifact like any
#: other.
POOL_ARTIFACT_FIELDS = ("normalized", "manifest", "predecessor")

ConfigT = TypeVar("ConfigT", bound=ConfigModel)


def resolve_artifact_roots(
    resolved: ResolvedConfig[ConfigT],
    *,
    fields: Sequence[str],
    overrides: Sequence[str] = (),
) -> ResolvedConfig[ConfigT]:
    """Root each named relative artifact path beneath the shared data root.

    A field is left alone when the data root is unset, when the configured
    path is already absolute, when it is unset, or when the caller named it in
    an explicit override: an explicit path is the caller's own and rooting it
    would silently move it somewhere else.
    """

    root = optional_root(DATA_ROOT_VARIABLE)
    if root is None:
        return resolved

    override_keys = {item.partition("=")[0] for item in overrides}
    value = resolved.value
    for field in fields:
        if field in override_keys:
            continue
        current = _read(value, field)
        if current is None or current.is_absolute():
            continue
        value = _write(value, field, rooted_artifact_path(root, current))
    if value is resolved.value:
        return resolved
    return ResolvedConfig(value=value, provenance=resolved.provenance)


def rooted_artifact_path(root: Path, configured_path: Path) -> Path:
    """Return a configured artifact path relocated beneath ``root``."""

    return root.joinpath(*_unprefixed(configured_path))


def artifact_name(path: Path, root: Path | None) -> str:
    """Return what a configured path names, with the machine taken off.

    Undoes :func:`rooted_artifact_path`, so a shipped selection and the same
    selection after rooting name one artifact rather than two. Kept beside the
    rooting it undoes for the reason this module exists: two halves of one
    convention living apart would drift, and here the drift would be silent — a
    benchmark's cost series would split with nothing to notice it.

    An absolute path keeps its full string when this machine's root does not
    contain it, and when there is no root to measure it against: both are the
    caller's own path rather than a named artifact, and no prefix can be taken
    off one without guessing. A machine with unset roots therefore keeps its own
    cost series, which is the honest reading of a path nothing here can name.
    """

    if root is not None:
        with suppress(ValueError):
            return str(path.relative_to(root))
    return str(Path(*_unprefixed(path)))


def _unprefixed(path: Path) -> tuple[str, ...]:
    """Return a configured path's parts without the repository's prefix."""

    parts = path.parts
    if parts and parts[0] == "artifacts":
        return parts[1:]
    return parts


def _read(config: ConfigModel, field: str) -> Path | None:
    """Return the path at one dotted field, or ``None`` when it is unset."""

    current: object = config
    for part in field.split("."):
        if current is None:
            return None
        current = getattr(current, part)
    if current is None:
        return None
    if not isinstance(current, Path):
        raise TypeError(f"configuration field {field!r} is not a path")
    return current


def _write(config: ConfigT, field: str, value: Path) -> ConfigT:
    """Return a copy of ``config`` with one dotted field replaced."""

    head, separator, tail = field.partition(".")
    if not separator:
        return config.model_copy(update={head: value})
    child = getattr(config, head)
    return config.model_copy(update={head: _write(child, tail, value)})
