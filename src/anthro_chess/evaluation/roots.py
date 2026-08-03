"""Rooting checked-in artifact paths beneath the shared machine data root.

Every shipped benchmark selection names its inputs the way the repository names
artifacts, as ``artifacts/<name>``. That path only resolves from a directory
which happens to hold an ``artifacts/`` tree, so it is rewritten beneath
``ANTHRO_CHESS_DATA_ROOT`` before the benchmark reads it.

Which fields a schema roots is declared here rather than at a call site, and
each benchmark's registry entry in :mod:`anthro_chess.evaluation.benchmarks`
names the list it is resolved through. A benchmark whose pool silently failed
to resolve is the one failure a sweep should never discover an hour in.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.machine import DATA_ROOT_VARIABLE, optional_root

#: ``anthro eval run``: the frozen pool, plus the training corpus the leakage
#: check reads when that corpus has moved.
CHECKPOINT_ARTIFACT_FIELDS = ("pool", "leakage.training_normalized")
#: ``anthro eval novelty``: the pool every arm derives its continuations from.
NOVELTY_ARTIFACT_FIELDS = ("pool", "leakage.training_normalized")
#: ``anthro eval puzzles``: the owned puzzle artifact and the overlap corpus.
PUZZLE_ARTIFACT_FIELDS = ("puzzle_set", "training_normalized")
#: ``anthro eval rollout``: optional, since a pool-free suite records the
#: rollout scalars alone.
ROLLOUT_ARTIFACT_FIELDS = ("pool",)
#: ``anthro eval termination``: required, since every reading is against human
#: endings.
TERMINATION_ARTIFACT_FIELDS = ("pool",)
#: ``anthro eval ladder``: the pool the frozen openings are drawn from.
LADDER_ARTIFACT_FIELDS = ("openings.pool",)
#: ``anthro eval freeze``: the normalized corpus and its manifest.
POOL_ARTIFACT_FIELDS = ("normalized", "manifest")

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

    parts = configured_path.parts
    if parts and parts[0] == "artifacts":
        parts = parts[1:]
    return root.joinpath(*parts)


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
