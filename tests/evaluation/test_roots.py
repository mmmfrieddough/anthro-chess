"""Rooting checked-in artifact paths beneath the shared machine data root."""

from __future__ import annotations

from pathlib import Path

import pytest

from anthro_chess.config import ConfigModel, ConfigProvenance, ResolvedConfig
from anthro_chess.evaluation.roots import resolve_artifact_roots


class Openings(ConfigModel):
    pool: Path | None = None


class Selection(ConfigModel):
    pool: Path | None = None
    openings: Openings = Openings()


def _resolved(**payload: object) -> ResolvedConfig[Selection]:
    return ResolvedConfig(
        value=Selection.model_validate(payload),
        provenance=ConfigProvenance(source=None, overrides=()),
    )


def test_a_relative_artifact_path_moves_beneath_the_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))

    rooted = resolve_artifact_roots(
        _resolved(pool="artifacts/example-pool"),
        fields=("pool",),
    )

    assert rooted.value.pool == tmp_path / "datasets" / "example-pool"


def test_a_nested_path_is_rooted_without_disturbing_its_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ladder names its pool inside its openings table."""

    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))

    rooted = resolve_artifact_roots(
        _resolved(openings={"pool": "artifacts/example-pool"}),
        fields=("openings.pool",),
    )

    assert rooted.value.openings.pool == tmp_path / "datasets" / "example-pool"


def test_an_unset_or_absolute_path_is_left_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))

    assert (
        resolve_artifact_roots(_resolved(), fields=("pool", "openings.pool")).value.pool
        is None
    )
    absolute = resolve_artifact_roots(
        _resolved(pool="/elsewhere/pool"),
        fields=("pool",),
    )
    assert absolute.value.pool == Path("/elsewhere/pool")


def test_an_explicitly_overridden_path_is_the_callers_own(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rooting it would silently move a path the caller named."""

    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))

    rooted = resolve_artifact_roots(
        _resolved(pool="artifacts/example-pool"),
        fields=("pool",),
        overrides=("pool=somewhere/else",),
    )

    assert rooted.value.pool == Path("artifacts/example-pool")


def test_no_data_root_leaves_every_path_as_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHRO_CHESS_DATA_ROOT", raising=False)

    rooted = resolve_artifact_roots(
        _resolved(pool="artifacts/example-pool"),
        fields=("pool",),
    )

    assert rooted.value.pool == Path("artifacts/example-pool")
