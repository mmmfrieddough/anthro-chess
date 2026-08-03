"""What the driver owns: resolving a selection, and calling the entry point.

The benchmarks themselves are covered by their own tests. What is covered here
is the envelope every caller shares, because a command and a sweep that resolve
or invoke a benchmark differently is the drift this module exists to end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from anthro_chess.config import (
    ConfigModel,
    ConfigProvenance,
    ResolvedConfig,
)
from anthro_chess.evaluation.benchmarks import (
    Benchmark,
    benchmark_registry,
    resolve_benchmark,
    run_benchmark,
)

ROLLOUT_SELECTION = 'pool = "artifacts/example-pool"\n\n[reference]\nenabled = false\n'
LADDER_SELECTION = '[openings]\npool = "artifacts/example-pool"\n'


def _selection(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "selection.toml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("name", "body", "read"),
    [
        ("rollout", ROLLOUT_SELECTION, lambda config: config.pool),
        ("ladder", LADDER_SELECTION, lambda config: config.openings.pool),
    ],
)
def test_a_selection_is_rooted_by_what_its_own_entry_declares(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    body: str,
    read: Any,
) -> None:
    """A shipped selection names its pool the way every artifact path is named.

    Rooting belongs to resolving rather than to each caller: without it the
    shipped configuration only resolves from a directory that happens to hold
    an `artifacts/` tree. The ladder is here because it is the one benchmark
    whose artifact path is nested, so an entry that declared the wrong field
    would fail differently.
    """

    data_root = tmp_path / "datasets"
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(data_root))

    resolved = resolve_benchmark(
        benchmark_registry()[name], path=_selection(tmp_path, body)
    )

    assert read(resolved.value) == data_root / "example-pool"


def test_an_overridden_path_is_the_callers_own(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rooting an explicitly named path would silently move it.

    A relative override, so the path is spared by having been named rather
    than by being absolute.
    """

    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))

    resolved = resolve_benchmark(
        benchmark_registry()["rollout"],
        path=_selection(tmp_path, ROLLOUT_SELECTION),
        overrides=['pool="somewhere/else"'],
    )

    assert resolved.value.pool == Path("somewhere/else")


def test_every_benchmark_is_called_the_same_way(tmp_path: Path) -> None:
    """One call shape for all of them, so an addition to it is one edit."""

    captured: dict[str, Any] = {}

    def invoke(resolved_config: ResolvedConfig[Any], **keywords: Any) -> str:
        captured["resolved"] = resolved_config
        captured.update(keywords)
        return "the reading"

    resolved = ResolvedConfig(
        value=ConfigModel(),
        provenance=ConfigProvenance(source=None, overrides=()),
    )
    benchmark = Benchmark(
        name="fake",
        schema=ConfigModel,
        artifact_fields=(),
        errors=(),
        invoke=invoke,
    )

    result = run_benchmark(benchmark, resolved, run_root=tmp_path / "runs")

    assert result == "the reading"
    assert captured == {
        "resolved": resolved,
        "run_root": tmp_path / "runs",
        "store": None,
        "detail": None,
    }
