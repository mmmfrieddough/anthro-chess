"""What the driver owns: resolving a selection, and calling the entry point.

The benchmarks themselves are covered by their own tests. What is covered here
is the envelope every caller shares, because a command and a sweep that resolve
or invoke a benchmark differently is the drift this module exists to end.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from anthro_chess.config import (
    ConfigProvenance,
    ResolvedConfig,
)
from anthro_chess.evaluation.benchmarks import (
    Benchmark,
    benchmark_registry,
    resolve_benchmark,
    run_benchmark,
)
from anthro_chess.evaluation.cost import BENCHMARK_COST_KIND
from anthro_chess.evaluation.recording import ResultRecording
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DetailStore,
    ResultEnvelope,
    ResultsStore,
    ResultsStoreError,
    measurement,
)
from anthro_chess.evaluation.results.metrics import BENCHMARK_WALL_CLOCK_SECONDS
from anthro_chess.evaluation.selection import CheckpointSelection
from anthro_chess.inference import ModelRunnerConfig

COST = BenchmarkReference(name="fake-benchmark", version=1)
ROLLOUT_SELECTION = 'pool = "artifacts/example-pool"\n\n[reference]\nenabled = false\n'
LADDER_SELECTION = '[openings]\npool = "artifacts/example-pool"\n'


@dataclass(frozen=True)
class _Reading:
    """The three fields the driver fills in on every benchmark's result."""

    envelopes: tuple[ResultEnvelope, ...] = ()
    recorded_paths: tuple[Path, ...] = ()
    detail_paths: tuple[Path, ...] = ()


class _FakeError(ValueError):
    """What one benchmark raises for a bad reading."""


def _fake(
    invoke: Callable[..., Any],
    *,
    cost: BenchmarkReference | None = None,
) -> Benchmark:
    """Return a registry entry over a measuring body written for one test."""

    return Benchmark(
        name="fake",
        schema=CheckpointSelection,
        artifact_fields=(),
        errors=(_FakeError,),
        error=_FakeError,
        invoke=invoke,
        cost=cost,
    )


def _resolved() -> ResolvedConfig[CheckpointSelection]:
    return ResolvedConfig(
        # A selection on the CPU, so a cost record is attributed to the same
        # device wherever the suite runs.
        value=CheckpointSelection(model=ModelRunnerConfig(device="cpu")),
        provenance=ConfigProvenance(source=None, overrides=()),
    )


def _measure(recording: Any) -> None:
    """Record one reading, which is all these tests need a benchmark to do."""

    recording.measuring(
        CheckpointReference(label="checkpoint-a", step=8000),
        kind="fake-reading",
        benchmark=BenchmarkReference(name="fake-reading", version=1),
    ).add(
        [measurement("training_health.gradient_norm", 1.5)],
        payload=dict,
        description="What the fake measured.",
    )


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

    def invoke(resolved_config: ResolvedConfig[Any], **keywords: Any) -> _Reading:
        captured["resolved"] = resolved_config
        captured.update(keywords)
        return _Reading()

    result = run_benchmark(
        _fake(invoke),
        _resolved(),
        run_root=tmp_path / "runs",
        # What one benchmark's entry point takes beyond the shared call. Three
        # of them accept a pre-loaded runner.
        runner="a pre-loaded runner",
    )

    assert isinstance(result, _Reading)
    # Exactly these: a benchmark is handed a recording rather than the store
    # and detail tier it used to open one with.
    assert set(captured) == {"resolved", "run_root", "recording", "runner"}
    assert captured["run_root"] == tmp_path / "runs"
    assert captured["runner"] == "a pre-loaded runner"
    assert isinstance(captured["recording"], ResultRecording)


def test_a_store_failure_while_measuring_is_the_benchmarks_own_error() -> None:
    """The conversion covers the measuring, not only the recording tail.

    A benchmark builds its measurements outside the calls it makes into the
    recorder, and `measurement` raises what the store does. One that escaped
    unconverted would end a whole sweep rather than failing one step.
    """

    def invoke(resolved_config: ResolvedConfig[Any], **keywords: Any) -> _Reading:
        raise ResultsStoreError("the store is locked")

    with pytest.raises(_FakeError, match="the store is locked"):
        run_benchmark(_fake(invoke), _resolved())


def test_the_driver_assembles_the_result_from_what_was_recorded(
    tmp_path: Path,
) -> None:
    """A benchmark returns what it measured; where it landed is added here.

    Seven copies of that assembly is what let one result drift to a different
    shape, so no benchmark writes it any more.
    """

    detail = DetailStore(tmp_path / "detail")

    def invoke(resolved_config: ResolvedConfig[Any], **keywords: Any) -> _Reading:
        recorder = keywords["recording"].measuring(
            CheckpointReference(label="checkpoint-a", step=8000),
            kind="fake-reading",
            benchmark=BenchmarkReference(name="fake-reading", version=1),
        )
        # No measurement, so this writes the payload and no envelope.
        recorder.add([], payload=dict, description="What the fake measured.")
        return _Reading()

    result = run_benchmark(_fake(invoke), _resolved(), detail=detail)

    (written,) = result.detail_paths
    assert written.parent == detail.root / "fake-reading" / "checkpoint-a"
    assert result.envelopes == ()
    assert result.recorded_paths == ()


def test_a_recording_benchmark_raises_what_its_entry_declares() -> None:
    """The error the store's own failures are converted into is one it names."""

    registry = benchmark_registry()
    recording_entries = [entry for entry in registry.values() if entry.records_results]

    assert recording_entries
    assert all(entry.error in entry.errors for entry in recording_entries)


def test_a_recording_benchmark_declares_what_its_cost_is_filed_under() -> None:
    """Checked rather than defaulted, because a default would fail silently.

    A benchmark that recorded a reading and no cost is exactly the drift this
    driver exists to end, and nothing about the reading itself would show it.
    """

    registry = benchmark_registry()
    recording_entries = [entry for entry in registry.values() if entry.records_results]

    assert all(entry.cost is not None for entry in recording_entries)
    # One cost series per benchmark. Two sharing a name would put two
    # unrelated amounts of work on one line.
    names = [entry.cost.name for entry in recording_entries if entry.cost is not None]
    assert len(set(names)) == len(names)


def test_every_recording_benchmark_selects_a_checkpoint_the_same_way() -> None:
    """A sweep replaces this field wholesale, and `model_copy` cannot check it.

    A schema off the shared base would take the replacement as a stray
    attribute and go on measuring its own checkpoint, and would drop nothing
    from its cost workload either.
    """

    schemas = [
        entry.schema
        for entry in benchmark_registry().values()
        if entry.records_results and entry.schema is not None
    ]

    assert schemas
    assert all(issubclass(schema, CheckpointSelection) for schema in schemas)


def test_the_driver_commits_what_the_invocation_cost(tmp_path: Path) -> None:
    """Timed here because here is the only place that sees one begin.

    Committed rather than machine-local: the failure this record exists to
    prevent is a stale cost claim nobody contradicts, and only a committed
    record is contradicted by a diff.
    """

    store = ResultsStore(tmp_path / "results")
    before = time.perf_counter()

    def invoke(resolved_config: ResolvedConfig[Any], **keywords: Any) -> _Reading:
        _measure(keywords["recording"])
        time.sleep(0.05)
        return _Reading()

    result = run_benchmark(_fake(invoke, cost=COST), _resolved(), store=store)

    reading, cost = result.envelopes
    assert reading.kind == "fake-reading"
    assert cost.kind == BENCHMARK_COST_KIND
    assert cost.benchmark == COST
    (recorded,) = cost.measurements
    assert recorded.metric == BENCHMARK_WALL_CLOCK_SECONDS.identifier
    # The window covers the whole invocation, so work done inside it counts.
    assert 0.05 <= recorded.value <= time.perf_counter() - before
    # Committed beside the reading rather than in a tier only this machine sees.
    assert len(result.recorded_paths) == 2
    assert cost in store.results()


def test_an_invocation_that_measured_nothing_costs_nothing_worth_recording(
    tmp_path: Path,
) -> None:
    """Those seconds are the cost of finding nothing, and belong to no series."""

    store = ResultsStore(tmp_path / "results")

    def invoke(resolved_config: ResolvedConfig[Any], **keywords: Any) -> _Reading:
        keywords["recording"].measuring(
            CheckpointReference(label="checkpoint-a", step=8000),
            kind="fake-reading",
            benchmark=BenchmarkReference(name="fake-reading", version=1),
        )
        return _Reading()

    result = run_benchmark(_fake(invoke, cost=COST), _resolved(), store=store)

    assert result.envelopes == ()
    assert store.results() == ()


def test_a_failed_invocation_records_no_cost(tmp_path: Path) -> None:
    """A failure's duration is the cost of the failure rather than a reading."""

    store = ResultsStore(tmp_path / "results")

    def invoke(resolved_config: ResolvedConfig[Any], **keywords: Any) -> _Reading:
        _measure(keywords["recording"])
        raise _FakeError("the pool ran out half way through")

    with pytest.raises(_FakeError):
        run_benchmark(_fake(invoke, cost=COST), _resolved(), store=store)

    assert store.results() == ()


def test_a_measured_invocation_that_records_nothing_still_reports_its_cost() -> None:
    """`--no-record` is how a shakedown reading is taken.

    The envelope is assembled exactly as a committed one is and simply never
    appended, so what a benchmark reports does not depend on the store.
    """

    def invoke(resolved_config: ResolvedConfig[Any], **keywords: Any) -> _Reading:
        _measure(keywords["recording"])
        return _Reading()

    result = run_benchmark(_fake(invoke, cost=COST), _resolved())

    assert [envelope.kind for envelope in result.envelopes] == [
        "fake-reading",
        BENCHMARK_COST_KIND,
    ]
    assert result.recorded_paths == ()
