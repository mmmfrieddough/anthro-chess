from __future__ import annotations

import json
from pathlib import Path

import pytest

from anthro_chess import machine
from anthro_chess.config import ConfigError
from anthro_chess.machine import (
    DATA_ROOT_VARIABLE,
    RESULT_DETAIL_ROOT_VARIABLE,
    RESULTS_ROOT_VARIABLE,
    RUN_ROOT_VARIABLE,
    inspect_machine,
    optional_root,
    required_root,
)

ROOT_VARIABLES = (
    DATA_ROOT_VARIABLE,
    RUN_ROOT_VARIABLE,
    RESULTS_ROOT_VARIABLE,
    RESULT_DETAIL_ROOT_VARIABLE,
)


@pytest.fixture(autouse=True)
def unconfigured_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a machine that has configured nothing.

    The inherited environment is a real configured machine, so a test that did
    not clear it would pass or fail depending on where it ran.
    """

    for variable in ROOT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def _write_run(root: Path, name: str, *, steps: tuple[int, ...]) -> Path:
    run_path = root / name
    checkpoints = run_path / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run_path / "run.json").write_text("{}", encoding="utf-8")
    for step in steps:
        (checkpoints / f"step-{step:08d}.pt").write_bytes(b"")
    if steps:
        (checkpoints / "latest.json").write_text(
            json.dumps({"global_step": steps[-1], "path": f"step-{steps[-1]:08d}.pt"}),
            encoding="utf-8",
        )
    return run_path


def test_optional_root_treats_unset_and_blank_alike(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert optional_root(RUN_ROOT_VARIABLE) is None
    monkeypatch.setenv(RUN_ROOT_VARIABLE, "   ")
    assert optional_root(RUN_ROOT_VARIABLE) is None
    monkeypatch.setenv(RUN_ROOT_VARIABLE, str(tmp_path))
    assert optional_root(RUN_ROOT_VARIABLE) == tmp_path.resolve()


def test_required_root_names_the_variable_and_what_it_holds() -> None:
    with pytest.raises(ConfigError) as failure:
        required_root(RUN_ROOT_VARIABLE, alternative="a path must be given")

    message = str(failure.value)
    assert "a path must be given" in message
    assert RUN_ROOT_VARIABLE in message
    # Naming the variable without saying what belongs in it leaves the reader
    # knowing they are misconfigured and not how to fix it.
    assert "retained training runs" in message


def test_report_reads_runs_artifacts_and_the_default_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    data = tmp_path / "data"
    _write_run(runs, "trained", steps=(100, 8000))
    (data / "corpus" / "manifests").mkdir(parents=True)
    (data / "corpus" / "manifests" / "manifest.json").write_text("{}", encoding="utf-8")
    (data / "pool").mkdir(parents=True)
    (data / "pool" / "manifest.json").write_text("{}", encoding="utf-8")
    (data / "pool" / "games.parquet").write_bytes(b"")
    (data / "puzzles").mkdir(parents=True)
    (data / "puzzles" / "manifest.json").write_text("{}", encoding="utf-8")
    (data / "puzzles" / "puzzles.csv").write_text("", encoding="utf-8")
    (data / "acquired-only" / "raw").mkdir(parents=True)
    monkeypatch.setenv(RUN_ROOT_VARIABLE, str(runs))
    monkeypatch.setenv(DATA_ROOT_VARIABLE, str(data))

    report = inspect_machine()

    assert [run.name for run in report.runs] == ["trained"]
    assert report.runs[0].checkpoints == 2
    assert report.runs[0].latest_checkpoint == "step-00008000.pt"
    assert {artifact.name: artifact.kind for artifact in report.artifacts} == {
        "acquired-only": "other",
        "corpus": "corpus",
        "pool": "evaluation-pool",
        "puzzles": "puzzle-set",
    }
    assert report.problems == ()
    # No selection record was written, so the report says so rather than
    # inventing a default from whichever run happens to be present.
    assert report.selection.resolved is None
    assert report.selection.record_path == runs / "selected-model.json"


def test_report_records_are_json_serializable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(RUN_ROOT_VARIABLE, str(tmp_path))

    record = inspect_machine().as_record()

    assert json.loads(json.dumps(record)) == record


def test_a_fresh_clone_configuring_neither_root_is_not_a_problem() -> None:
    report = inspect_machine()

    assert report.problems == ()
    assert report.runs == ()
    assert report.artifacts == ()
    assert all(root.path is None for root in report.roots)


def test_half_a_configured_pair_is_reported_as_the_defect_it_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(DATA_ROOT_VARIABLE, str(tmp_path))

    problems = inspect_machine().problems

    assert len(problems) == 1
    assert DATA_ROOT_VARIABLE in problems[0]
    assert RUN_ROOT_VARIABLE in problems[0]


def test_a_root_pointing_nowhere_is_reported_rather_than_read_as_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "absent"
    monkeypatch.setenv(RUN_ROOT_VARIABLE, str(missing))
    monkeypatch.setenv(DATA_ROOT_VARIABLE, str(tmp_path))

    problems = inspect_machine().problems

    assert len(problems) == 1
    assert str(missing) in problems[0]


def test_the_results_and_detail_roots_report_their_own_defaults() -> None:
    roots = {root.variable: root for root in inspect_machine().roots}

    assert "./results" in roots[RESULTS_ROOT_VARIABLE].fallback
    assert RUN_ROOT_VARIABLE in roots[RESULT_DETAIL_ROOT_VARIABLE].fallback
    # Both default without the pair, so leaving either unset is an ordinary
    # setup rather than something to warn about.
    assert inspect_machine().problems == ()


def test_a_run_directory_without_a_record_or_checkpoints_is_not_a_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "benchmark-sweeps").mkdir()
    _write_run(tmp_path, "real", steps=())
    monkeypatch.setenv(RUN_ROOT_VARIABLE, str(tmp_path))

    report = inspect_machine()

    assert [run.name for run in report.runs] == ["real"]
    assert report.runs[0].latest_checkpoint is None


def test_a_base_installation_reports_what_it_can_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The model extra is optional, and a broken environment reaches here first."""

    _write_run(tmp_path, "trained", steps=(8000,))
    monkeypatch.setenv(RUN_ROOT_VARIABLE, str(tmp_path))
    monkeypatch.setenv(DATA_ROOT_VARIABLE, str(tmp_path))
    monkeypatch.setattr(machine, "_model_extra_installed", lambda: False)

    report = inspect_machine()

    assert report.unavailable == (machine.MODEL_EXTRA_NOTE,)
    # The parts that need no optional dependency are still answered.
    assert [run.name for run in report.runs] == ["trained"]
    assert report.runs[0].checkpoints == 1
    assert report.runs[0].latest_checkpoint is None
    assert report.selection.resolved is None
    # Not a resolution failure either, so the reason is stated once.
    assert report.selection.error is None
    # A missing optional dependency is not a misconfigured machine.
    assert report.problems == ()
