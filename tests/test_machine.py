from __future__ import annotations

import json
import os
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


def _write_run(
    root: Path,
    name: str,
    *,
    steps: tuple[int, ...],
    record: object,
    modified: float | None = None,
) -> Path:
    run_path = root / name
    checkpoints = run_path / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run_path / "run.json").write_text(json.dumps(record), encoding="utf-8")
    for step in steps:
        (checkpoints / f"step-{step:08d}.pt").write_bytes(b"")
    if steps:
        (checkpoints / "latest.json").write_text(
            json.dumps({"global_step": steps[-1], "path": f"step-{steps[-1]:08d}.pt"}),
            encoding="utf-8",
        )
        if modified is not None:
            os.utime(checkpoints / f"step-{steps[-1]:08d}.pt", (modified, modified))
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loadable_run_record: dict[str, object],
) -> None:
    runs = tmp_path / "runs"
    data = tmp_path / "data"
    _write_run(runs, "trained", steps=(100, 8000), record=loadable_run_record)
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loadable_run_record: dict[str, object],
) -> None:
    (tmp_path / "benchmark-sweeps").mkdir()
    _write_run(tmp_path, "real", steps=(), record=loadable_run_record)
    monkeypatch.setenv(RUN_ROOT_VARIABLE, str(tmp_path))

    report = inspect_machine()

    assert [run.name for run in report.runs] == ["real"]
    assert report.runs[0].latest_checkpoint is None
    # Retained, but nothing a reading can be taken against.
    assert report.runs[0].loadable is False
    assert report.runs[0].blocker == "no checkpoints"


def test_a_run_record_this_code_would_write_itself_is_loadable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loadable_run_record: dict[str, object],
) -> None:
    """The question a session arrives with, answered without reading weights."""

    _write_run(tmp_path, "current", steps=(8000,), record=loadable_run_record)
    monkeypatch.setenv(RUN_ROOT_VARIABLE, str(tmp_path))

    run = inspect_machine().runs[0]

    assert run.loadable is True
    assert run.blocker is None


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("model", {"version": 4}, "model identity 4, and this code loads 5"),
        (
            "model",
            {"rating_conditioning": "pre-transformer"},
            "run model is incompatible with this model runner",
        ),
        (
            "action_vocabulary",
            {"version": 1},
            "run record action vocabulary is incompatible",
        ),
        (
            "encoding",
            {"version": 1},
            "run record model-facing encoding is incompatible",
        ),
        (
            "execution",
            {"precision": "bfloat16-mixed", "parameter_dtype": "float32"},
            "run parameter precision is unsupported",
        ),
    ],
)
def test_a_run_behind_any_gate_says_which_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loadable_run_record: dict[str, object],
    field: str,
    value: dict[str, object],
    expected: str,
) -> None:
    """Naming the gate is what separates a stale run root from a broken install."""

    model = loadable_run_record["model"]
    assert isinstance(model, dict)
    stale = {**model, **value} if field == "model" else value
    _write_run(
        tmp_path,
        "retired",
        steps=(8000,),
        record={**loadable_run_record, field: stale},
    )
    monkeypatch.setenv(RUN_ROOT_VARIABLE, str(tmp_path))

    run = inspect_machine().runs[0]

    assert run.loadable is False
    assert run.blocker == expected


def test_a_run_that_cannot_be_read_at_all_is_reported_rather_than_raised(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loadable_run_record: dict[str, object],
) -> None:
    """This is the command a broken machine reaches for first."""

    orphan = _write_run(
        tmp_path, "no-record", steps=(8000,), record=loadable_run_record
    )
    (orphan / "run.json").unlink()
    unpointed = _write_run(
        tmp_path, "no-pointer", steps=(8000,), record=loadable_run_record
    )
    (unpointed / "checkpoints" / "latest.json").unlink()
    truncated = _write_run(
        tmp_path, "half-written", steps=(8000,), record=loadable_run_record
    )
    (truncated / "run.json").write_text('{"model":', encoding="utf-8")
    undecodable = _write_run(
        tmp_path, "not-utf8", steps=(8000,), record=loadable_run_record
    )
    (undecodable / "run.json").write_bytes(b"\xff\xfe{")
    monkeypatch.setenv(RUN_ROOT_VARIABLE, str(tmp_path))

    runs = {run.name: run for run in inspect_machine().runs}

    assert runs["no-record"].blocker == "no run record"
    # A default selection resolves through the pointer, in the resolver's own
    # words, so weights it cannot reach are not weights a reading can name.
    assert "cannot resolve latest checkpoint" in str(runs["no-pointer"].blocker)
    assert runs["half-written"].loadable is False
    assert "cannot load run record" in str(runs["half-written"].blocker)
    assert "cannot load run record" in str(runs["not-utf8"].blocker)


def test_runs_are_listed_newest_first_on_the_latest_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loadable_run_record: dict[str, object],
) -> None:
    """Recency is what puts the runs that still load where a session looks."""

    _write_run(
        tmp_path,
        "aaa-oldest",
        steps=(8000,),
        record=loadable_run_record,
        modified=1_700_000_000,
    )
    _write_run(
        tmp_path,
        "zzz-newest",
        steps=(8000,),
        record=loadable_run_record,
        modified=1_800_000_000,
    )
    _write_run(tmp_path, "record-only", steps=(), record=loadable_run_record)
    monkeypatch.setenv(RUN_ROOT_VARIABLE, str(tmp_path))

    report = inspect_machine()

    # Alphabetical order is what hid the usable runs among the retired ones.
    assert [run.name for run in report.runs] == [
        "zzz-newest",
        "aaa-oldest",
        "record-only",
    ]
    newest = report.runs[0].latest_modified
    assert newest is not None
    assert newest.timestamp() == 1_800_000_000
    assert report.runs[2].latest_modified is None


def test_a_checkpoint_that_cannot_be_stated_costs_a_stamp_and_not_the_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loadable_run_record: dict[str, object],
) -> None:
    """A run root is a machine-local directory, not a structure this owns."""

    run_path = _write_run(
        tmp_path, "dangling", steps=(8000,), record=loadable_run_record
    )
    checkpoint = run_path / "checkpoints" / "step-00008000.pt"
    checkpoint.unlink()
    checkpoint.symlink_to(tmp_path / "absent.pt")
    monkeypatch.setenv(RUN_ROOT_VARIABLE, str(tmp_path))

    report = inspect_machine()

    assert [run.name for run in report.runs] == ["dangling"]
    assert report.runs[0].latest_modified is None


def test_a_base_installation_reports_what_it_can_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loadable_run_record: dict[str, object],
) -> None:
    """The model extra is optional, and a broken environment reaches here first."""

    _write_run(tmp_path, "trained", steps=(8000,), record=loadable_run_record)
    orphan = _write_run(
        tmp_path, "no-record", steps=(8000,), record=loadable_run_record
    )
    (orphan / "run.json").unlink()
    monkeypatch.setenv(RUN_ROOT_VARIABLE, str(tmp_path))
    monkeypatch.setenv(DATA_ROOT_VARIABLE, str(tmp_path))
    monkeypatch.setattr(machine, "_model_extra_installed", lambda: False)

    report = inspect_machine()
    runs = {run.name: run for run in report.runs}

    assert report.unavailable == (machine.MODEL_EXTRA_NOTE,)
    # The parts that need no optional dependency are still answered.
    assert sorted(runs) == ["no-record", "trained"]
    assert runs["trained"].checkpoints == 1
    assert runs["trained"].latest_checkpoint is None
    # The contract needs the missing dependencies to compare against, so it is
    # left undetermined rather than guessed. What the directory itself settles
    # is still settled.
    assert runs["trained"].loadable is None
    assert runs["no-record"].blocker == "no run record"
    assert report.selection.resolved is None
    # Not a resolution failure either, so the reason is stated once.
    assert report.selection.error is None
    # A missing optional dependency is not a misconfigured machine.
    assert report.problems == ()
