"""What the sweep driver owns: order, recording, resume, and failure.

The benchmarks themselves are covered by their own tests, so these run against
a fake registry. That is not a shortcut around a slow suite: the driver's job
is to compose benchmarks it did not write, and a fake registry is the only way
to assert that it composes *any* benchmark correctly rather than that one
particular benchmark happened to work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import chess
import pytest
from pydantic import ValidationError

from anthro_chess.chess import encode_move
from anthro_chess.config import ConfigModel, ConfigProvenance, ResolvedConfig
from anthro_chess.evaluation.benchmarks import Benchmark
from anthro_chess.evaluation.games import (
    DecisionPolicy,
    DecisionRecord,
    GameOutcome,
    GameRecord,
    GameTermination,
    SeatRecord,
    build_game_record,
)
from anthro_chess.evaluation.recording import ResultRecording
from anthro_chess.evaluation.suite import (
    StepOutcome,
    StepStatus,
    SuiteConfig,
    SuiteError,
    SuiteScale,
    resolve_suite,
    run_suite,
    sweep_directory,
)
from anthro_chess.inference.config import ModelRunnerConfig

PLAYED = ("e2e4", "e7e5", "g1f3", "b8c6")


class FakeDetail(ConfigModel):
    retain_games: bool = True


class FakeConfig(ConfigModel):
    """A benchmark schema with the fields every real one shares."""

    model: ModelRunnerConfig = ModelRunnerConfig()
    checkpoint_label: str | None = None
    pool: Path | None = None
    games_per_position: int = 4
    detail: FakeDetail = FakeDetail()


class FakeError(ValueError):
    """The error class one fake benchmark raises for a bad reading."""


@dataclass
class FakeEnvelope:
    measurements: tuple[str, ...] = ("one", "two")


@dataclass
class FakeResult:
    """The shape a benchmark result carries back to the driver."""

    config: FakeConfig | None = None
    recorded: bool = False
    envelopes: tuple[FakeEnvelope, ...] = ()
    recorded_paths: tuple[Path, ...] = ()
    detail_paths: tuple[Path, ...] = ()
    games: tuple[GameRecord, ...] = ()


class FakeStore:
    """The one call the driver makes on a store: appending what was read."""

    def __init__(self) -> None:
        self.appended: list[Any] = []

    def append(self, envelope: Any) -> Path:
        self.appended.append(envelope)
        return Path(f"appended-{len(self.appended)}.json")


@dataclass
class Recorder:
    """Records what the driver invoked, in the order it invoked it."""

    calls: list[str] = field(default_factory=list)
    failing: set[str] = field(default_factory=set)
    games: tuple[GameRecord, ...] = ()

    def invoke(self, name: str) -> Any:
        def run(
            resolved: ResolvedConfig[FakeConfig],
            *,
            run_root: Path | None = None,
            recording: ResultRecording,
        ) -> FakeResult:
            self.calls.append(name)
            if name in self.failing:
                raise FakeError(f"{name} could not be read")
            # What was read goes into the recording, the way a real benchmark
            # adds it; the driver is what puts it back on the result.
            recording.envelopes.append(cast("Any", FakeEnvelope()))
            return FakeResult(
                config=resolved.value,
                recorded=recording.store is not None,
                games=self.games,
            )

        return run

    def reader(self, name: str) -> Any:
        def run(path: Path) -> FakeResult:
            self.calls.append(name)
            if name in self.failing:
                raise FakeError(f"{name} could not be read")
            payload = json.loads(path.read_text(encoding="utf-8"))
            return FakeResult(envelopes=(), games=tuple(payload["games"]))

        return run


def _game_record(seed: int = 11) -> GameRecord:
    """Return one playable game whose decisions carry a policy."""

    action_ids = tuple(encode_move(chess.Move.from_uci(value)) for value in PLAYED)
    seat = SeatRecord(
        kind="model",
        label="model-a",
        seed=seed,
        configuration={
            "target_rating": 1500,
            "temperature": 1.0,
            "resignation_enabled": False,
        },
    )
    decisions = [
        DecisionRecord(
            ply_index=index,
            slot="white" if index % 2 == 0 else "black",
            action_id=action_id,
            policy=DecisionPolicy(
                enabled_action_count=20,
                selected_probability=0.4,
                selected_rank=1 + index % 2,
                preferred_action_id=action_id,
                preferred_probability=0.5,
            ),
        )
        for index, action_id in enumerate(action_ids)
    ]
    return build_game_record(
        initial_position=chess.STARTING_FEN,
        prefix_plies=0,
        action_ids=action_ids,
        white=seat,
        black=seat.model_copy(update={"label": "model-b"}),
        seed=seed,
        decisions=decisions,
        outcome=GameOutcome(
            result="*",
            termination=GameTermination.PLY_LIMIT,
            adjudicated=True,
        ),
    )


def _registry(recorder: Recorder) -> dict[str, Benchmark]:
    """Return three benchmarks: two ordinary and one that reads the first's games."""

    return {
        "alpha": Benchmark(
            name="alpha",
            schema=FakeConfig,
            artifact_fields=("pool",),
            errors=(FakeError,),
            error=FakeError,
            invoke=recorder.invoke("alpha"),
            games=lambda result: result.games,
            retains_games=lambda config: config.detail.retain_games,
        ),
        "beta": Benchmark(
            name="beta",
            schema=FakeConfig,
            artifact_fields=(),
            errors=(FakeError,),
            error=FakeError,
            invoke=recorder.invoke("beta"),
        ),
        "gamma": Benchmark(
            name="gamma",
            schema=None,
            artifact_fields=(),
            errors=(FakeError, OSError),
            error=FakeError,
            invoke=recorder.reader("gamma"),
            records_results=False,
            games_from="alpha",
        ),
    }


def _selection(tmp_path: Path, name: str = "alpha.toml", body: str = "") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _result(outcome: StepOutcome) -> FakeResult:
    """Return one completed step's reading, asserting that it has one."""

    assert isinstance(outcome.result, FakeResult)
    return outcome.result


def _suite(**payload: Any) -> ResolvedConfig[SuiteConfig]:
    return ResolvedConfig(
        value=SuiteConfig.model_validate(payload),
        provenance=ConfigProvenance(source="suite.toml", overrides=()),
    )


def test_a_dependent_step_runs_after_the_step_it_reads(tmp_path: Path) -> None:
    """Declaration order does not decide it; the dependency does."""

    recorder = Recorder()
    plan = resolve_suite(
        _suite(
            benchmarks={
                "gamma": {"record": False},
                "beta": {"config": str(_selection(tmp_path, "beta.toml"))},
                "alpha": {"config": str(_selection(tmp_path))},
            }
        ),
        registry=_registry(recorder),
    )

    assert [step.name for step in plan.steps] == ["beta", "alpha", "gamma"]


def test_declared_order_survives_where_nothing_forces_a_change(
    tmp_path: Path,
) -> None:
    """A stable order is what makes two sweeps' ledgers comparable."""

    recorder = Recorder()
    plan = resolve_suite(
        _suite(
            benchmarks={
                "beta": {"config": str(_selection(tmp_path, "beta.toml"))},
                "alpha": {"config": str(_selection(tmp_path))},
            }
        ),
        registry=_registry(recorder),
    )

    assert [step.name for step in plan.steps] == ["beta", "alpha"]


def test_a_dependency_the_sweep_does_not_include_is_refused(
    tmp_path: Path,
) -> None:
    """The dependent step would have nothing to read, so the plan fails."""

    with pytest.raises(SuiteError, match="which this sweep does not include"):
        resolve_suite(
            _suite(benchmarks={"gamma": {"record": False}}),
            registry=_registry(Recorder()),
        )


def test_a_producer_that_discards_its_games_is_refused_before_the_sweep_runs(
    tmp_path: Path,
) -> None:
    """The point of resolving the whole plan first: this cannot surface an hour in."""

    selection = _selection(tmp_path, body="[detail]\nretain_games = false\n")
    with pytest.raises(SuiteError, match="discards"):
        resolve_suite(
            _suite(
                benchmarks={
                    "alpha": {"config": str(selection)},
                    "gamma": {"record": False},
                }
            ),
            registry=_registry(Recorder()),
        )


def test_a_cycle_is_refused(tmp_path: Path) -> None:
    """Two steps each waiting for the other never becomes an order."""

    with pytest.raises(SuiteError, match="cycle"):
        resolve_suite(
            _suite(
                benchmarks={
                    "alpha": {
                        "config": str(_selection(tmp_path)),
                        "after": ["beta"],
                    },
                    "beta": {
                        "config": str(_selection(tmp_path, "beta.toml")),
                        "after": ["alpha"],
                    },
                }
            ),
            registry=_registry(Recorder()),
        )


def test_an_unknown_benchmark_names_what_this_build_runs(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="unknown benchmark"):
        resolve_suite(
            _suite(benchmarks={"delta": {"config": str(_selection(tmp_path))}}),
            registry=_registry(Recorder()),
        )


def test_a_step_with_no_result_kind_cannot_be_asked_to_record(
    tmp_path: Path,
) -> None:
    """Silently ignoring the request would make the ledger say something false."""

    with pytest.raises(SuiteError, match="no result kind"):
        resolve_suite(
            _suite(
                benchmarks={
                    "alpha": {"config": str(_selection(tmp_path))},
                    "gamma": {"record": True},
                }
            ),
            registry=_registry(Recorder()),
        )


def test_the_reduced_sweep_shrinks_and_the_full_one_does_not(
    tmp_path: Path,
) -> None:
    """Both scales read the same selection; only the sample counts differ."""

    entry = {
        "config": str(_selection(tmp_path)),
        "reduced": ["games_per_position=1"],
    }
    reduced = resolve_suite(
        _suite(benchmarks={"alpha": entry}),
        scale=SuiteScale.REDUCED,
        registry=_registry(Recorder()),
    )
    full = resolve_suite(
        _suite(benchmarks={"alpha": entry}),
        scale=SuiteScale.FULL,
        registry=_registry(Recorder()),
    )

    assert reduced.steps[0].resolved is not None
    assert reduced.steps[0].resolved.value.games_per_position == 1
    assert full.steps[0].resolved is not None
    assert full.steps[0].resolved.value.games_per_position == 4
    # Two scales are two series, so they must never resume into each other.
    assert reduced.sha256 != full.sha256


def test_the_sweeps_checkpoint_replaces_what_a_selection_carries(
    tmp_path: Path,
) -> None:
    """One entry point taking one checkpoint, not a batch of unrelated runs."""

    selection = _selection(
        tmp_path,
        body='[model]\nrun_path = "some-other-run"\n',
    )
    plan = resolve_suite(
        _suite(
            model={"run_path": "training-v4", "checkpoint": "step-00008000.pt"},
            checkpoint_label="v4-step-8000",
            benchmarks={"alpha": {"config": str(selection)}},
        ),
        registry=_registry(Recorder()),
    )

    resolved = plan.steps[0].resolved
    assert resolved is not None
    assert resolved.value.model.run_path == Path("training-v4")
    assert resolved.value.checkpoint_label == "v4-step-8000"
    # Provenance says where the checkpoint came from, so a committed result can
    # still explain itself.
    assert 'model.run_path="training-v4"' in resolved.provenance.overrides
    assert resolved.provenance.source is not None


def test_recording_is_decided_per_benchmark_within_one_sweep(
    tmp_path: Path,
) -> None:
    """A sweep commits the readings it meant to keep and no others."""

    recorder = Recorder()
    plan = resolve_suite(
        _suite(
            benchmarks={
                "alpha": {"config": str(_selection(tmp_path)), "record": True},
                "beta": {
                    "config": str(_selection(tmp_path, "beta.toml")),
                    "record": False,
                },
            }
        ),
        registry=_registry(recorder),
    )
    store = FakeStore()
    run = run_suite(
        plan,
        sweep_root=tmp_path / "sweep",
        # Reached only for the steps whose readings this sweep commits.
        store=cast("Any", store),
        detail=cast("Any", object()),
    )

    by_name = {outcome.name: outcome for outcome in run.outcomes}
    assert _result(by_name["alpha"]).recorded is True
    assert _result(by_name["beta"]).recorded is False
    assert plan.recording == ("alpha",)


def test_record_only_overrides_what_the_selection_decided(tmp_path: Path) -> None:
    """What #146 needs: sweep everything, commit one reading."""

    plan = resolve_suite(
        _suite(
            benchmarks={
                "alpha": {"config": str(_selection(tmp_path)), "record": True},
                "beta": {
                    "config": str(_selection(tmp_path, "beta.toml")),
                    "record": True,
                },
            }
        ),
        record_only=["beta"],
        registry=_registry(Recorder()),
    )

    assert plan.recording == ("beta",)


def test_no_record_leaves_the_whole_sweep_uncommitted(tmp_path: Path) -> None:
    plan = resolve_suite(
        _suite(benchmarks={"alpha": {"config": str(_selection(tmp_path))}}),
        no_record=True,
        registry=_registry(Recorder()),
    )

    assert plan.recording == ()


def test_a_recording_sweep_without_a_store_fails_before_it_runs(
    tmp_path: Path,
) -> None:
    plan = resolve_suite(
        _suite(benchmarks={"alpha": {"config": str(_selection(tmp_path))}}),
        registry=_registry(Recorder()),
    )

    with pytest.raises(SuiteError, match="needs a results store"):
        run_suite(plan, sweep_root=tmp_path / "sweep")


def test_the_games_one_step_generates_reach_the_step_that_reads_them(
    tmp_path: Path,
) -> None:
    """The dependency is a payload, not an ordering suggestion."""

    recorder = Recorder(games=(_game_record(),))
    plan = resolve_suite(
        _suite(
            benchmarks={
                "alpha": {"config": str(_selection(tmp_path))},
                "gamma": {"record": False},
            }
        ),
        no_record=True,
        registry=_registry(recorder),
    )
    sweep_root = tmp_path / "sweep"

    run = run_suite(plan, sweep_root=sweep_root)

    assert recorder.calls == ["alpha", "gamma"]
    assert all(outcome.status is StepStatus.COMPLETED for outcome in run.outcomes)
    assert (sweep_root / "alpha-games.json").is_file()
    gamma = next(item for item in run.outcomes if item.name == "gamma")
    assert len(_result(gamma).games) == 1


def test_a_failed_step_is_reported_and_the_rest_of_the_sweep_still_runs(
    tmp_path: Path,
) -> None:
    """One broken instrument is not a reason to discard the readings beside it."""

    recorder = Recorder(failing={"alpha"})
    plan = resolve_suite(
        _suite(
            benchmarks={
                "alpha": {"config": str(_selection(tmp_path))},
                "beta": {"config": str(_selection(tmp_path, "beta.toml"))},
            }
        ),
        no_record=True,
        registry=_registry(recorder),
    )

    run = run_suite(plan, sweep_root=tmp_path / "sweep")

    outcomes = {outcome.name: outcome for outcome in run.outcomes}
    assert outcomes["alpha"].status is StepStatus.FAILED
    assert "could not be read" in (outcomes["alpha"].note or "")
    assert outcomes["beta"].status is StepStatus.COMPLETED
    assert [item.name for item in run.failed] == ["alpha"]


def test_a_step_whose_input_failed_is_skipped_rather_than_run(
    tmp_path: Path,
) -> None:
    """It has nothing to read, and a confusing second error helps nobody."""

    recorder = Recorder(failing={"alpha"}, games=(_game_record(),))
    plan = resolve_suite(
        _suite(
            benchmarks={
                "alpha": {"config": str(_selection(tmp_path))},
                "gamma": {"record": False},
            }
        ),
        no_record=True,
        registry=_registry(recorder),
    )

    run = run_suite(plan, sweep_root=tmp_path / "sweep")

    assert recorder.calls == ["alpha"]
    gamma = next(item for item in run.outcomes if item.name == "gamma")
    assert gamma.status is StepStatus.SKIPPED
    assert "nothing to read" in (gamma.note or "")


def _obstruct_games_payload(sweep_root: Path, producer: str) -> None:
    """Make the hand-off's own write fail the way a bad sweep root does.

    A directory where the payload goes is refused by the same rename the store
    performs for any write, so the test exercises the real store rather than a
    stand-in that raises on request.
    """

    (sweep_root / f"{producer}-games.json").mkdir(parents=True)


def test_a_step_whose_games_cannot_be_handed_off_costs_only_its_dependent(
    tmp_path: Path,
) -> None:
    """The sweep root refusing a payload is not a reason to end the sweep."""

    recorder = Recorder(games=(_game_record(),))
    plan = resolve_suite(
        _suite(
            benchmarks={
                "alpha": {"config": str(_selection(tmp_path))},
                "beta": {"config": str(_selection(tmp_path, "beta.toml"))},
                "gamma": {"record": False},
            }
        ),
        no_record=True,
        registry=_registry(recorder),
    )
    sweep_root = tmp_path / "sweep"
    _obstruct_games_payload(sweep_root, "alpha")

    run = run_suite(plan, sweep_root=sweep_root)

    assert recorder.calls == ["alpha", "beta"]
    outcomes = {outcome.name: outcome for outcome in run.outcomes}
    assert outcomes["alpha"].status is StepStatus.COMPLETED
    assert outcomes["beta"].status is StepStatus.COMPLETED
    assert outcomes["gamma"].status is StepStatus.FAILED
    assert "could not hand off its games" in (outcomes["gamma"].note or "")
    assert [item.name for item in run.failed] == ["gamma"]


def test_a_step_whose_games_cannot_be_handed_off_is_still_in_the_ledger(
    tmp_path: Path,
) -> None:
    """Otherwise a resume replays the most expensive step in the sweep."""

    payload = {
        "benchmarks": {
            "alpha": {"config": str(_selection(tmp_path))},
            "gamma": {"record": False},
        }
    }
    first = Recorder(games=(_game_record(),))
    plan = resolve_suite(_suite(**payload), no_record=True, registry=_registry(first))
    sweep_root = sweep_directory(tmp_path / "sweeps", plan)
    _obstruct_games_payload(sweep_root, "alpha")

    run_suite(plan, sweep_root=sweep_root)

    ledger = json.loads((sweep_root / "sweep.json").read_text(encoding="utf-8"))
    statuses = {entry["benchmark"]: entry["status"] for entry in ledger["steps"]}
    assert statuses == {"alpha": "completed", "gamma": "failed"}

    second = Recorder(games=(_game_record(),))
    resumed = run_suite(
        resolve_suite(_suite(**payload), no_record=True, registry=_registry(second)),
        sweep_root=sweep_root,
        resume=True,
    )

    assert "alpha" not in second.calls
    outcomes = {outcome.name: outcome for outcome in resumed.outcomes}
    assert outcomes["alpha"].from_ledger is True
    # The games are gone with the run that generated them, so resuming reports
    # the dependent failing on the payload rather than pretending it can run.
    assert outcomes["gamma"].status is StepStatus.FAILED


def test_a_resumed_sweep_does_not_repeat_what_it_already_read(
    tmp_path: Path,
) -> None:
    """A sweep that dies late keeps what it already read."""

    first = Recorder(failing={"beta"}, games=(_game_record(),))
    payload = {
        "benchmarks": {
            "alpha": {"config": str(_selection(tmp_path))},
            "beta": {"config": str(_selection(tmp_path, "beta.toml"))},
        }
    }
    plan = resolve_suite(_suite(**payload), no_record=True, registry=_registry(first))
    sweep_root = sweep_directory(tmp_path / "sweeps", plan)

    run_suite(plan, sweep_root=sweep_root)
    assert first.calls == ["alpha", "beta"]

    second = Recorder(games=(_game_record(),))
    resumed = run_suite(
        resolve_suite(_suite(**payload), no_record=True, registry=_registry(second)),
        sweep_root=sweep_root,
        resume=True,
    )

    # Only the failed step is retried; the completed one comes off the ledger.
    assert second.calls == ["beta"]
    outcomes = {outcome.name: outcome for outcome in resumed.outcomes}
    assert outcomes["alpha"].from_ledger is True
    assert outcomes["alpha"].status is StepStatus.COMPLETED
    assert outcomes["beta"].from_ledger is False
    assert resumed.failed == ()


def test_a_resumed_sweep_refuses_a_ledger_from_a_different_sweep(
    tmp_path: Path,
) -> None:
    """A reduced sweep must not continue a full one, or another checkpoint's."""

    recorder = Recorder()
    payload = {"benchmarks": {"alpha": {"config": str(_selection(tmp_path))}}}
    plan = resolve_suite(
        _suite(**payload), no_record=True, registry=_registry(recorder)
    )
    sweep_root = tmp_path / "sweep"
    run_suite(plan, sweep_root=sweep_root)

    other = resolve_suite(
        _suite(model={"run_path": "another-run"}, **payload),
        no_record=True,
        registry=_registry(recorder),
    )

    with pytest.raises(SuiteError, match="records a different sweep"):
        run_suite(other, sweep_root=sweep_root, resume=True)


def test_the_ledger_is_written_after_every_step(tmp_path: Path) -> None:
    """Written as the sweep goes, so an interrupted process still leaves it."""

    recorder = Recorder(failing={"beta"})
    plan = resolve_suite(
        _suite(
            benchmarks={
                "alpha": {"config": str(_selection(tmp_path))},
                "beta": {"config": str(_selection(tmp_path, "beta.toml"))},
            }
        ),
        no_record=True,
        registry=_registry(recorder),
    )
    sweep_root = tmp_path / "sweep"

    run_suite(plan, sweep_root=sweep_root)

    ledger = json.loads((sweep_root / "sweep.json").read_text(encoding="utf-8"))
    assert ledger["plan_sha256"] == plan.sha256
    assert ledger["scale"] == "reduced"
    statuses = {entry["benchmark"]: entry["status"] for entry in ledger["steps"]}
    assert statuses == {"alpha": "completed", "beta": "failed"}


def test_two_checkpoints_never_share_a_sweep_directory(tmp_path: Path) -> None:
    """Otherwise a resume would silently continue the wrong checkpoint's sweep."""

    payload = {"benchmarks": {"alpha": {"config": str(_selection(tmp_path))}}}
    registry = _registry(Recorder())
    first = resolve_suite(_suite(**payload), registry=registry)
    second = resolve_suite(
        _suite(model={"checkpoint": "step-00008000.pt"}, **payload),
        registry=registry,
    )

    root = tmp_path / "sweeps"
    assert sweep_directory(root, first) != sweep_directory(root, second)


def test_a_sweep_summary_says_what_ran_and_what_it_cost(tmp_path: Path) -> None:
    recorder = Recorder(failing={"beta"})
    plan = resolve_suite(
        _suite(
            benchmarks={
                "alpha": {"config": str(_selection(tmp_path))},
                "beta": {"config": str(_selection(tmp_path, "beta.toml"))},
            }
        ),
        no_record=True,
        registry=_registry(recorder),
    )

    record = run_suite(plan, sweep_root=tmp_path / "sweep").as_record()

    assert record["scale"] == "reduced"
    assert [entry["benchmark"] for entry in record["steps"]] == ["alpha", "beta"]
    assert record["steps"][0]["measurements"] == 2
    assert record["steps"][1]["note"] is not None
    assert record["seconds"] >= 0.0


def test_record_only_cannot_ask_for_a_step_with_no_result_kind(
    tmp_path: Path,
) -> None:
    """Recording nothing while reporting success would make the ledger lie."""

    with pytest.raises(SuiteError, match="no result kind"):
        resolve_suite(
            _suite(
                benchmarks={
                    "alpha": {"config": str(_selection(tmp_path))},
                    "gamma": {"record": False},
                }
            ),
            record_only=["gamma"],
            registry=_registry(Recorder()),
        )


def test_a_step_can_declare_the_full_sweep_alone(tmp_path: Path) -> None:
    """A benchmark with no reduction worth reading is left out rather than shrunk."""

    payload = {
        "benchmarks": {
            "alpha": {"config": str(_selection(tmp_path))},
            "beta": {
                "config": str(_selection(tmp_path, "beta.toml")),
                "scales": ["full"],
            },
        }
    }
    registry = _registry(Recorder())

    reduced = resolve_suite(
        _suite(**payload), scale=SuiteScale.REDUCED, registry=registry
    )
    full = resolve_suite(_suite(**payload), scale=SuiteScale.FULL, registry=registry)

    assert [step.name for step in reduced.steps] == ["alpha"]
    assert [step.name for step in full.steps] == ["alpha", "beta"]


def test_a_reduction_no_sweep_would_apply_is_refused() -> None:
    """The refusal is the schema's, so no selection is read before it fires."""

    with pytest.raises(ValidationError, match="add the reduced scale"):
        _suite(
            benchmarks={
                "alpha": {
                    "config": "alpha.toml",
                    "scales": ["full"],
                    "reduced": ["games_per_position=1"],
                }
            }
        )


def test_a_step_that_runs_at_no_scale_is_refused() -> None:
    """An emptied scale list drops the step from every sweep and says nothing."""

    with pytest.raises(ValidationError, match="at least 1 item"):
        _suite(benchmarks={"alpha": {"config": "alpha.toml", "scales": []}})


@pytest.mark.parametrize("field", ["overrides", "reduced"])
def test_overrides_on_a_step_with_no_selection_are_refused(
    tmp_path: Path, field: str
) -> None:
    """There is no schema behind them, so nothing would ever read or check them."""

    with pytest.raises(SuiteError, match="no overrides of its own"):
        resolve_suite(
            _suite(
                benchmarks={
                    "alpha": {"config": str(_selection(tmp_path))},
                    "gamma": {"record": False, field: ["nothing=1"]},
                }
            ),
            registry=_registry(Recorder()),
        )


def test_a_scale_that_excludes_every_step_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="runs at the reduced scale"):
        resolve_suite(
            _suite(
                benchmarks={
                    "alpha": {
                        "config": str(_selection(tmp_path)),
                        "scales": ["full"],
                    }
                }
            ),
            scale=SuiteScale.REDUCED,
            registry=_registry(Recorder()),
        )


def test_a_dependent_step_excluded_with_its_producer_is_refused(
    tmp_path: Path,
) -> None:
    """Leaving the reader behind when the writer sits out is a plan error."""

    with pytest.raises(SuiteError, match="which this sweep does not include"):
        resolve_suite(
            _suite(
                benchmarks={
                    "alpha": {
                        "config": str(_selection(tmp_path)),
                        "scales": ["full"],
                    },
                    "beta": {"config": str(_selection(tmp_path, "beta.toml"))},
                    "gamma": {"record": False},
                }
            ),
            scale=SuiteScale.REDUCED,
            registry=_registry(Recorder()),
        )
