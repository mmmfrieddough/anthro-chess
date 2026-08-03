"""What a benchmark cost, and what two cost readings are safe to compare."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from anthro_chess.evaluation.checkpoint import (
    CHECKPOINT_COST_BENCHMARK,
    CheckpointEvaluationConfig,
)
from anthro_chess.evaluation.cost import (
    BENCHMARK_COST_KIND,
    benchmark_cost_result,
    cost_device,
    cost_workload,
    normalized_configuration,
)
from anthro_chess.evaluation.results import (
    CheckpointReference,
    ConfigurationReference,
)
from anthro_chess.evaluation.results.metrics import BENCHMARK_WALL_CLOCK_SECONDS
from anthro_chess.evaluation.roots import rooted_artifact_path
from anthro_chess.inference import ModelRunnerConfig
from anthro_chess.machine import DATA_ROOT_VARIABLE

CHECKPOINT = CheckpointReference(label="fixture-step-00000100", step=100)
CONFIGURATION = ConfigurationReference(sha256="a" * 64, source="config.toml")


def _config(**overrides: object) -> CheckpointEvaluationConfig:
    return CheckpointEvaluationConfig(**{"pool": Path("artifacts/pool"), **overrides})  # type: ignore[arg-type]


def _workload(config: CheckpointEvaluationConfig) -> dict[str, object]:
    return cost_workload(CHECKPOINT_COST_BENCHMARK, config)


def test_the_checkpoint_is_a_coordinate_rather_than_a_series() -> None:
    """A cost line answers whether the model got more expensive to read."""

    config = _config(model=ModelRunnerConfig(run_path=Path("run-a")))
    other = _config(model=ModelRunnerConfig(run_path=Path("run-b")))

    assert "model" not in normalized_configuration(config)
    assert _workload(config) == _workload(other)


def test_a_rooted_artifact_reads_the_same_as_an_unrooted_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two machines reading one pool are one line, not two three-point ones."""

    monkeypatch.setenv(DATA_ROOT_VARIABLE, str(tmp_path))
    shipped = _config(pool=Path("artifacts/blitz-pool"))
    rooted = _config(
        pool=rooted_artifact_path(tmp_path, Path("artifacts/blitz-pool")),
    )

    assert normalized_configuration(shipped)["pool"] == "blitz-pool"
    assert _workload(shipped) == _workload(rooted)


def test_two_corpora_ending_in_the_same_directory_stay_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final component alone is not a name: several artifacts end alike.

    A ten-times-larger corpus joined onto a smaller one's series would report
    the extra work it costs as a regression.
    """

    monkeypatch.delenv(DATA_ROOT_VARIABLE, raising=False)
    blitz = _config(leakage={"training_normalized": Path("artifacts/blitz/normalized")})
    rapid = _config(leakage={"training_normalized": Path("artifacts/rapid/normalized")})

    assert _workload(blitz) != _workload(rapid)


def test_a_reduction_measures_a_different_amount_of_work() -> None:
    """The sample count is identity here, unlike in every other workload.

    Measuring twice as many games estimates move loss more precisely and costs
    twice as much, so a reduced reading and a full one are two cost series.
    """

    full = _workload(_config())
    reduced = _workload(_config(view={"name": "canonical", "maximum_games": 400}))

    assert full != reduced


def test_the_recorded_cost_reproduces_its_own_series_identity() -> None:
    envelope = benchmark_cost_result(
        benchmark=CHECKPOINT_COST_BENCHMARK,
        checkpoint=CHECKPOINT,
        configuration=CONFIGURATION,
        config=_config(),
        device=torch.device("cpu"),
        seconds=64.28,
    )

    assert envelope.kind == BENCHMARK_COST_KIND
    assert envelope.benchmark == CHECKPOINT_COST_BENCHMARK
    (recorded,) = envelope.measurements
    assert recorded.metric == BENCHMARK_WALL_CLOCK_SECONDS.identifier
    assert recorded.value == 64.28
    assert envelope.execution is not None
    assert envelope.execution.device == "cpu"
    # verify() recomputes the fingerprint from the record alone, which is what
    # lets a reader check a committed cost claim without this machine.
    envelope.verify()


def test_two_benchmarks_never_share_a_cost_series() -> None:
    """The workload names the benchmark, so identical dials stay apart."""

    config = _config()
    mine = cost_workload(CHECKPOINT_COST_BENCHMARK, config)
    theirs = cost_workload(
        CHECKPOINT_COST_BENCHMARK.model_copy(update={"name": "somewhere-else"}),
        config,
    )

    assert mine != theirs


def test_the_configured_selection_decides_where_a_reading_is_attributed() -> None:
    """Resolved the way the runner resolves it, so the two cannot disagree.

    The driver deliberately never loads the runner — the inference benchmark
    times its own model load — so the declared selection is what it has.
    """

    assert cost_device(_config(model=ModelRunnerConfig(device="cpu"))) == torch.device(
        "cpu"
    )


def test_a_runner_the_driver_was_handed_is_where_the_work_actually_ran() -> None:
    """Nothing in a configuration says where a caller loaded its own runner."""

    class _Runner:
        device = torch.device("cpu")

    on_cuda = _config(model=ModelRunnerConfig(device="cuda"))

    assert cost_device(on_cuda, _Runner()) == torch.device("cpu")
