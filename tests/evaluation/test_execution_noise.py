"""Measuring what a timing reading costs in noise, beside the reading itself."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

import anthro_chess.evaluation.execution_noise as execution_noise_module
from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.evaluation.execution_noise import (
    ExecutionNoiseError,
    ProcessSample,
    execution_dispersion_record,
    measure_process_readings,
    sample_execution_noise,
    subprocess_sampler,
)
from anthro_chess.evaluation.inference import (
    InferenceBenchmarkConfig,
    LatencyWorkloadConfig,
    ThroughputWorkloadConfig,
)
from anthro_chess.evaluation.results import (
    PROCESS_REPLICATE_METHOD,
    CheckpointReference,
    ExecutionRecord,
    dispersion_bound,
    execution_reference,
    measurement,
    process_dispersion,
    replicate_dispersion,
)
from anthro_chess.evaluation.results.metrics import (
    INFERENCE_BATCH_THROUGHPUT,
    INFERENCE_MODEL_LOAD_SECONDS,
    INFERENCE_MOVE_LATENCY_BY_PERCENTILE,
)
from anthro_chess.inference import ModelRunnerConfig
from anthro_chess.runtime import RuntimeConfig

LATENCY = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
THROUGHPUT = INFERENCE_BATCH_THROUGHPUT.identifier
LOAD = INFERENCE_MODEL_LOAD_SECONDS.identifier


def _execution(*, device_name: str = "fixture-laptop") -> ExecutionRecord:
    return execution_reference(
        device="cpu",
        device_name=device_name,
        precision="float32",
        torch_version="2.7.0",
        platform_key="Fixture-x86",
        platform="fixture-1.2.3",
        cpu_threads=8,
        workload={"latency_reference_plies": 40},
    )


def _sample(
    reading: dict[str, float],
    *,
    device_name: str = "fixture-laptop",
    weights: str = "a" * 64,
) -> ProcessSample:
    execution = _execution(device_name=device_name)
    return ProcessSample(
        execution=execution,
        checkpoint=CheckpointReference(
            label="fixture-step-00000001",
            step=1,
            parameter_sha256=weights,
        ),
        values=tuple(
            measurement(metric, value, workload=execution.workload_component())
            for metric, value in reading.items()
        ),
    )


def _series(metric: str) -> str:
    """Return the fingerprint the fixture execution gives one metric."""

    (found,) = (
        value for value in _sample({metric: 0.0}).values if value.metric == metric
    )
    return found.fingerprint


def _sampler(
    samples: Sequence[ProcessSample],
) -> Callable[[], tuple[ProcessSample, ...]]:
    """Return a sampler that hands back one prepared sample per process."""

    remaining = list(samples)

    def sample() -> tuple[ProcessSample, ...]:
        return (remaining.pop(0),)

    return sample


def _spreads(
    own: ProcessSample,
    others: Sequence[ProcessSample],
    *,
    processes: int,
) -> dict[str, float]:
    """Return each series' spread the way the benchmark derives it."""

    readings = measure_process_readings((own,), _sampler(others), processes=processes)
    return {
        fingerprint: process_dispersion(values)
        for fingerprint, values in readings.items()
    }


def test_the_reading_being_qualified_is_one_of_the_replicates() -> None:
    """The spread has to describe the reading it travels with.

    Measuring it over neighbouring readings alone would describe a different
    set of processes than the one that produced the number being floored.
    """

    own = _sample({LATENCY: 10.0})
    others = [_sample({LATENCY: 12.0}), _sample({LATENCY: 14.0})]

    spreads = _spreads(own, others, processes=3)

    assert spreads == {
        _series(LATENCY): pytest.approx(replicate_dispersion([10.0, 12.0, 14.0]))
    }


def test_a_metric_the_processes_did_not_separate_is_measured_at_zero() -> None:
    """The measurement says so; withholding the floor is the recorder's job.

    A spread of zero is what these processes observed, and it stays in the
    reading's diagnostics for that reason. What must not follow is a stored
    floor of zero, which would clear every later delta on the metric.
    """

    own = _sample({LATENCY: 10.0, THROUGHPUT: 40.0})
    others = [
        _sample({LATENCY: 12.0, THROUGHPUT: 40.0}),
        _sample({LATENCY: 14.0, THROUGHPUT: 40.0}),
    ]

    spreads = _spreads(own, others, processes=3)

    assert spreads[_series(THROUGHPUT)] == 0.0
    with pytest.raises(ExecutionNoiseError, match="no bound to compute"):
        execution_dispersion_record(
            spreads[_series(THROUGHPUT)], processes=3, source="fixture replicates"
        )


def test_the_bound_counts_the_processes_behind_the_spread() -> None:
    own = _sample({LATENCY: 10.0})
    others = [_sample({LATENCY: 12.0}), _sample({LATENCY: 14.0})]

    spreads = _spreads(own, others, processes=3)
    record = execution_dispersion_record(
        spreads[_series(LATENCY)],
        processes=3,
        source="fixture replicates",
    )

    # Three processes, and therefore two degrees of freedom.
    assert record.bound == pytest.approx(
        dispersion_bound(record.value, degrees_of_freedom=2)
    )
    assert record.estimator == PROCESS_REPLICATE_METHOD


def test_more_processes_narrow_the_floor_a_thin_estimate_widens() -> None:
    """The process count is the only honest lever on a floor's width.

    The same spread measured across more processes supports a tighter bound,
    so the floor falls without anything about the machine having changed. This
    is what the default process count is chosen against.
    """

    readings = (10.0, 12.0, 14.0, 10.0, 12.0, 14.0)
    bounds = []
    for processes in (3, 6):
        samples = [_sample({LATENCY: readings[index]}) for index in range(processes)]
        spreads = _spreads(samples[0], samples[1:], processes=processes)
        bounds.append(
            execution_dispersion_record(
                spreads[_series(LATENCY)],
                processes=processes,
                source="fixture replicates",
            ).bound
        )

    assert bounds[1] < bounds[0]


def test_one_process_cannot_measure_execution_noise() -> None:
    with pytest.raises(ExecutionNoiseError, match="at least two processes"):
        _spreads(_sample({LATENCY: 10.0}), [], processes=1)


def test_replicates_from_two_machines_are_not_one_machines_noise() -> None:
    samples = [
        _sample({LATENCY: 10.0}, device_name="fixture-laptop"),
        _sample({LATENCY: 30.0}, device_name="fixture-workstation"),
    ]

    with pytest.raises(ExecutionNoiseError, match="different environments"):
        _spreads(samples[0], samples[1:], processes=2)


def test_replicates_of_two_checkpoints_include_a_model_difference() -> None:
    samples = [
        _sample({LATENCY: 10.0}, weights="a" * 64),
        _sample({LATENCY: 30.0}, weights="b" * 64),
    ]

    with pytest.raises(ExecutionNoiseError, match="different checkpoints"):
        _spreads(samples[0], samples[1:], processes=2)


def test_a_metric_missing_from_one_replicate_is_refused() -> None:
    samples = [
        _sample({LATENCY: 10.0, THROUGHPUT: 40.0}),
        _sample({LATENCY: 11.0}),
    ]

    with pytest.raises(ExecutionNoiseError, match="different series"):
        _spreads(samples[0], samples[1:], processes=2)


def test_a_sample_survives_the_round_trip_a_worker_process_makes() -> None:
    sample = _sample({LATENCY: 10.0, THROUGHPUT: 40.0})

    restored = ProcessSample.from_record(json.loads(json.dumps(sample.as_record())))

    assert restored == sample


def test_an_unreadable_sample_is_an_error_rather_than_a_crash() -> None:
    with pytest.raises(ExecutionNoiseError, match="unreadable sample"):
        ProcessSample.from_record({"values": []})


def test_the_subprocess_sampler_measures_in_a_fresh_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command records nothing, prints readings, and measures what it is given.

    The selection is handed over already resolved rather than as the file it
    came from: a replicate has to measure the checkpoint the parent loaded, and
    a sweep replaces a benchmark's model in memory rather than through the
    overrides a file could be re-read with.
    """

    sample = _sample({LATENCY: 10.0})
    seen: list[tuple[list[str], object]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        seen.append((command, kwargs.get("input")))
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=json.dumps([sample.as_record()]),
            stderr="",
        )

    monkeypatch.setattr(execution_noise_module.subprocess, "run", fake_run)
    selection = InferenceBenchmarkConfig(
        model=ModelRunnerConfig(checkpoint_path=Path("/pinned/step-00000200.pt")),
        processes=1,
    )

    measured = subprocess_sampler(selection)()

    assert measured == (sample,)
    ((command, piped),) = seen
    assert command[:2] == [sys.executable, "-m"]
    assert command[2:8] == [
        "anthro_chess",
        "eval",
        "noise",
        "sample",
        "--selection",
        "-",
    ]
    assert "--format" in command and command[command.index("--format") + 1] == "json"
    assert isinstance(piped, str)
    assert json.loads(piped)["model"]["checkpoint_path"] == "/pinned/step-00000200.pt"


def test_a_failed_replicate_process_reports_what_it_printed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            returncode=2,
            stdout="",
            stderr="anthro eval noise sample: no checkpoint is selected",
        )

    monkeypatch.setattr(execution_noise_module.subprocess, "run", fake_run)

    with pytest.raises(ExecutionNoiseError, match="no checkpoint is selected"):
        subprocess_sampler(InferenceBenchmarkConfig())()


def test_sampling_measures_repeatedly_and_records_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inference_run: Callable[..., Path],
) -> None:
    """The reading is evidence about the machine, not about the model."""

    checkpoint = inference_run(tmp_path / "run", seed=11)
    resolved = ResolvedConfig(
        value=InferenceBenchmarkConfig(
            model=ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu"),
            runtime=RuntimeConfig(seed=7),
            latency=LatencyWorkloadConfig(
                reference_plies=4,
                decisions=2,
                warmup_decisions=0,
                seed="test-latency",
            ),
            throughput=ThroughputWorkloadConfig(
                serving_batch_size=2,
                compute_batch_size=4,
                history_plies=4,
                batches=1,
                warmup_batches=0,
                seed="test-throughput",
            ),
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )
    monkeypatch.setenv("ANTHRO_CHESS_RESULTS_ROOT", str(tmp_path / "results"))

    samples = sample_execution_noise(resolved)

    reading = {
        value.metric: value.value for sample in samples for value in sample.values
    }
    assert reading[LATENCY] > 0.0
    assert not (tmp_path / "results").exists()
