"""Measuring what a timing reading costs in noise, and what that floor covers."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

import anthro_chess.evaluation.execution_noise as execution_noise_module
from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.evaluation.execution_noise import (
    DEFAULT_REPEATS,
    ExecutionNoiseError,
    ProcessSample,
    characterize_execution_noise,
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
    ResultsStore,
    dispersion_bound,
    execution_reference,
    series_fingerprint,
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

RECORDED_AT = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


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
    readings: Sequence[dict[str, float]],
    *,
    device_name: str = "fixture-laptop",
    weights: str = "a" * 64,
) -> ProcessSample:
    return ProcessSample(
        execution=_execution(device_name=device_name),
        checkpoint=CheckpointReference(
            label="fixture-step-00000001",
            step=1,
            parameter_sha256=weights,
        ),
        readings=tuple(readings),
    )


def _sampler(samples: Sequence[ProcessSample]) -> Callable[[], ProcessSample]:
    """Return a sampler that hands back prepared samples in order."""

    remaining = list(samples)

    def sample() -> ProcessSample:
        return remaining.pop(0)

    return sample


def test_the_floor_spans_processes_rather_than_repeats_within_one() -> None:
    """A reading a report compares is one process's, so the floor must be too.

    These readings agree closely inside each process and disagree between
    them, which is the shape a lazily compiled kernel or a warm allocator
    produces. Building the floor from within-process spread alone would report
    a floor an order of magnitude narrower than the numbers a reader compares.
    """

    samples = [
        _sample([{LATENCY: 10.0}, {LATENCY: 10.1}]),
        _sample([{LATENCY: 12.0}, {LATENCY: 12.1}]),
        _sample([{LATENCY: 14.0}, {LATENCY: 14.1}]),
    ]

    characterization = characterize_execution_noise(
        _sampler(samples),
        processes=3,
        source="fixture replicates",
        recorded_at=RECORDED_AT,
    )

    (entry,) = characterization.floors
    assert entry.metric == LATENCY
    assert entry.within_process_dispersion is not None
    assert entry.within_process_dispersion == pytest.approx(0.1 / 2**0.5, rel=1e-6)
    assert entry.dispersion > 10 * entry.within_process_dispersion
    assert entry.floor > entry.dispersion


def test_the_bound_counts_processes_rather_than_readings() -> None:
    """Readings inside one process are not independent evidence about spread.

    They widen the dispersion, which is why they are taken, but two of them
    share an allocator, a warm file cache and a compiled kernel. Counting them
    as replicates would claim the estimate is firmer than the design supports
    and narrow the bound on the strength of a repeat that cost nothing.
    """

    samples = [
        _sample([{LATENCY: 10.0}, {LATENCY: 10.1}]),
        _sample([{LATENCY: 12.0}, {LATENCY: 12.1}]),
        _sample([{LATENCY: 14.0}, {LATENCY: 14.1}]),
    ]

    characterization = characterize_execution_noise(
        _sampler(samples),
        processes=3,
        source="fixture replicates",
        recorded_at=RECORDED_AT,
    )

    (entry,) = characterization.floors
    assert characterization.replicates == 6
    assert entry.degrees_of_freedom == 2
    assert entry.dispersion_bound == pytest.approx(
        dispersion_bound(entry.dispersion, degrees_of_freedom=2)
    )


def test_more_processes_narrow_the_floor_a_thin_estimate_widens() -> None:
    """The process count is the only honest lever on a floor's width.

    The same spread measured across more processes supports a tighter bound,
    so the floor falls without anything about the machine having changed. This
    is what the default process count is chosen against.
    """

    readings = (10.0, 12.0, 14.0, 10.0, 12.0, 14.0)
    floors = []
    for processes in (3, 6):
        samples = [_sample([{LATENCY: readings[index]}]) for index in range(processes)]
        characterization = characterize_execution_noise(
            _sampler(samples),
            processes=processes,
            source="fixture replicates",
            recorded_at=RECORDED_AT,
        )
        (entry,) = characterization.floors
        floors.append(entry.floor)

    assert floors[1] < floors[0]


def test_a_cold_start_keeps_only_the_first_reading_of_each_process() -> None:
    """A reload inside one process is warm, so it is not a second cold start.

    Including the warm reloads would collapse the dispersion toward zero and
    produce a floor that licenses ordinary machine noise as a finding.
    """

    samples = [
        _sample([{LOAD: 1.0}, {LOAD: 0.1}]),
        _sample([{LOAD: 1.4}, {LOAD: 0.1}]),
    ]

    characterization = characterize_execution_noise(
        _sampler(samples),
        processes=2,
        source="fixture replicates",
        recorded_at=RECORDED_AT,
    )

    (entry,) = characterization.floors
    # Two cold starts, 1.0 and 1.4, rather than four readings around 0.55.
    assert entry.dispersion == pytest.approx(0.2 * 2**0.5, rel=1e-6)
    assert entry.within_process_dispersion is None


def test_the_characterization_records_the_machine_it_is_valid_on() -> None:
    execution = _execution()
    samples = [
        _sample([{LATENCY: 10.0}]),
        _sample([{LATENCY: 11.0}]),
    ]

    characterization = characterize_execution_noise(
        _sampler(samples),
        processes=2,
        source="fixture replicates",
        recorded_at=RECORDED_AT,
    )

    assert characterization.kind == "execution"
    assert characterization.method == PROCESS_REPLICATE_METHOD
    assert characterization.processes == 2
    assert characterization.replicates == 2
    assert characterization.execution == execution
    (entry,) = characterization.floors
    assert entry.fingerprint == series_fingerprint(
        LATENCY,
        None,
        execution.workload_component(),
    )


def test_one_process_cannot_characterize_execution_noise() -> None:
    with pytest.raises(ExecutionNoiseError, match="at least two processes"):
        characterize_execution_noise(
            _sampler([_sample([{LATENCY: 10.0}, {LATENCY: 10.5}])]),
            processes=1,
            source="fixture replicates",
        )


def test_replicates_from_two_machines_are_not_one_machines_noise() -> None:
    samples = [
        _sample([{LATENCY: 10.0}], device_name="fixture-laptop"),
        _sample([{LATENCY: 30.0}], device_name="fixture-workstation"),
    ]

    with pytest.raises(ExecutionNoiseError, match="different environments"):
        characterize_execution_noise(
            _sampler(samples),
            processes=2,
            source="fixture replicates",
        )


def test_replicates_of_two_checkpoints_include_a_model_difference() -> None:
    samples = [
        _sample([{LATENCY: 10.0}], weights="a" * 64),
        _sample([{LATENCY: 30.0}], weights="b" * 64),
    ]

    with pytest.raises(ExecutionNoiseError, match="different checkpoints"):
        characterize_execution_noise(
            _sampler(samples),
            processes=2,
            source="fixture replicates",
        )


def test_a_metric_missing_from_one_replicate_is_refused() -> None:
    samples = [
        _sample([{LATENCY: 10.0, THROUGHPUT: 40.0}]),
        _sample([{LATENCY: 11.0}]),
    ]

    with pytest.raises(ExecutionNoiseError, match="not every replicate reported"):
        characterize_execution_noise(
            _sampler(samples),
            processes=2,
            source="fixture replicates",
        )


def test_the_characterization_is_recordable(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results")
    characterization = characterize_execution_noise(
        _sampler([_sample([{LATENCY: 10.0}]), _sample([{LATENCY: 11.0}])]),
        processes=2,
        source="fixture replicates",
        recorded_at=RECORDED_AT,
    )

    path = store.append_characterization(characterization)

    assert path.exists()
    (recorded,) = store.characterizations()
    assert recorded == characterization


def test_a_sample_survives_the_round_trip_a_worker_process_makes() -> None:
    sample = _sample([{LATENCY: 10.0, THROUGHPUT: 40.0}, {LATENCY: 10.5}])

    restored = ProcessSample.from_record(json.loads(json.dumps(sample.as_record())))

    assert restored == sample


def test_an_unreadable_sample_is_an_error_rather_than_a_crash() -> None:
    with pytest.raises(ExecutionNoiseError, match="unreadable sample"):
        ProcessSample.from_record({"readings": []})


def test_the_subprocess_sampler_measures_in_a_fresh_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command has to be the one that records nothing and prints readings."""

    sample = _sample([{LATENCY: 10.0}])
    seen: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=json.dumps(sample.as_record()),
            stderr="",
        )

    monkeypatch.setattr(execution_noise_module.subprocess, "run", fake_run)

    measured = subprocess_sampler(
        config_path=Path("configs/evaluation/inference-efficiency.toml"),
        overrides=("model.device='cpu'",),
        repeats=2,
    )()

    assert measured == sample
    (command,) = seen
    assert command[:2] == [sys.executable, "-m"]
    assert command[2:8] == [
        "anthro_chess",
        "eval",
        "noise",
        "sample",
        "--config",
        "configs/evaluation/inference-efficiency.toml",
    ]
    assert "--repeats" in command and command[command.index("--repeats") + 1] == "2"
    assert "--format" in command and command[command.index("--format") + 1] == "json"
    assert "--set" in command and command[command.index("--set") + 1] == (
        "model.device='cpu'"
    )


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
        subprocess_sampler(config_path=Path("fixture.toml"))()


def test_sampling_measures_repeatedly_and_records_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inference_run: Callable[..., Path],
) -> None:
    """The readings are evidence about the machine, not about the model."""

    checkpoint = inference_run(tmp_path / "run", seed=11)
    resolved = ResolvedConfig(
        value=InferenceBenchmarkConfig(
            model=ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu"),
            runtime=RuntimeConfig(seed=7),
            latency=LatencyWorkloadConfig(
                reference_plies=4,
                sweep_plies=(4,),
                decisions=2,
                warmup_decisions=0,
                seed="test-latency",
            ),
            throughput=ThroughputWorkloadConfig(
                reference_batch_size=2,
                sweep_batch_sizes=(2,),
                history_plies=4,
                batches=1,
                warmup_batches=0,
                seed="test-throughput",
            ),
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )
    monkeypatch.setenv("ANTHRO_CHESS_RESULTS_ROOT", str(tmp_path / "results"))

    sample = sample_execution_noise(resolved, repeats=2)

    assert len(sample.readings) == 2
    assert LATENCY in sample.readings[0]
    assert sample.readings[0][LATENCY] > 0.0
    assert not (tmp_path / "results").exists()


def test_a_process_takes_at_least_one_reading() -> None:
    resolved = ResolvedConfig(
        value=InferenceBenchmarkConfig(),
        provenance=ConfigProvenance(source=None, overrides=()),
    )

    with pytest.raises(ExecutionNoiseError, match="at least one reading"):
        sample_execution_noise(resolved, repeats=0)


def test_the_default_repeat_count_sees_within_process_variation() -> None:
    """One reading per process would leave the diagnostic unmeasurable."""

    assert DEFAULT_REPEATS >= 2
