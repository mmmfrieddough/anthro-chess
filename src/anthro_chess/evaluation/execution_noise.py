"""Execution noise: how far a timing reading moves when nothing changed.

Every other floor this project characterizes can be estimated from numbers a
run already computed. This one cannot. The noise in an efficiency metric comes
from the machine — scheduler contention, thermal state, other processes,
allocator and kernel warmth — and no amount of resampling an already-measured
latency will reveal it. It has to be measured by measuring again.

So a characterization here *runs the benchmark*, several times, across several
processes:

**Across processes, because that is what a report compares.** Two efficiency
readings in the store were taken by two invocations of ``anthro eval
inference``, days apart. Each paid its own model load, its own lazy kernel
compilation, and its own allocator growth. A floor built from repeats inside
one process would omit exactly those and license a difference that is only
process luck.

**Within a process, because it says where the noise lives.** Repeating inside
one process measures the part a second process would *not* have to pay for
again. Reported beside the total, it tells a reader whether the cheap form of
replication would have been enough here, which is not knowable in advance and
differs by device.

Cold-start metrics are the exception and are treated as one reading per
process. A second load in the same process reads a warm file cache and an
already-imported Torch, so repeating it in process would report a dispersion
several times narrower than two real cold starts — a floor too narrow is worse
than none, because it licenses noise as a finding.

The floor this produces is valid on the machine that produced it and nowhere
else. ``anthro_chess.evaluation.results.noise`` owns that scoping rule and the
stored record; this module owns the driving.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from anthro_chess.config import ResolvedConfig
from anthro_chess.evaluation.benchmarks import benchmark_registry, run_benchmark
from anthro_chess.evaluation.inference import (
    COLD_START_METRICS,
    INFERENCE_KIND,
    InferenceBenchmarkConfig,
    InferenceBenchmarkError,
)
from anthro_chess.evaluation.results import (
    PROCESS_REPLICATE_METHOD,
    CheckpointReference,
    ExecutionRecord,
    NoiseCharacterization,
    NoiseCharacterizationError,
    build_characterization,
    environment_key,
    process_replicate_floors,
    series_fingerprint,
)
from anthro_chess.evaluation.results.noise import DEFAULT_CONFIDENCE, DEFAULT_COVERAGE

#: Processes to spread the replicates across. The process is the independent
#: replicate a floor's dispersion bound counts, so this is the one lever that
#: narrows a floor honestly rather than by assuming the spread is better known
#: than it is. Three processes leave two degrees of freedom, where the bound
#: sits more than four times above the measured dispersion and every floor is
#: too wide to resolve anything; six leave five, where it sits about twice
#: above. Past that the curve flattens — twelve processes would buy another
#: quarter for twice the model loads — so this is where the cost stops paying.
DEFAULT_PROCESSES = 6

#: Readings per process. Two is enough to see whether repeating in process
#: reproduces the spread; more of them buys precision in the diagnostic rather
#: than in the floor, which is dominated by the process count.
DEFAULT_REPEATS = 2


class ExecutionNoiseError(ValueError):
    """Raised when an execution noise floor cannot be measured or recorded."""


@dataclass(frozen=True)
class ProcessSample:
    """Everything one process measured, and the conditions it measured under."""

    execution: ExecutionRecord
    checkpoint: CheckpointReference
    #: One mapping of metric identifier to value per reading, in the order the
    #: process took them. The first reading is the only one whose cold start is
    #: a cold start.
    readings: tuple[Mapping[str, float], ...]

    def as_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record a worker process prints."""

        return {
            "execution": self.execution.model_dump(mode="json"),
            "checkpoint": self.checkpoint.model_dump(mode="json"),
            "readings": [dict(reading) for reading in self.readings],
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any]) -> ProcessSample:
        """Return the sample a worker process printed."""

        try:
            return cls(
                execution=ExecutionRecord.model_validate(payload["execution"]),
                checkpoint=CheckpointReference.model_validate(payload["checkpoint"]),
                readings=tuple(
                    {str(metric): float(value) for metric, value in reading.items()}
                    for reading in payload["readings"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ExecutionNoiseError(
                f"a replicate process produced an unreadable sample: {error}"
            ) from error


def sample_execution_noise(
    resolved_config: ResolvedConfig[InferenceBenchmarkConfig],
    *,
    repeats: int = DEFAULT_REPEATS,
    run_root: Path | None = None,
) -> ProcessSample:
    """Measure one checkpoint's efficiency ``repeats`` times in this process.

    Nothing is recorded. These readings are evidence about the machine rather
    than about the model, and committing them would put a checkpoint's history
    at the mercy of how many times its noise was characterized.
    """

    if repeats < 1:
        raise ExecutionNoiseError("a process has to take at least one reading")

    inference = benchmark_registry()["inference"]
    readings: list[Mapping[str, float]] = []
    execution: ExecutionRecord | None = None
    checkpoint: CheckpointReference | None = None
    for _ in range(repeats):
        try:
            # Through the driver, with no store: the envelopes are assembled
            # and nothing is appended, which is what this sampler wants.
            result = run_benchmark(inference, resolved_config, run_root=run_root)
        except InferenceBenchmarkError as error:
            raise ExecutionNoiseError(str(error)) from error
        # The invocation also records what it cost, on its own workload. A
        # characterization covers one workload, so folding that reading in here
        # would produce a floor keyed to neither.
        (envelope,) = (item for item in result.envelopes if item.kind == INFERENCE_KIND)
        readings.append({item.metric: item.value for item in envelope.measurements})
        execution, checkpoint = result.execution, result.checkpoint

    assert execution is not None and checkpoint is not None  # repeats >= 1
    return ProcessSample(
        execution=execution,
        checkpoint=checkpoint,
        readings=tuple(readings),
    )


def subprocess_sampler(
    *,
    config_path: Path,
    overrides: Sequence[str] = (),
    repeats: int = DEFAULT_REPEATS,
    timeout_seconds: float | None = None,
) -> Callable[[], ProcessSample]:
    """Return a sampler that measures in a fresh interpreter each time.

    A separate process is the whole point rather than an implementation
    detail: the state that makes the first reading in a process different —
    an unwarmed allocator, uncompiled kernels, an unread checkpoint file — is
    only reset by leaving the process.
    """

    command = [
        sys.executable,
        "-m",
        "anthro_chess",
        "eval",
        "noise",
        "sample",
        "--config",
        str(config_path),
        "--repeats",
        str(repeats),
        "--format",
        "json",
    ]
    for override in overrides:
        command.extend(("--set", override))

    def sample() -> ProcessSample:
        try:
            completed = subprocess.run(  # noqa: S603 - fixed command, no shell
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise ExecutionNoiseError(
                f"a replicate process did not finish within {timeout_seconds}s"
            ) from error
        if completed.returncode != 0:
            raise ExecutionNoiseError(
                "a replicate process failed:\n"
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ExecutionNoiseError(
                f"a replicate process printed no readable sample: {error}"
            ) from error
        return ProcessSample.from_record(payload)

    return sample


def characterize_execution_noise(
    sampler: Callable[[], ProcessSample],
    *,
    processes: int = DEFAULT_PROCESSES,
    source: str,
    coverage: float = DEFAULT_COVERAGE,
    confidence: float = DEFAULT_CONFIDENCE,
    recorded_at: datetime | None = None,
) -> NoiseCharacterization:
    """Return the recordable execution floor for one machine and workload.

    Fewer processes do not produce a narrower floor here, only a less certain
    one: the dispersion bound widens as the degrees of freedom fall, so cutting
    the count trades measurement time for a floor that resolves less. Two
    processes are permitted because a coarse floor beats none, but they leave
    one degree of freedom and a floor an order of magnitude above the spread.
    """

    if processes < 2:
        raise ExecutionNoiseError(
            "an execution floor needs at least two processes; one process "
            "cannot observe what a second one would pay for again"
        )
    samples = [sampler() for _ in range(processes)]
    _require_one_execution(samples)
    replicates = _grouped_readings(samples)
    execution = samples[0].execution
    workload = execution.workload_component()
    try:
        floors = process_replicate_floors(
            replicates,
            fingerprints={
                metric: series_fingerprint(metric, None, workload)
                for metric in replicates
            },
            coverage=coverage,
            confidence=confidence,
        )
        return build_characterization(
            kind="execution",
            method=PROCESS_REPLICATE_METHOD,
            replicates=sum(len(sample.readings) for sample in samples),
            processes=len(samples),
            coverage=coverage,
            confidence=confidence,
            source=source,
            execution=execution,
            floors=floors,
            recorded_at=recorded_at,
        )
    except NoiseCharacterizationError as error:
        raise ExecutionNoiseError(str(error)) from error


def _require_one_execution(samples: Sequence[ProcessSample]) -> None:
    """Reject replicates that did not measure one quantity on one machine.

    A floor covering two workloads or two devices would describe neither. This
    is reachable in practice: a device requested but unavailable falls back,
    and the reading is real but belongs to a different machine.
    """

    workloads = {sample.execution.workload_sha256 for sample in samples}
    if len(workloads) > 1:
        raise ExecutionNoiseError(
            "the replicate processes measured different declared workloads, so "
            "their spread is not one measurement's noise"
        )
    environments = {environment_key(sample.execution) for sample in samples}
    if len(environments) > 1:
        raise ExecutionNoiseError(
            "the replicate processes ran in different environments, so their "
            "spread is not one machine's noise"
        )
    parameters = {sample.checkpoint.parameter_sha256 for sample in samples}
    if len(parameters) > 1:
        raise ExecutionNoiseError(
            "the replicate processes measured different checkpoints, so their "
            "spread includes a model difference"
        )


def _grouped_readings(
    samples: Sequence[ProcessSample],
) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Return each metric's readings, grouped by the process that took them.

    A cold-start metric keeps only each process's first reading, since a
    reload inside one process is not a second cold start.
    """

    metrics = {
        metric
        for sample in samples
        for reading in sample.readings
        for metric in reading
    }
    missing = [
        metric
        for metric in sorted(metrics)
        if any(
            metric not in reading for sample in samples for reading in sample.readings
        )
    ]
    if missing:
        raise ExecutionNoiseError(
            f"not every replicate reported {', '.join(missing)}; the replicates "
            "do not describe one measurement"
        )
    grouped: dict[str, tuple[tuple[float, ...], ...]] = {}
    for metric in sorted(metrics):
        grouped[metric] = tuple(
            tuple(
                reading[metric]
                for reading in (
                    sample.readings[:1]
                    if metric in COLD_START_METRICS
                    else sample.readings
                )
            )
            for sample in samples
        )
    return grouped


__all__ = [
    "DEFAULT_PROCESSES",
    "DEFAULT_REPEATS",
    "ExecutionNoiseError",
    "ProcessSample",
    "characterize_execution_noise",
    "sample_execution_noise",
    "subprocess_sampler",
]
