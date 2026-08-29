"""Execution noise: how far a timing reading moves when nothing changed.

Every other dispersion this project reports is estimated from numbers a run
already computed. This one cannot be. The noise in an efficiency metric comes
from the machine: scheduler contention, thermal state, other processes,
allocator and kernel warmth. No amount of resampling an already-measured
latency will reveal it. It has to be measured by measuring again.

**And the process is what to measure again.** Repeating a measurement inside one
process reproduces it several times more closely than a fresh process does, so
where a process happens to land is nearly the whole of the noise, and measuring
more decisions inside one buys nothing. The inference benchmark therefore *runs
itself again*, in several processes, and those processes pay for themselves
twice: the committed value is their mean, and the floor a later delta is read
against is that mean's own spread rather than one process's.

**In the same session as the reading it qualifies.** A dispersion measured now
describes the machine now, and a machine drifts: #161 measured a floor taken on
a quiet machine licensing four times as many false findings once the machine was
hot, which is further than any arithmetic on the replicates moves anything.

One reading per process, and no repeats inside one. A second reading is cheap
and would widen the estimate rather than firm it up: it shares an allocator, a
warm file cache and a compiled kernel with the first, and #369 measured the
pooled estimator under-weighting the between-process term, the only term here,
whenever a process contributes more than one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from anthro_chess.config import ResolvedConfig
from anthro_chess.evaluation.benchmarks import benchmark_registry, run_benchmark
from anthro_chess.evaluation.inference import (
    INFERENCE_KIND,
    InferenceBenchmarkConfig,
    InferenceBenchmarkError,
)
from anthro_chess.evaluation.results import (
    PROCESS_REPLICATE_METHOD,
    CheckpointReference,
    ExecutionRecord,
    Measurement,
    MetricDispersion,
    NoiseCharacterizationError,
    measured_dispersion,
    pooled_process_reading,
)
from anthro_chess.evaluation.results.noise import DEFAULT_CONFIDENCE


class ExecutionNoiseError(ValueError):
    """Raised when an execution noise floor cannot be measured or recorded."""


@dataclass(frozen=True)
class ProcessSample:
    """What one process measured on one device, and the conditions it used.

    A reading taken on more than one device produces one of these per device,
    because a device is a declared condition of every timing here and two
    devices do not pool.
    """

    execution: ExecutionRecord
    checkpoint: CheckpointReference
    values: tuple[Measurement, ...]

    @classmethod
    def from_measurements(
        cls,
        execution: ExecutionRecord,
        checkpoint: CheckpointReference,
        values: Sequence[Measurement],
    ) -> ProcessSample:
        """Return the sample one device's measurements amount to."""

        return cls(execution=execution, checkpoint=checkpoint, values=tuple(values))

    def as_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record a worker process prints."""

        return {
            "execution": self.execution.model_dump(mode="json"),
            "checkpoint": self.checkpoint.model_dump(mode="json"),
            "values": [value.model_dump(mode="json") for value in self.values],
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any]) -> ProcessSample:
        """Return the sample a worker process printed."""

        try:
            return cls(
                execution=ExecutionRecord.model_validate(payload["execution"]),
                checkpoint=CheckpointReference.model_validate(payload["checkpoint"]),
                values=tuple(
                    Measurement.model_validate(value) for value in payload["values"]
                ),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ExecutionNoiseError(
                f"a replicate process produced an unreadable sample: {error}"
            ) from error


def sample_execution_noise(
    resolved_config: ResolvedConfig[InferenceBenchmarkConfig],
    *,
    run_root: Path | None = None,
) -> tuple[ProcessSample, ...]:
    """Measure one checkpoint's efficiency once in this process.

    Nothing is recorded. The reading is evidence about the machine rather than
    about the model, and committing it would put a checkpoint's history at the
    mercy of how many times its noise was measured.

    Pinning the selection to one process is what stops the recursion: the
    benchmark spawns this sampler, so a sampled run that spawned its own
    replicates would never terminate.
    """

    pinned = ResolvedConfig(
        value=resolved_config.value.model_copy(update={"processes": 1}),
        provenance=resolved_config.provenance,
    )
    try:
        # Through the driver, with no store: the envelopes are assembled and
        # nothing is appended, which is what this sampler wants.
        result = run_benchmark(
            benchmark_registry()["inference"],
            pinned,
            run_root=run_root,
        )
    except InferenceBenchmarkError as error:
        raise ExecutionNoiseError(str(error)) from error
    # The invocation also records what it cost, on its own workload. A
    # dispersion covers one workload, so folding that reading in here would
    # produce one keyed to neither.
    return tuple(
        ProcessSample.from_measurements(
            envelope.execution,
            envelope.checkpoint,
            envelope.measurements,
        )
        for envelope in result.envelopes
        if envelope.kind == INFERENCE_KIND and envelope.execution is not None
    )


def subprocess_sampler(
    selection: InferenceBenchmarkConfig,
    *,
    timeout_seconds: float | None = None,
) -> Callable[[], tuple[ProcessSample, ...]]:
    """Return a sampler that measures ``selection`` in a fresh interpreter.

    A separate process is the whole point rather than an implementation
    detail: the state that makes the first reading in a process different, an
    unwarmed allocator, uncompiled kernels, an unread checkpoint file, is only
    reset by leaving the process.

    The already-resolved selection is handed over rather than the file and
    overrides it came from. A replicate has to measure what the parent
    measured, and re-deriving that from a selection file cannot: the sweep
    replaces a benchmark's model in memory, and a default selection resolves
    against whatever the machine currently points at.
    """

    command = [
        sys.executable,
        "-m",
        "anthro_chess",
        "eval",
        "noise",
        "sample",
        "--selection",
        "-",
        "--format",
        "json",
    ]
    payload = json.dumps(selection.model_dump(mode="json"))
    # A fresh interpreter picks its own intra-op thread count, and the count is
    # part of the environment a reading declares, so a parent that pinned one
    # would refuse its own replicates for measuring a different machine. It also
    # changes what a host reading measures, which is the substantive half.
    environment = dict(os.environ, OMP_NUM_THREADS=str(torch.get_num_threads()))

    def sample() -> tuple[ProcessSample, ...]:
        try:
            completed = subprocess.run(  # noqa: S603 - fixed command, no shell
                command,
                input=payload,
                env=environment,
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
            sampled = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ExecutionNoiseError(
                f"a replicate process printed no readable sample: {error}"
            ) from error
        return tuple(ProcessSample.from_record(item) for item in sampled)

    return sample


def measure_pooled_readings(
    own: Sequence[ProcessSample],
    sampler: Callable[[], tuple[ProcessSample, ...]],
    *,
    processes: int,
) -> dict[str, tuple[float, float]]:
    """Return each series' pooled value and that value's own spread.

    ``own`` is the reading being pooled and qualified, and it counts as one of
    them: both halves have to describe the reading they travel with, not a
    neighbouring measurement of the same thing.

    Fewer processes give a wider floor and a noisier value, so cutting the count
    trades measurement time for a reading that resolves less. Two are permitted
    because a coarse floor beats none, but they leave one degree of freedom and
    a floor an order of magnitude above the spread.
    """

    if processes < 2:
        raise ExecutionNoiseError(
            "an execution reading needs at least two processes; one process "
            "cannot observe what a second one would pay for again"
        )
    rounds = [tuple(own)]
    for _ in range(processes - 1):
        rounds.append(sampler())
        # Checked as each arrives rather than at the end: a replicate that
        # measured something else makes every later one wasted work, and each is
        # a whole benchmark run.
        _require_one_measurement(rounds)
    readings = [_readings(round_) for round_ in rounds]
    try:
        return {
            fingerprint: pooled_process_reading(
                [reading[fingerprint] for reading in readings]
            )
            for fingerprint in sorted(readings[0])
        }
    except NoiseCharacterizationError as error:
        raise ExecutionNoiseError(str(error)) from error


def execution_dispersion_record(
    dispersion: float,
    *,
    processes: int,
    source: str,
    confidence: float = DEFAULT_CONFIDENCE,
) -> MetricDispersion:
    """Return the record a reading stores for what its replicates measured.

    No ``units``: this spread is a property of the machine rather than of a
    sample that could be enlarged, so no pool size follows from it.
    """

    try:
        return measured_dispersion(
            dispersion,
            degrees_of_freedom=processes - 1,
            confidence=confidence,
            source=source,
            estimator=PROCESS_REPLICATE_METHOD,
        )
    except NoiseCharacterizationError as error:
        raise ExecutionNoiseError(str(error)) from error


def _require_one_measurement(rounds: Sequence[Sequence[ProcessSample]]) -> None:
    """Reject replicates that did not measure one quantity on one model.

    The fingerprint set covers the workload and the device together, since a
    device requested but unavailable falls back and is recorded as the device
    that actually ran. The parameter digest is what a mis-resolved checkpoint
    would move, and the environment is what is left once both agree.
    """

    series = [frozenset(_readings(round_)) for round_ in rounds]
    if any(found != series[0] for found in series):
        raise ExecutionNoiseError(
            "the replicate processes measured different series, so their "
            "spread is not one measurement's noise"
        )
    conditions = [_conditions(round_) for round_ in rounds]
    if any(found != conditions[0] for found in conditions):
        raise ExecutionNoiseError(
            "the replicate processes ran in different environments, so their "
            "spread is not one machine's noise"
        )
    parameters = {
        sample.checkpoint.parameter_sha256 for round_ in rounds for sample in round_
    }
    if len(parameters) > 1:
        raise ExecutionNoiseError(
            "the replicate processes measured different checkpoints, so their "
            "spread includes a model difference"
        )


def _conditions(samples: Sequence[ProcessSample]) -> list[tuple[str, str]]:
    """Return each device's machine coordinates, ordered so rounds compare.

    The declared workload already separates two devices, so this is what is
    left: a host with unlike accelerators can hand two processes the same
    declared device and a different piece of silicon.
    """

    return sorted(
        (sample.execution.workload_sha256, repr(sample.execution.environment()))
        for sample in samples
    )


def _readings(samples: Sequence[ProcessSample]) -> dict[str, float]:
    """Return one process's whole reading, across every device it measured."""

    return {
        value.fingerprint: value.value for sample in samples for value in sample.values
    }


__all__ = [
    "ExecutionNoiseError",
    "ProcessSample",
    "execution_dispersion_record",
    "measure_pooled_readings",
    "sample_execution_noise",
    "subprocess_sampler",
]
