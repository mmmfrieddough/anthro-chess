"""Execution noise: how far a timing reading moves when nothing changed.

Every other dispersion this project reports is estimated from numbers a run
already computed. This one cannot be. The noise in an efficiency metric comes
from the machine — scheduler contention, thermal state, other processes,
allocator and kernel warmth — and no amount of resampling an already-measured
latency will reveal it. It has to be measured by measuring again.

So the inference benchmark *runs itself again*, in several processes, and
reports the spread across them beside its own value. That is the whole reason
this module exists as something other than a bootstrap.

**Across processes, because that is what a report compares.** Two efficiency
readings in the store were taken by two invocations of ``anthro eval
inference``, days apart. Each paid its own model load, its own lazy kernel
compilation, and its own allocator growth. A dispersion built from repeats
inside one process would omit exactly those and license a difference that is
only process luck.

**In the same session as the reading it qualifies.** A dispersion measured now
describes the machine now, and a machine drifts: #161 measured a floor taken on
a quiet machine licensing four times as many false findings once the machine was
hot, which is further than any arithmetic on the replicates moves anything.

One reading per process, and no repeats inside one. A second reading is cheap
and would widen the estimate rather than firm it up: it shares an allocator, a
warm file cache and a compiled kernel with the first, and #369 measured the
pooled estimator under-weighting the between-process term — the only term here —
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
    process_dispersion,
)
from anthro_chess.evaluation.results.noise import DEFAULT_CONFIDENCE


class ExecutionNoiseError(ValueError):
    """Raised when an execution noise floor cannot be measured or recorded."""


@dataclass(frozen=True)
class ProcessSample:
    """Everything one process measured, and the conditions it measured under."""

    execution: ExecutionRecord
    checkpoint: CheckpointReference
    reading: Mapping[str, float]

    @classmethod
    def from_measurements(
        cls,
        execution: ExecutionRecord,
        checkpoint: CheckpointReference,
        values: Sequence[Measurement],
    ) -> ProcessSample:
        """Return the sample one process's measurements amount to.

        Built here rather than at each call site because the process taking the
        reading and the processes qualifying it have to reduce their
        measurements the same way, or their values are not poolable.
        """

        return cls(
            execution=execution,
            checkpoint=checkpoint,
            reading={value.metric: value.value for value in values},
        )

    def as_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record a worker process prints."""

        return {
            "execution": self.execution.model_dump(mode="json"),
            "checkpoint": self.checkpoint.model_dump(mode="json"),
            "reading": dict(self.reading),
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any]) -> ProcessSample:
        """Return the sample a worker process printed."""

        try:
            return cls(
                execution=ExecutionRecord.model_validate(payload["execution"]),
                checkpoint=CheckpointReference.model_validate(payload["checkpoint"]),
                reading={
                    str(metric): float(value)
                    for metric, value in payload["reading"].items()
                },
            )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ExecutionNoiseError(
                f"a replicate process produced an unreadable sample: {error}"
            ) from error


def sample_execution_noise(
    resolved_config: ResolvedConfig[InferenceBenchmarkConfig],
    *,
    run_root: Path | None = None,
) -> ProcessSample:
    """Measure one checkpoint's efficiency once in this process.

    Nothing is recorded. The reading is evidence about the machine rather than
    about the model, and committing it would put a checkpoint's history at the
    mercy of how many times its noise was measured.

    Pinning the selection to one replicate is what stops the recursion: the
    benchmark spawns this sampler, so a sampled run that spawned its own
    replicates would never terminate.
    """

    pinned = ResolvedConfig(
        value=resolved_config.value.model_copy(update={"replicates": 1}),
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
    (envelope,) = (item for item in result.envelopes if item.kind == INFERENCE_KIND)
    return ProcessSample.from_measurements(
        result.execution,
        result.checkpoint,
        envelope.measurements,
    )


def subprocess_sampler(
    selection: InferenceBenchmarkConfig,
    *,
    timeout_seconds: float | None = None,
) -> Callable[[], ProcessSample]:
    """Return a sampler that measures ``selection`` in a fresh interpreter.

    A separate process is the whole point rather than an implementation
    detail: the state that makes the first reading in a process different —
    an unwarmed allocator, uncompiled kernels, an unread checkpoint file — is
    only reset by leaving the process.

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
    # part of the environment a reading declares — so a parent that pinned one
    # would refuse its own replicates for measuring a different machine. It also
    # changes what a CPU reading measures, which is the substantive half.
    environment = dict(os.environ, OMP_NUM_THREADS=str(torch.get_num_threads()))

    def sample() -> ProcessSample:
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
        return ProcessSample.from_record(sampled)

    return sample


def measure_execution_dispersions(
    own: ProcessSample,
    sampler: Callable[[], ProcessSample],
    *,
    processes: int,
) -> dict[str, float]:
    """Return one dispersion per metric, spread across ``processes`` processes.

    ``own`` is the reading being qualified, and it counts as one of them: the
    dispersion has to describe the reading it travels with, not a neighbouring
    measurement of the same thing.

    A metric the processes did not separate is absent from the result rather
    than carrying a zero, so a caller qualifies what was measured and leaves the
    rest bare.

    Fewer processes do not produce a narrower dispersion here, only a less
    certain one: the bound widens as the degrees of freedom fall, so cutting the
    count trades measurement time for a floor that resolves less. Two processes
    are permitted because a coarse floor beats none, but they leave one degree
    of freedom and a floor an order of magnitude above the spread.
    """

    if processes < 2:
        raise ExecutionNoiseError(
            "an execution dispersion needs at least two processes; one process "
            "cannot observe what a second one would pay for again"
        )
    samples = [own]
    for _ in range(processes - 1):
        samples.append(sampler())
        # Checked as each arrives rather than at the end: a replicate that fell
        # back to another device makes every later one wasted work, and each is
        # a whole benchmark run.
        _require_one_execution(samples)
    try:
        dispersions = {
            metric: process_dispersion(values)
            for metric, values in _readings_by_metric(samples).items()
        }
    except NoiseCharacterizationError as error:
        raise ExecutionNoiseError(str(error)) from error
    # A metric every process timed identically is omitted rather than qualified
    # by a floor of zero, which would clear every later delta. What the
    # replicates observed is that these processes did not separate the metric —
    # a clock too coarse for what was timed reads this way at any process count.
    return {metric: value for metric, value in dispersions.items() if value > 0.0}


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
    environments = [sample.execution.environment() for sample in samples]
    if any(environment != environments[0] for environment in environments):
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


def _readings_by_metric(
    samples: Sequence[ProcessSample],
) -> dict[str, tuple[float, ...]]:
    """Return each metric's one value per process, in process order."""

    metrics = {metric for sample in samples for metric in sample.reading}
    missing = [
        metric
        for metric in sorted(metrics)
        if any(metric not in sample.reading for sample in samples)
    ]
    if missing:
        raise ExecutionNoiseError(
            f"not every replicate reported {', '.join(missing)}; the replicates "
            "do not describe one measurement"
        )
    return {
        metric: tuple(sample.reading[metric] for sample in samples)
        for metric in sorted(metrics)
    }


__all__ = [
    "ExecutionNoiseError",
    "ProcessSample",
    "execution_dispersion_record",
    "measure_execution_dispersions",
    "sample_execution_noise",
    "subprocess_sampler",
]
