"""What a training configuration costs to run.

Training efficiency is a property of a **run**, measured while the run happens.
That is the whole difference from inference efficiency, which is a property of
a checkpoint and belongs in the end-of-run suite: there is no way to re-measure
what a finished run cost, so the instrumentation has to live inside the loop.

Three rules keep the numbers meaning something.

**Positions, not steps.** Sequences are padded, so a step that processed twice
the tokens may have learned from the same number of moves. The headline
denominator is active non-padding positions, and the realized padding fraction
is reported beside it rather than folded in.

**Overhead is identified, not averaged away.** Startup, checkpoint writes,
cadence evaluation, final validation, and the health monitor's own
instrumentation are timed separately and subtracted from the throughput window.
A run that evaluates itself every ten steps is not slower at training, and a
number that says otherwise would make the cadence look like a regression.

**Warmup is excluded.** The first steps pay for lazy kernel compilation and
allocator growth. They are real costs, but they are startup costs, and leaving
them in the window would make a short run look slower than a long one for
reasons that have nothing to do with the configuration.

**Almost nothing breaks the series.** The whole point of this family is to
answer what a model change, a batch change, or a year of drift cost, so the
model architecture, the effective batch, the corpus, the determinism setting,
and the machine are all recorded as *coordinates* rather than as identity. A
report subtracts across them and names whichever moved. Only the benchmark
version identifies the series, because only a changed definition makes the
delta mean nothing. See
``docs/decisions/0021-efficiency-identity-excludes-compared-conditions.md``,
which refines ``0018-workload-scoped-efficiency-series.md``.

The realized sequence-length distribution is not a coordinate either, because
it is an outcome of those choices rather than an input; it is reported as
measurements so a reader can see what the declared conditions produced.
"""

from __future__ import annotations

import logging
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from pydantic import Field, StrictBool, StrictInt
from torch import Tensor

from anthro_chess.config import ConfigModel
from anthro_chess.data import BatchUnit
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    ConfigurationReference,
    DetailReference,
    DetailStore,
    ExecutionRecord,
    Measurement,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    ResultsStoreError,
    build_result,
    execution_reference,
    measurement,
)
from anthro_chess.evaluation.results.metrics import (
    TRAINING_ACTIVE_POSITION_FRACTION,
    TRAINING_ACTIVE_POSITIONS_PER_SECOND,
    TRAINING_COMPILED_GRAPHS,
    TRAINING_OVERHEAD_FRACTION,
    TRAINING_PEAK_DEVICE_MEMORY_BYTES,
    TRAINING_PROCESSED_POSITIONS,
    TRAINING_SECONDS,
    TRAINING_STEP_SECONDS,
    TRAINING_STEP_SYNC_COST_SECONDS,
)
from anthro_chess.evaluation.results.records import canonical_json

TRAINING_EFFICIENCY_VERSION = 1

TRAINING_EFFICIENCY_KIND = "training-efficiency"
TRAINING_EFFICIENCY_BENCHMARK = BenchmarkReference(
    name="training-efficiency",
    version=TRAINING_EFFICIENCY_VERSION,
)

logger = logging.getLogger(__name__)


class TrainingEfficiencyError(ValueError):
    """Raised when training efficiency cannot be measured or recorded."""


class TrainingEfficiencyConfig(ConfigModel):
    """How a run measures what it costs.

    Every field here is a measurement choice rather than a training choice.
    None of them changes what the model learns, and none of them is part of
    series identity: measuring more precisely is not measuring something else.
    """

    #: Whether measurements are written to the results store. Off keeps an
    #: exploratory run out of committed history, matching the cadence posture.
    record: StrictBool = True
    #: Optimizer steps excluded from the throughput window. Lazy kernel
    #: compilation and allocator growth land here rather than depressing the
    #: steady-state figure.
    warmup_steps: StrictInt = Field(default=3, ge=0)
    #: How often a whole logging interval runs the per-micro-batch
    #: synchronization the loop otherwise defers, so the cost of that
    #: synchronization is measured on the real workload rather than estimated.
    #: Zero disables the probe.
    #:
    #: The unit is an interval rather than a step because a step that never
    #: synchronizes does not have a meaningful duration: it returns once the
    #: device work is queued. Only a span bracketed by two synchronizations is
    #: honestly timed, and a drained logging interval is exactly that span.
    #:
    #: Probe intervals are excluded from the throughput window, which is what
    #: sets the default. Alternating would resolve the cost most precisely and
    #: would also spend half the run in the arm the loop does not ship; one
    #: interval in four keeps the headline on most of the run while still
    #: interleaving often enough for drift to affect both arms alike.
    synchronization_probe_every_intervals: StrictInt = Field(default=4, ge=0)
    #: Whether an efficiency result is recorded at every cadence firing as well
    #: as at the end of the run. The intermediate points are what let a report
    #: join quality against a processed-position or wall-clock budget; without
    #: them a whole run contributes one point.
    record_at_cadence: StrictBool = True


@dataclass(frozen=True)
class StepTotals:
    """One drained interval's totals, read with a single synchronization."""

    steps: int
    loss_sum: float
    #: The final step's loss alone. Kept apart from the interval sum so the
    #: reported per-step loss stays the quantity it always was rather than
    #: becoming an interval mean as a side effect of deferring the read.
    final_step_loss_sum: float
    active_positions: int
    window_active_positions: int
    probe_active_positions: int
    finite: bool


class DeferredStepTotals:
    """Loss and active-position totals accumulated on device.

    Every quantity a training step reports used to be read with ``.item()``
    inside the micro-batch loop, which forces a device synchronization on every
    micro-batch regardless of whether anything was about to be logged. These
    stay device tensors until the existing logging interval drains them, which
    is the same trick :mod:`anthro_chess.training.health` already uses for
    optimizer statistics.

    Finiteness is drained with everything else rather than checked per
    micro-batch. The cost is that a non-finite loss is detected at the next
    logging boundary instead of immediately, so up to one logging interval of
    updates is applied from a corrupted gradient. The run fails either way; the
    reported step range says which steps are suspect rather than implying an
    exact one.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        #: MPS has no float64, so the widest accumulator a device supports is
        #: chosen rather than assumed. Losses are order-one values summed over
        #: a logging interval, so float32 is ample where that is the ceiling.
        self._float = torch.float32 if device.type == "mps" else torch.float64
        self._loss_sum = self._zero(self._float)
        self._step_loss_sum = self._zero(self._float)
        self._active = self._zero(torch.long)
        self._window_active = self._zero(torch.long)
        self._probe_active = self._zero(torch.long)
        self._finite = torch.ones((), dtype=torch.bool, device=device)
        self._steps = 0
        self._padded_positions = 0

    def _zero(self, dtype: torch.dtype) -> Tensor:
        return torch.zeros((), dtype=dtype, device=self._device)

    def begin_step(self) -> None:
        """Start one optimizer step's own loss accumulator."""

        self._step_loss_sum = self._zero(self._float)

    @property
    def padded_positions(self) -> int:
        """Return padded positions seen since the last drain.

        Free of any synchronization: a batch's padded extent is its shape,
        which the host already knows.
        """

        return self._padded_positions

    def observe(
        self,
        loss: Tensor,
        active_mask: Tensor,
        *,
        window: bool,
        probe: bool,
    ) -> None:
        """Accumulate one micro-batch without reading anything back."""

        detached = loss.detach().to(self._float)
        self._loss_sum = self._loss_sum + detached
        self._step_loss_sum = self._step_loss_sum + detached
        self._finite = self._finite & torch.isfinite(detached)
        active = active_mask.sum()
        self._active = self._active + active
        if probe:
            self._probe_active = self._probe_active + active
        elif window:
            self._window_active = self._window_active + active
        self._padded_positions += active_mask.numel()

    def end_step(self) -> None:
        """Record that one optimizer step's micro-batches are complete."""

        self._steps += 1

    def synchronize(self) -> None:
        """Force the read-back a deferred interval would otherwise skip.

        This exists for the synchronization probe. It reads one accumulator
        rather than three, because the second and third reads of a drained
        queue cost almost nothing: the first one is the stall being measured.
        """

        self._active.item()

    def drain(self) -> StepTotals:
        """Return the interval's totals and reset, synchronizing once.

        Counts and losses are read as two stacks rather than one. Widening the
        counts to float to share a stack would silently round a position total
        past sixteen million, which a real run reaches within an hour. The
        second read costs almost nothing: the first one drained the queue.
        """

        counts = torch.stack(
            [self._active, self._window_active, self._probe_active]
        ).tolist()
        values = torch.stack(
            [self._loss_sum, self._step_loss_sum, self._finite.to(self._float)]
        ).tolist()
        totals = StepTotals(
            steps=self._steps,
            loss_sum=float(values[0]),
            final_step_loss_sum=float(values[1]),
            active_positions=int(counts[0]),
            window_active_positions=int(counts[1]),
            probe_active_positions=int(counts[2]),
            finite=bool(values[2]),
        )
        self._loss_sum = self._zero(self._float)
        self._step_loss_sum = self._zero(self._float)
        self._active = self._zero(torch.long)
        self._window_active = self._zero(torch.long)
        self._probe_active = self._zero(torch.long)
        self._finite = torch.ones((), dtype=torch.bool, device=self._device)
        self._steps = 0
        self._padded_positions = 0
        return totals


@dataclass(frozen=True)
class TrainingEfficiencySummary:
    """Everything one run measured about what it cost."""

    #: Steps in the throughput window: after warmup, outside the probe arm.
    window_steps: int
    window_intervals: int
    window_active_positions: int
    window_training_seconds: float
    window_padded_positions: int
    processed_positions: int
    #: Training wall clock across every closed interval, overhead removed.
    #: Distinct from the window figure, which excludes warmup and the probe arm.
    training_seconds: float
    run_seconds: float
    startup_seconds: float
    checkpoint_seconds: float
    evaluation_seconds: float
    validation_seconds: float
    instrumentation_seconds: float
    #: The spread of per-step seconds across window intervals. Per *interval*
    #: rather than per step: a deferred step returns when its device work is
    #: queued, so only a drain-bracketed span has an honest duration.
    minimum_interval_step_seconds: float | None
    maximum_interval_step_seconds: float | None
    peak_allocated_memory_bytes: int | None
    peak_driver_memory_bytes: int | None
    probe_steps: int
    probe_seconds: float
    #: Graphs the compiler built while this run held the process. A delta
    #: rather than the process total, so a run measured after another one in
    #: the same interpreter reports its own.
    compiled_graphs: int

    @property
    def active_positions_per_second(self) -> float | None:
        """Return steady-state active positions per second of training."""

        if self.window_steps == 0 or self.window_training_seconds <= 0.0:
            return None
        return self.window_active_positions / self.window_training_seconds

    @property
    def step_seconds(self) -> float | None:
        """Return mean steady-state seconds per optimizer step."""

        if self.window_steps == 0:
            return None
        return self.window_training_seconds / self.window_steps

    @property
    def active_position_fraction(self) -> float | None:
        """Return the share of processed positions that were not padding."""

        if self.window_padded_positions == 0:
            return None
        return self.window_active_positions / self.window_padded_positions

    @property
    def overhead_fraction(self) -> float | None:
        """Return the share of the run spent outside measured training.

        Everything the run did other than optimizer steps: loading data,
        building the model, writing checkpoints, evaluating at a cadence,
        validating at the end, and instrumenting itself.
        """

        if self.run_seconds <= 0.0:
            return None
        return max(self.run_seconds - self.training_seconds, 0.0) / self.run_seconds

    @property
    def probe_step_seconds(self) -> float | None:
        """Return mean seconds per step in the synchronizing arm."""

        if self.probe_steps == 0:
            return None
        return self.probe_seconds / self.probe_steps

    @property
    def step_synchronization_cost_seconds(self) -> float | None:
        """Return added seconds per step from synchronizing per micro-batch.

        Both arms are whole drained intervals, so both are bracketed by a
        synchronization and neither can absorb the other's queued device work.
        They are interleaved through the same run, so a machine that slows down
        partway through affects both rather than only one of them.
        """

        probe = self.probe_step_seconds
        deferred = self.step_seconds
        if probe is None or deferred is None:
            return None
        return probe - deferred

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable summary."""

        return {
            "version": TRAINING_EFFICIENCY_VERSION,
            "window_steps": self.window_steps,
            "window_intervals": self.window_intervals,
            "window_active_positions": self.window_active_positions,
            "window_padded_positions": self.window_padded_positions,
            "window_training_seconds": self.window_training_seconds,
            "processed_positions": self.processed_positions,
            "training_seconds": self.training_seconds,
            "run_seconds": self.run_seconds,
            "overhead": {
                "startup_seconds": self.startup_seconds,
                "checkpoint_seconds": self.checkpoint_seconds,
                "evaluation_seconds": self.evaluation_seconds,
                "validation_seconds": self.validation_seconds,
                "instrumentation_seconds": self.instrumentation_seconds,
            },
            "step_seconds": {
                "mean": self.step_seconds,
                "interval_minimum": self.minimum_interval_step_seconds,
                "interval_maximum": self.maximum_interval_step_seconds,
            },
            "memory": {
                "peak_allocated_bytes": self.peak_allocated_memory_bytes,
                "peak_driver_bytes": self.peak_driver_memory_bytes,
            },
            "synchronization_probe": {
                "probe_steps": self.probe_steps,
                "probe_seconds": self.probe_seconds,
                "probe_step_seconds": self.probe_step_seconds,
                "deferred_step_seconds": self.step_seconds,
                "cost_seconds": self.step_synchronization_cost_seconds,
            },
            "derived": {
                "active_positions_per_second": self.active_positions_per_second,
                "active_position_fraction": self.active_position_fraction,
                "overhead_fraction": self.overhead_fraction,
            },
        }


class TrainingEfficiencyMonitor:
    """Time a training run, charge its overhead, and assemble its result.

    The monitor owns wall clock and device memory only. It never reads a
    tensor, so nothing it does forces a synchronization of its own; the
    position counts it reports arrive from :class:`DeferredStepTotals` when the
    logging interval drains them anyway.

    Its unit is the **interval** rather than the step, because that is the only
    span it can time honestly. Between two drains the host runs ahead of the
    device, so an individual step returns when its work is queued rather than
    when it is done, and the queue is paid off by whichever step happens to
    synchronize next. A drain-bracketed interval has the same start and end
    condition on both sides, so its duration is real.
    """

    def __init__(
        self,
        config: TrainingEfficiencyConfig,
        *,
        device: torch.device,
        started: float | None = None,
    ) -> None:
        self._config = config
        self._device = device
        self._run_started = time.perf_counter() if started is None else started
        self._compiled_graphs_at_start = _compiled_graph_count()
        self._startup_seconds = 0.0
        self._validation_seconds = 0.0
        self._checkpoint_seconds = 0.0
        self._evaluation_seconds = 0.0
        self._instrumentation_seconds = 0.0
        self._first_window_step = 1
        self._step_started: float | None = None
        self._training_seconds = 0.0
        self._interval_open = False
        self._interval_in_window = False
        self._interval_probing = False
        self._interval_seconds = 0.0
        self._interval_steps = 0
        self._closed_intervals = 0
        self._window_intervals = 0
        self._window_steps = 0
        self._window_seconds = 0.0
        self._window_padded_positions = 0
        self._window_active_positions = 0
        self._probe_steps = 0
        self._probe_seconds = 0.0
        self._minimum_interval_step_seconds: float | None = None
        self._maximum_interval_step_seconds: float | None = None
        self._peak_allocated: int | None = None
        self._peak_driver: int | None = None
        self._processed_positions = 0

    @property
    def peak_allocated_memory_bytes(self) -> int | None:
        """Return the largest tensor figure seen so far, if measurable."""

        return self._peak_allocated

    @property
    def peak_driver_memory_bytes(self) -> int | None:
        """Return the largest reserved figure seen so far, if measurable."""

        return self._peak_driver

    @property
    def interval_open(self) -> bool:
        """Return whether a measurement interval is currently accumulating."""

        return self._interval_open

    @property
    def interval_in_window(self) -> bool:
        """Return whether the open interval counts toward steady state."""

        return self._interval_in_window

    @property
    def interval_probing(self) -> bool:
        """Return whether the open interval runs the synchronizing arm."""

        return self._interval_probing

    def begin_optimization(self, *, starting_step: int) -> None:
        """Close the startup window and declare where the run resumes from."""

        self._startup_seconds = time.perf_counter() - self._run_started
        self._first_window_step = starting_step + self._config.warmup_steps + 1
        begin_peak_memory_measurement(self._device)
        self._sample_memory()

    def begin_interval(self, global_step: int) -> None:
        """Open a measurement interval at the step that starts it.

        Window membership and the probe arm are both decided here, from the
        interval's first step. Deciding either later would split an interval
        across two answers, and its positions and its seconds would then be
        attributed to different places.
        """

        if self._interval_open:  # pragma: no cover - misuse
            raise TrainingEfficiencyError("an interval is already open")
        self._interval_open = True
        self._interval_seconds = 0.0
        self._interval_steps = 0
        self._interval_in_window = global_step >= self._first_window_step
        period = self._config.synchronization_probe_every_intervals
        self._interval_probing = (
            self._interval_in_window
            and period > 0
            and self._closed_intervals % period == period - 1
        )

    def begin_step(self) -> None:
        """Start timing one optimizer step inside the open interval."""

        self._step_started = time.perf_counter()

    def charge(self, seconds: float, *, kind: str) -> None:
        """Attribute identified overhead that happened outside a timed step.

        ``kind`` names which bucket the time belongs to, so a report can say
        whether a slow run was checkpointing, evaluating, or training. Nothing
        is subtracted from the interval, because work between two steps was
        never inside one.
        """

        if kind == "checkpoint":
            self._checkpoint_seconds += seconds
        elif kind == "evaluation":
            self._evaluation_seconds += seconds
        elif kind == "instrumentation":
            self._instrumentation_seconds += seconds
        else:  # pragma: no cover - callers pass a known bucket
            raise TrainingEfficiencyError(f"unknown overhead bucket: {kind}")

    def charge_step_overhead(self, seconds: float, *, kind: str) -> None:
        """Attribute overhead that ran inside the current step's timed span.

        Only this form subtracts, and only the health monitor needs it: its
        measurements run between the optimizer step and the end of the step, so
        the span would otherwise charge training for a run measuring itself.
        """

        self.charge(seconds, kind=kind)
        self._interval_seconds -= seconds

    def end_step(self) -> None:
        """Close one optimizer step's wall clock into the open interval."""

        if self._step_started is None:  # pragma: no cover - misuse
            raise TrainingEfficiencyError("end_step was called before begin_step")
        self._interval_seconds += time.perf_counter() - self._step_started
        self._step_started = None
        self._interval_steps += 1
        self._sample_memory()

    def close_interval(self, totals: StepTotals, *, padded_positions: int) -> None:
        """Fold one drained interval into the window or the probe arm."""

        if not self._interval_open:  # pragma: no cover - misuse
            raise TrainingEfficiencyError("no interval is open")
        seconds = max(self._interval_seconds, 0.0)
        steps = self._interval_steps
        self._interval_open = False
        self._closed_intervals += 1
        self._training_seconds += seconds
        self._processed_positions += totals.active_positions
        if steps == 0:  # pragma: no cover - an interval always holds a step
            return
        if self._interval_probing:
            self._probe_steps += steps
            self._probe_seconds += seconds
            return
        if not self._interval_in_window:
            return
        self._window_intervals += 1
        self._window_steps += steps
        self._window_seconds += seconds
        self._window_active_positions += totals.window_active_positions
        self._window_padded_positions += padded_positions
        per_step = seconds / steps
        if (
            self._minimum_interval_step_seconds is None
            or per_step < self._minimum_interval_step_seconds
        ):
            self._minimum_interval_step_seconds = per_step
        if (
            self._maximum_interval_step_seconds is None
            or per_step > self._maximum_interval_step_seconds
        ):
            self._maximum_interval_step_seconds = per_step

    def charge_validation(self, seconds: float) -> None:
        """Attribute the end-of-run validation pass, which is not training."""

        self._validation_seconds += seconds

    def summary(
        self,
        *,
        processed_positions: int | None = None,
    ) -> TrainingEfficiencySummary:
        """Return everything measured so far.

        ``processed_positions`` overrides the monitor's own count for a resumed
        run, whose counter starts from the checkpoint rather than from zero.
        """

        return TrainingEfficiencySummary(
            window_steps=self._window_steps,
            window_intervals=self._window_intervals,
            window_active_positions=self._window_active_positions,
            window_training_seconds=self._window_seconds,
            window_padded_positions=self._window_padded_positions,
            processed_positions=(
                self._processed_positions
                if processed_positions is None
                else processed_positions
            ),
            training_seconds=self._training_seconds,
            run_seconds=time.perf_counter() - self._run_started,
            startup_seconds=self._startup_seconds,
            checkpoint_seconds=self._checkpoint_seconds,
            evaluation_seconds=self._evaluation_seconds,
            validation_seconds=self._validation_seconds,
            instrumentation_seconds=self._instrumentation_seconds,
            minimum_interval_step_seconds=self._minimum_interval_step_seconds,
            maximum_interval_step_seconds=self._maximum_interval_step_seconds,
            peak_allocated_memory_bytes=self._peak_allocated,
            peak_driver_memory_bytes=self._peak_driver,
            probe_steps=self._probe_steps,
            probe_seconds=self._probe_seconds,
            compiled_graphs=_compiled_graph_count() - self._compiled_graphs_at_start,
        )

    def _sample_memory(self) -> None:
        self._peak_allocated = _maximum(
            self._peak_allocated,
            maximum_allocated_memory_bytes(self._device),
        )
        self._peak_driver = _maximum(
            self._peak_driver,
            maximum_driver_memory_bytes(self._device),
        )


def _compiled_graph_count() -> int:
    """Return how many graphs Dynamo has compiled in this process so far.

    Dynamo publishes this nowhere public. The counter is a `Counter`, so a
    Torch that stops keeping the key reads as zero rather than raising, which
    is the right failure for a number reported beside a throughput rather than
    one anything gates on.
    """

    return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])


def allocated_memory_bytes(device: torch.device) -> int | None:
    """Return bytes the allocator currently holds for tensors, if measurable."""

    if device.type == "mps":  # pragma: no cover - no MPS in the CPU suite
        return int(torch.mps.current_allocated_memory())
    if device.type == "cuda":  # pragma: no cover - no CUDA in the CPU suite
        return int(torch.cuda.memory_allocated(device))
    return None


def driver_allocated_memory_bytes(device: torch.device) -> int | None:
    """Return bytes reserved from the driver, if the backend reports it.

    Reserved rather than allocated is what decides whether a configuration
    fits: an allocator that has cached freed blocks still holds them.
    """

    if device.type == "mps":  # pragma: no cover - no MPS in the CPU suite
        return int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":  # pragma: no cover - no CUDA in the CPU suite
        return int(torch.cuda.memory_reserved(device))
    return None


def begin_peak_memory_measurement(device: torch.device) -> None:
    """Scope the allocator's high-water marks to the run about to start.

    CUDA keeps its peaks per process rather than per run, and nothing decays
    them, so without this a second run in one process would inherit the first
    one's peak and a resumed run would report the checkpoint load it happened
    to do. Torch re-seeds each peak to the *current* figure rather than to
    zero, so memory the run is still holding — its parameters, its optimizer
    state — stays counted.

    What remains inside the mark is everything from here to the end of
    optimization, cadence evaluation included. That is deliberate: the figure
    answers whether a configuration fits on a device, and a run that evaluates
    itself has to fit with its evaluation.
    """

    if device.type == "cuda":  # pragma: no cover - no CUDA in the CPU suite
        torch.cuda.reset_peak_memory_stats(device)


def maximum_allocated_memory_bytes(device: torch.device) -> int | None:
    """Return the high-water mark of tensor bytes, if the backend keeps one.

    A step's activations are freed by the time it ends, so the *current* figure
    read at a step boundary misses the backward pass — the moment the run
    actually needs the most memory. Where the allocator keeps its own peak that
    is read instead, because no sampling cadence can catch a transient the
    caller only observes between steps. On a CUDA run of the shipped baseline
    at batch 16 the boundary figure was 23.6 MB against a peak of 134.6 MB.

    MPS exposes no peak, so its current figure is returned and the sampling in
    :class:`TrainingEfficiencyMonitor` remains the approximation it always was.
    """

    if device.type == "cuda":  # pragma: no cover - no CUDA in the CPU suite
        return int(torch.cuda.max_memory_allocated(device))
    return allocated_memory_bytes(device)


def maximum_driver_memory_bytes(device: torch.device) -> int | None:
    """Return the high-water mark of reserved bytes, if the backend keeps one.

    Reserved memory only grows while a run holds its cached blocks, so the
    sampled figure was already close here. It is read from the allocator's own
    peak for the same reason as its allocated counterpart, and so that a run
    calling ``empty_cache`` could not walk the reported peak back down.
    """

    if device.type == "cuda":  # pragma: no cover - no CUDA in the CPU suite
        return int(torch.cuda.max_memory_reserved(device))
    return driver_allocated_memory_bytes(device)


def workload_record() -> dict[str, Any]:
    """Return the declared workload a training-efficiency series is named by.

    Almost nothing, and deliberately. Active positions per second is the same
    quantity whether the model is large or small, the batch wide or narrow, the
    corpus old or regenerated, and the machine fast or slow. Every one of those
    deltas is interpretable, and most of them are the reason to measure at all,
    so freezing any of them into identity would refuse the comparisons this
    family exists to serve.

    Only a change to what the number *means* belongs here, which is the
    benchmark version. See
    ``docs/decisions/0021-efficiency-identity-excludes-compared-conditions.md``.
    """

    return {"benchmark_version": TRAINING_EFFICIENCY_VERSION}


def coordinate_record(
    *,
    dataset_sha256: str,
    loader_configuration_sha256: str,
    model_identity: Mapping[str, Any],
    batch_size: int,
    batch_unit: BatchUnit,
    gradient_accumulation_steps: int,
    determinism: str,
    matmul_precision: str,
    compilation: str,
    profile_phases: bool,
) -> dict[str, Any]:
    """Return the conditions a run was measured under, for attribution.

    These move the number without changing what it measures, so they are
    recorded and diffed rather than digested. A report names whichever of them
    moved, which is what stops a regenerated corpus from reading as a training
    slowdown.
    """

    return {
        "dataset_sha256": dataset_sha256,
        "loader_configuration_sha256": loader_configuration_sha256,
        "model_sha256": sha256(canonical_json(dict(model_identity))).hexdigest(),
        "batch_size": batch_size,
        # Which unit the two batch counts are in. A batch of four games and a
        # batch of four decisions are not the same coordinate, and nothing else
        # in this record tells them apart.
        "batch_unit": batch_unit,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "determinism": determinism,
        "matmul_precision": matmul_precision,
        "compilation": compilation,
        "phase_profiling": profile_phases,
    }


def execution_record(
    coordinates: Mapping[str, Any],
    *,
    device: torch.device,
    precision: str,
) -> ExecutionRecord:
    """Capture what a run was measured under, split by what identifies it."""

    return execution_reference(
        device=device.type,
        device_name=device_name(device),
        precision=precision,
        torch_version=torch.__version__,
        platform_key=platform_key(),
        platform=platform.platform(),
        cpu_threads=torch.get_num_threads() if device.type == "cpu" else None,
        workload=workload_record(),
        coordinates=coordinates,
    )


def platform_key() -> str:
    """Return the coarse machine identity an environment comparison keys on."""

    return f"{platform.system() or 'unknown'}-{platform.machine() or 'unknown'}"


def device_name(device: torch.device) -> str:
    """Return a stable name for the device a run trained on."""

    if device.type == "cuda":  # pragma: no cover - no CUDA in the CPU suite
        return torch.cuda.get_device_name(device)
    machine = platform.machine() or "unknown"
    processor = platform.processor() or machine
    return processor if device.type == "cpu" else f"{machine}-{device.type}"


def efficiency_measurements(
    summary: TrainingEfficiencySummary,
    execution: ExecutionRecord,
) -> tuple[Measurement, ...]:
    """Return the committed measurements for one efficiency reading.

    A quantity the current run cannot measure is omitted rather than reported
    as zero. Device memory on a CPU run is the usual case: no allocator figure
    exists, and a zero would enter the series as an improvement. A run too
    short to open a steady-state window omits the throughput headline the same
    way, and still records the budget axes a quality-versus-time report joins.
    """

    workload = execution.workload_component()
    values: list[Measurement] = []
    pairs: tuple[tuple[str, float | None, int | None], ...] = (
        (
            TRAINING_ACTIVE_POSITIONS_PER_SECOND.identifier,
            summary.active_positions_per_second,
            summary.window_steps,
        ),
        (
            TRAINING_STEP_SECONDS.identifier,
            summary.step_seconds,
            summary.window_steps,
        ),
        (
            TRAINING_PROCESSED_POSITIONS.identifier,
            float(summary.processed_positions),
            None,
        ),
        (TRAINING_SECONDS.identifier, summary.training_seconds, None),
        (
            TRAINING_ACTIVE_POSITION_FRACTION.identifier,
            summary.active_position_fraction,
            summary.window_steps,
        ),
        (TRAINING_OVERHEAD_FRACTION.identifier, summary.overhead_fraction, None),
        (
            TRAINING_PEAK_DEVICE_MEMORY_BYTES.identifier,
            (
                None
                if summary.peak_driver_memory_bytes is None
                else float(summary.peak_driver_memory_bytes)
            ),
            None,
        ),
        (
            TRAINING_STEP_SYNC_COST_SECONDS.identifier,
            summary.step_synchronization_cost_seconds,
            summary.probe_steps or None,
        ),
        (TRAINING_COMPILED_GRAPHS.identifier, float(summary.compiled_graphs), None),
    )
    for identifier, value, sample_size in pairs:
        if value is None:
            continue
        values.append(
            measurement(
                identifier,
                value,
                workload=workload,
                sample_size=sample_size,
            )
        )
    return tuple(values)


def build_efficiency_result(
    summary: TrainingEfficiencySummary,
    *,
    checkpoint: CheckpointReference,
    execution: ExecutionRecord,
    configuration: ConfigurationReference | None = None,
    detail: DetailReference | None = None,
    recorded_at: datetime | None = None,
) -> ResultEnvelope:
    """Assemble one training-efficiency result envelope."""

    try:
        return build_result(
            kind=TRAINING_EFFICIENCY_KIND,
            benchmark=TRAINING_EFFICIENCY_BENCHMARK,
            checkpoint=checkpoint,
            configuration=configuration,
            execution=execution,
            measurements=efficiency_measurements(summary, execution),
            detail=detail,
            recorded_at=recorded_at or datetime.now(tz=UTC),
        )
    except ResultRecordError as error:
        raise TrainingEfficiencyError(str(error)) from error


def record_efficiency(
    summary: TrainingEfficiencySummary,
    *,
    checkpoint: CheckpointReference,
    execution: ExecutionRecord,
    store: ResultsStore | None,
    configuration: ConfigurationReference | None = None,
    detail: DetailReference | None = None,
    recorded_at: datetime | None = None,
) -> tuple[ResultEnvelope, tuple[Path, ...]]:
    """Build one efficiency result and append it when a store is configured."""

    envelope = build_efficiency_result(
        summary,
        checkpoint=checkpoint,
        execution=execution,
        configuration=configuration,
        detail=detail,
        recorded_at=recorded_at,
    )
    if store is None:
        return envelope, ()
    try:
        return envelope, (store.append(envelope),)
    except (ResultRecordError, ResultsStoreError) as error:
        raise TrainingEfficiencyError(str(error)) from error


def write_efficiency_detail(
    detail: DetailStore | None,
    *,
    checkpoint: CheckpointReference,
    recorded_at: datetime,
    payload: Mapping[str, Any],
    paths: list[Path],
) -> DetailReference | None:
    """Write the per-run efficiency breakdown to the machine-local tier."""

    if detail is None:
        return None
    stamp = recorded_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    relative = Path(TRAINING_EFFICIENCY_KIND) / checkpoint.label / f"{stamp}.json"
    reference = detail.write(
        relative,
        dict(payload),
        description=(
            "Overhead breakdown, step-time spread, and synchronization probe."
        ),
    )
    paths.append(detail.root / relative)
    return reference


def render_efficiency(summary: TrainingEfficiencySummary) -> str:
    """Render one run's efficiency summary as text."""

    lines = ["Training efficiency"]
    lines.append(
        f"  steady state:   {_optional(summary.active_positions_per_second)} "
        f"active positions/s over {summary.window_steps} step(s)"
    )
    lines.append(
        f"  step time:      {_optional(summary.step_seconds)}s mean "
        f"(interval min {_optional(summary.minimum_interval_step_seconds)}s, "
        f"max {_optional(summary.maximum_interval_step_seconds)}s)"
    )
    lines.append(
        f"  positions:      {summary.processed_positions} processed, "
        f"{_optional(summary.active_position_fraction)} of the padded extent"
    )
    lines.append(
        f"  wall clock:     {summary.training_seconds:.3f}s training of "
        f"{summary.run_seconds:.3f}s total "
        f"({_optional(summary.overhead_fraction)} overhead)"
    )
    lines.append(
        f"  overhead:       startup {summary.startup_seconds:.3f}s, "
        f"checkpoint {summary.checkpoint_seconds:.3f}s, "
        f"evaluation {summary.evaluation_seconds:.3f}s, "
        f"validation {summary.validation_seconds:.3f}s, "
        f"instrumentation {summary.instrumentation_seconds:.3f}s"
    )
    if summary.peak_driver_memory_bytes is not None:
        lines.append(
            f"  peak memory:    {summary.peak_driver_memory_bytes} bytes reserved, "
            f"{summary.peak_allocated_memory_bytes} allocated"
        )
    cost = summary.step_synchronization_cost_seconds
    if cost is not None:
        lines.append(
            f"  per-step sync:  {cost:+.6f}s per step over "
            f"{summary.probe_steps} probe step(s)"
        )
    if summary.compiled_graphs:
        lines.append(f"  compilation:    {summary.compiled_graphs} graph(s) built")
    return "\n".join(lines) + "\n"


def _optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.6g}"


def _maximum(current: int | None, observed: int | None) -> int | None:
    if current is None:
        return observed
    if observed is None:
        return current
    return max(current, observed)


__all__: Sequence[str] = [
    "TRAINING_EFFICIENCY_BENCHMARK",
    "TRAINING_EFFICIENCY_KIND",
    "TRAINING_EFFICIENCY_VERSION",
    "DeferredStepTotals",
    "StepTotals",
    "TrainingEfficiencyConfig",
    "TrainingEfficiencyError",
    "TrainingEfficiencyMonitor",
    "TrainingEfficiencySummary",
    "allocated_memory_bytes",
    "begin_peak_memory_measurement",
    "build_efficiency_result",
    "coordinate_record",
    "device_name",
    "driver_allocated_memory_bytes",
    "efficiency_measurements",
    "execution_record",
    "maximum_allocated_memory_bytes",
    "maximum_driver_memory_bytes",
    "platform_key",
    "record_efficiency",
    "render_efficiency",
    "workload_record",
    "write_efficiency_detail",
]
