"""What a checkpoint costs to play with, and what a model change costs it.

Two jobs, and they want different instruments.

**What it costs to play with** is a wall clock on the device that serves: how
long one move takes with a person waiting for it, how many decisions a batch of
players resolves per second, and how long a process takes to become useful.
Measured end to end through :class:`GameSession`, spanning encoding, batch
construction, model execution, legal masking and sampling, because that is what
a decision actually costs.

**What a model change costs** is counted rather than timed. One 64-token
forward is too little work to register against kernel-launch overhead on an
accelerator, so batch-one latency there is nearly independent of model size and
stays that way even once the launches are collapsed into a graph replay. The
parameters and the operations one decision performs are therefore counted from
the loaded module, where they carry no noise and need no floor, and the clock
is read where it can still separate two models: at a batch large enough to
leave the launch-bound regime, and on the host, which has no launch floor to
hide the arithmetic under.

The host reading earns its place twice over. It is the only place a single
decision's timing tracks the model at all, and it answers whether the engine
needs an accelerator to be playable, which at these sizes it does not.

The workload is synthetic and self-contained: positions come from a seeded
random-legal-move walk rather than from the evaluation pool. Latency depends on
history length and legal-move count, not on which human played the game, so
binding this benchmark to the pool would break its series at every pool
generation for no gain in what it measures.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any, Protocol

import chess
import torch
from pydantic import Field, StrictInt
from torch import Tensor
from torch.utils.flop_counter import FlopCounterMode

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import DecisionContext
from anthro_chess.evaluation.execution import (
    execution_record,
    synchronize,
)
from anthro_chess.evaluation.recording import ResultRecording, checkpoint_reference
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    ExecutionRecord,
    Measurement,
    MetricDispersion,
    ResultEnvelope,
    measurement,
    process_dispersion,
)
from anthro_chess.evaluation.results.metrics import (
    INFERENCE_BATCH_THROUGHPUT,
    INFERENCE_DECISION_GFLOPS,
    INFERENCE_DECISION_OVERHEAD_MS,
    INFERENCE_FIRST_DECISION_SECONDS,
    INFERENCE_FORWARD_THROUGHPUT,
    INFERENCE_MODEL_LOAD_SECONDS,
    INFERENCE_MOVE_LATENCY_BY_PERCENTILE,
    INFERENCE_MOVE_LATENCY_MEAN,
    INFERENCE_PARAMETERS,
    INFERENCE_PEAK_MEMORY_MB,
    LATENCY_PERCENTILES,
)
from anthro_chess.evaluation.selection import CheckpointSelection
from anthro_chess.inference import CheckpointModelRunner
from anthro_chess.inference.config import LATEST_CHECKPOINT
from anthro_chess.inference.runner import ModelRunnerError
from anthro_chess.models import MoveModelBatch
from anthro_chess.runtime import DecisionRuntimeError, GameSession, RuntimeConfig

#: Bumped when what this benchmark times changes. It is part of the recorded
#: workload, so a bump ends every series here rather than letting a redefined
#: quantity continue an existing line.
INFERENCE_BENCHMARK_VERSION = 3

INFERENCE_KIND = "inference-efficiency"
INFERENCE_BENCHMARK = BenchmarkReference(
    name="inference-efficiency",
    version=INFERENCE_BENCHMARK_VERSION,
)

#: Processes one reading is taken in, including the one taking it.
#:
#: The noise in a timing reading is almost entirely *which process took it*:
#: repeating a measurement inside one reproduces it several times more closely
#: than a fresh one does. Measuring more decisions therefore buys nothing, and
#: processes are the only lever. Each costs a whole reading, and pays for it
#: twice: the committed value is their median, which narrows as their square
#: root, and their spread is the floor a later delta is read against.
DEFAULT_PROCESSES = 3

logger = logging.getLogger(__name__)


class _TimedRunner(Protocol):
    """The runner surface latency measurement needs.

    Narrower than the loaded runner on purpose: stage attribution needs the
    prediction call and the device to synchronize on, and nothing else.
    """

    device: torch.device

    def predict(self, context: DecisionContext) -> Tensor:
        """Return raw action logits for the current decision."""


class InferenceBenchmarkError(ValueError):
    """Raised when inference efficiency cannot be measured as configured."""


class LatencyWorkloadConfig(ConfigModel):
    """What batch-one latency is measured at."""

    #: Forty plies is move twenty: a real midgame decision rather than an
    #: opening position whose short history would flatter a full-history model.
    #: Latency is flat in this depth, because the model reads a fixed number of
    #: square tokens and folds history into their channels rather than into the
    #: sequence, so one depth says what every depth says.
    reference_plies: StrictInt = Field(default=40, ge=0)
    decisions: StrictInt = Field(default=100, ge=1)
    #: Excluded from the percentiles. The first decisions pay for lazy kernel
    #: compilation and allocator growth, which is a cold-start cost and is
    #: reported as one.
    warmup_decisions: StrictInt = Field(default=5, ge=0)
    seed: str = Field(default="anthro-inference-latency-v1", min_length=1)


class ThroughputWorkloadConfig(ConfigModel):
    """The two batch sizes a reading is taken at, and what they are for."""

    #: Concurrent players. This is a product figure and a poor detector: most
    #: of a batched decision is host work that does not change when the model
    #: does, so it barely moves between model sizes at any batch size.
    serving_batch_size: StrictInt = Field(default=64, ge=1)
    #: Where the device stops being launch bound and the forward pass starts
    #: reflecting how much arithmetic the model does. Nothing serves at this
    #: width; it is an instrument.
    compute_batch_size: StrictInt = Field(default=256, ge=1)
    history_plies: StrictInt = Field(default=40, ge=0)
    batches: StrictInt = Field(default=30, ge=1)
    warmup_batches: StrictInt = Field(default=3, ge=0)
    seed: str = Field(default="anthro-inference-throughput-v1", min_length=1)


class InferenceBenchmarkConfig(CheckpointSelection):
    """Code-owned schema for ``anthro eval inference``."""

    runtime: RuntimeConfig = RuntimeConfig(seed=0)
    latency: LatencyWorkloadConfig = LatencyWorkloadConfig()
    throughput: ThroughputWorkloadConfig = ThroughputWorkloadConfig()
    #: Processes this reading is taken in, itself included. Deliberately
    #: outside the declared workload: it decides how well the same quantity is
    #: known rather than what was measured. ``1`` takes one process's reading
    #: and reports it without a floor.
    processes: StrictInt = Field(default=DEFAULT_PROCESSES, ge=1)


@dataclass(frozen=True)
class LatencySample:
    """Measured batch-one latency at the declared depth, in milliseconds."""

    history_plies: int
    decisions: int
    percentiles: Mapping[int, float]
    mean_ms: float
    minimum_ms: float
    maximum_ms: float
    #: Attribution of the end-to-end figure, timed inside the very decisions it
    #: attributes rather than beside them. ``context`` assembles the encoded
    #: trajectory the decision is made from and ``predict`` is the prediction
    #: call, which covers batch construction, the forward pass, and the host
    #: copy; the remainder covers legal masking, sampling, and encoding the ply
    #: the chosen action creates, and is derived rather than timed, so it
    #: absorbs measurement overhead instead of hiding it. Because the parts are
    #: cut from one measured window, they cannot sum past it.
    #:
    #: There is deliberately no encode stage. A session encodes one ply as it
    #: advances rather than encoding a history per decision, so the only encode
    #: a decision pays for is that single ply, and it falls inside the
    #: remainder. Naming a stage for it would recreate the reading this
    #: attribution exists to correct.
    context_mean_ms: float
    predict_mean_ms: float

    @property
    def remainder_mean_ms(self) -> float:
        """Return mean milliseconds outside context assembly and prediction."""

        return self.mean_ms - self.context_mean_ms - self.predict_mean_ms

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable latency record."""

        return {
            "history_plies": self.history_plies,
            "decisions": self.decisions,
            "percentiles": {
                f"p{percentile}": value
                for percentile, value in sorted(self.percentiles.items())
            },
            "mean_ms": self.mean_ms,
            "minimum_ms": self.minimum_ms,
            "maximum_ms": self.maximum_ms,
            "context_mean_ms": self.context_mean_ms,
            "predict_mean_ms": self.predict_mean_ms,
            "remainder_mean_ms": self.remainder_mean_ms,
        }


@dataclass(frozen=True)
class ThroughputSample:
    """One batch size's measured decision throughput.

    The whole-decision figure runs the loop the generated benchmarks run:
    collect every pending context, resolve them in one forward pass, then mask
    and sample each result. The forward figure re-runs a batch built once, so
    the difference between them is everything a decision costs outside the
    model.

    Every batch is timed on its own so the reported rates come from the median
    rather than the mean, which stops one descheduled batch carrying either.
    """

    batch_size: int
    batches: int
    batch_median_ms: float
    batch_mean_ms: float
    forward_median_ms: float

    @property
    def decisions_per_second(self) -> float:
        """Return whole batched decisions per second at the median batch."""

        return _rate(self.batch_size, self.batch_median_ms)

    @property
    def forward_decisions_per_second(self) -> float:
        """Return decisions per second through the forward pass alone."""

        return _rate(self.batch_size, self.forward_median_ms)

    @property
    def decision_overhead_ms(self) -> float:
        """Return milliseconds one decision spends outside the model.

        Flat in batch size, unlike the launch cost the forward pass amortizes,
        because it is one decision's own host work: context assembly, batch
        construction, the host copy, legal masking and sampling.
        """

        return (self.batch_median_ms - self.forward_median_ms) / self.batch_size

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable per-batch-size record."""

        return {
            "batch_size": self.batch_size,
            "batches": self.batches,
            "decisions_per_second": self.decisions_per_second,
            "batch_median_ms": self.batch_median_ms,
            "batch_mean_ms": self.batch_mean_ms,
            "forward_decisions_per_second": self.forward_decisions_per_second,
            "forward_median_ms": self.forward_median_ms,
            "decision_overhead_ms": self.decision_overhead_ms,
        }


@dataclass(frozen=True)
class ColdStart:
    """What the process paid before it could serve a move."""

    model_load_seconds: float
    first_decision_seconds: float

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable cold-start record."""

        return {
            "model_load_seconds": self.model_load_seconds,
            "first_decision_seconds": self.first_decision_seconds,
        }


@dataclass(frozen=True)
class ModelCost:
    """What one decision costs the model, counted rather than timed."""

    parameters: int
    decision_gflops: float

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable counted-cost record."""

        return {
            "parameters": self.parameters,
            "decision_gflops": self.decision_gflops,
        }


@dataclass(frozen=True)
class ComputeSample:
    """The forward pass alone, where the device is no longer launch bound."""

    batch_size: int
    batches: int
    forward_median_ms: float
    peak_memory_mb: float | None

    @property
    def forward_decisions_per_second(self) -> float:
        """Return decisions per second through the forward pass alone."""

        return _rate(self.batch_size, self.forward_median_ms)

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable compute-batch record."""

        return {
            "batch_size": self.batch_size,
            "batches": self.batches,
            "forward_median_ms": self.forward_median_ms,
            "forward_decisions_per_second": self.forward_decisions_per_second,
            "peak_memory_mb": self.peak_memory_mb,
        }


@dataclass(frozen=True)
class InferenceDeviceReading:
    """Everything timed on one device, and the conditions it declared.

    A run on an accelerator produces two of these. The host is measured as
    well as the accelerator because a single decision is too small to occupy
    the accelerator, so the host is where a model change shows in a clock, and
    because whether the engine is playable without an accelerator is a question
    nothing else here answers.
    """

    execution: ExecutionRecord
    latency: LatencySample
    serving: ThroughputSample
    compute: ComputeSample | None = None
    cold_start: ColdStart | None = None

    @property
    def device(self) -> str:
        """Return the device type this reading was taken on."""

        return self.execution.device

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable per-device record."""

        return {
            "device": self.device,
            "execution": self.execution.model_dump(mode="json"),
            "latency": self.latency.as_record(),
            "serving": self.serving.as_record(),
            "compute": None if self.compute is None else self.compute.as_record(),
            "cold_start": (
                None if self.cold_start is None else self.cold_start.as_record()
            ),
        }


@dataclass(frozen=True)
class InferenceBenchmarkResult:
    """Everything one inference benchmark measured, and where it was written."""

    checkpoint: CheckpointReference
    cost: ModelCost
    readings: tuple[InferenceDeviceReading, ...]
    processes: int = 1
    #: Each committed metric's spread across those processes, keyed by the
    #: device that took it and the metric, because one reading commits the same
    #: metric on more than one device. Empty at one process.
    dispersions: Mapping[str, float] = field(default_factory=dict)
    envelopes: tuple[ResultEnvelope, ...] = ()
    recorded_paths: tuple[Path, ...] = ()
    detail_paths: tuple[Path, ...] = ()

    @property
    def execution(self) -> ExecutionRecord:
        """Return the serving device's execution record."""

        return self.readings[0].execution

    def reading(self, device: str) -> InferenceDeviceReading | None:
        """Return the reading taken on ``device``, when one was."""

        return next((item for item in self.readings if item.device == device), None)

    def as_record(self) -> dict[str, Any]:
        """Return the full structured result, detail tier included."""

        return {
            "version": INFERENCE_BENCHMARK_VERSION,
            "checkpoint": self.checkpoint.model_dump(mode="json"),
            "cost": self.cost.as_record(),
            "readings": [reading.as_record() for reading in self.readings],
            "processes": self.processes,
            "dispersions": dict(sorted(self.dispersions.items())),
            "recorded": [str(path) for path in self.recorded_paths],
        }


def benchmark_inference(
    resolved_config: ResolvedConfig[InferenceBenchmarkConfig],
    *,
    run_root: Path | None = None,
    recording: ResultRecording,
) -> InferenceBenchmarkResult:
    """Measure one checkpoint's decision cost, latency, and throughput.

    A recording opened without a store measures everything and records nothing,
    which is what an exploratory reading on a busy machine wants: a figure taken
    beside a training run is real but does not belong in the committed history.

    ``config.processes`` is how many processes the reading is taken in, this one
    included. Nothing inside a timing reading can be resampled into a spread, so
    the benchmark takes it again to pool the value and measure the floor at
    once. See :mod:`anthro_chess.evaluation.execution_noise`.
    """

    config = resolved_config.value
    started = time.perf_counter()
    try:
        runner = CheckpointModelRunner.load(config.model, run_root=run_root)
    except ModelRunnerError as error:
        raise InferenceBenchmarkError(str(error)) from error
    synchronize(runner.device)
    model_load_seconds = time.perf_counter() - started

    histories = _HistoryFactory()
    session = GameSession(runner, config=config.runtime)
    first_decision_seconds = _time_first_decision(
        session,
        runner.device,
        histories.history(config.latency.seed, config.latency.reference_plies),
    )
    logger.info(
        "Loaded in %.3fs; first decision took %.3fs",
        model_load_seconds,
        first_decision_seconds,
    )

    cost = _model_cost(runner, config.throughput, histories)
    logger.info(
        "Counted %.3fM parameters and %.4f GFLOP per decision",
        cost.parameters / 1e6,
        cost.decision_gflops,
    )

    readings = [
        _measure_device(
            runner,
            config,
            histories,
            session=session,
            compute_batch_size=config.throughput.compute_batch_size,
            cold_start=ColdStart(
                model_load_seconds=model_load_seconds,
                first_decision_seconds=first_decision_seconds,
            ),
        )
    ]
    if runner.device.type != "cpu":
        host = runner.replicated(torch.device("cpu"))
        readings.append(
            _measure_device(
                host,
                config,
                histories,
                session=GameSession(host, config=config.runtime),
                compute_batch_size=None,
                cold_start=None,
            )
        )

    checkpoint = checkpoint_reference(runner, label=config.checkpoint_label)
    result = InferenceBenchmarkResult(
        checkpoint=checkpoint,
        cost=cost,
        readings=tuple(readings),
    )

    units = _measurement_units(result)
    pooled, spreads = _pool_across_processes(
        config,
        result,
        units,
        checkpoint_path=runner.selection.checkpoint_path,
        processes=config.processes,
    )
    result = replace(
        result,
        processes=config.processes,
        dispersions=_labelled_spreads(units, spreads),
    )

    recorder = recording.measuring(
        checkpoint,
        kind=INFERENCE_KIND,
        benchmark=INFERENCE_BENCHMARK,
    )
    recorder.disperse(_dispersions(spreads, config.processes, result.execution))
    for reading, values in units:
        recorder.add(
            [pooled.get(value.fingerprint, value) for value in values],
            payload=result.as_record,
            description=f"Inference cost and timings on the {reading.device}.",
            slug=reading.device,
            execution=reading.execution,
        )
    return result


def replicate_selection(
    config: InferenceBenchmarkConfig,
    *,
    checkpoint_path: Path,
) -> InferenceBenchmarkConfig:
    """Return the selection a replicate process measures.

    Every dial the parent measured under, with two things pinned. The
    **checkpoint** becomes the absolute file the parent actually loaded, so a
    replicate cannot resolve a different one: the sweep replaces a benchmark's
    model selection in memory rather than through overrides, and a default
    selection resolves against whatever the machine currently points at.
    **Processes** becomes one, which is what stops the recursion.
    """

    return config.model_copy(
        update={
            "model": config.model.model_copy(
                update={
                    "checkpoint_path": checkpoint_path,
                    "run_path": None,
                    "checkpoint": LATEST_CHECKPOINT,
                }
            ),
            "processes": 1,
        }
    )


def _pool_across_processes(
    config: InferenceBenchmarkConfig,
    result: InferenceBenchmarkResult,
    units: Sequence[tuple[InferenceDeviceReading, tuple[Measurement, ...]]],
    *,
    checkpoint_path: Path,
    processes: int,
) -> tuple[dict[str, Measurement], dict[str, float]]:
    """Take this reading again in fresh processes, and pool what they read.

    The processes run one after another rather than together: they contend for
    the device they are timing, so measuring them concurrently would report the
    spread of the contention instead of the spread of the machine.

    A counted quantity reads identically in every process, so its spread is zero
    and it is left without a floor rather than qualified by one.
    """

    if processes < 2:
        return {}, {}
    # Imported here rather than at module scope because the sampler drives this
    # benchmark, so the two modules would otherwise import each other.
    from anthro_chess.evaluation.execution_noise import (
        ExecutionNoiseError,
        ProcessSample,
        measure_process_readings,
        subprocess_sampler,
    )

    own = tuple(
        ProcessSample.from_measurements(reading.execution, result.checkpoint, values)
        for reading, values in units
    )
    logger.info("Taking this reading in %d process(es) and pooling them", processes)
    try:
        readings = measure_process_readings(
            own,
            subprocess_sampler(
                replicate_selection(config, checkpoint_path=checkpoint_path)
            ),
            processes=processes,
        )
    except ExecutionNoiseError as error:
        raise InferenceBenchmarkError(str(error)) from error

    pooled = {
        value.fingerprint: value.model_copy(
            update={"value": median(readings[value.fingerprint])}
        )
        for _, values in units
        for value in values
        if value.fingerprint in readings
    }
    spreads = {
        fingerprint: process_dispersion(values)
        for fingerprint, values in readings.items()
    }
    return pooled, spreads


def _labelled_spreads(
    units: Sequence[tuple[InferenceDeviceReading, tuple[Measurement, ...]]],
    spreads: Mapping[str, float],
) -> dict[str, float]:
    """Return each spread under the device and metric a reader recognizes."""

    return {
        f"{reading.device} {value.metric}": spreads[value.fingerprint]
        for reading, values in units
        for value in values
        if value.fingerprint in spreads
    }


def _dispersions(
    spreads: Mapping[str, float],
    processes: int,
    execution: ExecutionRecord,
) -> dict[str, MetricDispersion]:
    """Return each series' spread, keyed the way the recorder carries one.

    A metric the processes read identically is left bare rather than qualified
    by a floor of zero, which would clear every later delta on it. For a counted
    quantity that is the right answer outright. For a timing it says the clock
    was too coarse for what was timed, and the measured zero still travels in
    the reading's own diagnostics.
    """

    from anthro_chess.evaluation.execution_noise import execution_dispersion_record

    source = f"{processes} process replicates on {execution.environment_label()}"
    return {
        fingerprint: execution_dispersion_record(
            spread,
            processes=processes,
            source=source,
        )
        for fingerprint, spread in spreads.items()
        if spread > 0.0
    }


class _HistoryFactory:
    """Deterministic synthetic histories of a requested ply depth.

    A seeded random-legal-move walk from the standard opening. Reaching a
    terminal position at or before the requested depth restarts the walk from
    the next attempt rather than returning a short history or a position with
    no decision left to make, so a depth means the depth it says and every
    history it returns is one a session can still move from. Successive offsets
    give distinct positions at one depth, so a run measures a spread of
    histories instead of one cached board.

    The end-of-walk check is what makes the deep end of the sweep reachable at
    all: random play thins the board down, and past roughly 250 plies a walk
    that never ran out of legal moves still lands on a dead draw often enough
    that one offset in six would abort the whole benchmark.
    """

    #: Restart budget before a depth is reported unreachable. Generous: a walk
    #: that keeps ending early is a signal about the depth, not bad luck.
    MAXIMUM_ATTEMPTS = 64

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], tuple[chess.Move, ...]] = {}

    def history(self, seed: str, plies: int, offset: int = 0) -> tuple[chess.Move, ...]:
        """Return one deterministic legal history of exactly ``plies`` moves."""

        key = (seed, plies, offset)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        for attempt in range(self.MAXIMUM_ATTEMPTS):
            moves = _random_walk(f"{seed}:{offset}:{attempt}", plies)
            if moves is not None:
                self._cache[key] = moves
                return moves
        raise InferenceBenchmarkError(
            f"could not build a {plies}-ply history from seed {seed!r}; every "
            "walk ended before that depth"
        )


def _random_walk(seed: str, plies: int) -> tuple[chess.Move, ...] | None:
    """Play ``plies`` seeded legal moves, or return ``None`` if the game ends.

    Ending covers the arrival position as well as the walk. A history whose
    final position is over by rule has legal moves on the board but no decision
    a session would make from it, and it would abort the measurement rather
    than merely shortening it.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed_value(seed))
    board = chess.Board()
    moves: list[chess.Move] = []
    for _ in range(plies):
        legal = list(board.legal_moves)
        if not legal:
            return None
        index = int(
            torch.randint(
                len(legal),
                (1,),
                generator=generator,
                dtype=torch.long,
            ).item()
        )
        move = legal[index]
        board.push(move)
        moves.append(move)
    # Claim-free, matching the rule a session ends a game on: a claimable
    # position is still playable until somebody claims it.
    if board.is_game_over():
        return None
    return tuple(moves)


def _seed_value(seed: str) -> int:
    """Derive a stable non-negative generator seed from a workload label.

    Derived from the label rather than drawn, so the same declared workload
    replays the same positions on every machine and in every process.
    """

    return int.from_bytes(sha256(seed.encode()).digest()[:7], "big")


def _time_first_decision(
    session: GameSession,
    device: torch.device,
    history: Sequence[chess.Move],
) -> float:
    """Return seconds the very first decision takes, warmup included."""

    _reset(session, history)
    started = time.perf_counter()
    _decide(session)
    synchronize(device)
    return time.perf_counter() - started


def _measure_latency(
    session: GameSession,
    runner: _TimedRunner,
    histories: _HistoryFactory,
    config: LatencyWorkloadConfig,
) -> LatencySample:
    """Measure end-to-end batch-one decision latency at the declared depth."""

    history_plies = config.reference_plies
    device = runner.device
    for offset in range(config.warmup_decisions):
        _reset(session, histories.history(config.seed, history_plies, offset))
        _decide(session)
        synchronize(device)

    durations: list[float] = []
    context_durations: list[float] = []
    predict_durations: list[float] = []
    for index in range(config.decisions):
        history = histories.history(
            config.seed,
            history_plies,
            config.warmup_decisions + index,
        )
        _reset(session, history)
        # The session's own three steps, cut apart where it cuts them, so the
        # attribution is a division of the measured window rather than a second
        # opinion about it.
        with _measured_decision():
            started = time.perf_counter()
            context = session.decision_context()
            assembled = time.perf_counter()
            logits = runner.predict(context)
            synchronize(device)
            predicted = time.perf_counter()
            session.decide_from_logits(logits)
            synchronize(device)
            finished = time.perf_counter()
        durations.append((finished - started) * 1000.0)
        context_durations.append((assembled - started) * 1000.0)
        predict_durations.append((predicted - assembled) * 1000.0)

    logger.info(
        "Measured %s decision(s) at %s plies: p50 %.1fms",
        config.decisions,
        history_plies,
        _percentile(durations, 50),
    )
    return LatencySample(
        history_plies=history_plies,
        decisions=config.decisions,
        percentiles={
            percentile: _percentile(durations, percentile)
            for percentile in LATENCY_PERCENTILES
        },
        mean_ms=_mean(durations),
        minimum_ms=min(durations),
        maximum_ms=max(durations),
        context_mean_ms=_mean(context_durations),
        predict_mean_ms=_mean(predict_durations),
    )


def _measure_throughput(
    runner: CheckpointModelRunner,
    runtime: RuntimeConfig,
    config: ThroughputWorkloadConfig,
    histories: _HistoryFactory,
    batch_size: int,
) -> ThroughputSample:
    """Measure decisions per second for one declared batch size.

    The headline loop is the runtime half of the one the generated benchmarks
    run: collect every pending context, resolve them in one padded forward
    pass, and hand each result back for masking and sampling. Building the
    batch is part of that, so timing a batch built once and reused would exclude
    the cost this benchmark exists to price.

    Deliberately the runtime half rather than :meth:`ModelPlayer.decide_batch`,
    which drives that loop for the generated benchmarks and then builds a
    validated record of every decision. That record is what playing a *scored*
    game costs, not what playing a move costs, and folding it in would move a
    checkpoint's cost series whenever the game-record schema changed.
    """

    sessions = tuple(GameSession(runner, config=runtime) for _ in range(batch_size))
    histories_used = tuple(
        histories.history(config.seed, config.history_plies, index)
        for index in range(batch_size)
    )

    for _ in range(config.warmup_batches):
        _reset_batch(sessions, histories_used)
        _decide_batch(runner, sessions)
        synchronize(runner.device)

    durations: list[float] = []
    for _ in range(config.batches):
        _reset_batch(sessions, histories_used)
        started = time.perf_counter()
        _decide_batch(runner, sessions)
        synchronize(runner.device)
        durations.append((time.perf_counter() - started) * 1000.0)

    forward_durations = _measure_forward(runner, sessions, histories_used, config)

    sample = ThroughputSample(
        batch_size=batch_size,
        batches=config.batches,
        batch_median_ms=_percentile(durations, 50),
        batch_mean_ms=_mean(durations),
        forward_median_ms=_percentile(forward_durations, 50),
    )
    logger.info(
        "Measured batch %s throughput: %.1f decisions/s (forward alone %.1f)",
        batch_size,
        sample.decisions_per_second,
        sample.forward_decisions_per_second,
    )
    return sample


def _measure_forward(
    runner: CheckpointModelRunner,
    sessions: Sequence[GameSession],
    histories_used: Sequence[Sequence[chess.Move]],
    config: ThroughputWorkloadConfig,
) -> list[float]:
    """Return per-batch milliseconds for the forward pass alone.

    The batch is built once, outside every timed window, and re-run. That is
    the point of this figure rather than an oversight: it isolates launch cost
    from the batch construction and host copy around it, which is what an
    optimization aimed at the kernels needs to see move.
    """

    _reset_batch(sessions, histories_used)
    with _measured_decision():
        contexts = [session.decision_context() for session in sessions]
        batch = MoveModelBatch.from_decision_contexts(contexts, device=runner.device)
        decisions = runner.decision_indices(contexts)
        for _ in range(config.warmup_batches):
            runner.decision_logits(batch, decisions)
        synchronize(runner.device)

        durations: list[float] = []
        for _ in range(config.batches):
            started = time.perf_counter()
            runner.decision_logits(batch, decisions)
            synchronize(runner.device)
            durations.append((time.perf_counter() - started) * 1000.0)
    return durations


def _reset_batch(
    sessions: Sequence[GameSession],
    histories_used: Sequence[Sequence[chess.Move]],
) -> None:
    """Return every session to its declared history, outside the timed window.

    Through the same ``sync_position`` the generated benchmarks call before
    every decision, which takes back the one ply the last batch played and
    keeps the encoded prefix. Rebuilding the session instead re-encoded all
    forty plies to discard thirty-nine of them, and at the wider batch sizes
    that setup cost several times the measurement it was preparing.
    """

    for session, history in zip(sessions, histories_used, strict=True):
        try:
            session.sync_position(moves=tuple(history))
        except DecisionRuntimeError as error:  # pragma: no cover - walks are legal
            raise InferenceBenchmarkError(str(error)) from error


def _decide_batch(
    runner: CheckpointModelRunner,
    sessions: Sequence[GameSession],
) -> None:
    """Resolve one decision for every session through a single forward pass."""

    with _measured_decision():
        contexts = [session.decision_context() for session in sessions]
        logits = runner.predict_batch(contexts)
        for session, row in zip(sessions, logits, strict=True):
            session.decide_from_logits(row)


def _rate(batch_size: int, batch_ms: float) -> float:
    """Return decisions per second from one batch's milliseconds."""

    if batch_ms <= 0.0:  # pragma: no cover - a clock this coarse is unusable
        raise InferenceBenchmarkError(
            "the measured throughput window was too short to time"
        )
    return batch_size / (batch_ms / 1000.0)


def _model_cost(
    runner: CheckpointModelRunner,
    config: ThroughputWorkloadConfig,
    histories: _HistoryFactory,
) -> ModelCost:
    """Count the parameters and the operations one decision performs.

    Counted at batch one because the count is per decision and does not vary
    with the batch it was taken in, and because the operator counter reads
    autograd metadata that serving deliberately strips, so this runs outside
    inference mode and a wider batch would build a graph for nothing.
    """

    batch, decisions = _built_batch(runner, config, histories, 1)
    counter = FlopCounterMode(display=False)
    with counter:
        runner.model.decide_at(batch, decisions)
    return ModelCost(
        parameters=sum(parameter.numel() for parameter in runner.model.parameters()),
        decision_gflops=counter.get_total_flops() / 1e9,
    )


def _built_batch(
    runner: CheckpointModelRunner,
    config: ThroughputWorkloadConfig,
    histories: _HistoryFactory,
    batch_size: int,
) -> tuple[MoveModelBatch, Tensor]:
    """Return one built batch of the declared width, and its decision rows."""

    sessions = tuple(
        GameSession(runner, config=RuntimeConfig(seed=0)) for _ in range(batch_size)
    )
    _reset_batch(
        sessions,
        tuple(
            histories.history(config.seed, config.history_plies, index)
            for index in range(batch_size)
        ),
    )
    contexts = [session.decision_context() for session in sessions]
    try:
        batch = MoveModelBatch.from_decision_contexts(contexts, device=runner.device)
    except (RuntimeError, ValueError) as error:
        raise InferenceBenchmarkError(f"could not build a batch: {error}") from error
    return batch, runner.decision_indices(contexts)


def _measure_device(
    runner: CheckpointModelRunner,
    config: InferenceBenchmarkConfig,
    histories: _HistoryFactory,
    *,
    session: GameSession,
    compute_batch_size: int | None,
    cold_start: ColdStart | None,
) -> InferenceDeviceReading:
    """Measure latency, serving throughput, and compute cost on one device."""

    latency = _measure_latency(session, runner, histories, config.latency)
    serving = _measure_throughput(
        runner,
        config.runtime,
        config.throughput,
        histories,
        config.throughput.serving_batch_size,
    )
    compute = (
        None
        if compute_batch_size is None
        else _measure_compute(runner, config.throughput, histories, compute_batch_size)
    )
    return InferenceDeviceReading(
        execution=_execution_record(
            config,
            runner.device,
            compute_batch_size=compute_batch_size,
        ),
        latency=latency,
        serving=serving,
        compute=compute,
        cold_start=cold_start,
    )


def _measure_compute(
    runner: CheckpointModelRunner,
    config: ThroughputWorkloadConfig,
    histories: _HistoryFactory,
    batch_size: int,
) -> ComputeSample:
    """Time the forward pass alone at the batch where compute is visible.

    Only the forward, because the host work around it does not change with the
    model and would dilute the one figure that does.
    """

    batch, decisions = _built_batch(runner, config, histories, batch_size)
    with _measured_decision():
        for _ in range(config.warmup_batches):
            runner.decision_logits(batch, decisions)
        synchronize(runner.device)
        peak_memory_mb = _peak_memory_mb(runner, batch, decisions)

        durations: list[float] = []
        for _ in range(config.batches):
            started = time.perf_counter()
            runner.decision_logits(batch, decisions)
            synchronize(runner.device)
            durations.append((time.perf_counter() - started) * 1000.0)

    sample = ComputeSample(
        batch_size=batch_size,
        batches=config.batches,
        forward_median_ms=_percentile(durations, 50),
        peak_memory_mb=peak_memory_mb,
    )
    logger.info(
        "Measured the forward pass at batch %s: %.1f decisions/s",
        batch_size,
        sample.forward_decisions_per_second,
    )
    return sample


def _peak_memory_mb(
    runner: CheckpointModelRunner,
    batch: MoveModelBatch,
    decisions: Tensor,
) -> float | None:
    """Return megabytes one forward pass peaks at, where the device tracks it.

    Reset to the resident allocation first, so the figure is what the pass
    itself needs on top of the weights already loaded.
    """

    if runner.device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(runner.device)
    runner.decision_logits(batch, decisions)
    synchronize(runner.device)
    return torch.cuda.max_memory_allocated(runner.device) / 1e6


def _measurement_units(
    result: InferenceBenchmarkResult,
) -> tuple[tuple[InferenceDeviceReading, tuple[Measurement, ...]], ...]:
    """Return the committed measurements, grouped by the device that took them.

    One group becomes one envelope, because a device is a declared condition of
    every timing here rather than an incidental coordinate: the same metric read
    on the host and on the accelerator is two quantities, not one series
    measured twice.

    The counted quantities ride with the serving device even though they are
    the same number anywhere, since emitting them once per device would enter
    one value into its own series repeatedly.
    """

    units: list[tuple[InferenceDeviceReading, tuple[Measurement, ...]]] = []
    for index, reading in enumerate(result.readings):
        workload = reading.execution.workload_component()
        latency = reading.latency
        values = [
            measurement(
                INFERENCE_MOVE_LATENCY_BY_PERCENTILE[percentile].identifier,
                latency.percentiles[percentile],
                workload=workload,
                sample_size=latency.decisions,
            )
            for percentile in LATENCY_PERCENTILES
        ]
        values.append(
            measurement(
                INFERENCE_MOVE_LATENCY_MEAN.identifier,
                latency.mean_ms,
                workload=workload,
                sample_size=latency.decisions,
            )
        )
        values.append(
            measurement(
                INFERENCE_BATCH_THROUGHPUT.identifier,
                reading.serving.decisions_per_second,
                workload=workload,
                sample_size=reading.serving.batches,
            )
        )
        values.append(
            measurement(
                INFERENCE_DECISION_OVERHEAD_MS.identifier,
                reading.serving.decision_overhead_ms,
                workload=workload,
                sample_size=reading.serving.batches,
            )
        )
        if reading.compute is not None:
            values.append(
                measurement(
                    INFERENCE_FORWARD_THROUGHPUT.identifier,
                    reading.compute.forward_decisions_per_second,
                    workload=workload,
                    sample_size=reading.compute.batches,
                )
            )
            if reading.compute.peak_memory_mb is not None:
                values.append(
                    measurement(
                        INFERENCE_PEAK_MEMORY_MB.identifier,
                        reading.compute.peak_memory_mb,
                        workload=workload,
                        sample_size=1,
                    )
                )
        if reading.cold_start is not None:
            values.append(
                measurement(
                    INFERENCE_MODEL_LOAD_SECONDS.identifier,
                    reading.cold_start.model_load_seconds,
                    workload=workload,
                    sample_size=1,
                )
            )
            values.append(
                measurement(
                    INFERENCE_FIRST_DECISION_SECONDS.identifier,
                    reading.cold_start.first_decision_seconds,
                    workload=workload,
                    sample_size=1,
                )
            )
        if index == 0:
            values.append(
                measurement(
                    INFERENCE_PARAMETERS.identifier,
                    float(result.cost.parameters),
                    sample_size=1,
                )
            )
            values.append(
                measurement(
                    INFERENCE_DECISION_GFLOPS.identifier,
                    result.cost.decision_gflops,
                    sample_size=1,
                )
            )
        units.append((reading, tuple(values)))
    return tuple(units)


def _execution_record(
    config: InferenceBenchmarkConfig,
    device: torch.device,
    *,
    compute_batch_size: int | None,
) -> ExecutionRecord:
    """Capture the device, precision, and declared workload identity.

    The device is declared rather than left to the environment because this
    benchmark measures two of them on purpose, and two readings that differ only
    by their coordinates would otherwise land on one series.
    """

    workload: dict[str, Any] = {
        "benchmark_version": INFERENCE_BENCHMARK_VERSION,
        "device": device.type,
        "latency_reference_plies": config.latency.reference_plies,
        "latency_seed": config.latency.seed,
        "serving_batch_size": config.throughput.serving_batch_size,
        "throughput_history_plies": config.throughput.history_plies,
        "throughput_seed": config.throughput.seed,
        "target_rating": config.runtime.target_rating,
        "temperature": config.runtime.temperature,
    }
    if compute_batch_size is not None:
        workload["compute_batch_size"] = compute_batch_size
    return execution_record(device, workload)


def _reset(session: GameSession, history: Sequence[chess.Move]) -> None:
    try:
        session.reset(moves=tuple(history))
    except DecisionRuntimeError as error:  # pragma: no cover - walks are legal
        raise InferenceBenchmarkError(str(error)) from error


def _decide(session: GameSession) -> None:
    with _measured_decision():
        session.decide()


@contextmanager
def _measured_decision() -> Iterator[None]:
    """Report a failure inside a measured window as this benchmark's error.

    Every timed block drives the session and the runner together, so both
    failures mean the same thing here and read the same way to a caller.
    """

    try:
        yield
    except (DecisionRuntimeError, ModelRunnerError) as error:
        raise InferenceBenchmarkError(f"a measured decision failed: {error}") from error


def _percentile(values: Sequence[float], percentile: int) -> float:
    """Return a linear-interpolated percentile of measured durations."""

    ordered = sorted(values)
    if not ordered:  # pragma: no cover - callers measure at least one decision
        raise InferenceBenchmarkError("cannot summarize an empty measurement")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


__all__ = [
    "INFERENCE_BENCHMARK_VERSION",
    "INFERENCE_KIND",
    "ColdStart",
    "ComputeSample",
    "InferenceDeviceReading",
    "InferenceBenchmarkConfig",
    "InferenceBenchmarkError",
    "InferenceBenchmarkResult",
    "LatencySample",
    "LatencyWorkloadConfig",
    "ModelCost",
    "ThroughputSample",
    "ThroughputWorkloadConfig",
    "benchmark_inference",
]
