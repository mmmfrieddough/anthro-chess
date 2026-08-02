"""What a checkpoint costs to play with.

Three questions, deliberately kept apart. How long one move takes when a
person is waiting for it; how many decisions the same checkpoint resolves per
second when a benchmark or a server batches them; and how long the process
takes to become useful at all. Reporting them as one number would let a
batching win hide an interactive regression, which is the usual way a bot ends
up too slow to play against while its dashboard improves.

Latency is measured end to end through :class:`GameSession`, spanning
encoding, model execution, legal masking, and sampling, because that is what a
move actually costs. Timing the forward pass alone would report a number no
player ever experiences and would go on looking healthy while the encoder
regressed.

The workload is synthetic and self-contained: positions come from a seeded
random-legal-move walk rather than from the evaluation pool. Latency depends
on history length and legal-move count, not on which human played the game, so
binding this benchmark to the pool would break its series at every pool
generation for no gain in what it measures.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, TypeVar

import chess
import torch
from pydantic import Field, StrictInt
from torch import Tensor

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import DecisionContext, EncodingError, build_decision_context
from anthro_chess.evaluation.cost import benchmark_cost_result
from anthro_chess.evaluation.execution import (
    execution_record,
    synchronize,
)
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DetailReference,
    DetailStore,
    ExecutionRecord,
    Measurement,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    ResultsStoreError,
    WorkloadComponent,
    build_result,
    configuration_reference,
    default_checkpoint_label,
    measurement,
)
from anthro_chess.evaluation.results.metrics import (
    INFERENCE_BATCH_THROUGHPUT,
    INFERENCE_FIRST_DECISION_SECONDS,
    INFERENCE_MODEL_LOAD_SECONDS,
    INFERENCE_MOVE_LATENCY_BY_PERCENTILE,
    INFERENCE_MOVE_LATENCY_MEAN,
    LATENCY_PERCENTILES,
)
from anthro_chess.inference import CheckpointModelRunner, ModelRunnerConfig
from anthro_chess.inference.runner import ModelRunnerError
from anthro_chess.models import MoveModelBatch
from anthro_chess.runtime import DecisionRuntimeError, GameSession, RuntimeConfig

INFERENCE_BENCHMARK_VERSION = 1

INFERENCE_KIND = "inference-efficiency"
INFERENCE_BENCHMARK = BenchmarkReference(
    name="inference-efficiency",
    version=INFERENCE_BENCHMARK_VERSION,
)

#: What a process pays once. Measuring these again inside the same process
#: reads a warm file cache, an imported Torch, and compiled kernels, so a
#: repeat is not a second cold start. Anything replicating this benchmark has
#: to know which readings are per-process rather than per-measurement.
COLD_START_METRICS: frozenset[str] = frozenset(
    {
        INFERENCE_MODEL_LOAD_SECONDS.identifier,
        INFERENCE_FIRST_DECISION_SECONDS.identifier,
    }
)

SampleT = TypeVar("SampleT")

logger = logging.getLogger(__name__)


class _TimedRunner(Protocol):
    """The runner surface latency measurement needs.

    Narrower than the loaded runner on purpose: stage attribution needs the
    decision call and the device to synchronize on, and nothing else.
    """

    device: torch.device

    def predict(self, context: DecisionContext) -> Tensor:
        """Return raw action logits for the current decision."""


class InferenceBenchmarkError(ValueError):
    """Raised when inference efficiency cannot be measured as configured."""


class LatencyWorkloadConfig(ConfigModel):
    """What batch-one latency is measured at.

    ``reference_plies`` alone decides the headline series. The sweep exists to
    show how latency grows with history, and is deliberately kept out of series
    identity so extending it does not end the headline series.
    """

    #: The depth the reported percentiles are taken at. Forty plies is move
    #: twenty: a real midgame decision rather than an opening position whose
    #: short history would flatter a full-history model.
    reference_plies: StrictInt = Field(default=40, ge=0)
    sweep_plies: tuple[StrictInt, ...] = (0, 20, 40, 60, 80)
    decisions: StrictInt = Field(default=30, ge=1)
    #: Excluded from the percentiles. The first decisions pay for lazy kernel
    #: compilation and allocator growth, which is a cold-start cost and is
    #: reported as one.
    warmup_decisions: StrictInt = Field(default=5, ge=0)
    seed: str = Field(default="anthro-inference-latency-v1", min_length=1)


class ThroughputWorkloadConfig(ConfigModel):
    """What declared-batch throughput is measured at."""

    reference_batch_size: StrictInt = Field(default=8, ge=1)
    sweep_batch_sizes: tuple[StrictInt, ...] = (1, 4, 8, 16)
    history_plies: StrictInt = Field(default=40, ge=0)
    batches: StrictInt = Field(default=10, ge=1)
    warmup_batches: StrictInt = Field(default=3, ge=0)
    seed: str = Field(default="anthro-inference-throughput-v1", min_length=1)


class InferenceBenchmarkConfig(ConfigModel):
    """Code-owned schema for ``anthro eval inference``."""

    model: ModelRunnerConfig = ModelRunnerConfig()
    runtime: RuntimeConfig = RuntimeConfig(seed=0)
    latency: LatencyWorkloadConfig = LatencyWorkloadConfig()
    throughput: ThroughputWorkloadConfig = ThroughputWorkloadConfig()
    checkpoint_label: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )


@dataclass(frozen=True)
class LatencySample:
    """One depth's measured batch-one latency, in milliseconds."""

    history_plies: int
    decisions: int
    percentiles: Mapping[int, float]
    mean_ms: float
    minimum_ms: float
    maximum_ms: float
    #: Attribution of the end-to-end figure. ``encode`` and ``model`` are
    #: measured by calling the same two functions the session calls; the
    #: remainder covers legal masking and sampling and is derived rather than
    #: timed, so it absorbs measurement overhead instead of hiding it.
    encode_mean_ms: float
    model_mean_ms: float

    @property
    def remainder_mean_ms(self) -> float:
        """Return mean milliseconds outside encoding and model execution."""

        return self.mean_ms - self.encode_mean_ms - self.model_mean_ms

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable per-depth record."""

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
            "encode_mean_ms": self.encode_mean_ms,
            "model_mean_ms": self.model_mean_ms,
            "remainder_mean_ms": self.remainder_mean_ms,
        }


@dataclass(frozen=True)
class ThroughputSample:
    """One batch size's measured decision throughput."""

    batch_size: int
    batches: int
    decisions_per_second: float
    batch_mean_ms: float

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable per-batch-size record."""

        return {
            "batch_size": self.batch_size,
            "batches": self.batches,
            "decisions_per_second": self.decisions_per_second,
            "batch_mean_ms": self.batch_mean_ms,
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
class InferenceBenchmarkResult:
    """Everything one inference benchmark measured, and where it was written."""

    checkpoint: CheckpointReference
    execution: ExecutionRecord
    cold_start: ColdStart
    reference_latency: LatencySample
    latency_sweep: tuple[LatencySample, ...]
    reference_throughput: ThroughputSample
    throughput_sweep: tuple[ThroughputSample, ...]
    envelopes: tuple[ResultEnvelope, ...] = ()
    recorded_paths: tuple[Path, ...] = ()
    detail_paths: tuple[Path, ...] = ()

    def as_record(self) -> dict[str, Any]:
        """Return the full structured result, detail tier included."""

        return {
            "version": INFERENCE_BENCHMARK_VERSION,
            "checkpoint": self.checkpoint.model_dump(mode="json"),
            "execution": self.execution.model_dump(mode="json"),
            "cold_start": self.cold_start.as_record(),
            "reference_latency": self.reference_latency.as_record(),
            "latency_sweep": [sample.as_record() for sample in self.latency_sweep],
            "reference_throughput": self.reference_throughput.as_record(),
            "throughput_sweep": [
                sample.as_record() for sample in self.throughput_sweep
            ],
            "recorded": [str(path) for path in self.recorded_paths],
        }


def benchmark_inference(
    resolved_config: ResolvedConfig[InferenceBenchmarkConfig],
    *,
    run_root: Path | None = None,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> InferenceBenchmarkResult:
    """Measure one checkpoint's move latency, throughput, and cold start.

    Passing no ``store`` measures everything and records nothing, which is what
    an exploratory reading on a busy machine wants: a figure taken beside a
    training run is real but does not belong in the committed history.
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
    cold_start = ColdStart(
        model_load_seconds=model_load_seconds,
        first_decision_seconds=first_decision_seconds,
    )
    logger.info(
        "Loaded in %.3fs; first decision took %.3fs",
        model_load_seconds,
        first_decision_seconds,
    )

    latency_sweep = tuple(
        _measure_latency(session, runner, histories, config.latency, plies)
        for plies in _sweep_depths(config.latency)
    )
    reference_latency = _select(
        latency_sweep,
        lambda sample: sample.history_plies == config.latency.reference_plies,
        "latency sweep",
    )

    throughput_sweep = tuple(
        _measure_throughput(runner, config.throughput, histories, batch_size)
        for batch_size in _sweep_batch_sizes(config.throughput)
    )
    reference_throughput = _select(
        throughput_sweep,
        lambda sample: sample.batch_size == config.throughput.reference_batch_size,
        "throughput sweep",
    )

    execution = _execution_record(config, runner.device)
    checkpoint = _checkpoint_reference(config, runner)
    result = InferenceBenchmarkResult(
        checkpoint=checkpoint,
        execution=execution,
        cold_start=cold_start,
        reference_latency=reference_latency,
        latency_sweep=latency_sweep,
        reference_throughput=reference_throughput,
        throughput_sweep=throughput_sweep,
    )

    recorded_at = datetime.now(tz=UTC)
    detail_paths: list[Path] = []
    configuration = configuration_reference(
        resolved_config.as_record(),
        source=resolved_config.provenance.source,
        overrides=resolved_config.provenance.overrides,
    )
    try:
        detail_reference = _write_detail(
            detail,
            checkpoint=checkpoint,
            recorded_at=recorded_at,
            payload=result.as_record(),
            paths=detail_paths,
        )
        envelope = build_result(
            kind=INFERENCE_KIND,
            benchmark=INFERENCE_BENCHMARK,
            checkpoint=checkpoint,
            configuration=configuration,
            execution=execution,
            measurements=_measurements(result, execution.workload_component()),
            detail=detail_reference,
            recorded_at=recorded_at,
        )
        cost = benchmark_cost_result(
            benchmark=INFERENCE_BENCHMARK,
            checkpoint=checkpoint,
            configuration=configuration,
            config=config,
            device=runner.device,
            seconds=time.perf_counter() - started,
            recorded_at=recorded_at,
        )
    except ResultRecordError as error:
        raise InferenceBenchmarkError(str(error)) from error

    envelopes = (envelope, cost)
    recorded_paths: list[Path] = []
    if store is not None:
        try:
            recorded_paths = [store.append(item) for item in envelopes]
        except (ResultRecordError, ResultsStoreError) as error:
            raise InferenceBenchmarkError(str(error)) from error

    return InferenceBenchmarkResult(
        checkpoint=checkpoint,
        execution=execution,
        cold_start=cold_start,
        reference_latency=reference_latency,
        latency_sweep=latency_sweep,
        reference_throughput=reference_throughput,
        throughput_sweep=throughput_sweep,
        envelopes=envelopes,
        recorded_paths=tuple(recorded_paths),
        detail_paths=tuple(detail_paths),
    )


class _HistoryFactory:
    """Deterministic synthetic histories of a requested ply depth.

    A seeded random-legal-move walk from the standard opening. Reaching a
    terminal position before the requested depth restarts the walk from the
    next attempt rather than returning a short history, so a depth means the
    depth it says. Successive offsets give distinct positions at one depth, so
    a run measures a spread of histories instead of one cached board.
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
    """Play ``plies`` seeded legal moves, or return ``None`` if the game ends."""

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
    history_plies: int,
) -> LatencySample:
    """Measure end-to-end batch-one decision latency at one history depth."""

    device = runner.device
    for offset in range(config.warmup_decisions):
        _reset(session, histories.history(config.seed, history_plies, offset))
        _decide(session)
        synchronize(device)

    durations: list[float] = []
    encode_durations: list[float] = []
    model_durations: list[float] = []
    for index in range(config.decisions):
        history = histories.history(
            config.seed,
            history_plies,
            config.warmup_decisions + index,
        )
        _reset(session, history)
        started = time.perf_counter()
        _decide(session)
        synchronize(device)
        durations.append((time.perf_counter() - started) * 1000.0)
        encoded, model_seconds = _stage_times(
            runner,
            session.config.target_rating,
            history,
        )
        encode_durations.append(encoded * 1000.0)
        model_durations.append(model_seconds * 1000.0)

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
        encode_mean_ms=_mean(encode_durations),
        model_mean_ms=_mean(model_durations),
    )


def _stage_times(
    runner: _TimedRunner,
    target_rating: int | None,
    history: Sequence[chess.Move],
) -> tuple[float, float]:
    """Return seconds spent encoding and executing the model for one decision.

    These call the same two functions the session calls, with the same
    arguments, rather than reimplementing the decision path. Masking and
    sampling are not timed here; they are reported as the remainder, so this
    attribution can never claim more of the end-to-end figure than it measured.
    """

    board = _board(history)
    encode_started = time.perf_counter()
    try:
        context = build_decision_context(
            board,
            tuple(board.move_stack),
            target_rating=target_rating,
        )
    except EncodingError as error:  # pragma: no cover - the session encoded it
        raise InferenceBenchmarkError(str(error)) from error
    encode_seconds = time.perf_counter() - encode_started

    model_started = time.perf_counter()
    try:
        runner.predict(context)
    except ModelRunnerError as error:  # pragma: no cover - the session ran it
        raise InferenceBenchmarkError(str(error)) from error
    synchronize(runner.device)
    return encode_seconds, time.perf_counter() - model_started


def _measure_throughput(
    runner: CheckpointModelRunner,
    config: ThroughputWorkloadConfig,
    histories: _HistoryFactory,
    batch_size: int,
) -> ThroughputSample:
    """Measure decisions per second for one declared batch size."""

    batch = _build_batch(runner, config, histories, batch_size)
    for _ in range(config.warmup_batches):
        runner.action_logits(batch)
    synchronize(runner.device)

    started = time.perf_counter()
    for _ in range(config.batches):
        runner.action_logits(batch)
    synchronize(runner.device)
    elapsed = time.perf_counter() - started
    if elapsed <= 0.0:  # pragma: no cover - a clock this coarse is unusable
        raise InferenceBenchmarkError(
            "the measured throughput window was too short to time"
        )

    decisions = batch_size * config.batches
    logger.info(
        "Measured batch %s throughput: %.1f decisions/s",
        batch_size,
        decisions / elapsed,
    )
    return ThroughputSample(
        batch_size=batch_size,
        batches=config.batches,
        decisions_per_second=decisions / elapsed,
        batch_mean_ms=(elapsed / config.batches) * 1000.0,
    )


def _build_batch(
    runner: CheckpointModelRunner,
    config: ThroughputWorkloadConfig,
    histories: _HistoryFactory,
    batch_size: int,
) -> MoveModelBatch:
    """Stack one equal-length decision batch for the declared batch size."""

    rows: list[MoveModelBatch] = []
    try:
        for index in range(batch_size):
            board = _board(histories.history(config.seed, config.history_plies, index))
            rows.append(
                MoveModelBatch.from_decision_context(
                    build_decision_context(
                        board,
                        tuple(board.move_stack),
                        target_rating=None,
                    ),
                    device=runner.device,
                )
            )
    except (EncodingError, ValueError) as error:
        raise InferenceBenchmarkError(
            f"cannot build a batch of {batch_size} decisions: {error}"
        ) from error
    return MoveModelBatch.stack(rows)


def _board(history: Sequence[chess.Move]) -> chess.Board:
    """Return the board reached by replaying one synthetic history."""

    board = chess.Board()
    for move in history:
        board.push(move)
    return board


def _measurements(
    result: InferenceBenchmarkResult,
    workload: WorkloadComponent,
) -> tuple[Measurement, ...]:
    """Return the committed measurements for one benchmark run."""

    latency = result.reference_latency
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
            result.reference_throughput.decisions_per_second,
            workload=workload,
            sample_size=result.reference_throughput.batches,
        )
    )
    values.append(
        measurement(
            INFERENCE_MODEL_LOAD_SECONDS.identifier,
            result.cold_start.model_load_seconds,
            workload=workload,
            sample_size=1,
        )
    )
    values.append(
        measurement(
            INFERENCE_FIRST_DECISION_SECONDS.identifier,
            result.cold_start.first_decision_seconds,
            workload=workload,
            sample_size=1,
        )
    )
    return tuple(values)


def _execution_record(
    config: InferenceBenchmarkConfig,
    device: torch.device,
) -> ExecutionRecord:
    """Capture the device, precision, and declared workload identity."""

    return execution_record(
        device,
        {
            "benchmark_version": INFERENCE_BENCHMARK_VERSION,
            "latency_reference_plies": config.latency.reference_plies,
            "latency_seed": config.latency.seed,
            "throughput_batch_size": config.throughput.reference_batch_size,
            "throughput_history_plies": config.throughput.history_plies,
            "throughput_seed": config.throughput.seed,
            "target_rating": config.runtime.target_rating,
            "temperature": config.runtime.temperature,
        },
    )


def _reset(session: GameSession, history: Sequence[chess.Move]) -> None:
    try:
        session.reset(moves=tuple(history))
    except DecisionRuntimeError as error:  # pragma: no cover - walks are legal
        raise InferenceBenchmarkError(str(error)) from error


def _decide(session: GameSession) -> None:
    try:
        session.decide()
    except DecisionRuntimeError as error:
        raise InferenceBenchmarkError(f"a measured decision failed: {error}") from error


def _sweep_depths(config: LatencyWorkloadConfig) -> tuple[int, ...]:
    """Return the depths to measure, with the reference depth guaranteed."""

    return tuple(sorted({*config.sweep_plies, config.reference_plies}))


def _sweep_batch_sizes(config: ThroughputWorkloadConfig) -> tuple[int, ...]:
    """Return the batch sizes to measure, with the declared one guaranteed."""

    return tuple(sorted({*config.sweep_batch_sizes, config.reference_batch_size}))


def _select(
    samples: Sequence[SampleT],
    matches: Callable[[SampleT], bool],
    label: str,
) -> SampleT:
    """Return the sweep entry the declared reference point names."""

    for sample in samples:
        if matches(sample):
            return sample
    raise InferenceBenchmarkError(  # pragma: no cover - the sweep includes it
        f"the {label} did not include the declared reference point"
    )


def _write_detail(
    detail: DetailStore | None,
    *,
    checkpoint: CheckpointReference,
    recorded_at: datetime,
    payload: Mapping[str, Any],
    paths: list[Path],
) -> DetailReference | None:
    if detail is None:
        return None
    stamp = recorded_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    relative = Path(INFERENCE_KIND) / checkpoint.label / f"{stamp}.json"
    reference = detail.write(
        relative,
        dict(payload),
        description="Latency depth sweep, batch-size sweep, and stage attribution.",
    )
    paths.append(detail.root / relative)
    return reference


def _checkpoint_reference(
    config: InferenceBenchmarkConfig,
    runner: CheckpointModelRunner,
) -> CheckpointReference:
    run_id = runner.selection.run_path.name
    label = config.checkpoint_label or default_checkpoint_label(
        run_id,
        runner.global_step,
    )
    return CheckpointReference(
        label=label,
        step=runner.global_step,
        run_id=run_id,
        parameter_sha256=runner.parameter_sha256(),
    )


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
    "InferenceBenchmarkConfig",
    "InferenceBenchmarkError",
    "InferenceBenchmarkResult",
    "LatencySample",
    "LatencyWorkloadConfig",
    "ThroughputSample",
    "ThroughputWorkloadConfig",
    "benchmark_inference",
]
