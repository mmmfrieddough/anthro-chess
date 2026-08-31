"""What a checkpoint costs to play with, and what makes two costs comparable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import chess
import pytest
import torch
from pydantic import ValidationError

import anthro_chess.evaluation.execution_noise as execution_noise_module
import anthro_chess.evaluation.inference as inference_module
from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.data import DecisionContext
from anthro_chess.evaluation.benchmarks import benchmark_registry, run_benchmark
from anthro_chess.evaluation.cost import BENCHMARK_COST_KIND
from anthro_chess.evaluation.execution import measured_precision
from anthro_chess.evaluation.execution_noise import ProcessSample
from anthro_chess.evaluation.inference import (
    INFERENCE_KIND,
    InferenceBenchmarkConfig,
    InferenceBenchmarkError,
    InferenceBenchmarkResult,
    LatencyWorkloadConfig,
    ThroughputWorkloadConfig,
    _HistoryFactory,
    _measure_latency,
    _percentile,
)
from anthro_chess.evaluation.results import (
    PROCESS_REPLICATE_METHOD,
    AxisChange,
    BenchmarkReference,
    BridgeIndex,
    CheckpointReference,
    Comparability,
    DeltaReport,
    DetailStore,
    ExecutionRecord,
    MetricDelta,
    MetricDispersion,
    Movement,
    NoiseVerdict,
    ReportError,
    ReportPivot,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    build_delta_report,
    build_environment_report,
    build_history,
    build_result,
    execution_reference,
    measurement,
    render_history,
    workload_digest,
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
)
from anthro_chess.inference import CheckpointModelRunner, ModelRunnerConfig
from anthro_chess.inference.config import InferenceDevice
from anthro_chess.runtime import GameSession, RuntimeConfig

from accelerators import inference_accelerator_parameters

#: A workload small enough for the CPU suite. The measured quantities are
#: unchanged; only the number of samples behind them is.
#: Devices named for a workload digest without asking a driver for one. Both
#: are types the project targets, so neither is a fiction, and constructing one
#: is inert on a host that has no such device.
_STANDIN_HOST = torch.device("cpu")
_STANDIN_ACCELERATOR = torch.device("mps")

FAST_LATENCY = LatencyWorkloadConfig(
    reference_plies=4,
    decisions=3,
    warmup_decisions=1,
    seed="test-latency",
)
FAST_THROUGHPUT = ThroughputWorkloadConfig(
    serving_batch_size=2,
    compute_batch_size=4,
    history_plies=4,
    batches=2,
    warmup_batches=1,
    seed="test-throughput",
)


def _measure(
    resolved_config: ResolvedConfig[InferenceBenchmarkConfig],
    *,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> InferenceBenchmarkResult:
    """Measure the benchmark the way both callers do, through the driver."""

    return cast(
        InferenceBenchmarkResult,
        run_benchmark(
            benchmark_registry()["inference"],
            resolved_config,
            store=store,
            detail=detail,
        ),
    )


def _config(
    checkpoint: Path, **overrides: object
) -> ResolvedConfig[InferenceBenchmarkConfig]:
    fields: dict[str, object] = {
        "model": ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu"),
        "runtime": RuntimeConfig(seed=7),
        "latency": FAST_LATENCY,
        "throughput": FAST_THROUGHPUT,
        # One process, so these read what the benchmark measures rather than
        # paying for the replicate runs that pool and qualify it.
        "processes": 1,
    }
    fields.update(overrides)
    return ResolvedConfig(
        value=InferenceBenchmarkConfig(**fields),  # type: ignore[arg-type]
        provenance=ConfigProvenance(source=None, overrides=()),
    )


def test_benchmark_reports_latency_throughput_and_cold_start(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    checkpoint = inference_run(tmp_path / "run", seed=5)
    store = ResultsStore(tmp_path / "results")
    detail = DetailStore(tmp_path / "detail")

    result = _measure(_config(checkpoint), store=store, detail=detail)

    reading = result.serving
    assert result.host is None
    latency = reading.latency
    assert latency.history_plies == FAST_LATENCY.reference_plies
    assert latency.decisions == FAST_LATENCY.decisions
    assert latency.percentiles[50] <= latency.percentiles[90]
    assert latency.percentiles[90] <= latency.percentiles[99]
    assert latency.minimum_ms <= latency.mean_ms <= latency.maximum_ms
    assert reading.serving.batch_size == 2
    assert reading.serving.decisions_per_second > 0.0
    assert reading.cold_start is not None
    assert reading.cold_start.model_load_seconds > 0.0
    assert reading.cold_start.first_decision_seconds > 0.0
    assert result.cost.parameters > 0
    assert result.cost.decision_gflops > 0.0

    play, compute, cost = result.envelopes
    assert play.kind == INFERENCE_KIND
    assert compute.kind == INFERENCE_KIND
    assert cost.kind == BENCHMARK_COST_KIND
    assert {item.metric for item in play.measurements} == {
        INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier,
        INFERENCE_MOVE_LATENCY_BY_PERCENTILE[90].identifier,
        INFERENCE_MOVE_LATENCY_BY_PERCENTILE[99].identifier,
        INFERENCE_MOVE_LATENCY_MEAN.identifier,
        INFERENCE_BATCH_THROUGHPUT.identifier,
        INFERENCE_DECISION_OVERHEAD_MS.identifier,
        INFERENCE_PARAMETERS.identifier,
        INFERENCE_DECISION_GFLOPS.identifier,
        INFERENCE_MODEL_LOAD_SECONDS.identifier,
        INFERENCE_FIRST_DECISION_SECONDS.identifier,
    }
    # The instrument is declared apart, so moving it ends only its own series.
    assert {item.metric for item in compute.measurements} == {
        INFERENCE_FORWARD_THROUGHPUT.identifier
    }
    assert play.execution is not None
    assert compute.execution is not None
    assert play.execution.workload_sha256 != compute.execution.workload_sha256
    assert result.recorded_paths and result.recorded_paths[0].exists()
    assert result.detail_paths and result.detail_paths[0].exists()


def _stub_sampler(
    monkeypatch: pytest.MonkeyPatch,
    parent: InferenceBenchmarkResult,
    offsets: Callable[[int], float],
) -> list[InferenceBenchmarkConfig]:
    """Answer every replicate with the parent's reading, shifted by a round.

    Stubbed rather than spawned wherever what is under test is how the readings
    combine: a real subprocess would put the machine's own noise into an
    assertion about arithmetic.
    """

    sampled: list[InferenceBenchmarkConfig] = []

    def fake_sampler(
        selection: InferenceBenchmarkConfig,
        **kwargs: object,
    ) -> Callable[[], tuple[ProcessSample, ...]]:
        def sample() -> tuple[ProcessSample, ...]:
            sampled.append(selection)
            offset = offsets(len(sampled))
            return tuple(
                ProcessSample(
                    execution=envelope.execution,
                    checkpoint=envelope.checkpoint,
                    values=tuple(
                        item.model_copy(update={"value": item.value + offset})
                        for item in envelope.measurements
                    ),
                )
                for envelope in parent.envelopes
                if envelope.kind == INFERENCE_KIND and envelope.execution is not None
            )

        return sample

    monkeypatch.setattr(execution_noise_module, "subprocess_sampler", fake_sampler)
    return sampled


def test_a_reading_carries_the_spread_its_own_replicate_processes_measured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inference_run: Callable[..., Path],
) -> None:
    """Nothing inside a timing reading can be resampled, so it is re-measured."""

    checkpoint = inference_run(tmp_path / "run", seed=23)
    parent = _measure(_config(checkpoint, processes=1))
    # Each replicate reads a little slower than the last, so the spread is
    # non-zero and the parent's own reading is not the whole of it.
    sampled = _stub_sampler(monkeypatch, parent, lambda round_: 0.5 * round_)

    result = _measure(_config(checkpoint, processes=3))

    # Two sampled, because the reading being qualified is the third.
    assert len(sampled) == 2
    assert {item.model.checkpoint_path for item in sampled} == {checkpoint}
    assert result.processes == 3
    for item in result.envelopes[0].measurements:
        assert item.dispersion is not None
        assert item.dispersion.estimator == PROCESS_REPLICATE_METHOD
        assert item.dispersion.bound >= item.dispersion.value
        # A machine spread does not shrink with a game count, so it sizes no pool.
        assert item.dispersion.units is None


def test_the_committed_value_is_pooled_over_the_processes_that_read_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inference_run: Callable[..., Path],
) -> None:
    """The extra processes pay for themselves twice, and this is the half that
    would otherwise be thrown away.

    Where a process lands is nearly the whole of the noise, so a reading is only
    improved by taking it in more of them. Committing the parent's own value and
    keeping the replicates for the floor alone would leave the number as noisy
    as one process while paying for several.
    """

    checkpoint = inference_run(tmp_path / "run", seed=29)
    parent = _measure(_config(checkpoint, processes=1))
    offsets = {1: 2000.0, 2: 10000.0}
    _stub_sampler(monkeypatch, parent, lambda round_: offsets[round_])

    result = _measure(_config(checkpoint, processes=3))

    latency = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    own = parent.envelopes[0].measurement(latency)
    pooled = result.envelopes[0].measurement(latency)
    assert own is not None
    assert pooled is not None
    # The offsets dwarf any real spread between two readings of this fixture, so
    # the live process contributes a value indistinguishable from the parent's
    # at this scale and the mean is theirs plus a third of the two offsets.
    assert pooled.value == pytest.approx(own.value + 4000.0, rel=1e-3)


def test_a_metric_the_replicates_read_identically_is_left_unqualified(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """A floor of zero would clear every later delta on that metric.

    What the replicates observed is that these processes did not separate the
    number, which is not the observation that nothing could. The measured spread
    stays in the reading's diagnostics; only the stored floor is withheld.
    """

    checkpoint = inference_run(tmp_path / "run", seed=27)
    result = _measure(_config(checkpoint, processes=1))
    values = result.envelopes[0].measurements
    unseparated = INFERENCE_MODEL_LOAD_SECONDS.identifier
    readings = {
        item.fingerprint: (item.value, 0.0 if item.metric == unseparated else 0.5)
        for item in values
    }

    records = inference_module._dispersions(
        inference_module._measurement_units(result), readings, 3
    )

    qualified = {item.metric for item in values if item.fingerprint in records}
    assert qualified == {item.metric for item in values} - {unseparated}


def test_a_single_replicate_reads_without_measuring_a_spread(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """The replicate processes are what stop the recursion, so they take this.

    It is also the exploratory reading: one process, no dispersion, and the
    report says the noise is unknown rather than claiming a floor.
    """

    checkpoint = inference_run(tmp_path / "run", seed=24)

    result = _measure(_config(checkpoint, processes=1))

    assert result.processes == 1
    assert result.pooled == {}
    assert all(item.dispersion is None for item in result.envelopes[0].measurements)


def test_a_replicate_process_measures_the_checkpoint_it_was_handed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inference_run: Callable[..., Path],
) -> None:
    """The whole handoff, through a real interpreter rather than a stub.

    Everything else about replication is tested with the sampler patched out,
    which cannot catch the selection failing to survive the trip: a replicate
    resolving its own checkpoint is the failure mode this hands the resolved
    path over to avoid, and it shows up as a refusal rather than a wrong number.
    """

    checkpoint = inference_run(tmp_path / "run", seed=26)
    # No default selection to fall back on, so a replicate that re-resolved
    # instead of reading what it was handed would fail to load anything at all.
    monkeypatch.setenv("ANTHRO_CHESS_RUN_ROOT", str(tmp_path / "empty"))

    result = _measure(_config(checkpoint, processes=2))

    assert result.processes == 2
    assert {label for label in result.pooled if label.startswith("cpu ")}
    # The floor a delta faces is the pooled value's own, not one process's.
    assert all(spread >= 0.0 for _, spread in result.pooled.values())


def test_the_stage_attribution_never_claims_more_than_the_whole(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """Two of three parts summing past the total is worse than no breakdown.

    They did, because the parts were timed beside the measured decision rather
    than inside it: a second, non-incremental encode and a second forward pass,
    both charged against an end-to-end figure that paid for neither.
    """

    checkpoint = inference_run(tmp_path / "run", seed=16)

    result = _measure(_config(checkpoint))

    for reading in result.readings:
        sample = reading.latency
        assert sample.context_mean_ms >= 0.0
        assert sample.predict_mean_ms >= 0.0
        # The parts are cut from the window they are subtracted from, so a
        # non-negative remainder says they did not sum past it.
        assert sample.remainder_mean_ms >= 0.0


def test_the_context_stage_uses_the_session_the_decision_came_from(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """The session assembles its context from plies it already encoded.

    Timing a from-scratch ``build_decision_context`` instead reported that
    assembly as two orders of magnitude above its real cost and grew it with
    history, which is how a reader got the encode-or-model question backwards.
    """

    checkpoint = inference_run(tmp_path / "run", seed=17)
    runner = CheckpointModelRunner.load(
        ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu")
    )
    session = GameSession(runner, config=RuntimeConfig(seed=7))
    contexts: list[DecisionContext] = []
    from_session = session.decision_context

    def recording() -> DecisionContext:
        context = from_session()
        contexts.append(context)
        return context

    config = LatencyWorkloadConfig(
        reference_plies=4,
        decisions=2,
        warmup_decisions=0,
        seed="test-encode",
    )

    with pytest.MonkeyPatch.context() as patch:
        # Patched on the session rather than on the benchmark, so this asserts
        # against the seam the engine itself uses.
        patch.setattr(session, "decision_context", recording)
        sample = _measure_latency(session, runner, _HistoryFactory(), config)

    # Every measured decision took its context from the session, once.
    assert len(contexts) == config.decisions
    assert sample.context_mean_ms < sample.mean_ms


def test_throughput_times_the_batch_it_builds(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """The suite's cost unit has to include what a decision actually builds.

    Both windows run the same model call, so what separates them is exactly the
    batch construction, masking, and sampling a generated decision pays for
    every ply. That difference is a fraction of a millisecond here, so this
    times more batches than the other tests need: at the shared two, one
    scheduling stall on a loaded machine moves the median past a real gap.
    """

    checkpoint = inference_run(tmp_path / "run", seed=18)

    sample = (
        _measure(
            _config(
                checkpoint,
                throughput=FAST_THROUGHPUT.model_copy(update={"batches": 9}),
            )
        )
        .readings[0]
        .serving
    )

    assert sample.decisions_per_second > 0.0
    # Re-running a built batch drops batch construction, masking, and sampling
    # from the window, and what it drops is what the overhead figure reports.
    assert sample.forward_median_ms < sample.batch_median_ms
    assert sample.decision_overhead_ms > 0.0


def test_a_reading_reaches_the_depth_the_generated_benchmarks_play_at(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """A 300-ply request used to abort the command rather than measure it.

    Completing is most of the claim: a walk that never ran out of legal moves
    could still arrive somewhere over by rule, and the session then refused to
    decide, which failed the whole command. This drives the real session, so it
    covers the half
    :func:`test_a_history_ending_on_a_dead_position_is_walked_again` cannot.
    """

    checkpoint = inference_run(tmp_path / "run", seed=19)

    result = _measure(
        _config(
            checkpoint,
            latency=FAST_LATENCY.model_copy(update={"reference_plies": 300}),
        )
    )

    deep = result.readings[0].latency
    # Decisions were resolved at that depth, not merely scheduled for it.
    assert deep.history_plies == 300
    assert deep.mean_ms > 0.0
    assert deep.predict_mean_ms > 0.0


def test_a_history_ending_on_a_dead_position_is_walked_again(
    tmp_path: Path,
) -> None:
    """A depth means a position a session can still decide from."""

    factory = _HistoryFactory()

    for offset in range(8):
        history = factory.history("anthro-inference-latency-v1", 300, offset)
        board = chess.Board()
        for move in history:
            board.push(move)
        assert len(history) == 300
        assert not board.is_game_over()


def test_cold_start_is_reported_apart_from_steady_state_latency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inference_run: Callable[..., Path],
) -> None:
    """Loading and first-call warmup must not inflate the percentiles.

    A benchmark that folded them in would report a checkpoint as slow to play
    with when it is only slow to start, and the two have different fixes.

    The startup cost is injected rather than measured, on the clock the warmup
    test injects. Loading this fixture checkpoint and deciding one move are
    both a few milliseconds on an idle machine, so a real clock makes the
    comparison a property of what else the machine is doing: under a sharded
    run this asserted that a contended decision beat an injected 50ms, and
    lost. Charging every window one tick keeps the durations ordered and
    nonzero, which is what the whole benchmark needs to report at all.
    """

    checkpoint = inference_run(tmp_path / "run", seed=6)
    tick_seconds = 0.001
    delay_seconds = 0.05
    clock = _FakeClock(tick_seconds=tick_seconds)
    monkeypatch.setattr(inference_module, "time", clock)
    real_decide = inference_module._decide
    served = 0

    def slow_first_decision(session: GameSession) -> None:
        nonlocal served
        if served == 0:
            served += 1
            clock.advance(delay_seconds)
        real_decide(session)

    monkeypatch.setattr(inference_module, "_decide", slow_first_decision)

    result = _measure(_config(checkpoint))

    delay_ms = delay_seconds * 1000.0
    assert served == 1
    # The cold reading carries the startup cost the first decision paid.
    cold_start = result.readings[0].cold_start
    assert cold_start is not None
    assert cold_start.first_decision_seconds * 1000.0 >= delay_ms
    # A measured decision reads the clock at its start, after encoding, after
    # predicting, and at its end, so a steady-state window carries three ticks
    # and nothing else. The separation is exact rather than a race against the
    # injected cost.
    assert result.readings[0].latency.maximum_ms == pytest.approx(
        3 * tick_seconds * 1000.0
    )
    # Loading is still reported, on its own, as the other half of cold start.
    assert cold_start.model_load_seconds > 0.0


def test_warmup_decisions_are_excluded_from_the_percentiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inference_run: Callable[..., Path],
) -> None:
    """The measured window must start after the slow first calls.

    Time is injected here for the reason the cold-start test injects it: the
    startup cost and a real decision are both a few milliseconds, so comparing
    them ambiently decides which piece of machine noise won rather than
    whether warmup landed inside the measured window.
    """

    checkpoint = inference_run(tmp_path / "run", seed=8)
    runner = CheckpointModelRunner.load(
        ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu")
    )
    warmup = 2
    delay_seconds = 0.05
    clock = _FakeClock()
    monkeypatch.setattr(inference_module, "time", clock)
    slow = _SlowFirstRunner(
        runner,
        slow_calls=warmup,
        delay_seconds=delay_seconds,
        clock=clock,
    )
    session = GameSession(slow, config=RuntimeConfig(seed=7))
    config = LatencyWorkloadConfig(
        reference_plies=2,
        decisions=4,
        warmup_decisions=warmup,
        seed="test-warmup",
    )

    sample = _measure_latency(session, slow, _HistoryFactory(), config)

    assert slow.slow_calls_served == warmup
    assert sample.decisions == 4
    # Nothing in the measured window advanced the clock, so a warmup decision
    # inside it would show up as the whole injected startup cost.
    assert sample.maximum_ms == 0.0


def test_a_faster_machine_is_not_reported_as_a_faster_model() -> None:
    """The delta is real, so it is shown; it just is not a model verdict."""

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    baseline = _efficiency_result("checkpoint-a", 12.0, device_name="laptop")
    same_machine = _efficiency_result("checkpoint-b", 9.0, device_name="laptop")
    other_machine = _efficiency_result("checkpoint-b", 9.0, device_name="workstation")

    within = build_delta_report(
        [baseline, same_machine], BridgeIndex(), metrics=[metric]
    )
    across = build_delta_report(
        [baseline, other_machine], BridgeIndex(), metrics=[metric]
    )

    # Same machine: an ordinary verdict on the model.
    assert _delta(within).movement is Movement.BETTER
    assert _delta(within).delta == -3.0

    # Other machine: the same arithmetic, and no claim about the model.
    row = _delta(across)
    assert row.comparability is Comparability.SAME_SERIES
    assert row.delta == -3.0
    assert row.movement is Movement.CONFOUNDED
    assert row.attribution is not None
    assert row.attribution.environment is AxisChange.CHANGED
    assert "device_name" in {difference.field for difference in row.environment}


def test_an_agent_reading_the_record_cannot_mistake_a_machine_for_progress() -> None:
    """Automation keys on ``movement``, so that is where the honesty has to be.

    Withholding ``delta`` would protect nothing, since the record carries both
    operands and any reader can subtract them.
    """

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    report = build_delta_report(
        [
            _efficiency_result("checkpoint-a", 12.0, device_name="laptop"),
            _efficiency_result(
                "checkpoint-b",
                9.0,
                device_name="workstation",
                weights="b" * 64,
            ),
        ],
        BridgeIndex(),
        metrics=[metric],
    )

    row = _delta(report).as_record()

    assert row["movement"] != "better"
    assert row["movement"] == "confounded"
    assert row["delta"] == -3.0
    assert row["attribution"] == {
        "model": "changed",
        "environment": "changed",
        "workload": "unchanged",
        # Inference declares no coordinates: its whole workload is identity,
        # so there is nothing to hold still beside it.
        "conditions": "unknown",
    }
    assert row["environment_differences"] == [
        {"field": "device_name", "baseline": "laptop", "current": "workstation"}
    ]


def test_a_changed_workload_is_still_refused_outright() -> None:
    """A different workload is a different measurement, not a confound."""

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    report = build_delta_report(
        [
            _efficiency_result("checkpoint-a", 12.0, device_name="laptop", plies=40),
            _efficiency_result("checkpoint-b", 24.0, device_name="laptop", plies=80),
        ],
        BridgeIndex(),
        metrics=[metric],
    )

    row = _delta(report)

    assert row.comparability is Comparability.INCOMPARABLE
    assert row.delta is None
    assert row.attribution is not None
    assert row.attribution.workload is AxisChange.CHANGED
    assert row.note is not None
    assert "workload changed" in row.note


def test_a_tampered_execution_record_cannot_reproduce_its_fingerprints() -> None:
    """Relabelling the workload after the fact must not launder a comparison."""

    result = _efficiency_result("checkpoint-a", 12.0, device_name="laptop")
    assert result.execution is not None
    relabelled = result.model_copy(
        update={
            "execution": result.execution.model_copy(
                update={
                    "workload": {"latency_reference_plies": 80},
                    "workload_sha256": workload_digest({"latency_reference_plies": 80}),
                }
            )
        }
    )

    with pytest.raises(ResultRecordError, match="does not reproduce"):
        relabelled.verify()


def test_the_recorded_execution_reproduces_its_own_series_identity(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    checkpoint = inference_run(tmp_path / "run", seed=10)

    envelope, _, _ = _measure(_config(checkpoint)).envelopes

    assert envelope.execution is not None
    assert envelope.execution.device == "cpu"
    assert envelope.execution.precision == measured_precision()
    assert envelope.execution.cpu_threads == torch.get_num_threads()
    assert envelope.execution.workload["latency_reference_plies"] == 4
    # verify() recomputes every fingerprint from the record alone.
    envelope.verify()


def test_a_declared_workload_change_starts_a_new_series(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    checkpoint = inference_run(tmp_path / "run", seed=12)
    shallow = _measure(_config(checkpoint)).envelopes[0]
    deeper = _measure(
        _config(
            checkpoint,
            latency=FAST_LATENCY.model_copy(update={"reference_plies": 0}),
        )
    ).envelopes[0]

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    shallow_measurement = shallow.measurement(metric)
    deeper_measurement = deeper.measurement(metric)

    assert shallow_measurement is not None
    assert deeper_measurement is not None
    assert shallow_measurement.fingerprint != deeper_measurement.fingerprint


def test_measuring_more_decisions_does_not_end_the_series(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """Sample counts estimate the same quantity more precisely."""

    checkpoint = inference_run(tmp_path / "run", seed=13)
    narrow = _measure(_config(checkpoint)).envelopes[0]
    wide = _measure(
        _config(
            checkpoint,
            latency=FAST_LATENCY.model_copy(update={"decisions": 5}),
        )
    ).envelopes[0]

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    narrow_measurement = narrow.measurement(metric)
    wide_measurement = wide.measurement(metric)

    assert narrow_measurement is not None
    assert wide_measurement is not None
    assert narrow_measurement.fingerprint == wide_measurement.fingerprint


def test_the_host_and_the_accelerator_do_not_land_on_one_series(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """One run measures two devices, so the device has to be declared.

    Left as an environment coordinate it would be comparison metadata, and a
    report following the machine across a hardware change would read a host
    reading and an accelerator reading as one line moving.
    """

    checkpoint = inference_run(tmp_path / "run", seed=31)
    config = _config(checkpoint).value

    host = inference_module._play_execution(config, _STANDIN_HOST)
    accelerator = inference_module._play_execution(config, _STANDIN_ACCELERATOR)

    assert host.workload_sha256 != accelerator.workload_sha256
    # And moving the instrument leaves the product timings' series alone.
    assert (
        accelerator.workload_sha256
        != inference_module._compute_execution(
            config, _STANDIN_ACCELERATOR, 4
        ).workload_sha256
    )


def test_what_a_decision_costs_the_model_is_counted_rather_than_timed(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """A count carries no noise, so it needs no floor and no replicates.

    It is also the only thing here that separates two model sizes reliably: a
    single forward is too small to occupy an accelerator, so a wall clock on
    one barely moves between them.
    """

    checkpoint = inference_run(tmp_path / "run", seed=32)

    first = _measure(_config(checkpoint))
    again = _measure(_config(checkpoint))

    assert first.cost.parameters == again.cost.parameters
    assert first.cost.decision_gflops == again.cost.decision_gflops
    counted = first.envelopes[0].measurement(INFERENCE_DECISION_GFLOPS.identifier)
    assert counted is not None
    assert counted.value == pytest.approx(first.cost.decision_gflops)
    # Declared unqualifiable rather than left unknown, so a report says there is
    # no spread to measure instead of pointing a reader at work to do.
    for definition in (INFERENCE_PARAMETERS, INFERENCE_DECISION_GFLOPS):
        assert definition.no_sampling_floor_reason is not None


def test_a_two_device_reading_splits_its_series_and_counts_once(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """The grouping a run on an accelerator produces, without needing one.

    The host reading is grafted onto a real result rather than measured, so this
    runs anywhere. What it pins is the part a device cannot change: which
    envelope each metric lands in, and that the counted quantities are entered
    once rather than once per device.
    """

    checkpoint = inference_run(tmp_path / "run", seed=33)
    measured = _measure(_config(checkpoint))
    accelerator = replace(
        measured.serving,
        execution=inference_module._play_execution(
            _config(checkpoint).value, _STANDIN_ACCELERATOR
        ),
    )
    # The host reading carries no compute instrument and no cold start, which
    # are both paid once by the process that loaded the checkpoint.
    host = replace(measured.serving, compute=None, cold_start=None)
    result = replace(measured, serving=accelerator, host=host)

    units = {unit.slug: unit for unit in inference_module._measurement_units(result)}

    device = _STANDIN_ACCELERATOR.type
    assert set(units) == {device, f"{device}-compute", "cpu"}
    counted = {INFERENCE_PARAMETERS.identifier, INFERENCE_DECISION_GFLOPS.identifier}
    assert counted <= {value.metric for value in units[device].values}
    assert not counted & {value.metric for value in units["cpu"].values}
    assert INFERENCE_MODEL_LOAD_SECONDS.identifier not in {
        value.metric for value in units["cpu"].values
    }
    # The same metric on two devices is two series rather than one read twice.
    latency = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    fingerprints = {
        slug: next(v.fingerprint for v in unit.values if v.metric == latency)
        for slug, unit in units.items()
        if any(v.metric == latency for v in unit.values)
    }
    assert len(set(fingerprints.values())) == len(fingerprints) == 2


@pytest.mark.gpu
@pytest.mark.parametrize("accelerator", inference_accelerator_parameters())
def test_an_accelerator_run_also_measures_the_host(
    accelerator: InferenceDevice,
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """The half of the two-device path a host without an accelerator cannot run.

    Everything downstream of the second reading is pinned on a grafted result,
    which cannot catch the graft itself being wrong: that the replica loads,
    decides, and reports as the host rather than as the device it was copied
    from.
    """

    checkpoint = inference_run(tmp_path / "run", seed=34)
    store = ResultsStore(tmp_path / "results")

    result = _measure(
        _config(
            checkpoint,
            model=ModelRunnerConfig(checkpoint_path=checkpoint, device=accelerator),
        ),
        store=store,
    )

    assert result.serving.device == accelerator
    assert result.host is not None
    assert result.host.device == "cpu"
    # The replica really decided, rather than reporting the accelerator's work.
    assert result.host.latency.mean_ms > 0.0
    assert result.host.serving.decisions_per_second > 0.0
    # The instrument and the cold start are the serving device's alone.
    assert result.host.compute is None
    assert result.host.cold_start is None
    kinds = [envelope.kind for envelope in result.envelopes]
    assert kinds.count(INFERENCE_KIND) == 3


def test_a_walk_that_ends_early_restarts_rather_than_returning_a_short_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A depth means the depth it says, or the benchmark says it cannot."""

    real_walk = inference_module._random_walk
    attempts: list[str] = []

    def failing_first(seed: str, plies: int) -> tuple[chess.Move, ...] | None:
        attempts.append(seed)
        return None if len(attempts) == 1 else real_walk(seed, plies)

    monkeypatch.setattr(inference_module, "_random_walk", failing_first)

    history = _HistoryFactory().history("restart", 6)

    assert len(attempts) == 2
    assert len(history) == 6


def test_a_depth_no_legal_walk_reaches_is_reported_rather_than_shortened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inference_module,
        "_random_walk",
        lambda seed, plies: None,
    )

    with pytest.raises(InferenceBenchmarkError, match="ended before that depth"):
        _HistoryFactory().history("unreachable", 6)


def test_percentiles_interpolate_between_measured_samples() -> None:
    samples = [10.0, 20.0, 30.0, 40.0]

    assert _percentile(samples, 0) == 10.0
    assert _percentile(samples, 50) == 25.0
    assert _percentile(samples, 100) == 40.0
    assert _percentile([7.0], 99) == 7.0


def test_history_factory_is_deterministic_and_varies_by_offset() -> None:
    factory = _HistoryFactory()

    first = factory.history("seed", 6, 0)
    again = _HistoryFactory().history("seed", 6, 0)
    other = factory.history("seed", 6, 1)

    assert len(first) == 6
    assert first == again
    assert first != other


def test_an_execution_metric_cannot_be_scheduled_at_a_training_cadence() -> None:
    """Timed beside a training step it would measure contention, not a model."""

    from anthro_chess.training.cadence import (
        CadenceConfig,
        CadenceError,
        TrainingEvaluationConfig,
        prepare_schedule,
    )

    with pytest.raises(CadenceError, match="measures execution time"):
        prepare_schedule(
            TrainingEvaluationConfig(
                cadences=(
                    CadenceConfig(
                        name="preview",
                        every_steps=10,
                        metrics=(INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier,),
                    ),
                ),
            ),
            None,
        )


class _FakeClock:
    """A clock only the test advances, so no ambient time is measured.

    A startup cost has to be compared against something, and comparing it
    against a real decision makes the comparison a property of what else the
    machine is doing: a contended runner stalls a fast decision for longer
    than the cost being injected. Advancing time only where the test says so
    removes the ambient side of that comparison, and costs no sleep.
    """

    def __init__(self, tick_seconds: float = 0.0) -> None:
        self.now = 0.0
        self._tick_seconds = tick_seconds

    def perf_counter(self) -> float:
        """Return the current fake time, in seconds.

        A caller measuring a whole benchmark needs every window to come out
        positive, since a load or a throughput batch that took no time at all
        is reported as unmeasurable rather than as fast. A tick charges each
        read a fixed cost, which keeps the durations ordered and nonzero
        without letting the machine decide any of them. It defaults to zero so
        a caller timing one window still sees real work as free.
        """

        self.now += self._tick_seconds
        return self.now

    def advance(self, seconds: float) -> None:
        """Charge the caller for time it did not really spend."""

        self.now += seconds


class _SlowFirstRunner:
    """A runner whose first calls are slow, standing in for a cold device."""

    def __init__(
        self,
        runner: CheckpointModelRunner,
        *,
        slow_calls: int,
        delay_seconds: float,
        clock: _FakeClock,
    ) -> None:
        self._runner = runner
        self._slow_calls = slow_calls
        self._delay_seconds = delay_seconds
        self._clock = clock
        self.device = runner.device
        self.slow_calls_served = 0

    def predict(self, context: DecisionContext) -> torch.Tensor:
        """Return real logits, paying a startup cost on the first calls."""

        if self.slow_calls_served < self._slow_calls:
            self.slow_calls_served += 1
            self._clock.advance(self._delay_seconds)
        return self._runner.predict(context)


def _efficiency_result(
    label: str,
    latency_ms: float,
    *,
    device_name: str,
    plies: int = 40,
    torch_version: str = "2.7.0",
    weights: str = "a" * 64,
    day: int | None = None,
    dispersion: float | None = None,
) -> ResultEnvelope:
    """Build one recorded efficiency result on a named machine.

    ``dispersion`` is the spread this reading measured across its own replicate
    processes, which is where an efficiency floor comes from.
    """

    execution = execution_reference(
        device="cpu",
        device_name=device_name,
        precision="float32",
        torch_version=torch_version,
        platform_key="Fixture-x86",
        platform="fixture-1.2.3",
        cpu_threads=8,
        workload={"latency_reference_plies": plies},
    )
    return build_result(
        kind=INFERENCE_KIND,
        benchmark=BenchmarkReference(name="inference-efficiency", version=1),
        checkpoint=CheckpointReference(
            label=label,
            step=1,
            parameter_sha256=weights,
        ),
        execution=execution,
        measurements=[
            measurement(
                INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier,
                latency_ms,
                workload=execution.workload_component(),
                dispersion=(
                    None
                    if dispersion is None
                    else MetricDispersion(
                        value=dispersion,
                        bound=dispersion,
                        source=f"6 process replicates on {device_name}",
                        estimator=PROCESS_REPLICATE_METHOD,
                    )
                ),
            )
        ],
        recorded_at=datetime(
            2026,
            7,
            day if day is not None else (1 if label.endswith("a") else 2),
            tzinfo=UTC,
        ),
    )


def _delta(report: DeltaReport) -> MetricDelta:
    families = [family for family in report.families if family.metrics]
    (family,) = families
    (delta,) = family.metrics
    return delta


def _comparability(report: DeltaReport) -> Comparability:
    return _delta(report).comparability


def test_a_recorded_workload_must_produce_its_own_digest() -> None:
    """The readable workload and the digest naming its series stay agreed."""

    record = execution_reference(
        device="cpu",
        device_name="laptop",
        precision="float32",
        torch_version="2.7.0",
        platform_key="Fixture-x86",
        platform="fixture-1.2.3",
        workload={"latency_reference_plies": 40},
    )

    with pytest.raises(ValidationError, match="does not produce the recorded digest"):
        ExecutionRecord(
            **{**record.model_dump(), "workload": {"latency_reference_plies": 80}}
        )


def test_the_environment_pivot_answers_whether_the_upgrade_helped() -> None:
    """The mirror of the default view: model pinned, machine varying."""

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    report = build_environment_report(
        [
            _efficiency_result("blitz-v3", 27.0, device_name="laptop", day=1),
            _efficiency_result("blitz-v3", 9.0, device_name="desktop", day=2),
        ],
        BridgeIndex(),
        metrics=[metric],
    )
    row = _delta(report)

    assert report.pivot is ReportPivot.ENVIRONMENT
    assert report.baseline is not None
    assert "laptop" in report.baseline.label
    assert "desktop" in report.current.label
    # With the model pinned this is a real verdict rather than a confound.
    assert row.delta == -18.0
    assert row.movement is Movement.BETTER
    assert row.attribution is not None
    assert row.attribution.model is AxisChange.UNCHANGED
    assert row.attribution.environment is AxisChange.CHANGED


def test_the_environment_pivot_refuses_when_the_weights_moved() -> None:
    """Otherwise a model change would be sold as a hardware win."""

    with pytest.raises(ReportError, match="more than one set of weights"):
        build_environment_report(
            [
                _efficiency_result("blitz-v3", 27.0, device_name="laptop", day=1),
                _efficiency_result(
                    "blitz-v3",
                    9.0,
                    device_name="desktop",
                    weights="b" * 64,
                    day=2,
                ),
            ],
            BridgeIndex(),
            metrics=[INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier],
        )


def test_the_environment_pivot_reports_a_torch_upgrade_on_one_machine() -> None:
    """The optimization case that motivated keeping the series continuous."""

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    report = build_environment_report(
        [
            _efficiency_result(
                "blitz-v3", 27.0, device_name="laptop", torch_version="2.7.0", day=1
            ),
            _efficiency_result(
                "blitz-v3", 21.0, device_name="laptop", torch_version="2.9.0", day=2
            ),
        ],
        BridgeIndex(),
        metrics=[metric],
    )
    row = _delta(report)

    assert row.movement is Movement.BETTER
    assert row.delta == -6.0
    assert [difference.field for difference in row.environment] == ["torch_version"]


def test_history_stays_one_line_across_machines_and_annotates_the_move() -> None:
    """The net-effect question is only answerable if the series never split."""

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    history = build_history(
        [
            _efficiency_result("v1", 30.0, device_name="laptop", day=1),
            _efficiency_result("v2", 27.0, device_name="laptop", day=2),
            _efficiency_result("v3", 11.0, device_name="desktop", day=3),
        ],
        BridgeIndex(),
        metric,
    )

    assert len(history.points) == 3
    assert {point.series for point in history.points} == {history.points[0].series}
    assert not any(point.starts_new_series for point in history.points)
    assert [point.environment_changed for point in history.points] == [
        False,
        False,
        True,
    ]
    assert "environment changed" in render_history(history)


def test_history_still_splits_when_the_workload_changes() -> None:
    """A workload change is a different measurement, so the line has to break."""

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    history = build_history(
        [
            _efficiency_result("v1", 27.0, device_name="laptop", plies=40, day=1),
            _efficiency_result("v2", 54.0, device_name="laptop", plies=80, day=2),
        ],
        BridgeIndex(),
        metric,
    )

    assert [point.starts_new_series for point in history.points] == [False, True]


def test_an_os_patch_alone_does_not_confound_a_delta() -> None:
    """Keying on the full platform string would flag every point release."""

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    baseline = _efficiency_result("checkpoint-a", 12.0, device_name="laptop")
    patched = _efficiency_result("checkpoint-b", 11.0, device_name="laptop")
    assert patched.execution is not None
    updated = patched.model_copy(
        update={
            "execution": patched.execution.model_copy(
                update={"platform": "fixture-1.2.4"}
            )
        }
    )

    report = build_delta_report([baseline, updated], BridgeIndex(), metrics=[metric])

    assert _delta(report).movement is Movement.BETTER
    assert _delta(report).environment == ()


def test_run_to_run_jitter_stops_reading_as_a_regression() -> None:
    """The reading this benchmark most often produces is noise, not a finding.

    Two readings of the same checkpoint minutes apart move by a fraction of a
    millisecond. A reading that measured no spread leaves the report able only
    to say the number moved, which is how sub-percent jitter gets written up as
    a regression.
    """

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    unknown = build_delta_report(
        [
            _efficiency_result("checkpoint-a", 12.000, device_name="laptop"),
            _efficiency_result("checkpoint-b", 12.008, device_name="laptop"),
        ],
        BridgeIndex(),
        metrics=[metric],
    )
    qualified = build_delta_report(
        [
            _efficiency_result(
                "checkpoint-a", 12.000, device_name="laptop", dispersion=0.1
            ),
            _efficiency_result(
                "checkpoint-b", 12.008, device_name="laptop", dispersion=0.1
            ),
        ],
        BridgeIndex(),
        metrics=[metric],
    )

    assert _delta(unknown).noise is NoiseVerdict.UNKNOWN
    assert _delta(unknown).noise_floor is None
    assert _delta(qualified).noise is NoiseVerdict.WITHIN
    assert _delta(qualified).noise_floor_source == ("6 process replicates on laptop")
    # The delta is still shown, so a small regression that repeats across
    # checkpoints stays visible rather than being filtered away.
    assert _delta(qualified).delta == pytest.approx(0.008)


def test_a_real_movement_still_clears_the_machine_floor() -> None:
    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    report = build_delta_report(
        [
            _efficiency_result(
                "checkpoint-a", 12.0, device_name="laptop", dispersion=0.1
            ),
            _efficiency_result(
                "checkpoint-b", 9.0, device_name="laptop", dispersion=0.1
            ),
        ],
        BridgeIndex(),
        metrics=[metric],
    )

    assert _delta(report).noise is NoiseVerdict.CLEARED
    assert _delta(report).movement is Movement.BETTER
