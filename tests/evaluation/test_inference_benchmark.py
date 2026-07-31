"""What a checkpoint costs to play with, and what makes two costs comparable."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import chess
import pytest
import torch
from pydantic import ValidationError

import anthro_chess.evaluation.inference as inference_module
from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.data import DecisionContext
from anthro_chess.evaluation.inference import (
    INFERENCE_KIND,
    InferenceBenchmarkConfig,
    InferenceBenchmarkError,
    LatencyWorkloadConfig,
    ThroughputWorkloadConfig,
    _HistoryFactory,
    _measure_latency,
    _percentile,
    benchmark_inference,
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
    FloorEntry,
    MetricDelta,
    Movement,
    NoiseCharacterization,
    NoiseFloorIndex,
    NoiseVerdict,
    ReportError,
    ReportPivot,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    build_characterization,
    build_delta_report,
    build_environment_report,
    build_history,
    build_result,
    execution_reference,
    measurement,
    render_history,
    series_fingerprint,
    workload_digest,
)
from anthro_chess.evaluation.results.metrics import (
    INFERENCE_BATCH_THROUGHPUT,
    INFERENCE_FIRST_DECISION_SECONDS,
    INFERENCE_MODEL_LOAD_SECONDS,
    INFERENCE_MOVE_LATENCY_BY_PERCENTILE,
    INFERENCE_MOVE_LATENCY_MEAN,
)
from anthro_chess.inference import CheckpointModelRunner, ModelRunnerConfig
from anthro_chess.models import MoveModelBatch
from anthro_chess.runtime import GameSession, RuntimeConfig

#: A workload small enough for the CPU suite. The measured quantities are
#: unchanged; only the number of samples behind them is.
FAST_LATENCY = LatencyWorkloadConfig(
    reference_plies=4,
    sweep_plies=(0, 4),
    decisions=3,
    warmup_decisions=1,
    seed="test-latency",
)
FAST_THROUGHPUT = ThroughputWorkloadConfig(
    reference_batch_size=2,
    sweep_batch_sizes=(1, 2),
    history_plies=4,
    batches=2,
    warmup_batches=1,
    seed="test-throughput",
)


def _config(
    checkpoint: Path, **overrides: object
) -> ResolvedConfig[InferenceBenchmarkConfig]:
    fields: dict[str, object] = {
        "model": ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu"),
        "runtime": RuntimeConfig(seed=7),
        "latency": FAST_LATENCY,
        "throughput": FAST_THROUGHPUT,
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

    result = benchmark_inference(_config(checkpoint), store=store, detail=detail)

    latency = result.reference_latency
    assert latency.history_plies == FAST_LATENCY.reference_plies
    assert latency.decisions == FAST_LATENCY.decisions
    assert latency.percentiles[50] <= latency.percentiles[90]
    assert latency.percentiles[90] <= latency.percentiles[99]
    assert latency.minimum_ms <= latency.mean_ms <= latency.maximum_ms
    assert result.reference_throughput.batch_size == 2
    assert result.reference_throughput.decisions_per_second > 0.0
    assert result.cold_start.model_load_seconds > 0.0
    assert result.cold_start.first_decision_seconds > 0.0

    (envelope,) = result.envelopes
    assert envelope.kind == INFERENCE_KIND
    assert {item.metric for item in envelope.measurements} == {
        INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier,
        INFERENCE_MOVE_LATENCY_BY_PERCENTILE[90].identifier,
        INFERENCE_MOVE_LATENCY_BY_PERCENTILE[99].identifier,
        INFERENCE_MOVE_LATENCY_MEAN.identifier,
        INFERENCE_BATCH_THROUGHPUT.identifier,
        INFERENCE_MODEL_LOAD_SECONDS.identifier,
        INFERENCE_FIRST_DECISION_SECONDS.identifier,
    }
    assert result.recorded_paths and result.recorded_paths[0].exists()
    assert result.detail_paths and result.detail_paths[0].exists()


def test_cold_start_is_reported_apart_from_steady_state_latency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inference_run: Callable[..., Path],
) -> None:
    """Loading and first-call warmup must not inflate the percentiles.

    A benchmark that folded them in would report a checkpoint as slow to play
    with when it is only slow to start, and the two have different fixes.

    The startup cost is injected rather than measured. Loading this fixture
    checkpoint and deciding one move are both a few milliseconds on an idle
    machine, and which of the two comes out larger is a property of what else
    the machine is doing rather than of the benchmark, so comparing them
    ambiently decides which piece of noise won.
    """

    checkpoint = inference_run(tmp_path / "run", seed=6)
    delay_seconds = 0.05
    real_decide = inference_module._decide
    served = 0

    def slow_first_decision(session: GameSession) -> None:
        nonlocal served
        if served == 0:
            served += 1
            time.sleep(delay_seconds)
        real_decide(session)

    monkeypatch.setattr(inference_module, "_decide", slow_first_decision)

    result = benchmark_inference(_config(checkpoint))

    delay_ms = delay_seconds * 1000.0
    assert served == 1
    # The cold reading carries the startup cost the first decision paid.
    assert result.cold_start.first_decision_seconds * 1000.0 >= delay_ms
    # The steady-state percentiles do not, which is the whole separation.
    assert result.reference_latency.maximum_ms < delay_ms
    # Loading is still reported, on its own, as the other half of cold start.
    assert result.cold_start.model_load_seconds > 0.0


def test_warmup_decisions_are_excluded_from_the_percentiles(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """The measured window must start after the slow first calls."""

    checkpoint = inference_run(tmp_path / "run", seed=8)
    runner = CheckpointModelRunner.load(
        ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu")
    )
    warmup = 2
    delay_seconds = 0.05
    slow = _SlowFirstRunner(runner, slow_calls=warmup, delay_seconds=delay_seconds)
    session = GameSession(slow, config=RuntimeConfig(seed=7))
    config = LatencyWorkloadConfig(
        reference_plies=2,
        sweep_plies=(2,),
        decisions=4,
        warmup_decisions=warmup,
        seed="test-warmup",
    )

    sample = _measure_latency(session, slow, _HistoryFactory(), config, 2)

    assert slow.slow_calls_served == warmup
    assert sample.decisions == 4
    assert sample.maximum_ms < delay_seconds * 1000.0


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

    (envelope,) = benchmark_inference(_config(checkpoint)).envelopes

    assert envelope.execution is not None
    assert envelope.execution.device == "cpu"
    assert envelope.execution.precision == "float32"
    assert envelope.execution.cpu_threads == torch.get_num_threads()
    assert envelope.execution.workload["latency_reference_plies"] == 4
    # verify() recomputes every fingerprint from the record alone.
    envelope.verify()


def test_a_declared_workload_change_starts_a_new_series(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    checkpoint = inference_run(tmp_path / "run", seed=12)
    shallow = benchmark_inference(_config(checkpoint)).envelopes[0]
    deeper = benchmark_inference(
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


def test_extending_a_sweep_does_not_end_the_headline_series(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """The sweep is drill-down. Only the reference point decides identity."""

    checkpoint = inference_run(tmp_path / "run", seed=13)
    narrow = benchmark_inference(_config(checkpoint)).envelopes[0]
    wide = benchmark_inference(
        _config(
            checkpoint,
            latency=FAST_LATENCY.model_copy(update={"sweep_plies": (0, 2, 4)}),
        )
    ).envelopes[0]

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    narrow_measurement = narrow.measurement(metric)
    wide_measurement = wide.measurement(metric)

    assert narrow_measurement is not None
    assert wide_measurement is not None
    assert narrow_measurement.fingerprint == wide_measurement.fingerprint


def test_the_reference_point_is_measured_even_when_the_sweep_omits_it(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    checkpoint = inference_run(tmp_path / "run", seed=14)

    result = benchmark_inference(
        _config(
            checkpoint,
            latency=FAST_LATENCY.model_copy(update={"sweep_plies": (0,)}),
            throughput=FAST_THROUGHPUT.model_copy(update={"sweep_batch_sizes": (1,)}),
        )
    )

    assert result.reference_latency.history_plies == 4
    assert result.reference_throughput.batch_size == 2


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


def test_stacked_batches_carry_every_row_through_the_model(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    checkpoint = inference_run(tmp_path / "run", seed=15)
    runner = CheckpointModelRunner.load(
        ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu")
    )
    factory = _HistoryFactory()
    rows = [
        MoveModelBatch.from_decision_context(
            _context(factory.history("stack", 4, index)),
            device=runner.device,
        )
        for index in range(3)
    ]

    stacked = MoveModelBatch.stack(rows)
    logits = runner.action_logits(stacked)

    # Four played plies plus the decision ply the context adds.
    assert stacked.action_targets.shape == (3, 5)
    assert logits.shape[:2] == (3, 5)
    assert len(stacked.legal_action_ids) == 3
    assert torch.isfinite(logits).all()


def test_stacking_refuses_batches_of_different_lengths(tmp_path: Path) -> None:
    factory = _HistoryFactory()
    rows = [
        MoveModelBatch.from_decision_context(_context(factory.history("stack", 4, 0))),
        MoveModelBatch.from_decision_context(_context(factory.history("stack", 6, 0))),
    ]

    with pytest.raises(ValueError, match="same sequence length"):
        MoveModelBatch.stack(rows)


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


class _SlowFirstRunner:
    """A runner whose first calls are slow, standing in for a cold device."""

    def __init__(
        self,
        runner: CheckpointModelRunner,
        *,
        slow_calls: int,
        delay_seconds: float,
    ) -> None:
        self._runner = runner
        self._slow_calls = slow_calls
        self._delay_seconds = delay_seconds
        self.device = runner.device
        self.slow_calls_served = 0

    def predict(self, context: DecisionContext) -> torch.Tensor:
        """Return real logits, paying a startup cost on the first calls."""

        if self.slow_calls_served < self._slow_calls:
            self.slow_calls_served += 1
            time.sleep(self._delay_seconds)
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
) -> ResultEnvelope:
    """Build one recorded efficiency result on a named machine."""

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


def _context(history: tuple[chess.Move, ...]) -> DecisionContext:
    from anthro_chess.data import build_decision_context

    board = chess.Board()
    for move in history:
        board.push(move)
    return build_decision_context(board, tuple(board.move_stack), target_rating=None)


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


def _execution_floor(
    *,
    device_name: str,
    floor: float,
    plies: int = 40,
) -> NoiseCharacterization:
    """Return a characterized machine floor for the p50 latency series."""

    execution = execution_reference(
        device="cpu",
        device_name=device_name,
        precision="float32",
        torch_version="2.7.0",
        platform_key="Fixture-x86",
        platform="fixture-1.2.3",
        cpu_threads=8,
        workload={"latency_reference_plies": plies},
    )
    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    return build_characterization(
        kind="execution",
        method=PROCESS_REPLICATE_METHOD,
        replicates=6,
        processes=3,
        source=f"three processes on {device_name}",
        execution=execution,
        floors=[
            FloorEntry(
                metric=metric,
                fingerprint=series_fingerprint(
                    metric,
                    None,
                    execution.workload_component(),
                ),
                floor=floor,
                dispersion=floor / 2,
            )
        ],
        recorded_at=datetime(2026, 7, 1, tzinfo=UTC),
    )


def test_run_to_run_jitter_stops_reading_as_a_regression() -> None:
    """The reading this benchmark most often produces is noise, not a finding.

    Two readings of the same checkpoint minutes apart move by a fraction of a
    millisecond. With no characterized floor the report can only say the number
    moved, which is how sub-percent jitter gets written up as a regression.
    """

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    results = [
        _efficiency_result("checkpoint-a", 12.000, device_name="laptop"),
        _efficiency_result("checkpoint-b", 12.008, device_name="laptop"),
    ]

    unknown = build_delta_report(results, BridgeIndex(), metrics=[metric])
    qualified = build_delta_report(
        results,
        BridgeIndex(),
        metrics=[metric],
        floors=NoiseFloorIndex([_execution_floor(device_name="laptop", floor=0.4)]),
    )

    assert _delta(unknown).noise is NoiseVerdict.UNKNOWN
    assert _delta(unknown).noise_floor is None
    assert _delta(qualified).noise is NoiseVerdict.WITHIN
    assert _delta(qualified).noise_floor_kind == "execution"
    # The delta is still shown, so a small regression that repeats across
    # checkpoints stays visible rather than being filtered away.
    assert _delta(qualified).delta == pytest.approx(0.008)


def test_a_real_movement_still_clears_the_machine_floor() -> None:
    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    report = build_delta_report(
        [
            _efficiency_result("checkpoint-a", 12.0, device_name="laptop"),
            _efficiency_result("checkpoint-b", 9.0, device_name="laptop"),
        ],
        BridgeIndex(),
        metrics=[metric],
        floors=NoiseFloorIndex([_execution_floor(device_name="laptop", floor=0.4)]),
    )

    assert _delta(report).noise is NoiseVerdict.CLEARED
    assert _delta(report).movement is Movement.BETTER


def test_a_floor_from_one_machine_does_not_qualify_another_machines_delta() -> None:
    """The series is continuous across machines; the noise in it is not."""

    metric = INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier
    report = build_delta_report(
        [
            _efficiency_result("checkpoint-a", 12.000, device_name="laptop"),
            _efficiency_result("checkpoint-b", 12.008, device_name="workstation"),
        ],
        BridgeIndex(),
        metrics=[metric],
        floors=NoiseFloorIndex([_execution_floor(device_name="laptop", floor=0.4)]),
    )

    assert _delta(report).noise is NoiseVerdict.UNKNOWN
    assert _delta(report).movement is Movement.CONFOUNDED
