"""What a checkpoint costs to play with, and what makes two costs comparable."""

from __future__ import annotations

import copy
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import chess
import pytest
import torch
from pydantic import ValidationError

import anthro_chess.evaluation.inference as inference_module
from anthro_chess.chess import action_vocabulary_identity
from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.data import DecisionContext, encoding_identity
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
    BenchmarkReference,
    BridgeIndex,
    CheckpointReference,
    Comparability,
    DeltaReport,
    DetailStore,
    ExecutionRecord,
    MetricDelta,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    build_delta_report,
    build_result,
    execution_reference,
    measurement,
)
from anthro_chess.evaluation.results.metrics import (
    INFERENCE_BATCH_THROUGHPUT,
    INFERENCE_FIRST_DECISION_SECONDS,
    INFERENCE_MODEL_LOAD_SECONDS,
    INFERENCE_MOVE_LATENCY_BY_PERCENTILE,
    INFERENCE_MOVE_LATENCY_MEAN,
)
from anthro_chess.inference import CheckpointModelRunner, ModelRunnerConfig
from anthro_chess.models import CausalMoveModel, MoveModelBatch, MoveModelConfig
from anthro_chess.runtime import GameSession, RuntimeConfig
from anthro_chess.training.checkpoints import save_training_checkpoint

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


def test_benchmark_reports_latency_throughput_and_cold_start(tmp_path: Path) -> None:
    checkpoint = _write_run(tmp_path / "run", seed=5)
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
) -> None:
    """Loading and first-call warmup must not inflate the percentiles.

    A benchmark that folded them in would report a checkpoint as slow to play
    with when it is only slow to start, and the two have different fixes.
    """

    checkpoint = _write_run(tmp_path / "run", seed=6)

    result = benchmark_inference(_config(checkpoint))

    load_ms = result.cold_start.model_load_seconds * 1000.0
    assert result.reference_latency.maximum_ms < load_ms
    assert (
        result.cold_start.first_decision_seconds * 1000.0
        >= result.reference_latency.minimum_ms
    )


def test_warmup_decisions_are_excluded_from_the_percentiles(tmp_path: Path) -> None:
    """The measured window must start after the slow first calls."""

    checkpoint = _write_run(tmp_path / "run", seed=8)
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


def test_a_measurement_on_another_machine_is_not_a_faster_checkpoint(
    tmp_path: Path,
) -> None:
    """The comparability layer already owns this; efficiency just declares it."""

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

    assert _comparability(within) is Comparability.SAME_SERIES
    assert _comparability(across) is Comparability.INCOMPARABLE
    # The faster machine posted the better number and is still not progress.
    assert _delta(across).current == 9.0
    assert _delta(across).delta is None
    assert "execution" in {difference.field for difference in across.provenance}


def test_a_tampered_execution_record_cannot_reproduce_its_fingerprints() -> None:
    """Relabelling the machine after the fact must not launder a comparison."""

    result = _efficiency_result("checkpoint-a", 12.0, device_name="laptop")
    relabelled = result.model_copy(
        update={
            "execution": result.execution.model_copy(  # type: ignore[union-attr]
                update={"device_name": "workstation"}
            )
        }
    )

    with pytest.raises(ResultRecordError, match="does not reproduce"):
        relabelled.verify()


def test_the_recorded_execution_reproduces_its_own_series_identity(
    tmp_path: Path,
) -> None:
    checkpoint = _write_run(tmp_path / "run", seed=10)

    (envelope,) = benchmark_inference(_config(checkpoint)).envelopes

    assert envelope.execution is not None
    assert envelope.execution.device == "cpu"
    assert envelope.execution.precision == "float32"
    assert envelope.execution.cpu_threads == torch.get_num_threads()
    assert envelope.execution.workload["latency_reference_plies"] == 4
    # verify() recomputes every fingerprint from the record alone.
    envelope.verify()


def test_a_declared_workload_change_starts_a_new_series(tmp_path: Path) -> None:
    checkpoint = _write_run(tmp_path / "run", seed=12)
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


def test_extending_a_sweep_does_not_end_the_headline_series(tmp_path: Path) -> None:
    """The sweep is drill-down. Only the reference point decides identity."""

    checkpoint = _write_run(tmp_path / "run", seed=13)
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
) -> None:
    checkpoint = _write_run(tmp_path / "run", seed=14)

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


def test_stacked_batches_carry_every_row_through_the_model(tmp_path: Path) -> None:
    checkpoint = _write_run(tmp_path / "run", seed=15)
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
) -> ResultEnvelope:
    """Build one recorded efficiency result on a named machine."""

    execution = execution_reference(
        device="cpu",
        device_name=device_name,
        precision="float32",
        torch_version="2.7.0",
        platform="fixture",
        cpu_threads=8,
        workload={"latency_reference_plies": 40},
    )
    return build_result(
        kind=INFERENCE_KIND,
        benchmark=BenchmarkReference(name="inference-efficiency", version=1),
        checkpoint=CheckpointReference(label=label, step=1),
        execution=execution,
        measurements=[
            measurement(
                INFERENCE_MOVE_LATENCY_BY_PERCENTILE[50].identifier,
                latency_ms,
                execution=execution.as_component(),
            )
        ],
        recorded_at=datetime(2026, 7, 1 if label.endswith("a") else 2, tzinfo=UTC),
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


def _write_run(path: Path, *, seed: int) -> Path:
    """Write a retained run holding one tiny compatible checkpoint."""

    torch.manual_seed(seed)
    path.mkdir(parents=True, exist_ok=True)
    config = MoveModelConfig(
        piece_embedding_dim=2,
        action_embedding_dim=2,
        model_dim=4,
        attention_heads=1,
        transformer_layers=1,
        feedforward_dim=8,
        dropout=0.0,
    )
    model = CausalMoveModel(config)
    model_identity = model.identity()
    resolved_config = {
        "config": {"model": config.model_dump(mode="json")},
        "provenance": {"source": None, "overrides": []},
    }
    execution = {
        "device": "cpu",
        "backend": "cpu",
        "precision": "float32",
        "parameter_dtype": "float32",
        "determinism": "strict",
        "gradient_accumulation_steps": 1,
        "phase_profiling": False,
    }
    metadata = {
        "resolved_config": copy.deepcopy(resolved_config),
        "code": {"package_version": "test", "git_revision": "test"},
        "data": {},
        "model": copy.deepcopy(model_identity),
        "action_vocabulary": action_vocabulary_identity(),
        "encoding": encoding_identity(),
        "execution": copy.deepcopy(execution),
    }
    checkpoint = path / "checkpoints" / "step-00000001.pt"
    save_training_checkpoint(
        checkpoint,
        global_step=1,
        counters={"processed_positions": 1},
        model_state=model.state_dict(),
        optimizer_state={},
        scheduler_state=None,
        scaler_state=None,
        loader_state={},
        compatibility={
            "training_config": {},
            "data": {},
            "model": copy.deepcopy(model_identity),
            "action_vocabulary": action_vocabulary_identity(),
            "encoding": encoding_identity(),
        },
        metadata=metadata,
        device="cpu",
    )
    (path / "run.json").write_text(
        json.dumps(
            {
                "version": 3,
                "resolved_config": copy.deepcopy(resolved_config),
                "model": copy.deepcopy(model_identity),
                "action_vocabulary": action_vocabulary_identity(),
                "encoding": encoding_identity(),
                "execution": copy.deepcopy(execution),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_a_recorded_workload_must_produce_its_own_digest() -> None:
    """The readable workload and the digest naming its series stay agreed."""

    record = execution_reference(
        device="cpu",
        device_name="laptop",
        precision="float32",
        torch_version="2.7.0",
        platform="fixture",
        workload={"latency_reference_plies": 40},
    )

    with pytest.raises(ValidationError, match="does not produce the recorded digest"):
        ExecutionRecord(
            **{**record.model_dump(), "workload": {"latency_reference_plies": 80}}
        )
