"""Deferred read-back, interval accounting, and the efficiency result shape."""

from __future__ import annotations

import platform
from datetime import UTC, datetime

import pytest
import torch

from anthro_chess.evaluation.results import (
    CheckpointReference,
    ExecutionRecord,
    ResultRecordError,
    ResultsStore,
    metric_definition,
    registered_metrics,
)
from anthro_chess.training.efficiency import (
    TRAINING_EFFICIENCY_KIND,
    DeferredStepTotals,
    StepTotals,
    TrainingEfficiencyConfig,
    TrainingEfficiencyError,
    TrainingEfficiencyMonitor,
    TrainingEfficiencySummary,
    build_efficiency_result,
    efficiency_measurements,
    execution_record,
    record_efficiency,
    render_efficiency,
    workload_record,
)

CPU = torch.device("cpu")

MODEL_IDENTITY = {"name": "fixture-model", "version": 1}


def _workload(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "dataset_sha256": "a" * 64,
        "loader_configuration_sha256": "b" * 64,
        "model_identity": MODEL_IDENTITY,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "determinism": "relaxed",
        "profile_phases": False,
    }
    record.update(overrides)
    return workload_record(**record)  # type: ignore[arg-type]


def _execution(**overrides: object) -> ExecutionRecord:
    return execution_record(_workload(**overrides), device=CPU, precision="float32")


def _summary(**overrides: object) -> TrainingEfficiencySummary:
    values: dict[str, object] = {
        "window_steps": 4,
        "window_intervals": 2,
        "window_active_positions": 800,
        "window_training_seconds": 2.0,
        "window_padded_positions": 1000,
        "processed_positions": 1200,
        "training_seconds": 3.0,
        "run_seconds": 5.0,
        "startup_seconds": 1.0,
        "checkpoint_seconds": 0.25,
        "evaluation_seconds": 0.5,
        "validation_seconds": 0.125,
        "instrumentation_seconds": 0.125,
        "minimum_interval_step_seconds": 0.4,
        "maximum_interval_step_seconds": 0.6,
        "peak_allocated_memory_bytes": 2048,
        "peak_driver_memory_bytes": 4096,
        "probe_steps": 2,
        "probe_seconds": 1.4,
    }
    values.update(overrides)
    return TrainingEfficiencySummary(**values)  # type: ignore[arg-type]


def _totals(**overrides: object) -> StepTotals:
    values: dict[str, object] = {
        "steps": 1,
        "loss_sum": 1.0,
        "final_step_loss_sum": 1.0,
        "active_positions": 10,
        "window_active_positions": 10,
        "probe_active_positions": 0,
        "finite": True,
    }
    values.update(overrides)
    return StepTotals(**values)  # type: ignore[arg-type]


def test_deferred_totals_bucket_positions_without_reading_them_back() -> None:
    totals = DeferredStepTotals(CPU)
    mask = torch.ones((2, 3), dtype=torch.bool)

    totals.begin_step()
    totals.observe(torch.tensor(1.5), mask, window=False, probe=False)
    totals.end_step()
    totals.begin_step()
    totals.observe(torch.tensor(2.5), mask, window=True, probe=False)
    totals.observe(torch.tensor(3.5), mask, window=True, probe=False)
    totals.end_step()

    assert totals.padded_positions == 18
    drained = totals.drain()

    assert drained.steps == 2
    assert drained.active_positions == 18
    # Only the second step declared itself in the window, and the probe arm
    # takes nothing, so the three buckets partition the same micro-batches.
    assert drained.window_active_positions == 12
    assert drained.probe_active_positions == 0
    assert drained.loss_sum == pytest.approx(7.5)
    # The interval summed three micro-batches; the reported step summed two.
    assert drained.final_step_loss_sum == pytest.approx(6.0)
    assert drained.finite is True


def test_probe_positions_leave_the_window_bucket() -> None:
    totals = DeferredStepTotals(CPU)
    mask = torch.ones((1, 4), dtype=torch.bool)

    totals.begin_step()
    totals.observe(torch.tensor(1.0), mask, window=True, probe=True)
    totals.end_step()
    drained = totals.drain()

    assert drained.active_positions == 4
    assert drained.probe_active_positions == 4
    assert drained.window_active_positions == 0


def test_deferred_totals_report_a_non_finite_loss_at_the_drain() -> None:
    totals = DeferredStepTotals(CPU)
    mask = torch.ones((1, 2), dtype=torch.bool)

    totals.begin_step()
    totals.observe(torch.tensor(1.0), mask, window=True, probe=False)
    totals.end_step()
    totals.begin_step()
    totals.observe(torch.tensor(float("nan")), mask, window=True, probe=False)
    totals.end_step()

    drained = totals.drain()

    assert drained.finite is False
    # The flag survives its own reset, so the next interval starts clean.
    assert totals.drain().finite is True


def test_drain_resets_every_accumulator() -> None:
    totals = DeferredStepTotals(CPU)
    mask = torch.ones((1, 2), dtype=torch.bool)

    totals.begin_step()
    totals.observe(torch.tensor(1.0), mask, window=True, probe=False)
    totals.end_step()
    totals.drain()

    empty = totals.drain()
    assert empty.steps == 0
    assert empty.active_positions == 0
    assert empty.window_active_positions == 0
    assert empty.loss_sum == 0.0


def test_synchronize_reads_back_without_disturbing_the_totals() -> None:
    totals = DeferredStepTotals(CPU)
    mask = torch.ones((1, 5), dtype=torch.bool)

    totals.begin_step()
    totals.observe(torch.tensor(2.0), mask, window=True, probe=True)
    totals.synchronize()
    totals.end_step()

    drained = totals.drain()
    assert drained.active_positions == 5
    assert drained.loss_sum == pytest.approx(2.0)


def test_warmup_steps_stay_outside_the_throughput_window() -> None:
    monitor = TrainingEfficiencyMonitor(
        TrainingEfficiencyConfig(
            warmup_steps=2,
            synchronization_probe_every_intervals=0,
        ),
        device=CPU,
    )
    monitor.begin_optimization(starting_step=0)

    for global_step in (1, 2, 3):
        monitor.begin_interval(global_step)
        assert monitor.interval_in_window is (global_step >= 3)
        monitor.begin_step()
        monitor.end_step()
        monitor.close_interval(_totals(active_positions=100), padded_positions=120)

    summary = monitor.summary()
    assert summary.window_steps == 1
    assert summary.window_intervals == 1
    # Every step's positions are processed; only the window's are measured.
    assert summary.processed_positions == 300
    assert summary.window_active_positions == 10
    assert summary.window_padded_positions == 120


def test_warmup_offset_follows_a_resumed_starting_step() -> None:
    monitor = TrainingEfficiencyMonitor(
        TrainingEfficiencyConfig(warmup_steps=1),
        device=CPU,
    )
    monitor.begin_optimization(starting_step=50)

    monitor.begin_interval(51)
    assert monitor.interval_in_window is False
    monitor.begin_step()
    monitor.end_step()
    monitor.close_interval(_totals(), padded_positions=1)

    monitor.begin_interval(52)
    assert monitor.interval_in_window is True


def test_the_probe_arm_alternates_whole_intervals() -> None:
    monitor = TrainingEfficiencyMonitor(
        TrainingEfficiencyConfig(
            warmup_steps=0,
            synchronization_probe_every_intervals=2,
        ),
        device=CPU,
    )
    monitor.begin_optimization(starting_step=0)

    arms: list[bool] = []
    for global_step in range(1, 5):
        monitor.begin_interval(global_step)
        arms.append(monitor.interval_probing)
        monitor.begin_step()
        monitor.end_step()
        monitor.close_interval(
            _totals(
                window_active_positions=0 if monitor.interval_probing else 10,
                probe_active_positions=10 if monitor.interval_probing else 0,
            ),
            padded_positions=12,
        )

    assert arms == [False, True, False, True]
    summary = monitor.summary()
    assert summary.window_steps == 2
    assert summary.probe_steps == 2


def test_a_disabled_probe_never_draws_the_synchronizing_arm() -> None:
    monitor = TrainingEfficiencyMonitor(
        TrainingEfficiencyConfig(
            warmup_steps=0,
            synchronization_probe_every_intervals=0,
        ),
        device=CPU,
    )
    monitor.begin_optimization(starting_step=0)

    for global_step in range(1, 5):
        monitor.begin_interval(global_step)
        assert monitor.interval_probing is False
        monitor.begin_step()
        monitor.end_step()
        monitor.close_interval(_totals(), padded_positions=1)

    assert monitor.summary().probe_steps == 0


def test_a_warmup_interval_never_draws_the_probe_arm() -> None:
    monitor = TrainingEfficiencyMonitor(
        TrainingEfficiencyConfig(
            warmup_steps=4,
            synchronization_probe_every_intervals=1,
        ),
        device=CPU,
    )
    monitor.begin_optimization(starting_step=0)

    monitor.begin_interval(1)
    assert monitor.interval_probing is False


def test_in_step_overhead_is_subtracted_and_between_step_overhead_is_not() -> None:
    monitor = TrainingEfficiencyMonitor(
        TrainingEfficiencyConfig(
            warmup_steps=0,
            synchronization_probe_every_intervals=0,
        ),
        device=CPU,
    )
    monitor.begin_optimization(starting_step=0)
    monitor.begin_interval(1)
    monitor.begin_step()
    monitor.charge_step_overhead(0.5, kind="instrumentation")
    monitor.end_step()
    # Work between two steps was never inside one, so it is bucketed without
    # being subtracted from a span that did not contain it.
    monitor.charge(2.0, kind="evaluation")
    monitor.charge(1.0, kind="checkpoint")
    monitor.close_interval(_totals(), padded_positions=1)

    summary = monitor.summary()
    assert summary.instrumentation_seconds == pytest.approx(0.5)
    assert summary.evaluation_seconds == pytest.approx(2.0)
    assert summary.checkpoint_seconds == pytest.approx(1.0)
    # The subtraction cannot drive an interval below zero.
    assert summary.window_training_seconds >= 0.0
    assert summary.window_training_seconds < 0.5


def test_an_unknown_overhead_bucket_is_refused() -> None:
    monitor = TrainingEfficiencyMonitor(TrainingEfficiencyConfig(), device=CPU)

    with pytest.raises(TrainingEfficiencyError, match="unknown overhead bucket"):
        monitor.charge(1.0, kind="mystery")


def test_the_monitor_refuses_overlapping_intervals() -> None:
    monitor = TrainingEfficiencyMonitor(TrainingEfficiencyConfig(), device=CPU)
    monitor.begin_optimization(starting_step=0)
    monitor.begin_interval(1)

    with pytest.raises(TrainingEfficiencyError, match="already open"):
        monitor.begin_interval(2)


def test_closing_without_an_open_interval_is_refused() -> None:
    monitor = TrainingEfficiencyMonitor(TrainingEfficiencyConfig(), device=CPU)

    with pytest.raises(TrainingEfficiencyError, match="no interval is open"):
        monitor.close_interval(_totals(), padded_positions=1)


def test_a_cpu_run_reports_no_device_memory() -> None:
    monitor = TrainingEfficiencyMonitor(TrainingEfficiencyConfig(), device=CPU)
    monitor.begin_optimization(starting_step=0)

    assert monitor.peak_allocated_memory_bytes is None
    assert monitor.peak_driver_memory_bytes is None


def test_derived_quantities_follow_the_window_rather_than_the_run() -> None:
    summary = _summary()

    assert summary.active_positions_per_second == pytest.approx(400.0)
    assert summary.step_seconds == pytest.approx(0.5)
    assert summary.active_position_fraction == pytest.approx(0.8)
    # Two of five seconds were training, so three fifths were overhead.
    assert summary.overhead_fraction == pytest.approx(0.4)
    assert summary.probe_step_seconds == pytest.approx(0.7)
    assert summary.step_synchronization_cost_seconds == pytest.approx(0.2)


def test_an_unmeasured_window_reports_absence_rather_than_zero() -> None:
    summary = _summary(
        window_steps=0,
        window_intervals=0,
        window_active_positions=0,
        window_training_seconds=0.0,
        window_padded_positions=0,
        probe_steps=0,
        probe_seconds=0.0,
    )

    assert summary.active_positions_per_second is None
    assert summary.step_seconds is None
    assert summary.active_position_fraction is None
    assert summary.step_synchronization_cost_seconds is None


def test_a_probe_with_no_deferred_arm_reports_no_cost() -> None:
    summary = _summary(window_steps=0, window_training_seconds=0.0)

    assert summary.step_synchronization_cost_seconds is None


def test_every_training_efficiency_metric_is_execution_sensitive() -> None:
    metrics = registered_metrics("training-efficiency")

    assert metrics
    for definition in metrics:
        assert definition.execution_sensitive is True
        assert definition.projection is None
        # Identity has to fit the report's metric column or its row loses
        # alignment; the registry is held to the width rather than the reverse.
        assert len(definition.identifier) <= 38


def test_measurements_carry_the_workload_fingerprint() -> None:
    execution = _execution()

    values = efficiency_measurements(_summary(), execution)

    assert {value.metric for value in values} == {
        "training.active_positions_per_second",
        "training.step_seconds",
        "training.processed_positions",
        "training.training_seconds",
        "training.active_position_fraction",
        "training.overhead_fraction",
        "training.peak_device_memory_bytes",
        "training.step_sync_cost_seconds",
    }
    other = execution_record(
        _workload(batch_size=8),
        device=CPU,
        precision="float32",
    )
    changed = efficiency_measurements(_summary(), other)
    by_metric = {value.metric: value.fingerprint for value in values}
    for value in changed:
        assert value.fingerprint != by_metric[value.metric]


def test_an_unmeasurable_quantity_is_omitted_rather_than_reported_as_zero() -> None:
    values = efficiency_measurements(
        _summary(peak_driver_memory_bytes=None, probe_steps=0, probe_seconds=0.0),
        _execution(),
    )

    metrics = {value.metric for value in values}
    assert "training.peak_device_memory_bytes" not in metrics
    assert "training.step_sync_cost_seconds" not in metrics
    assert "training.active_positions_per_second" in metrics


def test_a_run_too_short_to_reach_steady_state_still_records_its_budget() -> None:
    """A thin run is what a quality-versus-time report joins its early points to.

    Refusing to record it would leave the beginning of every training curve
    empty, which is where the interesting part of a budget comparison is.
    """

    short = _summary(
        window_steps=0,
        window_intervals=0,
        window_active_positions=0,
        window_training_seconds=0.0,
        window_padded_positions=0,
        peak_driver_memory_bytes=None,
        probe_steps=0,
        probe_seconds=0.0,
    )

    metrics = {value.metric for value in efficiency_measurements(short, _execution())}

    assert metrics == {
        "training.processed_positions",
        "training.training_seconds",
        "training.overhead_fraction",
    }


def test_the_workload_digests_what_decided_the_work() -> None:
    record = _workload()

    assert record["effective_batch_size"] == 8
    assert record["determinism"] == "relaxed"
    assert "warmup_steps" not in record
    assert "steps" not in record
    # The architecture decides the work; the weights do not.
    assert record["model_sha256"] != MODEL_IDENTITY


def test_the_environment_is_recorded_outside_series_identity() -> None:
    execution = _execution()

    assert execution.device == "cpu"
    assert execution.platform_key == (
        f"{platform.system() or 'unknown'}-{platform.machine() or 'unknown'}"
    )
    assert execution.cpu_threads == torch.get_num_threads()
    assert "device" not in execution.workload
    assert "platform_key" not in execution.workload


def test_a_result_verifies_its_own_workload_fingerprint() -> None:
    envelope = build_efficiency_result(
        _summary(),
        checkpoint=CheckpointReference(label="run-step-00000010", step=10),
        execution=_execution(),
        recorded_at=datetime(2026, 7, 30, 12, 0, tzinfo=UTC),
    )

    assert envelope.kind == TRAINING_EFFICIENCY_KIND
    assert envelope.execution is not None
    envelope.verify()
    for value in envelope.measurements:
        assert value.fingerprint == envelope.expected_fingerprint(value.metric)
        assert metric_definition(value.metric).family == "training-efficiency"


def test_recording_appends_one_result_to_the_store(tmp_path: object) -> None:
    store = ResultsStore(tmp_path)  # type: ignore[arg-type]
    checkpoint = CheckpointReference(label="run-step-00000010", step=10)

    envelope, paths = record_efficiency(
        _summary(),
        checkpoint=checkpoint,
        execution=_execution(),
        store=store,
    )

    assert len(paths) == 1
    assert paths[0].is_file()
    assert [result.result_id for result in store.results()] == [envelope.result_id]


def test_measuring_without_a_store_records_nothing() -> None:
    envelope, paths = record_efficiency(
        _summary(),
        checkpoint=CheckpointReference(label="run-step-00000010", step=10),
        execution=_execution(),
        store=None,
    )

    assert paths == ()
    assert envelope.measurements


def test_a_store_rejection_surfaces_as_an_efficiency_error(tmp_path: object) -> None:
    store = ResultsStore(tmp_path)  # type: ignore[arg-type]

    def refuse(_envelope: object) -> None:
        raise ResultRecordError("refused")

    store.append = refuse  # type: ignore[assignment,method-assign]
    with pytest.raises(TrainingEfficiencyError, match="refused"):
        record_efficiency(
            _summary(),
            checkpoint=CheckpointReference(label="run-step-00000010", step=10),
            execution=_execution(),
            store=store,
        )


def test_the_rendered_summary_names_each_overhead_bucket() -> None:
    text = render_efficiency(_summary())

    assert "active positions/s" in text
    assert "startup" in text
    assert "checkpoint" in text
    assert "evaluation" in text
    assert "validation" in text
    assert "instrumentation" in text
    assert "peak memory" in text
    assert "per-step sync" in text


def test_the_rendered_summary_omits_what_was_not_measured() -> None:
    text = render_efficiency(
        _summary(
            peak_driver_memory_bytes=None,
            peak_allocated_memory_bytes=None,
            probe_steps=0,
            probe_seconds=0.0,
        )
    )

    assert "peak memory" not in text
    assert "per-step sync" not in text
