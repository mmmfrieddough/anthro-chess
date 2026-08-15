"""Deferred read-back, interval accounting, and the efficiency result shape."""

from __future__ import annotations

import platform
from datetime import UTC, datetime
from hashlib import sha256

import pytest
import torch

from anthro_chess.evaluation.results import (
    AxisChange,
    BridgeIndex,
    CheckpointReference,
    Comparability,
    ExecutionRecord,
    Movement,
    NoiseVerdict,
    ReportError,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    build_delta_report,
    build_environment_report,
    build_history,
    metric_definition,
    registered_metrics,
    render_report,
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
    coordinate_record,
    efficiency_measurements,
    execution_record,
    record_efficiency,
    render_efficiency,
    workload_record,
)

from accelerators import requires_training_accelerator

CPU = torch.device("cpu")

MODEL_IDENTITY = {"name": "fixture-model", "version": 1}


def _coordinates(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "dataset_sha256": "a" * 64,
        "loader_configuration_sha256": "b" * 64,
        "model_identity": MODEL_IDENTITY,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "determinism": "relaxed",
        "matmul_precision": "highest",
        "profile_phases": False,
    }
    record.update(overrides)
    return coordinate_record(**record)  # type: ignore[arg-type]


def _execution(**overrides: object) -> ExecutionRecord:
    return execution_record(_coordinates(**overrides), device=CPU, precision="float32")


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


@pytest.mark.gpu
@requires_training_accelerator("cuda")
def test_cuda_peak_memory_holds_a_transient_freed_before_the_step_ends() -> None:
    """The reported peak is the step's high-water mark, not its boundary.

    A step's activations are gone by the time it ends, so a monitor reading the
    *current* allocator figure at ``end_step`` reports what survives the step
    rather than what it needed. On a real CUDA run of the shipped baseline at
    batch 16 that read 23.6 MB where the step had reached 134.6 MB, against
    207.6 MB reserved — a pair that says the allocator holds nine times what it
    uses.
    """

    device = torch.device("cuda")
    monitor = TrainingEfficiencyMonitor(TrainingEfficiencyConfig(), device=device)
    monitor.begin_optimization(starting_step=0)
    resident = torch.zeros(1024, dtype=torch.float32, device=device)
    boundary = monitor.peak_allocated_memory_bytes
    assert boundary is not None

    monitor.begin_interval(1)
    monitor.begin_step()
    transient = torch.zeros(8 * 1024 * 1024, dtype=torch.float32, device=device)
    transient_bytes = transient.element_size() * transient.numel()
    del transient
    monitor.end_step()

    assert resident.numel() == 1024  # the step's own allocation is not the peak
    peak = monitor.peak_allocated_memory_bytes
    assert peak is not None
    assert peak >= boundary + transient_bytes
    driver = monitor.peak_driver_memory_bytes
    assert driver is not None
    assert driver >= peak


@pytest.mark.gpu
@requires_training_accelerator("cuda")
def test_cuda_peak_memory_ignores_what_the_process_did_before_the_run() -> None:
    """A run reports its own peak rather than the process's.

    CUDA keeps its high-water marks per process and never decays them, so a
    second run in one process would otherwise inherit the first one's peak and
    a resumed run would report whatever its checkpoint load transiently held.
    """

    device = torch.device("cuda")
    earlier = torch.zeros(16 * 1024 * 1024, dtype=torch.float32, device=device)
    earlier_bytes = earlier.element_size() * earlier.numel()
    del earlier

    monitor = TrainingEfficiencyMonitor(TrainingEfficiencyConfig(), device=device)
    monitor.begin_optimization(starting_step=0)
    monitor.begin_interval(1)
    monitor.begin_step()
    monitor.end_step()

    peak = monitor.peak_allocated_memory_bytes
    assert peak is not None
    assert peak < earlier_bytes


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


def test_every_training_efficiency_metric_says_why_it_carries_no_floor() -> None:
    """The obligation decision 0043 leaves on a reading that measures no spread.

    A training reading cannot measure one: the only replicate of it is a second
    training run. Saying so is what makes a report read ``unqualifiable``
    instead of sending a reader after a spread nothing can produce.
    """

    for definition in registered_metrics("training-efficiency"):
        assert definition.no_sampling_floor_reason is not None


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
    # A different model, batch, and corpus stay on the same series, because
    # subtracting across them is the comparison this family exists for.
    other = execution_record(
        _coordinates(batch_size=64, dataset_sha256="f" * 64),
        device=CPU,
        precision="float32",
    )
    changed = efficiency_measurements(_summary(), other)
    by_metric = {value.metric: value.fingerprint for value in values}
    for value in changed:
        assert value.fingerprint == by_metric[value.metric]


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


def test_series_identity_holds_only_the_benchmark_version() -> None:
    """Anything a reader might subtract across must stay out of identity.

    The family exists to answer what a model or setup change cost, so freezing
    the model, the batch, or the corpus into the fingerprint would refuse its
    headline question.
    """

    assert workload_record() == {"benchmark_version": 1}


def test_the_conditions_are_recorded_without_reaching_the_digest() -> None:
    execution = _execution()

    assert execution.coordinates["effective_batch_size"] == 8
    assert execution.coordinates["determinism"] == "relaxed"
    # A setting that moves throughput has to be here or a report shows the jump
    # with nothing moved to attribute it to.
    assert execution.coordinates["matmul_precision"] == "highest"
    # The architecture is a coordinate, digested only into its own field.
    assert execution.coordinates["model_sha256"] != MODEL_IDENTITY
    assert "warmup_steps" not in execution.coordinates
    assert "steps" not in execution.coordinates
    # None of it reaches series identity.
    assert execution.workload == {"benchmark_version": 1}
    assert execution.workload_sha256 == _execution(batch_size=64).workload_sha256
    assert (
        execution.workload_sha256 == _execution(matmul_precision="high").workload_sha256
    )


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


def _recorded(
    label: str,
    at: datetime,
    *,
    training_sha256: str | None = None,
    **coordinates: object,
) -> ResultEnvelope:
    return build_efficiency_result(
        _summary(),
        checkpoint=CheckpointReference(
            label=label,
            step=100,
            parameter_sha256=sha256(label.encode()).hexdigest(),
            training_sha256=training_sha256,
        ),
        execution=_execution(**coordinates),
        recorded_at=at,
    )


def test_a_model_change_is_compared_and_attributed_rather_than_refused() -> None:
    """The family's headline question, which freezing the model would refuse."""

    before = _recorded("before", datetime(2026, 7, 1, tzinfo=UTC))
    after = _recorded(
        "after",
        datetime(2026, 7, 8, tzinfo=UTC),
        model_identity={"name": "fixture-model", "version": 2},
    )

    report = build_delta_report(
        [before, after],
        BridgeIndex(),
        metrics=["training.active_positions_per_second"],
    )
    row = report.families[0].metrics[0]

    assert row.comparability is Comparability.SAME_SERIES
    assert row.delta == pytest.approx(0.0)
    assert row.movement is Movement.CONFOUNDED
    assert row.attribution is not None
    assert row.attribution.conditions is AxisChange.CHANGED
    assert [difference.field for difference in row.conditions] == ["model_sha256"]
    rendered = render_report(report)
    assert "conditions changed: model_sha256" in rendered
    assert "the declared conditions moved as well" in rendered


def test_a_regenerated_corpus_is_named_rather_than_read_as_a_slowdown() -> None:
    """The confounder that changes neither machine nor checkpoint label."""

    before = _recorded("before", datetime(2026, 7, 1, tzinfo=UTC))
    after = _recorded(
        "after",
        datetime(2026, 7, 8, tzinfo=UTC),
        dataset_sha256="f" * 64,
    )

    row = (
        build_delta_report(
            [before, after],
            BridgeIndex(),
            metrics=["training.active_positions_per_second"],
        )
        .families[0]
        .metrics[0]
    )

    assert row.delta is not None
    assert row.movement is Movement.CONFOUNDED
    assert [difference.field for difference in row.conditions] == ["dataset_sha256"]


def test_an_unchanged_setup_still_reads_as_a_verdict() -> None:
    """Confounding has to be earned, or every row would carry the label.

    Two runs of one configuration on one machine differ only in their weights,
    which is the axis the checkpoint pivot varies on purpose.
    """

    before = _recorded("before", datetime(2026, 7, 1, tzinfo=UTC))
    after = build_efficiency_result(
        _summary(window_active_positions=1600),
        checkpoint=CheckpointReference(
            label="after",
            step=100,
            parameter_sha256=sha256(b"after").hexdigest(),
        ),
        execution=_execution(),
        recorded_at=datetime(2026, 7, 8, tzinfo=UTC),
    )

    row = (
        build_delta_report(
            [before, after],
            BridgeIndex(),
            metrics=["training.active_positions_per_second"],
        )
        .families[0]
        .metrics[0]
    )

    assert row.attribution is not None
    assert row.attribution.conditions is AxisChange.UNCHANGED
    assert row.attribution.environment is AxisChange.UNCHANGED
    assert row.movement is Movement.BETTER
    assert row.conditions == ()


def test_a_training_delta_is_unqualifiable_rather_than_left_unknown() -> None:
    """``unknown`` would send a reader after work nobody can do here.

    A floor for these would cost replicate training runs, so the verdict has to
    say the reading cannot produce one rather than that none has been produced.
    """

    before = _recorded("before", datetime(2026, 7, 1, tzinfo=UTC))
    after = _recorded("after", datetime(2026, 7, 8, tzinfo=UTC))

    report = build_delta_report([before, after], BridgeIndex())
    rows = [row for family in report.families for row in family.metrics]

    assert len(rows) == len(registered_metrics("training-efficiency"))
    for row in rows:
        assert row.noise is NoiseVerdict.UNQUALIFIABLE, row.metric
        assert row.noise_floor is None, row.metric


def test_the_environment_pivot_pins_conditions_rather_than_weights() -> None:
    """Two machines never share weights, so pinning them would forbid the ask."""

    laptop = _recorded("run-step-00000100", datetime(2026, 7, 1, tzinfo=UTC))
    workstation = build_efficiency_result(
        _summary(window_active_positions=3200),
        checkpoint=CheckpointReference(
            label="run-step-00000100",
            step=100,
            # A second run of the same configuration: same architecture and
            # corpus, necessarily different weights.
            parameter_sha256=sha256(b"other-weights").hexdigest(),
        ),
        execution=execution_record(
            _coordinates(),
            device=torch.device("cpu"),
            precision="float32",
        ).model_copy(update={"device_name": "workstation"}),
        recorded_at=datetime(2026, 7, 8, tzinfo=UTC),
    )

    row = (
        build_environment_report(
            [laptop, workstation],
            BridgeIndex(),
            metrics=["training.active_positions_per_second"],
        )
        .families[0]
        .metrics[0]
    )

    assert row.delta is not None
    # The conditions held, so this is a verdict on the machine despite the
    # weights differing.
    assert row.movement is Movement.BETTER


def test_a_training_identity_the_upgrade_moved_is_not_a_confound_here() -> None:
    """The arithmetic a machine works at is inside the training identity.

    So the two arms of an upgrade question land on two identities by
    construction, and reading that as a caveat would refuse the only comparison
    this pivot exists to make.
    """

    laptop = _recorded(
        "run-step-00000100",
        datetime(2026, 7, 1, tzinfo=UTC),
        training_sha256="4d" * 32,
    )
    workstation = build_efficiency_result(
        _summary(window_active_positions=3200),
        checkpoint=CheckpointReference(
            label="run-step-00000100",
            step=100,
            parameter_sha256=sha256(b"workstation-weights").hexdigest(),
            training_sha256="5e" * 32,
        ),
        execution=execution_record(
            _coordinates(),
            device=CPU,
            precision="bfloat16",
        ).model_copy(update={"device_name": "workstation"}),
        recorded_at=datetime(2026, 7, 8, tzinfo=UTC),
    )

    report = build_environment_report(
        [laptop, workstation],
        BridgeIndex(),
        metrics=["training.active_positions_per_second"],
    )
    row = report.families[0].metrics[0]

    assert row.training is AxisChange.CHANGED
    assert row.movement is Movement.BETTER
    assert "Training identity" not in render_report(report)


def test_the_environment_pivot_refuses_a_changed_configuration() -> None:
    """Pinning on conditions has to be as strict as pinning on parameters."""

    laptop = _recorded("run-step-00000100", datetime(2026, 7, 1, tzinfo=UTC))
    workstation = build_efficiency_result(
        _summary(),
        checkpoint=laptop.checkpoint,
        execution=execution_record(
            _coordinates(batch_size=64),
            device=torch.device("cpu"),
            precision="float32",
        ).model_copy(update={"device_name": "workstation"}),
        recorded_at=datetime(2026, 7, 8, tzinfo=UTC),
    )

    with pytest.raises(ReportError, match="declared conditions"):
        build_environment_report([laptop, workstation], BridgeIndex())


def test_long_run_history_stays_one_line_across_setup_changes() -> None:
    """The question a fragmented series destroys: are we drifting slower?"""

    points = [
        _recorded("first", datetime(2026, 7, 1, tzinfo=UTC)),
        _recorded(
            "second",
            datetime(2026, 7, 8, tzinfo=UTC),
            model_identity={"name": "fixture-model", "version": 2},
        ),
        _recorded("third", datetime(2026, 7, 15, tzinfo=UTC), batch_size=64),
        _recorded("fourth", datetime(2026, 7, 22, tzinfo=UTC), dataset_sha256="f" * 64),
    ]

    history = build_history(
        points,
        BridgeIndex(),
        "training.active_positions_per_second",
    )

    assert len(history.points) == 4
    assert not any(point.starts_new_series for point in history.points)
    assert len({point.series for point in history.points}) == 1


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
