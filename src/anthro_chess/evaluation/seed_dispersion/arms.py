"""Reading a set of trained arms into the spread they showed.

The arms live in two places and neither alone is enough. The run directory says
what a run was — its seed, the horizon it reached, and what it cost — and the
results store says what its checkpoint scored. This joins them on the checkpoint
label both sides derive from the run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from anthro_chess.evaluation.results.comparability import UNSCOPED_WORKLOAD
from anthro_chess.evaluation.results.metrics import (
    TRAINING_HEALTH_FAMILY,
    MetricRegistryError,
    metric_definition,
    registered_metrics,
)
from anthro_chess.evaluation.results.records import (
    ResultEnvelope,
    default_checkpoint_label,
)
from anthro_chess.evaluation.results.store import ResultsStore
from anthro_chess.evaluation.seed_dispersion.dispersion import (
    ArmReading,
    SeedDispersion,
    SeedDispersionError,
    characterize,
)

#: What a recording benchmark writes beside its reading to say what the
#: invocation cost. Summed across the arms rather than characterized: it is a
#: property of the machine and the hour, and a spread over it would describe
#: neither the model nor the seed.
BENCHMARK_COST_METRIC = "benchmark.wall_clock_seconds"


def characterize_runs(
    run_directories: Sequence[Path],
    store: ResultsStore,
    *,
    notes: str | None = None,
) -> SeedDispersion:
    """Return the seed dispersion a set of trained arms shows."""

    results = store.results()
    readings = [_arm_reading(directory, results) for directory in run_directories]
    identities = {_identity(directory, results) for directory in run_directories}
    if len(identities) != 1:
        raise SeedDispersionError(
            "the arms do not share one training identity, so no single "
            f"configuration is being characterized: {sorted(identities)}"
        )
    horizons = {_final_step(directory) for directory in run_directories}
    if len(horizons) != 1:
        raise SeedDispersionError(
            "the arms were trained to different horizons, which the training "
            f"identity does not hold: {sorted(horizons)} step(s)"
        )
    return characterize(
        readings,
        training_sha256=identities.pop(),
        horizon_steps=horizons.pop(),
        scoring_seconds=_scoring_seconds(
            results,
            labels={reading.checkpoint for reading in readings},
        ),
        measured_at=datetime.now(tz=UTC),
        notes=notes,
    )


def _arm_reading(
    directory: Path,
    results: Sequence[ResultEnvelope],
) -> ArmReading:
    record = _run_record(directory)
    step = _final_step(directory)
    label = default_checkpoint_label(directory.name, step)
    scored = [envelope for envelope in results if envelope.checkpoint.label == label]
    if not scored:
        raise SeedDispersionError(
            f"nothing in the results store scored {label}; the arm has to be "
            "read before its spread can be"
        )
    metrics, health = _measurements(scored)
    if not metrics:
        raise SeedDispersionError(f"{label} has no scored metric to characterize")
    seed = record.get("seed")
    if not isinstance(seed, int):
        raise SeedDispersionError(
            f"the run record at {directory} states no initialization seed, so "
            "which arm it is cannot be established"
        )
    return ArmReading(
        run_id=directory.name,
        seed=seed,
        checkpoint=label,
        training_seconds=_elapsed_seconds(directory),
        metrics=metrics,
        health=health,
    )


def _measurements(
    scored: Sequence[ResultEnvelope],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Split one checkpoint's readings into what is floored and what is scoped.

    Training health is withheld from the spread because it is what says whether
    the spread applies at all. Benchmark cost is withheld because it times the
    machine: a floor built from it would qualify a delta in what an invocation
    cost, which is not a quantity any comparison here reads.
    """

    health_metrics = {
        definition.identifier
        for definition in registered_metrics(TRAINING_HEALTH_FAMILY.identifier)
    }
    metrics: dict[str, dict[str, float]] = {}
    health: dict[str, float] = {}
    for envelope in sorted(scored, key=lambda item: (item.recorded_at, item.result_id)):
        for measurement in envelope.measurements:
            if measurement.metric in health_metrics:
                health[measurement.metric] = measurement.value
                continue
            if measurement.metric == BENCHMARK_COST_METRIC:
                continue
            workload = UNSCOPED_WORKLOAD
            if _execution_sensitive(measurement.metric) and envelope.execution:
                workload = envelope.execution.workload_sha256
            metrics.setdefault(measurement.metric, {})[workload] = measurement.value
    return metrics, health


def _execution_sensitive(metric: str) -> bool:
    """Return whether a metric's declared workload is an input to its value.

    An unregistered identifier is treated as declaring no workload rather than
    refused. A store holding a reading whose metric has since been retired is
    a thing that happens, and it should cost that row its floor rather than the
    whole characterization.
    """

    try:
        return metric_definition(metric).execution_sensitive
    except MetricRegistryError:
        return False


def _scoring_seconds(
    results: Sequence[ResultEnvelope],
    *,
    labels: set[str],
) -> float:
    """Return what scoring these arms' checkpoints cost, as recorded."""

    return sum(
        measurement.value
        for envelope in results
        if envelope.checkpoint.label in labels
        for measurement in envelope.measurements
        if measurement.metric == BENCHMARK_COST_METRIC
    )


def _run_record(directory: Path) -> dict[str, object]:
    path = directory / "run.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise SeedDispersionError(
            f"run record at {path} cannot be read: {error}"
        ) from (error)
    if not isinstance(record, dict):
        raise SeedDispersionError(f"run record at {path} is not a record")
    return record


def _final_step(directory: Path) -> int:
    """Return the optimizer step the run's last logged interval reached."""

    step = _last_step_entry(directory).get("global_step")
    if not isinstance(step, int):
        raise SeedDispersionError(
            f"the metrics at {directory} state no final optimizer step"
        )
    return step


def _elapsed_seconds(directory: Path) -> float:
    """Return the wall clock the run took, as the run itself measured it."""

    elapsed = _last_step_entry(directory).get("elapsed_seconds")
    if not isinstance(elapsed, int | float) or elapsed <= 0.0:
        raise SeedDispersionError(f"the metrics at {directory} state no wall clock")
    return float(elapsed)


def _last_step_entry(directory: Path) -> dict[str, object]:
    path = directory / "metrics.jsonl"
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise SeedDispersionError(f"metrics at {path} cannot be read: {error}") from (
            error
        )
    for line in reversed(lines):
        if not line.strip():
            continue
        entry = json.loads(line)
        if isinstance(entry, dict) and entry.get("record") == "step":
            return entry
    raise SeedDispersionError(f"metrics at {path} hold no logged training step")


def _identity(directory: Path, results: Sequence[ResultEnvelope]) -> str:
    label = default_checkpoint_label(directory.name, _final_step(directory))
    recorded = {
        envelope.checkpoint.training_sha256
        for envelope in results
        if envelope.checkpoint.label == label
    } - {None}
    if len(recorded) != 1:
        raise SeedDispersionError(
            f"the readings for {label} do not agree on one training identity, "
            "so the spread they carry has no key to be stored against"
        )
    identity = recorded.pop()
    assert identity is not None  # the ``None`` was removed above
    return identity


__all__ = ["BENCHMARK_COST_METRIC", "characterize_runs"]
