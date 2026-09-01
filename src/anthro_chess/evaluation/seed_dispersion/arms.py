"""Reading a set of trained arms into the spread they showed.

The arms live in two places and neither alone is enough. The run directory says
what a run was — its seed, the checkpoint it finished at, and what it cost — and
the results store says what that checkpoint scored. This joins them on the label
both sides derive from the run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

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


@dataclass(frozen=True)
class _Run:
    """What one arm's own artifacts say about it, read once each."""

    directory: Path
    seed: int
    #: The step whose checkpoint the run record describes, which lags the last
    #: step executed when a run died between checkpoints. Taking the last logged
    #: step instead would name a checkpoint that was never saved.
    completed_step: int
    elapsed_seconds: float

    @property
    def checkpoint(self) -> str:
        """Return the label every reading of this run's checkpoint carries."""

        return default_checkpoint_label(self.directory.name, self.completed_step)


def characterize_runs(
    run_directories: Sequence[Path],
    store: ResultsStore,
    *,
    notes: str | None = None,
) -> SeedDispersion:
    """Return the seed dispersion a set of trained arms shows."""

    results = store.results()
    runs = [_read_run(directory) for directory in run_directories]
    identities = {_identity(run, results) for run in runs}
    if len(identities) != 1:
        raise SeedDispersionError(
            "the arms do not share one training identity, so no single "
            f"configuration is being characterized: {sorted(identities)}"
        )
    horizons = {run.completed_step for run in runs}
    if len(horizons) != 1:
        raise SeedDispersionError(
            "the arms were trained to different horizons, which the training "
            f"identity does not hold: {sorted(horizons)} step(s)"
        )
    readings = [_arm_reading(run, results) for run in runs]
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


def _arm_reading(run: _Run, results: Sequence[ResultEnvelope]) -> ArmReading:
    scored = [
        envelope for envelope in results if envelope.checkpoint.label == run.checkpoint
    ]
    if not scored:
        raise SeedDispersionError(
            f"nothing in the results store scored {run.checkpoint}; the arm has "
            "to be read before its spread can be"
        )
    metrics, fingerprints, health = _measurements(scored)
    if not metrics:
        raise SeedDispersionError(
            f"{run.checkpoint} has no scored metric to characterize"
        )
    try:
        return ArmReading(
            run_id=run.directory.name,
            seed=run.seed,
            checkpoint=run.checkpoint,
            training_seconds=run.elapsed_seconds,
            metrics=metrics,
            fingerprints=fingerprints,
            health=health,
        )
    except ValidationError as error:
        raise SeedDispersionError(
            f"the run at {run.directory} does not describe an arm: {error}"
        ) from error


def _measurements(
    scored: Sequence[ResultEnvelope],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, str]],
    dict[str, float],
]:
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
    fingerprints: dict[str, dict[str, str]] = {}
    health: dict[str, float] = {}
    for envelope in sorted(scored, key=lambda item: (item.recorded_at, item.result_id)):
        for found in envelope.measurements:
            if found.metric in health_metrics:
                health[found.metric] = found.value
                continue
            if found.metric == BENCHMARK_COST_METRIC:
                continue
            workload = UNSCOPED_WORKLOAD
            if _execution_sensitive(found.metric) and envelope.execution:
                workload = envelope.execution.workload_sha256
            metrics.setdefault(found.metric, {})[workload] = found.value
            fingerprints.setdefault(found.metric, {})[workload] = found.fingerprint
    return metrics, fingerprints, health


def _execution_sensitive(metric: str) -> bool:
    """Return whether a metric's declared workload is an input to its value.

    An unregistered identifier is treated as declaring no workload rather than
    refused. A store holding a reading whose metric has since been retired is a
    thing that happens, and it should cost that row its floor rather than the
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
        found.value
        for envelope in results
        if envelope.checkpoint.label in labels
        for found in envelope.measurements
        if found.metric == BENCHMARK_COST_METRIC
    )


def _read_run(directory: Path) -> _Run:
    """Return what one run's two artifacts say, reading each of them once."""

    record = _json(directory / "run.json")
    if not isinstance(record, dict):
        raise SeedDispersionError(f"the run record at {directory} is not a record")
    seed = record.get("seed")
    if not isinstance(seed, int):
        raise SeedDispersionError(
            f"the run record at {directory} states no initialization seed, so "
            "which arm it is cannot be established"
        )
    optimization = record.get("optimization")
    step = (
        optimization.get("completed_steps") if isinstance(optimization, dict) else None
    )
    if not isinstance(step, int):
        raise SeedDispersionError(
            f"the run record at {directory} names no completed step, so which "
            "checkpoint its readings describe cannot be established"
        )
    elapsed = _last_step_entry(directory).get("elapsed_seconds")
    if not isinstance(elapsed, int | float) or elapsed <= 0.0:
        raise SeedDispersionError(f"the metrics at {directory} state no wall clock")
    return _Run(
        directory=directory,
        seed=seed,
        completed_step=step,
        elapsed_seconds=float(elapsed),
    )


def _last_step_entry(directory: Path) -> dict[str, object]:
    path = directory / "metrics.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SeedDispersionError(
            f"the metrics at {path} cannot be read: {error}"
        ) from (error)
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            # A run killed mid-write leaves a partial last line, which is the
            # first one a reversed scan reaches.
            continue
        if isinstance(entry, dict) and entry.get("record") == "step":
            return entry
    raise SeedDispersionError(f"the metrics at {path} hold no logged training step")


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SeedDispersionError(f"{path} cannot be read: {error}") from error


def _identity(run: _Run, results: Sequence[ResultEnvelope]) -> str:
    recorded = {
        envelope.checkpoint.training_sha256
        for envelope in results
        if envelope.checkpoint.label == run.checkpoint
    } - {None}
    if len(recorded) != 1:
        raise SeedDispersionError(
            f"the readings for {run.checkpoint} do not agree on one training "
            "identity, so the spread they carry has no key to be stored against"
        )
    identity = recorded.pop()
    assert identity is not None  # the ``None`` was removed above
    return identity


__all__ = ["BENCHMARK_COST_METRIC", "characterize_runs"]
