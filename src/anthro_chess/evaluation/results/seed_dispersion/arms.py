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

from anthro_chess.evaluation.results.comparability import (
    measurements_by_workload,
    training_health_readings,
)
from anthro_chess.evaluation.results.metrics import (
    BENCHMARK_COST_FAMILY,
    BENCHMARK_WALL_CLOCK_SECONDS,
    TRAINING_HEALTH_FAMILY,
    registered_metrics,
)
from anthro_chess.evaluation.results.records import (
    ResultEnvelope,
    default_checkpoint_label,
)
from anthro_chess.evaluation.results.seed_dispersion.dispersion import (
    ArmReading,
    SeedDispersion,
    SeedDispersionError,
    characterize,
)
from anthro_chess.evaluation.results.store import (
    ResultsStore,
    results_for_checkpoint,
)

#: The families withheld from the spread. Training health is what says whether
#: the spread applies at all, and benchmark cost times the machine rather than
#: the model, so a floor built from it would qualify a delta in what an
#: invocation cost.
_UNCHARACTERIZED_FAMILIES = frozenset(
    {TRAINING_HEALTH_FAMILY.identifier, BENCHMARK_COST_FAMILY.identifier}
)


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
    scored = results_for_checkpoint(results, run.checkpoint)
    if not scored:
        raise SeedDispersionError(
            f"nothing in the results store scored {run.checkpoint}; the arm has "
            "to be read before its spread can be"
        )
    metrics, fingerprints = _measurements(scored)
    if not metrics:
        raise SeedDispersionError(
            f"{run.checkpoint} has no scored metric to characterize"
        )
    try:
        return ArmReading(
            run_id=run.directory.name,
            seed=run.seed,
            checkpoint=run.checkpoint,
            wall_clock_seconds=run.elapsed_seconds,
            metrics=metrics,
            fingerprints=fingerprints,
            health=training_health_readings(scored),
        )
    except ValidationError as error:
        raise SeedDispersionError(
            f"the run at {run.directory} does not describe an arm: {error}"
        ) from error


def _measurements(
    scored: Sequence[ResultEnvelope],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, str]]]:
    """Return one checkpoint's values and series, per metric and workload.

    Grouped by the store's own rule rather than by a second one, because a
    report looks the stored spread up under the workload key it derives that
    way. A key derived differently here would miss every lookup and leave every
    row reading as though nothing had been characterized.
    """

    metrics: dict[str, dict[str, float]] = {}
    fingerprints: dict[str, dict[str, str]] = {}
    for definition in registered_metrics():
        if definition.family in _UNCHARACTERIZED_FAMILIES:
            continue
        cells = measurements_by_workload(
            scored,
            definition.identifier,
            workload_scoped=definition.execution_sensitive,
        )
        for workload, (_, found) in cells.items():
            metrics.setdefault(definition.identifier, {})[workload] = found.value
            fingerprints.setdefault(definition.identifier, {})[workload] = (
                found.fingerprint
            )
    return metrics, fingerprints


def _scoring_seconds(
    results: Sequence[ResultEnvelope],
    *,
    labels: set[str],
) -> float:
    """Return what scoring these arms' checkpoints cost, as recorded."""

    return sum(
        found.value
        for label in labels
        for envelope in results_for_checkpoint(results, label)
        if (found := envelope.measurement(BENCHMARK_WALL_CLOCK_SECONDS.identifier))
        is not None
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
        for envelope in results_for_checkpoint(results, run.checkpoint)
    } - {None}
    if len(recorded) != 1:
        raise SeedDispersionError(
            f"the readings for {run.checkpoint} do not agree on one training "
            "identity, so the spread they carry has no key to be stored against"
        )
    identity = recorded.pop()
    assert identity is not None  # the ``None`` was removed above
    return identity


__all__ = ["characterize_runs"]
