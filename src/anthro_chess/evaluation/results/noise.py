"""Characterized noise floors, and the rule for reading a delta against one.

A delta is not a finding until it is larger than the noise in the measurement.
Without a floor, a report can only say that a number moved, and a reader
comparing two checkpoints will confidently describe movement that is seed luck.

Four noise sources are kept apart, because conflating them is the usual
mistake. They are estimated differently and they answer different questions,
but they all reduce to the same reportable quantity: the **dispersion** of a
metric across independent replicates of one noise source.

A floor is that dispersion expressed as a delta. Two independent measurements
of an unchanged quantity differ with a standard deviation of ``sqrt(2)`` times
the dispersion of one of them, so a floor is that difference at a declared
normal coverage. The result is directly comparable to a reported delta, which
is what a report needs and what a raw standard deviation is not.

Floors are stored beside the results they qualify, keyed by the same series
fingerprint as any other measurement. That is deliberate: when the pool, the
view, or a metric definition moves, the floor stops matching and the report
says the floor is unknown, rather than silently applying a stale constant to a
measurement it no longer describes.

An execution floor is keyed by one thing more. Decision 0018 keeps the machine
out of an efficiency series on purpose, so that a latency history stays
continuous across a hardware change; but the noise in a timing measurement *is*
the machine, so a floor measured on a laptop describes nothing about a reading
taken on a workstation. Such a floor therefore carries the execution it was
characterized under, and a report applies it only where that environment
matches on both sides of the delta.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from pydantic import Field, model_validator

from anthro_chess.evaluation.results.comparability import BridgeIndex
from anthro_chess.evaluation.results.records import (
    MAXIMUM_SUMMARY_BYTES,
    EnvironmentRecord,
    ExecutionRecord,
    Identifier,
    NoiseFloor,
    NoiseFloorKind,
    ResultModel,
    Sha256Hex,
    canonical_json,
)

#: Version 2 added the execution scope an execution floor is only valid within.
CHARACTERIZATION_VERSION = 2

#: Two-sided normal coverage for a 95% interval. A floor is a claim about how
#: far apart two measurements land when nothing changed, so it needs a stated
#: confidence rather than a bare standard deviation.
DEFAULT_COVERAGE = 1.96

#: How a dispersion was estimated. The method is recorded because the three
#: noise kinds are not interchangeable and neither are their estimators.
BOOTSTRAP_METHOD = "bootstrap-over-games"
REPLICATE_METHOD = "independent-replicates"
PROCESS_REPLICATE_METHOD = "repeated-process-replicates"


class NoiseCharacterizationError(ValueError):
    """Raised when a noise floor cannot be estimated or recorded."""


class FloorEntry(ResultModel):
    """One metric's floor, on one series."""

    metric: str = Field(min_length=1)
    fingerprint: Sha256Hex
    floor: float = Field(ge=0.0)
    dispersion: float = Field(ge=0.0)
    #: How many independent sampling units the dispersion was measured over,
    #: when the floor scales with that count. Set for data-sampling floors,
    #: where it is the number of games and is what makes the sizing question
    #: computable; absent for the kinds that do not scale this way.
    sampling_units: int | None = Field(default=None, ge=1)
    #: How much of ``dispersion`` repeating the measurement inside one process
    #: already reproduces. Set for an execution floor measured with more than
    #: one reading per process, and absent otherwise. It is a diagnostic rather
    #: than a floor: a value close to ``dispersion`` says the machine's noise is
    #: visible without paying for a second process, and a much smaller one says
    #: process-level effects dominate and cheap in-process replication would
    #: understate the floor.
    within_process_dispersion: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _validate_values(self) -> FloorEntry:
        if not math.isfinite(self.floor) or not math.isfinite(self.dispersion):
            raise ValueError(f"floor for {self.metric} must be a finite number")
        if self.within_process_dispersion is not None and not math.isfinite(
            self.within_process_dispersion
        ):
            raise ValueError(f"floor for {self.metric} must be a finite number")
        return self


class NoiseCharacterization(ResultModel):
    """Every floor one calibration pass produced, and how it produced them.

    One record per pass rather than one per metric. A calibration covers every
    metric it could estimate at once, and splitting that into twenty committed
    files would grow the summary tier without making any floor easier to find.
    """

    characterization_version: int = Field(ge=1)
    characterization_id: str = Field(pattern=r"^[a-f0-9]{16}$")
    recorded_at: datetime
    kind: NoiseFloorKind
    method: Identifier
    replicates: int = Field(ge=2)
    coverage: float = Field(gt=0.0)
    source: str = Field(min_length=1)
    environment: EnvironmentRecord
    #: How many separate processes the replicates were spread across. Recorded
    #: for an execution floor, where a reading compared in a report is one
    #: process's, so the between-process component is the one that has to be in
    #: the floor at all.
    processes: int | None = Field(default=None, ge=2)
    #: The machine and workload an execution floor describes. Present only for
    #: that kind: every other floor is a property of the measurement rather than
    #: of where it ran, and scoping one of those to a machine would discard a
    #: floor that is still perfectly valid.
    execution: ExecutionRecord | None = None
    floors: tuple[FloorEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_floors(self) -> NoiseCharacterization:
        metrics = [entry.metric for entry in self.floors]
        if len(set(metrics)) != len(metrics):
            raise ValueError("a characterization may report each metric once")
        if metrics != sorted(metrics):
            raise ValueError("floors must be ordered by metric identifier")
        if self.recorded_at.tzinfo is None:
            raise ValueError("recorded_at must carry a time zone")
        return self

    @model_validator(mode="after")
    def _validate_execution_scope(self) -> NoiseCharacterization:
        """Keep the machine scope and the noise kind agreed.

        An execution floor without its execution would be applied to every
        machine that ever recorded the series, which is the one thing this kind
        must not do. An execution record on any other kind would narrow a floor
        that does not depend on where it was measured.
        """

        if self.kind == "execution":
            if self.execution is None:
                raise ValueError(
                    "an execution floor is a property of a machine and a "
                    "workload, so it must record the execution it was "
                    "characterized under"
                )
            if self.processes is None:
                raise ValueError(
                    "an execution floor must record how many processes its "
                    "replicates were spread across"
                )
        elif self.execution is not None or self.processes is not None:
            raise ValueError(
                f"a {self.kind} floor does not depend on where it was measured, "
                "so it carries no execution scope"
            )
        return self

    def environment_key(self) -> str:
        """Return the machine identity this characterization is valid on."""

        return environment_key(self.execution)

    def entry(self, metric: str) -> FloorEntry | None:
        """Return one metric's floor, if this characterization covers it."""

        for candidate in self.floors:
            if candidate.metric == metric:
                return candidate
        return None

    def as_floor(self, entry: FloorEntry) -> NoiseFloor:
        """Return the floor in the shape a measurement and a report read."""

        return NoiseFloor(value=entry.floor, kind=self.kind, source=self.source)

    def verify(self) -> None:
        """Reject a record too large for the committed summary tier."""

        size = len(canonical_json(self.as_record()))
        if size > MAXIMUM_SUMMARY_BYTES:
            raise NoiseCharacterizationError(
                f"noise characterization {self.characterization_id} is {size} "
                f"bytes; the committed summary tier caps a record at "
                f"{MAXIMUM_SUMMARY_BYTES}"
            )

    def as_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record written to the store."""

        return self.model_dump(mode="json")


class NoiseFloorIndex:
    """Resolve the floors that apply to a series.

    Floors are looked up through the same bridge equivalence as measurements,
    so a series that was legitimately rejoined keeps the floor characterized
    on either side of the seam. A fingerprint with no characterization
    resolves to nothing at all rather than to zero.

    An execution floor additionally has to match the machine, since that is
    what it measured. A reading from another machine, or a delta whose two
    sides ran on different ones, resolves to no execution floor rather than to
    a borrowed one.
    """

    def __init__(
        self,
        characterizations: Iterable[NoiseCharacterization] = (),
        bridges: BridgeIndex | None = None,
    ) -> None:
        self._bridges = bridges if bridges is not None else BridgeIndex()
        self._by_series: dict[
            tuple[str, str, str, str], tuple[datetime, NoiseFloor]
        ] = {}
        for characterization in sorted(
            characterizations,
            key=lambda item: (item.recorded_at, item.characterization_id),
        ):
            for entry in characterization.floors:
                key = (
                    self._bridges.series(entry.fingerprint),
                    entry.metric,
                    characterization.kind,
                    characterization.environment_key(),
                )
                recorded = (
                    characterization.recorded_at,
                    characterization.as_floor(entry),
                )
                previous = self._by_series.get(key)
                # The most recent characterization of a series wins. An older
                # one describes a measurement that has since been re-estimated,
                # not a second independent opinion to average in.
                if previous is None or previous[0] <= recorded[0]:
                    self._by_series[key] = recorded

    def floors(
        self,
        metric: str,
        fingerprint: str,
        *,
        executions: Sequence[ExecutionRecord | None] = (),
    ) -> tuple[NoiseFloor, ...]:
        """Return every characterized floor for one metric's series, by kind.

        ``executions`` are the executions the floors would qualify — both
        operands of a delta, or the single reading being annotated. A
        machine-scoped floor is returned only when every one of them was
        measured on the machine it was characterized on.
        """

        series = self._bridges.series(fingerprint)
        measured = {environment_key(execution) for execution in executions}
        return tuple(
            self._by_series[key][1]
            for key in sorted(self._by_series)
            if key[0] == series
            and key[1] == metric
            and (not key[3] or measured == {key[3]})
        )


def floor_from_dispersion(
    dispersion: float,
    *,
    coverage: float = DEFAULT_COVERAGE,
) -> float:
    """Return the delta that noise alone produces, at a declared coverage.

    Two independent measurements of an unchanged quantity differ with a
    standard deviation of ``sqrt(2)`` times the dispersion of either one, so
    this is the difference a reader should expect to see when nothing changed.
    """

    if dispersion < 0.0 or not math.isfinite(dispersion):
        raise NoiseCharacterizationError(
            "a dispersion must be a finite, non-negative number"
        )
    if coverage <= 0.0 or not math.isfinite(coverage):
        raise NoiseCharacterizationError("coverage must be a finite, positive factor")
    return coverage * math.sqrt(2.0) * dispersion


def environment_key(execution: ExecutionRecord | None) -> str:
    """Return the machine identity a floor measured here would be valid on.

    Built from the same coarse environment fields a report attributes a delta
    to, so an operating system point release does not invalidate a floor while
    a device or precision change does. No execution at all means no machine
    scope, which is what every floor other than an execution floor carries.
    """

    if execution is None:
        return ""
    return sha256(canonical_json(execution.environment())).hexdigest()[:16]


@dataclass(frozen=True)
class ProcessDispersion:
    """The spread of one metric across repeated measurements, decomposed.

    ``total`` is what a floor is built from: the spread of a single reading
    taken by a fresh process, which is what a report actually compares.
    ``within`` is how much of that a repeat inside one process reproduces, and
    is a diagnostic about where the noise lives rather than a second floor.
    """

    total: float
    within: float | None


def replicate_dispersion(values: Sequence[float]) -> float:
    """Return the dispersion across independent replicate measurements.

    This is the estimator for evaluation noise, where the replicates are the
    same checkpoint re-measured on the same data under different benchmark
    seeds, and for training noise, where they are the same configuration
    trained from different seeds.
    """

    if len(values) < 2:
        raise NoiseCharacterizationError(
            "a dispersion needs at least two independent replicates"
        )
    if any(not math.isfinite(value) for value in values):
        raise NoiseCharacterizationError("replicate values must be finite")
    return statistics.stdev(values)


def process_dispersion(groups: Sequence[Sequence[float]]) -> ProcessDispersion:
    """Return the dispersion of one metric across repeated measurements.

    ``groups`` holds one sequence of readings per process. The total is taken
    over every reading rather than over per-process means, because a reading
    compared in a report carries both the process-level and the within-process
    variation; averaging the repeats away first would report a floor narrower
    than the measurement a reader is actually judging.

    At least two processes are required. Repeating inside one process sees
    allocator and kernel state that a second process would pay for again, so it
    cannot observe the component most likely to dominate.
    """

    if len(groups) < 2:
        raise NoiseCharacterizationError(
            "an execution floor needs replicates from at least two processes; "
            "repeating inside one process cannot see process-level variation"
        )
    total = replicate_dispersion([value for group in groups for value in group])
    repeated = [group for group in groups if len(group) > 1]
    if not repeated:
        return ProcessDispersion(total=total, within=None)
    freedom = sum(len(group) - 1 for group in repeated)
    pooled = sum(statistics.variance(group) * (len(group) - 1) for group in repeated)
    return ProcessDispersion(total=total, within=math.sqrt(pooled / freedom))


def process_replicate_floors(
    replicates: Mapping[str, Sequence[Sequence[float]]],
    *,
    fingerprints: Mapping[str, str],
    coverage: float = DEFAULT_COVERAGE,
) -> tuple[FloorEntry, ...]:
    """Return one execution floor per metric from its repeated measurements.

    Each metric arrives as one sequence of readings per process. Every
    replicate of an execution floor is the same series by construction — the
    same checkpoint under the same declared workload — so the fingerprint is
    supplied once per metric rather than carried on each reading.
    """

    entries: list[FloorEntry] = []
    for metric in sorted(replicates):
        fingerprint = fingerprints.get(metric)
        if fingerprint is None:
            raise NoiseCharacterizationError(
                f"no series fingerprint was supplied for {metric}"
            )
        dispersion = process_dispersion(replicates[metric])
        entries.append(
            FloorEntry(
                metric=metric,
                fingerprint=fingerprint,
                floor=floor_from_dispersion(dispersion.total, coverage=coverage),
                dispersion=dispersion.total,
                within_process_dispersion=dispersion.within,
            )
        )
    return tuple(entries)


def games_to_resolve(entry: FloorEntry, effect: float) -> int:
    """Return how many games are needed to resolve an effect of a given size.

    A sampling floor shrinks with the square root of the games behind it, so
    the count an axis needs is computable from one measured floor rather than
    guessed. This is what sizes a pool generation.
    """

    if entry.sampling_units is None:
        raise NoiseCharacterizationError(
            f"the floor for {entry.metric} does not scale with a game count, so "
            "it cannot size an evaluation input"
        )
    if effect <= 0.0 or not math.isfinite(effect):
        raise NoiseCharacterizationError("an effect size must be finite and positive")
    if entry.floor == 0.0:
        return 1
    required = entry.sampling_units * (entry.floor / effect) ** 2
    return max(1, math.ceil(required))


def build_characterization(
    *,
    kind: NoiseFloorKind,
    method: str,
    replicates: int,
    source: str,
    floors: Sequence[FloorEntry],
    coverage: float = DEFAULT_COVERAGE,
    environment: EnvironmentRecord | None = None,
    execution: ExecutionRecord | None = None,
    processes: int | None = None,
    recorded_at: datetime | None = None,
) -> NoiseCharacterization:
    """Assemble a verified characterization with a content-derived identity."""

    ordered = tuple(sorted(floors, key=lambda entry: entry.metric))
    try:
        record = NoiseCharacterization(
            characterization_version=CHARACTERIZATION_VERSION,
            characterization_id="0" * 16,
            recorded_at=recorded_at or datetime.now(tz=UTC),
            kind=kind,
            method=method,
            replicates=replicates,
            coverage=coverage,
            source=source,
            environment=environment or EnvironmentRecord.capture(),
            processes=processes,
            execution=execution,
            floors=ordered,
        )
    except ValueError as error:
        raise NoiseCharacterizationError(str(error)) from error
    payload = record.model_dump(mode="json")
    payload.pop("characterization_id", None)
    identified = record.model_copy(
        update={
            "characterization_id": sha256(canonical_json(payload)).hexdigest()[:16],
        }
    )
    identified.verify()
    return identified


def replicate_floors(
    values_by_metric: Mapping[str, Sequence[tuple[str, float]]],
    *,
    coverage: float = DEFAULT_COVERAGE,
) -> tuple[FloorEntry, ...]:
    """Return one floor per metric from its recorded replicate measurements.

    Each replicate arrives as its own fingerprint and value. Replicates that
    do not share a series are a caller error rather than a wider floor, so
    that check belongs to whoever selected them.
    """

    entries: list[FloorEntry] = []
    for metric in sorted(values_by_metric):
        replicates = values_by_metric[metric]
        dispersion = replicate_dispersion([value for _, value in replicates])
        entries.append(
            FloorEntry(
                metric=metric,
                fingerprint=replicates[-1][0],
                floor=floor_from_dispersion(dispersion, coverage=coverage),
                dispersion=dispersion,
            )
        )
    return tuple(entries)


__all__ = [
    "BOOTSTRAP_METHOD",
    "CHARACTERIZATION_VERSION",
    "DEFAULT_COVERAGE",
    "PROCESS_REPLICATE_METHOD",
    "REPLICATE_METHOD",
    "FloorEntry",
    "NoiseCharacterization",
    "NoiseCharacterizationError",
    "NoiseFloorIndex",
    "ProcessDispersion",
    "build_characterization",
    "environment_key",
    "floor_from_dispersion",
    "games_to_resolve",
    "process_dispersion",
    "process_replicate_floors",
    "replicate_dispersion",
    "replicate_floors",
]
