"""How far apart two arms of one frozen training configuration land.

Every floor elsewhere in this package is combined from what the two readings'
own units could have moved, so none of them sees the training run. Two arms
differ by their initialization seed as well as by the change under test, and a
delta inside that spread is seed luck wearing a result's clothes.

The spread is measurable, but only for a configuration that stands still.
``docs/decisions/0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md``
owns why that is the ablation vehicle and nothing else, and
``docs/decisions/0076-the-vehicle-is-width-128-at-the-target-regime.md`` owns
the identity it is stored against.

Three things about the stored record are load-bearing rather than incidental.

**It is found by exact digest or not at all.** A floor that can be found
approximately can be applied to a configuration it did not measure, which is
the failure the whole design exists to prevent.

**It describes the control, not the arm.** It is measured on baseline arms, so
the treatment's own dispersion is assumed to match. Where a treatment's training
health departs from what the vehicle's arms showed, that assumption fails in the
one direction that matters: an unstable arm's spread is wider than the floor
allows, so the floor reads too narrow and noise clears it.

**It describes one horizon.** ``training_sha256`` excludes the step budget by
construction, which is what lets a cooldown branch match its trunk, so a reading
taken at another horizon matches this record's key without having been shown to
share its spread.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import Field, ValidationError, model_validator

from anthro_chess.evaluation.results.comparability import UNSCOPED_WORKLOAD
from anthro_chess.evaluation.results.noise import (
    DEFAULT_COVERAGE,
    NoiseCharacterizationError,
    measured_dispersion,
    replicate_dispersion,
    self_combined_floor,
)
from anthro_chess.evaluation.results.records import (
    MetricDispersion,
    ResultModel,
    Sha256Hex,
)
from anthro_chess.evaluation.results.store import canonical_readable_json

#: How a seed dispersion was estimated, named beside the two estimators in
#: ``noise`` so a stored spread never has to be guessed at. The replicate is a
#: whole training run rather than a resample or a process.
SEED_ARM_METHOD = "seed-replicate-arms"

#: Where the characterizations this package can find are checked in, one file
#: per training identity. Package data rather than a store directory: the
#: lookup has to answer from an installed wheel, where no results root exists.
DATA_DIRECTORY = Path(__file__).parent / "data"


class SeedDispersionError(ValueError):
    """Raised when a seed dispersion cannot be characterized or read."""


class SeedArm(ResultModel):
    """One training run behind a characterization."""

    run_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    #: The checkpoint label its readings were recorded under, which is what ties
    #: this arm to the measurements the spread was taken over.
    checkpoint: str = Field(min_length=1)
    #: What the run took end to end, as the run itself measured it. Not the
    #: store's ``training.training_seconds``, which times the optimizer loop
    #: alone; this is what re-characterizing would have to be scheduled around.
    wall_clock_seconds: float = Field(gt=0.0)


class HealthBand(ResultModel):
    """What one training-health reading looked like across the vehicle's arms.

    Stored as a center and a spread rather than as the range the arms happened
    to span, because a range over a handful of arms is as wide as its widest
    arm and no wider, and the check built on it would fire on a treatment that
    is merely at the edge of normal.
    """

    center: float
    dispersion: MetricDispersion

    @property
    def covered(self) -> float:
        """Return how far from ``center`` a reading may sit and still be in scope.

        Built from the measured spread rather than from the conservative bound
        every floor here rests on, which is the one place in this package that
        is right. A bound widens, and widening a floor errs safe while widening
        a scope check errs the other way: at three arms the bound is over four
        times the spread, so an arm well outside what the base showed would be
        passed as sharing its dispersion and quoted a floor measured on stable
        ones. Erring toward withholding costs a reader a floor they might have
        been entitled to; erring the other way is the failure this check exists
        for.
        """

        return DEFAULT_COVERAGE * self.dispersion.value

    def covers(self, value: float) -> bool:
        """Return whether one arm's reading sits inside the vehicle's spread."""

        return abs(value - self.center) <= self.covered


class SeedDispersion(ResultModel):
    """The spread one frozen training configuration's arms showed."""

    training_sha256: Sha256Hex
    #: The horizon the arms were trained to. Outside the digest by
    #: construction, so it is stated here and checked against the reading being
    #: qualified rather than assumed to match.
    horizon_steps: int = Field(gt=0)
    arms: tuple[SeedArm, ...] = Field(min_length=2)
    #: What the arms at distinct seeds moved each metric by, under each declared
    #: workload they reported it for. Keyed by workload as well as by metric
    #: because a benchmark that varies a rating and a temperature writes one
    #: reading per cell, and a spread pooled over the cells would describe none
    #: of them. The empty key is the metric whose value declares no workload.
    metrics: Mapping[str, Mapping[str, MetricDispersion]]
    #: The same spread over arms that shared a seed, which is nondeterminism
    #: rather than seed. Reported beside the total rather than subtracted from
    #: it: what a comparison faces is the total, and this says how much of it
    #: the seed is actually responsible for.
    nondeterminism: Mapping[str, Mapping[str, MetricDispersion]] = Field(
        default_factory=dict
    )
    health: Mapping[str, HealthBand] = Field(default_factory=dict)
    #: What scoring the arms cost, beside what training them cost. Together
    #: they are what re-characterizing a replacement vehicle would price at.
    scoring_seconds: float = Field(ge=0.0)
    measured_at: datetime
    notes: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_arms(self) -> SeedDispersion:
        if len({arm.seed for arm in self.arms}) < 2:
            raise ValueError(
                "a seed dispersion needs arms at two or more distinct seeds; "
                "arms sharing one seed measure nondeterminism instead"
            )
        if len({arm.run_id for arm in self.arms}) != len(self.arms):
            raise ValueError("each arm of a seed dispersion is a distinct run")
        return self

    @property
    def seeds(self) -> tuple[int, ...]:
        """Return the distinct initialization seeds behind the spread."""

        return tuple(sorted({arm.seed for arm in self.arms}))

    @property
    def training_wall_clock_seconds(self) -> float:
        """Return what training every arm cost."""

        return math.fsum(arm.wall_clock_seconds for arm in self.arms)

    @property
    def wall_clock_seconds(self) -> float:
        """Return what the whole characterization cost, training and scoring."""

        return self.training_wall_clock_seconds + self.scoring_seconds

    def floor(self, metric: str, workload: str = UNSCOPED_WORKLOAD) -> float | None:
        """Return the delta this configuration's seed spread alone can produce.

        The two arms of a comparison are replicate draws of one training
        configuration as far as this floor is concerned, so their spreads
        combine the way any two do. The treatment's spread is assumed equal to
        the control's, which is the assumption :attr:`health` exists to check.
        """

        dispersion = self.metrics.get(metric, {}).get(workload)
        return None if dispersion is None else self_combined_floor(dispersion)

    def departures(self, health: Mapping[str, float]) -> tuple[str, ...]:
        """Return the health readings sitting outside the vehicle's own spread.

        Readings this record has no band for are not departures. A band is
        evidence about a quantity the vehicle's arms reported, and silence about
        one they did not is not evidence that an arm is unstable.
        """

        return tuple(
            sorted(
                metric
                for metric, value in health.items()
                if metric in self.health and not self.health[metric].covers(value)
            )
        )


class ArmReading(ResultModel):
    """One arm's identity and everything the characterization reads off it."""

    run_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    checkpoint: str = Field(min_length=1)
    wall_clock_seconds: float = Field(gt=0.0)
    #: One value per metric per declared workload, keyed the way a report groups
    #: its rows. The empty workload key is the reading that declares none.
    metrics: Mapping[str, Mapping[str, float]]
    #: The series each of those values was measured on, keyed identically. Two
    #: arms can report one metric under one workload from different views — an
    #: in-training preview and a canonical pool pass both label their reading by
    #: the checkpoint — and a spread over those describes neither.
    fingerprints: Mapping[str, Mapping[str, str]] = Field(default_factory=dict)
    health: Mapping[str, float] = Field(default_factory=dict)


def characterize(
    readings: Sequence[ArmReading],
    *,
    training_sha256: str,
    horizon_steps: int,
    scoring_seconds: float,
    measured_at: datetime,
    notes: str | None = None,
) -> SeedDispersion:
    """Return the dispersion a set of arms of one configuration showed.

    One arm per distinct seed carries the spread. Where a seed was run more than
    once, the first of its arms is the one that counts and the rest measure
    nondeterminism, so a replicate pair cannot quietly halve the spread by
    contributing a near-duplicate value to it.
    """

    if len(readings) < 2:
        raise SeedDispersionError(
            "characterizing a seed dispersion needs at least two arms"
        )
    by_seed: dict[int, list[ArmReading]] = {}
    for reading in readings:
        by_seed.setdefault(reading.seed, []).append(reading)
    representatives = [arms[0] for _, arms in sorted(by_seed.items())]
    if len(representatives) < 2:
        raise SeedDispersionError(
            "characterizing a seed dispersion needs arms at two or more "
            "distinct seeds; every arm given here shares one seed"
        )

    source = f"{len(representatives)} seed arms of one training configuration"
    metrics = _spread(representatives, source=source)
    if not metrics:
        raise SeedDispersionError(
            "no metric was reported by every seed arm, so nothing here has a "
            "spread to characterize"
        )
    replicates = [arms for arms in by_seed.values() if len(arms) > 1]
    if len(replicates) > 1:
        # One record holds one nondeterminism term, and pooling several is not
        # the same arithmetic as taking one: the groups have their own degrees
        # of freedom and their own source. Refusing says so; merging them would
        # silently keep whichever group was read last.
        raise SeedDispersionError(
            "arms are repeated at more than one seed, and this record holds "
            "one nondeterminism term: repeat one seed, or characterize the "
            "groups separately"
        )
    nondeterminism = (
        _spread(
            replicates[0],
            source=f"{len(replicates[0])} arms at seed {replicates[0][0].seed}",
        )
        if replicates
        else {}
    )
    try:
        return SeedDispersion(
            training_sha256=training_sha256,
            horizon_steps=horizon_steps,
            arms=tuple(
                SeedArm(
                    run_id=reading.run_id,
                    seed=reading.seed,
                    checkpoint=reading.checkpoint,
                    wall_clock_seconds=reading.wall_clock_seconds,
                )
                for reading in readings
            ),
            metrics=metrics,
            nondeterminism=nondeterminism,
            health=_health(representatives, source=source),
            scoring_seconds=scoring_seconds,
            measured_at=measured_at,
            notes=notes,
        )
    except ValidationError as error:
        raise SeedDispersionError(
            f"these arms do not describe one characterization: {error}"
        ) from error


def _spread(
    readings: Sequence[ArmReading],
    *,
    source: str,
) -> dict[str, dict[str, MetricDispersion]]:
    """Return the spread over arms, per metric and declared workload.

    Only cells every arm reported. One an arm missed has fewer replicates behind
    it than the others, and a record mixing the two would hand a comparison a
    floor whose degrees of freedom depend on which row it asks about.
    """

    shared = _shared_cells(readings)
    spreads: dict[str, dict[str, MetricDispersion]] = {}
    for metric, workload in sorted(shared):
        if not _one_series(readings, metric, workload):
            continue
        values = [reading.metrics[metric][workload] for reading in readings]
        dispersion = _estimate(values, source=source)
        if dispersion is not None:
            spreads.setdefault(metric, {})[workload] = dispersion
    return spreads


def _estimate(
    values: Sequence[float],
    *,
    source: str,
) -> MetricDispersion | None:
    """Return the spread over replicate readings, or nothing where it is zero.

    Replicates that agreed to the last digit observed that they could not move
    the quantity rather than that nothing could, and the floor a zero spread
    implies clears every delta. Withheld for the same reason the bound refuses
    one.
    """

    try:
        return measured_dispersion(
            replicate_dispersion(values),
            degrees_of_freedom=len(values) - 1,
            source=source,
            estimator=SEED_ARM_METHOD,
        )
    except NoiseCharacterizationError:
        return None


def _one_series(
    readings: Sequence[ArmReading],
    metric: str,
    workload: str,
) -> bool:
    """Return whether every arm measured this cell on the same series.

    A cell no arm recorded a series for passes, since there is nothing saying
    they differ and a store predating the field is the ordinary case. A cell
    where they disagree is refused, the way a delta across two fingerprints is
    refused everywhere else here.
    """

    recorded = {
        reading.fingerprints.get(metric, {}).get(workload) for reading in readings
    } - {None}
    return len(recorded) <= 1


def _shared_cells(readings: Sequence[ArmReading]) -> set[tuple[str, str]]:
    """Return the metric and workload pairs every arm reported."""

    def cells(reading: ArmReading) -> set[tuple[str, str]]:
        return {
            (metric, workload)
            for metric, by_workload in reading.metrics.items()
            for workload in by_workload
        }

    shared = cells(readings[0])
    for reading in readings[1:]:
        shared &= cells(reading)
    return shared


def _health(
    readings: Sequence[ArmReading],
    *,
    source: str,
) -> dict[str, HealthBand]:
    """Return what the arms' training health looked like, metric by metric."""

    shared = set(readings[0].health)
    for reading in readings[1:]:
        shared &= set(reading.health)
    bands: dict[str, HealthBand] = {}
    for metric in sorted(shared):
        values = [reading.health[metric] for reading in readings]
        dispersion = _estimate(values, source=source)
        if dispersion is not None:
            bands[metric] = HealthBand(
                center=statistics.fmean(values),
                dispersion=dispersion,
            )
    return bands


def seed_dispersion_for(
    training_sha256: str,
    *,
    directory: Path | None = None,
) -> SeedDispersion | None:
    """Return the dispersion recorded for exactly this training identity.

    Exact or absent, with nothing in between. A configuration whose digest is
    not checked in has no floor here, and reporting that is the point: the
    nearest recorded configuration describes a different set of weights however
    close its file happens to sit.
    """

    root = DATA_DIRECTORY if directory is None else directory
    path = root / f"{training_sha256}.json"
    if not path.is_file():
        return None
    return read_seed_dispersion(path)


def read_seed_dispersion(path: Path) -> SeedDispersion:
    """Return the characterization stored at one path."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SeedDispersionError(
            f"seed dispersion at {path} cannot be read: {error}"
        ) from error
    try:
        record = SeedDispersion.model_validate(payload)
    except ValidationError as error:
        raise SeedDispersionError(
            f"seed dispersion at {path} is not a valid characterization: {error}"
        ) from error
    if path.stem != record.training_sha256:
        raise SeedDispersionError(
            f"seed dispersion at {path} records identity "
            f"{record.training_sha256}, so a lookup by that digest would never "
            "reach this file"
        )
    return record


def write_seed_dispersion(
    dispersion: SeedDispersion,
    *,
    directory: Path | None = None,
) -> Path:
    """Write a characterization where a lookup by its own digest will find it."""

    root = DATA_DIRECTORY if directory is None else directory
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{dispersion.training_sha256}.json"
    path.write_bytes(canonical_readable_json(dispersion.as_record()))
    return path


__all__ = [
    "DATA_DIRECTORY",
    "SEED_ARM_METHOD",
    "ArmReading",
    "HealthBand",
    "SeedArm",
    "SeedDispersion",
    "SeedDispersionError",
    "characterize",
    "read_seed_dispersion",
    "seed_dispersion_for",
    "write_seed_dispersion",
]
