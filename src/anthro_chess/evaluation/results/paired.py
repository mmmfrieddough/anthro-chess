"""Comparison-scoped sampling floors over matching frozen inputs.

Deterministic benchmarks compare checkpoints on the same frozen units. Their
population uncertainty is therefore uncertainty in the *paired delta*, not the
difference between two independently sampled benchmark sets. Detail payloads
retain the aligned unit contributions; this module validates them and
bootstraps checkpoint differences without moving bulk vectors into the
committed summary tier.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import Field, model_validator

from anthro_chess.evaluation.results.noise import (
    DEFAULT_CONFIDENCE,
    PAIRED_BOOTSTRAP_METHOD,
    dispersion_bound,
)
from anthro_chess.evaluation.results.records import (
    Identifier,
    NoiseFloor,
    ResultEnvelope,
    ResultModel,
)
from anthro_chess.evaluation.results.store import (
    DetailStore,
    ResultsStoreError,
)

#: Version 3 added per-unit weights, so a metric whose reported value is a mean
#: over something *inside* the resampling unit — a per-position mean carried by
#: games — can be retained without redefining it as a mean over units.
#: Version 2 added the confidence the floor's dispersion bound carries.
PAIRED_CONTRIBUTIONS_VERSION = 3
PAIRED_CONTRIBUTIONS_KEY = "paired_contributions"


class PairedContributions(ResultModel):
    """Aligned per-unit metric values retained for later checkpoint deltas.

    ``weights`` is how a unit-level retention stays faithful to a metric the
    benchmark reports as a mean over smaller things. A per-position mean is a
    ratio of sums, so a game contributes its own mean weighted by how many
    positions it holds; without the weight, the retained values would only
    reproduce the metric on units of identical size. Absent weights mean every
    unit counts once, which is the unweighted case earlier versions carried.
    """

    version: int = Field(default=PAIRED_CONTRIBUTIONS_VERSION, ge=1)
    unit: Identifier
    unit_ids: tuple[str, ...] = Field(min_length=2)
    stratum: Identifier | None = None
    strata: tuple[str, ...] | None = None
    metrics: dict[str, tuple[float, ...]] = Field(min_length=1)
    weights: tuple[float, ...] | None = None
    resamples: int = Field(ge=100)
    seed: int = Field(ge=0)
    coverage: float = Field(gt=0.0)
    confidence: float = Field(default=DEFAULT_CONFIDENCE, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate_alignment(self) -> PairedContributions:
        if self.version > PAIRED_CONTRIBUTIONS_VERSION:
            raise ValueError(
                f"paired contributions use version {self.version}; this build "
                f"understands {PAIRED_CONTRIBUTIONS_VERSION}"
            )
        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError("paired contribution unit ids must be unique")
        if (self.stratum is None) != (self.strata is None):
            raise ValueError(
                "paired contributions must provide both a stratum name and values"
            )
        if self.strata is not None and len(self.strata) != len(self.unit_ids):
            raise ValueError(
                f"paired contributions have {len(self.strata)} strata for "
                f"{len(self.unit_ids)} units"
            )
        if not math.isfinite(self.coverage):
            raise ValueError("paired contribution coverage must be finite")
        if self.weights is not None:
            if len(self.weights) != len(self.unit_ids):
                raise ValueError(
                    f"paired contributions have {len(self.weights)} weights for "
                    f"{len(self.unit_ids)} units"
                )
            if any(
                not math.isfinite(weight) or weight <= 0.0 for weight in self.weights
            ):
                raise ValueError(
                    "paired contribution weights must be finite and positive"
                )
        for metric, values in self.metrics.items():
            if not metric:
                raise ValueError("paired contribution metric names cannot be empty")
            if len(values) != len(self.unit_ids):
                raise ValueError(
                    f"paired contribution {metric} has {len(values)} values for "
                    f"{len(self.unit_ids)} units"
                )
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"paired contribution {metric} values must be finite")
        return self

    def as_record(self) -> dict[str, Any]:
        """Return the JSON-compatible detail-tier record."""

        return self.model_dump(mode="json")


def paired_contributions(
    *,
    unit: str,
    unit_ids: Sequence[str],
    stratum: str | None = None,
    strata: Sequence[str] | None = None,
    metrics: Mapping[str, Sequence[float]],
    weights: Sequence[float] | None = None,
    resamples: int,
    seed: int,
    coverage: float,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedContributions:
    """Build validated aligned values for one deterministic benchmark result."""

    return PairedContributions(
        unit=unit,
        unit_ids=tuple(unit_ids),
        stratum=stratum,
        strata=None if strata is None else tuple(strata),
        metrics={
            metric: tuple(float(value) for value in values)
            for metric, values in sorted(metrics.items())
        },
        weights=None if weights is None else tuple(float(value) for value in weights),
        resamples=resamples,
        seed=seed,
        coverage=coverage,
        confidence=confidence,
    )


@dataclass(frozen=True)
class PairedFloor:
    """One comparison's paired floor for one metric, or why it has none.

    Both fields are reported rather than one. A caller holding ``None`` where a
    floor was expected cannot tell a benchmark that never pairs from a pair
    whose inputs did not survive the machine boundary, and those are the two
    situations a reader most needs told apart: only the second is a reading
    qualified by the wrong estimator.
    """

    floor: NoiseFloor | None = None
    #: Why no paired floor could be built, in a phrase a report can print.
    #: ``None`` only when ``floor`` is set.
    unavailable: str | None = None


#: The fields two retained payloads have to agree on before their per-unit
#: values describe one comparison, named so a report can say which one differs.
_COMPARABLE_FIELDS = ("unit", "stratum", "resamples", "seed", "coverage", "confidence")


class PairedFloorIndex:
    """Resolve paired data-sampling floors from machine-local result details."""

    def __init__(self, detail: DetailStore) -> None:
        self._detail = detail
        self._contributions: dict[str, tuple[PairedContributions | None, str]] = {}
        self._floors: dict[tuple[str, str], tuple[dict[str, NoiseFloor], str]] = {}

    def floor(
        self,
        baseline: ResultEnvelope,
        current: ResultEnvelope,
        metric: str,
    ) -> PairedFloor:
        """Return the paired floor for one result comparison, or why there is none."""

        key = (baseline.result_id, current.result_id)
        if key not in self._floors:
            self._floors[key] = self._comparison_floors(baseline, current)
        floors, unavailable = self._floors[key]
        retained = floors.get(metric)
        if retained is not None:
            return PairedFloor(floor=retained)
        if unavailable:
            return PairedFloor(unavailable=unavailable)
        return PairedFloor(unavailable=self._metric_absence(baseline, current, metric))

    def _metric_absence(
        self,
        baseline: ResultEnvelope,
        current: ResultEnvelope,
        metric: str,
    ) -> str:
        """Name the side that retained nothing, for a pair that otherwise matched.

        Which side it is decides what a maintainer does about it, and the two
        sides disagreeing about what they retain is what a change in the
        retained set looks like from a store holding readings from either side
        of it.
        """

        left, _ = self._load(baseline)
        right, _ = self._load(current)
        assert left is not None and right is not None  # both loaded to get here
        missing = [
            side
            for side, contributions in (("baseline", left), ("current", right))
            if metric not in contributions.metrics
        ]
        if len(missing) == 1:
            return f"the {missing[0]} reading retained no contribution for {metric}"
        return f"neither reading retained a contribution for {metric}"

    def _comparison_floors(
        self,
        baseline: ResultEnvelope,
        current: ResultEnvelope,
    ) -> tuple[dict[str, NoiseFloor], str]:
        left, left_absence = self._load(baseline)
        right, right_absence = self._load(current)
        if left is None and right is None and left_absence == right_absence:
            # The usual shape of the failure, and the one worth reading as one
            # sentence: neither side of a comparison taken on another machine
            # has anything here to difference.
            return {}, f"both readings {left_absence}"
        if left is None or right is None:
            missing = [
                f"the {side} reading {absence}"
                for side, contributions, absence in (
                    ("baseline", left, left_absence),
                    ("current", right, right_absence),
                )
                if contributions is None
            ]
            return {}, " and ".join(missing)
        for field in _COMPARABLE_FIELDS:
            if getattr(left, field) != getattr(right, field):
                return {}, f"the retained contributions disagree on {field}"
        if set(left.unit_ids) != set(right.unit_ids):
            return {}, "the retained contributions cover different units"

        right_index = {unit_id: index for index, unit_id in enumerate(right.unit_ids)}
        right_order = np.asarray(
            [right_index[unit_id] for unit_id in left.unit_ids],
            dtype=np.int64,
        )
        left_strata = left.strata
        right_strata = (
            None
            if right.strata is None
            else tuple(right.strata[index] for index in right_order)
        )
        if left_strata != right_strata:
            return {}, "the retained contributions assign units to different strata"
        left_weights = left.weights
        right_weights = (
            None
            if right.weights is None
            else tuple(right.weights[index] for index in right_order)
        )
        # A weight describes the frozen view rather than the checkpoint, so two
        # comparable readings carry the same one. Where they do not, the two
        # sides weight their units differently and their difference is not the
        # delta either measurement reports.
        if left_weights != right_weights:
            return {}, "the retained contributions weight their units differently"
        common = sorted(set(left.metrics) & set(right.metrics))
        if not common:
            return {}, "the two readings retain no metric in common"
        baseline_values = np.column_stack(
            [np.asarray(left.metrics[metric], dtype=np.float64) for metric in common]
        )
        current_values = np.column_stack(
            [
                np.asarray(right.metrics[metric], dtype=np.float64)[right_order]
                for metric in common
            ]
        )
        unit_weights = (
            None if left_weights is None else np.asarray(left_weights, dtype=np.float64)
        )
        self._verify_means(baseline, common, baseline_values, unit_weights)
        self._verify_means(current, common, current_values, unit_weights)
        deltas = current_values - baseline_values
        replicates = _bootstrap_paired_means(
            deltas,
            strata=left_strata,
            weights=unit_weights,
            seed=left.seed,
            resamples=left.resamples,
        )
        dispersions = np.std(replicates, axis=0, ddof=1)
        # The matched units are the independent replicates, not the resamples
        # drawn from them. There is no sqrt(2) here either: this bootstrap
        # resamples the delta itself, so the spread it reports is already the
        # spread of a difference rather than of one side of one.
        freedom = len(left.unit_ids) - 1
        return {
            metric: NoiseFloor(
                value=float(
                    left.coverage
                    * dispersion_bound(
                        float(dispersion),
                        degrees_of_freedom=freedom,
                        confidence=left.confidence,
                    )
                ),
                kind="data-sampling",
                source=(
                    f"{left.resamples} "
                    f"{'stratified ' if left.stratum is not None else ''}"
                    f"{'weighted ' if left_weights is not None else ''}"
                    "paired bootstrap resamples of "
                    f"{len(left.unit_ids)} matching {left.unit} units"
                ),
                estimator=PAIRED_BOOTSTRAP_METHOD,
            )
            for metric, dispersion in zip(common, dispersions, strict=True)
        }, ""

    def _load(
        self,
        envelope: ResultEnvelope,
    ) -> tuple[PairedContributions | None, str]:
        """Return one reading's retained contributions, or why it has none.

        A payload this machine does not hold is an absence rather than a store
        fault. The detail tier is machine-local while the summary tier is
        committed, so a reading recorded elsewhere always resolves to nothing
        here, and raising would end a whole report over one row's floor. A
        payload that is present and wrong still raises: that is the store
        disagreeing with itself rather than a file that never arrived.
        """

        cached = self._contributions.get(envelope.result_id)
        if cached is not None:
            return cached
        loaded = self._read(envelope)
        self._contributions[envelope.result_id] = loaded
        return loaded

    def _read(
        self,
        envelope: ResultEnvelope,
    ) -> tuple[PairedContributions | None, str]:
        if envelope.detail is None:
            return None, "recorded no detail payload"
        if not self._detail.holds(envelope.detail):
            return None, "recorded a detail payload this machine does not hold"
        payload = self._detail.read(envelope.detail)
        if not isinstance(payload, Mapping):
            raise ResultsStoreError(
                f"detail payload for {envelope.result_id} is not an object"
            )
        raw = payload.get(PAIRED_CONTRIBUTIONS_KEY)
        if raw is None:
            return None, "retained no per-unit contributions"
        try:
            contributions = PairedContributions.model_validate(raw)
        except ValueError as error:
            raise ResultsStoreError(
                f"invalid paired contributions for {envelope.result_id}: {error}"
            ) from error
        return contributions, ""

    @staticmethod
    def _verify_means(
        envelope: ResultEnvelope,
        metrics: Sequence[str],
        values: np.ndarray,
        weights: np.ndarray | None,
    ) -> None:
        for column, metric in enumerate(metrics):
            measurement = envelope.measurement(metric)
            if measurement is None:
                continue
            retained = float(np.average(values[:, column], weights=weights))
            if not math.isclose(
                retained,
                measurement.value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ResultsStoreError(
                    f"paired contributions for {envelope.result_id} reproduce "
                    f"{metric} as {retained}, not {measurement.value}"
                )


def _bootstrap_paired_means(
    deltas: np.ndarray,
    *,
    strata: Sequence[str] | None = None,
    weights: np.ndarray | None = None,
    seed: int,
    resamples: int,
) -> np.ndarray:
    """Resample aligned units and return the mean delta for every metric.

    With ``weights`` the replicate is a ratio rather than a plain mean, and the
    denominator is recomputed per draw. That is the point: a resample that
    happens to draw the heavy units is a resample carrying more of the thing
    the metric averages over, and holding the denominator fixed would report
    that as movement in the metric instead.
    """

    units = int(deltas.shape[0])
    generator = np.random.default_rng(seed)
    replicates = np.empty((resamples, deltas.shape[1]), dtype=np.float64)
    buckets = _stratum_buckets(strata) if strata is not None else None
    for index in range(resamples):
        drawn: np.ndarray
        if buckets is None:
            drawn = generator.integers(0, units, size=units)
        else:
            sampled: list[np.ndarray] = []
            for size, grouped_indices in buckets:
                offsets = generator.integers(0, size, grouped_indices.shape)
                sampled.append(
                    np.take_along_axis(grouped_indices, offsets, axis=1).ravel()
                )
            drawn = np.concatenate(sampled)
        multiplicity = np.bincount(drawn, minlength=units).astype(np.float64)
        if weights is None:
            replicates[index] = multiplicity @ deltas / units
            continue
        drawn_weights = multiplicity * weights
        replicates[index] = drawn_weights @ deltas / float(drawn_weights.sum())
    return replicates


def _stratum_buckets(
    strata: Sequence[str],
) -> tuple[tuple[int, np.ndarray], ...]:
    """Group equal-size strata so one bootstrap draw remains vectorized."""

    grouped: dict[str, list[int]] = {}
    for index, stratum in enumerate(strata):
        grouped.setdefault(stratum, []).append(index)
    by_size: dict[int, list[list[int]]] = {}
    for indices in grouped.values():
        by_size.setdefault(len(indices), []).append(indices)
    return tuple(
        (size, np.asarray(groups, dtype=np.int64))
        for size, groups in sorted(by_size.items())
    )


__all__ = [
    "PAIRED_CONTRIBUTIONS_KEY",
    "PAIRED_CONTRIBUTIONS_VERSION",
    "PairedContributions",
    "PairedFloor",
    "PairedFloorIndex",
    "paired_contributions",
]
