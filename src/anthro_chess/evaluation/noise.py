"""Data-sampling noise, estimated by bootstrapping one scored run.

This is the cheap noise floor. It asks how far a metric would move on a
different draw of the same size from the same population, and it is answerable
from a single evaluation pass: resample the games that were scored, recompute
the metric, and read the spread. No repeat run and no second checkpoint.

Games are the resampling unit rather than positions. Positions within one game
are strongly dependent, so resampling positions would treat a game's worth of
correlated decisions as independent evidence and report a floor several times
too narrow.

The estimate scales as the inverse square root of the games behind it, which is
what turns "how many games does this axis need" into a computation rather than
a guess. ``anthro_chess.evaluation.results.noise`` owns that arithmetic and the
stored record; this module owns only the estimation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from pydantic import Field, StrictBool

from anthro_chess.config import ConfigModel
from anthro_chess.evaluation.results import DataComponent, series_fingerprint
from anthro_chess.evaluation.results.noise import (
    BOOTSTRAP_METHOD,
    DEFAULT_COVERAGE,
    FloorEntry,
    NoiseCharacterization,
    NoiseCharacterizationError,
    build_characterization,
    floor_from_dispersion,
)

#: Enough resamples for a stable dispersion without making a cheap estimate
#: expensive. The floor is reported to a few significant figures, so further
#: resamples buy precision nothing downstream reads.
DEFAULT_RESAMPLES = 1000


class NoiseConfig(ConfigModel):
    """How an evaluation characterizes its own data-sampling noise."""

    enabled: StrictBool = True
    resamples: int = Field(default=DEFAULT_RESAMPLES, ge=100)
    seed: int = Field(default=0, ge=0)
    coverage: float = Field(default=DEFAULT_COVERAGE, gt=0.0)


@dataclass(frozen=True)
class MetricTotal:
    """One game's contribution to one metric's position-weighted mean."""

    total: float
    positions: int


@dataclass(frozen=True)
class GameTotals:
    """Everything one scored game contributes to every reported metric.

    Sums and counts are carried per metric rather than sharing one count,
    because a sliced metric only counts the positions that fell in its slice.
    That is what lets phase, rating-band, and rule-case series be bootstrapped
    on the same pass as the pool-wide ones.
    """

    game_id: int
    metrics: Mapping[str, MetricTotal]


def bootstrap_floors(
    totals: Sequence[GameTotals],
    *,
    component: DataComponent,
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
    coverage: float = DEFAULT_COVERAGE,
) -> tuple[FloorEntry, ...]:
    """Return one data-sampling floor per metric the scored games support.

    A metric whose dispersion cannot be estimated is omitted rather than
    reported as zero. That happens for a rule case rare enough that resamples
    frequently contain none of it, and a floor of zero there would license
    every delta as a finding.
    """

    if len(totals) < 2:
        raise NoiseCharacterizationError(
            "bootstrapping data-sampling noise needs at least two scored games"
        )
    if resamples < 2:
        raise NoiseCharacterizationError("a bootstrap needs at least two resamples")

    metrics = sorted({metric for game in totals for metric in game.metrics})
    if not metrics:
        raise NoiseCharacterizationError(
            "the scored games report no metric to bootstrap"
        )

    sums = np.zeros((len(totals), len(metrics)), dtype=np.float64)
    counts = np.zeros((len(totals), len(metrics)), dtype=np.float64)
    for row, game in enumerate(totals):
        for column, metric in enumerate(metrics):
            contribution = game.metrics.get(metric)
            if contribution is None:
                continue
            sums[row, column] = contribution.total
            counts[row, column] = contribution.positions

    replicates = _bootstrap_replicates(sums, counts, seed=seed, resamples=resamples)
    entries: list[FloorEntry] = []
    for column, metric in enumerate(metrics):
        values = replicates[:, column]
        observed = values[np.isfinite(values)]
        if observed.size < 2:
            continue
        dispersion = float(np.std(observed, ddof=1))
        entries.append(
            FloorEntry(
                metric=metric,
                fingerprint=series_fingerprint(metric, component),
                floor=floor_from_dispersion(dispersion, coverage=coverage),
                dispersion=dispersion,
                sampling_units=len(totals),
            )
        )
    return tuple(entries)


def characterize_sampling_noise(
    totals: Sequence[GameTotals],
    *,
    component: DataComponent,
    config: NoiseConfig,
    source: str,
    recorded_at: datetime | None = None,
) -> NoiseCharacterization | None:
    """Return the recordable characterization for one evaluation's own noise.

    ``None`` means nothing could be estimated, which is a reportable state
    rather than an error: a view too small to resample says so instead of
    producing a floor nobody should trust.
    """

    entries = bootstrap_floors(
        totals,
        component=component,
        seed=config.seed,
        resamples=config.resamples,
        coverage=config.coverage,
    )
    if not entries:
        return None
    return build_characterization(
        kind="data-sampling",
        method=BOOTSTRAP_METHOD,
        replicates=config.resamples,
        coverage=config.coverage,
        source=source,
        floors=entries,
        recorded_at=recorded_at,
    )


def _bootstrap_replicates(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> np.ndarray:
    """Recompute every metric over ``resamples`` resamples of the games.

    Each resample is applied as a vector of game multiplicities rather than as
    an index gather, so one resample costs a matrix-vector product instead of
    copying the whole table.
    """

    games = sums.shape[0]
    generator = np.random.default_rng(seed)
    replicates = np.empty((resamples, sums.shape[1]), dtype=np.float64)
    for index in range(resamples):
        drawn = generator.integers(0, games, games)
        weights = np.bincount(drawn, minlength=games).astype(np.float64)
        resampled_counts = weights @ counts
        replicates[index] = np.divide(
            weights @ sums,
            resampled_counts,
            out=np.full(sums.shape[1], np.nan),
            where=resampled_counts > 0.0,
        )
    return replicates


__all__ = [
    "DEFAULT_RESAMPLES",
    "GameTotals",
    "MetricTotal",
    "NoiseConfig",
    "bootstrap_floors",
    "characterize_sampling_noise",
]
