"""Data-sampling noise, estimated by bootstrapping one scored run.

This is the cheap dispersion. It asks how far a metric would move on a
different draw of the same size from the same population, and it is answerable
from a single evaluation pass: resample the games that were scored, recompute
the metric, and read the spread. No repeat run and no second checkpoint.

The spread travels on the reading that measured it. Turning it into a floor
takes two readings, and that happens at comparison time.

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

import numpy as np
from pydantic import Field, StrictBool

from anthro_chess.config import ConfigModel
from anthro_chess.evaluation.results import (
    DataComponent,
    MetricDispersion,
    WorkloadComponent,
    series_fingerprint,
)
from anthro_chess.evaluation.results.noise import (
    BOOTSTRAP_METHOD,
    DEFAULT_CONFIDENCE,
    DEFAULT_COVERAGE,
    NoiseCharacterizationError,
    measured_dispersion,
)

#: Enough resamples for a stable dispersion without making a cheap estimate
#: expensive. The dispersion is reported to a few significant figures, so
#: further resamples buy precision nothing downstream reads.
DEFAULT_RESAMPLES = 1000


class NoiseConfig(ConfigModel):
    """How an evaluation estimates its own data-sampling noise."""

    enabled: StrictBool = True
    resamples: int = Field(default=DEFAULT_RESAMPLES, ge=100)
    seed: int = Field(default=0, ge=0)
    #: Read only by the estimators that report a covered spread within one
    #: reading. A benchmark that stores a dispersion and nothing else does not
    #: read it: a delta floor takes its coverage at comparison time, because the
    #: claim belongs to the comparison rather than to either reading.
    coverage: float = Field(default=DEFAULT_COVERAGE, gt=0.0)
    confidence: float = Field(default=DEFAULT_CONFIDENCE, gt=0.0, lt=1.0)


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


def bootstrap_dispersions(
    totals: Sequence[GameTotals],
    *,
    component: DataComponent,
    seed: int,
    source: str,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    workload: WorkloadComponent | None = None,
) -> dict[str, MetricDispersion]:
    """Return one data-sampling dispersion per metric the scored games support.

    Keyed by series fingerprint, which names the metric as well as the inputs,
    so a reading can attach each dispersion to the measurement it describes
    without matching on the identifier and hoping the series agreed.

    A metric whose dispersion cannot be estimated is omitted rather than
    reported as zero. That happens for a rule case rare enough that resamples
    frequently contain none of it, and a floor of zero there would license
    every delta as a finding.

    A dispersion the resample measured as exactly zero is omitted on the same
    ground. The draw observed that it could not move this metric, which is not
    the same as observing that nothing could: a quantity identical in every game
    scored reads this way at any sample size, and the wider sample that would
    move it is the thing nobody has taken.

    So is a metric only one game realized, and by its game count rather than by
    that test. One game is one replicate and observes no spread for a bound to
    rest on, while resampling it does not quite return a constant — floating
    point leaves the last bits of the divided total moving.

    The **games** are what the dispersion bound's degrees of freedom count, not
    the resamples. A bootstrap draws as many resamples as it is asked for, but
    every one of them is drawn from the same games, so more of them buy a
    steadier reading of a spread that is already pinned down by the sample in
    hand. Counting them as independent replicates would claim near-certainty
    about the dispersion from a number the caller chose for free, which is
    precisely the false precision the bound is here to remove.

    Which games is decided per metric, from the ones that scored a position for
    it. A sliced metric is realized in a fraction of the pass — a rule case, an
    opening tier, a rating band — and the games that never met it are evidence
    about nothing here. Counting the whole pass would size the bound for
    replicates the metric never had and hand the same inflated count to
    ``games_to_resolve``, both in the direction that overstates what a rare
    slice resolved.

    ``workload`` is required by a benchmark whose metrics are execution-
    sensitive, because a dispersion has to carry the same fingerprint as the
    measurement it qualifies. Omitting it there would key every dispersion
    against a series no measurement belongs to.
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
    realized = np.count_nonzero(counts, axis=0)
    dispersions: dict[str, MetricDispersion] = {}
    for column, metric in enumerate(metrics):
        games = int(realized[column])
        if games < 2:
            continue
        values = replicates[:, column]
        observed = values[np.isfinite(values)]
        if observed.size < 2:
            continue
        dispersion = float(np.std(observed, ddof=1))
        if dispersion == 0.0:
            continue
        dispersions[series_fingerprint(metric, component, workload)] = (
            measured_dispersion(
                dispersion,
                degrees_of_freedom=games - 1,
                confidence=confidence,
                units=games,
                source=source,
                estimator=BOOTSTRAP_METHOD,
            )
        )
    return dispersions


def sampling_dispersions(
    totals: Sequence[GameTotals],
    *,
    component: DataComponent,
    config: NoiseConfig,
    source: str,
    workload: WorkloadComponent | None = None,
) -> dict[str, MetricDispersion]:
    """Return what one evaluation's own draw of games moves each metric by.

    An empty mapping means nothing could be estimated, which is a reportable
    state rather than an error: a view too small to resample says so instead of
    producing a dispersion nobody should trust.
    """

    return bootstrap_dispersions(
        totals,
        component=component,
        seed=config.seed,
        source=source,
        resamples=config.resamples,
        confidence=config.confidence,
        workload=workload,
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
    "bootstrap_dispersions",
    "sampling_dispersions",
]
