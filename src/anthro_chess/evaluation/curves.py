"""The comparison shape every human-reference benchmark shares.

Opening repertoire, book depth, game length, result and termination patterns,
and repetition ask the same question in the same way: measure a quantity on
generated games, measure the same quantity on human games from the pool, and
compare the two as a function of rating. Building that once is what keeps five
benchmarks from drifting apart on the parts that are easy to get subtly wrong.

Rating is continuous here rather than banded. Each game contributes one
observation at its own rating, so a single quantity as a function of rating
needs no bins at all. Binning is forced only when a whole categorical
distribution has to be estimated at a point, and there the neighbourhood the
smoother already uses *is* the bin.

Two properties keep the comparison honest, and both are about the bandwidth.
The human reference's density over the rating range is uneven while the model's
is uniform by construction, so smoothing each side at its own bandwidth would
put an artifact into their difference near every peak: the human side forces
the local bandwidth and the model side is estimated at the same one. And
because smoothing bias scales with bandwidth, re-selecting it per run would
mean two checkpoints were measured differently and their series would not be
comparable. It is selected once from the human reference by cross-validation
and then declared in a frozen ``CurveSpec`` that a benchmark version owns.
Changing it is a version bump rather than a configuration change, which is why
a spec lives in code rather than in a config file.

One pass produces both readings a benchmark needs. The pooled reading asks
whether the model behaves like humans at all; the curve asks whether it
responds to rating the way humans do. They dissociate in a specific and likely
way, and a model that matches the pooled human distribution with a flat curve
is reported as such rather than as a pass. The pooled reading averages each
side's own curve across the grid rather than pooling raw games, because the two
sides are different rating designs and pooling raw games would report that
difference as a distance.

Every distance comes back with two things to read it against, and they are not
interchangeable. A floor says how far the number moves between two runs when
nothing changed, which qualifies a delta between checkpoints; it resamples the
model side alone, since two checkpoints share one fixed human reference and its
sampling error cancels in their difference. A reference level says what the
distance reads at when there is nothing to find, which is what a single reading
needs on its own; that is an absolute statement rather than a difference, so the
human side's own sampling error belongs in it and it keeps the paired
resampling. Two finite samples of one population never agree exactly, and a
sampled curve is never exactly flat.

The floor and the distance levels both resample the **stream** rather than the
game, because a stream is what a fresh seed redraws. A generated side plays one
game per rating from each of its streams, so a grid multiplies games without
multiplying the draws behind them, and an estimator that treated the games as
independent would answer a question about a sample nobody is going to take.

Curve points are data, not pictures. They go to the machine-local detail tier,
so overlaying several checkpoints against the human reference is a query rather
than another run, and only the scalar distances reach the committed summary
tier.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np

from anthro_chess.evaluation.results import (
    BridgeIndex,
    DataComponent,
    Measurement,
    MetricDispersion,
    WorkloadComponent,
    measurement,
)
from anthro_chess.evaluation.results.noise import (
    DEFAULT_CONFIDENCE,
    DEFAULT_COVERAGE,
    measured_dispersion,
    plug_in_rescale,
    self_combined_floor,
)

CURVE_COMPARISON_VERSION = 3

#: How a curve comparison estimates its own spread. A distributional distance
#: moves by an amount that is a function of its own configuration rather than of
#: a series, so it is estimated here and carried with the measurement instead of
#: being characterized separately and looked up. Decision 0060 measured why the
#: unit is the stream and not the game.
CURVE_BOOTSTRAP_METHOD = "bootstrap-over-generated-streams"

#: How a curve comparison states the spread of a model side that cannot vary.
#: Re-measuring such a side replays the same games, so its evaluation noise is
#: exactly zero rather than estimated, and the record says so instead of
#: bootstrapping games another seed was never going to change.
CURVE_DETERMINISTIC_METHOD = "deterministic-model-side"

#: Resamples behind a comparison's own floor. Lower than a scalar metric's
#: bootstrap because one resample here re-estimates whole curves rather than
#: recomputing a mean, and the floor is read to a couple of significant figures.
DEFAULT_RESAMPLES = 200

#: Streams a bootstrapped spread needs before it is an estimate at all. Two
#: leave three distinct resamples, where decision 0060 measured the estimator
#: ranging over a factor of four and landing on an exact zero whenever the two
#: agreed — and a zero floor clears every delta. ``SeedSpread`` withholds below
#: three replicates for the same reason.
MINIMUM_BOOTSTRAP_STREAMS = 3

#: Neighbour counts a bandwidth selection considers by default. Spread wide
#: because the right span depends on how densely the corpus covers the rating
#: range, which is a property of the corpus rather than of the quantity.
DEFAULT_NEIGHBOUR_CANDIDATES: tuple[int, ...] = (16, 32, 64, 128, 256, 512)

#: Label the human reference carries in an overlay of several checkpoints.
HUMAN_REFERENCE_LABEL = "human-reference"

_NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"

#: A radius is one of the observed distances, so the game that defines it must
#: land inside its own neighbourhood rather than on the wrong side of a float
#: comparison with itself.
_RADIUS_TOLERANCE = 1e-9


class CurveComparisonError(ValueError):
    """Raised when a curve comparison cannot be estimated as specified."""


class CurveQuantity(StrEnum):
    """What one game contributes to a curve.

    A scalar quantity is a number per game, such as game length or the share of
    available theory a game consumed. A categorical quantity is one label per
    game whose whole distribution is the measurement, such as opening family,
    result, or termination category.
    """

    SCALAR = "scalar"
    CATEGORICAL = "categorical"


class RatingResponse(StrEnum):
    """How the model's rating response reads against the human reference."""

    #: Pooled behavior and the curve both agree within sampling noise.
    MATCHES = "matches"
    #: Pooled behavior matches while the curve does not, and the model's curve
    #: is as flat as one with no rating response at all. The model learned the
    #: average human and is ignoring its rating input.
    AVERAGE_HUMAN = "average_human"
    #: Pooled behavior matches and the curve moves, but not the way humans do.
    DIVERGENT_RESPONSE = "divergent_response"
    #: The pooled distribution itself differs, so the curve reading is not the
    #: first thing to explain.
    MISMATCH = "mismatch"
    #: Nothing was resampled, so no distance can be judged against anything.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Observation:
    """One game's contribution: the rating it was played at, and its value.

    Games rather than positions, because positions within one game are far from
    independent and a curve is a statement about games anyway.

    ``stream`` names the random draw the game came out of, and is what a
    resample redraws. A generated side plays its whole rating grid from one set
    of streams, so the games one stream produced move together and are not
    independent of each other; a side whose games each came from their own draw
    leaves this unset and every game is its own unit.
    """

    rating: float
    value: float | str
    stream: int | None = None


@dataclass(frozen=True)
class PairedRateObservation:
    """One opportunity contributing to a point human-reference comparison."""

    game_id: int
    human_success: bool
    model_success: bool
    model_probability_mass: float


@dataclass(frozen=True)
class PointReferenceComparison:
    """A human and model rate compared at one point rather than over a curve."""

    games: int
    opportunities: int
    effective_sample_size: float
    human_rate: float
    model_rate: float
    model_probability_mass: float

    @property
    def human_gap(self) -> float:
        """Return model selected-action rate minus the human rate."""

        return self.model_rate - self.human_rate

    @property
    def selected_rate(self) -> float:
        """Return the model's legal-greedy success rate."""

        return self.model_rate

    @property
    def policy_mass(self) -> float:
        """Return the model's mean raw mass on successful actions."""

        return self.model_probability_mass

    def as_record(self) -> dict[str, object]:
        return {
            "games": self.games,
            "opportunities": self.opportunities,
            "effective_sample_size": self.effective_sample_size,
            "human_rate": self.human_rate,
            "selected_rate": self.model_rate,
            "policy_mass": self.model_probability_mass,
            "human_gap": self.human_gap,
        }


class ReferenceRateSupport:
    """One point human-reference comparison, summed rather than retained.

    One precondition: **all of a game's opportunities arrive in one call**,
        because the effective sample size counts a game's opportunities together
        and a game split across two calls would count as two.
    """

    def __init__(self) -> None:
        self._opportunities = 0
        self._human = 0.0
        self._model = 0.0
        self._mass = 0.0
        self._games = 0
        self._squared = 0

    def add(self, observations: Sequence[PairedRateObservation]) -> None:
        """Add a batch of opportunities, with each game's arriving together."""

        counts: dict[int, int] = {}
        for observation in observations:
            if not 0.0 <= observation.model_probability_mass <= 1.0:
                raise CurveComparisonError(
                    "model probability mass must be between zero and one"
                )
            counts[observation.game_id] = counts.get(observation.game_id, 0) + 1
        self._opportunities += len(observations)
        self._human += sum(float(item.human_success) for item in observations)
        self._model += sum(float(item.model_success) for item in observations)
        self._mass += sum(item.model_probability_mass for item in observations)
        self._games += len(counts)
        self._squared += sum(count * count for count in counts.values())

    def summary(self) -> PointReferenceComparison:
        """Return the comparison the added opportunities support."""

        if not self._opportunities:
            raise CurveComparisonError(
                "a point human-reference comparison needs at least one opportunity"
            )
        return PointReferenceComparison(
            games=self._games,
            opportunities=self._opportunities,
            effective_sample_size=(
                (self._opportunities * self._opportunities) / self._squared
            ),
            human_rate=self._human / self._opportunities,
            model_rate=self._model / self._opportunities,
            model_probability_mass=self._mass / self._opportunities,
        )


def compare_reference_rate(
    observations: Sequence[PairedRateObservation],
) -> PointReferenceComparison:
    """Compare paired human and model rates with game-clustered support.

    This is the point form of the human-reference curve shape. Rare decision
    predicates use it because each rating band contributes one rate rather
    than a curve, while retaining the same effective-support accounting.
    """

    support = ReferenceRateSupport()
    support.add(observations)
    return support.summary()


@dataclass(frozen=True)
class CurveSpec:
    """The frozen, declared shape of one human-reference curve comparison.

    ``neighbours`` is the bandwidth, expressed as a neighbour count so the span
    adapts to the human reference's density rather than fixing a rating width
    that is far too wide where the corpus is thick and far too narrow where it
    is thin. It belongs to the benchmark version: two results computed at
    different neighbour counts are not on the same line, however similar their
    inputs.
    """

    name: str
    version: int
    quantity: CurveQuantity
    neighbours: int
    grid: tuple[float, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(_NAME_PATTERN, self.name) is None:
            raise CurveComparisonError(
                f"curve name {self.name!r} must be lowercase and dashed"
            )
        if self.version < 1:
            raise CurveComparisonError(
                "a curve spec must declare a version of 1 or more"
            )
        if self.neighbours < 2:
            raise CurveComparisonError(
                "a curve bandwidth needs at least two neighbours to smooth over"
            )
        if len(self.grid) < 2:
            raise CurveComparisonError("a curve grid needs at least two rating points")
        if any(not math.isfinite(rating) for rating in self.grid):
            raise CurveComparisonError("curve grid ratings must be finite")
        if any(
            later <= earlier
            for earlier, later in zip(self.grid, self.grid[1:], strict=False)
        ):
            raise CurveComparisonError("curve grid ratings must strictly increase")

    def as_record(self) -> dict[str, Any]:
        """Return the spec record every artifact carries."""

        return {
            "name": self.name,
            "version": self.version,
            "quantity": self.quantity.value,
            "neighbours": self.neighbours,
            "grid": list(self.grid),
        }


def rating_grid(low: float, high: float, points: int) -> tuple[float, ...]:
    """Return an evenly spaced rating grid, which is what a model side plays."""

    if points < 2:
        raise CurveComparisonError("a rating grid needs at least two points")
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        raise CurveComparisonError("a rating grid needs a finite, increasing range")
    step = (high - low) / (points - 1)
    return tuple(low + step * index for index in range(points))


@dataclass(frozen=True)
class LocalEstimate:
    """One side's curve at one evaluation point.

    The effective sample size is Kish's, so a neighbourhood whose weight sits
    on a handful of games reports a small number even when many games are
    nominally inside it. That is what stops a difference where the human
    reference is thin from reading like one where it is dense.
    """

    value: float | None
    distribution: Mapping[str, float] | None
    effective_sample_size: float
    games: int

    @property
    def supported(self) -> bool:
        """Return whether any game fell inside this neighbourhood."""

        return self.games > 0

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one estimate."""

        return {
            "value": self.value,
            "distribution": (
                None if self.distribution is None else dict(self.distribution)
            ),
            "effective_sample_size": self.effective_sample_size,
            "games": self.games,
        }


@dataclass(frozen=True)
class CurvePoint:
    """One evaluation point of the comparison, with the bandwidth it used."""

    rating: float
    bandwidth: float
    human: LocalEstimate
    model: LocalEstimate
    #: ``None`` when no generated game fell inside the human-forced radius, so
    #: the point has a human curve and nothing to compare it against.
    distance: float | None

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one curve point."""

        return {
            "rating": self.rating,
            "bandwidth": self.bandwidth,
            "human": self.human.as_record(),
            "model": self.model.as_record(),
            "distance": self.distance,
        }


@dataclass(frozen=True)
class TracePoint:
    """One estimated curve point of one side, without its counterpart."""

    rating: float
    estimate: LocalEstimate

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one trace point."""

        return {"rating": self.rating, **self.estimate.as_record()}


@dataclass(frozen=True)
class CurveTrace:
    """One curve, labeled and pinned to the series it was measured on.

    A trace is what a chart draws. It carries its fingerprint because two
    curves may be drawn as one continuous comparison only when they sit on the
    same series.
    """

    label: str
    fingerprint: str
    quantity: CurveQuantity
    points: tuple[TracePoint, ...]

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one trace."""

        return {
            "label": self.label,
            "fingerprint": self.fingerprint,
            "quantity": self.quantity.value,
            "points": [point.as_record() for point in self.points],
        }


@dataclass(frozen=True)
class CurveOverlay:
    """Curves that may be drawn together, and the series they belong to."""

    series: str
    traces: tuple[CurveTrace, ...]


@dataclass(frozen=True)
class CurveDispersions:
    """A comparison's own spreads, estimated as the comparison was computed.

    These are what a **delta between two measurements** is floored by —
    normally two checkpoints read against the same human reference. Only the
    model side is resampled, because both checkpoints face the identical
    reference and its sampling error cancels in their difference; including it
    would inflate every floor and hide real movement.

    A distributional distance has no series-wide spread to look up, because it
    depends on how many games this particular reading generated. So it is
    estimated here and carried on the measurement rather than characterized
    once and stored separately.

    ``method`` says how these were arrived at, because not every model side is
    resampled to reach one. A side that cannot vary is stated at zero and
    carries neither resamples nor streams, and the two cases are not
    distinguishable from the values alone: a bootstrap over plentiful games also
    lands near zero.

    ``streams`` is what the bound rests on, and is not recoverable from the game
    count: a grid multiplies games without multiplying the draws behind them.
    """

    conditional: MetricDispersion
    pooled: MetricDispersion
    model_variation: MetricDispersion
    resamples: int
    method: str
    streams: int

    @property
    def conditional_floor(self) -> float:
        """Return what a delta against a reading like this one would clear."""

        return self_combined_floor(self.conditional)

    @property
    def pooled_floor(self) -> float:
        """Return the same for the pooled arm's distance."""

        return self_combined_floor(self.pooled)

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one comparison's spreads."""

        return {
            "method": self.method,
            "resamples": self.resamples,
            "streams": self.streams,
            "conditional": self.conditional.model_dump(mode="json"),
            "pooled": self.pooled.model_dump(mode="json"),
            "model_variation": self.model_variation.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class CurveReferences:
    """The levels a model that already matches would still read at.

    Floors and references answer different questions and are not
    interchangeable. A floor says how far a number moves between two runs when
    nothing changed, which is what qualifies a delta between checkpoints. A
    reference says what the number reads at when there is nothing to find at
    all, which is what a distance needs: two finite samples of one population
    never agree exactly, and neither does a smoothed curve with itself.

    Both nulls are estimated from the comparison's own resampling.
    ``conditional`` and ``pooled`` come from pairing each side's sampling noise
    independently, which is the distance the two curves would show if they
    estimated the same behavior. ``flat`` comes from permuting which generated
    game was played at which rating, which is the curve variation a model with
    no rating response would still show.
    """

    conditional: float
    pooled: float
    flat: float
    replicates: int
    coverage: float

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one comparison's reference levels."""

        return {
            "replicates": self.replicates,
            "coverage": self.coverage,
            "conditional": self.conditional,
            "pooled": self.pooled,
            "flat": self.flat,
        }


@dataclass(frozen=True)
class CategoryShare:
    """One category's share on each side, beside the delta between them.

    Family granularity follows naming convention rather than a uniform level of
    abstraction: the largest family holds a few hundred lines and the median
    holds a handful. A distributional distance is mass-weighted so tiny
    families contribute proportionally little, but a reader looking at a
    per-category delta needs the mass beside it, because a share change on a
    broad family and one on a narrow line are not the same finding.
    """

    category: str
    human: float
    model: float

    @property
    def delta(self) -> float:
        """Return how much more of the model's play this category takes."""

        return self.model - self.human

    @property
    def mass(self) -> float:
        """Return the category's size, taken from the human reference.

        The reference side rather than the mean of the two, because "how big is
        this family" is a question about human play; a model that invents mass
        in a narrow line should read as a large delta on a small category
        rather than inflating the category itself.
        """

        return self.human

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one category's drill-down row."""

        return {
            "category": self.category,
            "human": self.human,
            "model": self.model,
            "delta": self.delta,
            "mass": self.mass,
        }


@dataclass(frozen=True)
class CurveMetrics:
    """Which registered metrics a benchmark reports one comparison as.

    The shape does not own metric identities. Each benchmark names its own,
    because a repertoire distance and a game-length distance are different
    series however identically they are computed.
    """

    conditional: str
    pooled: str
    model_variation: str | None = None


@dataclass(frozen=True)
class CurveComparison:
    """One benchmark's whole reading: two curves, two distances, and a verdict."""

    spec: CurveSpec
    points: tuple[CurvePoint, ...]
    human_pooled: LocalEstimate
    model_pooled: LocalEstimate
    #: Mean distance between the curves over the evaluation points the model
    #: supports. This is the rating-conditional reading.
    conditional_distance: float
    #: Distance between the two rating-free readings, each of which is that
    #: side's own curve averaged across the grid.
    pooled_distance: float
    #: How much each curve moves across the rating range, measured as the mean
    #: distance of its points from its own grid average. Reported for both
    #: sides because a flat model curve only means something beside the human
    #: curve it is supposed to follow.
    human_variation: float
    model_variation: float
    human_games: int
    model_games: int
    #: The spreads a delta against another checkpoint's distance is floored by.
    dispersions: CurveDispersions | None
    #: Null levels, for reading one distance on its own.
    references: CurveReferences | None

    @property
    def unsupported_points(self) -> int:
        """Return how many grid points no generated game reached."""

        return sum(1 for point in self.points if point.distance is None)

    @property
    def response(self) -> RatingResponse:
        """Return how this comparison reads, judged against its own nulls.

        Order matters. A model whose pooled distribution is wrong is not first
        a rating-response finding, and a model whose curve already agrees with
        the human one needs no flatness reading at all: a flat model matching a
        flat human reference is matching, not ignoring its input.
        """

        if self.references is None:
            return RatingResponse.UNKNOWN
        if self.pooled_distance > self.references.pooled:
            return RatingResponse.MISMATCH
        if self.conditional_distance <= self.references.conditional:
            return RatingResponse.MATCHES
        if self.model_variation <= self.references.flat:
            return RatingResponse.AVERAGE_HUMAN
        return RatingResponse.DIVERGENT_RESPONSE

    def category_shares(self) -> tuple[CategoryShare, ...]:
        """Return the per-category drill-down behind a categorical distance.

        Taken from the rating-free reading, which is each side's own curve
        averaged over the grid, so the shares are on the same rating mixture the
        pooled distance is computed from and sum to that distance.

        Ordered by the larger of the two sides, because that is the order the
        rows matter in: a delta on a category neither side reaches is noise
        wearing the shape of a finding, while one only the model produces is
        the whole of its own contribution to the distance and would sort last
        behind every category the reference holds.
        """

        if self.spec.quantity is not CurveQuantity.CATEGORICAL:
            return ()
        human = self.human_pooled.distribution or {}
        model = self.model_pooled.distribution or {}
        shares = tuple(
            CategoryShare(
                category=category,
                human=float(human.get(category, 0.0)),
                model=float(model.get(category, 0.0)),
            )
            for category in sorted({*human, *model})
        )
        return tuple(
            sorted(
                shares,
                key=lambda share: (-max(share.mass, share.model), share.category),
            )
        )

    def trace(self, side: str, *, label: str, fingerprint: str) -> CurveTrace:
        """Return one side's curve as a drawable, series-pinned trace."""

        if side not in ("human", "model"):
            raise CurveComparisonError(f"unknown curve side: {side}")
        return CurveTrace(
            label=label,
            fingerprint=fingerprint,
            quantity=self.spec.quantity,
            points=tuple(
                TracePoint(
                    rating=point.rating,
                    estimate=point.human if side == "human" else point.model,
                )
                for point in self.points
            ),
        )

    def traces(
        self,
        *,
        model_label: str,
        fingerprint: str,
        human_label: str = HUMAN_REFERENCE_LABEL,
    ) -> tuple[CurveTrace, CurveTrace]:
        """Return the human and model traces of this comparison, in that order."""

        return (
            self.trace("human", label=human_label, fingerprint=fingerprint),
            self.trace("model", label=model_label, fingerprint=fingerprint),
        )

    def measurements(
        self,
        metrics: CurveMetrics,
        *,
        data: DataComponent | None = None,
        workload: WorkloadComponent | None = None,
    ) -> tuple[Measurement, ...]:
        """Return the scalar measurements the committed summary tier records.

        Each distance carries the spread estimated for it, so a report combines
        it with the other reading's rather than looking a floor up for a series
        that cannot have a series-wide one in the first place.

        A comparison whose model side was generated rather than scored passes a
        workload instead of a data component: what identifies that series is the
        recipe the games were played under, per decision 0020. The comparison
        itself does not care which, so it forwards whichever it is given.
        """

        games = self.human_games + self.model_games
        reported: list[tuple[str, float, MetricDispersion | None]] = [
            (
                metrics.conditional,
                self.conditional_distance,
                None if self.dispersions is None else self.dispersions.conditional,
            ),
            (
                metrics.pooled,
                self.pooled_distance,
                None if self.dispersions is None else self.dispersions.pooled,
            ),
        ]
        if metrics.model_variation is not None:
            reported.append(
                (
                    metrics.model_variation,
                    self.model_variation,
                    (
                        None
                        if self.dispersions is None
                        else self.dispersions.model_variation
                    ),
                )
            )
        return tuple(
            measurement(
                metric,
                value,
                data=data,
                workload=workload,
                sample_size=games,
                dispersion=dispersion,
            )
            for metric, value, dispersion in reported
        )

    def as_detail_record(self) -> dict[str, Any]:
        """Return the curve payload written to the machine-local detail tier.

        Points are stored as data rather than as a rendered image, so drawing
        several checkpoints against one human reference later is a query over
        payloads that already exist.
        """

        return {
            "version": CURVE_COMPARISON_VERSION,
            "spec": self.spec.as_record(),
            "human_games": self.human_games,
            "model_games": self.model_games,
            "conditional_distance": self.conditional_distance,
            "pooled_distance": self.pooled_distance,
            "human_variation": self.human_variation,
            "model_variation": self.model_variation,
            "unsupported_points": self.unsupported_points,
            "response": self.response.value,
            "dispersions": (
                None if self.dispersions is None else self.dispersions.as_record()
            ),
            "references": (
                None if self.references is None else self.references.as_record()
            ),
            "pooled": {
                "human": self.human_pooled.as_record(),
                "model": self.model_pooled.as_record(),
            },
            "categories": [share.as_record() for share in self.category_shares()],
            "points": [point.as_record() for point in self.points],
        }


@dataclass(frozen=True)
class BandwidthSelection:
    """The bandwidth one cross-validation pass chose, and what it considered."""

    neighbours: int
    error: float
    observations: int
    candidates: tuple[tuple[int, float], ...]

    def as_record(self) -> dict[str, Any]:
        """Return the record to quote when declaring a spec's bandwidth."""

        return {
            "neighbours": self.neighbours,
            "error": self.error,
            "observations": self.observations,
            "candidates": [
                {"neighbours": neighbours, "error": error}
                for neighbours, error in self.candidates
            ],
        }


def compare_curves(
    *,
    spec: CurveSpec,
    human: Sequence[Observation],
    model: Sequence[Observation],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    coverage: float = DEFAULT_COVERAGE,
    confidence: float = DEFAULT_CONFIDENCE,
    model_varies: bool = True,
    references: bool = True,
) -> CurveComparison:
    """Compare a generated curve against the human reference it should match.

    The human reference forces the local bandwidth at every evaluation point
    and the generated side is estimated at exactly that bandwidth, so the
    smoothing bias in the two curves cancels in their difference instead of
    accumulating in it.

    ``model_varies`` says whether re-measuring the model side draws different
    games at all. A caller whose model side is deterministic — greedy seats at
    temperature zero, or a fixed pass over fixed inputs — passes ``False``, and
    the spread is then stated at zero rather than bootstrapped: another
    measurement replays these games, so there is no evaluation noise to bound
    and resampling them would report the spread of a draw that is not going to
    be redrawn.

    ``references`` may be turned off by a caller that needs only the spread.
    The null levels cost a permutation pass per replicate on top of the
    resampling, which a sweep recomputing this comparison at every book ply
    pays once per ply for a reading it does not use.

    A caller whose model side plays several ratings from one random stream says
    so on each ``Observation``, because that is what the resampling unit is. A
    side that leaves it unset is drawn game by game, which is right where each
    game came out of its own draw and understates the spread where it did not.
    """

    if len(human) < spec.neighbours:
        raise CurveComparisonError(
            f"curve {spec.name!r} smooths over {spec.neighbours} neighbours but "
            f"the human reference has {len(human)} game(s)"
        )
    if not model:
        raise CurveComparisonError(
            f"curve {spec.name!r} has no generated games to compare"
        )

    categories = _categories(spec, human, model)
    human_side = _Side.prepare(human, spec, categories)
    model_side = _Side.prepare(model, spec, categories)

    weights = (
        np.ones((1, len(human)), dtype=np.float64),
        np.ones((1, len(model)), dtype=np.float64),
    )
    reading = _read(spec, human_side, model_side, *weights)
    generator = np.random.default_rng(seed)
    dispersions, reference_levels = _resample(
        spec,
        human_side,
        model_side,
        reading,
        resamples=resamples,
        generator=generator,
        coverage=coverage,
        confidence=confidence,
        model_varies=model_varies,
        references=references,
    )

    points = tuple(
        CurvePoint(
            rating=rating,
            bandwidth=float(reading.radii[0, index]),
            human=_estimate_at(spec, categories, reading.human, index),
            model=_estimate_at(spec, categories, reading.model, index),
            distance=(
                float(reading.distances[0, index])
                if bool(reading.supported[0, index])
                else None
            ),
        )
        for index, rating in enumerate(spec.grid)
    )
    return CurveComparison(
        spec=spec,
        points=points,
        human_pooled=_pooled_estimate(
            spec, categories, reading.human_pooled, reading.human, reading.supported
        ),
        model_pooled=_pooled_estimate(
            spec, categories, reading.model_pooled, reading.model, reading.supported
        ),
        conditional_distance=float(reading.conditional[0]),
        pooled_distance=float(reading.pooled[0]),
        human_variation=float(reading.human_variation[0]),
        model_variation=float(reading.model_variation[0]),
        human_games=len(human),
        model_games=len(model),
        dispersions=dispersions,
        references=reference_levels,
    )


@dataclass(frozen=True)
class ReferenceCurve:
    """One side's curve estimated on its own, with no counterpart to subtract.

    A comparison needs one observation per game on both sides. A model side
    computed exactly rather than played has no games at all — it is a
    distribution — so it cannot go through ``compare_curves``, but it still has
    to meet the human reference smoothed exactly as every other reading
    smooths it. This is that half.
    """

    spec: CurveSpec
    points: tuple[TracePoint, ...]
    pooled: LocalEstimate
    categories: tuple[str, ...]

    def distribution_at(self, rating: float) -> Mapping[str, float] | None:
        """Return the estimated categorical distribution at one grid point."""

        for point in self.points:
            if point.rating == rating:
                return point.estimate.distribution
        return None

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one estimated reference curve."""

        return {
            "spec": self.spec.as_record(),
            "categories": list(self.categories),
            "pooled": self.pooled.as_record(),
            "points": [point.as_record() for point in self.points],
        }


def estimate_curve(
    spec: CurveSpec,
    observations: Sequence[Observation],
    *,
    categories: Sequence[str] = (),
) -> ReferenceCurve:
    """Estimate one side's curve at the spec's grid, at its own bandwidth.

    ``categories`` extends the vocabulary beyond what these observations
    contain, which is how a category the counterpart produces and this side
    never does stays visible as a zero rather than vanishing from the
    comparison.
    """

    if len(observations) < spec.neighbours:
        raise CurveComparisonError(
            f"curve {spec.name!r} smooths over {spec.neighbours} neighbours but "
            f"the reference has {len(observations)} observation(s)"
        )
    vocabulary: tuple[str, ...] = ()
    if spec.quantity is CurveQuantity.CATEGORICAL:
        observed = _observed_categories(spec.quantity, observations)
        vocabulary = tuple(sorted({*observed, *categories}))
    side = _Side.prepare(observations, spec, vocabulary)
    weights = np.ones((1, len(observations)), dtype=np.float64)
    radii = _radii(side, weights, spec.neighbours)
    local = _local(side, weights, radii)
    supported = local.games > 0
    pooled = _masked_centre(local.values, supported)
    return ReferenceCurve(
        spec=spec,
        points=tuple(
            TracePoint(
                rating=rating,
                estimate=_estimate_at(spec, vocabulary, local, index),
            )
            for index, rating in enumerate(spec.grid)
        ),
        pooled=_pooled_estimate(spec, vocabulary, pooled, local, supported),
        categories=vocabulary,
    )


def distribution_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    """Return the total variation distance between two label distributions.

    The same distance ``compare_curves`` uses for a categorical quantity, so an
    exactly computed distribution and a sampled one land on one scale.
    """

    return 0.5 * sum(
        abs(float(left.get(category, 0.0)) - float(right.get(category, 0.0)))
        for category in {*left, *right}
    )


def select_neighbours(
    observations: Sequence[Observation],
    *,
    quantity: CurveQuantity,
    candidates: Sequence[int] = DEFAULT_NEIGHBOUR_CANDIDATES,
) -> BandwidthSelection:
    """Choose a bandwidth from the human reference by cross-validation.

    This is an offline step whose output is declared in a ``CurveSpec`` and
    then frozen. Calling it per run would re-select the bandwidth from whatever
    reference that run happened to read, which would mean two checkpoints were
    measured differently and their series would not be comparable.

    Leave-one-out squared error is the criterion for both kinds: on a scalar it
    is the usual prediction error, and on a categorical quantity it is the
    Brier score of the estimated distribution against the observed label, which
    a zero probability cannot blow up the way a log score can.
    """

    if len(observations) < 3:
        raise CurveComparisonError(
            "selecting a bandwidth needs at least three human observations"
        )
    usable = sorted({int(candidate) for candidate in candidates})
    usable = [
        candidate for candidate in usable if 2 <= candidate <= len(observations) - 1
    ]
    if not usable:
        raise CurveComparisonError(
            "no candidate bandwidth fits the human reference; every candidate is "
            "below two neighbours or larger than the reference itself"
        )

    categories = _observed_categories(quantity, observations)
    order = np.argsort(
        [observation.rating for observation in observations], kind="stable"
    )
    ratings = np.asarray(
        [observations[index].rating for index in order], dtype=np.float64
    )
    values = _values(quantity, [observations[index] for index in order], categories)

    scored = tuple(
        (candidate, _leave_one_out_error(ratings, values, candidate))
        for candidate in usable
    )
    best = min(scored, key=lambda entry: (entry[1], entry[0]))
    return BandwidthSelection(
        neighbours=best[0],
        error=best[1],
        observations=len(observations),
        candidates=scored,
    )


def curve_overlays(
    traces: Iterable[CurveTrace],
    *,
    bridges: BridgeIndex | None = None,
) -> tuple[CurveOverlay, ...]:
    """Group traces into the sets that may be drawn as one comparison.

    Curves whose fingerprints do not match are not one continuous comparison,
    so they come back as separate overlays rather than as one line a reader
    would take for a history. Recorded bridges are followed, because a series
    legitimately rejoined is one series.
    """

    index = bridges if bridges is not None else BridgeIndex()
    grouped: dict[str, list[CurveTrace]] = {}
    for trace in traces:
        grouped.setdefault(index.series(trace.fingerprint), []).append(trace)

    overlays: list[CurveOverlay] = []
    for series in sorted(grouped):
        seen: dict[str, CurveTrace] = {}
        for trace in grouped[series]:
            # One series has one human reference by construction, so a repeated
            # label is the same curve arriving with each checkpoint rather than
            # a second line to draw.
            seen.setdefault(trace.label, trace)
        overlays.append(CurveOverlay(series=series, traces=tuple(seen.values())))
    return tuple(overlays)


# Every dataclass below that holds an array compares by identity: a generated
# ``__eq__`` asks an elementwise comparison for a truth value it has none of.
# Compare the arrays a caller means, with ``numpy.array_equal``.
@dataclass(frozen=True, eq=False)
class _Side:
    """One side's observations, ordered for repeated local estimation."""

    ratings: np.ndarray
    values: np.ndarray
    order: np.ndarray
    distances: np.ndarray
    streams: np.ndarray

    @property
    def size(self) -> int:
        """Return how many observations this side holds."""

        return int(self.ratings.shape[0])

    @property
    def stream_count(self) -> int:
        """Return how many independent draws this side's observations came from."""

        return int(self.streams.max()) + 1 if self.streams.size else 0

    @classmethod
    def prepare(
        cls,
        observations: Sequence[Observation],
        spec: CurveSpec,
        categories: tuple[str, ...],
    ) -> _Side:
        """Sort observations by distance to each grid point, once."""

        ratings = np.asarray(
            [observation.rating for observation in observations], dtype=np.float64
        )
        if not np.all(np.isfinite(ratings)):
            raise CurveComparisonError("every observation needs a finite rating")
        values = _values(spec.quantity, observations, categories)
        grid = np.asarray(spec.grid, dtype=np.float64)
        distances = np.abs(ratings[None, :] - grid[:, None])
        order = np.argsort(distances, axis=1, kind="stable")
        return cls(
            ratings=ratings,
            values=values,
            order=order,
            distances=np.take_along_axis(distances, order, axis=1),
            streams=_stream_labels(observations),
        )


def _stream_labels(observations: Sequence[Observation]) -> np.ndarray:
    """Return which observations a resample has to redraw together.

    A side that names no stream anywhere — every human reference this shape
    compares against — takes the cheap path rather than a dictionary over
    thousands of games that would answer ``arange``.

    Labels run densely from zero, which is what lets a count be a maximum plus
    one and a draw be gathered by label.
    """

    if all(observation.stream is None for observation in observations):
        return np.arange(len(observations), dtype=np.int64)
    labels = np.empty(len(observations), dtype=np.int64)
    index: dict[object, int] = {}
    for position, observation in enumerate(observations):
        # Keyed apart from the stream ids so an unnamed game stays its own unit
        # rather than joining whichever stream shares its position.
        key = ("own", position) if observation.stream is None else observation.stream
        labels[position] = index.setdefault(key, len(index))
    return labels


@dataclass(frozen=True, eq=False)
class _Local:
    """Local estimates of one side across every replicate and grid point."""

    values: np.ndarray
    effective_sample_size: np.ndarray
    games: np.ndarray


@dataclass(frozen=True, eq=False)
class _Reading:
    """Everything one pass over a set of resampling weights produces."""

    radii: np.ndarray
    human: _Local
    model: _Local
    human_pooled: np.ndarray
    model_pooled: np.ndarray
    supported: np.ndarray
    distances: np.ndarray
    conditional: np.ndarray
    pooled: np.ndarray
    human_variation: np.ndarray
    model_variation: np.ndarray


def _read(
    spec: CurveSpec,
    human: _Side,
    model: _Side,
    human_weights: np.ndarray,
    model_weights: np.ndarray,
) -> _Reading:
    """Estimate both curves and reduce them, for every replicate at once."""

    radii = _radii(human, human_weights, spec.neighbours)
    return _reduce(
        spec, radii, _local(human, human_weights, radii), model, model_weights
    )


def _reduce(
    spec: CurveSpec,
    radii: np.ndarray,
    human_local: _Local,
    model: _Side,
    model_weights: np.ndarray,
) -> _Reading:
    """Reduce an already-estimated human curve against a model side.

    The human estimate may carry one replicate against a model side carrying
    many. That is what a bootstrap resampling the model alone reads: the human
    curve it holds fixed is the point pass's own, so it is broadcast across the
    replicates rather than re-estimated identically once per resample.

    Widening it here rather than letting each reduction broadcast on its own is
    what keeps a ``_Reading`` one shape throughout. ``np.broadcast_to`` returns
    a read-only view, so the saving this exists for survives it: the human curve
    is still estimated once, and only its shape is repeated.
    """

    replicates = int(model_weights.shape[0])
    grid_points = int(human_local.games.shape[1])
    radii = np.broadcast_to(radii, (replicates, grid_points))
    human_local = _Local(
        values=np.broadcast_to(
            human_local.values, (replicates, grid_points, human_local.values.shape[2])
        ),
        effective_sample_size=np.broadcast_to(
            human_local.effective_sample_size, (replicates, grid_points)
        ),
        games=np.broadcast_to(human_local.games, (replicates, grid_points)),
    )
    model_local = _local(model, model_weights, radii)

    supported = (model_local.games > 0) & (human_local.games > 0)
    distances = _distance(spec.quantity, human_local.values, model_local.values)
    conditional = _masked_mean(distances, supported)

    # The rating-free reading averages each side's own curve over the grid
    # rather than pooling raw observations. Pooling raw games would compare a
    # human corpus concentrated around the middle of the rating range against
    # a generated set spread evenly across it, so the distance would report the
    # difference between two rating designs rather than anything about the
    # model. Averaging the curves keeps both sides on one rating mixture, and
    # keeps the smoothing bias cancelling here as it does point by point.
    human_pooled = _masked_centre(human_local.values, supported)
    model_pooled = _masked_centre(model_local.values, supported)
    pooled = _distance(spec.quantity, human_pooled, model_pooled)

    return _Reading(
        radii=radii,
        human=human_local,
        model=model_local,
        human_pooled=human_pooled,
        model_pooled=model_pooled,
        supported=supported,
        distances=distances,
        conditional=conditional,
        pooled=pooled,
        human_variation=_variation(spec.quantity, human_local, human_pooled, supported),
        model_variation=_variation(spec.quantity, model_local, model_pooled, supported),
    )


def _radii(side: _Side, weights: np.ndarray, neighbours: int) -> np.ndarray:
    """Return the radius holding ``neighbours`` observations, per grid point.

    The bandwidth is a neighbour count rather than a rating width, so the span
    widens automatically where the human reference is thin and tightens where
    it is dense.
    """

    replicates = weights.shape[0]
    grid_points = side.order.shape[0]
    radii = np.empty((replicates, grid_points), dtype=np.float64)
    for index in range(grid_points):
        ordered = weights[:, side.order[index]]
        cumulative = np.cumsum(ordered, axis=1)
        reached = np.clip((cumulative < neighbours).sum(axis=1), 0, side.size - 1)
        radii[:, index] = side.distances[index][reached]
    return radii


def _local(side: _Side, weights: np.ndarray, radii: np.ndarray) -> _Local:
    """Estimate one side at every grid point, under the given radii.

    Weights inside a neighbourhood are tricube, which is the usual local-fit
    kernel: it goes smoothly to zero at the radius instead of jumping, so a
    curve does not step whenever one game enters or leaves the window.
    """

    replicates = weights.shape[0]
    grid_points = side.order.shape[0]
    dimensions = side.values.shape[1]
    values = np.full((replicates, grid_points, dimensions), np.nan, dtype=np.float64)
    effective = np.zeros((replicates, grid_points), dtype=np.float64)
    games = np.zeros((replicates, grid_points), dtype=np.int64)

    for index in range(grid_points):
        distances = side.distances[index]
        radius = radii[:, index][:, None]
        inside = distances[None, :] <= radius + _RADIUS_TOLERANCE
        scaled = np.divide(
            distances[None, :],
            radius,
            out=np.zeros((replicates, side.size), dtype=np.float64),
            where=radius > 0.0,
        )
        kernel = np.clip(1.0 - np.clip(scaled, 0.0, 1.0) ** 3, 0.0, 1.0) ** 3
        ordered_weights = weights[:, side.order[index]]
        weighted = ordered_weights * kernel * inside
        total = weighted.sum(axis=1)
        # Tricube weight is zero exactly at the radius, so a neighbourhood
        # whose games all sit on it would read as empty. Fall back to an
        # unweighted local mean there rather than dropping a point that does
        # have games behind it.
        degenerate = total <= 0.0
        if bool(degenerate.any()):
            uniform = ordered_weights * inside
            weighted = np.where(degenerate[:, None], uniform, weighted)
            total = weighted.sum(axis=1)
        squared = (weighted**2).sum(axis=1)
        ordered_values = side.values[side.order[index]]
        np.divide(
            weighted @ ordered_values,
            total[:, None],
            out=values[:, index, :],
            where=total[:, None] > 0.0,
        )
        np.divide(
            total**2,
            squared,
            out=effective[:, index],
            where=squared > 0.0,
        )
        games[:, index] = (weighted > 0.0).sum(axis=1)
    return _Local(values=values, effective_sample_size=effective, games=games)


def _distance(
    quantity: CurveQuantity,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Return the distance between two estimates, along their last axis.

    A scalar quantity uses the absolute difference and a categorical one uses
    total variation, so both land on a scale where zero is agreement and the
    number means the same thing at every rating.
    """

    absolute: np.ndarray = np.abs(left - right).sum(axis=-1)
    if quantity is CurveQuantity.CATEGORICAL:
        return 0.5 * absolute
    return absolute


def _variation(
    quantity: CurveQuantity,
    local: _Local,
    centre: np.ndarray,
    supported: np.ndarray,
) -> np.ndarray:
    """Return how far one curve moves from its own grid average.

    This is what makes a flat curve reportable. A model that matches the
    pooled human distribution while reading near zero here has learned the
    average human and is ignoring its rating input, which is a different
    finding from behaving unlike humans at all.
    """

    distances = _distance(quantity, local.values, centre[:, None, :])
    return _masked_mean(distances, supported)


def _masked_centre(values: np.ndarray, supported: np.ndarray) -> np.ndarray:
    """Return each replicate's average estimate over its supported points."""

    weights = supported.astype(np.float64)
    total = weights.sum(axis=1)
    filled = np.where(supported[:, :, None], values, 0.0)
    centre = np.full((values.shape[0], values.shape[2]), np.nan)
    np.divide(
        np.einsum("rg,rgd->rd", weights, filled),
        total[:, None],
        out=centre,
        where=total[:, None] > 0.0,
    )
    return centre


def _masked_mean(values: np.ndarray, supported: np.ndarray) -> np.ndarray:
    """Average over supported grid points, leaving unsupported ones out."""

    weights = supported.astype(np.float64)
    total = weights.sum(axis=1)
    summed = (np.where(supported, values, 0.0)).sum(axis=1)
    mean = np.full(values.shape[0], np.nan)
    np.divide(summed, total, out=mean, where=total > 0.0)
    return mean


def _resample(
    spec: CurveSpec,
    human: _Side,
    model: _Side,
    point: _Reading,
    *,
    resamples: int,
    generator: np.random.Generator,
    coverage: float,
    confidence: float,
    model_varies: bool,
    references: bool = True,
) -> tuple[CurveDispersions | None, CurveReferences | None]:
    """Estimate this comparison's own spreads and null levels in one pass.

    The two are estimated differently on purpose, because they answer different
    questions.

    A **spread** is what a delta between two measurements — in practice, two
    checkpoints — is floored by, once combined with the other reading's. Both
    are compared against the *same fixed* human reference, so human sampling
    error is common-mode and cancels in their difference: including it would
    inflate every floor and hide real movement. Only the model side is
    resampled, which is why this is evaluation noise rather than data-sampling
    noise. For a model side that varies the two coincide anyway, since a fresh
    draw of streams is exactly what another seed produces.

    Drawing its games as though each were independent, rather than the streams
    they came from, reports about half the movement a fresh seed produces;
    decision 0060 measured it.

    That coincidence is what ``model_varies`` denies, and ``compare_curves``
    documents what a caller passing it is claiming.

    A **reference** says what the distance would read at with nothing to find,
    which is an absolute statement about one reading rather than a difference.
    There the human side's own sampling error genuinely belongs, so it keeps
    the paired resampling.

    ``None`` means nothing could be estimated, which is a reportable state
    rather than an error: a spread invented from too few replicates would
    license every delta as a finding. The two are refused together for a model
    side that varies, since a null level's model half is read off the same
    replicates; a replayed side is the exception, and keeps its levels down to
    the two streams a resample needs to move it at all.
    """

    method = CURVE_BOOTSTRAP_METHOD if model_varies else CURVE_DETERMINISTIC_METHOD
    source = f"{spec.name} v{spec.version} {method}"

    # This family states a zero rather than bounding one, which the shared
    # arithmetic refuses. ``method`` says which of the two claims it carries: a
    # model side that replays exactly, or a resample that moved nothing.
    stated_zero = MetricDispersion(
        value=0.0,
        bound=0.0,
        source=source,
        estimator=method,
    )

    dispersions: CurveDispersions | None = None
    if not model_varies:
        # Exact rather than estimated, and the same for all three readings:
        # none of them can move when the games behind them cannot. Nothing
        # about it depends on resampling, so it stands where a bootstrap could
        # not — a model side too thin to resample is still replayed exactly.
        dispersions = CurveDispersions(
            conditional=stated_zero,
            pooled=stated_zero,
            model_variation=stated_zero,
            resamples=0,
            streams=0,
            method=method,
        )

    # Everything below is resampling, which only a bootstrapped spread or a
    # null level needs. A stated spread asked for on its own leaves nothing to
    # draw.
    if not model_varies and not references:
        return dispersions, None
    # A null level pairs each side's own sampling error, so it needs a model
    # side a resample can move at all; a spread needs one it can move often
    # enough to read a spread off, and three outcomes are not that. A replayed
    # side asks only the first of those, since its floor is stated rather than
    # estimated.
    minimum = MINIMUM_BOOTSTRAP_STREAMS if model_varies else 2
    if resamples < 2 or model.stream_count < minimum:
        return dispersions, None
    # The model side draws first, and the human side draws only when the null
    # levels that consume it were asked for. Both halves of that matter. Drawing
    # a whole multinomial over the reference for a floor-only caller to discard
    # is the largest cost left in that call; and taking the model's draw before
    # it is what keeps ``references`` a cost decision rather than a measurement
    # one, since the floor is then read off the same resamples either way.
    model_weights = _redraw(model, resamples, generator)
    paired = (
        _read(spec, human, model, _redraw(human, resamples, generator), model_weights)
        if references
        else None
    )
    if model_varies:
        # The human reference held exactly as measured, so this varies only in
        # which games the model produced — and the point pass already estimated
        # it at these radii.
        model_only = _reduce(spec, point.radii, point.human, model, model_weights)
        # The resample count says only how finely each replicate was read, so it
        # is how many streams the side holds that decides how far the bound sits
        # above the spread.
        streams = model.stream_count
        freedom = streams - 1
        rescale = plug_in_rescale(streams)
        # Asked of the curve rather than of the spread it reduces to. A draw
        # that could not move the model side leaves float noise rather than a
        # zero, which takes neither the stated zero below nor the shared
        # arithmetic's refusal of one — and a bound on 1e-16 is a floor that
        # clears every delta while reading as an ordinary estimate.
        curves = model_only.model.values
        unmoved = bool(
            np.all((curves == curves[:1]) | (np.isnan(curves) & np.isnan(curves[:1])))
        )
        bootstrapped: list[MetricDispersion] = []
        for values in (
            model_only.conditional,
            model_only.pooled,
            model_only.model_variation,
        ):
            observed = values[np.isfinite(values)]
            if observed.size < 2:
                return None, None
            dispersion = float(np.std(observed, ddof=1)) * rescale
            # Decision 0042 keeps this family's estimated zero where every other
            # estimator is refused one: the streams generated for a curve are
            # themselves the draw, so a distance no resample moved is a
            # statement about the play this reading produced rather than a
            # sample that came out flat. A curve the draw never moved states the
            # same zero, whatever the last bits of the reduction came out at.
            bootstrapped.append(
                stated_zero
                if unmoved or dispersion == 0.0
                else measured_dispersion(
                    dispersion,
                    degrees_of_freedom=freedom,
                    confidence=confidence,
                    source=source,
                    estimator=method,
                )
            )
        dispersions = CurveDispersions(
            conditional=bootstrapped[0],
            pooled=bootstrapped[1],
            model_variation=bootstrapped[2],
            resamples=resamples,
            method=method,
            streams=streams,
        )

    levels = (
        None
        if paired is None
        else _references(
            spec,
            point,
            paired,
            human,
            model,
            resamples=resamples,
            generator=generator,
            coverage=coverage,
        )
    )
    return dispersions, levels


def _redraw(
    side: _Side,
    resamples: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Draw this side's streams with replacement, one weight per observation.

    Drawing whole streams is also what stops a resample from moving how many
    games each grid point holds, wherever a stream reached every rating. Where a
    quantity drops games — a game that left the book on a waypoint chose no
    opening — a stream is missing from some points and a resample can still lose
    one; that variation is a re-run's too rather than an artifact of the draw.
    """

    count = side.stream_count
    drawn = generator.multinomial(
        count, np.full(count, 1.0 / count), size=resamples
    ).astype(np.float64)
    if count == side.size:
        # Every observation is its own stream, so the labels are already
        # ``arange`` and the gather below would copy the draw unchanged.
        return drawn
    return drawn[:, side.streams]


def _references(
    spec: CurveSpec,
    point: _Reading,
    resampled: _Reading,
    human: _Side,
    model: _Side,
    *,
    resamples: int,
    generator: np.random.Generator,
    coverage: float,
) -> CurveReferences | None:
    """Return the levels this comparison would read at with nothing to find.

    The distance nulls pair each side's own sampling noise independently, which
    is what two curves estimating the same behavior would differ by. The
    flatness null permutes which generated game was played at which rating,
    which destroys any rating response while leaving the sample, the grid, and
    the bandwidth exactly as they were.

    That permutation is the one thing here still drawn over games, and no
    permutation is the right shape for it. A model whose policy ignored the
    rating input would meet the same stream at every grid point and replay one
    game across the whole grid, so its curve would be exactly flat and the level
    this estimates is analytically zero. Permuting anything reports a level
    above that, and ``AVERAGE_HUMAN`` — the verdict it decides — fires too
    readily as a result. Decision 0060 records the argument and what replacing
    it would take.

    Each side's deviation is scaled to the fresh draw its plug-in resample
    understates, since these read that understatement off the same replicates a
    floor does. A distance is homogeneous in the deviations it is taken between,
    so the two scale before the distance rather than after it.
    """

    pairing = generator.permutation(resamples)
    supported = np.broadcast_to(point.supported, resampled.distances.shape)
    human_rescale = plug_in_rescale(human.stream_count)
    model_rescale = plug_in_rescale(model.stream_count)
    conditional = _masked_mean(
        _distance(
            spec.quantity,
            (resampled.human.values - point.human.values) * human_rescale,
            ((resampled.model.values - point.model.values) * model_rescale)[pairing],
        ),
        supported,
    )
    pooled = _distance(
        spec.quantity,
        (resampled.human_pooled - point.human_pooled) * human_rescale,
        ((resampled.model_pooled - point.model_pooled) * model_rescale)[pairing],
    )
    flat = _flat_variation(
        spec,
        model,
        point.radii,
        permutations=resamples,
        generator=generator,
    )

    levels: list[float] = []
    for values in (conditional, pooled, flat):
        observed = values[np.isfinite(values)]
        if observed.size < 2:
            return None
        levels.append(float(observed.mean() + coverage * observed.std(ddof=1)))
    return CurveReferences(
        conditional=levels[0],
        pooled=levels[1],
        flat=levels[2],
        replicates=resamples,
        coverage=coverage,
    )


def _flat_variation(
    spec: CurveSpec,
    model: _Side,
    radii: np.ndarray,
    *,
    permutations: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Return the curve variation of a model with its ratings permuted away."""

    weights = np.ones((1, model.size), dtype=np.float64)
    point_radii = radii[:1]
    observed = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled = replace(
            model, values=model.values[generator.permutation(model.size)]
        )
        local = _local(shuffled, weights, point_radii)
        supported = local.games > 0
        centre = _masked_centre(local.values, supported)
        observed[index] = _variation(spec.quantity, local, centre, supported)[0]
    return observed


def _estimate_at(
    spec: CurveSpec,
    categories: tuple[str, ...],
    local: _Local,
    index: int,
) -> LocalEstimate:
    """Return one stored estimate out of the array the point pass produced."""

    return _estimate(
        spec,
        categories,
        local.values[0, index],
        float(local.effective_sample_size[0, index]),
        int(local.games[0, index]),
    )


def _pooled_estimate(
    spec: CurveSpec,
    categories: tuple[str, ...],
    pooled: np.ndarray,
    local: _Local,
    supported: np.ndarray,
) -> LocalEstimate:
    """Return the rating-free estimate in the shape a reader stores.

    Its support is reported as the total across the grid points it averages,
    which is what stands behind the average even though neighbourhoods overlap
    and the games are therefore not independent of one another.
    """

    covered = supported[0]
    if not bool(covered.any()):
        return _estimate(spec, categories, pooled[0], 0.0, 0)
    return _estimate(
        spec,
        categories,
        pooled[0],
        float(local.effective_sample_size[0][covered].sum()),
        int(local.games[0][covered].sum()),
    )


def _estimate(
    spec: CurveSpec,
    categories: tuple[str, ...],
    values: np.ndarray,
    effective_sample_size: float,
    games: int,
) -> LocalEstimate:
    if games == 0:
        return LocalEstimate(
            value=None,
            distribution=None,
            effective_sample_size=0.0,
            games=0,
        )
    if spec.quantity is CurveQuantity.CATEGORICAL:
        return LocalEstimate(
            value=None,
            distribution={
                category: float(values[position])
                for position, category in enumerate(categories)
            },
            effective_sample_size=float(effective_sample_size),
            games=games,
        )
    return LocalEstimate(
        value=float(values[0]),
        distribution=None,
        effective_sample_size=float(effective_sample_size),
        games=games,
    )


def _categories(
    spec: CurveSpec,
    human: Sequence[Observation],
    model: Sequence[Observation],
) -> tuple[str, ...]:
    """Return the category vocabulary both sides are counted over.

    The union rather than the human side alone, so a family the model plays and
    humans never do shows up as a difference instead of vanishing.
    """

    if spec.quantity is not CurveQuantity.CATEGORICAL:
        return ()
    return _observed_categories(spec.quantity, (*human, *model))


def _observed_categories(
    quantity: CurveQuantity,
    observations: Iterable[Observation],
) -> tuple[str, ...]:
    if quantity is not CurveQuantity.CATEGORICAL:
        return ()
    categories = {
        observation.value
        for observation in observations
        if isinstance(observation.value, str)
    }
    if not categories:
        raise CurveComparisonError(
            "a categorical curve needs observations carrying category labels"
        )
    return tuple(sorted(categories))


def _values(
    quantity: CurveQuantity,
    observations: Sequence[Observation],
    categories: tuple[str, ...],
) -> np.ndarray:
    """Return the observation values in the shape the estimator averages.

    A categorical observation becomes an indicator row, so one weighted mean
    produces the whole local distribution rather than one pass per category.
    """

    if quantity is CurveQuantity.CATEGORICAL:
        positions = {category: index for index, category in enumerate(categories)}
        values = np.zeros((len(observations), len(categories)), dtype=np.float64)
        for row, observation in enumerate(observations):
            if not isinstance(observation.value, str):
                raise CurveComparisonError(
                    "a categorical curve needs a category label per observation, "
                    f"not {observation.value!r}"
                )
            values[row, positions[observation.value]] = 1.0
        return values

    numbers = []
    for observation in observations:
        if isinstance(observation.value, str):
            raise CurveComparisonError(
                "a scalar curve needs a number per observation, not "
                f"{observation.value!r}"
            )
        if not math.isfinite(observation.value):
            raise CurveComparisonError("scalar observations must be finite")
        numbers.append(observation.value)
    return np.asarray(numbers, dtype=np.float64).reshape(-1, 1)


def _leave_one_out_error(
    ratings: np.ndarray,
    values: np.ndarray,
    neighbours: int,
) -> float:
    """Return the mean leave-one-out squared error at one neighbour count.

    Ratings arrive sorted, so the nearest neighbours of an observation are
    inside a window around it and the search never scans the whole reference.
    """

    total = 0.0
    size = int(ratings.shape[0])
    for index in range(size):
        low = max(0, index - neighbours)
        high = min(size, index + neighbours + 1)
        window = np.concatenate(
            (np.arange(low, index), np.arange(index + 1, high)),
        )
        distances = np.abs(ratings[window] - ratings[index])
        nearest = window[np.argsort(distances, kind="stable")[:neighbours]]
        chosen = np.abs(ratings[nearest] - ratings[index])
        radius = float(chosen.max())
        if radius > 0.0:
            kernel = (1.0 - np.clip(chosen / radius, 0.0, 1.0) ** 3) ** 3
        else:
            kernel = np.ones_like(chosen)
        weight = kernel.sum()
        if weight <= 0.0:
            # Every neighbour sits exactly at the radius, which only happens
            # when the window is one tie: fall back to an unweighted mean.
            kernel = np.ones_like(chosen)
            weight = kernel.sum()
        predicted = (kernel @ values[nearest]) / weight
        total += float(((predicted - values[index]) ** 2).sum())
    return total / size


__all__ = [
    "CURVE_BOOTSTRAP_METHOD",
    "CURVE_COMPARISON_VERSION",
    "CURVE_DETERMINISTIC_METHOD",
    "DEFAULT_NEIGHBOUR_CANDIDATES",
    "DEFAULT_RESAMPLES",
    "HUMAN_REFERENCE_LABEL",
    "BandwidthSelection",
    "CategoryShare",
    "CurveComparison",
    "CurveComparisonError",
    "CurveDispersions",
    "CurveMetrics",
    "CurveOverlay",
    "CurvePoint",
    "CurveQuantity",
    "CurveReferences",
    "CurveSpec",
    "CurveTrace",
    "LocalEstimate",
    "Observation",
    "RatingResponse",
    "ReferenceCurve",
    "TracePoint",
    "compare_curves",
    "curve_overlays",
    "distribution_distance",
    "estimate_curve",
    "rating_grid",
    "select_neighbours",
]
