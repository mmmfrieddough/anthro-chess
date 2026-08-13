"""A reading's own dispersion, and the floor a delta between two of them faces.

A delta is not a finding until it is larger than the noise in the measurement.
Without a floor, a report can only say that a number moved, and a reader
comparing two checkpoints will confidently describe movement that is seed luck.

Every reading measures the spread of its own units and stores it: the
**dispersion**. Comparing two readings combines them, because the variance of a
difference is the sum of the two variances — :func:`combined_floor`. Nothing is
characterized ahead of a comparison, stored between runs, or looked up.

The dispersion a floor is built from is never the measured one. A dispersion
read off a handful of replicates is a point estimate sitting in the middle of
its own sampling distribution, so about half the time it lands below the spread
it is supposed to describe, and a floor built on it is too narrow exactly when
being too narrow matters. A floor that understates the noise licenses noise as
a finding, which is the failure floors exist to prevent, so every floor rests on
a **conservative upper confidence limit** on the dispersion instead — the
chi-squared bound for the degrees of freedom actually behind the estimate.

That makes a floor a tolerance bound rather than an interval, and the two
factors it carries answer two different questions. ``coverage`` says what
proportion of same-weights deltas the floor covers when the dispersion is
known. ``confidence`` says how sure the bound is that the dispersion is not
larger than assumed. They multiply, and the resulting claim is: with
``confidence``, this floor covers ``coverage`` of the deltas that noise alone
produces. Widening either one widens the floor; only more replicates narrow it
honestly.

``docs/decisions/0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md``
owns the design and what it deliberately gives up.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from anthro_chess.evaluation.results.records import MetricDispersion

#: Two-sided normal coverage for a 95% interval. A floor is a claim about how
#: far apart two measurements land when nothing changed, so it needs a stated
#: confidence rather than a bare standard deviation.
DEFAULT_COVERAGE = 1.96

#: One-sided confidence carried by the bound on the dispersion. Matched to the
#: coverage factor's 95% because a floor that is conservative about the spread
#: and lax about how well the spread is known is only conservative on paper.
DEFAULT_CONFIDENCE = 0.95

#: How a dispersion was estimated. The two are not interchangeable — one
#: resamples the units a reading scored and the other re-runs the whole
#: measurement — so a stored dispersion names the one behind it.
BOOTSTRAP_METHOD = "bootstrap-over-games"
PROCESS_REPLICATE_METHOD = "repeated-process-replicates"


class NoiseCharacterizationError(ValueError):
    """Raised when a dispersion cannot be estimated or recorded."""


def dispersion_bound(
    dispersion: float,
    *,
    degrees_of_freedom: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    """Return a conservative upper confidence limit on an estimated dispersion.

    A sample standard deviation ``s`` measured over ``degrees_of_freedom + 1``
    independent replicates of a normal quantity satisfies
    ``degrees_of_freedom * s**2 / sigma**2 ~ chi2(degrees_of_freedom)``, which
    inverts into an upper limit on the true ``sigma``. The limit is what a
    floor should be built from, because the estimate itself is below the truth
    about half the time and a floor is only useful when it errs the other way.

    The limit is punishing at small replicate counts, and honestly so: two
    replicates say almost nothing about a spread, and the arithmetic here is
    what makes that visible instead of letting a confident-looking floor come
    out of a measurement that could not support one. Adding replicates is the
    only way to narrow it.

    A dispersion of exactly zero is refused, because the limit is a multiple of
    what it bounds and cannot rescue one. An estimator whose replicates all
    agreed observed that it could not move this quantity, which is not the same
    as observing that nothing could, and the zero floor that would follow clears
    every delta. Deciding what to do about that is the estimator's — a caller
    with an answer withholds the quantity before reaching here, and a reading
    whose replicates genuinely cannot differ states its zero rather than
    estimating one.
    """

    if dispersion < 0.0 or not math.isfinite(dispersion):
        raise NoiseCharacterizationError(
            "a dispersion must be a finite, non-negative number"
        )
    if dispersion == 0.0:
        raise NoiseCharacterizationError(
            "a dispersion of exactly zero has no bound to compute; a floor of "
            "zero would clear every delta, so an estimator that could not move "
            "a quantity withholds it and a reading that cannot vary states its "
            "zero instead"
        )
    if degrees_of_freedom < 1:
        raise NoiseCharacterizationError(
            "bounding a dispersion needs at least one degree of freedom; a "
            "single replicate observes no spread to bound"
        )
    if not 0.0 < confidence < 1.0:
        raise NoiseCharacterizationError(
            "the confidence a dispersion bound carries must lie strictly "
            "between zero and one"
        )
    quantile = _chi_squared_quantile(1.0 - confidence, degrees_of_freedom)
    return dispersion * math.sqrt(degrees_of_freedom / quantile)


def plug_in_rescale(units: int) -> float:
    """Return what a draw from the units in hand understates a fresh draw by.

    A plug-in bootstrap resamples the units it holds, so its variance is
    ``(units - 1) / units`` of the variance a fresh sample of that size has,
    before it has estimated anything. Negligible wherever the unit count is
    large and worth 22% at three; decision 0039 named the correction and
    requires it of any draw small enough to notice.

    Exact for a mean, and first-order for the smooth reductions the estimators
    here take. It is the scalar form of the same correction a rescaled
    ``m``-out-of-``n`` draw applies to the counts.
    """

    if units < 2:
        raise NoiseCharacterizationError(
            "a plug-in correction needs at least two units; one unit has no "
            "spread for a factor to recover"
        )
    return math.sqrt(units / (units - 1))


def bounded_floor(
    dispersion: float,
    *,
    degrees_of_freedom: int,
    coverage: float = DEFAULT_COVERAGE,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    """Return the floor a spread of replicate draws of one quantity implies.

    The ``sqrt(2)`` is exact here rather than assumed: the two readings whose
    difference this floors are replicate draws of one quantity, so they share a
    dispersion by construction. A delta between two independent readings does
    not, and :func:`combined_floor` is what floors that.
    """

    if coverage <= 0.0 or not math.isfinite(coverage):
        raise NoiseCharacterizationError("coverage must be a finite, positive factor")
    return (
        coverage
        * math.sqrt(2.0)
        * dispersion_bound(
            dispersion,
            degrees_of_freedom=degrees_of_freedom,
            confidence=confidence,
        )
    )


def measured_dispersion(
    dispersion: float,
    *,
    degrees_of_freedom: int,
    confidence: float = DEFAULT_CONFIDENCE,
    units: int | None = None,
    source: str | None = None,
    estimator: str | None = None,
) -> MetricDispersion:
    """Return the record a reading stores for a dispersion it estimated."""

    return MetricDispersion(
        value=dispersion,
        bound=dispersion_bound(
            dispersion,
            degrees_of_freedom=degrees_of_freedom,
            confidence=confidence,
        ),
        units=units,
        source=source,
        estimator=estimator,
    )


def combined_floor(
    baseline: MetricDispersion,
    current: MetricDispersion,
) -> float:
    """Return the delta noise alone produces between two readings.

    The variance of a difference is the sum of the two variances, so the two
    readings' bounded dispersions combine in quadrature. The coverage is applied
    here rather than stored on either reading because a floor is a claim the
    comparison makes; the readings only say how far their own units move, and
    one factor for every comparison is what keeps two floors comparable.

    Resting on two bounds does not weaken the confidence either one carries.
    That would follow if the floor needed both to hold, but it needs only their
    combination to: a bound that overshoots covers for one that undershoots, and
    in quadrature it usually does. Coverage is exactly the declared confidence
    at the degenerate end, where one reading's variance is the whole pair, and
    higher wherever both readings contribute.
    """

    return DEFAULT_COVERAGE * math.hypot(baseline.bound, current.bound)


def self_combined_floor(dispersion: MetricDispersion) -> float:
    """Return the floor a delta against a reading like this one would face.

    What one reading resolves on its own is not a fact about any delta, since
    the other operand's spread is unknown until there is one. This is the
    stand-in for a display holding a single reading: the floor if the other
    reading turned out to match this one, which is also what a comparison
    between two readings of one benchmark at one size will land near.
    """

    return combined_floor(dispersion, dispersion)


def bounded_spread(
    dispersion: float,
    *,
    degrees_of_freedom: int,
    coverage: float = DEFAULT_COVERAGE,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    """Return the covered bound on one quantity's own spread.

    The same coverage and confidence as a floor, without the ``sqrt(2)``: this
    describes one quantity rather than the difference between two independent
    replicates of it. An estimator that resamples a difference directly, or
    that qualifies a single reading's own number, wants this one.
    """

    if coverage <= 0.0 or not math.isfinite(coverage):
        raise NoiseCharacterizationError("coverage must be a finite, positive factor")
    return coverage * dispersion_bound(
        dispersion,
        degrees_of_freedom=degrees_of_freedom,
        confidence=confidence,
    )


#: Convergence controls for the incomplete gamma function the chi-squared
#: quantile inverts. The project depends on neither SciPy nor NumPy in this
#: layer — the results package is importable from a bare install — so the
#: quantile is computed here rather than imported.
_GAMMA_ITERATIONS = 300
_GAMMA_EPSILON = 1e-14
_BISECTION_ITERATIONS = 200
_TINY = 1e-300


def _chi_squared_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Return the chi-squared value with ``probability`` mass below it.

    Bisected on the regularized lower incomplete gamma rather than solved in
    closed form, because there is no closed form. The result is only ever read
    at a handful of degrees of freedom per reading, so the cost of bisecting to
    machine precision is irrelevant next to being right at the small counts
    where the bound matters most.
    """

    shape = degrees_of_freedom / 2.0
    high = max(1.0, shape)
    for _ in range(_BISECTION_ITERATIONS):
        if _regularized_lower_gamma(shape, high) >= probability:
            break
        high *= 2.0
    else:  # pragma: no cover - unreachable for a probability below one
        raise NoiseCharacterizationError(
            "the chi-squared quantile could not be bracketed"
        )
    low = 0.0
    for _ in range(_BISECTION_ITERATIONS):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _regularized_lower_gamma(shape, middle) < probability:
            low = middle
        else:
            high = middle
    return 2.0 * high


def _regularized_lower_gamma(shape: float, x: float) -> float:
    """Return ``P(shape, x)``, the regularized lower incomplete gamma."""

    if x <= 0.0:
        return 0.0
    scale = math.exp(-x + shape * math.log(x) - math.lgamma(shape))
    if x < shape + 1.0:
        return _gamma_series(shape, x) * scale
    return 1.0 - _gamma_continued_fraction(shape, x) * scale


def _gamma_series(shape: float, x: float) -> float:
    """Return the series expansion that converges fastest below the mode."""

    term = 1.0 / shape
    total = term
    denominator = shape
    for _ in range(_GAMMA_ITERATIONS):
        denominator += 1.0
        term *= x / denominator
        total += term
        if abs(term) < abs(total) * _GAMMA_EPSILON:
            break
    return total


def _gamma_continued_fraction(shape: float, x: float) -> float:
    """Return the continued fraction that converges fastest above the mode."""

    b = x + 1.0 - shape
    c = 1.0 / _TINY
    d = 1.0 / b if abs(b) > _TINY else 1.0 / _TINY
    h = d
    for index in range(1, _GAMMA_ITERATIONS + 1):
        numerator = -index * (index - shape)
        b += 2.0
        d = numerator * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _GAMMA_EPSILON:
            break
    return h


def replicate_dispersion(values: Sequence[float]) -> float:
    """Return the dispersion across independent replicate measurements."""

    if len(values) < 2:
        raise NoiseCharacterizationError(
            "a dispersion needs at least two independent replicates"
        )
    if any(not math.isfinite(value) for value in values):
        raise NoiseCharacterizationError("replicate values must be finite")
    return statistics.stdev(values)


def process_dispersion(values: Sequence[float]) -> float:
    """Return the dispersion of one metric across separate processes.

    One value per process, because the process is the independent replicate: a
    second reading inside one process shares an allocator, a warm file cache and
    a compiled kernel with the first, so it cannot observe the component most
    likely to dominate. At least two are required for the same reason.
    """

    if len(values) < 2:
        raise NoiseCharacterizationError(
            "an execution dispersion needs readings from at least two "
            "processes; repeating inside one process cannot see process-level "
            "variation"
        )
    return replicate_dispersion(values)


def games_to_resolve(dispersion: MetricDispersion, *, effect: float) -> int:
    """Return how many games are needed to resolve an effect of a given size.

    A sampling dispersion shrinks with the square root of the games behind it,
    so the count an axis needs is computable from one reading rather than
    guessed. This is what sizes a pool generation.

    The count is in the games the dispersion was read over, which for a sliced
    metric are the ones that realized the slice rather than games in the pool;
    converting to a pool size takes a realization rate nothing here applies.

    The floor it extrapolates from is the one a delta against a reading like
    this one would face, since that is the comparison a larger pool is being cut
    for.

    The answer errs high, and deliberately. The bound behind that floor is sized
    for the degrees of freedom available now, and a larger pool would carry more
    of those and a tighter bound, so the projected floor is the widest one the
    larger pool could produce rather than the one it will. Erring the other way
    would size a pool that turns out not to resolve the effect it was cut for.
    """

    if dispersion.units is None:
        raise NoiseCharacterizationError(
            "this spread does not scale with a game count, so it cannot size "
            "an evaluation input"
        )
    if effect <= 0.0 or not math.isfinite(effect):
        raise NoiseCharacterizationError("an effect size must be finite and positive")
    floor = self_combined_floor(dispersion)
    if floor == 0.0:
        return 1
    return max(1, math.ceil(dispersion.units * (floor / effect) ** 2))


__all__ = [
    "BOOTSTRAP_METHOD",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_COVERAGE",
    "PROCESS_REPLICATE_METHOD",
    "NoiseCharacterizationError",
    "bounded_floor",
    "bounded_spread",
    "combined_floor",
    "dispersion_bound",
    "games_to_resolve",
    "measured_dispersion",
    "plug_in_rescale",
    "process_dispersion",
    "replicate_dispersion",
    "self_combined_floor",
]
