"""Characterized noise floors, and the rule for reading a delta against one.

A delta is not a finding until it is larger than the noise in the measurement.
Without a floor, a report can only say that a number moved, and a reader
comparing two checkpoints will confidently describe movement that is seed luck.

Four noise sources are kept apart, because conflating them is the usual
mistake. They are estimated differently and they answer different questions,
but they all reduce to the same reportable quantity: the **dispersion** of a
metric across independent replicates of one noise source.

A floor is that dispersion expressed as a delta, at a declared normal coverage.
The result is directly comparable to a reported delta, which is what a report
needs and what a raw standard deviation is not.

The variance of a difference is the sum of the two variances, so a delta
between two readings is floored by combining the dispersion each one carries:
:func:`combined_floor`. :func:`bounded_floor` is the same arithmetic where both
sides are one characterization's replicates and share a dispersion by
construction, which is what makes its ``sqrt(2)`` correct there and an
assumption anywhere else.

The dispersion a floor is built from is never the measured one. A dispersion
read off a handful of replicates is a point estimate sitting in the middle of
its own sampling distribution, so about half the time it lands below the spread
it is supposed to describe, and a floor built on it is too narrow exactly when
being too narrow matters. A floor that understates the noise licenses noise as
a finding, which is the failure floors exist to prevent, so this module builds
every floor from a **conservative upper confidence limit** on the dispersion
instead — the chi-squared bound for the degrees of freedom actually behind the
estimate.

That makes a floor a tolerance bound rather than an interval, and the two
factors it carries answer two different questions. ``coverage`` says what
proportion of same-weights deltas the floor covers when the dispersion is
known. ``confidence`` says how sure the bound is that the dispersion is not
larger than assumed. They multiply, and the resulting claim is: with
``confidence``, this floor covers ``coverage`` of the deltas that noise alone
produces. Widening either one widens the floor; only more replicates narrow it
honestly.

Floors are stored beside the results they qualify, keyed by the same series
fingerprint as any other measurement. That is deliberate: when the pool, the
view, or a metric definition moves, the floor stops matching and the report
says the floor is unknown, rather than silently applying a stale constant to a
measurement it no longer describes.

Two kinds are keyed by one thing more, for the same reason in two places.
Decision 0018 keeps the machine out of an efficiency series on purpose, so that
a latency history stays continuous across a hardware change; but the noise in a
timing measurement *is* the machine, so a floor measured on a laptop describes
nothing about a reading taken on a workstation. Decisions 0018 and 0021 keep the
training run out of series identity for the same reason, and the noise in a
training measurement *is* the configuration, so a floor characterized from one
configuration's seed replicates describes nothing about a delta between models
of another size, corpus, or arithmetic. Such a floor therefore carries the scope
it was characterized under, and a report applies it only where that scope
describes the delta being judged. What that takes differs by kind, and
:func:`in_scope` owns the difference.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from pydantic import Field, model_validator

from anthro_chess.evaluation.results.comparability import BridgeIndex
from anthro_chess.evaluation.results.records import (
    MAXIMUM_SUMMARY_BYTES,
    EnvironmentRecord,
    ExecutionRecord,
    Identifier,
    MetricDispersion,
    NoiseFloor,
    NoiseFloorKind,
    ResultEnvelope,
    ResultModel,
    Sha256Hex,
    canonical_json,
)

#: Version 5 dropped the sampled-unit count, which only a data-sampling floor
#: carried and which now travels beside the dispersion on the reading itself.
#: Version 4 added the training scope a training floor is only valid within.
#: Version 3 replaced the point-estimate dispersion in a floor with a
#: conservative upper bound, and records the confidence that bound carries.
#: Version 2 added the execution scope an execution floor is only valid within.
CHARACTERIZATION_VERSION = 5

#: Two-sided normal coverage for a 95% interval. A floor is a claim about how
#: far apart two measurements land when nothing changed, so it needs a stated
#: confidence rather than a bare standard deviation.
DEFAULT_COVERAGE = 1.96

#: One-sided confidence carried by the bound on the dispersion. Matched to the
#: coverage factor's 95% because a floor that is conservative about the spread
#: and lax about how well the spread is known is only conservative on paper.
DEFAULT_CONFIDENCE = 0.95

#: How a dispersion was estimated. The method is recorded because the three
#: noise kinds are not interchangeable and neither are their estimators.
BOOTSTRAP_METHOD = "bootstrap-over-games"
REPLICATE_METHOD = "independent-replicates"
PROCESS_REPLICATE_METHOD = "repeated-process-replicates"

#: How a delta floor built from the two readings in front of it was arrived at.
#: It fills the same field an estimator's name does while estimating nothing:
#: both readings already measured their spread, and this is only the arithmetic
#: that turns the pair into a floor.
COMBINED_DISPERSION_METHOD = "combined-reading-dispersions"


class NoiseCharacterizationError(ValueError):
    """Raised when a noise floor cannot be estimated or recorded."""


class FloorEntry(ResultModel):
    """One metric's floor, on one series."""

    metric: str = Field(min_length=1)
    fingerprint: Sha256Hex
    floor: float = Field(ge=0.0)
    #: The measured spread. Kept beside the bound because the two answer
    #: different questions: this one describes the machine or the sample as it
    #: was observed, and is what a later characterization is compared against.
    dispersion: float = Field(ge=0.0)
    #: The conservative upper limit on ``dispersion`` that ``floor`` was
    #: actually built from. Storing it rather than only the floor keeps how
    #: much of a wide floor is spread and how much is ignorance visible: a
    #: bound far above the dispersion means the estimate is thin, which more
    #: replicates fix, while the two close together means the noise is real.
    dispersion_bound: float = Field(ge=0.0)
    #: Independent replicates behind ``dispersion``, less one. This is what
    #: sets how far the bound sits above the estimate, and it counts genuinely
    #: independent units rather than numbers: bootstrap resamples of one sample
    #: and repeated readings inside one process are not independent replicates,
    #: and counting them here would restore the false precision the bound
    #: exists to remove.
    degrees_of_freedom: int = Field(ge=1)
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
        if (
            not math.isfinite(self.floor)
            or not math.isfinite(self.dispersion)
            or not math.isfinite(self.dispersion_bound)
        ):
            raise ValueError(f"floor for {self.metric} must be a finite number")
        if self.within_process_dispersion is not None and not math.isfinite(
            self.within_process_dispersion
        ):
            raise ValueError(f"floor for {self.metric} must be a finite number")
        if self.dispersion_bound < self.dispersion:
            raise ValueError(
                f"the bound for {self.metric} is below the dispersion it "
                "bounds, so it is not a conservative limit"
            )
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
    #: One-sided confidence the dispersion bound behind every floor carries.
    #: Declared once per pass, like coverage, because a characterization that
    #: mixed confidences across its metrics would have no single meaning.
    confidence: float = Field(gt=0.0, lt=1.0)
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
    #: The training configuration a training floor describes, as the digest a
    #: result records for the checkpoint it scored. Present only for that kind,
    #: on the same terms as ``execution``: seed variance is a property of the
    #: configuration that was trained, and every other kind is measured on
    #: whatever weights it was handed.
    training: Sha256Hex | None = None
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
    def _validate_scope(self) -> NoiseCharacterization:
        """Keep the scope a floor carries and the noise kind agreed.

        An execution floor without its execution would be applied to every
        machine that ever recorded the series, and a training floor without its
        training identity to every configuration that ever recorded it, which is
        the one thing either kind must not do. A scope on a kind that does not
        depend on it would narrow a floor that is valid wherever its series is.
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
        if self.kind == "training":
            if self.training is None:
                raise ValueError(
                    "a training floor is a property of the configuration its "
                    "replicates shared, so it must record the training identity "
                    "it was characterized under"
                )
        elif self.training is not None:
            raise ValueError(
                f"a {self.kind} floor does not depend on what was trained, so "
                "it carries no training scope"
            )
        return self

    def scope_key(self) -> str:
        """Return the scope this characterization is valid within.

        The empty string means no scope, and is what every kind other than
        ``execution`` and ``training`` carries; a floor keyed on it resolves for
        every reading of its series.
        """

        if self.kind == "training":
            return self.training or ""
        return environment_key(self.execution)

    def entry(self, metric: str) -> FloorEntry | None:
        """Return one metric's floor, if this characterization covers it."""

        for candidate in self.floors:
            if candidate.metric == metric:
                return candidate
        return None

    def as_floor(self, entry: FloorEntry) -> NoiseFloor:
        """Return the floor in the shape a measurement and a report read."""

        return NoiseFloor(
            value=entry.floor,
            kind=self.kind,
            source=self.source,
            estimator=self.method,
        )

    def verify(self) -> None:
        """Reject a record too large for the committed summary tier."""

        size = len(canonical_json(self.as_record()))
        if size > MAXIMUM_SUMMARY_BYTES:
            raise NoiseCharacterizationError(
                f"noise characterization {self.characterization_id} is {size} "
                f"bytes; the committed summary tier caps a record at "
                f"{MAXIMUM_SUMMARY_BYTES}"
            )


class NoiseFloorIndex:
    """Resolve the floors that apply to a series.

    Floors are looked up through the same bridge equivalence as measurements,
    so a series that was legitimately rejoined keeps the floor characterized
    on either side of the seam. A fingerprint with no characterization
    resolves to nothing at all rather than to zero.

    A scoped floor additionally has to describe the delta it would qualify,
    which :func:`in_scope` decides and the two scoped kinds answer differently.
    A floor that describes neither operand resolves to nothing rather than to a
    borrowed one.
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
                    characterization.scope_key(),
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
        trainings: Sequence[str | None] = (),
    ) -> tuple[NoiseFloor, ...]:
        """Return every characterized floor for one metric's series, by kind.

        ``executions`` and ``trainings`` are the scopes the floors would
        qualify — both operands of a delta, or the single reading being
        annotated. A reading that records no scope of a kind carries the empty
        scope, which no characterization is stored under, so it never resolves
        a floor of that kind on its own.
        """

        series = self._bridges.series(fingerprint)
        measured = {
            "execution": {environment_key(execution) for execution in executions},
            "training": {training or "" for training in trainings},
        }
        # ``measured`` covers only the scoped kinds; the ``not key[3]`` test is
        # what keeps an unscoped kind from reaching the lookup.
        return tuple(
            self._by_series[key][1]
            for key in sorted(self._by_series)
            if key[0] == series
            and key[1] == metric
            and (not key[3] or in_scope(key[2], key[3], measured[key[2]]))
        )


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
    """

    if dispersion < 0.0 or not math.isfinite(dispersion):
        raise NoiseCharacterizationError(
            "a dispersion must be a finite, non-negative number"
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


def bounded_floor(
    dispersion: float,
    *,
    degrees_of_freedom: int,
    coverage: float = DEFAULT_COVERAGE,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Return the bound a floor was built from, and the floor.

    Every characterization records both, so returning them together keeps the
    stored bound and the stored floor from drifting apart through two call sites
    that could disagree about which arguments produced which.

    The ``sqrt(2)`` is exact here rather than assumed: a characterization's
    replicates are draws of one quantity, so the two readings whose difference
    this floors share a dispersion by construction. A delta between two
    independent readings does not, and :func:`combined_floor` is what floors
    that.
    """

    if coverage <= 0.0 or not math.isfinite(coverage):
        raise NoiseCharacterizationError("coverage must be a finite, positive factor")
    bound = dispersion_bound(
        dispersion,
        degrees_of_freedom=degrees_of_freedom,
        confidence=confidence,
    )
    return bound, coverage * math.sqrt(2.0) * bound


def measured_dispersion(
    dispersion: float,
    *,
    kind: NoiseFloorKind,
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
        kind=kind,
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
    at a handful of degrees of freedom per characterization, so the cost of
    bisecting to machine precision is irrelevant next to being right at the
    small counts where the bound matters most.
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


def in_scope(kind: str, scope: str, measured: frozenset[str] | set[str]) -> bool:
    """Return whether a scoped floor describes the delta ``measured`` spans.

    The two scoped kinds answer this differently, because their scopes are
    different sorts of thing.

    An **execution** scope is a condition both readings were taken under, so a
    delta spanning two machines is described by neither machine's floor and is
    reported as unknown. Decision 0025 owns that rule.

    A **training** scope names the configuration whose seed spread the floor
    measured, and that spread is a null distribution rather than a condition:
    the question a delta asks is whether its other operand falls outside what a
    different seed of this configuration would have produced. So one operand
    carrying the characterized configuration is what makes the floor apply.
    Requiring both would refuse exactly the control-arm comparison decision 0029
    defines, since the change under test is what makes the two identities
    differ. Where both operands carry characterized configurations, both floors
    resolve and the widest binds, as it does everywhere else.
    """

    if kind == "training":
        return scope in measured
    return measured == {scope}


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
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[FloorEntry, ...]:
    """Return one execution floor per metric from its repeated measurements.

    Each metric arrives as one sequence of readings per process. Every
    replicate of an execution floor is the same series by construction — the
    same checkpoint under the same declared workload — so the fingerprint is
    supplied once per metric rather than carried on each reading.

    The **process** is the independent replicate here, so the degrees of
    freedom behind the bound come from the process count and not from the
    reading count. Two readings inside one process share an allocator, a warm
    file cache and a compiled kernel; they widen the dispersion the floor is
    built from, which is why they are taken, but treating them as independent
    would claim the estimate is firmer than the design can support. That is the
    same reason the within-process share is reported beside the floor rather
    than folded into it.
    """

    entries: list[FloorEntry] = []
    for metric in sorted(replicates):
        fingerprint = fingerprints.get(metric)
        if fingerprint is None:
            raise NoiseCharacterizationError(
                f"no series fingerprint was supplied for {metric}"
            )
        groups = replicates[metric]
        dispersion = process_dispersion(groups)
        freedom = len(groups) - 1
        bound, floor = bounded_floor(
            dispersion.total,
            degrees_of_freedom=freedom,
            coverage=coverage,
            confidence=confidence,
        )
        entries.append(
            FloorEntry(
                metric=metric,
                fingerprint=fingerprint,
                floor=floor,
                dispersion=dispersion.total,
                dispersion_bound=bound,
                degrees_of_freedom=freedom,
                within_process_dispersion=dispersion.within,
            )
        )
    return tuple(entries)


def games_to_resolve(dispersion: MetricDispersion, *, effect: float) -> int:
    """Return how many games are needed to resolve an effect of a given size.

    A sampling dispersion shrinks with the square root of the games behind it,
    so the count an axis needs is computable from one reading rather than
    guessed. This is what sizes a pool generation.

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


def build_characterization(
    *,
    kind: NoiseFloorKind,
    method: str,
    replicates: int,
    source: str,
    floors: Sequence[FloorEntry],
    coverage: float = DEFAULT_COVERAGE,
    confidence: float = DEFAULT_CONFIDENCE,
    environment: EnvironmentRecord | None = None,
    execution: ExecutionRecord | None = None,
    training: str | None = None,
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
            confidence=confidence,
            source=source,
            environment=environment or EnvironmentRecord.capture(),
            processes=processes,
            execution=execution,
            training=training,
            floors=ordered,
        )
        payload = record.as_record()
        payload.pop("characterization_id", None)
        identified = record.model_copy(
            update={
                "characterization_id": sha256(canonical_json(payload)).hexdigest()[:16],
            }
        )
    except ValueError as error:
        raise NoiseCharacterizationError(str(error)) from error
    identified.verify()
    return identified


def training_scope(readings: Sequence[ResultEnvelope]) -> str:
    """Return the training configuration every replicate reading was trained under.

    Reading the scope off the replicates is what makes it a fact about them
    rather than a caller's assertion.
    """

    identities = {reading.checkpoint.training_sha256 for reading in readings}
    named = [identity for identity in identities if identity is not None]
    if len(named) != len(identities):
        raise NoiseCharacterizationError(
            "a training floor is scoped to the configuration its replicates "
            "shared, and at least one named checkpoint records no training "
            "identity; re-record those readings before characterizing"
        )
    if len(named) != 1:
        raise NoiseCharacterizationError(
            "the named checkpoints were trained under different configurations, "
            "so their spread is not one configuration's seed variance"
        )
    return named[0]


def replicate_floors(
    values_by_metric: Mapping[str, Sequence[tuple[str, float]]],
    *,
    coverage: float = DEFAULT_COVERAGE,
    confidence: float = DEFAULT_CONFIDENCE,
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
        freedom = len(replicates) - 1
        bound, floor = bounded_floor(
            dispersion,
            degrees_of_freedom=freedom,
            coverage=coverage,
            confidence=confidence,
        )
        entries.append(
            FloorEntry(
                metric=metric,
                fingerprint=replicates[-1][0],
                floor=floor,
                dispersion=dispersion,
                dispersion_bound=bound,
                degrees_of_freedom=freedom,
            )
        )
    return tuple(entries)


__all__ = [
    "BOOTSTRAP_METHOD",
    "CHARACTERIZATION_VERSION",
    "COMBINED_DISPERSION_METHOD",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_COVERAGE",
    "PROCESS_REPLICATE_METHOD",
    "REPLICATE_METHOD",
    "FloorEntry",
    "NoiseCharacterization",
    "NoiseCharacterizationError",
    "NoiseFloorIndex",
    "ProcessDispersion",
    "bounded_floor",
    "bounded_spread",
    "build_characterization",
    "combined_floor",
    "dispersion_bound",
    "environment_key",
    "games_to_resolve",
    "measured_dispersion",
    "in_scope",
    "process_dispersion",
    "process_replicate_floors",
    "replicate_dispersion",
    "replicate_floors",
    "self_combined_floor",
    "training_scope",
]
