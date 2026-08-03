"""What the configured target rating buys, and what temperature costs it.

The configured target rating is the project's central dial, and nothing so far
says where it lands. Held-out prediction measures how well the model explains
human moves at a stated rating; it cannot say whether a seat configured at 1800
beats one configured at 1200. That is a question about play, so it is answered
by playing: a round robin among configured seats, reduced to an empirical rating
for each through a Bradley-Terry fit.

This is a measurement rather than a calibration. The deliverable is the transfer
function from configured to fitted rating — its ordering, its slope, and where
it stops resolving — not a dial tuned until the two agree. A slope below one
usually points at uneven rating coverage in the training data, whose expected
response is better data, weighting, or capacity, evaluated through these same
surfaces.

**One joint fit, not one per row.** Temperature is a second axis, and the
question asked of it is what it costs in strength. A ladder fitted separately at
each temperature could not answer that: a fit is invariant to a shift of every
rating in it, so two independent fits share no scale and their difference means
nothing. So a seat is a *(conditioning, temperature)* pair, every seat plays
every other, and one fit places them all on one internal scale. The rating
response is then read along one axis of that surface and the temperature
response along the other.

**Ablation is absence.** The control arm for the temperature response is the
same model with rating conditioning removed, which the runtime already supports
as an absent target rating. Ablated seats join the same round robin rather than
forming a second ladder, for the same reason the temperatures do: an attenuation
computed between two unrelated scales would be arithmetic rather than a
measurement. Absence is the treatment the dependency tests call ``absent``, and
it carries that form's caveat — a corpus that never contained rating-absent
positions makes this partly a reading about input presence rather than input
value.

**A degenerate fit is a result.** A checkpoint whose configured ratings are
indistinguishable produces a flat ladder, and one whose seats never lose
produces a fit that runs away rather than converging. Both are reported, with
the convergence state and the clamped seats named, because the reading is
establishing the instrument and the baseline. Neither is an error and neither is
a calibration verdict.

Nothing here waits in wall-clock time. Every game comes from the shared
generation harness under an explicit seed, so a suite reproduces exactly and a
single game reproduces on its own from the seed its record carries.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import torch
from pydantic import Field, StrictBool, StrictInt, model_validator

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data.artifacts import DataLoadingError, read_normalized_rows
from anthro_chess.data.schema import NormalizedColumn
from anthro_chess.evaluation.decisions import (
    DecisionCell,
    DecisionDecompositionError,
    DecisionSample,
    DecisionSet,
    DecisionSetting,
    collect_decisions,
    summarize_decisions,
)
from anthro_chess.evaluation.dependency import ConditioningKind
from anthro_chess.evaluation.execution import execution_record
from anthro_chess.evaluation.games import (
    GENERATION_VERSION,
    GameRecord,
    GenerationConfig,
    GenerationError,
    ModelPlayer,
    PlayerError,
    StartPosition,
    generate_games,
    prefix_positions,
    standard_positions,
)
from anthro_chess.evaluation.pool import EvaluationPoolError, FrozenPool, load_pool
from anthro_chess.evaluation.recording import (
    ResultRecorder,
    pool_dataset_reference,
    resolve_model,
    runner_device,
)
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DatasetReference,
    DetailStore,
    ExecutionRecord,
    Measurement,
    ResultEnvelope,
    ResultsStore,
    measurement,
)
from anthro_chess.evaluation.results.fingerprints import (
    FingerprintError,
    projection_content_digest,
)
from anthro_chess.evaluation.results.metrics import (
    LADDER_ABLATED_TEMPERATURE_RESPONSE,
    LADDER_ADJACENT_RATING_ORDER_ACCURACY,
    LADDER_DEPARTURE_POLICY_REGRET,
    LADDER_FITTED_RATING,
    LADDER_FITTED_RATING_SLOPE,
    LADDER_FITTED_RATING_SPAN,
    LADDER_POLICY_REGRET,
    LADDER_PREFERRED_SELECTION_RATE,
    LADDER_RATING_ERROR,
    LADDER_RATING_ORDER_ACCURACY,
    LADDER_SCORE_RATE,
    LADDER_SELECTED_RANK,
    LADDER_TEMPERATURE_RESPONSE,
    LADDER_TEMPERATURE_RESPONSE_ATTENUATION,
    MOVE_PREDICTION_PROJECTION,
)
from anthro_chess.evaluation.views import ViewConfig, ViewSelection, apply_view
from anthro_chess.inference import ModelRunnerConfig
from anthro_chess.runtime import ActionModelRunner, RuntimeConfig

LADDER_BENCHMARK_VERSION = 1

LADDER_KIND = "rating-ladder"
LADDER_BENCHMARK = BenchmarkReference(
    name="rating-ladder",
    version=LADDER_BENCHMARK_VERSION,
)

#: Rating points per factor-of-ten odds. The conventional Elo scale, declared
#: here because it is what makes a fitted number readable as a rating at all.
RATING_SCALE = 400.0

#: How the fit is identified in a declared workload. A different pairing model
#: is a different quantity even over identical games.
FIT_MODEL = "bradley-terry"

logger = logging.getLogger(__name__)


class LadderBenchmarkError(ValueError):
    """Raised when a rating ladder cannot be measured as configured."""


class SeatConditioning(StrEnum):
    """How one seat's rating input was supplied.

    Two values rather than a boolean because the ablated treatment is a named
    conditioning form rather than the absence of a setting: it is the
    dependency tests' ``absent`` treatment applied to whole games.
    """

    #: The configured target rating is supplied, which is ordinary play.
    CONDITIONED = "conditioned"
    #: No target rating is supplied at all, which is the control arm.
    ABLATED = "ablated"


#: What each seat's conditioning is called in the dependency tests' vocabulary.
#: The ladder ablates by withholding the rating entirely, which is exactly that
#: layer's ``absent`` treatment applied to whole games rather than to scored
#: positions, so the two families name one treatment rather than two.
CONDITIONING_TREATMENTS: Mapping[SeatConditioning, ConditioningKind] = {
    SeatConditioning.CONDITIONED: ConditioningKind.TRUE,
    SeatConditioning.ABLATED: ConditioningKind.ABSENT,
}


@dataclass(frozen=True)
class SeatKey:
    """One competitor of the ladder: a conditioning and a temperature.

    The unit is deliberately not the configured rating. A seat playing at
    temperature zero and the same rating playing at one are two different
    opponents, and treating them as one competitor would average away the
    quantity this benchmark exists to report.
    """

    conditioning: SeatConditioning
    #: ``None`` on an ablated seat, which has no configured rating by
    #: construction rather than by omission.
    target_rating: int | None
    temperature: float

    @property
    def label(self) -> str:
        """Return the short human label a report and a seat record print."""

        rating = "ablated" if self.target_rating is None else str(self.target_rating)
        return f"{rating}@t{self.temperature:g}"

    @property
    def setting(self) -> DecisionSetting:
        """Return the decomposition cell this seat's decisions land in."""

        return DecisionSetting(
            target_rating=self.target_rating,
            temperature=self.temperature,
        )

    @property
    def sort_key(self) -> tuple[str, bool, int, float]:
        """Return a total order over seats, ablated ones last.

        Spelled out rather than derived from the field order, because an
        ablated seat carries no configured rating and comparing ``None``
        against an integer is not an ordering at all.
        """

        return (
            self.conditioning.value,
            self.target_rating is None,
            self.target_rating or 0,
            self.temperature,
        )

    def as_record(self) -> dict[str, Any]:
        """Return the stable coordinates stored with every seat reading."""

        return {
            "conditioning": self.conditioning.value,
            "conditioning_kind": CONDITIONING_TREATMENTS[self.conditioning].value,
            "target_rating": self.target_rating,
            "temperature": self.temperature,
        }


class LadderGridConfig(ConfigModel):
    """The seats a ladder fields, before ablation adds its control arm."""

    #: Configured ratings to field. At least two, since a ladder of one has no
    #: ordering to report.
    target_ratings: tuple[StrictInt, ...] = (1200, 1500, 1800, 2100)
    #: Sampling temperatures to field each rating at. One temperature measures
    #: the rating transfer alone; two or more add the temperature response.
    temperatures: tuple[float, ...] = (1.0,)
    #: The temperature every calibration figure is declared against, and the
    #: row the joint fit is anchored on. Must be one of the grid's own.
    reference_temperature: float = Field(default=1.0, ge=0.0, le=3.0)
    #: Base seeds, each replaying the whole round robin. A sample size rather
    #: than a measurement setting.
    seeds: tuple[StrictInt, ...] = (0,)

    @model_validator(mode="after")
    def _validate_axes(self) -> LadderGridConfig:
        if len(self.target_ratings) < 2:
            raise ValueError("a rating ladder needs at least two configured ratings")
        if tuple(sorted(set(self.target_ratings))) != self.target_ratings:
            raise ValueError("target_ratings must be sorted and unique")
        if any(rating < 0 for rating in self.target_ratings):
            raise ValueError("a configured rating cannot be negative")
        if not self.temperatures:
            raise ValueError("a rating ladder needs at least one temperature")
        if tuple(sorted(set(self.temperatures))) != self.temperatures:
            raise ValueError("temperatures must be sorted and unique")
        if any(not 0.0 <= value <= 3.0 for value in self.temperatures):
            raise ValueError("a ladder temperature must be between zero and three")
        if self.reference_temperature not in self.temperatures:
            raise ValueError(
                "reference_temperature must be one of the grid's temperatures, "
                "since it is the row every calibration figure is declared "
                "against and the row the joint fit is anchored on"
            )
        if not self.seeds:
            raise ValueError("a rating ladder needs at least one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("a rating ladder must not repeat a seed")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("a ladder seed cannot be negative")
        return self


class LadderAblationConfig(ConfigModel):
    """The control arm the temperature response is read against."""

    #: One ablated seat per temperature, conditioned on no rating at all. Off
    #: leaves the conditioned response uncontrolled, which is a weaker reading
    #: rather than a cheaper one.
    enabled: StrictBool = True


class LadderOpeningsConfig(ConfigModel):
    """Where the ladder's games start.

    The standard position alone is a poor ladder at low temperature: two
    deterministic seats replay one game however many times they are asked, so a
    whole pairing collapses to two results. Frozen human openings restore the
    variety without making the measurement depend on the model's own sampling.
    """

    #: A frozen evaluation pool. Absent plays every game from the standard
    #: position, which is the right choice only when the temperatures are high
    #: enough for sampling to supply the variety itself.
    pool: Path | None = None
    view: ViewConfig = ViewConfig(name="ladder-openings", maximum_games=16)
    #: How many plies of each source game are replayed before the seats decide.
    plies: Annotated[StrictInt, Field(ge=1)] = 8


class LadderFitConfig(ConfigModel):
    """How the pairwise results are reduced to one rating per seat."""

    maximum_iterations: Annotated[StrictInt, Field(ge=1)] = 500
    #: Convergence threshold in rating points, so the stopping rule is stated
    #: in the units the result is read in.
    tolerance: Annotated[float, Field(gt=0.0)] = 1e-6
    #: How far a seat's fitted rating may sit from the anchor. A seat that won
    #: or lost every game has an unbounded maximum-likelihood rating, so the
    #: clamp is what turns a runaway fit into a reportable extreme. It joins
    #: series identity because it can change a reported value.
    maximum_spread: Annotated[float, Field(gt=0.0)] = 1200.0


class LadderDetailConfig(ConfigModel):
    """What the machine-local detail tier keeps beside a committed summary."""

    #: Whole game records. Large, and the input to any later analysis over the
    #: same games; a decision decomposition is computed during the run either
    #: way, so this is retained for re-analysis rather than for the reading.
    retain_games: StrictBool = True


class LadderBenchmarkConfig(ConfigModel):
    """Code-owned schema for ``anthro eval ladder``."""

    model: ModelRunnerConfig = ModelRunnerConfig()
    checkpoint_label: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    #: The base runtime settings every seat plays under. Rating, temperature,
    #: and seed are supplied per seat and per game, so setting them here would
    #: be overridden; the rest applies to the whole ladder.
    runtime: RuntimeConfig = RuntimeConfig()
    grid: LadderGridConfig = LadderGridConfig()
    generation: GenerationConfig = GenerationConfig()
    openings: LadderOpeningsConfig = LadderOpeningsConfig()
    ablation: LadderAblationConfig = LadderAblationConfig()
    fit: LadderFitConfig = LadderFitConfig()
    detail: LadderDetailConfig = LadderDetailConfig()


@dataclass(frozen=True)
class LadderPairing:
    """Every game two seats played against each other.

    Scored games only. A game that hit the ply limit has no result, so it
    informs no pairwise comparison; it is counted separately rather than
    adjudicated into a draw, which would report the ply limit as a level of
    play.
    """

    first: SeatKey
    second: SeatKey
    games: int
    first_points: float
    unfinished: int = 0

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one pairing."""

        return {
            "first": self.first.as_record(),
            "second": self.second.as_record(),
            "games": self.games,
            "first_points": self.first_points,
            "unfinished": self.unfinished,
        }


@dataclass(frozen=True)
class RatingFit:
    """One joint Bradley-Terry fit over every seat of the ladder."""

    ratings: dict[SeatKey, float]
    iterations: int
    converged: bool
    #: Seats whose maximum-likelihood rating is unbounded because they won or
    #: lost every game, and were therefore clamped. Named rather than hidden:
    #: their values are floor and ceiling readings, not estimates.
    clamped: tuple[SeatKey, ...]
    #: Seats no scored game reached, which the fit could not place at all.
    unscored: tuple[SeatKey, ...]
    anchor_rating: float
    anchor_basis: str

    def rating(self, seat: SeatKey) -> float | None:
        """Return one seat's fitted rating, or nothing when it has none."""

        return self.ratings.get(seat)

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of the fit and its convergence state."""

        return {
            "model": FIT_MODEL,
            "scale": RATING_SCALE,
            "iterations": self.iterations,
            "converged": self.converged,
            "anchor_rating": self.anchor_rating,
            "anchor_basis": self.anchor_basis,
            "clamped": [seat.as_record() for seat in self.clamped],
            "unscored": [seat.as_record() for seat in self.unscored],
        }


@dataclass(frozen=True)
class LadderSeat:
    """One competitor's play, and where the joint fit placed it."""

    key: SeatKey
    games: int
    points: float
    unfinished: int
    fitted_rating: float | None
    #: The error profile over this seat's own decisions, absent when the seat
    #: made none that could be classified.
    decisions: DecisionCell | None
    execution: ExecutionRecord

    @property
    def score_rate(self) -> float:
        """Return points per scored game, counting a draw as a half."""

        return self.points / self.games if self.games else 0.0

    @property
    def label(self) -> str:
        """Return the short human label a report prints."""

        return self.key.label

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one seat."""

        return {
            "seat": self.key.as_record(),
            "games": self.games,
            "points": self.points,
            "score_rate": self.score_rate,
            "unfinished": self.unfinished,
            "fitted_rating": self.fitted_rating,
            "workload_sha256": self.execution.workload_sha256,
            "workload": dict(self.execution.workload),
            "decisions": None if self.decisions is None else self.decisions.as_record(),
        }


@dataclass(frozen=True)
class LadderReading:
    """The rating transfer function read along one temperature row.

    One reading per temperature rather than one per seat, because ordering, a
    slope, and a ladder error are all statements about a whole row. The
    temperature is fixed across it because it is a separate dial rather than a
    point on this axis.
    """

    temperature: float
    ratings: tuple[int, ...]
    fitted: tuple[float, ...]
    order_accuracy: float
    adjacent_order_accuracy: float
    ladder_error: float
    slope: float
    span: float
    execution: ExecutionRecord

    @property
    def label(self) -> str:
        """Return the short human label a report prints."""

        return f"temperature={self.temperature:g}"

    @property
    def inversions(self) -> tuple[tuple[int, int], ...]:
        """Return the adjacent configured pairs the fit did not order.

        This is what localizes where the relationship degrades. A row can post
        a high pairwise accuracy while every neighbouring pair is
        indistinguishable, and only the pairs themselves say which end of the
        scale stopped resolving.
        """

        return tuple(
            (self.ratings[index], self.ratings[index + 1])
            for index in range(len(self.ratings) - 1)
            if self.fitted[index + 1] <= self.fitted[index]
        )

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one temperature row."""

        return {
            "temperature": self.temperature,
            "order_accuracy": self.order_accuracy,
            "adjacent_order_accuracy": self.adjacent_order_accuracy,
            "ladder_error": self.ladder_error,
            "slope": self.slope,
            "span": self.span,
            "inversions": [
                {"lower": lower, "upper": upper} for lower, upper in self.inversions
            ],
            "workload_sha256": self.execution.workload_sha256,
            "workload": dict(self.execution.workload),
            "seats": [
                {
                    "target_rating": rating,
                    "fitted_rating": fitted,
                    "error": fitted - rating,
                }
                for rating, fitted in zip(self.ratings, self.fitted, strict=True)
            ],
        }


@dataclass(frozen=True)
class TemperatureResponse:
    """What temperature costs in fitted rating, with and without conditioning.

    Both arms come out of one fit, which is the only reason their difference is
    a quantity at all: a Bradley-Terry fit is invariant to a shift of every
    rating in it, so two separate fits would share no origin and the ratio of
    their slopes would be arithmetic over unrelated scales.
    """

    temperatures: tuple[float, ...]
    #: One slope per configured rating, in fitted points per unit temperature.
    per_rating: tuple[tuple[int, float], ...]
    conditioned_response: float
    ablated_response: float | None
    #: One minus the ratio of the two responses, absent when the ablated arm
    #: did not run or read too close to zero to divide by.
    attenuation: float | None
    attenuation_unavailable: str | None
    execution: ExecutionRecord

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of the temperature response."""

        return {
            "temperatures": list(self.temperatures),
            "conditioned_response": self.conditioned_response,
            "ablated_response": self.ablated_response,
            "attenuation": self.attenuation,
            "attenuation_unavailable": self.attenuation_unavailable,
            "workload_sha256": self.execution.workload_sha256,
            "workload": dict(self.execution.workload),
            "per_rating": [
                {"target_rating": rating, "response": response}
                for rating, response in self.per_rating
            ],
        }


@dataclass(frozen=True)
class LadderBenchmarkResult:
    """Everything one ladder measured, and where it was written."""

    checkpoint: CheckpointReference
    seats: tuple[LadderSeat, ...]
    pairings: tuple[LadderPairing, ...]
    fit: RatingFit
    readings: tuple[LadderReading, ...] = ()
    response: TemperatureResponse | None = None
    #: Why a temperature row or the response could not be read, when one could
    #: not. A real state rather than an error: a row whose seats never finished
    #: a game has no ladder, and saying so beats reporting a zero.
    unavailable: dict[str, str] = field(default_factory=dict)
    view: ViewSelection | None = None
    dataset: DatasetReference | None = None
    records: tuple[GameRecord, ...] = field(default=(), repr=False)
    envelopes: tuple[ResultEnvelope, ...] = ()
    recorded_paths: tuple[Path, ...] = ()
    detail_paths: tuple[Path, ...] = ()

    @property
    def games(self) -> int:
        """Return how many scored games the whole ladder played."""

        return sum(pairing.games for pairing in self.pairings)

    @property
    def unfinished(self) -> int:
        """Return how many games ended at the ply limit with no result."""

        return sum(pairing.unfinished for pairing in self.pairings)

    def seat(self, key: SeatKey) -> LadderSeat:
        """Return one measured seat."""

        for candidate in self.seats:
            if candidate.key == key:
                return candidate
        raise LadderBenchmarkError(f"no seat was measured for {key.label}")

    def reading(self, temperature: float) -> LadderReading:
        """Return one temperature row's ladder."""

        for candidate in self.readings:
            if candidate.temperature == temperature:
                return candidate
        raise LadderBenchmarkError(f"no ladder was read at temperature {temperature:g}")

    def as_record(self) -> dict[str, Any]:
        """Return the full structured result, detail tier included."""

        return {
            "version": LADDER_BENCHMARK_VERSION,
            "generation_version": GENERATION_VERSION,
            "checkpoint": self.checkpoint.model_dump(mode="json"),
            "games": self.games,
            "unfinished": self.unfinished,
            "fit": self.fit.as_record(),
            "seats": [seat.as_record() for seat in self.seats],
            "pairings": [pairing.as_record() for pairing in self.pairings],
            "readings": [reading.as_record() for reading in self.readings],
            "response": None if self.response is None else self.response.as_record(),
            "unavailable": dict(sorted(self.unavailable.items())),
            "view": None if self.view is None else self.view.as_record(),
            "dataset": (
                None if self.dataset is None else self.dataset.model_dump(mode="json")
            ),
            "recorded": [str(path) for path in self.recorded_paths],
        }


@dataclass(frozen=True)
class _PositionSource:
    """The ladder's resolved roots, and how to describe them in a workload."""

    positions: tuple[StartPosition, ...]
    identity: dict[str, Any]


def benchmark_ladder(
    resolved_config: ResolvedConfig[LadderBenchmarkConfig],
    *,
    run_root: Path | None = None,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
    runner: ActionModelRunner | None = None,
    checkpoint: CheckpointReference | None = None,
) -> LadderBenchmarkResult:
    """Play the ladder, fit it, and report the rating and temperature response.

    Passing no ``store`` measures everything and records nothing, which is what
    an exploratory reading wants: a ladder played to look at one temperature is
    real but does not belong in the committed history.

    A ``runner`` may be supplied to measure an already-loaded checkpoint, in
    which case ``checkpoint`` identifies it. Otherwise both are resolved from
    the configuration.
    """

    config = resolved_config.value
    loaded, identity = resolve_model(
        config.model,
        runner,
        checkpoint,
        label=config.checkpoint_label,
        run_root=run_root,
        error=LadderBenchmarkError,
    )
    source, view, dataset = _position_source(config)
    seats = seat_keys(config)
    labels = {seat: f"{identity.label}-{seat.label}" for seat in seats}
    players = {
        seat: ModelPlayer(
            loaded,
            label=labels[seat],
            config=_seat_runtime(config, seat),
            checkpoint=identity,
        )
        for seat in seats
    }
    by_label = {label: seat for seat, label in labels.items()}

    pairings: list[LadderPairing] = []
    records: list[GameRecord] = []
    samples: list[DecisionSample] = []
    unscored = 0
    for first, second in _pairings(seats):
        played = _play_pairing(config, players[first], players[second], source)
        decisions = collect_decisions(played)
        samples.extend(decisions.samples)
        unscored += decisions.unscored_decisions
        pairings.append(_score_pairing(first, second, by_label, played))
        if config.detail.retain_games:
            records.extend(played)

    fit = fit_ratings(
        seats,
        pairings,
        anchor_seats=_anchor_seats(config),
        anchor_rating=_mean([float(value) for value in config.grid.target_ratings]),
        maximum_iterations=config.fit.maximum_iterations,
        tolerance=config.fit.tolerance,
        maximum_spread=config.fit.maximum_spread,
    )
    profiles = _error_profiles(DecisionSet(tuple(samples), unscored))
    device = runner_device(loaded)
    workload = _base_workload(config, source)
    measured_seats = tuple(
        LadderSeat(
            key=seat,
            games=_seat_games(seat, pairings),
            points=_seat_points(seat, pairings),
            unfinished=_seat_unfinished(seat, pairings),
            fitted_rating=fit.rating(seat),
            decisions=profiles.get(seat.setting),
            execution=execution_record(device, {**workload, **seat.as_record()}),
        )
        for seat in seats
    )

    unavailable: dict[str, str] = {}
    readings = _readings(config, fit, workload, device=device, unavailable=unavailable)
    response = _temperature_response(
        config,
        fit,
        workload,
        device=device,
        unavailable=unavailable,
    )
    result = LadderBenchmarkResult(
        checkpoint=identity,
        seats=measured_seats,
        pairings=tuple(pairings),
        fit=fit,
        readings=readings,
        response=response,
        unavailable=unavailable,
        view=view,
        dataset=dataset,
        records=tuple(records),
    )
    _log_summary(result)
    return _record(result, resolved_config, store=store, detail=detail)


def seat_keys(config: LadderBenchmarkConfig) -> tuple[SeatKey, ...]:
    """Return every competitor the configured ladder fields, in a fixed order.

    Ablated seats carry no configured rating, so there is one per temperature
    rather than one per cell: at a fixed temperature every ablated seat would be
    the same configuration, and pairing a configuration against itself measures
    nothing but the color balance.
    """

    seats = [
        SeatKey(SeatConditioning.CONDITIONED, rating, temperature)
        for temperature in config.grid.temperatures
        for rating in config.grid.target_ratings
    ]
    if config.ablation.enabled:
        seats.extend(
            SeatKey(SeatConditioning.ABLATED, None, temperature)
            for temperature in config.grid.temperatures
        )
    return tuple(seats)


def fit_ratings(
    seats: Sequence[SeatKey],
    pairings: Sequence[LadderPairing],
    *,
    anchor_seats: Sequence[SeatKey],
    anchor_rating: float,
    maximum_iterations: int = 500,
    tolerance: float = 1e-6,
    maximum_spread: float = 1200.0,
) -> RatingFit:
    """Fit one empirical rating per seat from the pairwise results.

    The estimator is Bradley-Terry by minorization-maximization, which is the
    maximum-likelihood fit of the same logistic model the expected-score formula
    states. Draws enter as half a point to each side, the conventional handling
    and the one that keeps a drawn ladder from being unfittable.

    The fit is invariant to multiplying every strength by a constant, so it is
    renormalized each iteration and shifted onto the configured scale at the
    end. That shift is the only place an absolute level enters: everything the
    benchmark reports beyond a seat's own fitted number — ordering, slope, span,
    and both temperature responses — is invariant to it.
    """

    if len(seats) < 2:
        raise LadderBenchmarkError("a rating fit needs at least two seats")
    points = {seat: 0.0 for seat in seats}
    games: dict[SeatKey, dict[SeatKey, int]] = {seat: {} for seat in seats}
    for pairing in pairings:
        if pairing.games == 0:
            continue
        for seat, opponent, scored in (
            (pairing.first, pairing.second, pairing.first_points),
            (pairing.second, pairing.first, pairing.games - pairing.first_points),
        ):
            if seat not in points:
                raise LadderBenchmarkError(
                    f"pairing names {seat.label}, which is not a seat of this ladder"
                )
            points[seat] += scored
            games[seat][opponent] = games[seat].get(opponent, 0) + pairing.games

    placed = tuple(seat for seat in seats if games[seat])
    unscored = tuple(seat for seat in seats if not games[seat])
    if len(placed) < 2:
        raise LadderBenchmarkError(
            "a rating fit needs at least two seats with a scored game; the "
            "ladder produced none, which is a generation failure rather than a "
            "degenerate fit"
        )

    ceiling = math.pow(10.0, maximum_spread / RATING_SCALE)
    floor = 1.0 / ceiling
    strengths = {seat: 1.0 for seat in placed}
    iterations = 0
    converged = False
    while iterations < maximum_iterations:
        iterations += 1
        updated: dict[SeatKey, float] = {}
        for seat in placed:
            denominator = sum(
                count / (strengths[seat] + strengths[opponent])
                for opponent, count in games[seat].items()
            )
            # A seat that scored nothing has a maximum-likelihood strength of
            # zero and one that scored everything has no finite maximum, so
            # both are pinned at the declared extreme and named in the result.
            value = (
                points[seat] / denominator if denominator and points[seat] else floor
            )
            updated[seat] = min(max(value, floor), ceiling)
        # The fit is scale-invariant, so it is renormalized every iteration.
        # Without it the whole ladder can drift toward one bound together and
        # every seat is reported as clamped.
        scale = _geometric_mean(list(updated.values()))
        updated = {
            seat: min(max(value / scale, floor), ceiling)
            for seat, value in updated.items()
        }
        shift = max(
            abs(_rating(updated[seat]) - _rating(strengths[seat])) for seat in placed
        )
        strengths = updated
        if shift < tolerance:
            converged = True
            break
    clamped = {
        seat
        for seat in placed
        if math.isclose(strengths[seat], floor, rel_tol=1e-9)
        or math.isclose(strengths[seat], ceiling, rel_tol=1e-9)
    }
    if not converged:
        logger.info(
            "Rating fit did not converge within %s iteration(s); reporting the "
            "last estimate, which is a result about the ladder rather than an "
            "error",
            maximum_iterations,
        )

    ratings = {seat: _rating(strength) for seat, strength in strengths.items()}
    basis = [seat for seat in anchor_seats if seat in ratings]
    anchor_basis = "reference-temperature"
    if not basis:
        basis = list(ratings)
        anchor_basis = "every-fitted-seat"
    offset = anchor_rating - _mean([ratings[seat] for seat in basis])
    return RatingFit(
        ratings={seat: value + offset for seat, value in ratings.items()},
        iterations=iterations,
        converged=converged,
        clamped=tuple(sorted(clamped, key=lambda seat: seat.sort_key)),
        unscored=unscored,
        anchor_rating=anchor_rating,
        anchor_basis=anchor_basis,
    )


def _anchor_seats(config: LadderBenchmarkConfig) -> tuple[SeatKey, ...]:
    """Return the seats the fit's absolute level is pinned to.

    The conditioned row at the reference temperature, so that row's ladder error
    reads as calibration shape while the other rows carry the offset temperature
    imposed on them. Anchoring on every seat instead would spread that offset
    across the grid and make each row's error depend on which temperatures the
    grid happened to include.
    """

    return tuple(
        SeatKey(
            SeatConditioning.CONDITIONED,
            rating,
            config.grid.reference_temperature,
        )
        for rating in config.grid.target_ratings
    )


def _pairings(seats: Sequence[SeatKey]) -> tuple[tuple[SeatKey, SeatKey], ...]:
    """Return every unordered pair of seats, in a fixed order.

    A full round robin rather than a scheduled subset. It keeps the comparison
    graph connected without a connectivity check, which is what a fit needs to
    place every seat on one scale, and the cost is the caller's to size through
    the grid and the games per pairing.
    """

    return tuple(
        (seats[left], seats[right])
        for left in range(len(seats))
        for right in range(left + 1, len(seats))
    )


def _seat_runtime(config: LadderBenchmarkConfig, seat: SeatKey) -> RuntimeConfig:
    """Return the runtime settings one seat plays every game under."""

    return config.runtime.model_copy(
        update={
            "target_rating": seat.target_rating,
            "temperature": seat.temperature,
        }
    )


def _play_pairing(
    config: LadderBenchmarkConfig,
    first: ModelPlayer,
    second: ModelPlayer,
    source: _PositionSource,
) -> tuple[GameRecord, ...]:
    """Play every seed's games between two seats."""

    played: list[GameRecord] = []
    for seed in config.grid.seeds:
        generation = config.generation.model_copy(update={"seed": seed})
        try:
            played.extend(
                generate_games(first, second, source.positions, config=generation)
            )
        except (GenerationError, PlayerError) as error:
            raise LadderBenchmarkError(f"cannot generate games: {error}") from error
    return tuple(played)


def _score_pairing(
    first: SeatKey,
    second: SeatKey,
    by_label: Mapping[str, SeatKey],
    played: Sequence[GameRecord],
) -> LadderPairing:
    """Reduce one pairing's games to points for the first seat.

    Which seat held which color is read from the seat labels the harness
    recorded rather than from the order the games were planned in, so a change
    to how colors are assigned cannot silently transpose a result.
    """

    points = 0.0
    scored = 0
    unfinished = 0
    for record in played:
        result = record.outcome.result
        if result == "*":
            unfinished += 1
            continue
        white = by_label.get(record.seat("white").label)
        black = by_label.get(record.seat("black").label)
        if {white, black} != {first, second}:
            raise LadderBenchmarkError(
                "a generated game names a seat outside the pairing that played it"
            )
        scored += 1
        if result == "1/2-1/2":
            points += 0.5
            continue
        if (white if result == "1-0" else black) == first:
            points += 1.0
    return LadderPairing(
        first=first,
        second=second,
        games=scored,
        first_points=points,
        unfinished=unfinished,
    )


def _error_profiles(decisions: DecisionSet) -> dict[DecisionSetting, DecisionCell]:
    """Return the decomposition cell for every seat that made a decision.

    The shared decomposition rather than a private one: it already groups by
    the dials a decision was made under, and a ladder's seats are exactly those
    groups. A suite whose seats made no classifiable decision reports no error
    profile rather than failing, since the strength reading stands without it.
    """

    if not decisions.samples:
        logger.info("No ladder decision carried a policy, so no error profile was read")
        return {}
    try:
        decomposition = summarize_decisions(decisions)
    except DecisionDecompositionError as error:  # pragma: no cover - guarded above
        raise LadderBenchmarkError(str(error)) from error
    return {
        cell.setting: cell for cell in decomposition.cells if cell.setting is not None
    }


def _readings(
    config: LadderBenchmarkConfig,
    fit: RatingFit,
    workload: Mapping[str, Any],
    *,
    device: torch.device,
    unavailable: dict[str, str],
) -> tuple[LadderReading, ...]:
    """Read the rating transfer function along each temperature row."""

    readings: list[LadderReading] = []
    for temperature in config.grid.temperatures:
        placed = [
            (rating, value)
            for rating in config.grid.target_ratings
            if (
                value := fit.rating(
                    SeatKey(SeatConditioning.CONDITIONED, rating, temperature)
                )
            )
            is not None
        ]
        if len(placed) < 2:
            unavailable[f"ladder@t{temperature:g}"] = (
                "fewer than two configured seats finished a scored game"
            )
            continue
        ratings = tuple(rating for rating, _ in placed)
        fitted = tuple(value for _, value in placed)
        readings.append(
            LadderReading(
                temperature=temperature,
                ratings=ratings,
                fitted=fitted,
                order_accuracy=_order_accuracy(fitted),
                adjacent_order_accuracy=_adjacent_order_accuracy(fitted),
                ladder_error=_mean(
                    [
                        abs(value - rating)
                        for rating, value in zip(ratings, fitted, strict=True)
                    ]
                ),
                slope=_slope([float(rating) for rating in ratings], list(fitted)),
                span=max(fitted) - min(fitted),
                execution=execution_record(
                    device,
                    {**workload, "temperature": temperature},
                ),
            )
        )
    return tuple(readings)


def _temperature_response(
    config: LadderBenchmarkConfig,
    fit: RatingFit,
    workload: Mapping[str, Any],
    *,
    device: torch.device,
    unavailable: dict[str, str],
) -> TemperatureResponse | None:
    """Measure what temperature costs in fitted rating, and what resists it."""

    if len(config.grid.temperatures) < 2:
        unavailable["temperature-response"] = (
            "a temperature response needs at least two temperatures"
        )
        return None
    per_rating: list[tuple[int, float]] = []
    for rating in config.grid.target_ratings:
        response = _axis_slope(
            [
                (
                    temperature,
                    fit.rating(
                        SeatKey(SeatConditioning.CONDITIONED, rating, temperature)
                    ),
                )
                for temperature in config.grid.temperatures
            ]
        )
        if response is not None:
            per_rating.append((rating, response))
    if not per_rating:
        unavailable["temperature-response"] = (
            "no configured rating was fitted at two or more temperatures"
        )
        return None

    ablated = None
    if config.ablation.enabled:
        ablated = _axis_slope(
            [
                (
                    temperature,
                    fit.rating(SeatKey(SeatConditioning.ABLATED, None, temperature)),
                )
                for temperature in config.grid.temperatures
            ]
        )
    conditioned = _mean([response for _, response in per_rating])
    attenuation: float | None = None
    reason: str | None = None
    if ablated is None:
        reason = (
            "no ablated arm was fitted at two or more temperatures"
            if config.ablation.enabled
            else "rating-conditioning ablation was disabled"
        )
    elif abs(ablated) < 1.0:
        # A ratio against an ablated response of a rating point or two is
        # dominated by the noise in both arms, and would report a confident
        # attenuation from two numbers that are indistinguishable from zero.
        reason = "the ablated temperature response is too close to zero to divide by"
    else:
        attenuation = 1.0 - conditioned / ablated
    if reason is not None:
        unavailable["temperature-response-attenuation"] = reason
    return TemperatureResponse(
        temperatures=tuple(config.grid.temperatures),
        per_rating=tuple(per_rating),
        conditioned_response=conditioned,
        ablated_response=ablated,
        attenuation=attenuation,
        attenuation_unavailable=reason,
        execution=execution_record(device, dict(workload)),
    )


def _position_source(
    config: LadderBenchmarkConfig,
) -> tuple[_PositionSource, ViewSelection | None, DatasetReference | None]:
    """Resolve the roots every pairing plays from.

    Every pairing plays the same roots, which is what makes the round robin
    balanced: a seat that met a different set of openings from its opponents
    would carry that difference into its fitted rating.
    """

    if config.openings.pool is None:
        return (
            _PositionSource(
                positions=standard_positions(label="standard-start"),
                identity={"kind": "standard-start"},
            ),
            None,
            None,
        )
    pool = _load_pool(config.openings.pool)
    view_config = config.openings.view.model_copy(
        update={"prefix_plies": config.openings.plies}
    )
    selection = apply_view(pool.games, view_config)
    if not selection.game_ids:
        raise LadderBenchmarkError(
            f"view {view_config.name!r} selected no opening games from the pool"
        )
    wanted = set(selection.game_ids)
    try:
        rows = [
            _prefix_row(row, config.openings.plies)
            for row in read_normalized_rows(pool.games_path)
            if int(row[NormalizedColumn.GAME_ID]) in wanted
        ]
    except DataLoadingError as error:
        raise LadderBenchmarkError(str(error)) from error
    if len(rows) != len(wanted):
        raise LadderBenchmarkError(
            "the evaluation pool does not contain every selected opening game"
        )
    games = sorted(
        (
            int(row[NormalizedColumn.GAME_ID.value]),
            tuple(int(value) for value in row[NormalizedColumn.ACTION_IDS.value]),
        )
        for row in rows
    )
    try:
        positions = prefix_positions(games, plies=config.openings.plies)
    except GenerationError as error:
        raise LadderBenchmarkError(str(error)) from error
    dataset = _dataset_reference(pool, selection, rows)
    return (
        _PositionSource(
            positions=positions,
            identity={
                "kind": "human-prefix",
                "prefix_plies": config.openings.plies,
                "pool_id": dataset.pool_id,
                "pool_version": dataset.pool_version,
                "view": dataset.view,
                "game_ids_sha256": dataset.game_ids_sha256,
            },
        ),
        selection,
        dataset,
    )


def _load_pool(path: Path) -> FrozenPool:
    """Load the frozen pool the ladder's openings are projected out of."""

    try:
        return load_pool(path)
    except EvaluationPoolError as error:
        raise LadderBenchmarkError(str(error)) from error


def _prefix_row(row: Mapping[str, Any], plies: int) -> dict[str, Any]:
    """Truncate one pool game to the opening the seats were actually given."""

    updated = dict(row)
    values = updated.get(NormalizedColumn.ACTION_IDS.value)
    if values is not None:
        updated[NormalizedColumn.ACTION_IDS.value] = list(values)[:plies]
    return updated


def _dataset_reference(
    pool: FrozenPool,
    selection: ViewSelection,
    rows: Sequence[Mapping[str, Any]],
) -> DatasetReference:
    """Describe the human games the openings were projected out of.

    Provenance rather than series identity. A ladder metric declares no
    projection, so this digest never enters a fingerprint; the openings reach
    it through the workload's position source instead.
    """

    try:
        component = projection_content_digest(rows, MOVE_PREDICTION_PROJECTION)
    except FingerprintError as error:
        raise LadderBenchmarkError(str(error)) from error
    return pool_dataset_reference(
        pool,
        selection,
        component,
        error=LadderBenchmarkError,
    )


def _base_workload(
    config: LadderBenchmarkConfig,
    source: _PositionSource,
) -> dict[str, Any]:
    """Declare what every reading of this ladder measured.

    The whole grid is in every workload, including a single seat's, because a
    fitted rating is an output of the joint fit rather than of that seat's own
    games: adding a temperature or an ablated arm changes the population every
    seat was placed against, and therefore changes what its number means.

    Seed count and games per pairing are deliberately absent. More games
    estimate the same ladder more precisely, so putting either in identity would
    end a series for a precision change.
    """

    return {
        "generation_version": GENERATION_VERSION,
        "positions": source.identity,
        "target_ratings": list(config.grid.target_ratings),
        "temperatures": list(config.grid.temperatures),
        "reference_temperature": config.grid.reference_temperature,
        "ablation": (
            CONDITIONING_TREATMENTS[SeatConditioning.ABLATED].value
            if config.ablation.enabled
            else "none"
        ),
        "pairing": "round-robin",
        "fit_model": FIT_MODEL,
        "fit_scale": RATING_SCALE,
        "fit_maximum_spread": config.fit.maximum_spread,
        "maximum_generated_plies": config.generation.maximum_generated_plies,
        "swap_colors": config.generation.swap_colors,
        "claim_draws": config.generation.claim_draws,
        "resignation_enabled": config.runtime.resignation_enabled,
        "draw_claim_enabled": config.runtime.draw_claim_enabled,
    }


def _record(
    result: LadderBenchmarkResult,
    resolved_config: ResolvedConfig[LadderBenchmarkConfig],
    *,
    store: ResultsStore | None,
    detail: DetailStore | None,
) -> LadderBenchmarkResult:
    """Write one envelope per seat, per temperature row, and per response.

    Three units rather than one record, because they are scoped differently. A
    seat carries what one configuration played like, a row carries the transfer
    function at one temperature, and the response spans the whole grid. Each
    declares its own workload and so owns its own series.
    """

    with ResultRecorder(
        resolved_config,
        kind=LADDER_KIND,
        benchmark=LADDER_BENCHMARK,
        checkpoint=result.checkpoint,
        store=store,
        detail=detail,
        error=LadderBenchmarkError,
    ) as recorder:
        for seat in result.seats:
            measurements = _seat_measurements(seat)
            if not measurements:
                continue
            recorder.add(
                measurements,
                payload=partial(_seat_payload, result, seat),
                description=f"Rating-ladder seat: {seat.label}",
                slug=f"seat-{_slug(seat.label)}",
                data=result.dataset,
                execution=seat.execution,
            )
        for reading in result.readings:
            recorder.add(
                _reading_measurements(reading),
                payload=partial(_reading_payload, result, reading),
                description=f"Rating ladder: {reading.label}",
                slug=f"ladder-t{_slug(f'{reading.temperature:g}')}",
                data=result.dataset,
                execution=reading.execution,
            )
        if result.response is not None:
            recorder.add(
                _response_measurements(result.response),
                payload=result.response.as_record,
                description="Rating-ladder temperature response",
                slug="temperature-response",
                data=result.dataset,
                execution=result.response.execution,
            )
        return replace(result, **recorder.commit())


def _seat_measurements(seat: LadderSeat) -> tuple[Measurement, ...]:
    """Return one seat's committed strength reading and error profile.

    A seat the fit could not place reports nothing rather than a zero, and the
    seat's own record says why. The error profile is reported beside strength
    rather than in place of it: a temperature that preserves average score while
    moving the profile has changed the shape of the mistakes, which no strength
    number can show.
    """

    workload = seat.execution.workload_component()
    values: list[tuple[str, float, int | None]] = []
    if seat.fitted_rating is not None:
        values.append((LADDER_FITTED_RATING.identifier, seat.fitted_rating, seat.games))
    if seat.games:
        values.append((LADDER_SCORE_RATE.identifier, seat.score_rate, seat.games))
    profile = seat.decisions
    if profile is not None and profile.decisions:
        values.extend(
            [
                (
                    LADDER_PREFERRED_SELECTION_RATE.identifier,
                    profile.preferred_selection_rate,
                    profile.decisions,
                ),
                (
                    LADDER_POLICY_REGRET.identifier,
                    profile.policy_regret,
                    profile.decisions,
                ),
                (
                    LADDER_SELECTED_RANK.identifier,
                    profile.selected_rank,
                    profile.decisions,
                ),
            ]
        )
        # A greedy seat departs from its own preference never, and a zero there
        # would read as a measured near tie rather than as a quantity that does
        # not exist.
        if profile.departures:
            values.append(
                (
                    LADDER_DEPARTURE_POLICY_REGRET.identifier,
                    profile.departure_policy_regret,
                    profile.departures,
                )
            )
    return tuple(
        measurement(identifier, value, workload=workload, sample_size=sample_size)
        for identifier, value, sample_size in values
    )


def _reading_measurements(reading: LadderReading) -> tuple[Measurement, ...]:
    """Return one temperature row's committed transfer-function reading."""

    workload = reading.execution.workload_component()
    seats = len(reading.ratings)
    values: tuple[tuple[str, float, int], ...] = (
        (LADDER_RATING_ORDER_ACCURACY.identifier, reading.order_accuracy, seats),
        (
            LADDER_ADJACENT_RATING_ORDER_ACCURACY.identifier,
            reading.adjacent_order_accuracy,
            seats,
        ),
        (LADDER_RATING_ERROR.identifier, reading.ladder_error, seats),
        (LADDER_FITTED_RATING_SLOPE.identifier, reading.slope, seats),
        (LADDER_FITTED_RATING_SPAN.identifier, reading.span, seats),
    )
    return tuple(
        measurement(identifier, value, workload=workload, sample_size=sample_size)
        for identifier, value, sample_size in values
    )


def _response_measurements(response: TemperatureResponse) -> tuple[Measurement, ...]:
    """Return the committed temperature response and its attenuation."""

    workload = response.execution.workload_component()
    values: list[tuple[str, float, int]] = [
        (
            LADDER_TEMPERATURE_RESPONSE.identifier,
            response.conditioned_response,
            len(response.per_rating),
        )
    ]
    if response.ablated_response is not None:
        values.append(
            (
                LADDER_ABLATED_TEMPERATURE_RESPONSE.identifier,
                response.ablated_response,
                len(response.temperatures),
            )
        )
    if response.attenuation is not None:
        values.append(
            (
                LADDER_TEMPERATURE_RESPONSE_ATTENUATION.identifier,
                response.attenuation,
                len(response.per_rating),
            )
        )
    return tuple(
        measurement(identifier, value, workload=workload, sample_size=sample_size)
        for identifier, value, sample_size in values
    )


def _seat_payload(
    result: LadderBenchmarkResult,
    seat: LadderSeat,
) -> dict[str, Any]:
    """Return one seat's diagnostics, with the games it held white in.

    Every game has exactly one white seat, so keying retained records on that
    writes each game once across the whole suite rather than once per seat that
    played it. A record carries both seats either way, so nothing is lost.
    """

    payload = seat.as_record()
    payload["fit"] = result.fit.as_record()
    payload["pairings"] = [
        pairing.as_record()
        for pairing in result.pairings
        if seat.key in (pairing.first, pairing.second)
    ]
    payload["games_detail"] = [
        record.as_record() for record in result.records if _held_white(record, seat.key)
    ]
    return payload


def _reading_payload(
    result: LadderBenchmarkResult,
    reading: LadderReading,
) -> dict[str, Any]:
    """Return one temperature row's ladder, fit state, and pairwise results."""

    payload = reading.as_record()
    payload["fit"] = result.fit.as_record()
    payload["pairings"] = [pairing.as_record() for pairing in result.pairings]
    payload["unavailable"] = dict(sorted(result.unavailable.items()))
    return payload


def _held_white(record: GameRecord, seat: SeatKey) -> bool:
    """Return whether one seat configuration held white in one recorded game."""

    configuration = record.seat("white").configuration
    return (
        configuration.get("target_rating") == seat.target_rating
        and configuration.get("temperature") == seat.temperature
    )


def _seat_games(seat: SeatKey, pairings: Iterable[LadderPairing]) -> int:
    return sum(
        pairing.games for pairing in pairings if seat in (pairing.first, pairing.second)
    )


def _seat_unfinished(seat: SeatKey, pairings: Iterable[LadderPairing]) -> int:
    return sum(
        pairing.unfinished
        for pairing in pairings
        if seat in (pairing.first, pairing.second)
    )


def _seat_points(seat: SeatKey, pairings: Iterable[LadderPairing]) -> float:
    return sum(
        pairing.first_points
        if pairing.first == seat
        else pairing.games - pairing.first_points
        for pairing in pairings
        if seat in (pairing.first, pairing.second)
    )


def _log_summary(result: LadderBenchmarkResult) -> None:
    """Report what the ladder found, before anything is written."""

    logger.info(
        "Rating ladder: %s scored game(s) over %s pairing(s), %s unfinished; "
        "fit %s after %s iteration(s)",
        result.games,
        len(result.pairings),
        result.unfinished,
        "converged" if result.fit.converged else "did not converge",
        result.fit.iterations,
    )
    for reading in result.readings:
        logger.info(
            "Ladder at %s: order %.3f, adjacent %.3f, error %.1f, slope %.3f",
            reading.label,
            reading.order_accuracy,
            reading.adjacent_order_accuracy,
            reading.ladder_error,
            reading.slope,
        )


def _rating(strength: float) -> float:
    """Return the rating one Bradley-Terry strength corresponds to."""

    return RATING_SCALE * math.log10(strength)


def _geometric_mean(values: Sequence[float]) -> float:
    return math.exp(_mean([math.log(value) for value in values]))


def _order_accuracy(values: Sequence[float]) -> float:
    """Return how often a later value exceeds an earlier one, ties scoring half."""

    outcomes: list[float] = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            difference = values[right] - values[left]
            outcomes.append(1.0 if difference > 0 else 0.5 if difference == 0 else 0.0)
    return _mean(outcomes)


def _adjacent_order_accuracy(values: Sequence[float]) -> float:
    """Return the same accuracy over neighbouring pairs alone."""

    return _order_accuracy_of(
        [(values[index], values[index + 1]) for index in range(len(values) - 1)]
    )


def _order_accuracy_of(pairs: Sequence[tuple[float, float]]) -> float:
    outcomes = [
        1.0 if upper > lower else 0.5 if upper == lower else 0.0
        for lower, upper in pairs
    ]
    return _mean(outcomes)


def _slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    """Return the least-squares slope of ``y`` against ``x``."""

    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0.0:
        raise LadderBenchmarkError("a slope needs at least two distinct inputs")
    return (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / denominator
    )


def _axis_slope(points: Sequence[tuple[float, float | None]]) -> float | None:
    """Return the slope over the points that have a value, or nothing."""

    placed = [(x_value, y_value) for x_value, y_value in points if y_value is not None]
    if len({x_value for x_value, _ in placed}) < 2:
        return None
    return _slope(
        [x_value for x_value, _ in placed],
        [y_value for _, y_value in placed],
    )


def _slug(value: str) -> str:
    """Return a filename-safe form of a label."""

    return value.replace(".", "_").replace("@", "-")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise LadderBenchmarkError("cannot average an empty ladder measurement")
    return float(sum(values) / len(values))


__all__ = [
    "FIT_MODEL",
    "LADDER_BENCHMARK",
    "LADDER_BENCHMARK_VERSION",
    "LADDER_KIND",
    "RATING_SCALE",
    "LadderAblationConfig",
    "LadderBenchmarkConfig",
    "LadderBenchmarkError",
    "LadderBenchmarkResult",
    "LadderDetailConfig",
    "LadderFitConfig",
    "LadderGridConfig",
    "LadderOpeningsConfig",
    "LadderPairing",
    "LadderReading",
    "LadderSeat",
    "RatingFit",
    "CONDITIONING_TREATMENTS",
    "SeatConditioning",
    "SeatKey",
    "TemperatureResponse",
    "benchmark_ladder",
    "fit_ratings",
    "seat_keys",
]
