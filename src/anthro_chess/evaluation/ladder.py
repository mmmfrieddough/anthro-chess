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

**Every reported number carries what it can resolve.** A flat ladder and a
sample too thin to see a slope in look identical without one, which is the
distinction the first full-size reading could not draw. Ordering, slope, span,
ladder error and both temperature responses are functions of the fitted
ratings, so the floor beside them is estimated the only way that reaches all of
them at once: redraw the games each pairing played, refit, and read the spread
of everything the fit yields. Pairings whose seats are all greedy replay rather
than redraw and are held fixed, so a reading is qualified against the games that
would actually have differed.

Nothing here waits in wall-clock time. Every game comes from the shared
generation harness under an explicit seed, so a suite reproduces exactly and a
single game reproduces on its own from the seed its record carries.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
from pydantic import Field, StrictBool, StrictInt, model_validator

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data.schema import (
    NormalizedColumn,
    row_game_id,
)
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
from anthro_chess.evaluation.execution import execution_record, runner_device
from anthro_chess.evaluation.games import (
    GENERATION_VERSION,
    GameRecord,
    GenerationConfig,
    GenerationError,
    ModelPlayer,
    PlayerError,
    StartPosition,
    collapse_replicates,
    generate_games,
    prefix_positions,
    replicates_vary,
    standard_positions,
)
from anthro_chess.evaluation.noise import NoiseConfig
from anthro_chess.evaluation.pool import (
    EvaluationPoolError,
    FrozenPool,
    PoolGenerationPin,
    load_pool,
    pool_rows,
)
from anthro_chess.evaluation.recording import (
    ResultRecording,
    pool_dataset_reference,
    resolve_model,
)
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DatasetReference,
    ExecutionRecord,
    Measurement,
    MetricDispersion,
    ResultEnvelope,
    measured_dispersion,
    measurement,
    self_combined_floor,
)
from anthro_chess.evaluation.results.fingerprints import (
    FingerprintError,
    WorkloadComponent,
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
    LADDER_SCORED_GAME_RATE,
    LADDER_SELECTED_RANK,
    LADDER_TEMPERATURE_RESPONSE,
    LADDER_TEMPERATURE_RESPONSE_ATTENUATION,
    MOVE_PREDICTION_PROJECTION,
)
from anthro_chess.evaluation.results.records import NoiseFloorKind
from anthro_chess.evaluation.selection import CheckpointSelection
from anthro_chess.evaluation.views import ViewConfig, ViewSelection, apply_view
from anthro_chess.runtime import ActionModelRunner, RuntimeConfig

#: Version 2 stores each quantity's own dispersion where version 1 stored the
#: floor and the coverage that scaled it.
LADDER_BENCHMARK_VERSION = 2

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

#: How the ladder estimates what its own reading can resolve. Every quantity it
#: reports is an output of one fit rather than a per-game average, so the
#: resample redraws each pairing's games and runs the fit again; the spread of
#: the refits is the spread of everything downstream of them.
LADDER_BOOTSTRAP_METHOD = "refit-over-resampled-games"

#: How the ladder states the floor of a reading nothing would redraw. A grid of
#: greedy seats replays every game it played, so its evaluation noise is exactly
#: zero rather than small, per decision 0032.
LADDER_DETERMINISTIC_METHOD = "deterministic-seats"

#: What a reading records when neither of the above ran: the pairings a fresh
#: seed would redraw hold too few games to show a spread. Named rather than left
#: as the bootstrap's method, so a stored record does not claim a resample that
#: was never drawn.
LADDER_UNRESOLVED_METHOD = "too-few-redrawn-games"

#: The noise a ladder spread bounds. A rollout has no fixed data to re-measure
#: on — the games are the draw — so resampling them and re-running under another
#: seed estimate the same quantity, and that quantity is what qualifies a delta
#: between two checkpoints.
LADDER_FLOOR_KIND: NoiseFloorKind = "evaluation"

#: What the temperature response's spreads are filed under. Seats and rows name
#: themselves; the response spans the whole grid and has no narrower scope.
RESPONSE_SCOPE = "response"

logger = logging.getLogger(__name__)

#: What the opening projection reads from a pool row: the moves it continues from,
#: and nothing beyond what the provenance digest already declares.
_OPENING_COLUMNS = (
    NormalizedColumn.SOURCE_ID.value,
    NormalizedColumn.SOURCE_GAME_KEY.value,
    *MOVE_PREDICTION_PROJECTION.columns,
)


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
    #: than a measurement setting, and a pairing of two greedy seats plays the
    #: first of them alone, since it would replay one game.
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


class LadderOpeningsConfig(PoolGenerationPin):
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

    @model_validator(mode="after")
    def _validate_pool(self) -> LadderOpeningsConfig:
        if self.pool is None and self.expected_pool_game_ids_sha256 is not None:
            raise ValueError(
                "expected_pool_game_ids_sha256 names the generation of a pool "
                "this selection does not read"
            )
        return self


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


class LadderBenchmarkConfig(CheckpointSelection):
    """Code-owned schema for ``anthro eval ladder``."""

    #: The base runtime settings every seat plays under. Rating, temperature,
    #: and seed are supplied per seat and per game, so setting them here would
    #: be overridden; the rest applies to the whole ladder.
    runtime: RuntimeConfig = RuntimeConfig()
    grid: LadderGridConfig = LadderGridConfig()
    generation: GenerationConfig = GenerationConfig()
    openings: LadderOpeningsConfig = LadderOpeningsConfig()
    ablation: LadderAblationConfig = LadderAblationConfig()
    fit: LadderFitConfig = LadderFitConfig()
    #: How the reading is qualified. Named as every other benchmark names it,
    #: so one override key reaches the whole suite. A precision setting rather
    #: than a measurement one — it decides how finely the floor beside a number
    #: is read, never the number — so it stays out of series identity like the
    #: seeds and the games per pairing.
    noise: NoiseConfig = NoiseConfig()
    detail: LadderDetailConfig = LadderDetailConfig()


@dataclass(frozen=True)
class LadderPairing:
    """Every game two seats played against each other.

    How the games ended rather than what they came to. A point total is what
    the fit reads, but it is not what another seed would redraw: a resample of
    this pairing draws outcomes, and half a point is not one of them. The
    totals are derived so the two cannot disagree.

    A game that hit the ply limit has no result, so it informs no pairwise
    comparison; it is counted separately rather than adjudicated into a draw,
    which would report the ply limit as a level of play.
    """

    first: SeatKey
    second: SeatKey
    first_wins: int
    draws: int
    first_losses: int
    unfinished: int = 0
    #: The seeds this pairing actually played, which ``collapse_replicates``
    #: cuts to one when both seats are greedy.
    seeds: tuple[int, ...] = ()

    @property
    def games(self) -> int:
        """Return how many of this pairing's games were scored."""

        return self.first_wins + self.draws + self.first_losses

    @property
    def first_points(self) -> float:
        """Return the first seat's points, counting a draw as a half."""

        return self.first_wins + 0.5 * self.draws

    @property
    def played(self) -> int:
        """Return every game played, whether or not it reached a result."""

        return self.games + self.unfinished

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one pairing."""

        return {
            "first": self.first.as_record(),
            "second": self.second.as_record(),
            "games": self.games,
            "first_points": self.first_points,
            "first_wins": self.first_wins,
            "draws": self.draws,
            "first_losses": self.first_losses,
            "unfinished": self.unfinished,
            "seeds": list(self.seeds),
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
    def played(self) -> int:
        """Return every game this seat played, scored or not."""

        return self.games + self.unfinished

    @property
    def scored_game_rate(self) -> float:
        """Return the share of this seat's games that reached a result."""

        return self.games / self.played if self.played else 0.0

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
            "scored_game_rate": self.scored_game_rate,
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
class LadderResolution:
    """What each quantity the ladder reports can resolve, and how that is known.

    One estimate for the whole ladder rather than one per reported number,
    because there is one ladder: ordering, slope, span, ladder error and both
    temperature responses are functions of the same fitted ratings, so a
    resample that redraws the games moves all of them together and their
    spreads come off the same refits.

    A quantity may have none, and that is a state rather than a gap. A seat the
    fit clamped has no finite maximum-likelihood rating, so its number is a
    bound rather than an estimate and no resample can say how precise it is:
    every resample of a seat that won every game wins every game again.
    ``unqualifiable`` names those and says why, so a reader is not left holding
    a number with nothing beside it.
    """

    #: Spreads keyed by the scope they were read at and the metric they
    #: qualify. The scope is a seat's label, a temperature row's label, or
    #: ``response``.
    dispersions: Mapping[tuple[str, str], MetricDispersion]
    #: Why a quantity has no spread, keyed the same way.
    unqualifiable: Mapping[tuple[str, str], str]
    method: str
    #: The confidence the bound beside every spread carries, per decision 0026.
    #: A stored bound cannot be read without it.
    confidence: float
    replayed_pairings: int
    #: Everything below counts what a resample did, so all of it is zero where
    #: none was drawn. Defaulted rather than restated, so the two paths that
    #: draw nothing cannot come to disagree about what that looks like.
    #:
    #: Resamples drawn. Zero on a stated spread, which does not depend on
    #: resampling and stands where a bootstrap could not.
    resamples: int = 0
    #: Resamples that produced a fit. Below ``resamples`` where a draw left
    #: fewer than two seats with a scored game, which is a thin reading rather
    #: than an error but is not what the configured count says it is.
    fitted_resamples: int = 0
    #: Games in the pairings a fresh seed would redraw, which is what the
    #: dispersion bound's degrees of freedom count. Games in a pairing of greedy
    #: seats are excluded: they replay, so they contribute no spread to bound.
    redrawn_games: int = 0
    #: Resamples whose refit ran out of iterations. A diagnostic beside the
    #: spreads rather than a filter on them: the spread of an estimator that is
    #: struggling is still the spread of the number the benchmark reports, and a
    #: reader deciding how much to trust it needs to see it.
    non_convergent_resamples: int = 0

    def dispersion(self, scope: str, metric: str) -> MetricDispersion | None:
        """Return the spread beside one reported quantity, if it has one."""

        return self.dispersions.get((scope, metric))

    def floor(self, scope: str, metric: str) -> float | None:
        """Return what a delta against a reading like this one would clear.

        A ladder reading resolves nothing on its own, so what a single reading
        can show is the floor the other operand matching it would produce.
        """

        spread = self.dispersion(scope, metric)
        return None if spread is None else self_combined_floor(spread)

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of the whole resolution estimate."""

        return {
            "method": self.method,
            "confidence": self.confidence,
            "resamples": self.resamples,
            "fitted_resamples": self.fitted_resamples,
            "redrawn_games": self.redrawn_games,
            "replayed_pairings": self.replayed_pairings,
            "non_convergent_resamples": self.non_convergent_resamples,
            "dispersions": [
                {
                    "scope": scope,
                    "metric": metric,
                    "dispersion": dispersion.model_dump(mode="json"),
                }
                for (scope, metric), dispersion in sorted(self.dispersions.items())
            ],
            "unqualifiable": [
                {"scope": scope, "metric": metric, "reason": reason}
                for (scope, metric), reason in sorted(self.unqualifiable.items())
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
    #: What the reading can resolve. Absent only where it was switched off.
    resolution: LadderResolution | None = None
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
            "resolution": (
                None if self.resolution is None else self.resolution.as_record()
            ),
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
    recording: ResultRecording,
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
        seeds, generation = collapse_replicates(
            config.grid.seeds,
            config.generation,
            temperatures=(first.temperature, second.temperature),
        )
        played = _play_pairing(
            players[first],
            players[second],
            source,
            seeds=seeds,
            generation=generation,
        )
        decisions = collect_decisions(played)
        samples.extend(decisions.samples)
        unscored += decisions.unscored_decisions
        pairings.append(_score_pairing(first, second, by_label, played, seeds=seeds))
        if config.detail.retain_games:
            records.extend(played)

    fit = _fit(config, seats, pairings)
    profiles = _error_profiles(DecisionSet(tuple(samples), unscored))
    device = runner_device(loaded)
    workload = _base_workload(config, source)
    totals = _seat_totals(seats, pairings)
    measured_seats = tuple(
        LadderSeat(
            key=seat,
            games=games,
            points=points,
            unfinished=played - games,
            fitted_rating=fit.rating(seat),
            decisions=profiles.get(seat.setting),
            execution=execution_record(device, {**workload, **seat.as_record()}),
        )
        for seat, (games, points, played) in totals.items()
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
    resolution = _resolution(
        config,
        seats,
        pairings,
        fit,
        workload,
        device=device,
    )
    result = LadderBenchmarkResult(
        checkpoint=identity,
        seats=measured_seats,
        pairings=tuple(pairings),
        fit=fit,
        readings=readings,
        response=response,
        resolution=resolution,
        unavailable=unavailable,
        view=view,
        dataset=dataset,
        records=tuple(records),
    )
    _log_summary(result)
    _record(result, recording)
    return result


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
            # The seat's own strength is loop-invariant here, and this runs once
            # per opponent per iteration per resample — hashing a seat key to
            # fetch it again each time is the largest cost in the resolution
            # step that buys nothing.
            own = strengths[seat]
            denominator = sum(
                count / (own + strengths[opponent])
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
    # Convergence is returned rather than logged here. The reading logs its own
    # state once, and this runs a thousand more times inside the resample that
    # qualifies it, where a line per fit would bury the reading's own.
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
    first: ModelPlayer,
    second: ModelPlayer,
    source: _PositionSource,
    *,
    seeds: tuple[int, ...],
    generation: GenerationConfig,
) -> tuple[GameRecord, ...]:
    """Play every replicate's games between two seats."""

    played: list[GameRecord] = []
    for seed in seeds:
        try:
            played.extend(
                generate_games(
                    first,
                    second,
                    source.positions,
                    config=generation.model_copy(update={"seed": seed}),
                )
            )
        except (GenerationError, PlayerError) as error:
            raise LadderBenchmarkError(f"cannot generate games: {error}") from error
    return tuple(played)


def _score_pairing(
    first: SeatKey,
    second: SeatKey,
    by_label: Mapping[str, SeatKey],
    played: Sequence[GameRecord],
    *,
    seeds: tuple[int, ...],
) -> LadderPairing:
    """Reduce one pairing's games to points for the first seat.

    Which seat held which color is read from the seat labels the harness
    recorded rather than from the order the games were planned in, so a change
    to how colors are assigned cannot silently transpose a result.
    """

    wins = 0
    draws = 0
    losses = 0
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
        if result == "1/2-1/2":
            draws += 1
        elif (white if result == "1-0" else black) == first:
            wins += 1
        else:
            losses += 1
    return LadderPairing(
        first=first,
        second=second,
        first_wins=wins,
        draws=draws,
        first_losses=losses,
        unfinished=unfinished,
        seeds=seeds,
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


def _resolution(
    config: LadderBenchmarkConfig,
    seats: Sequence[SeatKey],
    pairings: Sequence[LadderPairing],
    fit: RatingFit,
    workload: Mapping[str, Any],
    *,
    device: torch.device,
) -> LadderResolution | None:
    """Return what this reading can resolve, quantity by quantity.

    Refitting a resample rather than perturbing the numbers the fit produced.
    Ordering is a step function of rating differences and a ladder error is a
    mean of absolute ones, so neither has a derivative for uncertainty to be
    propagated through; running the whole reduction again on redrawn games
    reaches every quantity by the same route the reading itself took.

    A pairing is redrawn at the count of games it *played* rather than the
    count it scored, so a game reaching the ply limit is an outcome the redraw
    can produce. That is what makes the scored-game count a measured quantity
    rather than a denominator, and it is the quantity that has discriminated
    most sharply between checkpoints so far.

    The draw is over a pairing's games without regard to which opening each came
    from, and deliberately so: the between-opening spread a stratified draw
    would remove measures at nothing.
    ``docs/decisions/0034-qualifying-a-rating-ladder-reading.md`` owns the rest,
    and
    ``docs/decisions/0039-stratifying-the-ladder-redraw-costs-more-than-it-removes.md``
    owns that measurement.
    """

    settings = config.noise
    if not settings.enabled:
        return None
    # What a re-run would redraw is decided by the seats rather than by what
    # they managed to play, per decision 0032: a pairing that produced no game
    # is a thin reading, not a replayed one, and calling it replayed would state
    # a floor of zero over a generation failure.
    varying = {
        index: pairing
        for index, pairing in enumerate(pairings)
        if replicates_vary((pairing.first.temperature, pairing.second.temperature))
    }
    redrawn = {index: pairing for index, pairing in varying.items() if pairing.played}
    replayed = len(pairings) - len(varying)
    observed = _quantities(config, seats, pairings, fit, workload, device=device)
    if not varying:
        # Nothing to draw. Every seat is greedy, so a fresh seed replays the
        # ladder move for move and the delta between two such readings is
        # attributable to the weights alone — a seat pinned at the declared
        # spread included, since it is pinned identically both times.
        return LadderResolution(
            dispersions={
                key: MetricDispersion(
                    value=0.0,
                    bound=0.0,
                    kind=LADDER_FLOOR_KIND,
                    source=_floor_source(LADDER_DETERMINISTIC_METHOD),
                    estimator=LADDER_DETERMINISTIC_METHOD,
                )
                for key in observed
            },
            unqualifiable={},
            method=LADDER_DETERMINISTIC_METHOD,
            confidence=settings.confidence,
            replayed_pairings=replayed,
        )

    unqualifiable = _unbounded_seats(seats, pairings)
    redrawn_games = sum(pairing.played for pairing in redrawn.values())
    if redrawn_games < 2:
        # One game shows no spread, and a bound needs a degree of freedom to be
        # computed at all. Saying so beats failing a reading whose games are
        # already played.
        return LadderResolution(
            dispersions={},
            unqualifiable={
                key: (
                    "the pairings a fresh seed would redraw hold fewer than two "
                    "games between them, which shows no spread to bound"
                )
                for key in observed
            }
            | unqualifiable,
            method=LADDER_UNRESOLVED_METHOD,
            confidence=settings.confidence,
            redrawn_games=redrawn_games,
            replayed_pairings=replayed,
        )

    source = _floor_source(LADDER_BOOTSTRAP_METHOD)

    generator = np.random.default_rng(settings.seed)
    outcomes = {
        index: _resample_outcomes(pairing, generator, settings.resamples)
        for index, pairing in redrawn.items()
    }
    replicates: dict[tuple[str, str], list[float]] = {key: [] for key in observed}
    non_convergent = 0
    fitted = 0
    for resample in range(settings.resamples):
        resampled = [
            _resampled_pairing(pairing, outcomes[index][resample])
            if index in outcomes
            else pairing
            for index, pairing in enumerate(pairings)
        ]
        try:
            replicate = _fit(config, seats, resampled)
        except LadderBenchmarkError:
            # A resample that left fewer than two seats with a scored game is
            # not a fit at all. It is dropped rather than failing the reading,
            # and the quantities it would have carried lose a replicate.
            continue
        fitted += 1
        non_convergent += not replicate.converged
        for key, value in _quantities(
            config, seats, resampled, replicate, workload, device=device
        ).items():
            if key in replicates:
                replicates[key].append(value)

    dispersions: dict[tuple[str, str], MetricDispersion] = {}
    for key in observed:
        if key in unqualifiable:
            continue
        values = [value for value in replicates[key] if math.isfinite(value)]
        if len(values) < 2:
            unqualifiable[key] = (
                "fewer than two resamples produced this quantity, so there is "
                "no spread to read"
            )
            continue
        dispersion = statistics.stdev(values)
        if dispersion == 0.0:
            # Not a measured absence of noise. The games were redrawn a
            # thousand times and this number did not move, which says the
            # resample could not move it rather than that a re-run would not:
            # a step function saturated at its bound, or a quantity computed
            # from a seat the fit pinned. A floor of zero here reads as perfect
            # resolution and licenses every delta, which is the failure a floor
            # exists to prevent. The genuine zero is stated, not estimated, and
            # comes out of the branch above.
            unqualifiable[key] = (
                "every resample returned the same value, so the redraw could "
                "not move it and there is no spread to bound; a stated spread "
                "of zero is a different claim and is reported as one"
            )
            continue
        dispersions[key] = measured_dispersion(
            dispersion,
            kind=LADDER_FLOOR_KIND,
            # The games are the independent replicates, and the resample count
            # is not — but a spread read off three surviving refits is known no
            # better than three numbers allow, whichever is the scarcer of the
            # two.
            degrees_of_freedom=min(redrawn_games, len(values)) - 1,
            confidence=settings.confidence,
            source=source,
            estimator=LADDER_BOOTSTRAP_METHOD,
        )
    return LadderResolution(
        dispersions=dispersions,
        unqualifiable=unqualifiable,
        method=LADDER_BOOTSTRAP_METHOD,
        confidence=settings.confidence,
        resamples=settings.resamples,
        fitted_resamples=fitted,
        redrawn_games=redrawn_games,
        replayed_pairings=replayed,
        non_convergent_resamples=non_convergent,
    )


def _floor_source(method: str) -> str:
    """Return what a stored floor names as the thing that produced it."""

    return f"{LADDER_BENCHMARK.name} v{LADDER_BENCHMARK_VERSION} {method}"


def _seat_totals(
    seats: Sequence[SeatKey],
    pairings: Sequence[LadderPairing],
) -> dict[SeatKey, tuple[int, float, int]]:
    """Return each seat's scored games, points, and games played.

    One pass over the pairings rather than one per quantity per seat. This runs
    once per resample, where a scan per seat is quadratic in the grid on the
    hottest loop the benchmark has, and it is the only place the three are
    counted, so the number a floor describes and the number it sits beside come
    from one reduction.
    """

    totals = {seat: (0, 0.0, 0) for seat in seats}
    for pairing in pairings:
        for seat, points in (
            (pairing.first, pairing.first_points),
            (pairing.second, pairing.games - pairing.first_points),
        ):
            total = totals.get(seat)
            if total is None:
                continue
            totals[seat] = (
                total[0] + pairing.games,
                total[1] + points,
                total[2] + pairing.played,
            )
    return totals


def _unbounded_seats(
    seats: Sequence[SeatKey],
    pairings: Sequence[LadderPairing],
) -> dict[tuple[str, str], str]:
    """Return the seats whose fitted rating is plainly a bound, not an estimate.

    A seat that scored nothing or scored everything has no finite
    maximum-likelihood rating, so the fit reports the declared spread instead.
    Read from the seat's own record rather than from ``RatingFit.clamped``,
    which also fires where the spread merely binds on an ordinary record — that
    seat's rating moves under resampling and is qualified like any other.

    A seat sweeping on its own is the case this catches. A *group* that jointly
    beats everything outside it is unbounded in the same way and is not named
    here, because finding one is a connectivity question over the whole
    comparison graph rather than a fact about a seat. Such a rating does not
    move under resampling either, so it is caught by the spread instead, one
    step later and with a vaguer reason.
    """

    unqualifiable: dict[tuple[str, str], str] = {}
    for seat, (games, points, _) in _seat_totals(seats, pairings).items():
        if not games or 0.0 < points < games:
            continue
        unqualifiable[(seat.label, LADDER_FITTED_RATING.identifier)] = (
            "the seat scored nothing or scored everything, so it has no finite "
            "maximum-likelihood rating and the fit reports the declared spread; "
            "every resample of it sweeps the same way and reproduces the bound"
        )
    return unqualifiable


def _quantities(
    config: LadderBenchmarkConfig,
    seats: Sequence[SeatKey],
    pairings: Sequence[LadderPairing],
    fit: RatingFit,
    workload: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[tuple[str, str], float]:
    """Return every quantity one fit yields, keyed by scope and metric.

    The reading's own reductions rather than a second copy of them, so a floor
    cannot end up describing a slightly different quantity from the number it
    sits beside.
    """

    values: dict[tuple[str, str], float] = {}
    totals = _seat_totals(seats, pairings)
    for seat in seats:
        rating = fit.rating(seat)
        if rating is not None:
            values[(seat.label, LADDER_FITTED_RATING.identifier)] = rating
        games, points, played = totals[seat]
        if games:
            values[(seat.label, LADDER_SCORE_RATE.identifier)] = points / games
        if played:
            values[(seat.label, LADDER_SCORED_GAME_RATE.identifier)] = games / played
    for reading in _readings(config, fit, workload, device=device, unavailable={}):
        scope = reading.label
        values[(scope, LADDER_RATING_ORDER_ACCURACY.identifier)] = (
            reading.order_accuracy
        )
        values[(scope, LADDER_ADJACENT_RATING_ORDER_ACCURACY.identifier)] = (
            reading.adjacent_order_accuracy
        )
        values[(scope, LADDER_RATING_ERROR.identifier)] = reading.ladder_error
        values[(scope, LADDER_FITTED_RATING_SLOPE.identifier)] = reading.slope
        values[(scope, LADDER_FITTED_RATING_SPAN.identifier)] = reading.span
    response = _temperature_response(
        config,
        fit,
        workload,
        device=device,
        unavailable={},
    )
    if response is not None:
        values[(RESPONSE_SCOPE, LADDER_TEMPERATURE_RESPONSE.identifier)] = (
            response.conditioned_response
        )
        if response.ablated_response is not None:
            values[(RESPONSE_SCOPE, LADDER_ABLATED_TEMPERATURE_RESPONSE.identifier)] = (
                response.ablated_response
            )
        if response.attenuation is not None:
            values[
                (RESPONSE_SCOPE, LADDER_TEMPERATURE_RESPONSE_ATTENUATION.identifier)
            ] = response.attenuation
    return values


def _resample_outcomes(
    pairing: LadderPairing,
    generator: np.random.Generator,
    resamples: int,
) -> np.ndarray:
    """Return one pairing's redrawn outcome counts, one row per resample.

    A multinomial rather than an index draw. Resampling a pairing's games with
    replacement and counting how each one ended is the same distribution, and
    this way one pairing costs one draw rather than one per resample.
    """

    played = pairing.played
    return generator.multinomial(
        played,
        (
            pairing.first_wins / played,
            pairing.draws / played,
            pairing.first_losses / played,
            pairing.unfinished / played,
        ),
        size=resamples,
    )


def _resampled_pairing(pairing: LadderPairing, counts: np.ndarray) -> LadderPairing:
    """Return one pairing as a resample drew it."""

    return LadderPairing(
        first=pairing.first,
        second=pairing.second,
        first_wins=int(counts[0]),
        draws=int(counts[1]),
        first_losses=int(counts[2]),
        unfinished=int(counts[3]),
        seeds=pairing.seeds,
    )


def _fit(
    config: LadderBenchmarkConfig,
    seats: Sequence[SeatKey],
    pairings: Sequence[LadderPairing],
) -> RatingFit:
    """Fit one ladder under this configuration's declared fit controls."""

    return fit_ratings(
        seats,
        pairings,
        anchor_seats=_anchor_seats(config),
        anchor_rating=_mean([float(value) for value in config.grid.target_ratings]),
        maximum_iterations=config.fit.maximum_iterations,
        tolerance=config.fit.tolerance,
        maximum_spread=config.fit.maximum_spread,
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
    pool = _load_pool(
        config.openings.pool, config.openings.expected_pool_game_ids_sha256
    )
    view_config = config.openings.view.model_copy(
        update={"prefix_plies": config.openings.plies}
    )
    selection = apply_view(pool.games, view_config)
    if not selection.game_ids:
        raise LadderBenchmarkError(
            f"view {view_config.name!r} selected no opening games from the pool"
        )
    rows = [
        _prefix_row(row, config.openings.plies)
        for row in pool_rows(
            pool,
            selection.game_ids,
            _OPENING_COLUMNS,
            error=LadderBenchmarkError,
        )
    ]
    games = sorted(
        (
            row_game_id(row),
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


def _load_pool(path: Path, expected_game_ids_sha256: str | None) -> FrozenPool:
    """Load the frozen pool the ladder's openings are projected out of."""

    try:
        return load_pool(path, expected_game_ids_sha256=expected_game_ids_sha256)
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
    recording: ResultRecording,
) -> None:
    """Write one envelope per seat, per temperature row, and per response.

    Three units rather than one record, because they are scoped differently. A
    seat carries what one configuration played like, a row carries the transfer
    function at one temperature, and the response spans the whole grid. Each
    declares its own workload and so owns its own series.
    """

    recorder = recording.measuring(
        result.checkpoint,
        kind=LADDER_KIND,
        benchmark=LADDER_BENCHMARK,
    )
    for seat in result.seats:
        measurements = _seat_measurements(seat, result.resolution)
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
            _reading_measurements(reading, result.resolution),
            payload=partial(_reading_payload, result, reading),
            description=f"Rating ladder: {reading.label}",
            slug=f"ladder-t{_slug(f'{reading.temperature:g}')}",
            data=result.dataset,
            execution=reading.execution,
        )
    if result.response is not None:
        recorder.add(
            _response_measurements(result.response, result.resolution),
            payload=result.response.as_record,
            description="Rating-ladder temperature response",
            slug="temperature-response",
            data=result.dataset,
            execution=result.response.execution,
        )


def _seat_measurements(
    seat: LadderSeat,
    resolution: LadderResolution | None,
) -> tuple[Measurement, ...]:
    """Return one seat's committed strength reading and error profile.

    A seat the fit could not place reports nothing rather than a zero, and the
    seat's own record says why. The error profile is reported beside strength
    rather than in place of it: a temperature that preserves average score while
    moving the profile has changed the shape of the mistakes, which no strength
    number can show.

    The error profile carries no spread. It is a mean over decisions rather than
    an output of the fit, so the refit the others come from does not reach it,
    and a delta in one reports its noise as unknown — which is what a floor
    somebody could still produce should read as.
    """

    workload = seat.execution.workload_component()
    values: list[tuple[str, float, int]] = []
    if seat.fitted_rating is not None:
        values.append((LADDER_FITTED_RATING.identifier, seat.fitted_rating, seat.games))
    if seat.games:
        values.append((LADDER_SCORE_RATE.identifier, seat.score_rate, seat.games))
    if seat.played:
        values.append(
            (LADDER_SCORED_GAME_RATE.identifier, seat.scored_game_rate, seat.played)
        )
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
    return _measurements(seat.label, values, workload, resolution)


def _reading_measurements(
    reading: LadderReading,
    resolution: LadderResolution | None,
) -> tuple[Measurement, ...]:
    """Return one temperature row's committed transfer-function reading."""

    workload = reading.execution.workload_component()
    seats = len(reading.ratings)
    values: list[tuple[str, float, int]] = [
        (LADDER_RATING_ORDER_ACCURACY.identifier, reading.order_accuracy, seats),
        (
            LADDER_ADJACENT_RATING_ORDER_ACCURACY.identifier,
            reading.adjacent_order_accuracy,
            seats,
        ),
        (LADDER_RATING_ERROR.identifier, reading.ladder_error, seats),
        (LADDER_FITTED_RATING_SLOPE.identifier, reading.slope, seats),
        (LADDER_FITTED_RATING_SPAN.identifier, reading.span, seats),
    ]
    return _measurements(reading.label, values, workload, resolution)


def _response_measurements(
    response: TemperatureResponse,
    resolution: LadderResolution | None,
) -> tuple[Measurement, ...]:
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
    return _measurements(RESPONSE_SCOPE, values, workload, resolution)


def _measurements(
    scope: str,
    values: Sequence[tuple[str, float, int]],
    workload: WorkloadComponent,
    resolution: LadderResolution | None,
) -> tuple[Measurement, ...]:
    """Return one unit's measurements, each beside what the reading can resolve.

    The spread travels on the measurement rather than being characterized
    against the series. Sample size is deliberately outside a ladder's identity
    — more seeds estimate the same ladder more precisely — so a spread filed
    against the series would be looked up later beside a reading taken at a
    different size, and be wrong by whatever the two sizes differ by.
    """

    return tuple(
        measurement(
            identifier,
            value,
            workload=workload,
            sample_size=sample_size,
            dispersion=(
                None if resolution is None else resolution.dispersion(scope, identifier)
            ),
        )
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
    payload["resolution"] = (
        None if result.resolution is None else result.resolution.as_record()
    )
    return payload


def _held_white(record: GameRecord, seat: SeatKey) -> bool:
    """Return whether one seat configuration held white in one recorded game."""

    configuration = record.seat("white").configuration
    return (
        configuration.get("target_rating") == seat.target_rating
        and configuration.get("temperature") == seat.temperature
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
    resolution = result.resolution
    if resolution is not None:
        logger.info(
            "Resolution: %s over %s redrawn game(s), %s replayed pairing(s); "
            "%s quantity(s) qualified, %s unqualifiable",
            resolution.method,
            resolution.redrawn_games,
            resolution.replayed_pairings,
            len(resolution.dispersions),
            len(resolution.unqualifiable),
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
    "LADDER_BOOTSTRAP_METHOD",
    "LADDER_DETERMINISTIC_METHOD",
    "LADDER_UNRESOLVED_METHOD",
    "LADDER_FLOOR_KIND",
    "LADDER_KIND",
    "RATING_SCALE",
    "RESPONSE_SCOPE",
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
    "LadderResolution",
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
