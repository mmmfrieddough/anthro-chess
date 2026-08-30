"""Dependency tests for the rating conditioning input.

Score the same held-out positions under true and corrupted conditioning, and
see whether the prediction gets worse. The ladder and puzzle families would
also catch a rating input the model ignores; what this one adds is that it
changes the input and reads the same move at the same position, with no
sampling or game dynamics in between, cheaply enough to run on any checkpoint.

Three forms are computed, each answering more than the last:

- **corruption** shows sensitivity. A conditioning input the model uses should
  predict worse when the value is shuffled or removed. Absence is the weaker
  of the two here: the corpus rates every game, so the model's rating-absent
  embedding is untrained and the treatment reads as an out-of-distribution
  probe rather than as a measure of reliance on the value.
- **cross-conditioning** shows direction. Scoring every rating slice under
  every conditioning value should put each slice's best result on the matching
  pair, which separates a model that reacts to the input from one that learned
  its meaning. Reported both as the fraction of slices that match, which
  saturates, and as what a position pays for being scored outside its own
  band, which does not.
- **within-game response** shows whether rating is tracked or treated as a
  static prior, by asking whether the policy at a fixed stated rating leans
  toward the strong-conditioned policy when the play so far has looked strong.

Results are degradations, not verdicts. An undertrained checkpoint can show
weak dependency because it has not learned the conditioning yet, so every
result carries the training maturity it was measured at, and nothing here
returns a pass or a fail.

This module is deliberately free of torch: it consumes per-position scores and
scalar trajectory signals that the scoring session already computed.
"""

from __future__ import annotations

import logging
from array import array
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean, median

from pydantic import Field, StrictBool, model_validator

from anthro_chess.config import ConfigModel
from anthro_chess.evaluation.noise import GameTotals, MetricTotal
from anthro_chess.evaluation.policy import (
    PositionColumns,
    TrajectoryColumns,
)
from anthro_chess.evaluation.results.metrics import (
    DEPENDENCY_RATING_ABSENT_DEGRADATION,
    DEPENDENCY_RATING_ANCHOR_POLICY_DIVERGENCE,
    DEPENDENCY_RATING_ANCHOR_TOP1_AGREEMENT,
    DEPENDENCY_RATING_CROSS_CONDITIONING_PENALTY,
    DEPENDENCY_RATING_SHUFFLED_DEGRADATION,
)
from anthro_chess.evaluation.slices import (
    DEFAULT_RATING_BANDS,
    RatingBand,
    rating_band_name,
)

#: Version 2 reads ``maturity`` off the evaluated checkpoint. Version 1 records
#: carry the position count their run finished on for every checkpoint in it, so
#: the two are not comparable per position.
DEPENDENCY_TEST_VERSION = 2

#: Key identifying one scored position across conditioning passes.
PositionKey = tuple[int, int]

WEAKER_PREFIX_GROUP = "weaker_prefix"
STRONGER_PREFIX_GROUP = "stronger_prefix"

logger = logging.getLogger(__name__)


class DependencyError(ValueError):
    """Raised when a dependency test cannot be computed from its inputs."""


class ConditioningKind(StrEnum):
    """How one scoring pass supplied the rating conditioning input."""

    TRUE = "true"
    SHUFFLED = "shuffled"
    CONSTANT = "constant"
    ABSENT = "absent"


#: ``TRUE`` has no entry: it is the baseline the others degrade from, and
#: ``CONSTANT`` none because the cross-conditioning table already scores every
#: position at each fixed rating, so a dedicated pass would buy one point of a
#: curve the reading holds in full.
DEGRADATION_METRICS = {
    ConditioningKind.SHUFFLED: DEPENDENCY_RATING_SHUFFLED_DEGRADATION,
    ConditioningKind.ABSENT: DEPENDENCY_RATING_ABSENT_DEGRADATION,
}


@dataclass(frozen=True)
class Conditioning:
    """One conditioning treatment applied to a scoring pass."""

    name: str
    kind: ConditioningKind
    rating: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a conditioning treatment needs a name")
        if self.kind is ConditioningKind.CONSTANT and self.rating is None:
            raise ValueError("constant conditioning needs an explicit rating")
        if self.kind is not ConditioningKind.CONSTANT and self.rating is not None:
            raise ValueError(f"{self.kind} conditioning does not take a rating")

    def as_record(self) -> dict[str, object]:
        """Return the stable record stored with a dependency result."""

        return {"name": self.name, "kind": str(self.kind), "rating": self.rating}


class DependencyTestConfig(ConfigModel):
    """Code-owned schema for the rating dependency-test mode."""

    enabled: StrictBool = True
    cross_conditioning_ratings: tuple[int, ...] = (1000, 1400, 1800, 2200)
    shuffle_seed: str = Field(default="anthro-dependency-shuffle-v1", min_length=1)
    #: Slices thinner than this are reported but excluded from the headline
    #: cross-conditioning and within-game numbers, where a handful of positions
    #: would produce a confident-looking accident.
    minimum_slice_positions: int = Field(default=50, ge=1)
    #: How many earlier decisions by the same player a position needs before
    #: its prefix carries a usable strength signal.
    minimum_prefix_decisions: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _validate_conditioning_ratings(self) -> DependencyTestConfig:
        # Counted after deduplication, because it is distinct ratings both
        # readers need: a grid naming one rating twice would put every slice's
        # best result on the only column there is, and would collapse the
        # anchor comparison onto a distribution and itself.
        if len(set(self.cross_conditioning_ratings)) < 2:
            raise ValueError(
                "cross-conditioning needs at least two distinct conditioning ratings"
            )
        return self

    def conditioning_values(self) -> tuple[int, ...]:
        """Return the deduplicated conditioning ratings, in ascending order."""

        return tuple(sorted(set(self.cross_conditioning_ratings)))


@dataclass(frozen=True)
class PositionContext:
    """The true, uncorrupted facts about one scored position."""

    game_id: int
    ply_index: int
    color: str
    rating: int | None
    rating_band: str | None

    @property
    def key(self) -> PositionKey:
        """Return this position's identity across conditioning passes."""

        return (self.game_id, self.ply_index)


@dataclass(frozen=True)
class MaturityContext:
    """How far the evaluated checkpoint had trained when it was measured.

    Both coordinates come from the evaluated checkpoint rather than from its
    run, so an intermediate checkpoint reports the maturity it was saved at
    instead of inheriting the count its run finished on.
    """

    step: int
    processed_positions: int

    def as_record(self) -> dict[str, object]:
        """Return the record every dependency result is read against."""

        return {
            "step": self.step,
            "processed_positions": self.processed_positions,
        }


@dataclass(frozen=True)
class CorruptionResult:
    """How much one corrupted conditioning treatment costs in move loss."""

    conditioning: Conditioning
    position_count: int
    move_loss: float
    degradation: float

    def as_record(self) -> dict[str, object]:
        """Return the stable JSON-serializable record."""

        return {
            "conditioning": self.conditioning.as_record(),
            "position_count": self.position_count,
            "move_loss": self.move_loss,
            "degradation": self.degradation,
        }


@dataclass(frozen=True)
class CrossConditioningCell:
    """One true rating slice scored under one conditioning value."""

    rating_band: str
    conditioning_rating: int
    position_count: int
    move_loss: float


@dataclass(frozen=True)
class CrossConditioningResult:
    """The full slice-by-conditioning table and what its diagonal says."""

    cells: tuple[CrossConditioningCell, ...]
    compared_bands: tuple[str, ...]
    matched_bands: tuple[str, ...]
    excluded_bands: tuple[str, ...]
    #: Mean extra loss a position pays under a conditioning outside its own
    #: band. The match rate below reads one on any checkpoint that has learned
    #: the ordering at all, so it reports a regression and never progress;
    #: this is the same comparison left graded.
    penalty: float | None

    @property
    def match_rate(self) -> float | None:
        """Return the fraction of slices whose best value is the matching one."""

        if not self.compared_bands:
            return None
        return len(self.matched_bands) / len(self.compared_bands)

    def as_record(self) -> dict[str, object]:
        """Return the stable JSON-serializable record."""

        return {
            "penalty": self.penalty,
            "match_rate": self.match_rate,
            "compared_bands": list(self.compared_bands),
            "matched_bands": list(self.matched_bands),
            "excluded_bands": list(self.excluded_bands),
            "cells": [
                {
                    "rating_band": cell.rating_band,
                    "conditioning_rating": cell.conditioning_rating,
                    "position_count": cell.position_count,
                    "move_loss": cell.move_loss,
                }
                for cell in self.cells
            ],
        }


@dataclass(frozen=True)
class WithinGameGroup:
    """One prefix-strength half of one rating slice."""

    rating_band: str
    group: str
    position_count: int
    mean_prefix_strength: float
    mean_alignment: float
    move_loss: float


@dataclass(frozen=True)
class WithinGameResult:
    """Whether the policy tracks the trajectory at a fixed stated rating."""

    groups: tuple[WithinGameGroup, ...]
    response: float | None
    compared_bands: tuple[str, ...]
    excluded_bands: tuple[str, ...]
    positions_with_prefix: int
    anchor_low_rating: int
    anchor_high_rating: int

    def as_record(self) -> dict[str, object]:
        """Return the stable JSON-serializable record."""

        return {
            "response": self.response,
            "compared_bands": list(self.compared_bands),
            "excluded_bands": list(self.excluded_bands),
            "positions_with_prefix": self.positions_with_prefix,
            "anchor_low_rating": self.anchor_low_rating,
            "anchor_high_rating": self.anchor_high_rating,
            "groups": [
                {
                    "rating_band": group.rating_band,
                    "group": group.group,
                    "position_count": group.position_count,
                    "mean_prefix_strength": group.mean_prefix_strength,
                    "mean_alignment": group.mean_alignment,
                    "move_loss": group.move_loss,
                }
                for group in self.groups
            ],
        }


@dataclass(frozen=True)
class DependencyTestResult:
    """Everything the rating dependency-test mode measured."""

    rated_position_count: int
    true_move_loss: float
    corruptions: tuple[CorruptionResult, ...]
    cross_conditioning: CrossConditioningResult
    within_game: WithinGameResult
    anchor_divergence: float
    anchor_agreement_rate: float
    maturity: MaturityContext
    #: Per-game shares of the quantities this reading can resample, which is
    #: what its own dispersion is estimated from.
    per_game_totals: tuple[GameTotals, ...]

    def corruption(self, kind: ConditioningKind) -> CorruptionResult | None:
        """Return one corruption result by treatment kind."""

        for result in self.corruptions:
            if result.conditioning.kind is kind:
                return result
        return None

    def as_record(self) -> dict[str, object]:
        """Return the versioned structured dependency-test record."""

        return {
            "version": DEPENDENCY_TEST_VERSION,
            "rated_position_count": self.rated_position_count,
            "true_move_loss": self.true_move_loss,
            "maturity": self.maturity.as_record(),
            "interpretation": (
                "Degradations are measurements, not gates. A weak result on an "
                "undertrained checkpoint means the conditioning has not been "
                "learned yet, which is not the same as a miswired input."
            ),
            "corruptions": [item.as_record() for item in self.corruptions],
            "cross_conditioning": self.cross_conditioning.as_record(),
            "within_game": self.within_game.as_record(),
            "anchors": {
                "policy_divergence": self.anchor_divergence,
                "top1_agreement_rate": self.anchor_agreement_rate,
            },
        }


@dataclass(frozen=True)
class DependencyColumns:
    """Every per-position quantity the dependency reductions read, as columns.

    This family is the one part of a checkpoint reading that no running total
    finishes: the within-game split takes a median of each slice, and an
    accumulator recovers no median. So a pool-scale pass holds an entry per
    scored position, and what decides whether that is affordable is the shape
    it is held in rather than the fact of holding it.

    Columns are filled in scoring order, so a reduction sums the same values in
    the same sequence and lands on the same float.
    """

    game_ids: Sequence[int]
    ply_indices: Sequence[int]
    #: An opaque per-position grouping key, so a prefix is read over one
    #: player's own earlier decisions rather than over the game's.
    colors: Sequence[int]
    band_names: tuple[str, ...]
    #: An index into ``band_names``, or negative where the position has no band.
    bands: Sequence[int]
    rated: Sequence[int]
    true_move_nll: Sequence[float]
    corrupted: Mapping[str, tuple[Conditioning, Sequence[float]]]
    conditioned: Mapping[int, Sequence[float]]
    has_signal: Sequence[int]
    strength_signal: Sequence[float]
    alignment: Sequence[float]
    anchor_divergence: Sequence[float]
    anchor_agreement: Sequence[int]

    def __post_init__(self) -> None:
        widths = {
            len(self.game_ids),
            len(self.ply_indices),
            len(self.colors),
            len(self.bands),
            len(self.rated),
            len(self.true_move_nll),
            len(self.has_signal),
            len(self.strength_signal),
            len(self.alignment),
            len(self.anchor_divergence),
            len(self.anchor_agreement),
            *(len(values) for _, values in self.corrupted.values()),
            *(len(values) for values in self.conditioned.values()),
        }
        if len(widths) > 1:
            raise DependencyError(
                "dependency columns must all cover the same scored positions"
            )

    @property
    def positions(self) -> int:
        """Return how many scored positions these columns describe."""

        return len(self.game_ids)

    def band(self, index: int) -> str | None:
        """Return one position's rating band, absent where it has none."""

        offset = self.bands[index]
        return None if offset < 0 else self.band_names[offset]

    def rated_indices(self) -> tuple[int, ...]:
        """Return the positions a dependency test may compare, in pass order.

        Only positions whose true rating is present take part. Comparing a
        rated position against an unrated one would mix the effect of
        corrupting the input with the effect of its being absent to begin with.
        """

        return tuple(index for index in range(self.positions) if self.rated[index])


def _signal_columns(
    trajectory: TrajectoryColumns | None,
    scored: int,
) -> tuple[list[float], list[float], list[float], list[bool]]:
    """Return the four trajectory columns as lists, or zeros where absent."""

    if trajectory is None:
        return [0.0] * scored, [0.0] * scored, [0.0] * scored, [False] * scored
    if len(trajectory.strength_signal) != scored:
        raise DependencyError(
            "the trajectory columns do not cover every scored position"
        )
    return (
        list(trajectory.strength_signal),
        list(trajectory.alignment),
        list(trajectory.anchor_divergence),
        list(trajectory.anchor_agreement),
    )


class DependencyColumnBuilder:
    """Fills :class:`DependencyColumns` from scored conditioning passes.

    Takes one call per scored batch, or one call for a whole view. Which it is
    does not change the columns: the treatments of a batch are aligned to its
    true-conditioning pass by position, so a caller that hands over a batch at a
    time builds the same record as one holding everything at once.
    """

    def __init__(self) -> None:
        # Unsigned: a game id is a 64-bit hash, and a signed column wraps every
        # id past the signed maximum onto one that matches no game.
        self._game_ids: array[int] = array("Q")
        self._ply_indices: array[int] = array("i")
        self._colors = _Interner()
        self._bands = _Interner()
        self._rated = bytearray()
        self._true: array[float] = array("d")
        self._corrupted: dict[str, tuple[Conditioning, array[float]]] = {}
        self._conditioned: dict[int, array[float]] = {}
        self._has_signal = bytearray()
        self._strength: array[float] = array("d")
        self._alignment: array[float] = array("d")
        self._divergence: array[float] = array("d")
        self._agreement = bytearray()

    def add(
        self,
        contexts: Mapping[PositionKey, PositionContext],
        true_positions: PositionColumns,
        corrupted_losses: Mapping[str, tuple[Conditioning, Sequence[float]]],
        conditioned_losses: Mapping[int, Sequence[float]],
        trajectory: TrajectoryColumns | None,
    ) -> None:
        """Append one pass over some positions to every column.

        A treatment is aligned to the true pass by position rather than joined
        to it: a conditioning changes the rating the model saw and nothing
        about which rows a batch enables, so its losses arrive in the order
        this pass scored.
        """

        strength, alignment, divergence, agreement = _signal_columns(
            trajectory, len(true_positions)
        )
        losses = true_positions.move_nll.tolist()
        keys = zip(true_positions.game_ids, true_positions.ply_indices, strict=True)
        for offset, key in enumerate(keys):
            context = contexts.get(key)
            if context is None:
                raise DependencyError(
                    f"scored position {key} has no recorded true context"
                )
            self._game_ids.append(key[0])
            self._ply_indices.append(key[1])
            self._colors.append(context.color)
            self._bands.append(context.rating_band)
            self._rated.append(context.rating is not None)
            self._true.append(losses[offset])
            self._has_signal.append(trajectory is not None)
            self._strength.append(strength[offset])
            self._alignment.append(alignment[offset])
            self._divergence.append(divergence[offset])
            self._agreement.append(agreement[offset])

        for name, (conditioning, losses) in corrupted_losses.items():
            self._extend(
                self._corrupted.setdefault(name, (conditioning, array("d")))[1],
                losses,
                len(true_positions),
                f"conditioning pass {name!r}",
            )
        for rating, losses in conditioned_losses.items():
            self._extend(
                self._conditioned.setdefault(rating, array("d")),
                losses,
                len(true_positions),
                f"conditioning rating {rating}",
            )

    def build(self) -> DependencyColumns:
        """Return the filled columns every dependency reduction reads."""

        return DependencyColumns(
            game_ids=self._game_ids,
            ply_indices=self._ply_indices,
            band_names=self._bands.names(),
            bands=self._bands.offsets,
            colors=self._colors.offsets,
            rated=self._rated,
            true_move_nll=self._true,
            corrupted=dict(self._corrupted),
            conditioned=dict(self._conditioned),
            has_signal=self._has_signal,
            strength_signal=self._strength,
            alignment=self._alignment,
            anchor_divergence=self._divergence,
            anchor_agreement=self._agreement,
        )

    def _extend(
        self,
        column: array[float],
        losses: Sequence[float],
        scored: int,
        what: str,
    ) -> None:
        if len(losses) != scored:
            raise DependencyError(
                f"{what} scored {len(losses)} of {scored} scored position(s)"
            )
        column.extend(losses)


class _Interner:
    """A column of repeated names, kept as offsets into the names themselves."""

    def __init__(self) -> None:
        self.offsets: array[int] = array("h")
        self._names: dict[str, int] = {}

    def append(self, name: str | None) -> None:
        """Append one value, recording an absent one as a negative offset."""

        if name is None:
            self.offsets.append(-1)
            return
        offset = self._names.get(name)
        if offset is None:
            offset = len(self._names)
            self._names[name] = offset
        self.offsets.append(offset)

    def names(self) -> tuple[str, ...]:
        """Return the distinct names, in the order they were first seen."""

        return tuple(self._names)


def reduce_dependency_columns(
    *,
    config: DependencyTestConfig,
    columns: DependencyColumns,
    maturity: MaturityContext,
    rating_bands: Sequence[RatingBand] = DEFAULT_RATING_BANDS,
) -> DependencyTestResult:
    """Assemble every dependency test from one pass' aligned columns."""

    rated = columns.rated_indices()
    if not rated:
        raise DependencyError(
            "dependency tests need held-out positions with a present rating"
        )
    true_move_loss = fmean(columns.true_move_nll[index] for index in rated)

    corruptions = tuple(
        _corruption_result(columns, name, rated, true_move_loss)
        for name in sorted(columns.corrupted)
    )
    penalties = _cross_penalties(config, columns, rating_bands)
    cross = _cross_conditioning(config, columns, rating_bands, penalties)
    values = config.conditioning_values()
    within = _within_game(
        config,
        columns,
        rated,
        rating_bands,
        anchor_low=values[0],
        anchor_high=values[-1],
    )
    signals = tuple(index for index in rated if columns.has_signal[index])
    if not signals:
        raise DependencyError(
            "dependency tests need anchor policies for scored positions"
        )
    return DependencyTestResult(
        rated_position_count=len(rated),
        true_move_loss=true_move_loss,
        corruptions=corruptions,
        cross_conditioning=cross,
        within_game=within,
        anchor_divergence=fmean(columns.anchor_divergence[index] for index in signals),
        anchor_agreement_rate=fmean(
            float(columns.anchor_agreement[index]) for index in signals
        ),
        maturity=maturity,
        per_game_totals=_per_game_totals(columns, rated, penalties),
    )


def _require_conditioning_passes(
    config: DependencyTestConfig,
    conditioned: Mapping[int, object],
) -> tuple[int, ...]:
    """Return the configured conditioning ratings, refusing an unscored one."""

    values = config.conditioning_values()
    missing = tuple(value for value in values if value not in conditioned)
    if missing:
        raise DependencyError(
            f"cross-conditioning is missing a scoring pass for rating {missing[0]}"
        )
    return values


def _per_game_totals(
    columns: DependencyColumns,
    rated: Sequence[int],
    penalties: Mapping[int, float],
) -> tuple[GameTotals, ...]:
    """Return each game's share of the position-mean dependency results.

    The game is the resampling unit, for the reason
    ``anthro_chess.evaluation.noise`` gives. Each quantity carries its own
    count, because they are means over different positions: the degradations
    over every rated one, the anchors over those a trajectory signal was
    computed for, and the cross-conditioning penalty over those whose band the
    grid names a value for.

    Two of the family's quantities are absent because no resampling of games
    could recompute them; each declares why in its
    ``no_sampling_floor_reason``.
    """

    positions: dict[int, int] = defaultdict(int)
    true_totals: dict[int, float] = defaultdict(float)
    anchor_positions: dict[int, int] = defaultdict(int)
    divergence: dict[int, float] = defaultdict(float)
    agreement: dict[int, float] = defaultdict(float)
    penalty_positions: dict[int, int] = defaultdict(int)
    penalty_totals: dict[int, float] = defaultdict(float)
    for index, penalty in penalties.items():
        game_id = columns.game_ids[index]
        penalty_positions[game_id] += 1
        penalty_totals[game_id] += penalty
    for index in rated:
        game_id = columns.game_ids[index]
        positions[game_id] += 1
        true_totals[game_id] += columns.true_move_nll[index]
        if not columns.has_signal[index]:
            continue
        anchor_positions[game_id] += 1
        divergence[game_id] += columns.anchor_divergence[index]
        agreement[game_id] += float(columns.anchor_agreement[index])

    # Name-sorted and first-wins, matching how a corruption result is selected
    # by kind, so a share is drawn from the pass whose value it qualifies.
    degradations: dict[ConditioningKind, dict[int, float]] = {}
    for name in sorted(columns.corrupted):
        conditioning, losses = columns.corrupted[name]
        if conditioning.kind in degradations:
            continue
        totals: dict[int, float] = defaultdict(float)
        for index in rated:
            totals[columns.game_ids[index]] += losses[index]
        degradations[conditioning.kind] = {
            game_id: total - true_totals[game_id] for game_id, total in totals.items()
        }

    # A game's rated count is also the count behind its degradation share:
    # ``_corruption_result`` refuses a pass that does not cover every rated
    # position, so no treatment can be missing one of these games.
    return tuple(
        GameTotals(
            game_id=game_id,
            metrics={
                **{
                    DEGRADATION_METRICS[kind].identifier: MetricTotal(
                        total=totals[game_id],
                        positions=count,
                    )
                    for kind, totals in degradations.items()
                    if kind in DEGRADATION_METRICS
                },
                DEPENDENCY_RATING_ANCHOR_POLICY_DIVERGENCE.identifier: MetricTotal(
                    total=divergence[game_id],
                    positions=anchor_positions[game_id],
                ),
                DEPENDENCY_RATING_ANCHOR_TOP1_AGREEMENT.identifier: MetricTotal(
                    total=agreement[game_id],
                    positions=anchor_positions[game_id],
                ),
                DEPENDENCY_RATING_CROSS_CONDITIONING_PENALTY.identifier: MetricTotal(
                    total=penalty_totals[game_id],
                    positions=penalty_positions[game_id],
                ),
            },
        )
        for game_id, count in sorted(positions.items())
    )


def _corruption_result(
    columns: DependencyColumns,
    name: str,
    rated: Sequence[int],
    true_move_loss: float,
) -> CorruptionResult:
    conditioning, losses = columns.corrupted[name]
    move_loss = fmean(losses[index] for index in rated)
    return CorruptionResult(
        conditioning=conditioning,
        position_count=len(rated),
        move_loss=move_loss,
        degradation=move_loss - true_move_loss,
    )


def _band_conditioning(
    values: Sequence[int],
    rating_bands: Sequence[RatingBand],
) -> dict[str, int]:
    """Return the conditioning value that belongs to each band the grid names.

    A grid naming two values inside one band would leave that band without a
    single value of its own, so the first wins and the second is read as an
    away value like any other.
    """

    matching: dict[str, int] = {}
    for value in values:
        name = rating_band_name(value, rating_bands)
        if name is not None:
            matching.setdefault(name, value)
    return matching


def _cross_penalties(
    config: DependencyTestConfig,
    columns: DependencyColumns,
    rating_bands: Sequence[RatingBand],
) -> dict[int, float]:
    """Return what each position pays for being conditioned outside its band.

    Held per position rather than per band, because this is the one
    cross-conditioning quantity a game carries a share of, and a floor needs
    that share. A band the grid names no value for contributes nothing: there
    is no matching column to price the others against.
    """

    values = _require_conditioning_passes(config, columns.conditioned)
    matching = _band_conditioning(values, rating_bands)
    penalties: dict[int, float] = {}
    for index in range(columns.positions):
        band = columns.band(index)
        own = None if band is None else matching.get(band)
        if own is None:
            continue
        away = [value for value in values if value != own]
        if not away:
            continue
        base = columns.conditioned[own][index]
        penalties[index] = fmean(
            columns.conditioned[value][index] - base for value in away
        )
    return penalties


def _cross_conditioning(
    config: DependencyTestConfig,
    columns: DependencyColumns,
    rating_bands: Sequence[RatingBand],
    penalties: Mapping[int, float],
) -> CrossConditioningResult:
    values = _require_conditioning_passes(config, columns.conditioned)

    band_losses: dict[str, dict[int, list[float]]] = {}
    for value in values:
        losses = columns.conditioned[value]
        for index in range(columns.positions):
            band = columns.band(index)
            if band is None:
                continue
            band_losses.setdefault(band, {}).setdefault(value, []).append(losses[index])

    cells: list[CrossConditioningCell] = []
    compared: list[str] = []
    matched: list[str] = []
    excluded: list[str] = []
    for definition in rating_bands:
        by_value = band_losses.get(definition.name)
        if by_value is None:
            continue
        for value in values:
            losses = by_value.get(value, [])
            if not losses:
                continue
            cells.append(
                CrossConditioningCell(
                    rating_band=definition.name,
                    conditioning_rating=value,
                    position_count=len(losses),
                    move_loss=fmean(losses),
                )
            )
        counts = {value: len(by_value.get(value, [])) for value in values}
        if min(counts.values()) < config.minimum_slice_positions:
            excluded.append(definition.name)
            continue
        best = min(values, key=lambda value: fmean(by_value[value]))
        compared.append(definition.name)
        if rating_band_name(best, rating_bands) == definition.name:
            matched.append(definition.name)

    return CrossConditioningResult(
        cells=tuple(cells),
        compared_bands=tuple(compared),
        matched_bands=tuple(matched),
        excluded_bands=tuple(excluded),
        penalty=fmean(penalties.values()) if penalties else None,
    )


def _within_game(
    config: DependencyTestConfig,
    columns: DependencyColumns,
    rated: Sequence[int],
    rating_bands: Sequence[RatingBand],
    *,
    anchor_low: int,
    anchor_high: int,
) -> WithinGameResult:
    prefixes = _prefix_strengths(config, columns, rated)
    by_band: dict[str, list[tuple[float, int]]] = {}
    for index in rated:
        strength = prefixes.get(index)
        band = columns.band(index)
        if strength is None or band is None:
            continue
        by_band.setdefault(band, []).append((strength, index))

    groups: list[WithinGameGroup] = []
    compared: list[str] = []
    excluded: list[str] = []
    weaker: list[float] = []
    stronger: list[float] = []
    for definition in rating_bands:
        band = definition.name
        entries = by_band.get(band)
        if entries is None:
            continue
        if len(entries) < 2 * config.minimum_slice_positions:
            excluded.append(band)
            continue
        threshold = median(strength for strength, _ in entries)
        halves = {
            WEAKER_PREFIX_GROUP: [entry for entry in entries if entry[0] <= threshold],
            STRONGER_PREFIX_GROUP: [entry for entry in entries if entry[0] > threshold],
        }
        if any(len(half) < config.minimum_slice_positions for half in halves.values()):
            excluded.append(band)
            continue
        compared.append(band)
        for name, half in halves.items():
            alignments = [columns.alignment[index] for _, index in half]
            groups.append(
                WithinGameGroup(
                    rating_band=band,
                    group=name,
                    position_count=len(half),
                    mean_prefix_strength=fmean(strength for strength, _ in half),
                    mean_alignment=fmean(alignments),
                    move_loss=fmean(columns.true_move_nll[index] for _, index in half),
                )
            )
            if name == WEAKER_PREFIX_GROUP:
                weaker.extend(alignments)
            else:
                stronger.extend(alignments)

    response = (
        fmean(stronger) - fmean(weaker) if compared and weaker and stronger else None
    )
    if response is None:
        logger.info(
            "Within-game rating response was not computed; no rating band held "
            "enough held-out prefixes"
        )
    return WithinGameResult(
        groups=tuple(groups),
        response=response,
        compared_bands=tuple(compared),
        excluded_bands=tuple(excluded),
        positions_with_prefix=len(prefixes),
        anchor_low_rating=anchor_low,
        anchor_high_rating=anchor_high,
    )


def _prefix_strengths(
    config: DependencyTestConfig,
    columns: DependencyColumns,
    rated: Sequence[int],
) -> dict[int, float]:
    """Return each position's mean prefix strength for the player to move.

    Only the same player's earlier decisions count. A player's own moves are
    what says whether *their* play has looked strong, and mixing in the
    opponent's would measure the game rather than the player.
    """

    by_player: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for index in range(columns.positions):
        if not columns.has_signal[index]:
            continue
        by_player.setdefault(
            (columns.game_ids[index], columns.colors[index]), []
        ).append((columns.ply_indices[index], columns.strength_signal[index]))
    for entries in by_player.values():
        entries.sort()

    strengths: dict[int, float] = {}
    for index in rated:
        entries = by_player.get((columns.game_ids[index], columns.colors[index]), [])
        ply_index = columns.ply_indices[index]
        earlier = [value for earlier_ply, value in entries if earlier_ply < ply_index]
        if len(earlier) < config.minimum_prefix_decisions:
            continue
        strengths[index] = fmean(earlier)
    return strengths
