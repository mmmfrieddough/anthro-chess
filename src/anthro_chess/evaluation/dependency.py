"""Dependency tests for the rating conditioning input.

Rating-sliced loss cannot show that a model reads the rating input. It
measures how hard each band's positions are to predict, and it looks
unremarkable on a model that ignores conditioning entirely. These tests answer
the different question: score the same held-out positions under true and
corrupted conditioning, and see whether the prediction gets worse.

Three forms are computed, each answering more than the last:

- **corruption** shows sensitivity. A conditioning input the model uses should
  predict worse when the value is shuffled, pinned to a constant, or removed.
- **cross-conditioning** shows direction. Scoring every rating slice under
  every conditioning value should put each slice's best result on the matching
  pair, which separates a model that reacts to the input from one that learned
  its meaning.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean, median

from pydantic import Field, StrictBool

from anthro_chess.config import ConfigModel
from anthro_chess.evaluation.policy import PositionPolicy
from anthro_chess.evaluation.slices import (
    DEFAULT_RATING_BANDS,
    RatingBand,
    rating_band_name,
)

DEPENDENCY_TEST_VERSION = 1

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
    constant_rating: int = Field(default=1500, ge=0)
    cross_conditioning_ratings: tuple[int, ...] = (1000, 1400, 1800, 2200)
    shuffle_seed: str = Field(default="anthro-dependency-shuffle-v1", min_length=1)
    #: Slices thinner than this are reported but excluded from the headline
    #: cross-conditioning and within-game numbers, where a handful of positions
    #: would produce a confident-looking accident.
    minimum_slice_positions: int = Field(default=50, ge=1)
    #: How many earlier decisions by the same player a position needs before
    #: its prefix carries a usable strength signal.
    minimum_prefix_decisions: int = Field(default=5, ge=1)

    def conditioning_values(self) -> tuple[int, ...]:
        """Return the deduplicated conditioning ratings, in ascending order."""

        if len(self.cross_conditioning_ratings) < 2:
            raise ValueError(
                "cross-conditioning needs at least two conditioning ratings"
            )
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
class TrajectorySignal:
    """What two anchor conditionings say about one position and its prefix.

    ``strength_signal`` is positive when the strong-rating conditioning
    explains the move actually played better than the weak one, which is the
    available proxy for how strong the play looked. ``alignment`` is positive
    when the policy at the position's true rating sits closer to the
    strong-conditioned policy than to the weak-conditioned one.
    """

    strength_signal: float
    alignment: float
    anchor_divergence: float
    anchor_agreement: bool


@dataclass(frozen=True)
class MaturityContext:
    """How far the evaluated checkpoint had trained when it was measured."""

    step: int | None
    processed_positions: int | None

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

    @property
    def match_rate(self) -> float | None:
        """Return the fraction of slices whose best value is the matching one."""

        if not self.compared_bands:
            return None
        return len(self.matched_bands) / len(self.compared_bands)

    def as_record(self) -> dict[str, object]:
        """Return the stable JSON-serializable record."""

        return {
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


def build_dependency_result(
    *,
    config: DependencyTestConfig,
    contexts: Mapping[PositionKey, PositionContext],
    true_positions: Sequence[PositionPolicy],
    corrupted_positions: Mapping[str, tuple[Conditioning, Sequence[PositionPolicy]]],
    conditioned_positions: Mapping[int, Sequence[PositionPolicy]],
    trajectory: Mapping[PositionKey, TrajectorySignal],
    maturity: MaturityContext,
    rating_bands: Sequence[RatingBand] = DEFAULT_RATING_BANDS,
) -> DependencyTestResult:
    """Assemble every dependency test from already-scored conditioning passes.

    Only positions whose true rating is present take part. Comparing a rated
    position against an unrated one would mix the effect of corrupting the
    input with the effect of the input being absent to begin with.
    """

    rated = tuple(
        position
        for position in true_positions
        if _context(contexts, position).rating is not None
    )
    if not rated:
        raise DependencyError(
            "dependency tests need held-out positions with a present rating"
        )
    rated_keys = {(position.game_id, position.ply_index) for position in rated}
    true_move_loss = fmean(position.move_nll for position in rated)

    corruptions = tuple(
        _corruption_result(conditioning, positions, rated_keys, true_move_loss)
        for conditioning, positions in (
            corrupted_positions[name] for name in sorted(corrupted_positions)
        )
    )
    cross = _cross_conditioning(
        config,
        contexts,
        conditioned_positions,
        rating_bands,
    )
    values = config.conditioning_values()
    within = _within_game(
        config,
        contexts,
        rated,
        trajectory,
        anchor_low=values[0],
        anchor_high=values[-1],
    )
    signals = tuple(trajectory[key] for key in sorted(rated_keys) if key in trajectory)
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
        anchor_divergence=fmean(signal.anchor_divergence for signal in signals),
        anchor_agreement_rate=fmean(
            float(signal.anchor_agreement) for signal in signals
        ),
        maturity=maturity,
    )


def _corruption_result(
    conditioning: Conditioning,
    positions: Sequence[PositionPolicy],
    rated_keys: set[PositionKey],
    true_move_loss: float,
) -> CorruptionResult:
    losses = [
        position.move_nll
        for position in positions
        if (position.game_id, position.ply_index) in rated_keys
    ]
    if len(losses) != len(rated_keys):
        raise DependencyError(
            f"conditioning pass {conditioning.name!r} scored "
            f"{len(losses)} of {len(rated_keys)} rated position(s)"
        )
    move_loss = fmean(losses)
    return CorruptionResult(
        conditioning=conditioning,
        position_count=len(losses),
        move_loss=move_loss,
        degradation=move_loss - true_move_loss,
    )


def _cross_conditioning(
    config: DependencyTestConfig,
    contexts: Mapping[PositionKey, PositionContext],
    conditioned_positions: Mapping[int, Sequence[PositionPolicy]],
    rating_bands: Sequence[RatingBand],
) -> CrossConditioningResult:
    values = config.conditioning_values()
    missing = tuple(value for value in values if value not in conditioned_positions)
    if missing:
        raise DependencyError(
            f"cross-conditioning is missing a scoring pass for rating {missing[0]}"
        )

    band_losses: dict[str, dict[int, list[float]]] = {}
    for value in values:
        for position in conditioned_positions[value]:
            context = _context(contexts, position)
            if context.rating_band is None:
                continue
            band_losses.setdefault(context.rating_band, {}).setdefault(
                value, []
            ).append(position.move_nll)

    cells: list[CrossConditioningCell] = []
    compared: list[str] = []
    matched: list[str] = []
    excluded: list[str] = []
    for band in rating_bands:
        by_value = band_losses.get(band.name)
        if by_value is None:
            continue
        for value in values:
            losses = by_value.get(value, [])
            if not losses:
                continue
            cells.append(
                CrossConditioningCell(
                    rating_band=band.name,
                    conditioning_rating=value,
                    position_count=len(losses),
                    move_loss=fmean(losses),
                )
            )
        counts = {value: len(by_value.get(value, [])) for value in values}
        if min(counts.values()) < config.minimum_slice_positions:
            excluded.append(band.name)
            continue
        best = min(values, key=lambda value: fmean(by_value[value]))
        compared.append(band.name)
        if rating_band_name(best, rating_bands) == band.name:
            matched.append(band.name)

    return CrossConditioningResult(
        cells=tuple(cells),
        compared_bands=tuple(compared),
        matched_bands=tuple(matched),
        excluded_bands=tuple(excluded),
    )


def _within_game(
    config: DependencyTestConfig,
    contexts: Mapping[PositionKey, PositionContext],
    rated: Sequence[PositionPolicy],
    trajectory: Mapping[PositionKey, TrajectorySignal],
    *,
    anchor_low: int,
    anchor_high: int,
) -> WithinGameResult:
    prefixes = _prefix_strengths(config, contexts, rated, trajectory)
    by_band: dict[str, list[tuple[float, PositionPolicy]]] = {}
    for position in rated:
        context = _context(contexts, position)
        strength = prefixes.get(context.key)
        if strength is None or context.rating_band is None:
            continue
        by_band.setdefault(context.rating_band, []).append((strength, position))

    groups: list[WithinGameGroup] = []
    compared: list[str] = []
    excluded: list[str] = []
    weaker: list[float] = []
    stronger: list[float] = []
    for band in sorted(by_band):
        entries = by_band[band]
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
            alignments = [
                trajectory[(position.game_id, position.ply_index)].alignment
                for _, position in half
            ]
            groups.append(
                WithinGameGroup(
                    rating_band=band,
                    group=name,
                    position_count=len(half),
                    mean_prefix_strength=fmean(strength for strength, _ in half),
                    mean_alignment=fmean(alignments),
                    move_loss=fmean(position.move_nll for _, position in half),
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
    contexts: Mapping[PositionKey, PositionContext],
    rated: Sequence[PositionPolicy],
    trajectory: Mapping[PositionKey, TrajectorySignal],
) -> dict[PositionKey, float]:
    """Return each position's mean prefix strength for the player to move.

    Only the same player's earlier decisions count. A player's own moves are
    what says whether *their* play has looked strong, and mixing in the
    opponent's would measure the game rather than the player.
    """

    by_player: dict[tuple[int, str], list[tuple[int, float]]] = {}
    for key, context in contexts.items():
        signal = trajectory.get(key)
        if signal is None:
            continue
        by_player.setdefault((context.game_id, context.color), []).append(
            (context.ply_index, signal.strength_signal)
        )
    for entries in by_player.values():
        entries.sort()

    strengths: dict[PositionKey, float] = {}
    for position in rated:
        context = _context(contexts, position)
        entries = by_player.get((context.game_id, context.color), [])
        earlier = [
            value for ply_index, value in entries if ply_index < context.ply_index
        ]
        if len(earlier) < config.minimum_prefix_decisions:
            continue
        strengths[context.key] = fmean(earlier)
    return strengths


def _context(
    contexts: Mapping[PositionKey, PositionContext],
    position: PositionPolicy,
) -> PositionContext:
    key = (position.game_id, position.ply_index)
    context = contexts.get(key)
    if context is None:
        raise DependencyError(f"scored position {key} has no recorded true context")
    return context
