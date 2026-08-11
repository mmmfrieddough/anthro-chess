"""Slice aggregation over scored held-out positions.

Averaging a whole pool hides the populations inside it. Legality varies by
nearly an order of magnitude between opening and endgame positions, and a rule
case that appears in a fraction of a percent of positions cannot move a
pool-wide mean at all. So every metric is aggregated per slice as well as
overall, and a comparison that does not hold those slices fixed is reading
composition rather than model quality.

Aggregation is deliberately separate from scoring: it consumes per-position
records and knows nothing about models, batches, or torch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from anthro_chess.evaluation.policy import PositionPolicy
from anthro_chess.evaluation.slices import PositionCharacteristic, PositionSlices

SLICE_TABLE_VERSION = 2

#: Reported top-k human-move accuracies. Top-1 says how often the model would
#: play the human move outright; the wider cutoffs say whether it was close.
TOP_K_ACCURACIES: tuple[int, ...] = (1, 3, 5)

#: Slice name used for positions whose player rating is unavailable. Missing
#: ratings are reported as their own slice rather than folded into a band.
UNRATED_SLICE = "unrated"

#: Slice name used for positions whose game carries no readable time control.
#: A game played without a clock lands here too, since the normalized columns
#: record an unlimited control as an absent one, so the slice says the class is
#: unknown rather than naming correspondence.
UNTIMED_SLICE = "untimed"

#: Rule cases that can hold at a scored decision. Terminal, checkmate, and
#: stalemate positions offer no move to predict, so they never appear here.
REPORTED_CHARACTERISTICS: tuple[PositionCharacteristic, ...] = (
    PositionCharacteristic.CASTLING_AVAILABLE,
    PositionCharacteristic.CASTLING_RIGHTS,
    PositionCharacteristic.CHECK,
    PositionCharacteristic.EN_PASSANT,
    PositionCharacteristic.ONLY_MOVE,
    PositionCharacteristic.PIN,
    PositionCharacteristic.PROMOTION,
)

PHASE_DIMENSION = "phase"
COLOR_DIMENSION = "color"
RATING_DIMENSION = "rating_band"
SPEED_DIMENSION = "speed"
LEGAL_MOVE_COUNT_DIMENSION = "legal_move_count"
RULE_CASE_DIMENSION = "rule_case"
OPENING_FAMILY_DIMENSION = "opening_family"
OPENING_TIER_DIMENSION = "opening_frequency_tier"

SLICE_DIMENSIONS: tuple[str, ...] = (
    PHASE_DIMENSION,
    COLOR_DIMENSION,
    RATING_DIMENSION,
    SPEED_DIMENSION,
    LEGAL_MOVE_COUNT_DIMENSION,
    RULE_CASE_DIMENSION,
    OPENING_FAMILY_DIMENSION,
    OPENING_TIER_DIMENSION,
)


@dataclass(frozen=True)
class PositionSummary:
    """Mean metrics over one set of scored positions."""

    position_count: int
    move_loss: float
    legal_move_loss: float
    uniform_over_legal_move_loss: float
    mask_penalty: float
    legal_mass: float
    top1_illegal_rate: float
    top_illegal_fraction: float
    legal_margin: float
    legality_lift: float
    top_k_accuracy: tuple[tuple[int, float], ...]

    def accuracy(self, k: int) -> float:
        """Return the reported top-``k`` human-move accuracy."""

        for cutoff, value in self.top_k_accuracy:
            if cutoff == k:
                return value
        raise KeyError(f"top-{k} accuracy is not reported")

    def as_record(self) -> dict[str, object]:
        """Return the stable JSON-serializable summary record."""

        return {
            "position_count": self.position_count,
            "move_loss": self.move_loss,
            "legal_move_loss": self.legal_move_loss,
            "uniform_over_legal_move_loss": self.uniform_over_legal_move_loss,
            "legality": {
                "mask_penalty": self.mask_penalty,
                "legal_mass": self.legal_mass,
                "illegal_mass": 1.0 - self.legal_mass,
                "top1_illegal_rate": self.top1_illegal_rate,
                "top_illegal_fraction": self.top_illegal_fraction,
                "legal_margin": self.legal_margin,
                "legality_lift": self.legality_lift,
            },
            "accuracy": {
                f"top{cutoff}": value for cutoff, value in self.top_k_accuracy
            },
        }


class PositionAccumulator:
    """Sum per-position quantities without depending on batch boundaries."""

    def __init__(self) -> None:
        self._count = 0
        self._sums: dict[str, float] = {
            "move_loss": 0.0,
            "legal_move_loss": 0.0,
            "uniform_over_legal_move_loss": 0.0,
            "mask_penalty": 0.0,
            "legal_mass": 0.0,
            "top1_illegal": 0.0,
            "top_illegal_fraction": 0.0,
            "legal_margin": 0.0,
            "legality_lift": 0.0,
        }
        self._within_top = dict.fromkeys(TOP_K_ACCURACIES, 0)

    @property
    def position_count(self) -> int:
        """Return how many positions have been accumulated."""

        return self._count

    def add(self, position: PositionPolicy) -> None:
        """Add one scored position."""

        self._count += 1
        self._sums["move_loss"] += position.move_nll
        self._sums["legal_move_loss"] += position.legal_move_nll
        self._sums["uniform_over_legal_move_loss"] += (
            position.uniform_over_legal_move_nll
        )
        self._sums["mask_penalty"] += position.mask_penalty
        self._sums["legal_mass"] += position.legal_mass
        self._sums["top1_illegal"] += float(position.top1_illegal)
        self._sums["top_illegal_fraction"] += position.top_illegal_fraction
        self._sums["legal_margin"] += position.legal_margin
        self._sums["legality_lift"] += position.legality_lift
        for cutoff in TOP_K_ACCURACIES:
            if position.within_top(cutoff):
                self._within_top[cutoff] += 1

    def summary(self) -> PositionSummary | None:
        """Return the aggregate, or ``None`` when the slice is empty.

        An empty slice is reported as absent rather than as a zero. A rule
        case the pool never realized is a coverage fact, and inventing a value
        for it would be indistinguishable from a perfect score.
        """

        if self._count == 0:
            return None
        count = float(self._count)
        return PositionSummary(
            position_count=self._count,
            move_loss=self._sums["move_loss"] / count,
            legal_move_loss=self._sums["legal_move_loss"] / count,
            uniform_over_legal_move_loss=(
                self._sums["uniform_over_legal_move_loss"] / count
            ),
            mask_penalty=self._sums["mask_penalty"] / count,
            legal_mass=self._sums["legal_mass"] / count,
            top1_illegal_rate=self._sums["top1_illegal"] / count,
            top_illegal_fraction=self._sums["top_illegal_fraction"] / count,
            legal_margin=self._sums["legal_margin"] / count,
            legality_lift=self._sums["legality_lift"] / count,
            top_k_accuracy=tuple(
                (cutoff, self._within_top[cutoff] / count)
                for cutoff in TOP_K_ACCURACIES
            ),
        )


@dataclass(frozen=True)
class SliceTable:
    """Every reported slice of one evaluated position set."""

    overall: PositionSummary
    dimensions: Mapping[str, Mapping[str, PositionSummary]]

    def slice_summary(self, dimension: str, name: str) -> PositionSummary | None:
        """Return one slice summary, or ``None`` when it holds no positions."""

        return self.dimensions.get(dimension, {}).get(name)

    def as_record(self) -> dict[str, object]:
        """Return the detail-tier slice tables."""

        return {
            "version": SLICE_TABLE_VERSION,
            "overall": self.overall.as_record(),
            "dimensions": {
                dimension: {
                    name: summary.as_record()
                    for name, summary in sorted(slices.items())
                }
                for dimension, slices in sorted(self.dimensions.items())
            },
        }


class SliceAggregator:
    """Accumulate one scored position into every slice it belongs to."""

    def __init__(self) -> None:
        self._overall = PositionAccumulator()
        self._dimensions: dict[str, dict[str, PositionAccumulator]] = {
            dimension: {} for dimension in SLICE_DIMENSIONS
        }

    def add(
        self,
        position: PositionPolicy,
        slices: PositionSlices,
        characteristics: Iterable[PositionCharacteristic],
        *,
        opening_family: str | None = None,
        opening_tier: str | None = None,
    ) -> None:
        """Add one scored position under its derived slice labels.

        The two opening labels describe the game rather than the position, so
        every decision in a game classified as a Sicilian counts toward that
        family, endgame included.
        """

        self._overall.add(position)
        self._bucket(PHASE_DIMENSION, str(slices.phase)).add(position)
        self._bucket(COLOR_DIMENSION, str(slices.color)).add(position)
        self._bucket(
            RATING_DIMENSION,
            slices.rating_band if slices.rating_band is not None else UNRATED_SLICE,
        ).add(position)
        self._bucket(
            SPEED_DIMENSION,
            str(slices.speed) if slices.speed is not None else UNTIMED_SLICE,
        ).add(position)
        self._bucket(
            LEGAL_MOVE_COUNT_DIMENSION,
            slices.legal_move_count_bucket,
        ).add(position)
        observed = frozenset(characteristics)
        for characteristic in REPORTED_CHARACTERISTICS:
            if characteristic in observed:
                self._bucket(RULE_CASE_DIMENSION, str(characteristic)).add(position)
        if opening_family is not None:
            self._bucket(OPENING_FAMILY_DIMENSION, opening_family).add(position)
        if opening_tier is not None:
            self._bucket(OPENING_TIER_DIMENSION, opening_tier).add(position)

    def compute(self) -> SliceTable:
        """Return every slice summary, rejecting an empty evaluation."""

        overall = self._overall.summary()
        if overall is None:
            raise ValueError("evaluation requires at least one scored position")
        dimensions: dict[str, dict[str, PositionSummary]] = {}
        for dimension, buckets in self._dimensions.items():
            summaries = {
                name: summary
                for name, accumulator in buckets.items()
                if (summary := accumulator.summary()) is not None
            }
            dimensions[dimension] = summaries
        return SliceTable(overall=overall, dimensions=dimensions)

    def _bucket(self, dimension: str, name: str) -> PositionAccumulator:
        buckets = self._dimensions[dimension]
        accumulator = buckets.get(name)
        if accumulator is None:
            accumulator = PositionAccumulator()
            buckets[name] = accumulator
        return accumulator


def summarize(positions: Sequence[PositionPolicy]) -> PositionSummary | None:
    """Return the aggregate over an explicit set of scored positions."""

    accumulator = PositionAccumulator()
    for position in positions:
        accumulator.add(position)
    return accumulator.summary()
