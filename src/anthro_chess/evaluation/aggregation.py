"""Slice aggregation over scored held-out positions.

Averaging a whole pool hides the populations inside it. Legality varies by
nearly an order of magnitude between opening and endgame positions, and a rule
case that appears in a fraction of a percent of positions cannot move a
pool-wide mean at all. So every metric is aggregated per slice as well as
overall, and a comparison that does not hold those slices fixed is reading
composition rather than model quality.

Aggregation is deliberately separate from scoring: it knows nothing about
models or forward passes. It does read a batch of positions as columns rather
than one record at a time, because a pool holds millions of them and a slice is
a sum.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from anthro_chess.evaluation.policy import PositionColumns, PositionPolicy
from anthro_chess.evaluation.slices import PositionCharacteristic, PositionSlices

SLICE_TABLE_VERSION = 2

#: Reported top-k human-move accuracies. Top-1 says how often the model would
#: play the human move outright; the wider cutoffs say whether it was close.
TOP_K_ACCURACIES: tuple[int, ...] = (1, 3, 5)

#: Slice name used for positions whose player rating is unavailable. Missing
#: ratings are reported as their own slice rather than folded into a band.
UNRATED_SLICE = "unrated"

#: Speed slice for positions whose game names no readable time control, which
#: is where a clockless game lands as well: the normalized columns record an
#: unlimited control as an absent one.
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

#: What a slice sums, in the order a batch's value matrix carries it. One
#: order, read by the columns a batch is summed from and by the accumulator
#: those sums land in.
SUMMED_METRICS: tuple[str, ...] = (
    "move_loss",
    "legal_move_loss",
    "uniform_over_legal_move_loss",
    "mask_penalty",
    "legal_mass",
    "top1_illegal",
    "top_illegal_fraction",
    "legal_margin",
    "legality_lift",
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

    def add_block(
        self,
        count: int,
        sums: NDArray[np.float64],
        within_top: NDArray[np.float64],
    ) -> None:
        """Add positions already summed, in :data:`SUMMED_METRICS` order."""

        self._count += count
        for metric, total in zip(SUMMED_METRICS, sums.tolist(), strict=True):
            self._sums[metric] += total
        for cutoff, total in zip(TOP_K_ACCURACIES, within_top.tolist(), strict=True):
            self._within_top[cutoff] += int(total)

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


@dataclass(frozen=True)
class SliceMembership:
    """Which buckets of one dimension each position of a batch falls in.

    A position holds exactly one phase and every rule case it realizes, so
    membership is a column per bucket rather than an index into one. The two
    kinds of dimension then sum the same way.
    """

    names: tuple[str, ...]
    columns: NDArray[np.bool_]

    def select(self, rows: NDArray[np.int64]) -> SliceMembership:
        """Return the same membership restricted to some of its positions."""

        return SliceMembership(names=self.names, columns=self.columns[rows])


def one_of(names: Sequence[str], indices: NDArray[np.int64]) -> SliceMembership:
    """Return membership for a dimension every position falls in exactly once."""

    return SliceMembership(
        names=tuple(names),
        columns=indices[:, None] == np.arange(len(names)),
    )


def value_matrix(columns: PositionColumns) -> NDArray[np.float64]:
    """Return one row per position: what a slice sums, then what it counts.

    The order is :data:`SUMMED_METRICS` followed by :data:`TOP_K_ACCURACIES`,
    which is the order :meth:`PositionAccumulator.add_block` reads back.
    """

    return np.stack(
        (
            columns.move_nll,
            columns.legal_move_nll,
            columns.uniform_over_legal_move_nll,
            columns.mask_penalty,
            columns.legal_mass,
            columns.top1_illegal.astype(np.float64),
            columns.top_illegal_fraction,
            columns.legal_margin,
            columns.legality_lift,
            *(
                (columns.target_rank <= cutoff).astype(np.float64)
                for cutoff in TOP_K_ACCURACIES
            ),
        ),
        axis=1,
    )


def position_memberships(
    slices: Sequence[PositionSlices],
    characteristics: Sequence[Collection[PositionCharacteristic]],
    *,
    opening_families: Sequence[str] | None = None,
    opening_tiers: Sequence[str] | None = None,
) -> dict[str, SliceMembership]:
    """Return which slice of every dimension each position falls in.

    Derived from the labels rather than from the scores, so a reading that
    knows a batch's slices before it scores it can build these off the pass
    that scores it.
    """

    observed = np.zeros((len(slices), len(REPORTED_CHARACTERISTICS)), dtype=np.bool_)
    for row, present in enumerate(characteristics):
        for column, characteristic in enumerate(REPORTED_CHARACTERISTICS):
            observed[row, column] = characteristic in present

    memberships = {
        PHASE_DIMENSION: _labelled([str(item.phase) for item in slices]),
        COLOR_DIMENSION: _labelled([str(item.color) for item in slices]),
        RATING_DIMENSION: _labelled(
            [item.rating_band or UNRATED_SLICE for item in slices]
        ),
        SPEED_DIMENSION: _labelled(
            [
                UNTIMED_SLICE if item.speed is None else str(item.speed)
                for item in slices
            ]
        ),
        LEGAL_MOVE_COUNT_DIMENSION: _labelled(
            [item.legal_move_count_bucket for item in slices]
        ),
        RULE_CASE_DIMENSION: SliceMembership(
            names=tuple(str(item) for item in REPORTED_CHARACTERISTICS),
            columns=observed,
        ),
    }
    if opening_families is not None:
        memberships[OPENING_FAMILY_DIMENSION] = _labelled(list(opening_families))
    if opening_tiers is not None:
        memberships[OPENING_TIER_DIMENSION] = _labelled(list(opening_tiers))
    return memberships


def _labelled(values: Sequence[str]) -> SliceMembership:
    """Return membership over whichever names these positions realized."""

    names = tuple(dict.fromkeys(values))
    index = {name: offset for offset, name in enumerate(names)}
    return one_of(
        names,
        np.fromiter(
            (index[value] for value in values), dtype=np.int64, count=len(values)
        ),
    )


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
        """Add one scored position under its derived slice labels."""

        self.accumulate(
            PositionColumns.from_records((position,)),
            position_memberships(
                (slices,),
                (frozenset(characteristics),),
                opening_families=None if opening_family is None else (opening_family,),
                opening_tiers=None if opening_tier is None else (opening_tier,),
            ),
        )

    def accumulate(
        self,
        columns: PositionColumns,
        memberships: Mapping[str, SliceMembership],
    ) -> None:
        """Add a batch of scored positions under their slice labels.

        Every slice a batch touches is summed in one pass over it. A slice is a
        sum, and a pool holds more positions than a walk through them one at a
        time can afford.

        The two opening labels describe the game rather than the position, so
        every decision in a game classified as a Sicilian counts toward that
        family, endgame included.
        """

        positions = len(columns)
        if not positions:
            return
        blocks: list[PositionAccumulator] = [self._overall]
        membership = [np.ones((positions, 1), dtype=np.bool_)]
        for dimension in SLICE_DIMENSIONS:
            present = memberships.get(dimension)
            if present is None:
                continue
            blocks.extend(self._bucket(dimension, name) for name in present.names)
            membership.append(present.columns)

        counted = np.concatenate(membership, axis=1).astype(np.float64)
        totals = counted.T @ value_matrix(columns)
        counts = counted.sum(axis=0)
        summed = len(SUMMED_METRICS)
        for offset, accumulator in enumerate(blocks):
            count = int(counts[offset])
            if count:
                accumulator.add_block(
                    count, totals[offset, :summed], totals[offset, summed:]
                )

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

    if not positions:
        return None
    accumulator = PositionAccumulator()
    columns = PositionColumns.from_records(positions)
    totals = value_matrix(columns).sum(axis=0)
    summed = len(SUMMED_METRICS)
    accumulator.add_block(len(positions), totals[:summed], totals[summed:])
    return accumulator.summary()
