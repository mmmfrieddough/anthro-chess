"""Held-out move-time readings, taken beside move prediction in one pass.

Every reading is over decisions whose time the clock recorded, and each is
reported overall and per speed and per clock bucket. Both the marginal and the
distance are taken within cells of one rating band, speed, clock bucket and
phase, and a slice reads its cells weighted by how many moves each holds.
Distances are measured in
``log(1 + ms)``, the space the time head writes in, where a second's error on a
two-second move and a minute's on a two-minute think weigh alike.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from anthro_chess.chess import MOVE_ACTION_COUNT
from anthro_chess.evaluation.dependency import PositionKey
from anthro_chess.evaluation.policy import ActiveBatch
from anthro_chess.evaluation.results import DataComponent, Measurement, measurement
from anthro_chess.evaluation.results.metrics import (
    TIMING_MARGINAL_MOVE_TIME_LOSS,
    TIMING_MARGINAL_MOVE_TIME_LOSS_BY_SLICE,
    TIMING_MOVE_TIME_DISTANCE,
    TIMING_MOVE_TIME_DISTANCE_BY_SLICE,
    TIMING_MOVE_TIME_LOSS,
    TIMING_MOVE_TIME_LOSS_BY_SLICE,
)
from anthro_chess.evaluation.slices import PositionSlices
from anthro_chess.models import MoveModelBatch, MoveTimeDistribution, move_time_targets
from anthro_chess.models.timing import MOVE_TIME_RESOLUTION_MS

#: The marginal histogram's bin width in ``log(1 + ms)``. Coarse enough that a
#: cell of a few hundred moves is not fitted move by move, which would flatter
#: the bar the head is read against.
MARGINAL_BIN_WIDTH = 0.1
#: The distance histograms' bin width in ``log(1 + ms)``, which bounds how far
#: a distance read off them can sit from the exact one.
DISTANCE_BIN_WIDTH = 0.01

_OVERALL = ""


@dataclass(frozen=True)
class TimingColumns:
    """One batch's timed decisions, in the order its active rows came."""

    human_ms: NDArray[np.int64]
    sampled_ms: NDArray[np.int64]
    loss: NDArray[np.float64]
    #: Rating band, speed, clock bucket, and phase: what a comparison matches.
    cells: tuple[tuple[str, str, str, str], ...]


def timing_columns(
    batch: MoveModelBatch,
    active: ActiveBatch,
    times: Callable[[Tensor], MoveTimeDistribution],
    slices: Mapping[PositionKey, PositionSlices],
    generator: torch.Generator,
) -> TimingColumns:
    """Score the human's move time and sample one after a sampled legal move.

    The move is drawn from the legal policy at temperature one, which is what
    a player meets, and the time is drawn after it, as the runtime would.
    Resigning or claiming a draw ends the game rather than spending time on a
    move, so the draw is over moves alone.
    """

    mask = batch.action_loss_mask
    targets = move_time_targets(batch)
    moves = active.legal_mask.clone()
    moves[:, MOVE_ACTION_COUNT:] = False
    sampled_moves = torch.multinomial(
        torch.softmax(active.logits.masked_fill(~moves, -torch.inf), -1),
        1,
        generator=generator,
    ).squeeze(1)
    sampled_actions = batch.action_targets.clone()
    sampled_actions[mask] = sampled_moves.to(sampled_actions.dtype)

    human = times(batch.action_targets)
    timed = targets.present[mask]
    recorded = targets.values[mask]
    human_ms = recorded[timed]
    loss = -_active(human, mask).log_prob(recorded)[timed]
    sampled_ms = _active(times(sampled_actions), mask).sample(generator)[timed]

    keys = [
        key
        for key, keep in zip(
            zip(active.game_ids, active.ply_indices, strict=True),
            timed.tolist(),
            strict=True,
        )
        if keep
    ]
    labelled = [slices[key] for key in keys]
    return TimingColumns(
        human_ms=human_ms.cpu().numpy().astype(np.int64),
        sampled_ms=sampled_ms.cpu().numpy().astype(np.int64),
        loss=loss.cpu().numpy().astype(np.float64),
        # A decision with a time has a clock, so it has a clock bucket. It can
        # lack a speed, when the game recorded an increment and no initial
        # clock, and is then read in the overall reading alone.
        cells=tuple(
            (
                str(item.rating_band),
                str(item.speed),
                str(item.clock_pressure),
                str(item.phase),
            )
            for item in labelled
        ),
    )


@dataclass
class TimingAccumulator:
    """Running totals that outlive the batches they came from."""

    _loss: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    _count: Counter[str] = field(default_factory=Counter)
    _marginal: dict[tuple[str, str, str, str], Counter[int]] = field(
        default_factory=dict
    )
    _human: dict[tuple[str, str, str, str], Counter[int]] = field(default_factory=dict)
    _sampled: dict[tuple[str, str, str, str], Counter[int]] = field(
        default_factory=dict
    )

    def add(self, columns: TimingColumns) -> None:
        """Fold one batch in."""

        logged_human = np.log1p(np.maximum(columns.human_ms, 0))
        marginal_bins = np.floor(logged_human / MARGINAL_BIN_WIDTH).astype(np.int64)
        human_bins = np.floor(logged_human / DISTANCE_BIN_WIDTH).astype(np.int64)
        sampled_bins = np.floor(
            np.log1p(columns.sampled_ms) / DISTANCE_BIN_WIDTH
        ).astype(np.int64)
        for index, cell in enumerate(columns.cells):
            for name in _slices_of(cell):
                self._loss[name] += float(columns.loss[index])
                self._count[name] += 1
            for histograms, bins in (
                (self._marginal, marginal_bins),
                (self._human, human_bins),
                (self._sampled, sampled_bins),
            ):
                histograms.setdefault(cell, Counter())[int(bins[index])] += 1

    def measurements(self, component: DataComponent) -> tuple[Measurement, ...]:
        """Return every timing reading this pass supports."""

        marginal = self._marginal_losses()
        distance = self._distances()
        values: list[Measurement] = []
        for name, count in sorted(self._count.items()):
            for overall, by_slice, value in (
                (
                    TIMING_MOVE_TIME_LOSS,
                    TIMING_MOVE_TIME_LOSS_BY_SLICE,
                    self._loss[name] / count,
                ),
                (
                    TIMING_MARGINAL_MOVE_TIME_LOSS,
                    TIMING_MARGINAL_MOVE_TIME_LOSS_BY_SLICE,
                    marginal[name] / count,
                ),
                (
                    TIMING_MOVE_TIME_DISTANCE,
                    TIMING_MOVE_TIME_DISTANCE_BY_SLICE,
                    distance[name] / count,
                ),
            ):
                definition = overall if name == _OVERALL else by_slice.get(name)
                if definition is None:
                    continue
                values.append(
                    measurement(
                        definition.identifier,
                        value,
                        data=component,
                        sample_size=count,
                    )
                )
        return tuple(values)

    def as_record(self) -> dict[str, object]:
        """Return each slice's readings and its human and sampled quantiles."""

        marginal = self._marginal_losses()
        distance = self._distances()
        human: defaultdict[str, Counter[int]] = defaultdict(Counter)
        sampled: defaultdict[str, Counter[int]] = defaultdict(Counter)
        for cell in self._human:
            for name in _slices_of(cell):
                human[name].update(self._human[cell])
                sampled[name].update(self._sampled[cell])
        return {
            "marginal_bin_width": MARGINAL_BIN_WIDTH,
            "distance_bin_width": DISTANCE_BIN_WIDTH,
            "slices": {
                name or "all": {
                    "moves": count,
                    "move_time_loss": self._loss[name] / count,
                    "marginal_move_time_loss": marginal[name] / count,
                    "move_time_distance": distance[name] / count,
                    "human_ms": _quantiles(human[name]),
                    "sampled_ms": _quantiles(sampled[name]),
                }
                for name, count in sorted(self._count.items())
            },
        }

    def _marginal_losses(self) -> defaultdict[str, float]:
        """Return each slice's summed loss under the matched histograms.

        A move's mass is its bin's share of its cell, spread evenly over the
        recorded times the bin holds.
        """

        totals: defaultdict[str, float] = defaultdict(float)
        for cell, histogram in self._marginal.items():
            moves = sum(histogram.values())
            loss = -sum(
                count * math.log(count / moves / _recorded_times_in(bin_index))
                for bin_index, count in histogram.items()
            )
            for name in _slices_of(cell):
                totals[name] += loss
        return totals

    def _distances(self) -> defaultdict[str, float]:
        """Return each slice's cell distances, each weighted by its moves."""

        totals: defaultdict[str, float] = defaultdict(float)
        for cell, human in self._human.items():
            weighted = sum(human.values()) * _distance(human, self._sampled[cell])
            for name in _slices_of(cell):
                totals[name] += weighted
        return totals


def _active(distribution: MoveTimeDistribution, mask: Tensor) -> MoveTimeDistribution:
    return MoveTimeDistribution(
        distribution.logits[mask],
        distribution.locations[mask],
        distribution.log_scales[mask],
    )


def _slices_of(cell: tuple[str, str, str, str]) -> tuple[str, str, str]:
    """Return the readings a cell counts toward: overall, its speed, its clock."""

    _, speed, clock, _ = cell
    return (_OVERALL, speed, clock)


def _recorded_times_in(bin_index: int) -> int:
    """Return how many recorded times fall in one marginal bin, at least one."""

    def first_at_or_above(logged: float) -> int:
        return math.ceil(math.expm1(logged) / MOVE_TIME_RESOLUTION_MS)

    return max(
        1,
        first_at_or_above((bin_index + 1) * MARGINAL_BIN_WIDTH)
        - first_at_or_above(bin_index * MARGINAL_BIN_WIDTH),
    )


#: The quantiles a record reports of each move-time distribution.
_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _quantiles(histogram: Counter[int]) -> dict[str, int]:
    """Return a distance histogram's quantiles in milliseconds, to its bin width."""

    bins = sorted(histogram)
    cumulative = np.cumsum([histogram[index] for index in bins])
    total = cumulative[-1]
    return {
        f"p{round(quantile * 100)}": round(
            math.expm1(
                (bins[int(np.searchsorted(cumulative, quantile * total))] + 0.5)
                * DISTANCE_BIN_WIDTH
            )
        )
        for quantile in _QUANTILES
    }


def _distance(human: Counter[int], sampled: Counter[int]) -> float:
    """Return the Wasserstein distance between two histograms of one bin width.

    In one dimension it is the area between the two cumulative distributions.
    """

    bins = range(min(*human, *sampled), max(*human, *sampled) + 1)
    human_total = sum(human.values())
    sampled_total = sum(sampled.values())
    human_cdf = np.cumsum([human.get(index, 0) for index in bins]) / human_total
    sampled_cdf = np.cumsum([sampled.get(index, 0) for index in bins]) / sampled_total
    return float(np.abs(human_cdf - sampled_cdf).sum() * DISTANCE_BIN_WIDTH)


__all__: Sequence[str] = [
    "MARGINAL_BIN_WIDTH",
    "TimingAccumulator",
    "TimingColumns",
    "timing_columns",
]
