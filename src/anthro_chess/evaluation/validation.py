"""Deterministic validation metrics over the ordinary move-model boundary."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from anthro_chess.chess import ACTION_VOCABULARY_SIZE
from anthro_chess.data import SequenceBatch
from anthro_chess.evaluation.policy import PositionPolicy, score_positions
from anthro_chess.evaluation.slices import (
    DEFAULT_RATING_BANDS,
    RatingBand,
    _rating_band_index,
    _validate_rating_bands,
)
from anthro_chess.models import MoveModelBatch

VALIDATION_METRICS_VERSION = 2


@dataclass(frozen=True)
class RatingSliceMetrics:
    """Move loss for one explicit normalized-rating interval."""

    band: RatingBand
    position_count: int
    move_loss: float | None

    def as_record(self) -> dict[str, object]:
        """Return a stable JSON-serializable metric record."""

        return {
            "name": self.band.name,
            "minimum_rating": self.band.minimum_rating,
            "maximum_rating": self.band.maximum_rating,
            "position_count": self.position_count,
            "move_loss": self.move_loss,
        }


@dataclass(frozen=True)
class MoveValidationMetrics:
    """Compact default validation results plus explicit rating slices."""

    position_count: int
    move_loss: float
    legal_move_loss: float
    mask_penalty: float
    legal_mass: float
    top1_illegal_rate: float
    uniform_over_legal_move_loss: float
    uniform_over_vocabulary_move_loss: float
    rated_position_count: int
    missing_rating_position_count: int
    missing_rating_move_loss: float | None
    rating_slices: tuple[RatingSliceMetrics, ...]

    def as_record(self) -> dict[str, object]:
        """Return the versioned structured result used by run artifacts."""

        return {
            "version": VALIDATION_METRICS_VERSION,
            "position_count": self.position_count,
            "move_loss": self.move_loss,
            "legal_move_loss": self.legal_move_loss,
            "legality": {
                "mask_penalty": self.mask_penalty,
                "legal_mass": self.legal_mass,
                "top1_illegal_rate": self.top1_illegal_rate,
            },
            "baselines": {
                "uniform_over_legal_move_loss": (self.uniform_over_legal_move_loss),
                "uniform_over_vocabulary_move_loss": (
                    self.uniform_over_vocabulary_move_loss
                ),
            },
            "ratings": {
                "rated_position_count": self.rated_position_count,
                "missing_rating_position_count": self.missing_rating_position_count,
                "missing_rating_move_loss": self.missing_rating_move_loss,
                "slices": [item.as_record() for item in self.rating_slices],
            },
        }


class MoveValidationAccumulator:
    """Aggregate per-position metrics without depending on batch boundaries."""

    def __init__(
        self,
        rating_bands: Sequence[RatingBand] = DEFAULT_RATING_BANDS,
    ) -> None:
        self._rating_bands = _validate_rating_bands(rating_bands)
        self._position_count = 0
        self._move_loss_sum = 0.0
        self._legal_move_loss_sum = 0.0
        self._mask_penalty_sum = 0.0
        self._legal_mass_sum = 0.0
        self._top1_illegal_count = 0
        self._uniform_legal_loss_sum = 0.0
        self._rated_position_count = 0
        self._missing_rating_position_count = 0
        self._missing_rating_loss_sum = 0.0
        self._rating_counts = [0 for _ in self._rating_bands]
        self._rating_loss_sums = [0.0 for _ in self._rating_bands]

    def update(self, logits: Tensor, batch: MoveModelBatch) -> None:
        """Add one aligned raw-logit batch to the validation result."""

        self.add(score_positions(logits, batch))

    def add(self, positions: Iterable[PositionPolicy]) -> None:
        """Add already-scored positions, so callers can score a batch once."""

        for position in positions:
            self._position_count += 1
            self._move_loss_sum += position.move_nll
            self._legal_move_loss_sum += position.legal_move_nll
            self._mask_penalty_sum += position.mask_penalty
            self._legal_mass_sum += position.legal_mass
            self._top1_illegal_count += int(position.top1_illegal)
            self._uniform_legal_loss_sum += position.uniform_over_legal_move_nll
            if position.conditioned_rating is None:
                self._missing_rating_position_count += 1
                self._missing_rating_loss_sum += position.move_nll
            else:
                band_index = _rating_band_index(
                    position.conditioned_rating,
                    self._rating_bands,
                )
                self._rated_position_count += 1
                self._rating_counts[band_index] += 1
                self._rating_loss_sums[band_index] += position.move_nll

    def compute(self) -> MoveValidationMetrics:
        """Return the aggregate result, rejecting an empty validation input."""

        if self._position_count == 0:
            raise ValueError("validation requires at least one enabled action target")

        rating_slices = tuple(
            RatingSliceMetrics(
                band=band,
                position_count=count,
                move_loss=(self._rating_loss_sums[index] / count if count else None),
            )
            for index, (band, count) in enumerate(
                zip(self._rating_bands, self._rating_counts, strict=True)
            )
        )
        return MoveValidationMetrics(
            position_count=self._position_count,
            move_loss=self._move_loss_sum / self._position_count,
            legal_move_loss=self._legal_move_loss_sum / self._position_count,
            mask_penalty=self._mask_penalty_sum / self._position_count,
            legal_mass=self._legal_mass_sum / self._position_count,
            top1_illegal_rate=(self._top1_illegal_count / self._position_count),
            uniform_over_legal_move_loss=(
                self._uniform_legal_loss_sum / self._position_count
            ),
            uniform_over_vocabulary_move_loss=math.log(ACTION_VOCABULARY_SIZE),
            rated_position_count=self._rated_position_count,
            missing_rating_position_count=self._missing_rating_position_count,
            missing_rating_move_loss=(
                self._missing_rating_loss_sum / self._missing_rating_position_count
                if self._missing_rating_position_count
                else None
            ),
            rating_slices=rating_slices,
        )


def evaluate_move_model(
    model: nn.Module,
    batches: Iterable[SequenceBatch],
    *,
    device: torch.device | str | None = None,
    rating_bands: Sequence[RatingBand] = DEFAULT_RATING_BANDS,
) -> MoveValidationMetrics:
    """Evaluate a model over ordinary loader batches using raw action logits."""

    accumulator = MoveValidationAccumulator(rating_bands)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for sequence_batch in batches:
                model_batch = MoveModelBatch.from_sequence_batch(
                    sequence_batch,
                    device=device,
                )
                logits = model(model_batch)
                if not isinstance(logits, Tensor):
                    raise TypeError("move model must return an action-logit tensor")
                accumulator.update(logits, model_batch)
    finally:
        model.train(was_training)
    return accumulator.compute()
