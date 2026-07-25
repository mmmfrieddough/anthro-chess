"""Deterministic validation metrics over the ordinary move-model boundary."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from anthro_chess.chess import ACTION_VOCABULARY_SIZE
from anthro_chess.data import SequenceBatch
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

        batch.validate()
        _validate_logits(logits, batch)

        active_indices = (
            torch.nonzero(
                batch.action_loss_mask,
                as_tuple=False,
            )
            .detach()
            .cpu()
        )
        active_logits = (
            logits[batch.action_loss_mask].detach().cpu().to(dtype=torch.float64)
        )
        active_targets = (
            batch.action_targets[batch.action_loss_mask].detach().to(device="cpu")
        )
        log_probabilities = torch.log_softmax(active_logits, dim=-1)
        move_losses = -log_probabilities.gather(
            1,
            active_targets.unsqueeze(1),
        ).squeeze(1)

        rating_values = (
            batch.inputs.target_rating.values[batch.action_loss_mask]
            .detach()
            .to(device="cpu")
        )
        rating_present = (
            batch.inputs.target_rating.present[batch.action_loss_mask]
            .detach()
            .to(device="cpu")
        )
        _validate_ratings(rating_values, rating_present)

        active_targets_list = active_targets.detach().cpu().tolist()
        legal_rows: list[tuple[int, ...]] = []
        for (batch_index, sequence_index), target in zip(
            active_indices.tolist(),
            active_targets_list,
            strict=True,
        ):
            legal_actions = batch.legal_action_ids[batch_index][sequence_index]
            _validate_legal_actions(legal_actions, target)
            legal_rows.append(legal_actions)

        legal_mask = torch.zeros_like(active_logits, dtype=torch.bool)
        for active_offset, legal_actions in enumerate(legal_rows):
            legal_indices = torch.tensor(
                legal_actions,
                dtype=torch.long,
                device=active_logits.device,
            )
            legal_mask[active_offset, legal_indices] = True

        log_legal_mass = torch.logsumexp(
            active_logits.masked_fill(~legal_mask, -torch.inf),
            dim=-1,
        ) - torch.logsumexp(active_logits, dim=-1)
        legal_mass = torch.exp(log_legal_mass)
        top_actions = torch.argmax(active_logits, dim=-1, keepdim=True)
        top1_illegal = ~legal_mask.gather(1, top_actions).squeeze(1)

        active_count = len(legal_rows)
        self._position_count += active_count
        self._move_loss_sum += float(move_losses.sum().item())
        self._legal_move_loss_sum += float((move_losses + log_legal_mass).sum().item())
        self._mask_penalty_sum += float((-log_legal_mass).sum().item())
        self._legal_mass_sum += float(legal_mass.sum().item())
        self._top1_illegal_count += int(top1_illegal.sum().item())
        self._uniform_legal_loss_sum += sum(
            math.log(len(legal_actions)) for legal_actions in legal_rows
        )

        move_loss_values = move_losses.detach().cpu().tolist()
        rating_value_list = rating_values.detach().cpu().tolist()
        rating_present_list = rating_present.detach().cpu().tolist()
        for move_loss, rating, present in zip(
            move_loss_values,
            rating_value_list,
            rating_present_list,
            strict=True,
        ):
            if present:
                band_index = _rating_band_index(rating, self._rating_bands)
                self._rated_position_count += 1
                self._rating_counts[band_index] += 1
                self._rating_loss_sums[band_index] += move_loss
            else:
                self._missing_rating_position_count += 1
                self._missing_rating_loss_sum += move_loss

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


def _validate_logits(logits: Tensor, batch: MoveModelBatch) -> None:
    expected_shape = (*batch.action_targets.shape, ACTION_VOCABULARY_SIZE)
    if logits.shape != expected_shape:
        raise ValueError(
            "action logits must align with model targets and the action vocabulary"
        )
    if not logits.is_floating_point():
        raise ValueError("action logits must use a floating-point dtype")
    if not torch.all(torch.isfinite(logits[batch.action_loss_mask])):
        raise ValueError("enabled action logits must all be finite")


def _validate_legal_actions(
    legal_actions: tuple[int, ...],
    target: int,
) -> None:
    if not legal_actions:
        raise ValueError("enabled validation position has no legal actions")
    if tuple(sorted(set(legal_actions))) != legal_actions:
        raise ValueError("legal actions must be sorted and unique")
    if legal_actions[0] < 0 or legal_actions[-1] >= ACTION_VOCABULARY_SIZE:
        raise ValueError("legal action is outside the action vocabulary")
    if target not in legal_actions:
        raise ValueError("enabled validation target is not legal")


def _validate_ratings(ratings: Tensor, present: Tensor) -> None:
    if torch.any(ratings[present] < 0):
        raise ValueError("present player ratings must be nonnegative")
