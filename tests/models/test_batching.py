"""Tests for the alignment contract every model batch is validated against."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
import torch

from anthro_chess.chess import ACTION_VOCABULARY_SIZE
from anthro_chess.data import (
    EN_PASSANT_TOKEN_COUNT,
    REPETITION_STATE_COUNT,
    SequenceBatch,
)
from anthro_chess.models import MoveModelBatch
from anthro_chess.models.batching import OptionalTensor


def _batch(sequence_batch: Callable[..., SequenceBatch]) -> MoveModelBatch:
    return MoveModelBatch.from_sequence_batch(
        sequence_batch((("e2e4", "e7e5"), 1500, 1600))
    )


def _with_inputs(batch: MoveModelBatch, **overrides: Any) -> MoveModelBatch:
    return replace(batch, inputs=replace(batch.inputs, **overrides))


def _holed(mask: torch.Tensor) -> torch.Tensor:
    """Return the mask with an interior timestep cleared, breaking alignment."""

    holed = mask.clone()
    holed[0, 0] = False
    return holed


def _corrupted(batch: MoveModelBatch, field: str, value: int) -> MoveModelBatch:
    """Return the batch with one entry of an ordinary input tensor rewritten."""

    original: torch.Tensor = getattr(batch.inputs, field)
    replacement = original.clone()
    replacement.view(-1)[0] = value
    return _with_inputs(batch, **{field: replacement})


def test_a_valid_batch_is_accepted(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    _batch(sequence_batch).validate()


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        pytest.param(
            lambda batch: replace(
                batch,
                ply_indices=batch.ply_indices - 1,
            ),
            "ply indices must be nonnegative",
            id="negative-ply-index",
        ),
        pytest.param(
            lambda batch: replace(
                batch,
                action_loss_mask=torch.ones_like(batch.action_loss_mask),
                attention_mask=torch.zeros_like(batch.attention_mask),
            ),
            "action loss cannot include padded timesteps",
            id="padded-loss",
        ),
        pytest.param(
            # The model reads no padding mask, so an interior gap would let a
            # real timestep attend to padding with nothing to notice.
            lambda batch: replace(
                batch,
                attention_mask=_holed(batch.attention_mask),
                action_loss_mask=_holed(batch.action_loss_mask),
            ),
            "padding must follow every real timestep in a row",
            id="padding-before-a-real-timestep",
        ),
        pytest.param(
            lambda batch: _corrupted(batch, "piece_ids", 13),
            "piece ids are outside the board encoding",
            id="piece-id",
        ),
        pytest.param(
            lambda batch: _corrupted(batch, "side_to_move", 2),
            "side-to-move ids are outside the board encoding",
            id="side-to-move",
        ),
        pytest.param(
            lambda batch: _corrupted(batch, "castling_rights", 16),
            "castling rights are outside the board encoding",
            id="castling-rights",
        ),
        pytest.param(
            lambda batch: _corrupted(batch, "en_passant_token", EN_PASSANT_TOKEN_COUNT),
            "en-passant tokens are outside the board encoding",
            id="en-passant",
        ),
        pytest.param(
            lambda batch: _corrupted(batch, "repetition_count", REPETITION_STATE_COUNT),
            "repetition counts are outside the board encoding",
            id="repetition",
        ),
        pytest.param(
            lambda batch: _with_inputs(
                batch,
                target_rating=OptionalTensor(
                    values=torch.full_like(batch.inputs.target_rating.values, -1),
                    present=torch.ones_like(batch.inputs.target_rating.present),
                ),
            ),
            "target ratings must be nonnegative",
            id="negative-rating",
        ),
    ],
)
def test_out_of_range_values_are_rejected_by_name(
    sequence_batch: Callable[..., SequenceBatch],
    corrupt: Callable[[MoveModelBatch], MoveModelBatch],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        corrupt(_batch(sequence_batch)).validate()


def test_the_tensor_boundary_widens_the_columns_the_model_indexes_with(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    """The loader hands over narrow columns; embeddings and the loss need long.

    Widening belongs on this side of the copy, so what crosses to a device is
    the loader's own width and what the model reads is what it can index with.
    """

    source = sequence_batch((("e2e4", "e7e5"), 1500, 1600))
    batch = MoveModelBatch.from_sequence_batch(source)

    assert source.inputs.piece_ids.dtype.name == "uint8"
    assert batch.inputs.piece_ids.dtype is torch.long
    assert batch.action_targets.dtype is torch.long
    assert batch.ply_indices.dtype is torch.long
    assert batch.inputs.repetition_count.dtype is torch.long
    assert batch.attention_mask.dtype is torch.bool
    assert batch.inputs.piece_ids.tolist() == source.inputs.piece_ids.tolist()
    assert batch.action_targets.tolist() == source.action_targets.tolist()
    assert batch.game_ids.tolist() == source.game_ids.tolist()


def test_comparing_two_batches_answers_instead_of_asking_a_tensor_for_a_bool(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    """Comparison is total and returns a bool, whatever the tensors hold.

    ``False`` for a copy holding the same tensors is the point: no caller may
    read ``==`` as "same contents". That is ``torch.equal`` per field.
    """

    batch = _batch(sequence_batch)

    for value in (batch, batch.inputs, batch.inputs.target_rating):
        assert (value == replace(value)) is False


def test_the_tensor_boundary_rejects_a_corrupted_source_column(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    """The factory checks the arrays it was handed, not the tensors it built.

    The copy only widens, so those are the same values. What must not change is
    that the factory asks at all.
    """

    source = sequence_batch((("e2e4", "e7e5"), 1500, 1600))
    piece_ids = source.inputs.piece_ids.copy()
    piece_ids.flat[0] = 13

    with pytest.raises(ValueError, match="piece ids are outside"):
        MoveModelBatch.from_sequence_batch(
            replace(source, inputs=replace(source.inputs, piece_ids=piece_ids))
        )


def test_an_active_target_outside_the_vocabulary_is_rejected(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    batch = _batch(sequence_batch)
    targets = batch.action_targets.clone()
    targets[batch.action_loss_mask] = ACTION_VOCABULARY_SIZE

    with pytest.raises(ValueError, match="active action target is outside"):
        replace(batch, action_targets=targets).validate()


def test_an_inactive_target_outside_the_vocabulary_is_ignored(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    """Only enabled timesteps carry a target; padding is not held to one."""

    batch = _batch(sequence_batch)
    targets = batch.action_targets.clone()
    targets[~batch.action_loss_mask] = ACTION_VOCABULARY_SIZE

    replace(batch, action_targets=targets).validate()


def test_an_active_target_that_is_not_legal_is_rejected(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    batch = _batch(sequence_batch)
    assert batch.legal_action_ids is not None
    legal = batch.legal_action_ids[0][0]
    substitute = next(
        action for action in range(ACTION_VOCABULARY_SIZE) if action not in legal
    )
    targets = batch.action_targets.clone()
    targets[0, 0] = substitute

    with pytest.raises(ValueError, match="active target must be legal"):
        replace(batch, action_targets=targets).validate()


def test_misaligned_legal_actions_are_rejected(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    batch = _batch(sequence_batch)
    assert batch.legal_action_ids is not None

    with pytest.raises(ValueError, match="legal actions must align"):
        replace(batch, legal_action_ids=batch.legal_action_ids[:0]).validate()


def test_a_batch_without_legal_actions_skips_only_the_legality_checks(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    """A training batch carries none of them, and must still be validated.

    The alignment and range checks are what a training step depends on. Only
    the two that read legal actions can be skipped, so an out-of-range value
    still has to be named rather than waved through with them.
    """

    batch = replace(_batch(sequence_batch), legal_action_ids=None)

    batch.validate()

    with pytest.raises(ValueError, match="piece ids are outside"):
        _corrupted(batch, "piece_ids", 13).validate()


def test_validation_never_asks_the_device_for_a_scalar(
    sequence_batch: Callable[..., SequenceBatch],
    device_read_trap: Callable[[Any], Any],
) -> None:
    """Every reduction is read back together rather than one question at a time.

    A validation that queries the device per check, or worse per timestep,
    passes on CPU while stalling a real accelerator once per query. Nothing
    about the result would say so, which is why this asserts on the access
    pattern rather than on the verdict.
    """

    device_read_trap(_batch(sequence_batch)).validate()


def test_the_trap_still_reports_a_real_violation(
    sequence_batch: Callable[..., SequenceBatch],
    device_read_trap: Callable[[Any], Any],
) -> None:
    """The sync-free path must reject what the blocking one rejected."""

    batch = _batch(sequence_batch)
    corrupted = _corrupted(batch, "piece_ids", 13)

    with pytest.raises(ValueError, match="piece ids are outside"):
        device_read_trap(corrupted).validate()
