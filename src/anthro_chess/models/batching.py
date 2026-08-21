"""Tensor conversion at the framework-neutral sequence-batch boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch
from torch import Tensor

from anthro_chess.chess import ACTION_VOCABULARY_SIZE
from anthro_chess.data import (
    BOARD_SQUARE_COUNT,
    EN_PASSANT_TOKEN_COUNT,
    REPETITION_STATE_COUNT,
    DecisionColumn,
    DecisionContext,
    SequenceBatch,
)
from anthro_chess.data.loading import LegalActionTensor

#: What the alignment and range rules read. The loader's arrays and the tensors
#: built from them name their columns alike, so the rules are written once and
#: the type checker holds both families to every name they use.
_Batch: TypeAlias = "SequenceBatch | MoveModelBatch"


# Every dataclass below that holds a tensor compares by identity: a generated
# ``__eq__`` asks an elementwise comparison for a truth value it has none of.
# Compare the tensors a caller means, with ``torch.equal``.
@dataclass(frozen=True, eq=False)
class OptionalTensor:
    """Nullable integer values with an explicit presence mask."""

    values: Tensor
    present: Tensor


@dataclass(frozen=True, eq=False)
class MoveModelInputs:
    """Tensorized exact state and context shaped batch by sequence."""

    piece_ids: Tensor
    side_to_move: Tensor
    castling_rights: Tensor
    en_passant_token: Tensor
    halfmove_clock: Tensor
    fullmove_number: Tensor
    repetition_count: Tensor
    target_rating: OptionalTensor


@dataclass(frozen=True, eq=False)
class MoveModelBatch:
    """Tensor model boundary plus alignment metadata for inspection.

    Every factory here validates every batch it returns, so a batch a factory
    returned is a batch that passed and the model rechecks nothing. What
    :meth:`from_sequence_batch` checks is the loader's arrays rather than the
    tensors it built out of them, for the reason :func:`_reject_invalid_batch`
    gives.
    :func:`dataclasses.replace` goes around that: substituting a field of equal
    shape and range is safe, and changing a shape, a ply index, or the padding
    layout owns calling :meth:`validate` again.

    ``decision_columns`` names the column each timestep sits at in its own row.
    Counted here rather than inside the model, because an ``arange`` built
    inside a compiled graph is folded into the index arithmetic of the gathers
    that consume it and Inductor then fails to simplify it.

    ``legal_action_ids`` is ``None`` when the batch came from a loader that was
    not asked for them. Legality checking and policy scoring are the only
    readers, so a training batch omits them and :meth:`validate` skips the
    checks that need them.
    """

    inputs: MoveModelInputs
    action_targets: Tensor
    action_loss_mask: Tensor
    attention_mask: Tensor
    legal_action_ids: LegalActionTensor | None
    game_ids: Tensor
    ply_indices: Tensor
    decision_columns: Tensor

    @property
    def history_floor(self) -> Tensor:
        """Return the earliest column of its own row each timestep may read.

        A timestep's own game starts where its column runs as far ahead of its
        ply index as it ever will, so the difference of the two indices names
        that column without a row having to carry its game boundaries. A game
        the row inherited from the batch before it, or a chunk that starts
        partway through one, subtracts to a negative column and clamps onto the
        row's first, which is the repeat a game's own opening plies read anyway.
        """

        return (self.decision_columns - self.ply_indices).clamp(min=0)

    @classmethod
    def from_sequence_batch(
        cls,
        batch: SequenceBatch,
        *,
        device: torch.device | str | None = None,
    ) -> MoveModelBatch:
        """Wrap the loader's arrays without changing target or mask alignment.

        Widening to what the model indexes with happens after the device copy
        rather than before it, so the copy carries the width the loader chose
        and the cast is a device kernel on what has already landed.

        The arrays are checked before any of that, which is the same check the
        result would have been held to.
        """

        _reject_invalid_batch(batch)
        tensor_device = torch.device(device) if device is not None else None

        def required(values: np.ndarray) -> Tensor:
            return torch.from_numpy(values).to(device=tensor_device).to(torch.long)

        def boolean(values: np.ndarray) -> Tensor:
            return torch.from_numpy(values).to(device=tensor_device)

        def optional(values: np.ndarray, present: np.ndarray) -> OptionalTensor:
            return OptionalTensor(required(values), boolean(present))

        inputs = batch.inputs
        return cls(
            inputs=MoveModelInputs(
                piece_ids=required(inputs.piece_ids),
                side_to_move=required(inputs.side_to_move),
                castling_rights=required(inputs.castling_rights),
                en_passant_token=required(inputs.en_passant_token),
                halfmove_clock=required(inputs.halfmove_clock),
                fullmove_number=required(inputs.fullmove_number),
                repetition_count=required(inputs.repetition_count),
                target_rating=optional(
                    inputs.target_rating.values,
                    inputs.target_rating.present,
                ),
            ),
            action_targets=required(batch.action_targets),
            action_loss_mask=boolean(batch.action_loss_mask),
            attention_mask=boolean(batch.attention_mask),
            legal_action_ids=batch.legal_action_ids,
            game_ids=torch.from_numpy(batch.game_ids).to(device=tensor_device),
            ply_indices=required(batch.ply_indices),
            decision_columns=_decision_columns(
                batch.action_targets.shape, tensor_device
            ),
        )

    @classmethod
    def from_decision_context(
        cls,
        context: DecisionContext,
        *,
        device: torch.device | str | None = None,
    ) -> MoveModelBatch:
        """Tensorize one target-free full history for its current decision."""

        return cls.from_decision_contexts((context,), device=device)

    @classmethod
    def from_decision_contexts(
        cls,
        contexts: Sequence[DecisionContext],
        *,
        device: torch.device | str | None = None,
    ) -> MoveModelBatch:
        """Tensorize several pending decisions into one padded batch.

        Games in flight are at different plies, so their histories cannot be
        stacked. Rows are padded to the longest history, and a caller reads each
        decision at its own length rather than at a shared last column.

        Padding goes after the history rather than before it, which is what
        keeps the boards behind a decision real: it reaches at or before its own
        column, and every column before a real one is itself real. So the row's
        last real timestep sees exactly what the same history would present
        alone.

        Each history arrives as the buffers it accumulated while it was played,
        so a row is a memory copy rather than a walk of the plies behind it and
        every column of every history crosses to the device together.
        """

        if not contexts:
            raise ValueError("a decision batch needs at least one context")
        tensor_device = torch.device(device) if device is not None else None
        histories = tuple(context.columns for context in contexts)
        lengths = tuple(history.length for history in histories)
        count = len(contexts)
        width = max(lengths)

        boards = np.zeros((count, width, BOARD_SQUARE_COUNT), dtype=np.uint8)
        packed = np.zeros((len(DecisionColumn), count, width), dtype=np.int64)
        attention = np.zeros((count, width), dtype=np.bool_)
        ratings = np.zeros((count, width), dtype=np.int64)
        rated = np.zeros((count, width), dtype=np.bool_)
        for index, history in enumerate(histories):
            length = history.length
            boards[index, :length] = np.frombuffer(
                history.piece_ids, dtype=np.uint8
            ).reshape(length, BOARD_SQUARE_COUNT)
            packed[:, index, :length] = (
                np.frombuffer(history.values, dtype=np.int64)
                .reshape(length, len(DecisionColumn))
                .T
            )
            attention[index, :length] = True
            # Every column of a served row is a decision by the player who
            # asked, so every one of them carries that player's rating. A
            # decision reads only its own column, so this matters to whichever
            # column a caller names rather than to the row.
            target_rating = contexts[index].target_rating
            if target_rating is not None:
                ratings[index, :length] = target_rating
                rated[index, :length] = True

        def transferred(values: np.ndarray) -> Tensor:
            return torch.from_numpy(values).to(device=tensor_device)

        # Already the width the model indexes with, so the columns cross once
        # and are read where they land. The board is the payload worth
        # narrowing and is widened on the far side of its own copy.
        crossed = transferred(packed)

        def column(name: DecisionColumn) -> Tensor:
            return crossed[name]

        result = cls(
            inputs=MoveModelInputs(
                piece_ids=transferred(boards).to(torch.long),
                side_to_move=column(DecisionColumn.SIDE_TO_MOVE),
                castling_rights=column(DecisionColumn.CASTLING_RIGHTS),
                en_passant_token=column(DecisionColumn.EN_PASSANT_TOKEN),
                halfmove_clock=column(DecisionColumn.HALFMOVE_CLOCK),
                fullmove_number=column(DecisionColumn.FULLMOVE_NUMBER),
                repetition_count=column(DecisionColumn.REPETITION_COUNT),
                target_rating=OptionalTensor(
                    transferred(ratings),
                    transferred(rated),
                ),
            ),
            action_targets=torch.zeros(
                (count, width), dtype=torch.long, device=tensor_device
            ),
            action_loss_mask=torch.zeros(
                (count, width), dtype=torch.bool, device=tensor_device
            ),
            attention_mask=transferred(attention),
            legal_action_ids=None,
            game_ids=torch.zeros(
                (count, width), dtype=torch.long, device=tensor_device
            ),
            ply_indices=column(DecisionColumn.PLY_INDEX),
            decision_columns=_decision_columns((count, width), tensor_device),
        )
        result.validate()
        return result

    def validate(self) -> None:
        """Hold a batch assembled any other way to what a factory guarantees.

        A batch given a new shape owes new decision columns; a stale one
        indexes past its own row.
        """

        if self.decision_columns.shape != self.action_targets.shape:
            raise ValueError("decision columns must align with targets")
        _reject_invalid_batch(self)


def _decision_columns(
    shape: tuple[int, ...],
    device: torch.device | None,
) -> Tensor:
    """Return the column each timestep of a batch of ``shape`` sits at."""

    return torch.arange(shape[1], device=device).expand(shape[0], shape[1])


def _read_together(columns: Sequence[Any]) -> list[Any]:
    """Return every column's values on the host, reading a device at most once.

    Asking a device column a question at a time would evaluate it in boolean
    context and drain the command queue between questions, so the columns are
    stacked and come back in one transfer. Columns that are already the
    loader's host arrays are read where they lie and cost no transfer at all.

    Stacking needs one dtype, and it is the widest of the columns rather than
    the first one's, so which column is passed first cannot decide what the
    rest come back as.
    """

    if not isinstance(columns[0], Tensor):
        return [column.tolist() for column in columns]
    dtype = columns[0].dtype
    for column in columns[1:]:
        dtype = torch.promote_types(dtype, column.dtype)
    stacked = torch.stack(tuple(column.to(dtype=dtype) for column in columns))
    return stacked.detach().cpu().tolist()


def _reject_invalid_batch(batch: _Batch) -> None:
    """Reject shapes or values that would corrupt model alignment.

    Written against column names rather than against a library. The loader's
    arrays and the tensors built from them carry the same fields and, because
    the copy only widens, the same values — so this answers the same questions
    of either, and :meth:`MoveModelBatch.from_sequence_batch` asks them on the
    side of the copy where they are cheap.
    """

    expected_shape = batch.action_targets.shape
    if batch.action_targets.ndim != 2:
        raise ValueError("action targets must have batch and sequence dimensions")
    if batch.inputs.piece_ids.shape != (*expected_shape, 64):
        raise ValueError("piece ids must align with targets and contain 64 squares")
    aligned = (
        batch.action_loss_mask,
        batch.attention_mask,
        batch.game_ids,
        batch.ply_indices,
        batch.inputs.side_to_move,
        batch.inputs.castling_rights,
        batch.inputs.en_passant_token,
        batch.inputs.halfmove_clock,
        batch.inputs.fullmove_number,
        batch.inputs.repetition_count,
    )
    if any(value.shape != expected_shape for value in aligned):
        raise ValueError("model inputs, targets, and masks must align")
    rating = batch.inputs.target_rating
    if rating.values.shape != expected_shape or rating.present.shape != expected_shape:
        raise ValueError("nullable model inputs must align with targets")
    _reject_invalid_values(batch)
    legal_action_ids = batch.legal_action_ids
    if legal_action_ids is None:
        return
    if len(legal_action_ids) != expected_shape[0] or any(
        len(row) != expected_shape[1] for row in legal_action_ids
    ):
        raise ValueError("legal actions must align with model timesteps")
    _reject_illegal_active_targets(batch, legal_action_ids)


def _reject_invalid_values(batch: _Batch) -> None:
    """Raise for any out-of-range value, naming the first rule that failed.

    Every check is a whole-column reduction queued before any answer is read,
    so the rules cost one pass each and their verdicts arrive together. A
    bounded column is held to its bounds by its own extremes rather than by a
    mask of the values outside them, which asks the same question without
    allocating a boolean copy of the widest column in the batch.
    """

    en_passant = batch.inputs.en_passant_token
    repetitions = batch.inputs.repetition_count
    ratings = batch.inputs.target_rating
    checks: tuple[tuple[str, Any], ...] = (
        (
            # A negative one lifts the history floor above its decision
            # column, and the gather reaches past its own row: a crash in the
            # forward pass, or a device-side assert that kills a run.
            "ply indices must be nonnegative",
            batch.ply_indices.min() < 0,
        ),
        (
            "action loss cannot include padded timesteps",
            (batch.action_loss_mask & ~batch.attention_mask).any(),
        ),
        (
            "padding must follow every real timestep in a row",
            (batch.attention_mask[:, :-1] < batch.attention_mask[:, 1:]).any(),
        ),
        (
            "piece ids are outside the board encoding",
            (batch.inputs.piece_ids.min() < 0) | (batch.inputs.piece_ids.max() >= 13),
        ),
        (
            "side-to-move ids are outside the board encoding",
            (batch.inputs.side_to_move.min() < 0)
            | (batch.inputs.side_to_move.max() >= 2),
        ),
        (
            "castling rights are outside the board encoding",
            (batch.inputs.castling_rights.min() < 0)
            | (batch.inputs.castling_rights.max() >= 16),
        ),
        (
            "en-passant tokens are outside the board encoding",
            (en_passant.min() < 0) | (en_passant.max() >= EN_PASSANT_TOKEN_COUNT),
        ),
        (
            "repetition counts are outside the board encoding",
            (repetitions.min() < 0) | (repetitions.max() >= REPETITION_STATE_COUNT),
        ),
        (
            "active action target is outside the action vocabulary",
            (
                batch.action_loss_mask
                & (
                    (batch.action_targets < 0)
                    | (batch.action_targets >= ACTION_VOCABULARY_SIZE)
                )
            ).any(),
        ),
        (
            "target ratings must be nonnegative",
            (ratings.present & (ratings.values < 0)).any(),
        ),
    )
    rejected = _read_together([flag for _, flag in checks])
    for (message, _), failed in zip(checks, rejected, strict=True):
        if failed:
            raise ValueError(message)


def _reject_illegal_active_targets(
    batch: _Batch,
    legal_action_ids: LegalActionTensor,
) -> None:
    """Raise when an enabled target is not legal at its own timestep.

    Legal actions are a host-side structure, so this comparison happens on the
    host either way. Both columns are read across together so that it costs one
    device read rather than one per timestep.
    """

    targets, enabled = _read_together((batch.action_targets, batch.action_loss_mask))
    for batch_index, row in enumerate(legal_action_ids):
        for sequence_index, legal_actions in enumerate(row):
            if (
                enabled[batch_index][sequence_index]
                and targets[batch_index][sequence_index] not in legal_actions
            ):
                raise ValueError("active target must be legal at its timestep")
