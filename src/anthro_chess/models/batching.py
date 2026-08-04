"""Tensor conversion at the framework-neutral sequence-batch boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch
from torch import Tensor

from anthro_chess.chess import ACTION_VOCABULARY_SIZE
from anthro_chess.data import (
    BOARD_SQUARE_COUNT,
    DecisionContext,
    PlyContext,
    SequenceBatch,
)
from anthro_chess.data.loading import LegalActionTensor

#: What the alignment and range rules read. The loader's arrays and the tensors
#: built from them name their columns alike, so the rules are written once and
#: the type checker holds both families to every name they use.
_Batch: TypeAlias = "SequenceBatch | MoveModelBatch"


def _validated_plies(context: DecisionContext) -> tuple[PlyContext, ...]:
    """Return one decision history after rejecting misaligned inputs."""

    plies = context.plies
    if tuple(ply.ply_index for ply in plies) != tuple(range(len(plies))):
        raise ValueError("decision context plies must be a complete zero-based history")
    if plies[0].previous_action_id is not None or any(
        ply.previous_action_id is None for ply in plies[1:]
    ):
        raise ValueError("decision context previous actions must align with history")
    if any(
        value is not None
        for ply in plies
        for value in (
            ply.time_initial_ms,
            ply.time_increment_ms,
            ply.player_clock_ms,
            ply.opponent_clock_ms,
        )
    ):
        raise ValueError("the current move model does not support timing inputs")
    return plies


@dataclass(frozen=True)
class OptionalTensor:
    """Nullable integer values with an explicit presence mask."""

    values: Tensor
    present: Tensor


@dataclass(frozen=True)
class MoveModelInputs:
    """Tensorized exact state and context shaped batch by sequence."""

    piece_ids: Tensor
    side_to_move: Tensor
    castling_rights: Tensor
    en_passant_square: OptionalTensor
    halfmove_clock: Tensor
    fullmove_number: Tensor
    previous_action_id: OptionalTensor
    target_rating: OptionalTensor


@dataclass(frozen=True)
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
    chunk_start_plies: tuple[int, ...]

    @property
    def position_bound(self) -> int:
        """Return one past the furthest ply index this batch can hold.

        A row starts at its chunk's first ply and runs no further than the
        padded width, so this is the batch's own statement of how far its
        indices reach. :meth:`validate` holds ``ply_indices`` to it and the
        model refuses a batch whose reach passes the context length it
        declares. Between them a forward pass indexes a fixed position table in
        range, having read nothing back from the device.
        """

        return _position_bound(self)

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
                en_passant_square=optional(
                    inputs.en_passant_square.values,
                    inputs.en_passant_square.present,
                ),
                halfmove_clock=required(inputs.halfmove_clock),
                fullmove_number=required(inputs.fullmove_number),
                previous_action_id=optional(
                    inputs.previous_action_id.values,
                    inputs.previous_action_id.present,
                ),
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
            chunk_start_plies=batch.chunk_start_plies,
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
        stacked. Rows are padded to the longest history and the padded
        timesteps are marked absent in the attention mask, which is what the
        model reads to exclude them as attention keys.

        Padding goes after the history rather than before it. Every real ply
        then keeps the index it would have had on its own, and since the
        position encoding reads that index and the model lets a timestep attend
        only to earlier ones, the row's last real timestep sees exactly the
        inputs the same history would present alone. A caller reads each
        decision at its own history length rather than at a shared last
        column.
        """

        if not contexts:
            raise ValueError("a decision batch needs at least one context")
        tensor_device = torch.device(device) if device is not None else None
        histories = tuple(_validated_plies(context) for context in contexts)
        lengths = tuple(len(plies) for plies in histories)
        width = max(lengths)

        def padded(
            select: Callable[[PlyContext], int | None],
            fill: int | None = 0,
        ) -> tuple[tuple[int | None, ...], ...]:
            return tuple(
                tuple(select(ply) for ply in plies) + (fill,) * (width - len(plies))
                for plies in histories
            )

        def required(rows: tuple[tuple[int | None, ...], ...]) -> Tensor:
            return torch.as_tensor(rows, dtype=torch.long, device=tensor_device)

        def optional(rows: tuple[tuple[int | None, ...], ...]) -> OptionalTensor:
            return OptionalTensor(
                required(
                    tuple(
                        tuple(value if value is not None else 0 for value in row)
                        for row in rows
                    )
                ),
                torch.as_tensor(
                    tuple(tuple(value is not None for value in row) for row in rows),
                    dtype=torch.bool,
                    device=tensor_device,
                ),
            )

        boards = np.zeros((len(contexts), width, BOARD_SQUARE_COUNT), dtype=np.uint8)
        for index, plies in enumerate(histories):
            boards[index, : len(plies)] = np.frombuffer(
                b"".join([ply.board.piece_ids for ply in plies]),
                dtype=np.uint8,
            ).reshape(len(plies), BOARD_SQUARE_COUNT)
        ratings = tuple(
            (None,) * (length - 1)
            + (context.target_rating,)
            + (None,) * (width - length)
            for context, length in zip(contexts, lengths, strict=True)
        )
        result = cls(
            inputs=MoveModelInputs(
                piece_ids=torch.from_numpy(boards)
                .to(device=tensor_device)
                .to(torch.long),
                side_to_move=required(padded(lambda ply: ply.board.side_to_move)),
                castling_rights=required(padded(lambda ply: ply.board.castling_rights)),
                en_passant_square=optional(
                    padded(lambda ply: ply.board.en_passant_square, fill=None)
                ),
                halfmove_clock=required(padded(lambda ply: ply.board.halfmove_clock)),
                fullmove_number=required(padded(lambda ply: ply.board.fullmove_number)),
                previous_action_id=optional(
                    padded(lambda ply: ply.previous_action_id, fill=None)
                ),
                target_rating=optional(ratings),
            ),
            action_targets=torch.zeros(
                (len(contexts), width), dtype=torch.long, device=tensor_device
            ),
            action_loss_mask=torch.zeros(
                (len(contexts), width), dtype=torch.bool, device=tensor_device
            ),
            attention_mask=torch.as_tensor(
                tuple(
                    (True,) * length + (False,) * (width - length) for length in lengths
                ),
                dtype=torch.bool,
                device=tensor_device,
            ),
            legal_action_ids=None,
            game_ids=torch.zeros(
                (len(contexts), width), dtype=torch.long, device=tensor_device
            ),
            ply_indices=required(padded(lambda ply: ply.ply_index)),
            chunk_start_plies=tuple(plies[0].ply_index for plies in histories),
        )
        result.validate()
        return result

    def validate(self) -> None:
        """Hold a batch assembled any other way to what a factory guarantees."""

        _reject_invalid_batch(self)


def _position_bound(batch: _Batch) -> int:
    """Return one past the furthest ply index a batch's rows can hold."""

    return max(batch.chunk_start_plies) + batch.action_targets.shape[1]


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
        batch.inputs.halfmove_clock,
        batch.inputs.fullmove_number,
    )
    if any(value.shape != expected_shape for value in aligned):
        raise ValueError("model inputs, targets, and masks must align")
    optional_inputs = (
        batch.inputs.en_passant_square,
        batch.inputs.previous_action_id,
        batch.inputs.target_rating,
    )
    if any(
        item.values.shape != expected_shape or item.present.shape != expected_shape
        for item in optional_inputs
    ):
        raise ValueError("nullable model inputs must align with targets")
    if len(batch.chunk_start_plies) != expected_shape[0]:
        raise ValueError("chunk start plies must name every sequence in the batch")
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

    en_passant = batch.inputs.en_passant_square
    previous_actions = batch.inputs.previous_action_id
    ratings = batch.inputs.target_rating
    # A negative index would gather real position features from the wrong
    # end of the table rather than fail, so the lower bound is checked even
    # though no factory can produce one.
    bound = _position_bound(batch)
    checks: tuple[tuple[str, Any], ...] = (
        (
            "ply indices must lie inside the plies the batch declares",
            (batch.ply_indices.min() < 0) | (batch.ply_indices.max() >= bound),
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
            "en-passant squares are outside the board encoding",
            (
                en_passant.present
                & ((en_passant.values < 0) | (en_passant.values >= 64))
            ).any(),
        ),
        (
            "previous action is outside the action vocabulary",
            (
                previous_actions.present
                & (
                    (previous_actions.values < 0)
                    | (previous_actions.values >= ACTION_VOCABULARY_SIZE)
                )
            ).any(),
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
