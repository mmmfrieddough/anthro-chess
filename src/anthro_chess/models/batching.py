"""Tensor conversion at the framework-neutral sequence-batch boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

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

    Every factory here validates what it built, so a batch a factory returned is
    a batch that passed and the model rechecks nothing.
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

        return max(self.chunk_start_plies) + self.action_targets.shape[1]

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
        """

        tensor_device = torch.device(device) if device is not None else None

        def required(values: np.ndarray) -> Tensor:
            return torch.from_numpy(values).to(device=tensor_device).to(torch.long)

        def boolean(values: np.ndarray) -> Tensor:
            return torch.from_numpy(values).to(device=tensor_device)

        def optional(values: np.ndarray, present: np.ndarray) -> OptionalTensor:
            return OptionalTensor(required(values), boolean(present))

        inputs = batch.inputs
        result = cls(
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
        result.validate()
        return result

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
        """Reject shapes or values that would corrupt model alignment."""

        expected_shape = self.action_targets.shape
        if self.action_targets.ndim != 2:
            raise ValueError("action targets must have batch and sequence dimensions")
        if self.inputs.piece_ids.shape != (*expected_shape, 64):
            raise ValueError("piece ids must align with targets and contain 64 squares")
        aligned = (
            self.action_loss_mask,
            self.attention_mask,
            self.game_ids,
            self.ply_indices,
            self.inputs.side_to_move,
            self.inputs.castling_rights,
            self.inputs.halfmove_clock,
            self.inputs.fullmove_number,
        )
        if any(value.shape != expected_shape for value in aligned):
            raise ValueError("model inputs, targets, and masks must align")
        optional_inputs = (
            self.inputs.en_passant_square,
            self.inputs.previous_action_id,
            self.inputs.target_rating,
        )
        if any(
            item.values.shape != expected_shape or item.present.shape != expected_shape
            for item in optional_inputs
        ):
            raise ValueError("nullable model inputs must align with targets")
        if len(self.chunk_start_plies) != expected_shape[0]:
            raise ValueError("chunk start plies must name every sequence in the batch")
        self._reject_invalid_values()
        legal_action_ids = self.legal_action_ids
        if legal_action_ids is None:
            return
        if len(legal_action_ids) != expected_shape[0] or any(
            len(row) != expected_shape[1] for row in legal_action_ids
        ):
            raise ValueError("legal actions must align with model timesteps")
        self._reject_illegal_active_targets(legal_action_ids)

    def _reject_invalid_values(self) -> None:
        """Raise for any out-of-range value, reading the device exactly once.

        Every check is a whole-tensor reduction, so asking each one separately
        would evaluate a device tensor in boolean context and drain the command
        queue between them. The reductions are queued together and read back in
        one transfer instead; the first failure still names itself.
        """

        en_passant = self.inputs.en_passant_square
        previous_actions = self.inputs.previous_action_id
        ratings = self.inputs.target_rating
        # A negative index would gather real position features from the wrong
        # end of the table rather than fail, so the lower bound is checked even
        # though no factory can produce one.
        bound = self.position_bound
        checks: tuple[tuple[str, Tensor], ...] = (
            (
                "ply indices must lie inside the plies the batch declares",
                torch.any((self.ply_indices < 0) | (self.ply_indices >= bound)),
            ),
            (
                "action loss cannot include padded timesteps",
                torch.any(self.action_loss_mask & ~self.attention_mask),
            ),
            (
                "padding must follow every real timestep in a row",
                torch.any(self.attention_mask[:, :-1] < self.attention_mask[:, 1:]),
            ),
            (
                "piece ids are outside the board encoding",
                torch.any((self.inputs.piece_ids < 0) | (self.inputs.piece_ids >= 13)),
            ),
            (
                "side-to-move ids are outside the board encoding",
                torch.any(
                    (self.inputs.side_to_move < 0) | (self.inputs.side_to_move >= 2)
                ),
            ),
            (
                "castling rights are outside the board encoding",
                torch.any(
                    (self.inputs.castling_rights < 0)
                    | (self.inputs.castling_rights >= 16)
                ),
            ),
            (
                "en-passant squares are outside the board encoding",
                torch.any(
                    en_passant.present
                    & ((en_passant.values < 0) | (en_passant.values >= 64))
                ),
            ),
            (
                "previous action is outside the action vocabulary",
                torch.any(
                    previous_actions.present
                    & (
                        (previous_actions.values < 0)
                        | (previous_actions.values >= ACTION_VOCABULARY_SIZE)
                    )
                ),
            ),
            (
                "active action target is outside the action vocabulary",
                torch.any(
                    self.action_loss_mask
                    & (
                        (self.action_targets < 0)
                        | (self.action_targets >= ACTION_VOCABULARY_SIZE)
                    )
                ),
            ),
            (
                "target ratings must be nonnegative",
                torch.any(ratings.present & (ratings.values < 0)),
            ),
        )
        rejected = torch.stack([flag for _, flag in checks]).detach().cpu().tolist()
        for (message, _), failed in zip(checks, rejected, strict=True):
            if failed:
                raise ValueError(message)

    def _reject_illegal_active_targets(
        self,
        legal_action_ids: LegalActionTensor,
    ) -> None:
        """Raise when an enabled target is not legal at its own timestep.

        Legal actions are a host-side structure, so this comparison happens on
        the host either way. Both tensors are read across in one transfer so
        that it costs one device read rather than one per timestep.
        """

        targets, enabled = (
            torch.stack(
                (
                    self.action_targets.to(dtype=torch.long),
                    self.action_loss_mask.to(dtype=torch.long),
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        for batch_index, row in enumerate(legal_action_ids):
            for sequence_index, legal_actions in enumerate(row):
                if (
                    enabled[batch_index][sequence_index]
                    and targets[batch_index][sequence_index] not in legal_actions
                ):
                    raise ValueError("active target must be legal at its timestep")
