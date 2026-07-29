"""Tensor conversion at the framework-neutral sequence-batch boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from anthro_chess.chess import ACTION_VOCABULARY_SIZE
from anthro_chess.data import DecisionContext, SequenceBatch
from anthro_chess.data.loading import LegalActionTensor


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
    """Tensor model boundary plus alignment metadata for inspection."""

    inputs: MoveModelInputs
    action_targets: Tensor
    action_loss_mask: Tensor
    attention_mask: Tensor
    causal_attention_mask: Tensor
    legal_action_ids: LegalActionTensor
    game_ids: Tensor
    ply_indices: Tensor
    chunk_start_plies: tuple[int, ...]

    @classmethod
    def from_sequence_batch(
        cls,
        batch: SequenceBatch,
        *,
        device: torch.device | str | None = None,
    ) -> MoveModelBatch:
        """Convert the loader output without changing target or mask alignment."""

        tensor_device = torch.device(device) if device is not None else None

        def required(values: object) -> Tensor:
            return torch.as_tensor(
                values,
                dtype=torch.long,
                device=tensor_device,
            )

        def boolean(values: object) -> Tensor:
            return torch.as_tensor(
                values,
                dtype=torch.bool,
                device=tensor_device,
            )

        def optional(values: object, present: object) -> OptionalTensor:
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
            causal_attention_mask=boolean(batch.causal_attention_mask),
            legal_action_ids=batch.legal_action_ids,
            game_ids=torch.as_tensor(
                batch.game_ids,
                dtype=torch.uint64,
                device=tensor_device,
            ),
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

        tensor_device = torch.device(device) if device is not None else None
        plies = context.plies
        length = len(plies)
        if tuple(ply.ply_index for ply in plies) != tuple(range(length)):
            raise ValueError(
                "decision context plies must be a complete zero-based history"
            )
        if plies[0].previous_action_id is not None or any(
            ply.previous_action_id is None for ply in plies[1:]
        ):
            raise ValueError(
                "decision context previous actions must align with history"
            )
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

        def required(values: object) -> Tensor:
            return torch.as_tensor(
                (values,),
                dtype=torch.long,
                device=tensor_device,
            )

        def optional(values: tuple[int | None, ...]) -> OptionalTensor:
            return OptionalTensor(
                required(tuple(value if value is not None else 0 for value in values)),
                torch.as_tensor(
                    (tuple(value is not None for value in values),),
                    dtype=torch.bool,
                    device=tensor_device,
                ),
            )

        ratings = (None,) * (length - 1) + (context.target_rating,)
        result = cls(
            inputs=MoveModelInputs(
                piece_ids=required(tuple(ply.board.piece_ids for ply in plies)),
                side_to_move=required(tuple(ply.board.side_to_move for ply in plies)),
                castling_rights=required(
                    tuple(ply.board.castling_rights for ply in plies)
                ),
                en_passant_square=optional(
                    tuple(ply.board.en_passant_square for ply in plies)
                ),
                halfmove_clock=required(
                    tuple(ply.board.halfmove_clock for ply in plies)
                ),
                fullmove_number=required(
                    tuple(ply.board.fullmove_number for ply in plies)
                ),
                previous_action_id=optional(
                    tuple(ply.previous_action_id for ply in plies)
                ),
                target_rating=optional(ratings),
            ),
            action_targets=torch.zeros(
                (1, length), dtype=torch.long, device=tensor_device
            ),
            action_loss_mask=torch.zeros(
                (1, length), dtype=torch.bool, device=tensor_device
            ),
            attention_mask=torch.ones(
                (1, length), dtype=torch.bool, device=tensor_device
            ),
            causal_attention_mask=torch.tril(
                torch.ones((length, length), dtype=torch.bool, device=tensor_device)
            ),
            legal_action_ids=(((),) * length,),
            game_ids=torch.zeros((1, length), dtype=torch.long, device=tensor_device),
            ply_indices=required(tuple(ply.ply_index for ply in plies)),
            chunk_start_plies=(plies[0].ply_index,),
        )
        result.validate()
        return result

    @classmethod
    def stack(cls, batches: Sequence[MoveModelBatch]) -> MoveModelBatch:
        """Combine equal-length batches into one wider batch.

        This is the narrow case: every input covers the same number of plies,
        so the rows concatenate with no padding and the causal mask is already
        shared. It exists for declared-batch throughput measurement, where the
        workload fixes one history length on purpose. Batching in-flight games
        of differing lengths is a padding problem rather than a stacking one
        and does not belong here.
        """

        if not batches:
            raise ValueError("stacking needs at least one batch")
        length = batches[0].action_targets.shape[1]
        if any(batch.action_targets.shape[1] != length for batch in batches):
            raise ValueError("stacked batches must cover the same sequence length")

        def joined(select: Callable[[MoveModelBatch], Tensor]) -> Tensor:
            return torch.cat([select(batch) for batch in batches], dim=0)

        def joined_optional(
            select: Callable[[MoveModelInputs], OptionalTensor],
        ) -> OptionalTensor:
            return OptionalTensor(
                torch.cat([select(batch.inputs).values for batch in batches], dim=0),
                torch.cat([select(batch.inputs).present for batch in batches], dim=0),
            )

        first = batches[0]
        result = cls(
            inputs=MoveModelInputs(
                piece_ids=joined(lambda batch: batch.inputs.piece_ids),
                side_to_move=joined(lambda batch: batch.inputs.side_to_move),
                castling_rights=joined(lambda batch: batch.inputs.castling_rights),
                en_passant_square=joined_optional(
                    lambda inputs: inputs.en_passant_square
                ),
                halfmove_clock=joined(lambda batch: batch.inputs.halfmove_clock),
                fullmove_number=joined(lambda batch: batch.inputs.fullmove_number),
                previous_action_id=joined_optional(
                    lambda inputs: inputs.previous_action_id
                ),
                target_rating=joined_optional(lambda inputs: inputs.target_rating),
            ),
            action_targets=joined(lambda batch: batch.action_targets),
            action_loss_mask=joined(lambda batch: batch.action_loss_mask),
            attention_mask=joined(lambda batch: batch.attention_mask),
            causal_attention_mask=first.causal_attention_mask,
            legal_action_ids=tuple(
                row for batch in batches for row in batch.legal_action_ids
            ),
            game_ids=joined(lambda batch: batch.game_ids),
            ply_indices=joined(lambda batch: batch.ply_indices),
            chunk_start_plies=tuple(
                start for batch in batches for start in batch.chunk_start_plies
            ),
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
        sequence_length = expected_shape[1]
        if self.causal_attention_mask.shape != (
            sequence_length,
            sequence_length,
        ):
            raise ValueError("causal attention mask must be sequence by sequence")
        if torch.any(torch.triu(self.causal_attention_mask, diagonal=1)):
            raise ValueError("causal attention mask cannot expose future timesteps")
        if torch.any(self.action_loss_mask & ~self.attention_mask):
            raise ValueError("action loss cannot include padded timesteps")
        if torch.any(self.inputs.piece_ids < 0) or torch.any(
            self.inputs.piece_ids >= 13
        ):
            raise ValueError("piece ids are outside the board encoding")
        if torch.any(self.inputs.side_to_move < 0) or torch.any(
            self.inputs.side_to_move >= 2
        ):
            raise ValueError("side-to-move ids are outside the board encoding")
        if torch.any(self.inputs.castling_rights < 0) or torch.any(
            self.inputs.castling_rights >= 16
        ):
            raise ValueError("castling rights are outside the board encoding")
        en_passant = self.inputs.en_passant_square
        if torch.any(en_passant.values[en_passant.present] < 0) or torch.any(
            en_passant.values[en_passant.present] >= 64
        ):
            raise ValueError("en-passant squares are outside the board encoding")
        previous_actions = self.inputs.previous_action_id
        if torch.any(
            previous_actions.values[previous_actions.present] < 0
        ) or torch.any(
            previous_actions.values[previous_actions.present] >= ACTION_VOCABULARY_SIZE
        ):
            raise ValueError("previous action is outside the action vocabulary")
        active_targets = self.action_targets[self.action_loss_mask]
        if torch.any(active_targets < 0) or torch.any(
            active_targets >= ACTION_VOCABULARY_SIZE
        ):
            raise ValueError("active action target is outside the action vocabulary")
        ratings = self.inputs.target_rating
        if torch.any(ratings.values[ratings.present] < 0):
            raise ValueError("target ratings must be nonnegative")
        if len(self.legal_action_ids) != expected_shape[0] or any(
            len(row) != expected_shape[1] for row in self.legal_action_ids
        ):
            raise ValueError("legal actions must align with model timesteps")
        for batch_index, row in enumerate(self.legal_action_ids):
            for sequence_index, legal_actions in enumerate(row):
                if (
                    self.action_loss_mask[batch_index, sequence_index]
                    and int(self.action_targets[batch_index, sequence_index].item())
                    not in legal_actions
                ):
                    raise ValueError("active target must be legal at its timestep")
