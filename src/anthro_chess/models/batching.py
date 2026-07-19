"""Tensor conversion at the framework-neutral sequence-batch boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from anthro_chess.chess import ACTION_VOCABULARY_SIZE
from anthro_chess.data import SequenceBatch
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
    player_rating: OptionalTensor
    opponent_rating: OptionalTensor


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
                player_rating=optional(
                    inputs.player_rating.values,
                    inputs.player_rating.present,
                ),
                opponent_rating=optional(
                    inputs.opponent_rating.values,
                    inputs.opponent_rating.present,
                ),
            ),
            action_targets=required(batch.action_targets),
            action_loss_mask=boolean(batch.action_loss_mask),
            attention_mask=boolean(batch.attention_mask),
            causal_attention_mask=boolean(batch.causal_attention_mask),
            legal_action_ids=batch.legal_action_ids,
            game_ids=required(batch.game_ids),
            ply_indices=required(batch.ply_indices),
            chunk_start_plies=batch.chunk_start_plies,
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
            self.inputs.player_rating,
            self.inputs.opponent_rating,
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
        if active_targets.numel() == 0:
            raise ValueError("model batch must contain at least one action target")
        if torch.any(active_targets < 0) or torch.any(
            active_targets >= ACTION_VOCABULARY_SIZE
        ):
            raise ValueError("active action target is outside the action vocabulary")
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
