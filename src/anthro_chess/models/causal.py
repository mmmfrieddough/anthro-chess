"""Minimal action-only causal model over exact per-ply chess context."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    action_vocabulary_identity,
)
from anthro_chess.data import encoding_identity
from anthro_chess.models.batching import MoveModelBatch, OptionalTensor
from anthro_chess.models.config import MoveModelConfig

_PIECE_ID_COUNT = 13
_SIDE_TO_MOVE_COUNT = 2
_CASTLING_RIGHTS_COUNT = 16
_EN_PASSANT_TOKEN_COUNT = 65
_RATING_CONTEXT_COUNT = 2


class BoardEncoder(nn.Module):
    """Learn an embedding of the exact pre-move standard-chess state."""

    def __init__(self, config: MoveModelConfig) -> None:
        super().__init__()
        embedding_dim = config.piece_embedding_dim
        self.piece_embedding = nn.Embedding(_PIECE_ID_COUNT, embedding_dim)
        self.side_embedding = nn.Embedding(_SIDE_TO_MOVE_COUNT, embedding_dim)
        self.castling_embedding = nn.Embedding(_CASTLING_RIGHTS_COUNT, embedding_dim)
        self.en_passant_embedding = nn.Embedding(
            _EN_PASSANT_TOKEN_COUNT,
            embedding_dim,
        )
        input_dim = (64 + 3) * embedding_dim + 2
        self.projection = nn.Sequential(
            nn.Linear(input_dim, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )

    def forward(self, batch: MoveModelBatch) -> Tensor:
        """Encode board pieces and rule state for every timestep."""

        inputs = batch.inputs
        piece_features = self.piece_embedding(inputs.piece_ids).flatten(-2)
        en_passant_tokens = torch.where(
            inputs.en_passant_square.present,
            inputs.en_passant_square.values + 1,
            0,
        )
        rule_counts = torch.log1p(
            torch.stack(
                (
                    inputs.halfmove_clock.float(),
                    inputs.fullmove_number.float(),
                ),
                dim=-1,
            )
        )
        features = torch.cat(
            (
                piece_features,
                self.side_embedding(inputs.side_to_move),
                self.castling_embedding(inputs.castling_rights),
                self.en_passant_embedding(en_passant_tokens),
                rule_counts,
            ),
            dim=-1,
        )
        return cast(Tensor, self.projection(features))


class RatingConditioner(nn.Module):
    """Modulate completed history features for one decision-maker rating."""

    def __init__(self, config: MoveModelConfig) -> None:
        super().__init__()
        self.modulation = nn.Sequential(
            nn.Linear(_RATING_CONTEXT_COUNT, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim * 2),
        )
        self.normalization = nn.LayerNorm(config.model_dim)

    def forward(self, hidden: Tensor, rating: OptionalTensor) -> Tensor:
        """Apply nonlinear, position-dependent rating feature modulation."""

        rating_features = torch.stack(
            (
                _nullable_log_value(rating),
                rating.present.float(),
            ),
            dim=-1,
        )
        scale, shift = self.modulation(rating_features).chunk(2, dim=-1)
        return cast(
            Tensor,
            self.normalization(hidden * (1.0 + torch.tanh(scale)) + shift),
        )


class CausalMoveModel(nn.Module):
    """Predict human action logits from exact state and causal trajectory."""

    causal_mask: Tensor
    position_table: Tensor

    def __init__(self, config: MoveModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or MoveModelConfig()
        self.board_encoder = BoardEncoder(self.config)
        self.previous_action_embedding = nn.Embedding(
            ACTION_VOCABULARY_SIZE + 1,
            self.config.action_embedding_dim,
        )
        context_input_dim = self.config.model_dim + self.config.action_embedding_dim
        self.context_combiner = nn.Sequential(
            nn.Linear(context_input_dim, self.config.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.model_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.model_dim,
            nhead=self.config.attention_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=self.config.transformer_layers,
            norm=nn.LayerNorm(self.config.model_dim),
            enable_nested_tensor=False,
        )
        self.rating_conditioner = RatingConditioner(self.config)
        self.action_head = nn.Linear(
            self.config.model_dim,
            ACTION_VOCABULARY_SIZE,
        )
        # Derived constants, not state: a pure function of the configuration, so
        # neither belongs in a checkpoint. The mask is additive rather than
        # boolean because attention converts a boolean one to exactly this on
        # the way in, once per forward pass.
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(
                self.config.maximum_context_plies
            ),
            persistent=False,
        )
        self.register_buffer(
            "position_table",
            _sinusoidal_table(
                self.config.maximum_context_plies,
                self.config.model_dim,
            ),
            persistent=False,
        )

    def forward(self, batch: MoveModelBatch) -> Tensor:
        """Return raw action logits shaped batch by sequence by vocabulary.

        Every batch is validated by whichever factory built it, so nothing is
        rechecked here. Padded timesteps produce ordinary finite logits that no
        caller reads: the loss ignores them by target, and every scoring path
        gathers by the loss mask before looking.
        """

        hidden = self.encode_history(batch)
        conditioned = self.rating_conditioner(hidden, batch.inputs.target_rating)
        return cast(Tensor, self.action_head(conditioned))

    def encode_history(self, batch: MoveModelBatch) -> Tensor:
        """Encode rating-neutral exact state and causal move history."""

        declared = self.config.maximum_context_plies
        if batch.position_bound > declared:
            raise ValueError(
                f"batch reaches ply index {batch.position_bound - 1}, past the "
                f"{declared} plies this model declares as its context"
            )
        inputs = batch.inputs
        previous_action_tokens = torch.where(
            inputs.previous_action_id.present,
            inputs.previous_action_id.values,
            ACTION_VOCABULARY_SIZE,
        )
        context = torch.cat(
            (
                self.board_encoder(batch),
                self.previous_action_embedding(previous_action_tokens),
            ),
            dim=-1,
        )
        hidden = self.context_combiner(context)
        # A chunked selection carries ply indices past the row's own width.
        hidden = hidden + self.position_table[batch.ply_indices].to(hidden.dtype)
        # No key padding mask. Padding is right-aligned, so a real query attends
        # only to keys that are themselves real, and a padded query's output is
        # discarded by target and by loss mask downstream. Leaving it out also
        # removes the all-padding-row hazard a key padding mask carries.
        #
        # ``is_causal=True`` does not stand in for the mask. The encoder reads
        # the flag as a hint accompanying one, and refuses it alone.
        width = hidden.shape[1]
        hidden = self.transformer(
            hidden,
            mask=self.causal_mask[:width, :width].to(hidden.dtype),
            is_causal=True,
        )
        return cast(Tensor, hidden)

    def identity(self) -> dict[str, object]:
        """Return compatibility metadata for future runs and checkpoints.

        The whole configuration is carried, so a checkpoint says what to
        rebuild without a second record having to supply the rest. Anything
        left out here is a value the runner cannot recover, and it would then
        rebuild the model at that field's default instead of the run's.
        """

        return {
            "name": "anthro-causal-move-model",
            "version": 4,
            "config": self.config.model_dump(mode="json"),
            "action_vocabulary": action_vocabulary_identity(),
            "encoding": encoding_identity(),
            "rating_conditioning": "post-transformer-feature-modulation",
            "timing_inputs": False,
            "timing_head": False,
        }


def _nullable_log_value(value: OptionalTensor) -> Tensor:
    return torch.where(
        value.present,
        torch.log1p(value.values.float()),
        0.0,
    )


def _sinusoidal_table(length: int, dimension: int) -> Tensor:
    dtype = torch.get_default_dtype()
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, dtype=dtype) * (-math.log(10_000.0) / dimension)
    )
    angles = torch.arange(length, dtype=dtype).unsqueeze(-1) * frequencies
    return torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(-2)
