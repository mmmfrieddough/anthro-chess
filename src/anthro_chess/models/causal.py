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
_SCALAR_CONTEXT_COUNT = 4


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
            torch.zeros_like(inputs.en_passant_square.values),
        )
        rule_counts = torch.stack(
            (
                torch.log1p(inputs.halfmove_clock.float()),
                torch.log1p(inputs.fullmove_number.float()),
            ),
            dim=-1,
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


class CausalMoveModel(nn.Module):
    """Predict human action logits from exact state and causal trajectory."""

    def __init__(self, config: MoveModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or MoveModelConfig()
        self.board_encoder = BoardEncoder(self.config)
        self.previous_action_embedding = nn.Embedding(
            ACTION_VOCABULARY_SIZE + 1,
            self.config.action_embedding_dim,
        )
        context_input_dim = (
            self.config.model_dim
            + self.config.action_embedding_dim
            + _SCALAR_CONTEXT_COUNT
        )
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
        self.action_head = nn.Linear(
            self.config.model_dim,
            ACTION_VOCABULARY_SIZE,
        )

    def forward(self, batch: MoveModelBatch) -> Tensor:
        """Return raw action logits shaped batch by sequence by vocabulary."""

        batch.validate()
        inputs = batch.inputs
        previous_action_tokens = torch.where(
            inputs.previous_action_id.present,
            inputs.previous_action_id.values,
            torch.full_like(
                inputs.previous_action_id.values,
                ACTION_VOCABULARY_SIZE,
            ),
        )
        context = torch.cat(
            (
                self.board_encoder(batch),
                self.previous_action_embedding(previous_action_tokens),
                self._scalar_context(batch),
            ),
            dim=-1,
        )
        hidden = self.context_combiner(context)
        hidden = hidden + _sinusoidal_positions(
            batch.ply_indices,
            self.config.model_dim,
            hidden.dtype,
        )
        hidden = self.transformer(
            hidden,
            mask=~batch.causal_attention_mask,
            src_key_padding_mask=~batch.attention_mask,
        )
        logits = self.action_head(hidden)
        return cast(
            Tensor,
            logits.masked_fill(~batch.attention_mask.unsqueeze(-1), 0.0),
        )

    def identity(self) -> dict[str, object]:
        """Return compatibility metadata for future runs and checkpoints."""

        return {
            "name": "anthro-causal-move-model",
            "version": 1,
            "config": self.config.model_dump(mode="json"),
            "action_vocabulary": action_vocabulary_identity(),
            "encoding": encoding_identity(),
            "timing_inputs": False,
            "timing_head": False,
        }

    @staticmethod
    def _scalar_context(batch: MoveModelBatch) -> Tensor:
        inputs = batch.inputs
        nullable = (
            inputs.player_rating,
            inputs.opponent_rating,
        )
        values = tuple(_nullable_log_value(item) for item in nullable)
        presence = tuple(item.present.float() for item in nullable)
        return torch.stack(
            (
                *values,
                *presence,
            ),
            dim=-1,
        )


def _nullable_log_value(value: OptionalTensor) -> Tensor:
    return torch.where(
        value.present,
        torch.log1p(value.values.float()),
        torch.zeros_like(value.values, dtype=torch.float),
    )


def _sinusoidal_positions(
    ply_indices: Tensor,
    dimension: int,
    dtype: torch.dtype,
) -> Tensor:
    frequencies = torch.exp(
        torch.arange(
            0,
            dimension,
            2,
            device=ply_indices.device,
            dtype=dtype,
        )
        * (-math.log(10_000.0) / dimension)
    )
    angles = ply_indices.to(dtype).unsqueeze(-1) * frequencies
    positions = torch.zeros(
        (*ply_indices.shape, dimension),
        device=ply_indices.device,
        dtype=dtype,
    )
    positions[..., 0::2] = torch.sin(angles)
    positions[..., 1::2] = torch.cos(angles)
    return positions
