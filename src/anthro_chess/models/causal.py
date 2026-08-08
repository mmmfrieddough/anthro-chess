"""Minimal action-only causal model over exact per-ply chess context."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    action_vocabulary_identity,
)
from anthro_chess.data import (
    EN_PASSANT_TOKEN_COUNT,
    PREVIOUS_ACTION_TOKEN_COUNT,
    encoding_identity,
)
from anthro_chess.models.batching import MoveModelBatch, OptionalTensor
from anthro_chess.models.config import MoveModelConfig

_PIECE_ID_COUNT = 13
_SIDE_TO_MOVE_COUNT = 2
_CASTLING_RIGHTS_COUNT = 16
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
            EN_PASSANT_TOKEN_COUNT,
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
        # A transform rather than a token vocabulary, so it stays here while
        # the two nullable inputs moved. Decision 0035 says why.
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
                self.en_passant_embedding(inputs.en_passant_token),
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


class CausalSelfAttention(nn.Module):
    """Attend over earlier timesteps, stating causality as the flag it is.

    ``scaled_dot_product_attention`` reads ``is_causal`` as the mask rather than
    as a hint accompanying one, which is what the encoder wrapper this replaced
    could not do. Flash and cuDNN attention refuse a non-null ``attn_mask``, so
    handing one over is what put them out of reach.
    """

    def __init__(self, config: MoveModelConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.model_dim // config.attention_heads
        self.dropout = config.dropout
        self.qkv_projection = nn.Linear(config.model_dim, 3 * config.model_dim)
        self.output_projection = nn.Linear(config.model_dim, config.model_dim)
        # What the wrapper's attention initialized itself to. ``nn.Linear``'s
        # own default is a generic one, and restating this is what keeps a
        # fresh model drawn the way it was before.
        nn.init.xavier_uniform_(self.qkv_projection.weight)
        nn.init.zeros_(self.qkv_projection.bias)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        """Return attended features for every timestep, batch dimension first."""

        query, key, value = (
            self.qkv_projection(hidden)
            .unflatten(-1, (3, self.heads, self.head_dim))
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return cast(
            Tensor,
            self.output_projection(attended.transpose(1, 2).flatten(-2)),
        )


class TransformerBlock(nn.Module):
    """One pre-norm residual pair: causal attention, then a feed-forward."""

    def __init__(self, config: MoveModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.attention = CausalSelfAttention(config)
        self.feedforward_norm = nn.LayerNorm(config.model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dim, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.model_dim),
        )
        # Dropout carries no state, so both residuals read the same module.
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        """Return the block's output for a whole batch of timesteps."""

        hidden = hidden + self.residual_dropout(
            self.attention(self.attention_norm(hidden))
        )
        hidden = hidden + self.residual_dropout(
            self.feedforward(self.feedforward_norm(hidden))
        )
        return hidden


class CausalMoveModel(nn.Module):
    """Predict human action logits from exact state and causal trajectory."""

    position_table: Tensor

    def __init__(self, config: MoveModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or MoveModelConfig()
        self.board_encoder = BoardEncoder(self.config)
        self.previous_action_embedding = nn.Embedding(
            PREVIOUS_ACTION_TOKEN_COUNT,
            self.config.action_embedding_dim,
        )
        context_input_dim = self.config.model_dim + self.config.action_embedding_dim
        self.context_combiner = nn.Sequential(
            nn.Linear(context_input_dim, self.config.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.model_dim),
        )
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(self.config) for _ in range(self.config.transformer_layers)
        )
        # A pre-norm block leaves its output unnormalized for whatever follows,
        # so the stack ends with one. Named rather than the last element of a
        # sequence, because a checkpoint key that moves with the layer count
        # cannot be read without also reading the configuration.
        self.transformer_norm = nn.LayerNorm(self.config.model_dim)
        self.rating_conditioner = RatingConditioner(self.config)
        self.action_head = nn.Linear(
            self.config.model_dim,
            ACTION_VOCABULARY_SIZE,
        )
        # A derived constant, not state: a pure function of the configuration,
        # so it does not belong in a checkpoint.
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
        context = torch.cat(
            (
                self.board_encoder(batch),
                self.previous_action_embedding(batch.inputs.previous_action_token),
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
        for block in self.transformer_blocks:
            hidden = block(hidden)
        return cast(Tensor, self.transformer_norm(hidden))

    def identity(self) -> dict[str, object]:
        """Return compatibility metadata for future runs and checkpoints."""

        return model_identity(self.config)


def model_identity(config: MoveModelConfig) -> dict[str, object]:
    """Return the compatibility metadata a model built from ``config`` carries.

    The whole configuration is carried, so a checkpoint says what to rebuild
    without a second record having to supply the rest. Anything left out here
    is a value the runner cannot recover, and it would then rebuild the model
    at that field's default instead of the run's.

    A function of the configuration rather than of a built model, because
    every field here is one: a caller comparing an artifact against what this
    code would produce should not have to allocate a network to do it.
    """

    return {
        "name": "anthro-causal-move-model",
        # Version 5 renamed every transformer parameter, so this is what
        # refuses an older checkpoint by name rather than letting it fail as
        # missing state-dict keys.
        "version": 5,
        "config": config.model_dump(mode="json"),
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
