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
        # Derived constants, not state: a pure function of the configuration and
        # the longest sequence seen, so neither belongs in a checkpoint. Both are
        # built under `torch.inference_mode(False)`, because the first caller may
        # be a benchmark scoring under `torch.inference_mode` and a tensor made
        # there can never afterwards join a training backward pass. `#227` would
        # retire the laziness by declaring a maximum context length.
        self._cached_causal_mask: Tensor | None = None
        self._cached_position_table: Tensor | None = None

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
            ),
            dim=-1,
        )
        hidden = self.context_combiner(context)
        hidden = hidden + self._positions(batch, hidden.dtype)
        # No key padding mask. Padding is right-aligned, so a real query attends
        # only to keys that are themselves real, and a padded query's output is
        # discarded by target and by loss mask downstream. Leaving it out also
        # removes the all-padding-row hazard a key padding mask carries.
        hidden = self.transformer(
            hidden,
            mask=self._causal_mask(hidden.shape[1], hidden.device, hidden.dtype),
            is_causal=True,
        )
        return cast(Tensor, hidden)

    def _causal_mask(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return the additive mask that hides every future timestep.

        The mask states only that a query cannot attend past itself, which is a
        property of this model rather than of any batch. It is held at the
        longest length seen and sliced for shorter ones, so a step neither
        builds it in Python nor copies it to the device.

        Additive rather than boolean, and in the attention input's own dtype,
        because a boolean mask is converted to exactly this one on the way in:
        ``F._canonical_mask`` writes a fresh length-by-length float tensor per
        forward pass, which is 1.44 MB at 600 plies. Caching the boolean form
        would avoid rebuilding a tensor the framework then rebuilds anyway.

        ``is_causal=True`` does not stand in for a mask. The encoder reads the
        flag as a hint accompanying one, and refuses it alone.
        """

        cached = self._cached_causal_mask
        if (
            cached is None
            or cached.shape[0] < length
            or cached.device != device
            or cached.dtype != dtype
        ):
            with torch.inference_mode(False):
                cached = nn.Transformer.generate_square_subsequent_mask(
                    length,
                    device=device,
                    dtype=dtype,
                )
            self._cached_causal_mask = cached
        return cached[:length, :length]

    def _positions(self, batch: MoveModelBatch, dtype: torch.dtype) -> Tensor:
        """Return each timestep's sinusoidal features for its own ply index.

        The features depend only on the ply index and the model width, so the
        table is held and gathered rather than recomputed every forward pass.
        How far it must reach is the batch's own
        :attr:`~MoveModelBatch.position_bound`, which :meth:`MoveModelBatch.validate`
        has already held its indices to.
        """

        indices = batch.ply_indices
        bound = batch.position_bound
        table = self._cached_position_table
        if (
            table is None
            or table.shape[0] < bound
            or table.device != indices.device
            or table.dtype != dtype
        ):
            with torch.inference_mode(False):
                table = _sinusoidal_table(
                    bound,
                    self.config.model_dim,
                    device=indices.device,
                    dtype=dtype,
                )
            self._cached_position_table = table
        return table[indices]

    def identity(self) -> dict[str, object]:
        """Return compatibility metadata for future runs and checkpoints."""

        return {
            "name": "anthro-causal-move-model",
            "version": 3,
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
        torch.zeros_like(value.values, dtype=torch.float),
    )


def _sinusoidal_table(
    length: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=dtype)
        * (-math.log(10_000.0) / dimension)
    )
    angles = (
        torch.arange(length, device=device, dtype=dtype).unsqueeze(-1) * frequencies
    )
    return torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(-2)
