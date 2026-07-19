"""Code-owned configuration for the causal move model."""

from __future__ import annotations

from pydantic import Field, model_validator

from anthro_chess.config import ConfigModel


class MoveModelConfig(ConfigModel):
    """Hyperparameters for the first action-only causal model."""

    piece_embedding_dim: int = Field(default=8, ge=1)
    action_embedding_dim: int = Field(default=16, ge=1)
    model_dim: int = Field(default=64, ge=2)
    attention_heads: int = Field(default=4, ge=1)
    transformer_layers: int = Field(default=2, ge=1)
    feedforward_dim: int = Field(default=128, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_attention_shape(self) -> MoveModelConfig:
        """Require dimensions supported by multi-head attention and positions."""

        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.model_dim % 2:
            raise ValueError("model_dim must be even")
        return self
