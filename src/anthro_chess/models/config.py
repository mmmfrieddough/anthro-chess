"""Code-owned configuration for the move model."""

from __future__ import annotations

from pydantic import Field, model_validator

from anthro_chess.config import ConfigModel


class MoveModelConfig(ConfigModel):
    """Hyperparameters for the action-only move model."""

    piece_embedding_dim: int = Field(default=8, ge=1)
    model_dim: int = Field(default=64, ge=2)
    attention_heads: int = Field(default=2, ge=1)
    #: Spatial layers over the 64 square tokens of one decision. Depth is held
    #: fixed and width is what grows, which is what every published
    #: Chessformer size runs; `docs/scaling.md` owns the rule.
    layers: int = Field(default=8, ge=1)
    feedforward_dim: int = Field(default=128, ge=1)
    #: Boards stacked into each square token's depth, the decision's own
    #: included. Seven prior boards is Chessformer's default and their ablation
    #: reads thirty-one as worth nothing further.
    history_positions: int = Field(default=8, ge=1)
    #: How often a decision is trained on a truncated history, so that the
    #: short histories a game's opening plies present are not out of
    #: distribution.
    history_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    #: Width each square token is compressed to before the geometric bias
    #: generator flattens the board into one vector.
    geometric_token_dim: int = Field(default=8, ge=1)
    #: Width of the geometric bias generator's hidden stages, and the number of
    #: 64-by-64 bias templates the shared template bank holds. Every template
    #: is 4096 values whatever the model width is, so this width's cost does
    #: not fall with ``model_dim``. The bank is shared by every layer, which is
    #: what keeps depth from multiplying it.
    geometric_bias_dim: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_attention_shape(self) -> MoveModelConfig:
        """Require a width multi-head attention can split."""

        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        return self
