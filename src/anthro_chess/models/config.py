"""Code-owned configuration for the move model."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from anthro_chess.config import ConfigModel


class MoveModelConfig(ConfigModel):
    """Hyperparameters for the action-only move model."""

    piece_embedding_dim: int = Field(default=8, ge=1)
    model_dim: int = Field(default=64, ge=2)
    attention_heads: int = Field(default=2, ge=1)
    #: Spatial layers over the 64 square tokens of one decision. Depth is held
    #: fixed and width is what grows, which is what every published Chessformer
    #: size runs.
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
    #: Whether a decision reads the game's time control and both clocks.
    clock_inputs: bool = False
    #: How often a training decision is shown each partial time context it can
    #: be served with: none at all, a control without a clock, or clocks
    #: without the initial one. The corpus presents almost none of these on its
    #: own.
    clock_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    #: How the rating reaches the trunk. ``line`` adds a vector placed on the
    #: segment between two anchors to every square token, so its length varies
    #: with the rating; ``arc`` turns between the anchors' directions at one
    #: learned length; ``modulation`` leaves the tokens alone and scales and
    #: shifts every block's normalized input instead.
    rating_path: Literal["line", "arc", "modulation"] = "line"
    #: ``post`` normalizes after each residual sum and not the embedding, as
    #: Chessformer does; ``pre`` normalizes the embedding and each sublayer's
    #: input.
    block_norm: Literal["pre", "post"] = "pre"

    @model_validator(mode="after")
    def validate_attention_shape(self) -> MoveModelConfig:
        """Require a width multi-head attention can split."""

        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.rating_path == "modulation" and self.block_norm == "post":
            raise ValueError("rating modulation reads a pre-norm block's input")
        return self
