"""Code-owned configuration for the causal move model."""

from __future__ import annotations

from pydantic import Field, model_validator

from anthro_chess.config import ConfigModel


class MoveModelConfig(ConfigModel):
    """Hyperparameters for the action-only causal move model."""

    piece_embedding_dim: int = Field(default=8, ge=1)
    action_embedding_dim: int = Field(default=16, ge=1)
    model_dim: int = Field(default=64, ge=2)
    attention_heads: int = Field(default=4, ge=1)
    #: Spatial layers over the 64 square tokens of one position, run before the
    #: ply trunk sees that position at all.
    spatial_layers: int = Field(default=2, ge=1)
    #: Causal layers over the ply axis. The only stage that reads more than one
    #: position, and the only one whose cost grows with game length.
    transformer_layers: int = Field(default=2, ge=1)
    #: Spatial layers that read the trunk's history feature back onto the square
    #: tokens, so the move head scores squares that know the history.
    decision_layers: int = Field(default=1, ge=1)
    feedforward_dim: int = Field(default=128, ge=1)
    #: Width each square token is compressed to before the geometric bias
    #: generator flattens the board into one vector.
    geometric_token_dim: int = Field(default=8, ge=1)
    #: Width of the geometric bias generator's hidden stages, and the number of
    #: 64-by-64 bias templates its output layer mixes. Every template is 4096
    #: values, so this is the one width whose cost does not fall with
    #: ``model_dim``: left at a large model's setting it would dominate a small
    #: model's parameter count outright.
    geometric_bias_dim: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    #: One past the furthest ply index the model can encode, and the length of
    #: the position table derived from it. A shape assertion rather than a dial:
    #: the longest game in the million-game blitz corpus is 306 plies, which
    #: this covers at every chunk length. ``data.maximum_position_bound`` owns
    #: what a corpus actually reaches, and a training run refuses to start when
    #: that passes this. It costs one row of that table per ply — attention is
    #: quadratic in the padded batch width, which this does not set.
    maximum_context_plies: int = Field(default=1024, ge=1)

    @model_validator(mode="after")
    def validate_attention_shape(self) -> MoveModelConfig:
        """Require dimensions supported by multi-head attention and positions."""

        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.model_dim % 2:
            raise ValueError("model_dim must be even")
        return self
