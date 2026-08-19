"""The smallest model configuration the suite builds, in one place.

Most fixtures want a model that loads and runs and is otherwise as cheap as it
goes. Written out per fixture, that meant every new width had to be added to
each of them, and the geometric bias generator is the width where omitting it
stops being cosmetic: it emits a 64-by-64 template per head whatever
``model_dim`` is, so a fixture that leaves it at its default builds a model an
order of magnitude larger than the four-wide one it is asking for.

``history_positions`` is held down for the same reason: every stacked board
widens the projection that reads them.
"""

from __future__ import annotations

from typing import Any

from anthro_chess.models import MoveModelConfig


def tiny_model_config(**overrides: Any) -> MoveModelConfig:
    """Return the smallest usable model configuration, with any field replaced."""

    return MoveModelConfig(
        **{
            "piece_embedding_dim": 2,
            "model_dim": 4,
            "attention_heads": 1,
            "layers": 1,
            "feedforward_dim": 8,
            "history_positions": 2,
            "geometric_token_dim": 1,
            "geometric_bias_dim": 2,
            "dropout": 0.0,
            **overrides,
        }
    )
