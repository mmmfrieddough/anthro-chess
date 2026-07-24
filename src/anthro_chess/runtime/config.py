"""Strict settings for untimed action selection."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StrictBool, StrictInt

from anthro_chess.config import ConfigModel

TargetRating = Annotated[StrictInt, Field(ge=0)]
Temperature = Annotated[float, Field(ge=0.0, le=3.0, allow_inf_nan=False)]
RandomSeed = Annotated[StrictInt, Field(ge=0)]


class RuntimeConfig(ConfigModel):
    """Code-owned controls for one decision runtime."""

    target_rating: TargetRating | None = 1500
    temperature: Temperature = 1.0
    # ``None`` draws a fresh per-game stream from operating-system entropy so
    # ordinary interactive play varies; an explicit non-negative seed makes a
    # game reproducible for debugging and benchmarks.
    seed: RandomSeed | None = None
    resignation_enabled: StrictBool = False
