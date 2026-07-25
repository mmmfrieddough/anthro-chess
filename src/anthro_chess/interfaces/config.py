"""Strict configuration for the directly invoked UCI process."""

from __future__ import annotations

import math

from pydantic import Field, model_validator

from anthro_chess.config import ConfigModel
from anthro_chess.inference import ModelRunnerConfig
from anthro_chess.runtime import RuntimeConfig

UCI_MIN_RATING = 400
UCI_MAX_RATING = 2500
UCI_TEMPERATURE_SCALE = 100
UCI_MAX_TEMPERATURE = 300
# The seed spin exposes ``-1`` as the fresh-per-game sentinel and otherwise an
# explicit non-negative seed bounded to a value common GUIs store safely.
UCI_RANDOM_SEED = -1
UCI_MAX_SEED = 2**31 - 1


class UciConfig(ConfigModel):
    """Model selection and runtime defaults for one UCI process."""

    model: ModelRunnerConfig = Field(default_factory=ModelRunnerConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def validate_portable_runtime(self) -> UciConfig:
        """Keep advertised portable-UCI controls representable and safe."""

        rating = self.runtime.target_rating
        if rating is None or not UCI_MIN_RATING <= rating <= UCI_MAX_RATING:
            raise ValueError(
                "UCI target_rating must be between "
                f"{UCI_MIN_RATING} and {UCI_MAX_RATING}"
            )
        scaled_temperature = self.runtime.temperature * UCI_TEMPERATURE_SCALE
        if not math.isclose(
            scaled_temperature,
            round(scaled_temperature),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("UCI temperature must use increments of 0.01")
        if self.runtime.resignation_enabled:
            raise ValueError("portable UCI mode does not support resignation")
        seed = self.runtime.seed
        if seed is not None and not 0 <= seed <= UCI_MAX_SEED:
            raise ValueError(f"UCI seed must be between 0 and {UCI_MAX_SEED}")
        return self
