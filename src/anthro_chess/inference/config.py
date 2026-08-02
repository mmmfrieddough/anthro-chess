"""Strict configuration for checkpoint-backed model inference."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from anthro_chess.config import ConfigModel

InferenceDevice = Literal["auto", "cpu", "mps", "cuda"]
LATEST_CHECKPOINT = "latest"
_CHECKPOINT_NAME = re.compile(r"^step-\d{8}\.pt$")


class ModelRunnerConfig(ConfigModel):
    """Selection and device settings for one checkpoint-backed model runner."""

    checkpoint_path: Path | None = None
    run_path: Path | None = None
    checkpoint: str = LATEST_CHECKPOINT
    device: InferenceDevice = "auto"

    @model_validator(mode="after")
    def validate_selection(self) -> ModelRunnerConfig:
        """Keep explicit checkpoint and retained-run selections unambiguous."""

        if self.checkpoint_path is not None and (
            self.run_path is not None or self.checkpoint != LATEST_CHECKPOINT
        ):
            raise ValueError(
                "checkpoint_path is authoritative and cannot be combined with "
                "run_path or checkpoint"
            )
        if (
            self.checkpoint != LATEST_CHECKPOINT
            and _CHECKPOINT_NAME.fullmatch(self.checkpoint) is None
        ):
            raise ValueError(
                "checkpoint must be 'latest' or a step-########.pt file name"
            )
        return self
