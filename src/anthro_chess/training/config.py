"""Code-owned configuration for the first executable training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, StrictBool

from anthro_chess.config import ConfigModel
from anthro_chess.data import SequenceDataConfig
from anthro_chess.models import MoveModelConfig
from anthro_chess.training.devices import DeviceSelection


class TrainingConfig(ConfigModel):
    """Configuration for a bounded action-model training run."""

    output_directory: Path = Path("artifacts/training")
    seed: int = Field(default=17, ge=0)
    steps: int = Field(default=10, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    log_every_steps: int = Field(default=1, ge=1)
    checkpoint_every_steps: int = Field(default=100, ge=1)
    resume_from: Literal["latest"] | Path | None = None
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    device: DeviceSelection = "auto"
    precision: Literal["float32"] = "float32"
    determinism: Literal["strict", "relaxed"] = "relaxed"
    profile_phases: StrictBool = False
    model: MoveModelConfig = MoveModelConfig()
    train: SequenceDataConfig
    validation: SequenceDataConfig | None = None
