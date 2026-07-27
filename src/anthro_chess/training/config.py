"""Code-owned configuration for the first executable training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, StrictBool, model_validator

from anthro_chess.config import ConfigModel
from anthro_chess.data import SequenceDataConfig
from anthro_chess.models import MoveModelConfig
from anthro_chess.training.cadence import TrainingEvaluationConfig
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
    evaluation: TrainingEvaluationConfig = TrainingEvaluationConfig()
    train: SequenceDataConfig
    validation: SequenceDataConfig | None = None

    @model_validator(mode="after")
    def _reject_test_split(self) -> TrainingConfig:
        """Keep the held-out benchmark partition out of every training run.

        The test split exists so checkpoint comparisons are not reported on
        data the training loop selected against. Consuming it here, even for
        validation metrics, would quietly destroy that guarantee.
        """

        selections = (("train", self.train), ("validation", self.validation))
        for name, selection in selections:
            if selection is not None and selection.loader.split == "test":
                raise ValueError(
                    f"{name} selection must not use the held-out test split"
                )
        return self
