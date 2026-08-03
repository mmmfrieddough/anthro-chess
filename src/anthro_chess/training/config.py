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
from anthro_chess.training.efficiency import TrainingEfficiencyConfig

#: Parameters always stay float32. ``bfloat16-mixed`` autocasts the forward
#: pass, so activations are held at half the width while the optimizer keeps
#: full-precision master weights. bfloat16 rather than float16 because its
#: exponent range matches float32, so no gradient scaler is involved and a
#: checkpoint's scaler slot stays empty on every supported path.
#:
#: Its measured benefit is memory rather than speed, which is why it is off by
#: default: activation width is what decides whether a larger model fits, and
#: the throughput it costs is only worth paying when that is the constraint.
#: ``docs/planning/cuda-training-proof.md`` holds the readings.
TrainingPrecision = Literal["float32", "bfloat16-mixed"]


class TrainingConfig(ConfigModel):
    """Configuration for a bounded action-model training run."""

    #: What the run root names the run's directory, rather than where it goes:
    #: the caller decides that, so a checked-in value cannot put a run on the
    #: wrong filesystem. The pattern holds the name to one path component,
    #: which is what keeps it from escaping the root it is joined to.
    run_name: str = Field(default="training", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    seed: int = Field(default=17, ge=0)
    steps: int = Field(default=10, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    log_every_steps: int = Field(default=1, ge=1)
    checkpoint_every_steps: int = Field(default=100, ge=1)
    resume_from: Literal["latest"] | Path | None = None
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    device: DeviceSelection = "auto"
    precision: TrainingPrecision = "float32"
    determinism: Literal["strict", "relaxed"] = "relaxed"
    profile_phases: StrictBool = False
    model: MoveModelConfig = MoveModelConfig()
    evaluation: TrainingEvaluationConfig = TrainingEvaluationConfig()
    efficiency: TrainingEfficiencyConfig = TrainingEfficiencyConfig()
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
