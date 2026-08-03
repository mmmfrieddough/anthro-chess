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
#: What it is worth depends entirely on whether the step is launch-bound. At
#: the batch this project trains today it costs throughput and returns
#: activation memory; at a batch that fills the device it is the largest single
#: win measured on this backend. It stays off by default because the default
#: batch is the first of those.  ``docs/planning/cuda-training-proof.md`` holds
#: the readings on both.
TrainingPrecision = Literal["float32", "bfloat16-mixed"]

#: Whether float32 matrix multiplication may use the tensor cores' reduced
#: internal precision. ``highest`` keeps the full float32 path; ``high`` is
#: TF32, which rounds the inputs of a matmul while accumulating in float32.
#:
#: Same shape of tradeoff as the precision dial and the same reason to measure
#: rather than assume: it returns nothing on a launch-bound step, because
#: tensor cores accelerate arithmetic and a launch-bound step is not doing any.
#: Off by default; ``docs/planning/cuda-training-proof.md`` holds the readings.
#:
#: Declared rather than derived, so it is one of the settings a continuation has
#: to match: it decides the arithmetic every gradient is computed in, and a run
#: that changed it partway would have no way to say which half produced its
#: weights.
MatmulPrecision = Literal["highest", "high"]


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
    precision: TrainingPrecision = "float32"
    matmul_precision: MatmulPrecision = "highest"
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
