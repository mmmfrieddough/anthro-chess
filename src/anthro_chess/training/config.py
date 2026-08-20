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
from anthro_chess.training.schedule import LearningRateSchedule, resolve_schedule

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
#: batch is the first of those.  ``docs/training-and-runtime.md`` holds the
#: readings on both.
TrainingPrecision = Literal["float32", "bfloat16-mixed"]

#: Whether float32 matrix multiplication may use the tensor cores' reduced
#: internal precision. ``highest`` keeps the full float32 path; ``high`` is
#: TF32, which rounds the inputs of a matmul while accumulating in float32.
#:
#: Same shape of tradeoff as the precision dial and the same reason to measure
#: rather than assume: it returns nothing on a launch-bound step, because
#: tensor cores accelerate arithmetic and a launch-bound step is not doing any.
#: Off by default; ``docs/training-and-runtime.md`` holds the readings,
#: and ``_EXECUTION_COMPATIBILITY_KEYS`` in the runner holds why a continuation
#: has to match it.
MatmulPrecision = Literal["highest", "high"]

#: Whether the forward pass is handed to `torch.compile`.
#:
#: Fusion acts on exactly what the precision dials cannot: a step that spends
#: itself issuing kernels is one where the launches themselves are the cost.
#: Unlike those dials it compounds with them rather than competing, because
#: reduced precision makes a step faster and so makes it more launch-bound.
#:
#: Only the plain mode is offered. ``reduce-overhead`` and ``max-autotune``
#: both capture CUDA graphs, both were measured, and both came in below this
#: one; ``docs/training-and-runtime.md`` holds those readings and the rest.
#:
#: Dynamo specializes on shape and the loader buckets games by length, so a run
#: presents several. ``training.compiled_graphs`` reports how many the compiler
#: built, which is what separates a run that settled from one recompiling
#: underneath a disappointing speedup.
TrainingCompilation = Literal["off", "default"]


class TrainingConfig(ConfigModel):
    """Configuration for a bounded action-model training run."""

    #: What the run root names the run's directory, rather than where it goes:
    #: the caller decides that, so a checked-in value cannot put a run on the
    #: wrong filesystem. The pattern holds the name to one path component,
    #: which is what keeps it from escaping the root it is joined to.
    run_name: str = Field(default="training", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    seed: int = Field(default=17, ge=0)
    steps: int = Field(default=10, ge=1)
    #: The rate the trunk holds, which warmup rises to and the cooldown decays
    #: from.
    learning_rate: float = Field(default=1e-3, gt=0.0)
    #: How much training data warmup spans, converted to steps by this run's own
    #: batch and accumulation. Declared as data rather than as steps or as a
    #: share of the run so that a branch and the trunk it resumes warm up over
    #: the same prefix; sequences rather than positions because games vary in
    #: length, so only the first is known before a run starts.
    warmup_sequences: int = Field(default=0, ge=0)
    #: The share of the horizon the cooldown occupies. A step count would
    #: survive a horizon change syntactically and reshape the curve silently.
    cooldown_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    #: Decoupled from the gradient, so what it sets is a timescale against the
    #: learning rate rather than a term the second moment rescales.
    weight_decay: float = Field(default=0.0, ge=0.0)
    #: Adam's second moment, which averages over a number of steps rather than
    #: over a quantity of data, so it moves when the batch does.
    second_moment_decay: float = Field(default=0.999, gt=0.0, lt=1.0)
    #: The global gradient norm a step is scaled back to. The default sits above
    #: the norms an ordinary step reaches, so it catches a spike rather than
    #: capping every step.
    gradient_clip_norm: float = Field(default=10.0, gt=0.0)
    log_every_steps: int = Field(default=1, ge=1)
    checkpoint_every_steps: int = Field(default=100, ge=1)
    resume_from: Literal["latest"] | Path | None = None
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    device: DeviceSelection = "auto"
    precision: TrainingPrecision = "float32"
    matmul_precision: MatmulPrecision = "highest"
    compilation: TrainingCompilation = "default"
    determinism: Literal["strict", "relaxed"] = "relaxed"
    profile_phases: StrictBool = False
    model: MoveModelConfig = MoveModelConfig()
    evaluation: TrainingEvaluationConfig = TrainingEvaluationConfig()
    efficiency: TrainingEfficiencyConfig = TrainingEfficiencyConfig()
    train: SequenceDataConfig
    validation: SequenceDataConfig | None = None

    def learning_rate_schedule(self) -> LearningRateSchedule:
        """Resolve the rate curve this run's steps are taken at."""

        return resolve_schedule(
            peak=self.learning_rate,
            steps=self.steps,
            warmup_sequences=self.warmup_sequences,
            cooldown_fraction=self.cooldown_fraction,
            sequences_per_step=(
                self.train.loader.batch_size * self.gradient_accumulation_steps
            ),
        )

    @model_validator(mode="after")
    def _resolve_schedule(self) -> TrainingConfig:
        """Refuse a schedule the declared horizon cannot carry.

        At validation rather than at the first optimizer step, so a horizon
        that cannot carry its schedule fails before the corpus loads rather
        than minutes into a run.
        """

        self.learning_rate_schedule()
        return self

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
