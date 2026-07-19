"""Training loops, losses, validation, and checkpoint orchestration."""

from anthro_chess.training.config import TrainingConfig
from anthro_chess.training.losses import masked_action_cross_entropy
from anthro_chess.training.runner import (
    RUN_ARTIFACT_VERSION,
    TrainingError,
    TrainingResult,
    run_training,
)

__all__ = [
    "RUN_ARTIFACT_VERSION",
    "TrainingConfig",
    "TrainingError",
    "TrainingResult",
    "masked_action_cross_entropy",
    "run_training",
]
