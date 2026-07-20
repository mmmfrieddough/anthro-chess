"""Training loops, losses, validation, and checkpoint orchestration."""

from anthro_chess.training.checkpoints import (
    CHECKPOINT_VERSION,
    CheckpointError,
    checkpoint_path,
    latest_checkpoint_path,
    load_training_checkpoint,
)
from anthro_chess.training.config import TrainingConfig
from anthro_chess.training.losses import masked_action_cross_entropy
from anthro_chess.training.runner import (
    RUN_ARTIFACT_VERSION,
    TrainingError,
    TrainingResult,
    run_training,
)

__all__ = [
    "CHECKPOINT_VERSION",
    "RUN_ARTIFACT_VERSION",
    "CheckpointError",
    "TrainingConfig",
    "TrainingError",
    "TrainingResult",
    "checkpoint_path",
    "latest_checkpoint_path",
    "load_training_checkpoint",
    "masked_action_cross_entropy",
    "run_training",
]
