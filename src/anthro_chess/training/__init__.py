"""Training losses, checkpoint persistence, and lazy runner orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anthro_chess.training.checkpoints import (
    CHECKPOINT_VERSION,
    CheckpointError,
    checkpoint_path,
    latest_checkpoint_path,
    load_training_checkpoint,
)
from anthro_chess.training.config import TrainingConfig
from anthro_chess.training.losses import masked_action_cross_entropy

if TYPE_CHECKING:
    from anthro_chess.training.runner import TrainingError, TrainingResult

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


def __getattr__(name: str) -> Any:
    """Load training-loop orchestration only when its public API is requested."""

    if name in {
        "RUN_ARTIFACT_VERSION",
        "TrainingError",
        "TrainingResult",
        "run_training",
    }:
        from anthro_chess.training import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
