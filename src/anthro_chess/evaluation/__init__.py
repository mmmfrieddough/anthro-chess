"""Model metrics, benchmarks, and rollout evaluation.

Building and reading evaluation inputs needs only exact chess logic and the
data extra, so those names import eagerly. The move-model metrics need torch,
and are resolved lazily so a pool can be frozen or inspected without it.
"""

from typing import TYPE_CHECKING, Any

from anthro_chess.evaluation.pool import (
    BENCHMARK_VERSION,
    EvaluationPoolError,
    FrozenPool,
    PoolConfig,
    PoolGame,
    PoolResult,
    freeze_pool,
    load_pool,
    pool_game,
)
from anthro_chess.evaluation.slices import (
    DEFAULT_RATING_BANDS,
    LEGAL_MOVE_COUNT_BUCKETS,
    SLICE_SCHEME_VERSION,
    GamePhase,
    PlayerColor,
    PositionCharacteristic,
    PositionSlices,
    RatingBand,
    board_characteristics,
    board_from_encoding,
    board_phase,
    board_piece_ids,
    game_phase,
    legal_move_count_bucket,
    ply_characteristics,
    position_slices,
    rating_band_name,
)
from anthro_chess.evaluation.views import (
    VIEW_SPEC_VERSION,
    ViewConfig,
    ViewSelection,
    apply_view,
    game_ids_sha256,
)

if TYPE_CHECKING:
    from anthro_chess.evaluation.checkpoint import (
        CHECKPOINT_EVALUATION_VERSION,
        CheckpointEvaluationConfig,
        CheckpointEvaluationError,
        CheckpointEvaluationResult,
        evaluate_checkpoint,
    )
    from anthro_chess.evaluation.leakage import (
        LeakageCheck,
        LeakageError,
        check_leakage,
    )
    from anthro_chess.evaluation.validation import (
        VALIDATION_METRICS_VERSION,
        MoveValidationAccumulator,
        MoveValidationMetrics,
        RatingSliceMetrics,
        evaluate_move_model,
    )

_LAZY_EXPORTS = {
    "VALIDATION_METRICS_VERSION": "validation",
    "MoveValidationAccumulator": "validation",
    "MoveValidationMetrics": "validation",
    "RatingSliceMetrics": "validation",
    "evaluate_move_model": "validation",
    "CHECKPOINT_EVALUATION_VERSION": "checkpoint",
    "CheckpointEvaluationConfig": "checkpoint",
    "CheckpointEvaluationError": "checkpoint",
    "CheckpointEvaluationResult": "checkpoint",
    "evaluate_checkpoint": "checkpoint",
    "LeakageCheck": "leakage",
    "LeakageError": "leakage",
    "check_leakage": "leakage",
}


def __getattr__(name: str) -> Any:
    """Resolve torch-backed metric names only when they are actually used."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        from importlib import import_module

        module = import_module(f"anthro_chess.evaluation.{module_name}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BENCHMARK_VERSION",
    "CHECKPOINT_EVALUATION_VERSION",
    "CheckpointEvaluationConfig",
    "CheckpointEvaluationError",
    "CheckpointEvaluationResult",
    "DEFAULT_RATING_BANDS",
    "LeakageCheck",
    "LeakageError",
    "check_leakage",
    "evaluate_checkpoint",
    "LEGAL_MOVE_COUNT_BUCKETS",
    "SLICE_SCHEME_VERSION",
    "VALIDATION_METRICS_VERSION",
    "VIEW_SPEC_VERSION",
    "EvaluationPoolError",
    "FrozenPool",
    "GamePhase",
    "MoveValidationAccumulator",
    "MoveValidationMetrics",
    "PlayerColor",
    "PoolConfig",
    "PoolGame",
    "PoolResult",
    "PositionCharacteristic",
    "PositionSlices",
    "RatingBand",
    "RatingSliceMetrics",
    "ViewConfig",
    "ViewSelection",
    "apply_view",
    "board_characteristics",
    "board_from_encoding",
    "board_phase",
    "board_piece_ids",
    "evaluate_move_model",
    "freeze_pool",
    "game_ids_sha256",
    "game_phase",
    "legal_move_count_bucket",
    "ply_characteristics",
    "load_pool",
    "pool_game",
    "position_slices",
    "rating_band_name",
]
