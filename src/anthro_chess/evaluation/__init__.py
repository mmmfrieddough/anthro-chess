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
    from anthro_chess.evaluation.validation import (
        VALIDATION_METRICS_VERSION,
        MoveValidationAccumulator,
        MoveValidationMetrics,
        RatingSliceMetrics,
        evaluate_move_model,
    )

_VALIDATION_EXPORTS = frozenset(
    {
        "VALIDATION_METRICS_VERSION",
        "MoveValidationAccumulator",
        "MoveValidationMetrics",
        "RatingSliceMetrics",
        "evaluate_move_model",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve torch-backed metric names only when they are actually used."""

    if name in _VALIDATION_EXPORTS:
        from anthro_chess.evaluation import validation

        return getattr(validation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BENCHMARK_VERSION",
    "DEFAULT_RATING_BANDS",
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
    "position_slices",
    "rating_band_name",
]
