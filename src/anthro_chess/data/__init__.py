"""Data ingestion, normalization, and training-example construction."""

from anthro_chess.data.config import (
    FilterConfig,
    PrepareConfig,
    SourceConfig,
    SplitConfig,
)
from anthro_chess.data.prepare import (
    DataPreparationError,
    PreparationResult,
    prepare_pgn,
)

__all__ = [
    "DataPreparationError",
    "FilterConfig",
    "PreparationResult",
    "PrepareConfig",
    "SourceConfig",
    "SplitConfig",
    "prepare_pgn",
]
