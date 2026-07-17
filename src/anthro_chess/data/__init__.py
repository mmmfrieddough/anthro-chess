"""Data ingestion, normalization, and training-example construction."""

from anthro_chess.data.config import (
    FilterConfig,
    PrepareConfig,
    SourceConfig,
    SplitConfig,
)
from anthro_chess.data.encoding import (
    BOARD_SQUARE_COUNT,
    ENCODING_NAME,
    ENCODING_SCHEMA_SHA256,
    ENCODING_VERSION,
    BoardEncoding,
    EncodingError,
    GameEncodingInput,
    PlyEncoding,
    encode_game,
    encoding_identity,
)
from anthro_chess.data.prepare import (
    DataPreparationError,
    PreparationResult,
    prepare_pgn,
)

__all__ = [
    "BOARD_SQUARE_COUNT",
    "ENCODING_NAME",
    "ENCODING_SCHEMA_SHA256",
    "ENCODING_VERSION",
    "BoardEncoding",
    "DataPreparationError",
    "EncodingError",
    "FilterConfig",
    "GameEncodingInput",
    "PlyEncoding",
    "PreparationResult",
    "PrepareConfig",
    "SourceConfig",
    "SplitConfig",
    "encode_game",
    "encoding_identity",
    "prepare_pgn",
]
