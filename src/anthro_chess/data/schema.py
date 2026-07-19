"""Canonical schema contract for normalized game artifacts."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from pyarrow import Schema  # type: ignore[import-untyped]

SCHEMA_VERSION = 1
PREPROCESSING_VERSION = 2

FieldStatus = Literal["present", "unavailable", "rejected"]


class NormalizedColumn(StrEnum):
    """Stable column names in normalized game artifacts."""

    SCHEMA_VERSION = "schema_version"
    GAME_ID = "game_id"
    SOURCE_ID = "source_id"
    SOURCE_GAME_KEY = "source_game_key"
    RULESET = "ruleset"
    INITIAL_POSITION = "initial_position"
    RESULT = "result"
    TERMINATION = "termination"
    TERMINATION_STATUS = "termination_status"
    PLY_COUNT = "ply_count"
    ACTION_IDS = "action_ids"
    WHITE_SOURCE_RATING = "white_source_rating"
    WHITE_SOURCE_RATING_STATUS = "white_source_rating_status"
    BLACK_SOURCE_RATING = "black_source_rating"
    BLACK_SOURCE_RATING_STATUS = "black_source_rating_status"
    SOURCE_RATING_NAMESPACE = "source_rating_namespace"
    SOURCE_RATING_SYSTEM = "source_rating_system"
    WHITE_NORMALIZED_RATING = "white_normalized_rating"
    WHITE_NORMALIZED_RATING_STATUS = "white_normalized_rating_status"
    BLACK_NORMALIZED_RATING = "black_normalized_rating"
    BLACK_NORMALIZED_RATING_STATUS = "black_normalized_rating_status"
    TIME_INITIAL_MS = "time_initial_ms"
    TIME_INITIAL_STATUS = "time_initial_status"
    TIME_INCREMENT_MS = "time_increment_ms"
    TIME_INCREMENT_STATUS = "time_increment_status"
    CLOCK_REMAINING_MS = "clock_remaining_ms"
    CLOCK_STATUS = "clock_status"
    CLOCK_PRECISION_MS = "clock_precision_ms"
    SPLIT = "split"


NORMALIZED_COLUMNS = tuple(column.value for column in NormalizedColumn)


def normalized_parquet_schema() -> Schema:
    """Return the canonical Arrow schema for normalized game artifacts."""

    import pyarrow as pa

    column = NormalizedColumn
    return cast(
        "Schema",
        pa.schema(
            [
                pa.field(column.SCHEMA_VERSION, pa.int16(), nullable=False),
                pa.field(column.GAME_ID, pa.uint64(), nullable=False),
                pa.field(column.SOURCE_ID, pa.string(), nullable=False),
                pa.field(column.SOURCE_GAME_KEY, pa.string(), nullable=False),
                pa.field(column.RULESET, pa.string(), nullable=False),
                pa.field(column.INITIAL_POSITION, pa.string(), nullable=False),
                pa.field(column.RESULT, pa.string(), nullable=False),
                pa.field(column.TERMINATION, pa.string()),
                pa.field(column.TERMINATION_STATUS, pa.string(), nullable=False),
                pa.field(column.PLY_COUNT, pa.int32(), nullable=False),
                pa.field(column.ACTION_IDS, pa.list_(pa.uint16()), nullable=False),
                pa.field(column.WHITE_SOURCE_RATING, pa.int32()),
                pa.field(
                    column.WHITE_SOURCE_RATING_STATUS, pa.string(), nullable=False
                ),
                pa.field(column.BLACK_SOURCE_RATING, pa.int32()),
                pa.field(
                    column.BLACK_SOURCE_RATING_STATUS, pa.string(), nullable=False
                ),
                pa.field(column.SOURCE_RATING_NAMESPACE, pa.string()),
                pa.field(column.SOURCE_RATING_SYSTEM, pa.string()),
                pa.field(column.WHITE_NORMALIZED_RATING, pa.int32()),
                pa.field(
                    column.WHITE_NORMALIZED_RATING_STATUS,
                    pa.string(),
                    nullable=False,
                ),
                pa.field(column.BLACK_NORMALIZED_RATING, pa.int32()),
                pa.field(
                    column.BLACK_NORMALIZED_RATING_STATUS,
                    pa.string(),
                    nullable=False,
                ),
                pa.field(column.TIME_INITIAL_MS, pa.int32()),
                pa.field(column.TIME_INITIAL_STATUS, pa.string(), nullable=False),
                pa.field(column.TIME_INCREMENT_MS, pa.int32()),
                pa.field(column.TIME_INCREMENT_STATUS, pa.string(), nullable=False),
                pa.field(
                    column.CLOCK_REMAINING_MS,
                    pa.list_(pa.int32()),
                    nullable=False,
                ),
                pa.field(
                    column.CLOCK_STATUS,
                    pa.list_(pa.string()),
                    nullable=False,
                ),
                pa.field(
                    column.CLOCK_PRECISION_MS,
                    pa.list_(pa.int32()),
                    nullable=False,
                ),
                pa.field(column.SPLIT, pa.string(), nullable=False),
            ],
            metadata={
                b"anthro_schema_version": str(SCHEMA_VERSION).encode(),
                b"anthro_preprocessing_version": str(PREPROCESSING_VERSION).encode(),
            },
        ),
    )
