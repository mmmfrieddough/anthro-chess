"""Canonical schema contract for normalized game artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from pyarrow import Schema  # type: ignore[import-untyped]

SCHEMA_VERSION = 3
PREPROCESSING_VERSION = 7

FieldStatus = Literal["present", "unavailable", "rejected"]

SplitName = Literal["train", "validation", "test"]

#: Canonical split order. ``test`` is held back from training entirely so
#: benchmark comparisons are not reported on data the training loop selects
#: against; ``validation`` remains the in-training split.
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "test")


class NormalizedColumn(StrEnum):
    """Stable column names in normalized game artifacts.

    ``action_ids`` carries the game's moves and, when the ending was a player's
    decision made on their own turn, one trailing terminal action.
    ``ply_count`` counts moves only, so a terminal action never changes a
    game's length, its ply filters, or its prefix depth. The per-ply clock
    columns stay aligned one-to-one with ``action_ids``, so a trailing terminal
    action carries an unavailable clock observation rather than a synthesized
    one. ``terminal_action_status`` records whether that action was appended
    and, when it was not, why.

    ``clock_remaining_delta_ms`` stores what
    :func:`encode_clock_remaining_deltas` produces rather than the clock itself;
    read a row through :func:`clock_remaining_ms`, or a bare column through
    :func:`decode_clock_remaining_deltas`. The name carries the
    difference because the two are indistinguishable by inspection, and reading
    one as the other yields plausible wrong move times rather than an error.

    ``white_player_digest`` and ``black_player_digest`` come from
    ``anthro_chess.data.accounts.account_row_digest``, which truncates the same
    salted hash a marked-account snapshot stores, so a corpus row and a snapshot
    are comparable without the corpus carrying readable usernames.
    """

    SCHEMA_VERSION = "schema_version"
    GAME_ID = "game_id"
    SOURCE_ID = "source_id"
    SOURCE_GAME_KEY = "source_game_key"
    WHITE_PLAYER_DIGEST = "white_player_digest"
    BLACK_PLAYER_DIGEST = "black_player_digest"
    RULESET = "ruleset"
    INITIAL_POSITION = "initial_position"
    RESULT = "result"
    TERMINATION = "termination"
    TERMINATION_STATUS = "termination_status"
    TERMINATION_CATEGORY = "termination_category"
    TERMINATION_BY_SIDE_TO_MOVE = "termination_by_side_to_move"
    TERMINAL_ACTION_STATUS = "terminal_action_status"
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
    CLOCK_REMAINING_DELTA_MS = "clock_remaining_delta_ms"
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
                pa.field(column.WHITE_PLAYER_DIGEST, pa.uint64()),
                pa.field(column.BLACK_PLAYER_DIGEST, pa.uint64()),
                pa.field(column.RULESET, pa.string(), nullable=False),
                pa.field(column.INITIAL_POSITION, pa.string(), nullable=False),
                pa.field(column.RESULT, pa.string(), nullable=False),
                pa.field(column.TERMINATION, pa.string()),
                pa.field(column.TERMINATION_STATUS, pa.string(), nullable=False),
                pa.field(column.TERMINATION_CATEGORY, pa.string(), nullable=False),
                pa.field(column.TERMINATION_BY_SIDE_TO_MOVE, pa.bool_()),
                pa.field(column.TERMINAL_ACTION_STATUS, pa.string(), nullable=False),
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
                    column.CLOCK_REMAINING_DELTA_MS,
                    pa.list_(pa.int32()),
                    nullable=False,
                ),
                pa.field(
                    column.CLOCK_STATUS,
                    pa.list_(pa.string()),
                    nullable=False,
                ),
                pa.field(column.CLOCK_PRECISION_MS, pa.int32()),
                pa.field(column.SPLIT, pa.string(), nullable=False),
            ],
            metadata={
                b"anthro_schema_version": str(SCHEMA_VERSION).encode(),
                b"anthro_preprocessing_version": str(PREPROCESSING_VERSION).encode(),
            },
        ),
    )


#: A clock trace lists both players alternately, so the previous reading by the
#: player to move is two entries back and the difference between them is a move
#: time. Differencing adjacent entries instead subtracts one player's clock from
#: the other's, which is not a quantity.
_CLOCK_STRIDE = 2


def encode_clock_remaining_deltas(
    remaining_ms: Sequence[int | None],
) -> list[int | None]:
    """Return the stored form of one game's clock trace.

    Each entry becomes the drop since the same player's previous reading. The
    first entry for each player has no predecessor and stays absolute, as does
    any entry whose predecessor is missing, so a trace with holes survives the
    round trip instead of losing every reading after one.

    An entry depends only on entries before it, so a prefix of the stored form
    decodes to the same prefix of the trace — which is what lets a benchmark
    shorten a game by slicing the column.
    """

    stored: list[int | None] = []
    for index, value in enumerate(remaining_ms):
        previous = (
            remaining_ms[index - _CLOCK_STRIDE] if index >= _CLOCK_STRIDE else None
        )
        if value is None or previous is None:
            stored.append(value)
        else:
            stored.append(previous - value)
    return stored


def clock_remaining_ms(row: Mapping[str, Any]) -> tuple[int | None, ...]:
    """Return one normalized row's clock trace as remaining time."""

    return tuple(
        decode_clock_remaining_deltas(row[NormalizedColumn.CLOCK_REMAINING_DELTA_MS])
    )


def decode_clock_remaining_deltas(
    stored_ms: Sequence[int | None],
) -> list[int | None]:
    """Return the clock trace :func:`encode_clock_remaining_deltas` was given.

    The absolute-or-delta choice is not recorded per entry because it follows
    from the decoded predecessor, which is already known by the time an entry is
    read.
    """

    remaining: list[int | None] = []
    for index, value in enumerate(stored_ms):
        previous = remaining[index - _CLOCK_STRIDE] if index >= _CLOCK_STRIDE else None
        if value is None or previous is None:
            remaining.append(value)
        else:
            remaining.append(previous - value)
    return remaining
