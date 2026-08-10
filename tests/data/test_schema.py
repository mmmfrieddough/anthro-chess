"""The stored form of a normalized column and what reading it back must give."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from anthro_chess.data.accounts import account_digest, account_row_digest
from anthro_chess.data.artifacts import write_normalized_rows
from anthro_chess.data.schema import (
    NormalizedColumn,
    decode_clock_remaining_deltas,
    encode_clock_remaining_deltas,
    normalized_parquet_schema,
)


@pytest.mark.parametrize(
    "remaining",
    [
        pytest.param([], id="empty"),
        pytest.param([300_000], id="one-ply"),
        pytest.param([300_000, 300_000], id="one-move"),
        pytest.param([180_000, 180_000, 178_000, 177_500], id="descending"),
        # An increment can leave a player with more time than they had, so a
        # delta is signed and a decoder that assumed otherwise would drift.
        pytest.param([60_000, 60_000, 61_000, 62_000], id="rising-with-increment"),
        pytest.param([300_000, None, 298_000, None], id="interior-holes"),
        pytest.param([None, None, None], id="no-clock-data"),
        pytest.param([300_000, 299_000, 298_000, None], id="trailing-terminal-action"),
    ],
)
def test_a_clock_trace_survives_the_stored_form(remaining: list[int | None]) -> None:
    assert decode_clock_remaining_deltas(encode_clock_remaining_deltas(remaining)) == (
        remaining
    )


def test_arbitrary_traces_survive_the_stored_form() -> None:
    """Holes, lengths, and directions a hand-written case would not think of."""

    rng = random.Random(20260809)
    for _ in range(2_000):
        length = rng.randrange(0, 24)
        trace: list[int | None] = [
            None if rng.random() < 0.2 else rng.randrange(0, 10_800_000)
            for _ in range(length)
        ]
        assert decode_clock_remaining_deltas(encode_clock_remaining_deltas(trace)) == (
            trace
        )


def test_truncating_the_stored_form_truncates_the_clock_it_decodes_to() -> None:
    """Two benchmarks shorten a game by slicing the column without decoding it.

    ``evaluation.checkpoint`` and ``evaluation.novelty`` both cut a stored trace
    to the plies a derivation reached. That is sound only because an entry
    depends on entries before it and never after, which is a property of the
    codec rather than of how they call it.
    """

    remaining: list[int | None] = [180_000, 60_000, 178_000, 59_000, 175_000, 55_000]
    stored = encode_clock_remaining_deltas(remaining)

    for plies in range(len(remaining) + 1):
        assert decode_clock_remaining_deltas(stored[:plies]) == remaining[:plies]


def test_the_stored_form_differences_one_player_rather_than_two() -> None:
    """The saving comes from move times, which are two entries apart."""

    # White spends 2s then 3s; Black spends 1s then 4s.
    remaining = [180_000, 60_000, 178_000, 59_000, 175_000, 55_000]

    assert encode_clock_remaining_deltas(remaining) == [
        180_000,
        60_000,
        2_000,
        1_000,
        3_000,
        4_000,
    ]


def test_a_row_digest_matches_the_snapshot_digest_for_one_account() -> None:
    """The corpus column and a marked-account snapshot must be comparable."""

    assert account_row_digest("Cheater") == int(account_digest("cheater")[:16], 16)
    assert account_row_digest("cheater") != account_row_digest("someone-else")


def test_columns_unique_to_every_row_are_not_dictionary_encoded(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
) -> None:
    """A dictionary of unique values is as large as the values it indexes."""

    path = tmp_path / "games.parquet"
    write_normalized_rows([normalized_row(index) for index in range(64)], path)

    row_group = pq.ParquetFile(path).metadata.row_group(0)
    columns = [row_group.column(index) for index in range(row_group.num_columns)]
    undictionaried = {
        column.path_in_schema
        for column in columns
        if column.dictionary_page_offset is None
    }
    # Parquet bit-packs booleans and never offers them a dictionary, which is
    # its decision rather than an exemption this project chose.
    boolean_columns = {
        field.name
        for field in normalized_parquet_schema()
        if pa.types.is_boolean(field.type)
    }
    # Asserted as the whole partition rather than as the two exempt columns, so
    # a column added later cannot inherit an encoding nobody chose for it.
    assert undictionaried - boolean_columns == {
        NormalizedColumn.SOURCE_GAME_KEY.value,
    }


def test_stored_deltas_match_the_definition_over_every_short_hole_pattern() -> None:
    """The encoder reads a stride behind, so a hole moves what the next entry is.

    Enumerating the patterns is what says the loop agrees with the rule the
    docstring states, rather than with the traces the other tests happen to use.
    """

    for length in range(6):
        for holes in range(1 << length):
            trace: list[int | None] = [
                None if holes >> index & 1 else 100_000 - 500 * index
                for index in range(length)
            ]
            expected = [
                value
                if value is None or index < 2 or trace[index - 2] is None
                else cast(int, trace[index - 2]) - value
                for index, value in enumerate(trace)
            ]
            assert encode_clock_remaining_deltas(trace) == expected
