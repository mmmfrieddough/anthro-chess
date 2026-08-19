"""Tests for the position-label artifact a pool's readings share."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.data import encode_game
from anthro_chess.evaluation import PoolConfig, freeze_pool
from anthro_chess.evaluation import labels as labels_module
from anthro_chess.evaluation.labels import (
    POSITION_LABEL_FILE_NAME,
    PositionLabelStore,
    artifact_key,
    open_position_labels,
)
from anthro_chess.evaluation.pool import PoolProjection, load_pool
from anthro_chess.evaluation.scoring import SCORED_COLUMNS, encoding_input, row_game_id
from anthro_chess.evaluation.slices import board_from_encoding, position_labels


@pytest.fixture
def pool_path(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> Path:
    normalized, manifest = write_corpus(
        tmp_path,
        [
            normalized_row(1, split="test", plies=10, rating=1500),
            normalized_row(2, split="test", plies=8, rating=2100),
            normalized_row(3, split="test", plies=6, rating=None),
            normalized_row(4, split="train", plies=10, rating=1100),
            # A mate the side to move is threatened with and cannot prevent,
            # which realizes a predicate whose successful action set is empty.
            normalized_row(
                5,
                split="test",
                rating=1800,
                moves=("f8e8", "h5h8"),
                initial_position=(
                    "r2q1rk1/pbpn2p1/1p2ppP1/3p3Q/3P4/4P3/PPPN1PP1/R3K2R b KQ - 0 14"
                ),
            ),
        ],
    )
    output = tmp_path / "pool"
    freeze_pool(
        ResolvedConfig(
            value=PoolConfig.model_validate(
                {
                    "pool_id": "fixture-test",
                    "normalized": str(normalized),
                    "manifest": str(manifest),
                }
            ),
            provenance=ConfigProvenance(source=None, overrides=()),
        ),
        output,
    )
    return output


def _open(path: Path) -> tuple[PositionLabelStore, PoolProjection, object]:
    pool = load_pool(path)
    projection = PoolProjection(pool, SCORED_COLUMNS, error=ValueError)
    return open_position_labels(pool, projection), projection, pool


def test_the_stored_labels_are_what_deriving_them_gives(pool_path: Path) -> None:
    """The artifact holds the same answer the derivation would have given.

    It exists to stop every reading of a frozen pool resolving the predicates
    again, so the whole of its value rests on the two being the same labels.
    """

    store, projection, pool = _open(pool_path)

    game_ids = pool.game_ids  # type: ignore[attr-defined]
    stored = store.labels(game_ids)
    derived = {
        (row_game_id(row), ply.ply_index): position_labels(
            board_from_encoding(ply.board)
        )
        for row in projection.rows(game_ids)
        for ply in encode_game(encoding_input(row))
    }

    assert stored == derived
    assert store.position_count == len(derived)
    assert any(
        not match.successful_action_ids
        for labels in derived.values()
        for match in labels.predicates.values()
    ), "the fixture no longer covers a predicate with no successful action"


def test_a_second_reading_reads_the_artifact_rather_than_deriving_it(
    pool_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deriving happens on the first reading of a pool and on no later one."""

    _open(pool_path)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a stored artifact was derived again")

    monkeypatch.setattr(labels_module, "_build", refuse)
    store, _, _ = _open(pool_path)

    assert store.position_count > 0


def test_a_changed_slice_scheme_is_a_miss_rather_than_a_stale_hit(
    pool_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A label derived under one scheme is never read back under another.

    Nothing invalidates the artifact by hand, so the key covers every scheme
    the labels are derived under and a change to one is a miss.
    """

    store, _, pool = _open(pool_path)
    before = artifact_key(pool)  # type: ignore[arg-type]
    written = (pool_path / POSITION_LABEL_FILE_NAME).stat().st_mtime_ns

    monkeypatch.setattr(labels_module, "SLICE_SCHEME_VERSION", 99)
    rebuilt, _, pool = _open(pool_path)

    assert artifact_key(pool) != before  # type: ignore[arg-type]
    assert (pool_path / POSITION_LABEL_FILE_NAME).stat().st_mtime_ns != written
    assert rebuilt.position_count == store.position_count
