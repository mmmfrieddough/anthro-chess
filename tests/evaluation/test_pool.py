"""Tests for freezing and loading the evaluation pool."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.evaluation import (
    BENCHMARK_VERSION,
    EvaluationPoolError,
    PoolConfig,
    freeze_pool,
    load_pool,
)


def _resolved(
    normalized: Path,
    manifest: Path,
    **overrides: object,
) -> ResolvedConfig[PoolConfig]:
    return ResolvedConfig(
        value=PoolConfig.model_validate(
            {
                "pool_id": "fixture-test",
                "normalized": str(normalized),
                "manifest": str(manifest),
                **overrides,
            }
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )


@pytest.fixture
def corpus(
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[[Path, list[dict[str, Any]]], tuple[Path, Path]],
) -> Callable[[Path], tuple[Path, Path]]:
    """Return a factory writing a mixed-split corpus beneath a directory."""

    def build(tmp_path: Path) -> tuple[Path, Path]:
        return write_corpus(
            tmp_path / "corpus",
            [
                normalized_row(1, split="train"),
                normalized_row(2, split="train"),
                normalized_row(3, split="validation"),
                normalized_row(4, split="test", plies=4, result="0-1"),
                normalized_row(5, split="test", plies=8, rating=900, clocks=False),
            ],
        )

    return build


def test_freeze_selects_only_the_test_split_and_records_provenance(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)

    result = freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    record = json.loads(result.manifest_path.read_text())
    assert result.games == 2
    assert result.plies == 12
    assert record["benchmark_version"] == BENCHMARK_VERSION
    assert record["pool"] == {"id": "fixture-test", "version": 1, "split": "test"}
    assert record["source"]["manifest_sha256"]
    assert record["output"]["sha256"]
    assert set(record) >= {
        "action_vocabulary",
        "coverage",
        "encoding",
        "identity",
        "leakage",
        "preprocessing_version",
        "resolved_config",
        "schema_version",
    }


def test_manifest_records_ids_and_content_hashes_for_later_leakage_checks(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """#29 compares these against an evaluated checkpoint's training identity."""

    normalized, manifest = corpus(tmp_path)

    result = freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    record = json.loads(result.manifest_path.read_text())
    games = record["identity"]["games"]
    assert [entry["game_id"] for entry in games] == [4, 5]
    assert all(len(entry["content_sha256"]) == 64 for entry in games)


def test_build_time_overlap_check_compares_against_the_train_split(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)

    result = freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    leakage = json.loads(result.manifest_path.read_text())["leakage"]
    assert leakage["compared_split"] == "train"
    assert leakage["compared_games"] == 2
    assert leakage["overlapping_games"] == 0


def test_a_game_in_both_train_and_test_fails_the_build(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[[Path, list[dict[str, Any]]], tuple[Path, Path]],
) -> None:
    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [normalized_row(7, split="train"), normalized_row(7, split="test")],
    )

    with pytest.raises(EvaluationPoolError, match="also appear in the train split"):
        freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")


def test_coverage_makes_thin_slices_visible(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)

    result = freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    coverage = json.loads(result.manifest_path.read_text())["coverage"]
    assert coverage["games"] == 2
    assert coverage["plies"] == {
        "total": 12,
        "minimum_per_game": 4,
        "maximum_per_game": 8,
    }
    assert coverage["results"] == {"0-1": 1, "1-0": 1}
    assert coverage["clock_presence_games"] == {"absent": 1, "present": 1}
    assert coverage["color_positions"] == {"black": 6, "white": 6}
    assert sum(coverage["phase_positions"].values()) == 12
    assert sum(coverage["legal_move_count_positions"].values()) == 12
    assert coverage["rating_band_positions"] == {"1200_to_1599": 4, "under_1200": 8}
    assert coverage["positions_without_rating"] == 0


def test_expected_identity_rejects_a_pool_that_changed(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """A checked-in digest turns silent drift into a build failure."""

    normalized, manifest = corpus(tmp_path)
    baseline = freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    matching = _resolved(
        normalized,
        manifest,
        expected_game_ids_sha256=baseline.game_ids_sha256,
    )
    assert freeze_pool(matching, tmp_path / "pool").games == 2

    with pytest.raises(EvaluationPoolError, match="expected identity"):
        freeze_pool(
            _resolved(normalized, manifest, expected_game_ids_sha256="0" * 64),
            tmp_path / "pool",
        )


def test_freezing_is_reproducible(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)

    first = freeze_pool(_resolved(normalized, manifest), tmp_path / "first")
    second = freeze_pool(_resolved(normalized, manifest), tmp_path / "second")

    assert first.game_ids_sha256 == second.game_ids_sha256
    assert first.games_path.read_bytes() == second.games_path.read_bytes()


def test_load_pool_round_trips_and_exposes_game_level_facts(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)
    freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    pool = load_pool(tmp_path / "pool")

    assert pool.game_ids == (4, 5)
    by_id = {game.game_id: game for game in pool.games}
    assert by_id[4].ply_count == 4
    assert by_id[4].has_clocks is True
    assert by_id[5].has_clocks is False
    assert by_id[5].has_ratings is True


def test_load_pool_rejects_a_tampered_artifact(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)
    freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")
    (tmp_path / "pool/games.parquet").write_bytes(b"corrupted")

    with pytest.raises(EvaluationPoolError, match="checksum mismatch"):
        load_pool(tmp_path / "pool")


def test_load_pool_rejects_an_incompatible_benchmark_version(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)
    freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")
    manifest_path = tmp_path / "pool/manifest.json"
    record = json.loads(manifest_path.read_text())
    record["benchmark_version"] = BENCHMARK_VERSION + 1
    manifest_path.write_text(json.dumps(record))

    with pytest.raises(EvaluationPoolError, match="benchmark version"):
        load_pool(tmp_path / "pool")


def test_an_empty_selection_fails_rather_than_writing_an_empty_pool(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[[Path, list[dict[str, Any]]], tuple[Path, Path]],
) -> None:
    normalized, manifest = write_corpus(
        tmp_path / "corpus", [normalized_row(1, split="train")]
    )

    with pytest.raises(EvaluationPoolError, match="no normalized games"):
        freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")


def test_incompatible_source_preprocessing_is_rejected(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)
    record = json.loads(manifest.read_text())
    record["preprocessing_version"] = 1
    manifest.write_text(json.dumps(record))

    with pytest.raises(EvaluationPoolError, match="preprocessing version"):
        freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")
