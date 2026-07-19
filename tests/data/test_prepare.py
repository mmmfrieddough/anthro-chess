import json
from hashlib import sha256
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import chess
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import zstandard

from anthro_chess.chess import decode_move
from anthro_chess.config import load_config
from anthro_chess.data import (
    DataPreparationError,
    PrepareConfig,
    acquire_archive,
    prepare_pgn,
)
from anthro_chess.data.schema import NORMALIZED_COLUMNS

REPOSITORY_ROOT = Path(__file__).parents[2]
SAMPLE_PGN = REPOSITORY_ROOT / "samples/lichess/standard-export-sample.pgn"
SAMPLE_CONFIG = REPOSITORY_ROOT / "configs/data/lichess-sample.toml"
BASELINE_CONFIG = REPOSITORY_ROOT / "configs/data/lichess-blitz-2017-04.toml"


def test_baseline_selection_pins_one_bounded_verified_rating_namespace() -> None:
    config = load_config(PrepareConfig, path=BASELINE_CONFIG).value

    assert config.source.rating_namespace == "lichess_blitz"
    assert config.filters.event_speed == "blitz"
    assert config.filters.maximum_games is not None
    assert config.output.games_per_shard is not None
    assert config.split.require_nonempty is True
    assert config.archive is not None
    assert config.archive.sha256 == (
        "559222b0e933bc02281643724fbd9bd46690074b10d784c4f264e0bff6c5992c"
    )


def test_prepares_checked_in_sample_with_shared_actions_and_provenance(
    tmp_path: Path,
) -> None:
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    result = prepare_pgn(SAMPLE_PGN, tmp_path / "artifacts", resolved)

    rows = pq.read_table(result.normalized_path).to_pylist()
    assert tuple(pq.read_schema(result.normalized_path).names) == NORMALIZED_COLUMNS
    assert result.accepted_games == 1
    assert result.rejected_games == 0
    assert len(rows) == 1
    row = rows[0]
    assert row["source_id"] == "lichess"
    assert row["source_game_key"] == "PpwPOZMq"
    assert row["result"] == "0-1"
    assert row["termination"] == "time_forfeit"
    assert row["ply_count"] == 26
    assert decode_move(row["action_ids"][0]) == chess.Move.from_uci("e2e4")
    assert row["white_source_rating"] == 2100
    assert row["white_normalized_rating"] == 2100
    assert row["clock_remaining_ms"][:3] == [30000, 30000, 29000]
    assert set(row["clock_status"]) == {"present"}
    assert set(row["clock_precision_ms"]) == {1000}

    manifest = _read_json(result.manifest_path)
    assert manifest["input"]["sha256"] == _sha256(SAMPLE_PGN)
    assert manifest["output"]["sha256"] == _sha256(result.normalized_path)
    assert manifest["source"]["license"] == "CC0-1.0"
    assert manifest["games"]["accepted"] == 1
    assert manifest["games"]["rejected"] == 0
    assert manifest["games"]["rejection_reasons"] == {}
    assert manifest["games"]["plies"]["total"] == 26
    assert manifest["output"]["shards"][0]["games"] == 1
    assert manifest["resolved_config"] == resolved.as_record()
    assert manifest["action_vocabulary"]["sha256"]
    assert manifest["split"]["counts"] == result.split_counts


def test_repeated_runs_produce_equivalent_records_and_splits(tmp_path: Path) -> None:
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    first = prepare_pgn(SAMPLE_PGN, tmp_path / "first", resolved)
    second = prepare_pgn(SAMPLE_PGN, tmp_path / "second", resolved)

    assert (
        pq.read_table(first.normalized_path).to_pylist()
        == pq.read_table(second.normalized_path).to_pylist()
    )
    assert first.split_counts == second.split_counts
    assert _sha256(first.normalized_path) == _sha256(second.normalized_path)


def test_preserves_present_unavailable_and_rejected_optional_values(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "missingness.pgn"
    input_path.write_text(
        """
[Event "Rated test game"]
[Site "https://example.test/zero-values"]
[Date "2026.07.16"]
[Round "-"]
[White "White"]
[Black "Black"]
[Result "1-0"]
[WhiteElo "0"]
[BlackElo "not-a-rating"]
[TimeControl "60+0"]

1. e4 { [%clk 0:00:00] } e5 { [%clk nope] }
2. Nf3 { [%clkc 1234] } Nc6 1-0
""".lstrip(),
        encoding="utf-8",
    )
    resolved = load_config(
        PrepareConfig,
        path=SAMPLE_CONFIG,
        overrides=(
            'source.id="test"',
            'source.version="fixture"',
            'source.url="https://example.test/"',
            'source.license="CC0-1.0"',
        ),
    )

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    row = pq.read_table(result.normalized_path).to_pylist()[0]
    assert row["white_source_rating"] == 0
    assert row["white_source_rating_status"] == "present"
    assert row["black_source_rating"] is None
    assert row["black_source_rating_status"] == "rejected"
    assert row["clock_remaining_ms"] == [0, None, 12340, None]
    assert row["clock_status"] == [
        "present",
        "rejected",
        "present",
        "unavailable",
    ]
    assert row["clock_precision_ms"] == [1000, None, 10, None]
    assert row["termination"] is None
    assert row["termination_status"] == "unavailable"


def test_filters_games_and_records_rejection_reasons(tmp_path: Path) -> None:
    input_path = tmp_path / "filtered.pgn"
    input_path.write_text(
        _short_game(site="accepted")
        + _short_game(site="bot", extra_headers='[WhiteTitle "BOT"]\n')
        + _short_game(site="unrated", event="Casual test game"),
        encoding="utf-8",
    )
    resolved = load_config(
        PrepareConfig,
        path=SAMPLE_CONFIG,
        overrides=(
            'source.id="test"',
            'source.version="fixture"',
            'source.url="https://example.test/"',
            'source.license="CC0-1.0"',
        ),
    )

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    assert result.accepted_games == 1
    assert result.rejected_games == 2
    manifest = _read_json(result.manifest_path)
    assert manifest["games"]["rejection_reasons"] == {
        "bot_game": 1,
        "unrated_game": 1,
    }


def test_rejects_a_run_when_no_games_pass_filters(tmp_path: Path) -> None:
    input_path = tmp_path / "unrated.pgn"
    input_path.write_text(
        _short_game(site="unrated", event="Casual test game"),
        encoding="utf-8",
    )
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    with pytest.raises(DataPreparationError, match="unrated_game=1"):
        prepare_pgn(input_path, tmp_path / "artifacts", resolved)


def test_acquires_and_reuses_only_a_checksum_verified_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"pinned archive bytes"
    expected_sha256 = sha256(payload).hexdigest()
    resolved = load_config(
        PrepareConfig,
        path=SAMPLE_CONFIG,
        overrides=(
            'archive.url="https://example.test/archive.pgn.zst"',
            'archive.file_name="archive.pgn.zst"',
            f'archive.sha256="{expected_sha256}"',
            'archive.compression="zstd"',
        ),
    )
    prepare_module = import_module("anthro_chess.data.prepare")
    calls = 0

    def fake_urlopen(_request: object, *, timeout: int) -> BytesIO:
        nonlocal calls
        assert timeout == 60
        calls += 1
        return BytesIO(payload)

    monkeypatch.setattr(prepare_module, "urlopen", fake_urlopen)

    acquired = acquire_archive(tmp_path, resolved)
    reused = acquire_archive(tmp_path, resolved)

    assert acquired.archive_path.read_bytes() == payload
    assert acquired.sha256 == expected_sha256
    assert acquired.reused is False
    assert reused.reused is True
    assert calls == 1


def test_rejects_download_with_wrong_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = load_config(
        PrepareConfig,
        path=SAMPLE_CONFIG,
        overrides=(
            'archive.url="https://example.test/archive.pgn.zst"',
            'archive.file_name="archive.pgn.zst"',
            f'archive.sha256="{"0" * 64}"',
            'archive.compression="zstd"',
        ),
    )
    prepare_module = import_module("anthro_chess.data.prepare")
    monkeypatch.setattr(
        prepare_module,
        "urlopen",
        lambda _request, timeout: BytesIO(b"wrong bytes"),
    )

    with pytest.raises(DataPreparationError, match="checksum mismatch"):
        acquire_archive(tmp_path, resolved)

    assert not (tmp_path / "raw/archive.pgn.zst").exists()
    assert not (tmp_path / "raw/archive.pgn.zst.part").exists()


def test_prepare_rejects_input_that_does_not_match_pinned_archive(
    tmp_path: Path,
) -> None:
    resolved = load_config(
        PrepareConfig,
        path=SAMPLE_CONFIG,
        overrides=(
            'archive.url="https://example.test/archive.pgn.zst"',
            'archive.file_name="archive.pgn.zst"',
            f'archive.sha256="{"0" * 64}"',
            'archive.compression="zstd"',
        ),
    )

    with pytest.raises(DataPreparationError, match="input archive checksum mismatch"):
        prepare_pgn(SAMPLE_PGN, tmp_path / "artifacts", resolved)


def test_streams_zstandard_input_into_bounded_shards_and_one_namespace(
    tmp_path: Path,
) -> None:
    games = (
        _short_game(site="rapid", event="Rated Rapid game")
        + _short_game(site="game-0")
        + _short_game(site="game-0")
        + "".join(_short_game(site=f"game-{index}") for index in range(1, 8))
    )
    input_path = tmp_path / "games.pgn.zst"
    input_path.write_bytes(zstandard.ZstdCompressor().compress(games.encode()))
    resolved = load_config(
        PrepareConfig,
        path=SAMPLE_CONFIG,
        overrides=(
            'source.id="test"',
            'source.version="fixture"',
            'source.url="https://example.test/"',
            'source.license="CC0-1.0"',
            'source.rating_namespace="test_blitz"',
            'filters.event_speed="blitz"',
            "filters.require_ratings=true",
            "filters.maximum_games=5",
            "output.games_per_shard=2",
            "split.validation_fraction=0.5",
            "split.require_nonempty=true",
        ),
    )
    stale_shard = tmp_path / "artifacts/normalized/games-99999.parquet"
    stale_shard.parent.mkdir(parents=True)
    stale_shard.write_bytes(b"stale")

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    assert result.accepted_games == 5
    assert result.rejected_games == 2
    assert len(result.normalized_paths) == 3
    assert [pq.read_table(path).num_rows for path in result.normalized_paths] == [
        2,
        2,
        1,
    ]
    assert all(result.split_counts.values())
    assert not stale_shard.exists()
    manifest = _read_json(result.manifest_path)
    assert manifest["selection"] == {
        "algorithm": "source-order-first-accepted-v1",
        "limit_reached": True,
        "maximum_games": 5,
    }
    assert manifest["games"]["scanned"] == 7
    assert manifest["games"]["rejection_reasons"] == {
        "duplicate_game": 1,
        "rating_namespace_mismatch": 1,
    }
    assert [shard["games"] for shard in manifest["output"]["shards"]] == [2, 2, 1]
    assert manifest["coverage"]["source_rating"] == {
        "maximum": 1200,
        "minimum": 1200,
        "values_present": 10,
    }


def _short_game(
    *,
    site: str,
    event: str = "Rated Blitz game",
    extra_headers: str = "",
) -> str:
    return f"""
[Event "{event}"]
[Site "https://example.test/{site}"]
[Date "2026.07.16"]
[Round "-"]
[White "White"]
[Black "Black"]
[Result "1-0"]
[WhiteElo "1200"]
[BlackElo "1200"]
{extra_headers}
1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0

""".lstrip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
