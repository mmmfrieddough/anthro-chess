import json
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import chess
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from anthro_chess.chess import decode_move
from anthro_chess.config import load_config
from anthro_chess.data import (
    DataPreparationError,
    PrepareConfig,
    prepare_pgn,
)
from anthro_chess.data.schema import NORMALIZED_COLUMNS

REPOSITORY_ROOT = Path(__file__).parents[2]
SAMPLE_PGN = REPOSITORY_ROOT / "samples/lichess/standard-export-sample.pgn"
SAMPLE_CONFIG = REPOSITORY_ROOT / "configs/data/lichess-sample.toml"


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
    assert manifest["games"] == {
        "accepted": 1,
        "rejected": 0,
        "rejection_reasons": {},
    }
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


def _short_game(
    *,
    site: str,
    event: str = "Rated test game",
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
