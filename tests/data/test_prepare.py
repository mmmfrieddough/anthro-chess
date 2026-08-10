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

from anthro_chess.chess import (
    DRAW_CLAIM_ACTION_ID,
    RESIGNATION_ACTION_ID,
    decode_move,
    is_terminal_action,
)
from anthro_chess.config import load_config
from anthro_chess.data import (
    ArchiveConfig,
    DataPreparationError,
    PrepareConfig,
    SourceConfig,
    acquire_configured_archive,
    prepare_pgn,
)
from anthro_chess.data.accounts import (
    account_row_digest,
    marked_accounts_from_usernames,
)
from anthro_chess.data.schema import (
    NORMALIZED_COLUMNS,
    decode_clock_remaining_deltas,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
SAMPLE_PGN = REPOSITORY_ROOT / "samples/lichess/standard-export-sample.pgn"
SAMPLE_CONFIG = REPOSITORY_ROOT / "configs/data/lichess-sample.toml"
BASELINE_CONFIG = REPOSITORY_ROOT / "configs/data/lichess-blitz-2017-04.toml"
UNIV_CONFIG = REPOSITORY_ROOT / "configs/data/lichess-univ-2017-04-2021-06.toml"


def test_baseline_selection_pins_one_bounded_verified_rating_namespace() -> None:
    config = load_config(PrepareConfig, path=BASELINE_CONFIG).value

    assert config.source.rating_namespace == "lichess_blitz"
    assert config.filters.event_speed == "blitz"
    assert config.filters.maximum_games is not None
    assert config.output.games_per_shard is not None
    assert config.split.require_nonempty is True
    assert len(config.archives) == 1
    assert config.archives[0].sha256 == (
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
    assert decode_clock_remaining_deltas(row["clock_remaining_delta_ms"])[:3] == [
        30000,
        30000,
        29000,
    ]
    assert set(row["clock_status"]) == {"present"}
    assert row["clock_precision_ms"] == 1000

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
    assert decode_clock_remaining_deltas(row["clock_remaining_delta_ms"]) == [
        0,
        None,
        12340,
        None,
    ]
    assert row["clock_status"] == [
        "present",
        "rejected",
        "present",
        "unavailable",
    ]
    assert row["clock_precision_ms"] == 10
    assert row["termination"] is None
    assert row["termination_status"] == "unavailable"


def test_filters_games_and_records_rejection_reasons(tmp_path: Path) -> None:
    input_path = tmp_path / "filtered.pgn"
    input_path.write_text(
        _short_game(site="accepted")
        + _short_game(site="bot", extra_headers='[WhiteTitle "BOT"]\n')
        + _short_game(site="unrated", event="Casual test game")
        + _short_game(
            site="infraction",
            extra_headers='[Termination "Rules infraction"]\n',
        ),
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
    assert result.rejected_games == 3
    manifest = _read_json(result.manifest_path)
    assert manifest["games"]["rejection_reasons"] == {
        "bot_game": 1,
        "rules_infraction": 1,
        "unrated_game": 1,
    }


def test_keeps_a_game_whose_clock_precision_varies_at_its_finest_tick(
    tmp_path: Path,
) -> None:
    """A game whose plies print clocks at two ticks records the finer one."""

    input_path = tmp_path / "mixed-precision.pgn"
    input_path.write_text(
        _short_game(site="mixed", moves="1. e4 { [%clk 0:01:00] } e5 { [%clkc 5900] }"),
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
            "filters.minimum_plies=1",
        ),
    )

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    row = pq.read_table(result.normalized_path).to_pylist()[0]
    assert result.accepted_games == 1
    assert decode_clock_remaining_deltas(row["clock_remaining_delta_ms"]) == [
        60_000,
        59_000,
    ]
    assert row["clock_precision_ms"] == 10


def test_records_the_player_digests_a_marked_snapshot_is_matched_against(
    tmp_path: Path,
) -> None:
    """The corpus carries digests rather than names, and they must line up."""

    input_path = tmp_path / "players.pgn"
    input_path.write_text(
        _short_game(site="named", white="Alice", black="Bob"), encoding="utf-8"
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
    assert row["white_player_digest"] == account_row_digest("alice")
    assert row["black_player_digest"] == account_row_digest("bob")


def test_rejects_every_game_a_marked_account_played(tmp_path: Path) -> None:
    input_path = tmp_path / "marked.pgn"
    input_path.write_text(
        _short_game(site="clean")
        + _short_game(site="marked-white", white="Cheater")
        + _short_game(site="marked-black", black="Cheater"),
        encoding="utf-8",
    )
    snapshot = marked_accounts_from_usernames(
        ["cheater"],
        archive_sha256=sha256(input_path.read_bytes()).hexdigest(),
        queried_at="2026-08-08",
        accounts_queried=6,
    ).write(tmp_path / "marked-accounts.txt")
    resolved = load_config(
        PrepareConfig,
        path=SAMPLE_CONFIG,
        overrides=(f'filters.marked_accounts="{snapshot}"',),
    )

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    # Both colours are checked.
    assert result.accepted_games == 1
    manifest = _read_json(result.manifest_path)
    assert manifest["games"]["rejection_reasons"] == {"marked_account": 2}
    assert manifest["marked_accounts"]["accounts_marked"] == 1
    assert manifest["marked_accounts"]["accounts_queried"] == 6


def test_refuses_to_prepare_an_archive_the_snapshot_does_not_cover(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "widened.pgn"
    input_path.write_text(_short_game(site="accepted"), encoding="utf-8")
    snapshot = marked_accounts_from_usernames(
        ["cheater"],
        archive_sha256="f" * 64,
        queried_at="2026-08-08",
        accounts_queried=6,
    ).write(tmp_path / "marked-accounts.txt")
    resolved = load_config(
        PrepareConfig,
        path=SAMPLE_CONFIG,
        overrides=(f'filters.marked_accounts="{snapshot}"',),
    )

    with pytest.raises(DataPreparationError, match="does not cover archive"):
        prepare_pgn(input_path, tmp_path / "artifacts", resolved)


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
    archive = ArchiveConfig(
        url="https://example.test/archive.pgn.zst",
        file_name="archive.pgn.zst",
        sha256=expected_sha256,
    )
    prepare_module = import_module("anthro_chess.data.prepare")
    calls = 0

    def fake_urlopen(_request: object, *, timeout: int) -> BytesIO:
        nonlocal calls
        assert timeout == 60
        calls += 1
        return BytesIO(payload)

    monkeypatch.setattr(prepare_module, "urlopen", fake_urlopen)

    acquired = acquire_configured_archive(tmp_path, archive)
    reused = acquire_configured_archive(tmp_path, archive)

    assert acquired.archive_path.read_bytes() == payload
    assert acquired.sha256 == expected_sha256
    assert acquired.reused is False
    assert reused.reused is True
    assert calls == 1


def test_rejects_download_with_wrong_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = ArchiveConfig(
        url="https://example.test/archive.pgn.zst",
        file_name="archive.pgn.zst",
        sha256="0" * 64,
    )
    prepare_module = import_module("anthro_chess.data.prepare")
    monkeypatch.setattr(
        prepare_module,
        "urlopen",
        lambda _request, timeout: BytesIO(b"wrong bytes"),
    )

    with pytest.raises(DataPreparationError, match="checksum mismatch"):
        acquire_configured_archive(tmp_path, archive)

    assert not (tmp_path / "raw/archive.pgn.zst").exists()
    assert not (tmp_path / "raw/archive.pgn.zst.part").exists()


def test_prepare_rejects_input_that_does_not_match_pinned_archive(
    tmp_path: Path,
) -> None:
    # The baseline selection pins the real monthly archive, which the checked-in
    # sample is not.
    resolved = load_config(PrepareConfig, path=BASELINE_CONFIG)

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
            "split.validation_fraction=0.4",
            "split.test_fraction=0.4",
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


#: One game per ending preparation has to classify from a real PGN. The clock
#: comments carry the abandonment evidence: ``flagged`` is a genuine flag fall
#: and ``walked-away`` is a player who quit with most of their time left.
_ENDING_GAMES: tuple[tuple[str, str], ...] = (
    ("mated", "checkmate"),
    ("resigned-on-turn", "resignation"),
    ("resigned-off-turn", "resignation"),
    ("flagged", "clock_expiry"),
    ("walked-away", "abandonment"),
    ("unterminated", "unknown"),
)


def _ending_corpus() -> str:
    return (
        _ended_game(
            site="mated",
            result="1-0",
            movetext="1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#",
            termination="Normal",
        )
        # Three plies leave Black, the losing player, on move.
        + _ended_game(
            site="resigned-on-turn",
            result="1-0",
            movetext="1. e4 e5 2. Nf3",
            termination="Normal",
        )
        # Two plies leave White on move while Black is the losing player, so
        # the resignation was made on the opponent's clock.
        + _ended_game(
            site="resigned-off-turn",
            result="1-0",
            movetext="1. e4 e5",
            termination="Normal",
        )
        + _ended_game(
            site="flagged",
            result="1-0",
            movetext="1. e4 { [%clk 0:04:55] } e5 { [%clk 0:00:01] }",
            termination="Time forfeit",
        )
        + _ended_game(
            site="walked-away",
            result="1-0",
            movetext="1. e4 { [%clk 0:04:55] } e5 { [%clk 0:04:50] }",
            termination="Time forfeit",
        )
        + _ended_game(
            site="unterminated",
            result="1-0",
            movetext="1. e4 e5",
            termination="Unterminated",
        )
    )


def test_derives_a_termination_category_for_every_ending(tmp_path: Path) -> None:
    input_path = tmp_path / "endings.pgn"
    input_path.write_text(_ending_corpus(), encoding="utf-8")
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    rows = {
        row["source_game_key"]: row
        for row in pq.read_table(result.normalized_path).to_pylist()
    }
    assert {
        site: rows[site]["termination_category"] for site, _ in _ENDING_GAMES
    } == dict(_ENDING_GAMES)


def test_retains_the_raw_source_termination_beside_the_derived_category(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "endings.pgn"
    input_path.write_text(_ending_corpus(), encoding="utf-8")
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    rows = {
        row["source_game_key"]: row
        for row in pq.read_table(result.normalized_path).to_pylist()
    }
    # The source collapses all three into one value; only the derivation
    # separates them.
    assert {rows[site]["termination"] for site in ("mated", "resigned-on-turn")} == {
        "normal"
    }
    assert rows["flagged"]["termination"] == "time_forfeit"
    assert rows["walked-away"]["termination"] == "time_forfeit"


def test_records_whether_an_ending_was_attributable_to_the_side_to_move(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "endings.pgn"
    input_path.write_text(_ending_corpus(), encoding="utf-8")
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    rows = {
        row["source_game_key"]: row
        for row in pq.read_table(result.normalized_path).to_pylist()
    }
    assert rows["resigned-on-turn"]["termination_by_side_to_move"] is True
    assert rows["resigned-off-turn"]["termination_by_side_to_move"] is False
    # A checkmate is nobody's decision, which is a different statement from a
    # decision made off turn.
    assert rows["mated"]["termination_by_side_to_move"] is None


def test_reports_termination_composition_without_re_deriving_it(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "endings.pgn"
    input_path.write_text(_ending_corpus(), encoding="utf-8")
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    coverage = _read_json(result.manifest_path)["coverage"]["termination"]
    assert coverage["category_games"]["resignation"] == 2
    assert coverage["category_games"]["checkmate"] == 1
    assert coverage["category_games"]["abandonment"] == 1
    assert coverage["category_games"]["clock_expiry"] == 1
    assert coverage["category_games"]["unknown"] == 1
    assert coverage["category_games"]["stalemate"] == 0
    assert coverage["attribution_games"] == {
        "side_to_move": 1,
        "opponent_to_move": 1,
        "not_applicable": 4,
    }
    assert coverage["abandonment"] == {
        "clock_share_threshold": 0.3,
        "clock_share_judged_games": 2,
    }


def test_the_abandonment_threshold_is_configurable(tmp_path: Path) -> None:
    input_path = tmp_path / "endings.pgn"
    input_path.write_text(_ending_corpus(), encoding="utf-8")
    resolved = load_config(
        PrepareConfig,
        path=SAMPLE_CONFIG,
        overrides=("termination.abandonment_clock_share=0.99",),
    )

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    coverage = _read_json(result.manifest_path)["coverage"]["termination"]
    assert coverage["category_games"]["abandonment"] == 0
    assert coverage["category_games"]["clock_expiry"] == 2


def _claimed_draw_corpus() -> str:
    return (
        # Eight reversible plies return the starting position for a third time,
        # so the player to move can claim without announcing anything.
        _ended_game(
            site="claimed-on-turn",
            result="1/2-1/2",
            movetext="1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8",
            termination="Normal",
        )
        # One ply earlier the claim exists only alongside the move that would
        # repeat the position, which the claim action cannot express.
        + _ended_game(
            site="claimed-with-announced-move",
            result="1/2-1/2",
            movetext="1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1",
            termination="Normal",
        )
    )


def test_appends_a_terminal_action_for_a_decision_made_on_the_players_turn(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "endings.pgn"
    input_path.write_text(_ending_corpus(), encoding="utf-8")
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    rows = {
        row["source_game_key"]: row
        for row in pq.read_table(result.normalized_path).to_pylist()
    }
    resigned = rows["resigned-on-turn"]
    assert resigned["terminal_action_status"] == "appended"
    assert resigned["action_ids"][-1] == RESIGNATION_ACTION_ID
    # The ply count stays the move count, so a terminal action never changes a
    # game's length or the prefixes taken from it.
    assert resigned["ply_count"] == 3
    assert len(resigned["action_ids"]) == 4
    # The per-ply columns stay aligned with the actions, with no invented clock.
    for column in ("clock_remaining_delta_ms", "clock_status"):
        assert len(resigned[column]) == 4
    assert (
        decode_clock_remaining_deltas(resigned["clock_remaining_delta_ms"])[-1] is None
    )
    assert resigned["clock_status"][-1] == "unavailable"


def test_omits_a_terminal_action_the_player_could_not_have_chosen(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "endings.pgn"
    input_path.write_text(_ending_corpus(), encoding="utf-8")
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    rows = {
        row["source_game_key"]: row
        for row in pq.read_table(result.normalized_path).to_pylist()
    }
    off_turn = rows["resigned-off-turn"]
    assert off_turn["terminal_action_status"] == "omitted_opponent_to_move"
    assert len(off_turn["action_ids"]) == off_turn["ply_count"] == 2
    assert all(not is_terminal_action(action) for action in off_turn["action_ids"])
    # An ending nobody decided is a different statement from an omission.
    assert rows["mated"]["terminal_action_status"] == "not_applicable"
    assert rows["flagged"]["terminal_action_status"] == "not_applicable"


def test_appends_a_draw_claim_only_where_the_final_position_allows_one(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "claims.pgn"
    input_path.write_text(_claimed_draw_corpus(), encoding="utf-8")
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    rows = {
        row["source_game_key"]: row
        for row in pq.read_table(result.normalized_path).to_pylist()
    }
    claimed = rows["claimed-on-turn"]
    assert claimed["termination_category"] == "threefold_repetition"
    assert claimed["terminal_action_status"] == "appended"
    assert claimed["action_ids"][-1] == DRAW_CLAIM_ACTION_ID
    assert claimed["ply_count"] == 8

    announced = rows["claimed-with-announced-move"]
    assert announced["termination_category"] == "threefold_repetition"
    assert announced["terminal_action_status"] == "omitted_claim_unavailable"
    assert len(announced["action_ids"]) == announced["ply_count"] == 7


def test_reports_terminal_action_composition_beside_the_endings(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "endings.pgn"
    input_path.write_text(_ending_corpus(), encoding="utf-8")
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    coverage = _read_json(result.manifest_path)["coverage"]["termination"]
    assert coverage["terminal_action_games"] == {
        "appended": 1,
        "not_applicable": 4,
        "omitted_opponent_to_move": 1,
        "omitted_claim_unavailable": 0,
    }
    # Clock coverage still describes the moves the source reported, so the
    # appended action's empty observation is not counted as a source gap.
    clock_statuses = _read_json(result.manifest_path)["coverage"]["clock"][
        "status_plies"
    ]
    assert (
        sum(clock_statuses.values())
        == _read_json(result.manifest_path)["games"]["plies"]["total"]
    )


def test_derives_an_ending_from_the_position_when_the_source_is_silent(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "silent.pgn"
    input_path.write_text(
        _ended_game(
            site="mated-unlabelled",
            result="1-0",
            movetext="1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#",
        )
        + _ended_game(site="quiet-unlabelled", result="1-0", movetext="1. e4 e5"),
        encoding="utf-8",
    )
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    rows = {
        row["source_game_key"]: row
        for row in pq.read_table(result.normalized_path).to_pylist()
    }
    # Exact logic still classifies the mate, while a decided game with no rule
    # ending and no source field cannot be classified at all.
    assert rows["mated-unlabelled"]["termination_category"] == "checkmate"
    assert rows["mated-unlabelled"]["termination_status"] == "unavailable"
    assert rows["quiet-unlabelled"]["termination_category"] == "unknown"


def _ended_game(
    *,
    site: str,
    result: str,
    movetext: str,
    termination: str | None = None,
    time_control: str = "300+0",
) -> str:
    """Return one rated game whose ending the derivation has to classify."""

    termination_header = (
        f'[Termination "{termination}"]\n' if termination is not None else ""
    )
    return f"""
[Event "Rated Blitz game"]
[Site "https://example.test/{site}"]
[Date "2026.07.16"]
[Round "-"]
[White "White"]
[Black "Black"]
[Result "{result}"]
[WhiteElo "1200"]
[BlackElo "1200"]
[TimeControl "{time_control}"]
{termination_header}
{movetext} {result}

""".lstrip()


def _short_game(
    *,
    site: str,
    event: str = "Rated Blitz game",
    extra_headers: str = "",
    white: str = "White",
    black: str = "Black",
    moves: str = "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0",
) -> str:
    return f"""
[Event "{event}"]
[Site "https://example.test/{site}"]
[Date "2026.07.16"]
[Round "-"]
[White "{white}"]
[Black "{black}"]
[Result "1-0"]
[WhiteElo "1200"]
[BlackElo "1200"]
{extra_headers}
{moves}

""".lstrip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_centisecond_selection_pins_every_month_that_carries_clocks() -> None:
    """The corpus selection is a checked-in fact rather than prose in an issue."""

    config = load_config(PrepareConfig, path=UNIV_CONFIG).value

    assert len(config.archives) == 51
    assert {archive.compression for archive in config.archives} == {"bzip2"}
    months = [archive.file_name.split("_")[-1][:7] for archive in config.archives]
    # Centisecond clocks begin at 2017-04 and the export ends at 2021-06, so a
    # month outside the span carries nothing this source was chosen for.
    assert months[0] == "2017-04"
    assert months[-1] == "2021-06"
    assert months == sorted(months)
    assert len(set(months)) == len(months)
    # Left unset until the namespace is derived per game, because a corpus
    # spanning speeds cannot be described by one namespace.
    assert config.source.rating_namespace is None


def test_a_selection_refuses_two_archives_that_would_overwrite_each_other() -> None:
    """Two archives at one path overwrite each other, so neither ever stays."""

    archive = ArchiveConfig(
        artifact_name="shared",
        url="https://example.test/a.pgn.zst",
        file_name="same.pgn.zst",
        sha256="a" * 64,
    )
    twin = archive.model_copy(update={"sha256": "b" * 64})

    with pytest.raises(ValueError, match="acquired to the same path"):
        PrepareConfig(
            artifact_name="fixture",
            source=SourceConfig(
                id="test", version="v", url="https://example.test/", license="CC0-1.0"
            ),
            archives=(archive, twin),
        )


def test_reads_a_bzip2_archive_the_universal_export_publishes(tmp_path: Path) -> None:
    """The chosen source publishes bzip2 and nothing else."""

    import bz2

    archive_path = tmp_path / "games.pgn.bz2"
    archive_path.write_bytes(bz2.compress(SAMPLE_PGN.read_bytes()))
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    result = prepare_pgn(archive_path, tmp_path / "artifacts", resolved)

    assert result.accepted_games == 1


def _two_archive_selection(tmp_path: Path) -> Path:
    """Written as TOML so the selection is validated the way a real config is."""

    path = tmp_path / "two-archives.toml"
    path.write_text(
        f"""
artifact_name = "fixture"

[source]
id = "test"
version = "fixture"
url = "https://example.test/"
license = "CC0-1.0"

[[archives]]
artifact_name = "fixture-one"
url = "https://example.test/one.pgn.zst"
file_name = "one.pgn.zst"
sha256 = "{"1" * 64}"

[[archives]]
artifact_name = "fixture-two"
url = "https://example.test/two.pgn.bz2"
file_name = "two.pgn.bz2"
sha256 = "{"2" * 64}"
compression = "bzip2"
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_preparation_refuses_a_selection_it_cannot_build_one_corpus_from(
    tmp_path: Path,
) -> None:
    """One run writes one manifest, so a second archive would replace the first."""

    resolved = load_config(PrepareConfig, path=_two_archive_selection(tmp_path))

    with pytest.raises(DataPreparationError, match="pins 2 archives"):
        prepare_pgn(SAMPLE_PGN, tmp_path / "artifacts", resolved)


def test_an_archive_may_not_declare_a_compression_its_name_contradicts() -> None:
    """Readers dispatch on the suffix, so the declaration has to agree with it."""

    with pytest.raises(ValueError, match="does not end in .bz2"):
        ArchiveConfig(
            url="https://example.test/a.pgn.zst",
            file_name="a.pgn.zst",
            sha256="a" * 64,
            compression="bzip2",
        )


def test_a_truncated_bzip2_archive_reports_rather_than_escapes(tmp_path: Path) -> None:
    """A truncated stream raises EOFError, not the OSError callers catch."""

    import bz2

    archive_path = tmp_path / "games.pgn.bz2"
    complete = bz2.compress(SAMPLE_PGN.read_bytes())
    archive_path.write_bytes(complete[: len(complete) // 2])
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)

    with pytest.raises(DataPreparationError, match="cannot decompress input PGN"):
        prepare_pgn(archive_path, tmp_path / "artifacts", resolved)
