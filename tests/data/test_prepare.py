import json
from hashlib import sha256
from importlib import import_module
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, cast

import chess
import chess.pgn
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
    Speed,
    acquire_configured_archive,
    prepare_archives,
    prepare_pgn,
)
from anthro_chess.data.accounts import (
    MarkedAccounts,
    account_digest,
    account_row_digest,
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
    assert config.filters.speed is Speed.BLITZ
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
    assert [entry["sha256"] for entry in manifest["inputs"]] == [_sha256(SAMPLE_PGN)]
    assert manifest["output"]["shards"][0]["sha256"] == _sha256(result.normalized_path)
    assert manifest["source"]["license"] == "CC0-1.0"
    assert manifest["games"]["accepted"] == 1
    assert manifest["games"]["rejected"] == 0
    assert manifest["games"]["rejection_reasons"] == {}
    assert manifest["games"]["plies"]["total"] == 26
    assert manifest["output"]["shards"][0]["games"] == 1
    assert manifest["resolved_config"] == resolved.as_record()
    assert manifest["action_vocabulary"]["sha256"]
    assert manifest["split"]["counts"] == result.split_counts
    assert result.corpus_archives == 1
    assert result.disposition == "prepared"


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


def test_a_speed_filter_rejects_a_game_whose_time_control_says_nothing(
    tmp_path: Path,
) -> None:
    """A speed label in the event text is not evidence of the speed."""

    input_path = tmp_path / "unclocked.pgn"
    input_path.write_text(
        _short_game(site="blitz-by-clock")
        + _short_game(site="blitz-by-label-only", time_control="-"),
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
            'filters.speed="blitz"',
        ),
    )

    result = prepare_pgn(input_path, tmp_path / "artifacts", resolved)

    assert result.accepted_games == 1
    manifest = _read_json(result.manifest_path)
    assert manifest["games"]["rejection_reasons"] == {"speed_mismatch": 1}


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
    snapshot = MarkedAccounts(
        covers_archives=(sha256(input_path.read_bytes()).hexdigest(),),
        queried_at="2026-08-08",
        accounts_total=8,
        accounts_queried=6,
        slots_total=100,
        slots_queried=90,
        digests=frozenset({account_digest("cheater")}),
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
    # Recorded per archive, with the coverage the census had reached, because a
    # snapshot rejects what it caught rather than everything there is.
    assert manifest["inputs"][0]["marked_accounts"]["accounts_marked"] == 1
    assert manifest["inputs"][0]["marked_accounts"]["accounts_queried"] == 6
    assert manifest["inputs"][0]["marked_accounts"]["slots_queried"] == 90


def test_refuses_to_prepare_an_archive_the_snapshot_does_not_cover(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "widened.pgn"
    input_path.write_text(_short_game(site="accepted"), encoding="utf-8")
    snapshot = MarkedAccounts(
        covers_archives=("f" * 64,),
        queried_at="2026-08-08",
        accounts_total=8,
        accounts_queried=6,
        slots_total=100,
        slots_queried=90,
        digests=frozenset({account_digest("cheater")}),
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

    with pytest.raises(DataPreparationError, match="matches none of the 1 archive"):
        prepare_pgn(SAMPLE_PGN, tmp_path / "artifacts", resolved)


def test_streams_zstandard_input_into_bounded_shards_and_one_speed(
    tmp_path: Path,
) -> None:
    games = (
        _short_game(site="rapid", time_control="600+0")
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
            'filters.speed="blitz"',
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
        "speed_mismatch": 1,
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
    time_control: str = "300+0",
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
[TimeControl "{time_control}"]
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


#: Games in each archive :func:`_monthly_archives` writes. Several tests count
#: against it, as a corpus total and as a bound one archive exactly fills.
_GAMES_PER_ARCHIVE = 4


def _monthly_archives(tmp_path: Path) -> tuple[Path, Path]:
    """Write two archives that stand in for two months of one source."""

    paths = []
    for month in ("2017-04", "2017-05"):
        path = tmp_path / f"games-{month}.pgn.zst"
        games = "".join(
            _short_game(site=f"{month}-{index}") for index in range(_GAMES_PER_ARCHIVE)
        )
        path.write_bytes(zstandard.ZstdCompressor().compress(games.encode()))
        paths.append(path)
    return paths[0], paths[1]


def _selection_over(tmp_path: Path, *archives: Path) -> Path:
    """Pin real archives by their observed digests, as a checked-in one does.

    Written as TOML so the selection is validated the way a real config is.
    """

    entries = "\n".join(
        f"""
[[archives]]
artifact_name = "fixture-{index}"
url = "https://example.test/{path.name}"
file_name = "{path.name}"
sha256 = "{_sha256(path)}"
"""
        for index, path in enumerate(archives)
    )
    path = tmp_path / "selection.toml"
    path.write_text(
        f"""
artifact_name = "fixture"

[source]
id = "test"
version = "fixture"
url = "https://example.test/"
license = "CC0-1.0"

[output]
games_per_shard = 3
{entries}
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_appends_a_second_archive_to_the_corpus_the_first_built(
    tmp_path: Path,
) -> None:
    """One corpus is built from many archives, one run at a time."""

    first, second = _monthly_archives(tmp_path)
    resolved = load_config(PrepareConfig, path=_selection_over(tmp_path, first, second))
    output = tmp_path / "artifacts"

    opening = prepare_pgn(first, output, resolved)
    appended = prepare_pgn(second, output, resolved)

    assert opening.corpus_archives == 1
    assert appended.corpus_archives == 2
    manifest = _read_json(appended.manifest_path)
    assert [entry["sha256"] for entry in manifest["inputs"]] == [
        _sha256(first),
        _sha256(second),
    ]
    assert [entry["file_name"] for entry in manifest["inputs"]] == [
        first.name,
        second.name,
    ]
    # Every shard says which archive it came from, and the earlier archive's
    # shards are still there beside the later one's.
    shards = manifest["output"]["shards"]
    assert {shard["input_sha256"] for shard in shards} == {
        _sha256(first),
        _sha256(second),
    }
    assert len({shard["path"] for shard in shards}) == len(shards)
    assert all((output / shard["path"]).is_file() for shard in shards)
    assert sorted(
        path.name for path in (output / "normalized").glob("games*.parquet")
    ) == sorted(Path(shard["path"]).name for shard in shards)
    assert manifest["games"]["accepted"] == sum(
        entry["games"]["accepted"] for entry in manifest["inputs"]
    )
    assert manifest["games"]["scanned"] == 8
    assert sum(appended.split_counts.values()) == manifest["games"]["accepted"]
    assert appended.accepted_games == 4


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_preparing_archives_at_once_writes_what_preparing_them_in_turn_writes(
    tmp_path: Path,
) -> None:
    """How many archives a machine decodes at once cannot reach the artifact.

    The manifest a concurrent run writes has to order its archives by the
    inputs rather than by whichever finished first, and every shard has to
    land where one run at a time would have put it.
    """

    first, second = _monthly_archives(tmp_path)
    resolved = load_config(PrepareConfig, path=_selection_over(tmp_path, first, second))
    in_turn = tmp_path / "in-turn"
    at_once = tmp_path / "at-once"

    prepare_pgn(first, in_turn, resolved)
    prepare_pgn(second, in_turn, resolved)
    appended = prepare_archives([first, second], at_once, resolved, concurrency=2)

    assert appended.corpus_archives == 2
    assert _tree_digests(in_turn) == _tree_digests(at_once)


def test_preparing_archives_at_once_leaves_the_ones_already_recorded_alone(
    tmp_path: Path,
) -> None:
    """Re-running an interrupted pass from its beginning still costs nothing."""

    first, second = _monthly_archives(tmp_path)
    resolved = load_config(PrepareConfig, path=_selection_over(tmp_path, first, second))
    output = tmp_path / "artifacts"

    prepare_archives([first, second], output, resolved, concurrency=2)
    before = _tree_digests(output)
    again = prepare_archives([first, second], output, resolved, concurrency=2)

    assert again.disposition == "already_prepared"
    assert again.corpus_archives == 2
    assert _tree_digests(output) == before


def test_a_bounded_selection_prepares_its_archives_in_turn(tmp_path: Path) -> None:
    """What an archive may admit depends on what the ones before it took.

    Deciding that for two archives at once would overshoot the bound by
    whatever the second one contributed, so the bound outranks the request.
    """

    first, second = _monthly_archives(tmp_path)
    resolved = load_config(
        PrepareConfig,
        path=_selection_over(tmp_path, first, second),
        overrides=[f"filters.maximum_games={_GAMES_PER_ARCHIVE + 1}"],
    )
    output = tmp_path / "artifacts"

    result = prepare_archives([first, second], output, resolved, concurrency=4)

    manifest = _read_json(result.manifest_path)
    assert manifest["games"]["accepted"] == _GAMES_PER_ARCHIVE + 1
    assert manifest["selection"]["limit_reached"] is True


def test_the_corpus_blocks_report_every_field_the_archive_ones_do(
    tmp_path: Path,
) -> None:
    """A field added to one and not the other would go silently missing."""

    first, _ = _monthly_archives(tmp_path)
    resolved = load_config(PrepareConfig, path=_selection_over(tmp_path, first))

    result = prepare_pgn(first, tmp_path / "artifacts", resolved)

    manifest = _read_json(result.manifest_path)
    (entry,) = manifest["inputs"]
    for block in ("games", "coverage"):
        assert _shape_of(manifest[block]) == _shape_of(entry[block]), block
    # With one archive in, the roll-up is that archive, so the numbers agree too.
    assert manifest["games"] == entry["games"]
    assert manifest["coverage"] == entry["coverage"]
    assert manifest["split"]["counts"] == entry["split_counts"]


def _shape_of(block: Any) -> Any:
    """Return a block's nested key structure, ignoring the values under it."""

    if not isinstance(block, dict):
        return None
    return {key: _shape_of(value) for key, value in sorted(block.items())}


def test_leaves_an_archive_the_corpus_already_holds_alone(tmp_path: Path) -> None:
    """Re-running an archive is what makes an interrupted pass resumable."""

    first, second = _monthly_archives(tmp_path)
    resolved = load_config(PrepareConfig, path=_selection_over(tmp_path, first, second))
    output = tmp_path / "artifacts"

    opening = prepare_pgn(first, output, resolved)
    prepare_pgn(second, output, resolved)
    manifest_before = (output / "manifests/manifest.json").read_text(encoding="utf-8")

    result = prepare_pgn(first, output, resolved)

    assert result.disposition == "already_prepared"
    assert result.accepted_games == 4
    assert result.corpus_archives == 2
    assert result.normalized_paths == opening.normalized_paths
    assert (output / "manifests/manifest.json").read_text(
        encoding="utf-8"
    ) == manifest_before


def test_refuses_to_append_an_archive_under_a_different_selection(
    tmp_path: Path,
) -> None:
    """A corpus half-filtered one way and half another is not one corpus."""

    first, second = _monthly_archives(tmp_path)
    selection = _selection_over(tmp_path, first, second)
    output = tmp_path / "artifacts"
    prepare_pgn(first, output, load_config(PrepareConfig, path=selection))
    refiltered = load_config(
        PrepareConfig,
        path=selection,
        overrides=("filters.minimum_plies=4",),
    )

    with pytest.raises(DataPreparationError, match="different filters"):
        prepare_pgn(second, output, refiltered)


def test_appending_stops_once_the_corpus_reaches_its_bound(tmp_path: Path) -> None:
    """The bound counts the corpus, not each archive, so 51 do not multiply it."""

    first, second = _monthly_archives(tmp_path)
    resolved = load_config(
        PrepareConfig,
        path=_selection_over(tmp_path, first, second),
        overrides=("filters.maximum_games=4",),
    )
    output = tmp_path / "artifacts"

    prepare_pgn(first, output, resolved)
    result = prepare_pgn(second, output, resolved)

    assert result.disposition == "corpus_complete"
    assert result.normalized_paths == ()
    manifest = _read_json(result.manifest_path)
    assert len(manifest["inputs"]) == 1
    assert manifest["games"]["accepted"] == 4
    assert manifest["selection"]["limit_reached"] is True


def test_a_raised_bound_admits_another_archive_into_the_same_corpus(
    tmp_path: Path,
) -> None:
    """Raising a bound keeps every accepted game and adds more, so it appends."""

    first, second = _monthly_archives(tmp_path)
    selection = _selection_over(tmp_path, first, second)
    output = tmp_path / "artifacts"
    prepare_pgn(
        first,
        output,
        load_config(
            PrepareConfig, path=selection, overrides=("filters.maximum_games=4",)
        ),
    )

    result = prepare_pgn(
        second,
        output,
        load_config(
            PrepareConfig, path=selection, overrides=("filters.maximum_games=6",)
        ),
    )

    assert result.disposition == "prepared"
    assert result.accepted_games == 2
    assert _read_json(result.manifest_path)["games"]["accepted"] == 6


def test_appending_clears_an_interrupted_archive_and_keeps_the_others(
    tmp_path: Path,
) -> None:
    """A previous attempt's orphans go; another archive's shards must not."""

    first, second = _monthly_archives(tmp_path)
    resolved = load_config(PrepareConfig, path=_selection_over(tmp_path, first, second))
    output = tmp_path / "artifacts"
    kept = prepare_pgn(first, output, resolved).normalized_paths
    orphan = output / "normalized/games-abcdefabcdef-00000.parquet"
    orphan.write_bytes(b"interrupted")

    prepare_pgn(second, output, resolved)

    assert not orphan.exists()
    assert all(path.is_file() for path in kept)


def test_refuses_a_bound_lowered_below_what_the_corpus_already_holds(
    tmp_path: Path,
) -> None:
    """Adding nothing cannot satisfy a bound the corpus is already past."""

    first, second = _monthly_archives(tmp_path)
    selection = _selection_over(tmp_path, first, second)
    output = tmp_path / "artifacts"
    prepare_pgn(
        first,
        output,
        load_config(
            PrepareConfig, path=selection, overrides=("filters.maximum_games=4",)
        ),
    )

    with pytest.raises(DataPreparationError, match="past the configured maximum"):
        prepare_pgn(
            second,
            output,
            load_config(
                PrepareConfig, path=selection, overrides=("filters.maximum_games=2",)
            ),
        )


def test_records_an_archive_every_filter_rejected_as_an_empty_append(
    tmp_path: Path,
) -> None:
    """A pass over many archives has to be able to mark such a month done."""

    first, second = _monthly_archives(tmp_path)
    casual = tmp_path / "games-casual.pgn.zst"
    casual.write_bytes(
        zstandard.ZstdCompressor().compress(
            _short_game(site="casual", event="Casual Blitz game").encode()
        )
    )
    resolved = load_config(
        PrepareConfig, path=_selection_over(tmp_path, first, second, casual)
    )
    output = tmp_path / "artifacts"
    prepare_pgn(first, output, resolved)

    result = prepare_pgn(casual, output, resolved)

    assert result.disposition == "prepared"
    assert result.accepted_games == 0
    assert result.normalized_paths == ()
    manifest = _read_json(result.manifest_path)
    assert [entry["sha256"] for entry in manifest["inputs"]] == [
        _sha256(first),
        _sha256(casual),
    ]
    assert manifest["inputs"][-1]["games"]["rejection_reasons"] == {"unrated_game": 1}
    # Recorded, so the pass moves on rather than re-reading it every restart.
    assert prepare_pgn(casual, output, resolved).disposition == "already_prepared"


def test_refuses_an_empty_first_archive_rather_than_starting_a_corpus_on_none(
    tmp_path: Path,
) -> None:
    """With nothing before it, an archive that accepts nothing is a mistake."""

    casual = tmp_path / "games-casual.pgn.zst"
    casual.write_bytes(
        zstandard.ZstdCompressor().compress(
            _short_game(site="casual", event="Casual Blitz game").encode()
        )
    )
    resolved = load_config(PrepareConfig, path=_selection_over(tmp_path, casual))

    with pytest.raises(DataPreparationError, match="no games passed preparation"):
        prepare_pgn(casual, tmp_path / "artifacts", resolved)


def test_refuses_a_corpus_whose_recorded_shards_have_gone(tmp_path: Path) -> None:
    """A manifest asserting absent shards must not gain another archive."""

    first, second = _monthly_archives(tmp_path)
    resolved = load_config(PrepareConfig, path=_selection_over(tmp_path, first, second))
    output = tmp_path / "artifacts"
    for shard in prepare_pgn(first, output, resolved).normalized_paths:
        shard.unlink()

    with pytest.raises(DataPreparationError, match="no longer present"):
        prepare_pgn(second, output, resolved)


def test_refuses_a_corpus_manifest_missing_the_blocks_the_rollup_reads(
    tmp_path: Path,
) -> None:
    """A manifest this code did not write reports rather than dying mid-append."""

    first, second = _monthly_archives(tmp_path)
    resolved = load_config(PrepareConfig, path=_selection_over(tmp_path, first, second))
    output = tmp_path / "artifacts"
    manifest_path = prepare_pgn(first, output, resolved).manifest_path
    manifest = _read_json(manifest_path)
    manifest["inputs"][0]["coverage"] = {"clock": {}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataPreparationError, match="input record missing"):
        prepare_pgn(second, output, resolved)


def test_refuses_to_append_under_a_selection_section_it_does_not_know(
    tmp_path: Path,
) -> None:
    """A configuration gaining a section fails closed rather than appending."""

    first, second = _monthly_archives(tmp_path)
    resolved = load_config(PrepareConfig, path=_selection_over(tmp_path, first, second))
    output = tmp_path / "artifacts"
    manifest_path = prepare_pgn(first, output, resolved).manifest_path
    manifest = _read_json(manifest_path)
    manifest["resolved_config"]["config"]["future_section"] = {"shapes_records": True}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataPreparationError, match="different future_section"):
        prepare_pgn(second, output, resolved)


def test_replaces_the_corpus_manifest_atomically(tmp_path: Path) -> None:
    """The manifest is the only record of what is in, so a partial write loses it."""

    first, second = _monthly_archives(tmp_path)
    resolved = load_config(PrepareConfig, path=_selection_over(tmp_path, first, second))
    output = tmp_path / "artifacts"
    prepare_pgn(first, output, resolved)
    written: list[Path] = []

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "anthro_chess.data.prepare.write_text_atomically",
            lambda path, text: written.append(path),
        )
        result = prepare_pgn(second, output, resolved)

    assert written == [result.manifest_path]


def test_two_writers_of_one_path_do_not_share_a_staging_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation and the census both write an archive's counts, unsynchronized."""

    from anthro_chess.data import artifacts
    from anthro_chess.data.artifacts import write_text_atomically

    path = tmp_path / "counts.tsv"
    monkeypatch.setattr(artifacts.os, "getpid", lambda: 111)
    other = path.with_suffix(f"{path.suffix}.222.writing")
    other.write_text("what the other writer is part-way through", encoding="utf-8")

    write_text_atomically(path, "mine")

    assert path.read_text(encoding="utf-8") == "mine"
    assert other.read_text(encoding="utf-8") == (
        "what the other writer is part-way through"
    )
    assert not list(tmp_path.glob("*.111.writing"))


def test_refuses_a_corpus_prepared_before_a_manifest_could_span_archives(
    tmp_path: Path,
) -> None:
    """An older manifest records one input and no per-archive counts."""

    first, second = _monthly_archives(tmp_path)
    resolved = load_config(PrepareConfig, path=_selection_over(tmp_path, first, second))
    output = tmp_path / "artifacts"
    manifest_path = prepare_pgn(first, output, resolved).manifest_path
    manifest = _read_json(manifest_path)
    (entry,) = manifest.pop("inputs")
    manifest["input"] = {"file_name": entry["file_name"], "sha256": entry["sha256"]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataPreparationError, match="records no per-archive inputs"):
        prepare_pgn(second, output, resolved)


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


def test_decoding_across_processes_writes_the_bytes_one_process_would(
    tmp_path: Path,
) -> None:
    """Worker count is a runtime choice, so it may not reach the artifact.

    The corpus spans several decoding jobs and stops at a bound partway
    through one, so the comparison covers the boundaries between them and the
    early exit that leaves jobs queued for games past the bound.
    """

    from anthro_chess.data.prepare import _GAMES_PER_JOB

    games = "".join(
        _short_game(site=f"game-{index}") for index in range(_GAMES_PER_JOB * 2 + 40)
    )
    input_path = tmp_path / "games.pgn"
    input_path.write_text(games + _short_game(site="game-0"), encoding="utf-8")
    overrides = (
        'source.id="test"',
        f"filters.maximum_games={_GAMES_PER_JOB * 2 + 10}",
        "output.games_per_shard=200",
        "split.test_fraction=0.2",
        "split.require_nonempty=true",
    )

    written = []
    for name, workers in (("one", 0), ("many", 3)):
        resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG, overrides=overrides)
        output = tmp_path / name
        result = prepare_pgn(input_path, output, resolved, workers=workers)
        assert result.accepted_games == _GAMES_PER_JOB * 2 + 10
        written.append(
            {
                path.relative_to(output): path.read_bytes()
                for path in sorted(output.rglob("*"))
                if path.is_file()
            }
        )

    assert written[0] == written[1]


#: PGN the framing has to survive, since a run splits the stream on game
#: boundaries before anything parses it: leading noise, an escaped line, one
#: blank line between headers, and a comment holding a blank line of its own.
_AWKWARDLY_FRAMED_PGN = """

% an escaped line the parser ignores
[Event "Rated Blitz game"]
[Site "https://example.test/first"]

[Result "1-0"]
1. e4 { a comment

carrying a blank line } e5 2. Qh5 Nc6 3. Qxf7# 1-0

[Event "Rated Blitz game"]
[Site "https://example.test/second"]
[Result "0-1"]
1. f3 e5 2. g4 Qh4# 0-1"""


@pytest.mark.parametrize("workers", [0, 2])
def test_counts_the_accounts_the_census_orders_by_while_it_reads(
    tmp_path: Path,
    workers: int,
) -> None:
    """Whichever pass counts an archive has to produce the same counts."""

    from anthro_chess.data.census import count_archive_accounts, read_account_games

    input_path = tmp_path / "counted.pgn"
    input_path.write_text(
        _short_game(site="one", white="Busy", black="Quiet")
        + _short_game(site="two", white="BUSY", black="Middling")
        # A game preparation rejects still holds accounts the corpus's archive
        # holds, so the census asks about them too.
        + _short_game(site="three", white="Busy", black="Botted", event="Casual game"),
        encoding="utf-8",
    )
    resolved = load_config(PrepareConfig, path=SAMPLE_CONFIG)
    counts_path = tmp_path / "census/counted.pgn.accounts.tsv"

    prepare_pgn(
        input_path,
        tmp_path / "artifacts",
        resolved,
        workers=workers,
        counts_path=counts_path,
    )

    counted = read_account_games(counts_path)
    assert counted.archive_sha256 == _sha256(input_path)
    assert counted.games_by_account == {
        "busy": 3,
        "quiet": 1,
        "middling": 1,
        "botted": 1,
    }
    assert counted.games_by_account == count_archive_accounts(input_path)


def test_leaves_no_counts_for_an_archive_the_game_bound_cut_short(
    tmp_path: Path,
) -> None:
    """Counts that spoke for part of an archive would read as the whole of it."""

    input_path = tmp_path / "bounded.pgn"
    input_path.write_text(
        _short_game(site="one") + _short_game(site="two"), encoding="utf-8"
    )
    resolved = load_config(
        PrepareConfig, path=SAMPLE_CONFIG, overrides=("filters.maximum_games=1",)
    )
    counts_path = tmp_path / "census/bounded.pgn.accounts.tsv"

    prepare_pgn(input_path, tmp_path / "artifacts", resolved, counts_path=counts_path)

    assert not counts_path.exists()


def test_framing_splits_the_stream_where_the_parser_ends_a_game() -> None:
    """A framed game reparses into the game reading the whole stream gives."""

    from anthro_chess.data.census import ArchiveAccountCounter
    from anthro_chess.data.prepare import _framed_games

    framed = list(
        _framed_games(StringIO(_AWKWARDLY_FRAMED_PGN), ArchiveAccountCounter())
    )
    streamed = StringIO(_AWKWARDLY_FRAMED_PGN)

    assert "".join(framed) == _AWKWARDLY_FRAMED_PGN
    assert len(framed) == 2
    for text in framed:
        expected = chess.pgn.read_game(streamed)
        actual = chess.pgn.read_game(StringIO(text))
        assert expected is not None and actual is not None
        assert dict(actual.headers) == dict(expected.headers)
        assert list(actual.mainline_moves()) == list(expected.mainline_moves())
    assert chess.pgn.read_game(streamed) is None


def _decoded(moves: str, site: str = "edge") -> Any:
    """Decode one rated game and return its single parsed result."""

    from anthro_chess.data.prepare import _decode_batch

    config = load_config(PrepareConfig, path=SAMPLE_CONFIG).value
    (parsed,) = _decode_batch(_short_game(site=site, moves=moves), config, None)
    return parsed


@pytest.mark.parametrize("null", ["--", "Z0", "0000", "@@@@"])
def test_rejects_a_null_move_the_parser_reports_no_error_for(null: str) -> None:
    """The one move ``parse_san`` returns without vouching for its legality.

    Nothing else covers this branch, and the guard that catches it is what
    stands between a null move and ``encode_move`` raising mid-archive.
    """

    parsed = _decoded(f"1. e4 {null} 1-0")

    assert parsed.record is None
    assert parsed.rejection == "illegal_move"


def test_a_comment_after_a_move_less_variation_belongs_to_no_ply() -> None:
    """``in_variation`` is cleared by the variation and no move restores it."""

    parsed = _decoded("1. e4 e5 ( { [%clk 0:01:01] } ) { [%clk 0:04:00] } 2. Nf3 1-0")

    assert parsed.record is not None
    assert parsed.record["clock_status"] == ["unavailable"] * 3
    assert parsed.record["clock_precision_ms"] is None


def test_a_comment_after_a_variation_holding_a_move_belongs_to_the_mainline() -> None:
    """A move inside the variation sets ``in_variation``, and nothing clears it."""

    parsed = _decoded("1. e4 e5 ( 1... c5 ) { [%clk 0:04:00] } 2. Nf3 1-0")

    assert parsed.record is not None
    assert parsed.record["clock_status"] == [
        "unavailable",
        "present",
        "unavailable",
    ]


def test_an_illegal_move_inside_a_variation_rejects_the_whole_game() -> None:
    """Skipping variations outright would accept this game instead.

    The move has to be legal SAN that the position refuses; an unparseable
    token is dropped by the movetext scanner and never reaches the board.
    """

    parsed = _decoded("1. e4 e5 ( 1... Qh4 ) 2. Nf3 1-0")

    assert parsed.record is None
    assert parsed.rejection == "pgn_parse_error"


def _headerless_game(headers: str, moves: str) -> str:
    return f'[Event "Rated Blitz game"]\n[Site "https://example.test/edge"]\n{headers}\n{moves}\n\n'


def test_a_game_with_no_result_tag_and_no_result_token_is_unfinished() -> None:
    """``read_game``'s own headers default ``Result`` to ``*``, and so must these.

    Collecting headers into a plain dict is what makes the decode one pass, and
    a dict starts without the roster defaults a game is entitled to.
    """

    from anthro_chess.data.prepare import _decode_batch

    config = load_config(PrepareConfig, path=SAMPLE_CONFIG).value
    text = _headerless_game('[WhiteElo "1200"]\n[BlackElo "1200"]\n', "1. e4 e5 2. Nf3")

    (parsed,) = _decode_batch(text, config, None)

    assert parsed.record is not None
    assert parsed.record["result"] == "*"


def test_a_result_token_fills_a_result_tag_that_is_still_open() -> None:
    """The movetext token is the only writer of a header this never reads."""

    from anthro_chess.data.prepare import _decode_batch

    config = load_config(PrepareConfig, path=SAMPLE_CONFIG).value
    text = _headerless_game(
        '[Result "*"]\n[WhiteElo "1200"]\n[BlackElo "1200"]\n',
        "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0",
    )

    (parsed,) = _decode_batch(text, config, None)

    assert parsed.record is not None
    assert parsed.record["result"] == "1-0"


def test_an_illegal_mainline_move_is_a_parse_error_not_an_encoded_action() -> None:
    """Pins the ``parse_san`` contract the mainline legality test used to enforce.

    Nothing else would notice if a ``python-chess`` bump started returning a
    move the position refuses, and the pin admits a minor one.
    """

    parsed = _decoded("1. e4 e5 2. Qh4 1-0")

    assert parsed.record is None
    assert parsed.rejection == "pgn_parse_error"


def test_a_header_rejection_outranks_an_error_in_the_movetext() -> None:
    """The movetext of a game the headers reject is skipped, not parsed.

    Reaching the same verdict by parsing it anyway would report whichever of
    the two the movetext raised, which is the reason this changed.
    """

    from anthro_chess.data.prepare import _decode_batch

    config = load_config(PrepareConfig, path=SAMPLE_CONFIG).value
    text = _short_game(
        site="edge", event="Casual Blitz game", moves="1. e4 e5 2. Qh4 1-0"
    )

    (parsed,) = _decode_batch(text, config, None)

    assert parsed.record is None
    assert parsed.rejection == "unrated_game"


def test_a_skipped_game_leaves_the_stream_where_the_next_one_starts() -> None:
    """A batch is one handle, so the skip has to end a game where a parse would.

    Comments and variations are what the skipping scanner tracks instead of
    parsing, and getting either wrong would swallow the game after it.
    """

    from anthro_chess.data.prepare import _decode_batch

    config = load_config(PrepareConfig, path=SAMPLE_CONFIG).value
    text = _short_game(
        site="skipped",
        event="Casual Blitz game",
        moves="1. e4 { good } e5 ( 1... c5 { sicilian } ) 2. Nf3 1-0",
    ) + _short_game(site="kept")

    skipped, kept = _decode_batch(text, config, None)

    assert skipped.rejection == "unrated_game"
    assert kept.record is not None
    assert kept.record["source_game_key"] == "kept"


def test_a_header_the_parser_itself_chokes_on_is_still_a_header_rejection() -> None:
    """The other shape of the rule, which a real archive does not supply.

    ``read_game`` builds the board from ``FEN`` before it reads a move, so an
    unparseable one used to be reported as a parse error. Nothing in the pinned
    export carries one, so only this says which reason such a game now gets.
    """

    from anthro_chess.data.prepare import _decode_batch

    config = load_config(PrepareConfig, path=SAMPLE_CONFIG).value
    text = _short_game(
        site="edge", extra_headers='[SetUp "1"]\n[FEN "not a position"]\n'
    )

    (parsed,) = _decode_batch(text, config, None)

    assert parsed.record is None
    assert parsed.rejection == "nonstandard_initial_position"
