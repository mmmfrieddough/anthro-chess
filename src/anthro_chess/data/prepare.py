"""Normalize standard-chess PGN games into deterministic Parquet artifacts."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import chess
import chess.pgn

from anthro_chess.chess import action_vocabulary_identity, encode_move
from anthro_chess.config import ResolvedConfig
from anthro_chess.data.accounts import (
    MarkedAccountError,
    MarkedAccounts,
    account_row_digest,
    load_marked_accounts,
    resolve_snapshot_path,
)
from anthro_chess.data.artifacts import (
    SOURCE_USER_AGENT,
    DataLoadingError,
    file_sha256,
    open_pgn_text,
    write_normalized_rows,
)
from anthro_chess.data.config import ArchiveConfig, PrepareConfig, SplitConfig
from anthro_chess.data.schema import (
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    SPLIT_NAMES,
    FieldStatus,
    NormalizedColumn,
    SplitName,
    derive_game_id,
    encode_clock_remaining_deltas,
    row_game_id,
)
from anthro_chess.data.termination import (
    TERMINAL_ACTION_STATUSES,
    TERMINATION_CATEGORIES,
    DerivedTermination,
    derive_termination,
    terminal_action_for,
)

_STATUS_PRESENT: FieldStatus = "present"
_STATUS_UNAVAILABLE: FieldStatus = "unavailable"
_STATUS_REJECTED: FieldStatus = "rejected"
_CLOCK_RE = re.compile(r"\[%clk\s+([^\]\s]+)\]")
_CLOCK_CENTISECONDS_RE = re.compile(r"\[%clkc\s+([^\]\s]+)\]")
_CLOCK_SHAPED_RE = re.compile(r"\[%clk(?:c)?\b")
_TIME_CONTROL_RE = re.compile(r"(\d+)\+(\d+)")
_SOURCE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
#: PGN ``Termination "Rules infraction"`` in the form ``_parse_text`` yields.
_RULES_INFRACTION_TERMINATION = "rules_infraction"
_EVENT_SPEED_RE = re.compile(
    r"\b(bullet|blitz|rapid|classical|correspondence)\b",
    re.IGNORECASE,
)
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
#: Manifest labels for whether a player's decision ended the game while they
#: held the move. ``not_applicable`` covers endings no player decided, which is
#: a different statement from a decision made on the opponent's clock.
_ATTRIBUTION_LABELS: dict[bool | None, str] = {
    True: "side_to_move",
    False: "opponent_to_move",
    None: "not_applicable",
}
logger = logging.getLogger(__name__)


class DataPreparationError(ValueError):
    """Raised when a data-preparation run cannot produce a valid artifact."""


@dataclass(frozen=True)
class AcquisitionResult:
    """Verified local archive produced by one acquisition run."""

    archive_path: Path
    sha256: str
    size_bytes: int
    reused: bool


@dataclass(frozen=True)
class PreparationResult:
    """Paths and counts produced by one preparation run."""

    normalized_paths: tuple[Path, ...]
    manifest_path: Path
    accepted_games: int
    rejected_games: int
    split_counts: dict[str, int]

    @property
    def normalized_path(self) -> Path:
        """Return the first shard for compatibility with single-shard callers."""
        return self.normalized_paths[0]


@dataclass(frozen=True)
class _OptionalInteger:
    value: int | None
    status: FieldStatus


@dataclass(frozen=True)
class _ParsedClock:
    value: int | None
    status: FieldStatus
    precision_ms: int | None


@dataclass(frozen=True)
class _ParsedGame:
    record: dict[str, object] | None
    rejection: str | None
    termination: DerivedTermination | None = None


def acquire_archive(
    output_directory: str | Path,
    resolved_config: ResolvedConfig[PrepareConfig],
) -> AcquisitionResult:
    """Download a pinned source archive and verify its configured digest."""
    archive = resolved_config.value.archive
    if archive is None:
        raise DataPreparationError(
            "configuration has no archive selection for data acquisition"
        )
    return acquire_configured_archive(output_directory, archive)


def acquire_configured_archive(
    output_directory: str | Path,
    archive: ArchiveConfig,
) -> AcquisitionResult:
    """Download one explicitly pinned archive into an artifact's raw directory."""

    raw_directory = Path(output_directory) / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    archive_path = raw_directory / archive.file_name
    logger.info("Acquiring configured archive %s", archive.file_name)
    if archive_path.is_file():
        observed_sha256 = file_sha256(archive_path)
        if observed_sha256 == archive.sha256:
            logger.info("Reusing verified archive %s", archive_path)
            return AcquisitionResult(
                archive_path=archive_path,
                sha256=observed_sha256,
                size_bytes=archive_path.stat().st_size,
                reused=True,
            )

    partial_path = raw_directory / f"{archive.file_name}.part"
    request = Request(
        archive.url,
        headers={"User-Agent": SOURCE_USER_AGENT},
    )
    try:
        with (
            urlopen(request, timeout=60) as response,  # noqa: S310
            partial_path.open("wb") as output_file,
        ):
            for chunk in iter(lambda: response.read(_DOWNLOAD_CHUNK_SIZE), b""):
                output_file.write(chunk)
    except (HTTPError, URLError, OSError) as error:
        raise DataPreparationError(
            f"cannot acquire source archive {archive.url}: {error}"
        ) from error

    observed_sha256 = file_sha256(partial_path)
    if observed_sha256 != archive.sha256:
        partial_path.unlink(missing_ok=True)
        raise DataPreparationError(
            f"downloaded archive checksum mismatch: expected {archive.sha256}, "
            f"observed {observed_sha256}"
        )
    partial_path.replace(archive_path)
    logger.info(
        "Acquired and verified archive %s (%s bytes)",
        archive_path,
        archive_path.stat().st_size,
    )
    return AcquisitionResult(
        archive_path=archive_path,
        sha256=observed_sha256,
        size_bytes=archive_path.stat().st_size,
        reused=False,
    )


def prepare_pgn(
    input_path: str | Path,
    output_directory: str | Path,
    resolved_config: ResolvedConfig[PrepareConfig],
) -> PreparationResult:
    """Prepare a PGN stream and write normalized and manifest artifacts."""
    source_path = Path(input_path)
    output_path = Path(output_directory)
    logger.info("Preparing normalized data from %s", source_path)
    if not source_path.is_file():
        raise DataPreparationError(f"input PGN does not exist: {source_path}")

    input_sha256 = file_sha256(source_path)
    configured_archive = resolved_config.value.archive
    if configured_archive is not None and input_sha256 != configured_archive.sha256:
        raise DataPreparationError(
            f"input archive checksum mismatch: expected {configured_archive.sha256}, "
            f"observed {input_sha256}"
        )
    marked_accounts = _resolve_marked_accounts(resolved_config, input_sha256)
    records: list[dict[str, object]] = []
    rejections: Counter[str] = Counter()
    seen_game_ids: set[int] = set()
    split_counts: Counter[str] = Counter()
    clock_status_counts: Counter[str] = Counter()
    clock_precision_counts: Counter[int] = Counter()
    time_initial_status_counts: Counter[str] = Counter()
    time_increment_status_counts: Counter[str] = Counter()
    time_initial_values: list[int] = []
    time_increment_values: list[int] = []
    rating_values: list[int] = []
    termination_category_counts: Counter[str] = Counter()
    termination_attribution_counts: Counter[str] = Counter()
    terminal_action_status_counts: Counter[str] = Counter()
    abandonment_judged_games = 0
    total_plies = 0
    minimum_plies: int | None = None
    maximum_plies: int | None = None
    normalized_paths: list[Path] = []
    output_shards: list[dict[str, object]] = []
    accepted_games = 0
    stopped_at_limit = False

    normalized_directory = output_path / "normalized"
    manifest_directory = output_path / "manifests"

    try:
        with open_pgn_text(source_path) as pgn_file:
            for game in _read_games(pgn_file):
                parsed = _parse_game(game, resolved_config.value, marked_accounts)
                if parsed.record is None:
                    assert parsed.rejection is not None
                    rejections[parsed.rejection] += 1
                else:
                    game_id = _record_game_id(parsed.record)
                    if game_id in seen_game_ids:
                        rejections["duplicate_game"] += 1
                        continue
                    seen_game_ids.add(game_id)
                    records.append(parsed.record)
                    accepted_games += 1
                    split_counts[str(parsed.record[NormalizedColumn.SPLIT])] += 1
                    ply_count_value = parsed.record[NormalizedColumn.PLY_COUNT]
                    if not isinstance(ply_count_value, int):
                        raise TypeError("normalized ply count must be an integer")
                    ply_count = ply_count_value
                    total_plies += ply_count
                    minimum_plies = (
                        ply_count
                        if minimum_plies is None
                        else min(minimum_plies, ply_count)
                    )
                    maximum_plies = (
                        ply_count
                        if maximum_plies is None
                        else max(maximum_plies, ply_count)
                    )
                    clock_statuses = parsed.record[NormalizedColumn.CLOCK_STATUS]
                    if not isinstance(clock_statuses, list):
                        raise TypeError("normalized clock statuses must be a list")
                    # Clock coverage describes the moves a source reported, so
                    # a trailing terminal action's empty observation is left
                    # out rather than counted as a source gap.
                    clock_status_counts.update(
                        str(status) for status in clock_statuses[:ply_count]
                    )
                    clock_precision = parsed.record[NormalizedColumn.CLOCK_PRECISION_MS]
                    if isinstance(clock_precision, int):
                        clock_precision_counts[clock_precision] += 1
                    assert parsed.termination is not None
                    termination_category_counts[parsed.termination.category.value] += 1
                    termination_attribution_counts[
                        _ATTRIBUTION_LABELS[parsed.termination.by_side_to_move]
                    ] += 1
                    terminal_action_status_counts[
                        str(parsed.record[NormalizedColumn.TERMINAL_ACTION_STATUS])
                    ] += 1
                    if parsed.termination.losing_clock_share is not None:
                        abandonment_judged_games += 1
                    time_initial_status_counts[
                        str(parsed.record[NormalizedColumn.TIME_INITIAL_STATUS])
                    ] += 1
                    time_increment_status_counts[
                        str(parsed.record[NormalizedColumn.TIME_INCREMENT_STATUS])
                    ] += 1
                    time_initial = parsed.record[NormalizedColumn.TIME_INITIAL_MS]
                    if isinstance(time_initial, int):
                        time_initial_values.append(time_initial)
                    time_increment = parsed.record[NormalizedColumn.TIME_INCREMENT_MS]
                    if isinstance(time_increment, int):
                        time_increment_values.append(time_increment)
                    for rating_column in (
                        NormalizedColumn.WHITE_SOURCE_RATING,
                        NormalizedColumn.BLACK_SOURCE_RATING,
                    ):
                        rating = parsed.record[rating_column]
                        if isinstance(rating, int):
                            rating_values.append(rating)

                    games_per_shard = resolved_config.value.output.games_per_shard
                    if games_per_shard is not None and len(records) >= games_per_shard:
                        _flush_records(
                            records,
                            output_path=output_path,
                            normalized_directory=normalized_directory,
                            normalized_paths=normalized_paths,
                            output_shards=output_shards,
                            sharded=True,
                        )

                    game_limit = resolved_config.value.filters.maximum_games
                    if game_limit is not None and accepted_games >= game_limit:
                        stopped_at_limit = True
                        break
    except (OSError, UnicodeError, DataLoadingError) as error:
        raise DataPreparationError(
            f"cannot read input PGN {source_path}: {error}"
        ) from error

    if accepted_games == 0:
        detail = ", ".join(
            f"{reason}={count}" for reason, count in sorted(rejections.items())
        )
        raise DataPreparationError(
            "no games passed preparation filters" + (f" ({detail})" if detail else "")
        )

    _flush_records(
        records,
        output_path=output_path,
        normalized_directory=normalized_directory,
        normalized_paths=normalized_paths,
        output_shards=output_shards,
        sharded=resolved_config.value.output.games_per_shard is not None,
    )
    manifest_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_directory / "manifest.json"

    observed_split_counts: dict[str, int] = {
        split_name: split_counts[split_name] for split_name in SPLIT_NAMES
    }
    if resolved_config.value.split.require_nonempty:
        empty_splits = tuple(
            split_name
            for split_name in _requested_splits(resolved_config.value.split)
            if observed_split_counts[split_name] == 0
        )
        if empty_splits:
            raise DataPreparationError(
                "prepared corpus did not produce nonempty "
                f"{', '.join(empty_splits)} split(s)"
            )

    selected_paths = set(normalized_paths)
    for stale_path in normalized_directory.glob("games*.parquet"):
        if stale_path not in selected_paths:
            stale_path.unlink()

    output_record: dict[str, object] = {
        "format": "parquet",
        "compression": "zstd",
        "shards": output_shards,
    }
    if len(output_shards) == 1:
        output_record["path"] = output_shards[0]["path"]
        output_record["sha256"] = output_shards[0]["sha256"]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "source": resolved_config.value.source.model_dump(mode="json"),
        "input": {
            "file_name": source_path.name,
            "sha256": input_sha256,
        },
        "output": output_record,
        "action_vocabulary": action_vocabulary_identity(),
        "selection": {
            "algorithm": "source-order-first-accepted-v1",
            "maximum_games": resolved_config.value.filters.maximum_games,
            "limit_reached": stopped_at_limit,
        },
        "marked_accounts": (
            None
            if marked_accounts is None
            else {
                "covers_archive": marked_accounts.covers_archive,
                "queried_at": marked_accounts.queried_at,
                "accounts_queried": marked_accounts.accounts_queried,
                "accounts_marked": marked_accounts.accounts_marked,
            }
        ),
        "split": {
            "algorithm": "sha256-threshold-v2",
            "seed": resolved_config.value.split.seed,
            "validation_fraction": resolved_config.value.split.validation_fraction,
            "test_fraction": resolved_config.value.split.test_fraction,
            "counts": observed_split_counts,
        },
        "games": {
            "accepted": accepted_games,
            "rejected": sum(rejections.values()),
            "scanned": accepted_games + sum(rejections.values()),
            "rejection_reasons": dict(sorted(rejections.items())),
            "plies": {
                "total": total_plies,
                "minimum_per_game": minimum_plies,
                "maximum_per_game": maximum_plies,
            },
        },
        "coverage": {
            "clock": {
                "status_plies": dict(sorted(clock_status_counts.items())),
                "precision_ms_games": {
                    str(precision): count
                    for precision, count in sorted(clock_precision_counts.items())
                },
            },
            "time_initial_ms": _integer_coverage(
                time_initial_values,
                time_initial_status_counts,
            ),
            "time_increment_ms": _integer_coverage(
                time_increment_values,
                time_increment_status_counts,
            ),
            "source_rating": {
                "values_present": len(rating_values),
                "minimum": min(rating_values) if rating_values else None,
                "maximum": max(rating_values) if rating_values else None,
            },
            "termination": {
                "category_games": {
                    category: termination_category_counts[category]
                    for category in TERMINATION_CATEGORIES
                },
                "attribution_games": {
                    label: termination_attribution_counts[label]
                    for label in _ATTRIBUTION_LABELS.values()
                },
                "terminal_action_games": {
                    status: terminal_action_status_counts[status]
                    for status in TERMINAL_ACTION_STATUSES
                },
                "abandonment": {
                    "clock_share_threshold": (
                        resolved_config.value.termination.abandonment_clock_share
                    ),
                    "clock_share_judged_games": abandonment_judged_games,
                },
            },
        },
        "resolved_config": resolved_config.as_record(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Prepared %s game(s), rejected %s, and wrote %s shard(s)",
        accepted_games,
        sum(rejections.values()),
        len(normalized_paths),
    )
    return PreparationResult(
        normalized_paths=tuple(normalized_paths),
        manifest_path=manifest_path,
        accepted_games=accepted_games,
        rejected_games=sum(rejections.values()),
        split_counts=observed_split_counts,
    )


def _read_games(pgn_file: TextIO) -> Iterator[chess.pgn.Game]:
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            return
        yield game


def _resolve_marked_accounts(
    resolved_config: ResolvedConfig[PrepareConfig],
    input_sha256: str,
) -> MarkedAccounts | None:
    """Load the configured marked-account snapshot and check it applies here."""

    configured = resolved_config.value.filters.marked_accounts
    if configured is None:
        return None
    try:
        snapshot_path = resolve_snapshot_path(
            configured, resolved_config.provenance.source
        )
        snapshot = load_marked_accounts(snapshot_path)
        snapshot.require_archive(input_sha256)
    except MarkedAccountError as error:
        raise DataPreparationError(str(error)) from error
    logger.info(
        "Rejecting games of %s marked account(s) observed %s across %s account(s)",
        snapshot.accounts_marked,
        snapshot.queried_at,
        snapshot.accounts_queried,
    )
    return snapshot


def _parse_game(
    game: chess.pgn.Game,
    config: PrepareConfig,
    marked_accounts: MarkedAccounts | None,
) -> _ParsedGame:
    if game.errors:
        return _ParsedGame(None, "pgn_parse_error")
    variant = game.headers.get("Variant", "Standard")
    if variant.casefold() not in {"standard", "from position"}:
        return _ParsedGame(None, "unsupported_variant")
    if game.headers.get("SetUp") == "1" or game.headers.get("FEN"):
        return _ParsedGame(None, "nonstandard_initial_position")
    termination_text, _ = _parse_text(game.headers.get("Termination"))
    if termination_text == _RULES_INFRACTION_TERMINATION:
        # Ended by the platform on a client-side report rather than played to
        # a finish, so the record is a fragment of a game rather than a
        # completed human one. Rejected for that rather than as a cheating
        # filter, which 0041 measures this label as far too narrow to serve.
        return _ParsedGame(None, "rules_infraction")
    if config.filters.exclude_bots and _has_bot_player(game):
        return _ParsedGame(None, "bot_game")
    if (
        config.filters.event_speed is not None
        and _event_speed(game.headers.get("Event")) != config.filters.event_speed
    ):
        return _ParsedGame(None, "rating_namespace_mismatch")
    if (
        config.filters.require_rated
        and "rated" not in game.headers.get("Event", "").casefold()
    ):
        return _ParsedGame(None, "unrated_game")
    if marked_accounts is not None and _has_marked_player(game, marked_accounts):
        # Every game a marked account played is rejected rather than the moves
        # that were assisted, because no method separates the two per game and
        # a marked player's honest games are not what this corpus is for.
        #
        # Ordered after the speed, rated, and bot filters so the manifest's
        # count is a share of the corpus being built rather than of the whole
        # archive, which is the number 0041 is checked against.
        return _ParsedGame(None, "marked_account")

    source_game_key = _source_game_key(game.headers.get("Site"))
    if source_game_key is None:
        return _ParsedGame(None, "missing_source_game_key")

    white_rating = _parse_nonnegative_integer(game.headers.get("WhiteElo"))
    black_rating = _parse_nonnegative_integer(game.headers.get("BlackElo"))
    if config.filters.require_ratings and (
        white_rating.status != _STATUS_PRESENT or black_rating.status != _STATUS_PRESENT
    ):
        return _ParsedGame(None, "missing_or_invalid_rating")
    time_initial, time_increment = _parse_time_control(game.headers.get("TimeControl"))
    actions: list[int] = []
    clock_values: list[int | None] = []
    clock_statuses: list[FieldStatus] = []
    clock_precision_ms: int | None = None
    board = game.board()
    for node in game.mainline():
        move = node.move
        if move not in board.legal_moves:
            return _ParsedGame(None, "illegal_move")
        actions.append(encode_move(move))
        clock = _parse_clock(node.comment)
        clock_values.append(clock.value)
        clock_statuses.append(clock.status)
        if clock.precision_ms is not None:
            # Precision is inferred per ply from how the source printed the
            # clock, so an exporter that strips a trailing zero infers a coarser
            # tick for that ply alone. The finest tick describes every reading,
            # since a coarser one is representable in it.
            clock_precision_ms = (
                clock.precision_ms
                if clock_precision_ms is None
                else min(clock_precision_ms, clock.precision_ms)
            )
        board.push(move)

    if len(actions) < config.filters.minimum_plies:
        return _ParsedGame(None, "too_short")

    result = game.headers.get("Result")
    if result not in {"1-0", "0-1", "1/2-1/2", "*"}:
        return _ParsedGame(None, "invalid_result")

    termination, termination_status = _parse_text(game.headers.get("Termination"))
    derived_termination = derive_termination(
        result=result,
        source_termination=termination,
        final_board=board,
        clock_remaining_ms=clock_values,
        time_initial_ms=time_initial.value,
        abandonment_clock_share=config.termination.abandonment_clock_share,
    )
    ply_count = len(actions)
    terminal_action_id, terminal_action_status = terminal_action_for(
        derived_termination,
        board,
    )
    if terminal_action_id is not None:
        # The per-ply columns stay aligned with the action sequence, and no
        # source reports a clock for an ending, so the terminal action's
        # observation is explicitly unavailable rather than invented.
        actions.append(terminal_action_id)
        clock_values.append(None)
        clock_statuses.append(_STATUS_UNAVAILABLE)
    game_id = derive_game_id(config.source.id, source_game_key)
    split = _split_name(
        game_id,
        seed=config.split.seed,
        validation_fraction=config.split.validation_fraction,
        test_fraction=config.split.test_fraction,
    )
    normalized_white = (
        white_rating.value if config.source.ratings_are_normalized else None
    )
    normalized_black = (
        black_rating.value if config.source.ratings_are_normalized else None
    )
    normalized_white_status: FieldStatus = (
        white_rating.status
        if config.source.ratings_are_normalized
        else _STATUS_UNAVAILABLE
    )
    normalized_black_status: FieldStatus = (
        black_rating.status
        if config.source.ratings_are_normalized
        else _STATUS_UNAVAILABLE
    )
    return _ParsedGame(
        {
            NormalizedColumn.SCHEMA_VERSION: SCHEMA_VERSION,
            NormalizedColumn.SOURCE_ID: config.source.id,
            NormalizedColumn.SOURCE_GAME_KEY: source_game_key,
            NormalizedColumn.WHITE_PLAYER_DIGEST: _player_digest(
                game.headers.get("White")
            ),
            NormalizedColumn.BLACK_PLAYER_DIGEST: _player_digest(
                game.headers.get("Black")
            ),
            NormalizedColumn.RULESET: "standard",
            NormalizedColumn.INITIAL_POSITION: chess.STARTING_FEN,
            NormalizedColumn.RESULT: result,
            NormalizedColumn.TERMINATION: termination,
            NormalizedColumn.TERMINATION_STATUS: termination_status,
            NormalizedColumn.TERMINATION_CATEGORY: derived_termination.category.value,
            NormalizedColumn.TERMINATION_BY_SIDE_TO_MOVE: (
                derived_termination.by_side_to_move
            ),
            NormalizedColumn.TERMINAL_ACTION_STATUS: terminal_action_status.value,
            NormalizedColumn.PLY_COUNT: ply_count,
            NormalizedColumn.ACTION_IDS: actions,
            NormalizedColumn.WHITE_SOURCE_RATING: white_rating.value,
            NormalizedColumn.WHITE_SOURCE_RATING_STATUS: white_rating.status,
            NormalizedColumn.BLACK_SOURCE_RATING: black_rating.value,
            NormalizedColumn.BLACK_SOURCE_RATING_STATUS: black_rating.status,
            NormalizedColumn.SOURCE_RATING_NAMESPACE: config.source.rating_namespace,
            NormalizedColumn.SOURCE_RATING_SYSTEM: config.source.rating_system,
            NormalizedColumn.WHITE_NORMALIZED_RATING: normalized_white,
            NormalizedColumn.WHITE_NORMALIZED_RATING_STATUS: (normalized_white_status),
            NormalizedColumn.BLACK_NORMALIZED_RATING: normalized_black,
            NormalizedColumn.BLACK_NORMALIZED_RATING_STATUS: (normalized_black_status),
            NormalizedColumn.TIME_INITIAL_MS: time_initial.value,
            NormalizedColumn.TIME_INITIAL_STATUS: time_initial.status,
            NormalizedColumn.TIME_INCREMENT_MS: time_increment.value,
            NormalizedColumn.TIME_INCREMENT_STATUS: time_increment.status,
            NormalizedColumn.CLOCK_REMAINING_DELTA_MS: encode_clock_remaining_deltas(
                clock_values
            ),
            NormalizedColumn.CLOCK_STATUS: clock_statuses,
            NormalizedColumn.CLOCK_PRECISION_MS: clock_precision_ms,
            NormalizedColumn.SPLIT: split,
        },
        None,
        derived_termination,
    )


def _has_marked_player(game: chess.pgn.Game, marked: MarkedAccounts) -> bool:
    return any(
        marked.contains(name)
        for name in (game.headers.get("White"), game.headers.get("Black"))
        if name
    )


def _player_digest(name: str | None) -> int | None:
    if name is None or not name.strip() or name.strip() == "?":
        return None
    return account_row_digest(name)


def _has_bot_player(game: chess.pgn.Game) -> bool:
    return any(
        game.headers.get(header, "").casefold() == "bot"
        for header in ("WhiteTitle", "BlackTitle")
    )


def _event_speed(event: str | None) -> str | None:
    if event is None:
        return None
    match = _EVENT_SPEED_RE.search(event)
    return match.group(1).casefold() if match is not None else None


def _source_game_key(site: str | None) -> str | None:
    if site is None:
        return None
    parsed = urlparse(site)
    candidate = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if not candidate or not _SOURCE_KEY_RE.fullmatch(candidate):
        return None
    return candidate


def _parse_nonnegative_integer(value: str | None) -> _OptionalInteger:
    if value is None or not value.strip():
        return _OptionalInteger(None, _STATUS_UNAVAILABLE)
    try:
        parsed = int(value)
    except ValueError:
        return _OptionalInteger(None, _STATUS_REJECTED)
    if parsed < 0:
        return _OptionalInteger(None, _STATUS_REJECTED)
    return _OptionalInteger(parsed, _STATUS_PRESENT)


def _parse_time_control(
    value: str | None,
) -> tuple[_OptionalInteger, _OptionalInteger]:
    if value is None or not value.strip() or value in {"-", "?"}:
        unavailable = _OptionalInteger(None, _STATUS_UNAVAILABLE)
        return unavailable, unavailable
    match = _TIME_CONTROL_RE.fullmatch(value.strip())
    if match is None:
        rejected = _OptionalInteger(None, _STATUS_REJECTED)
        return rejected, rejected
    return (
        _OptionalInteger(int(match.group(1)) * 1000, _STATUS_PRESENT),
        _OptionalInteger(int(match.group(2)) * 1000, _STATUS_PRESENT),
    )


def _parse_clock(comment: str) -> _ParsedClock:
    clock_match = _CLOCK_RE.search(comment)
    if clock_match is not None:
        clock_text = clock_match.group(1)
        value = _clock_text_to_milliseconds(clock_text)
        return (
            _ParsedClock(value, _STATUS_PRESENT, _clock_precision_ms(clock_text))
            if value is not None
            else _ParsedClock(None, _STATUS_REJECTED, None)
        )
    centiseconds_match = _CLOCK_CENTISECONDS_RE.search(comment)
    if centiseconds_match is not None:
        try:
            centiseconds = int(centiseconds_match.group(1))
        except ValueError:
            return _ParsedClock(None, _STATUS_REJECTED, None)
        if centiseconds < 0:
            return _ParsedClock(None, _STATUS_REJECTED, None)
        return _ParsedClock(centiseconds * 10, _STATUS_PRESENT, 10)
    if _CLOCK_SHAPED_RE.search(comment):
        return _ParsedClock(None, _STATUS_REJECTED, None)
    return _ParsedClock(None, _STATUS_UNAVAILABLE, None)


def _clock_text_to_milliseconds(value: str) -> int | None:
    parts = value.split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return None
    milliseconds = round((hours * 3600 + minutes * 60 + seconds) * 1000)
    return milliseconds


def _clock_precision_ms(value: str) -> int:
    seconds = value.rsplit(":", 1)[-1]
    decimal_places = len(seconds.partition(".")[2])
    return (1000, 100, 10, 1)[min(decimal_places, 3)]


def _parse_text(value: str | None) -> tuple[str | None, FieldStatus]:
    if value is None or not value.strip():
        return None, _STATUS_UNAVAILABLE
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not normalized:
        return None, _STATUS_REJECTED
    return normalized, _STATUS_PRESENT


def _requested_splits(split: SplitConfig) -> tuple[SplitName, ...]:
    """Return the splits a selection actually asked for, ignoring zero shares."""

    fractions: dict[SplitName, float] = {
        "train": 1.0 - split.validation_fraction - split.test_fraction,
        "validation": split.validation_fraction,
        "test": split.test_fraction,
    }
    return tuple(name for name in SPLIT_NAMES if fractions[name] > 0.0)


def _split_name(
    game_id: int,
    *,
    seed: str,
    validation_fraction: float,
    test_fraction: float,
) -> SplitName:
    """Assign one game to a split as a pure function of the seed and game id.

    Holding the seed fixed keeps every game's assignment stable as a corpus
    grows or its filters change, so a frozen test pool can never leak into a
    later training selection.

    ``test`` claims the lowest hash range so its membership also survives a
    later change to ``validation_fraction``. That ordering costs a one-time
    reassignment of games relative to the previous two-way split, which the
    preprocessing version bump already forces.
    """

    digest = sha256(f"{seed}\0{game_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < test_fraction:
        return "test"
    if fraction < test_fraction + validation_fraction:
        return "validation"
    return "train"


def _write_parquet(records: list[dict[str, object]], path: Path) -> None:
    try:
        write_normalized_rows(records, path)
    except DataLoadingError as error:
        raise DataPreparationError(str(error)) from error


def _flush_records(
    records: list[dict[str, object]],
    *,
    output_path: Path,
    normalized_directory: Path,
    normalized_paths: list[Path],
    output_shards: list[dict[str, object]],
    sharded: bool,
) -> None:
    if not records:
        return
    records.sort(key=_record_game_id)
    normalized_directory.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"games-{len(normalized_paths):05d}.parquet" if sharded else "games.parquet"
    )
    normalized_path = normalized_directory / file_name
    _write_parquet(records, normalized_path)
    split_counts = Counter(str(record[NormalizedColumn.SPLIT]) for record in records)
    normalized_paths.append(normalized_path)
    output_shards.append(
        {
            "path": normalized_path.relative_to(output_path).as_posix(),
            "sha256": file_sha256(normalized_path),
            "games": len(records),
            "split_counts": {
                split_name: split_counts[split_name] for split_name in SPLIT_NAMES
            },
        }
    )
    records.clear()


def _record_game_id(record: dict[str, object]) -> int:
    return row_game_id(record)


def _integer_coverage(
    values: list[int],
    statuses: Counter[str],
) -> dict[str, object]:
    return {
        "status_games": dict(sorted(statuses.items())),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }
