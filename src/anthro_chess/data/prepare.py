"""Normalize standard-chess PGN games into deterministic Parquet artifacts."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, deque
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import BrokenExecutor, Future, ProcessPoolExecutor
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from functools import partial
from io import StringIO
from itertools import islice
from pathlib import Path
from typing import Any, Literal, TextIO, cast
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
    load_snapshot_covering,
    resolve_snapshot_path,
)
from anthro_chess.data.artifacts import (
    SOURCE_USER_AGENT,
    DataLoadingError,
    file_sha256,
    open_pgn_text,
    validate_manifest_compatibility,
    write_normalized_rows,
    write_text_atomically,
)
from anthro_chess.data.census import (
    ArchiveAccountCounter,
    ArchiveAccounts,
    write_account_games,
)
from anthro_chess.data.config import ArchiveConfig, PrepareConfig, SplitConfig
from anthro_chess.data.rating_namespace import rating_namespace_from_event
from anthro_chess.data.schema import (
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    SPLIT_ALGORITHM,
    SPLIT_NAMES,
    FieldStatus,
    NormalizedColumn,
    SplitName,
    derive_game_id,
    encode_clock_remaining_deltas,
    row_game_id,
    split_name,
)
from anthro_chess.data.speed import (
    UNCLASSIFIED_SPEED,
    parse_time_control,
    speed_from_clock_ms,
    speed_from_time_control,
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
_SOURCE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
#: PGN ``Termination "Rules infraction"`` in the form ``_parse_text`` yields.
_RULES_INFRACTION_TERMINATION = "rules_infraction"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
#: What ``chess.pgn.Headers`` seeds a game with when a tag is absent. Collecting
#: headers into a plain dict is what makes the decode one pass, and it is these
#: the dict would otherwise be missing: a game carrying no ``Result`` tag and no
#: result token is unfinished rather than invalid.
_TAG_ROSTER_DEFAULTS: dict[str, str] = {
    "Event": "?",
    "Site": "?",
    "Date": "????.??.??",
    "Round": "?",
    "White": "?",
    "Black": "?",
    "Result": "*",
}
#: Games framed into one decoding job. A job has to cost far more than the
#: round trip that dispatched it, and a full pool's worth of the records they
#: return has to stay small beside the shard being filled.
_GAMES_PER_JOB = 256
#: Jobs queued beyond one per worker. ``StreamingLoaderConfig`` argues why the
#: two add rather than compete; the number is smaller than the loader's because
#: a job here is a quarter-thousand games of records rather than one batch.
_JOB_PREFETCH = 2
#: How many hex characters of an input's digest name that input's shards. A
#: shard name has to be unique across every archive one corpus is built from,
#: and deriving it from the content rather than from a running count keeps a
#: retried archive overwriting its own shards instead of another month's.
_SHARD_KEY_LENGTH = 12
#: Configuration a second archive may differ in and still belong in the same
#: corpus. ``archives`` grows when a source publishes another month;
#: ``artifact_name`` and ``output`` decide where shards go and how large they
#: are rather than what is in them; ``filters.maximum_games`` is a corpus bound,
#: and raising one keeps every game already accepted while admitting more, which
#: is the expansion ``docs/data.md`` describes; ``split.require_nonempty``
#: checks the result rather than shaping it.
#:
#: Named as exemptions rather than as the sections to compare, so that a
#: configuration gaining a field or a whole section fails closed: appending
#: refuses until someone decides the new one is safe.
_IDENTITY_EXEMPTIONS: tuple[tuple[str, ...], ...] = (
    ("archives",),
    ("artifact_name",),
    ("output",),
    ("filters", "maximum_games"),
    ("split", "require_nonempty"),
)
#: Manifest labels for whether a player's decision ended the game while they
#: held the move. ``not_applicable`` covers endings no player decided, which is
#: a different statement from a decision made on the opponent's clock.
_ATTRIBUTION_LABELS: dict[bool | None, str] = {
    True: "side_to_move",
    False: "opponent_to_move",
    None: "not_applicable",
}

#: Width of the rating buckets the axis coverage counts player-slots into.
#: Deliberately finer than any band a benchmark slices on, so a reader can
#: re-add buckets into whatever banding it uses instead of the manifest
#: pinning one benchmark's current choice into every corpus ever prepared.
_RATING_BUCKET_POINTS = 200


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


PreparationDisposition = Literal["prepared", "already_prepared", "corpus_complete"]


@dataclass(frozen=True)
class PreparationResult:
    """Paths and counts produced by one preparation run.

    A run prepares one archive and appends it to whatever corpus already exists
    beneath the output directory, so these are read at two levels:
    ``normalized_paths``, ``accepted_games`` and ``rejected_games`` describe the
    one archive, while ``split_counts`` and ``corpus_archives`` describe the
    corpus that archive now belongs to.
    """

    normalized_paths: tuple[Path, ...]
    manifest_path: Path
    accepted_games: int
    rejected_games: int
    split_counts: dict[str, int]
    corpus_archives: int
    disposition: PreparationDisposition = "prepared"

    @property
    def normalized_path(self) -> Path:
        """Return the first shard for compatibility with single-shard callers."""
        return self.normalized_paths[0]


@dataclass(frozen=True)
class _ExistingCorpus:
    """The archives and shards a corpus manifest already records."""

    inputs: tuple[dict[str, Any], ...]
    shards: tuple[dict[str, Any], ...]

    @property
    def accepted_games(self) -> int:
        return sum(int(entry["games"]["accepted"]) for entry in self.inputs)

    def entry_for(self, input_sha256: str) -> dict[str, Any] | None:
        return next(
            (entry for entry in self.inputs if entry["sha256"] == input_sha256),
            None,
        )

    def shard_paths(self, output_path: Path, input_sha256: str) -> tuple[Path, ...]:
        return tuple(
            output_path / shard["path"]
            for shard in self.shards
            if shard["input_sha256"] == input_sha256
        )


@dataclass(frozen=True)
class _OptionalInteger:
    value: int | None
    status: FieldStatus


#: One ply's clock reading: value, status, and the tick the source printed it
#: at. A plain tuple rather than a record, because building one runs once per
#: ply of every game prepared and nothing holds onto it.
_ParsedClock = tuple[int | None, FieldStatus, int | None]
_REJECTED_CLOCK: _ParsedClock = (None, _STATUS_REJECTED, None)
_UNAVAILABLE_CLOCK: _ParsedClock = (None, _STATUS_UNAVAILABLE, None)


@dataclass(frozen=True)
class _ParsedGame:
    record: dict[str, object] | None
    rejection: str | None
    termination: DerivedTermination | None = None


@dataclass(frozen=True)
class _ScreenedHeaders:
    """What the headers gave up, once they have not ruled the game out."""

    source_game_key: str
    white_rating: _OptionalInteger
    black_rating: _OptionalInteger
    termination: str | None
    termination_status: FieldStatus


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
    *,
    workers: int = 0,
    counts_path: Path | None = None,
) -> PreparationResult:
    """Append one PGN archive to the corpus beneath an output directory.

    A corpus is built one archive at a time, because a selection can name more
    archives than the machine has disk to hold at once and a run then fetches,
    prepares and deletes each in turn. An archive the manifest already records
    is left alone rather than prepared twice, which is what lets an interrupted
    pass be re-run from its beginning.

    One corpus directory takes one writer at a time: two runs appending at once
    each rewrite the manifest without the other's archive, and the loser's
    shards are then swept as orphans.

    ``workers`` decodes games on that many processes, and none of it reaches
    what is written: the artifact of a run on thirty processes is the artifact
    of a run on none, byte for byte.

    ``counts_path`` is where this pass leaves the account census its per-archive
    counts. Producing them here is what lets an archive be deleted as soon as it
    is prepared: the census orders its queue by games played, and reading that
    from the pass the archive is already getting costs a comparison per line
    rather than a second pass over hundreds of gigabytes.
    """
    source_path = Path(input_path)
    output_path = Path(output_directory)
    logger.info("Preparing normalized data from %s", source_path)
    if not source_path.is_file():
        raise DataPreparationError(f"input PGN does not exist: {source_path}")

    config = resolved_config.value
    manifest_directory = output_path / "manifests"
    manifest_path = manifest_directory / "manifest.json"
    # Ahead of digesting the input, which reads a whole archive — tens of
    # gigabytes at the pinned selection's sizes — so a corpus this selection
    # cannot append to is refused without that read.
    corpus = _read_existing_corpus(
        manifest_path,
        _selection_identity(_recorded_mapping(resolved_config.as_record()["config"])),
    )

    input_sha256 = _pinned_archive_digest(source_path, config)
    prepared_entry = corpus.entry_for(input_sha256)
    if prepared_entry is not None:
        logger.info("Archive %s is already in this corpus", source_path.name)
        return _recorded_result(corpus, manifest_path, output_path, prepared_entry)
    maximum_games = config.filters.maximum_games
    remaining_games = (
        None if maximum_games is None else maximum_games - corpus.accepted_games
    )
    if remaining_games is not None and remaining_games < 0:
        # Raising a bound admits more games and is why the bound is not part of
        # the selection identity. Lowering one below what a corpus already holds
        # cannot be honoured by adding nothing, because the corpus is already
        # past it.
        raise DataPreparationError(
            f"{manifest_path} records {corpus.accepted_games} accepted game(s), "
            f"past the configured maximum of {maximum_games}; remove the "
            "artifact directory to rebuild it, or prepare into another one"
        )
    if remaining_games == 0:
        logger.info(
            "Corpus already holds its configured maximum of %s game(s)",
            maximum_games,
        )
        return _recorded_result(corpus, manifest_path, output_path, None)
    marked_accounts = _resolve_marked_accounts(resolved_config, input_sha256)
    prepared = _prepare_archive(
        source_path,
        output_path,
        resolved_config,
        input_sha256=input_sha256,
        marked_accounts=marked_accounts,
        remaining_games=remaining_games,
        workers=workers,
        counts_path=counts_path,
    )
    return _append_to_corpus(
        corpus,
        (prepared,),
        output_path=output_path,
        manifest_path=manifest_path,
        resolved_config=resolved_config,
    )


def prepare_archives(
    input_paths: Sequence[str | Path],
    output_directory: str | Path,
    resolved_config: ResolvedConfig[PrepareConfig],
    *,
    workers: int = 0,
    concurrency: int = 1,
    counts_paths: Sequence[Path | None] | None = None,
) -> PreparationResult:
    """Append several archives to one corpus, decoding them at the same time.

    One archive cannot use a whole machine. The reader that frames its games
    runs in a single process and is what a pool of more than about a dozen
    decoders waits on, so the way to spend the rest of the machine is another
    archive rather than a wider pool. `0053` measures both.

    The manifest is written once, for all of them, after every archive is done.
    That is the part a run cannot share: the corpus totals, the split check and
    the sweep of shards no manifest claims each read the whole corpus, and a
    sweep running while another archive was still writing would delete that
    archive's shards.

    An archive already recorded is skipped rather than prepared twice, exactly
    as one at a time does, so an interrupted pass is still re-run from its
    beginning at no cost.
    """

    output_path = Path(output_directory)
    sources = [Path(path) for path in input_paths]
    counts = list(counts_paths) if counts_paths is not None else [None] * len(sources)
    config = resolved_config.value
    if config.filters.maximum_games is not None:
        # What an archive may admit is the bound less what every archive before
        # it contributed, so the archives are ordered by construction and a
        # second one cannot start before the first has been counted.
        return _prepare_in_turn(sources, output_path, resolved_config, workers, counts)

    manifest_path = output_path / "manifests" / "manifest.json"
    corpus = _read_existing_corpus(
        manifest_path,
        _selection_identity(_recorded_mapping(resolved_config.as_record()["config"])),
    )
    pending: list[tuple[Path, str, Path | None]] = []
    for source_path, counts_path in zip(sources, counts, strict=True):
        if not source_path.is_file():
            raise DataPreparationError(f"input PGN does not exist: {source_path}")
        input_sha256 = _pinned_archive_digest(source_path, config)
        if corpus.entry_for(input_sha256) is not None:
            logger.info("Archive %s is already in this corpus", source_path.name)
            continue
        pending.append((source_path, input_sha256, counts_path))
    if not pending:
        return PreparationResult(
            normalized_paths=(),
            manifest_path=manifest_path,
            accepted_games=0,
            rejected_games=0,
            split_counts=_corpus_split_counts(corpus.inputs),
            corpus_archives=len(corpus.inputs),
            disposition="already_prepared",
        )

    jobs = [
        (
            source_path,
            output_path,
            resolved_config,
            input_sha256,
            workers,
            counts_path,
        )
        for source_path, input_sha256, counts_path in pending
    ]
    if concurrency > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(concurrency, len(jobs))) as pool:
            # Ordered by the inputs rather than by which archive finished, so
            # the manifest a concurrent run writes is the one a sequential run
            # over the same inputs writes.
            prepared = list(pool.map(_prepare_archive_from_job, jobs))
    else:
        prepared = [_prepare_archive_from_job(job) for job in jobs]
    return _append_to_corpus(
        corpus,
        prepared,
        output_path=output_path,
        manifest_path=manifest_path,
        resolved_config=resolved_config,
    )


def _prepare_in_turn(
    sources: Sequence[Path],
    output_path: Path,
    resolved_config: ResolvedConfig[PrepareConfig],
    workers: int,
    counts: Sequence[Path | None],
) -> PreparationResult:
    """Append archives one at a time, each seeing what the last contributed."""

    result: PreparationResult | None = None
    accepted = 0
    rejected = 0
    paths: list[Path] = []
    for source_path, counts_path in zip(sources, counts, strict=True):
        result = prepare_pgn(
            source_path,
            output_path,
            resolved_config,
            workers=workers,
            counts_path=counts_path,
        )
        if result.disposition == "prepared":
            accepted += result.accepted_games
            rejected += result.rejected_games
            paths.extend(result.normalized_paths)
    if result is None:
        raise DataPreparationError("no input archives were given to prepare")
    return PreparationResult(
        normalized_paths=tuple(paths),
        manifest_path=result.manifest_path,
        accepted_games=accepted,
        rejected_games=rejected,
        split_counts=result.split_counts,
        corpus_archives=result.corpus_archives,
    )


def _pinned_archive_digest(source_path: Path, config: PrepareConfig) -> str:
    digest = file_sha256(source_path)
    if config.archives and digest not in {
        archive.sha256 for archive in config.archives
    }:
        raise DataPreparationError(
            f"input archive checksum {digest} matches none of the "
            f"{len(config.archives)} archive(s) this selection pins"
        )
    return digest


def _prepare_archive_from_job(
    job: tuple[
        Path,
        Path,
        ResolvedConfig[PrepareConfig],
        str,
        int,
        Path | None,
    ],
) -> _PreparedArchive:
    """Prepare one archive from a job another process can be handed.

    The marked-account snapshot is loaded here rather than sent, because it
    dwarfs everything else in the job and every archive needs its own copy
    regardless.
    """

    (
        source_path,
        output_path,
        resolved_config,
        input_sha256,
        workers,
        counts_path,
    ) = job
    return _prepare_archive(
        source_path,
        output_path,
        resolved_config,
        input_sha256=input_sha256,
        marked_accounts=_resolve_marked_accounts(resolved_config, input_sha256),
        remaining_games=None,
        workers=workers,
        counts_path=counts_path,
    )


@dataclass(frozen=True)
class _PreparedArchive:
    """One archive's shards and the manifest entry describing them.

    Deliberately free of corpus state, so that preparing it says nothing about
    the archives beside it and can therefore happen in another process while
    those are still being read.
    """

    record: dict[str, Any]
    shards: list[dict[str, object]]
    normalized_paths: list[Path]
    accepted_games: int
    rejected_games: int


def _prepare_archive(
    source_path: Path,
    output_path: Path,
    resolved_config: ResolvedConfig[PrepareConfig],
    *,
    input_sha256: str,
    marked_accounts: MarkedAccounts | None,
    remaining_games: int | None,
    workers: int,
    counts_path: Path | None,
) -> _PreparedArchive:
    """Decode one archive and write its shards, touching no manifest."""

    config = resolved_config.value
    normalized_directory = output_path / "normalized"
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
    rating_namespace_counts: Counter[str] = Counter()
    source_date_status_counts: Counter[str] = Counter()
    speed_split_counts: Counter[tuple[str, str]] = Counter()
    clock_split_counts: Counter[tuple[str, str]] = Counter()
    rating_slot_split_counts: Counter[tuple[str, str]] = Counter()
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
    counter = ArchiveAccountCounter()

    try:
        with (
            open_pgn_text(source_path) as pgn_file,
            closing(
                _decoded_games(pgn_file, config, marked_accounts, workers, counter)
            ) as decoded,
        ):
            for parsed in decoded:
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
                    split_name = str(parsed.record[NormalizedColumn.SPLIT])
                    split_counts[split_name] += 1
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
                    rating_namespace = parsed.record[
                        NormalizedColumn.SOURCE_RATING_NAMESPACE
                    ]
                    rating_namespace_counts[
                        _STATUS_UNAVAILABLE
                        if rating_namespace is None
                        else str(rating_namespace)
                    ] += 1
                    source_date_status_counts[
                        str(parsed.record[NormalizedColumn.SOURCE_DATE_STATUS])
                    ] += 1
                    # Banded off the normalized clock, which is how a pool and a
                    # training selection read this axis.
                    speed = speed_from_clock_ms(
                        time_initial if isinstance(time_initial, int) else None,
                        time_increment if isinstance(time_increment, int) else None,
                    )
                    speed_split_counts[
                        (
                            UNCLASSIFIED_SPEED if speed is None else str(speed),
                            split_name,
                        )
                    ] += 1
                    clock_split_counts[
                        (
                            "present" if isinstance(clock_precision, int) else "absent",
                            split_name,
                        )
                    ] += 1
                    # Bucketed on the normalized rating rather than the source
                    # one, because that is the column a pool admits on and a
                    # selection bands by. A source left on an unconverted scale
                    # would otherwise report a fully covered rating axis for a
                    # corpus no rating-banded reading can draw a game from.
                    for normalized_column in (
                        NormalizedColumn.WHITE_NORMALIZED_RATING,
                        NormalizedColumn.BLACK_NORMALIZED_RATING,
                    ):
                        rating_slot_split_counts[
                            (
                                _rating_bucket(parsed.record[normalized_column]),
                                split_name,
                            )
                        ] += 1

                    games_per_shard = config.output.games_per_shard
                    if games_per_shard is not None and len(records) >= games_per_shard:
                        _flush_records(
                            records,
                            output_path=output_path,
                            normalized_directory=normalized_directory,
                            normalized_paths=normalized_paths,
                            output_shards=output_shards,
                            input_sha256=input_sha256,
                        )

                    if (
                        remaining_games is not None
                        and accepted_games >= remaining_games
                    ):
                        stopped_at_limit = True
                        break
    except (OSError, UnicodeError, DataLoadingError) as error:
        raise DataPreparationError(
            f"cannot read input PGN {source_path}: {error}"
        ) from error
    except BrokenExecutor as error:
        # A decoder killed from outside — the out-of-memory killer is the one
        # to expect on a run this long — takes the pool down with it.
        raise DataPreparationError(
            f"a decoding worker died while preparing {source_path}: {error}"
        ) from error

    if accepted_games == 0:
        logger.warning(
            "Archive %s contributed no games and is recorded as an empty append",
            source_path.name,
        )
    if counts_path is not None and not stopped_at_limit:
        # Only a pass that reached the end of the archive counted all of it, and
        # a counts file that spoke for part of one would understate the census's
        # population while reading as though it covered the archive. A pass the
        # corpus bound cut short leaves the counting to the census, which needs
        # the archive back to do it.
        write_account_games(
            counts_path,
            ArchiveAccounts(
                archive_sha256=input_sha256,
                games_by_account=counter.account_games(),
            ),
        )

    _flush_records(
        records,
        output_path=output_path,
        normalized_directory=normalized_directory,
        normalized_paths=normalized_paths,
        output_shards=output_shards,
        input_sha256=input_sha256,
    )

    archive_record: dict[str, Any] = {
        "file_name": source_path.name,
        "sha256": input_sha256,
        "limit_reached": stopped_at_limit,
        "marked_accounts": (
            None if marked_accounts is None else marked_accounts.as_record()
        ),
        "split_counts": {
            split_name: split_counts[split_name] for split_name in SPLIT_NAMES
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
                # Keeps an archive whose labels named no pool visible here
                # rather than in whatever first reads the column.
                "namespace_games": dict(sorted(rating_namespace_counts.items())),
            },
            # An archive whose games carry no readable date prepares
            # successfully and reads as an era nothing can name, and the only
            # repair is parsing it again.
            "source_date": {
                "status_games": dict(sorted(source_date_status_counts.items())),
            },
            # Split-wise so that what a held-out reading can resolve on an axis
            # is readable before a pool is cut from it, rather than after the
            # generation that fixes it permanently.
            "axes": {
                "speed_games": _counts_by_split(speed_split_counts),
                "clock_games": _counts_by_split(clock_split_counts),
                "rating_slots": _counts_by_split(rating_slot_split_counts),
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
                        config.termination.abandonment_clock_share
                    ),
                    "clock_share_judged_games": abandonment_judged_games,
                },
            },
        },
    }

    return _PreparedArchive(
        archive_record,
        output_shards,
        normalized_paths,
        accepted_games,
        sum(rejections.values()),
    )


def _append_to_corpus(
    corpus: _ExistingCorpus,
    prepared: Sequence[_PreparedArchive],
    *,
    output_path: Path,
    manifest_path: Path,
    resolved_config: ResolvedConfig[PrepareConfig],
) -> PreparationResult:
    """Record every prepared archive in one rewrite of the corpus manifest.

    Taking them together rather than one at a time is what lets archives be
    prepared at once: the totals, the split check and the sweep of shards no
    manifest claims all read the whole corpus, and a sweep run while another
    archive was still writing would delete that archive's shards.
    """

    config = resolved_config.value
    normalized_directory = output_path / "normalized"
    maximum_games = config.filters.maximum_games
    inputs = (*corpus.inputs, *(archive.record for archive in prepared))
    corpus_shards = [
        *corpus.shards,
        *(shard for archive in prepared for shard in archive.shards),
    ]
    normalized_paths = [
        path for archive in prepared for path in archive.normalized_paths
    ]
    accepted_games = sum(archive.accepted_games for archive in prepared)
    rejected_games = sum(archive.rejected_games for archive in prepared)
    if accepted_games == 0 and not corpus.inputs:
        # With no earlier archive there is no corpus for an empty append to
        # extend, so archives that accept nothing between them are a
        # misconfigured selection rather than months with nothing eligible in.
        reasons: Counter[str] = Counter()
        for archive in prepared:
            reasons.update(archive.record["games"]["rejection_reasons"])
        detail = ", ".join(
            f"{reason}={count}" for reason, count in sorted(reasons.items())
        )
        raise DataPreparationError(
            "no games passed preparation filters" + (f" ({detail})" if detail else "")
        )
    corpus_games = _corpus_games(inputs)
    observed_split_counts = _corpus_split_counts(inputs)
    if config.split.require_nonempty:
        empty_splits = tuple(
            split_name
            for split_name in _requested_splits(config.split)
            if observed_split_counts[split_name] == 0
        )
        if empty_splits:
            raise DataPreparationError(
                "prepared corpus did not produce nonempty "
                f"{', '.join(empty_splits)} split(s)"
            )

    recorded_paths = {output_path / shard["path"] for shard in corpus_shards}
    present_paths = set(normalized_directory.glob("games*.parquet"))
    # A manifest asserting shards that are gone is otherwise not detected until
    # a training or freeze run reads the corpus, by which point more archives
    # sit on top of the claim.
    missing_paths = sorted(recorded_paths - present_paths)
    if missing_paths:
        raise DataPreparationError(
            f"{manifest_path} records {len(missing_paths)} shard(s) that are no "
            f"longer present, the first being {missing_paths[0]}; remove the "
            "artifact directory to rebuild it"
        )
    for stale_path in present_paths - recorded_paths:
        stale_path.unlink()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "source": config.source.model_dump(mode="json"),
        "inputs": list(inputs),
        "output": {
            "format": "parquet",
            "compression": "zstd",
            "shards": corpus_shards,
        },
        "action_vocabulary": action_vocabulary_identity(),
        "selection": {
            "algorithm": "source-order-first-accepted-v1",
            "maximum_games": maximum_games,
            "limit_reached": maximum_games is not None
            and corpus_games["accepted"] >= maximum_games,
        },
        "split": {
            "algorithm": SPLIT_ALGORITHM,
            "seed": config.split.seed,
            "validation_fraction": config.split.validation_fraction,
            "test_fraction": config.split.test_fraction,
            "counts": observed_split_counts,
        },
        "games": corpus_games,
        "coverage": _corpus_coverage(inputs),
        "resolved_config": resolved_config.as_record(),
    }
    write_text_atomically(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    logger.info(
        "Prepared %s game(s), rejected %s, and wrote %s shard(s); the corpus "
        "now spans %s archive(s)",
        accepted_games,
        rejected_games,
        len(normalized_paths),
        len(inputs),
    )
    return PreparationResult(
        normalized_paths=tuple(normalized_paths),
        manifest_path=manifest_path,
        accepted_games=accepted_games,
        rejected_games=rejected_games,
        split_counts=observed_split_counts,
        corpus_archives=len(inputs),
    )


def _selection_identity(config: Mapping[str, Any]) -> dict[str, object]:
    """Reduce a recorded selection to what decides what a game becomes.

    The data contract — schema, preprocessing and vocabulary versions — is not
    here because ``validate_manifest_compatibility`` owns it for every consumer.
    """

    identity: dict[str, Any] = deepcopy(dict(config))
    for *sections, field in _IDENTITY_EXEMPTIONS:
        holder: Any = identity
        for section in sections:
            holder = holder.get(section) if isinstance(holder, dict) else None
        if isinstance(holder, dict):
            holder.pop(field, None)
    return identity


def _recorded_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_existing_corpus(
    manifest_path: Path,
    identity: Mapping[str, object],
) -> _ExistingCorpus:
    """Read the corpus a run appends to, refusing one built another way."""

    if not manifest_path.is_file():
        return _ExistingCorpus(inputs=(), shards=())
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest is not a JSON object")
        validate_manifest_compatibility(manifest, manifest_path)
    except (OSError, ValueError, DataLoadingError) as error:
        raise DataPreparationError(
            f"cannot read corpus manifest {manifest_path}: {error}"
        ) from error
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise DataPreparationError(
            f"{manifest_path} records no per-archive inputs, so it predates a "
            "corpus that can span them; remove the artifact directory to "
            "rebuild it, or prepare into another one"
        )
    recorded_config = _recorded_mapping(manifest.get("resolved_config")).get("config")
    recorded = _selection_identity(_recorded_mapping(recorded_config))
    if recorded != identity:
        # Over the union, so a section the recorded selection has and this one
        # does not is named rather than reported as an unexplained difference.
        differing = ", ".join(
            sorted(
                key
                for key in recorded.keys() | identity.keys()
                if recorded.get(key) != identity.get(key)
            )
        )
        raise DataPreparationError(
            f"{manifest_path} was prepared under a different {differing}, and "
            "one corpus is built from one selection; remove the artifact "
            "directory to rebuild it, or prepare into another one"
        )
    shards = _recorded_mapping(manifest.get("output")).get("shards")
    if not isinstance(shards, list) or not shards:
        raise DataPreparationError(f"{manifest_path} records no output shards")
    if not all(_records_an_archive(entry) for entry in inputs) or not all(
        _records_a_shard(shard) for shard in shards
    ):
        raise DataPreparationError(
            f"{manifest_path} has an input record without a digest, or a shard "
            "record without a path and the digest of the archive it came from"
        )
    corpus = _ExistingCorpus(inputs=tuple(inputs), shards=tuple(shards))
    try:
        # Rolling the recorded archives up walks every remaining field this run
        # will index, so a manifest written by other code reports rather than
        # dying on a bare lookup halfway through the append.
        _corpus_games(corpus.inputs)
        _corpus_coverage(corpus.inputs)
    except (KeyError, TypeError) as error:
        raise DataPreparationError(
            f"{manifest_path} has an input record missing {error}"
        ) from error
    return corpus


def _records_an_archive(entry: object) -> bool:
    return isinstance(entry, dict) and isinstance(entry.get("sha256"), str)


def _records_a_shard(shard: object) -> bool:
    return (
        isinstance(shard, dict)
        and isinstance(shard.get("path"), str)
        and isinstance(shard.get("input_sha256"), str)
    )


def _recorded_result(
    corpus: _ExistingCorpus,
    manifest_path: Path,
    output_path: Path,
    entry: Mapping[str, Any] | None,
) -> PreparationResult:
    """Report a run that added nothing, from what the manifest already records."""

    if entry is None:
        return PreparationResult(
            normalized_paths=(),
            manifest_path=manifest_path,
            accepted_games=0,
            rejected_games=0,
            split_counts=_corpus_split_counts(corpus.inputs),
            corpus_archives=len(corpus.inputs),
            disposition="corpus_complete",
        )
    return PreparationResult(
        normalized_paths=corpus.shard_paths(output_path, entry["sha256"]),
        manifest_path=manifest_path,
        accepted_games=entry["games"]["accepted"],
        rejected_games=entry["games"]["rejected"],
        split_counts=_corpus_split_counts(corpus.inputs),
        corpus_archives=len(corpus.inputs),
        disposition="already_prepared",
    )


def _decode_stream(
    handle: TextIO,
    config: PrepareConfig,
    marked_accounts: MarkedAccounts | None,
) -> Iterator[_ParsedGame]:
    """Yield every game a handle holds, decoded but not yet accepted."""

    builder = partial(_RecordBuilder, config, marked_accounts)
    while True:
        parsed: _ParsedGame | None = chess.pgn.read_game(handle, Visitor=builder)
        if parsed is None:
            return
        yield parsed


class _ScanningReader:
    """Hand ``read_game`` its lines, counting accounts as they go past.

    ``read_game`` touches the handle only through ``readline``, which is what
    lets this stand in for the ``TextIO`` it is cast to.

    Counting here is what lets an archive be reclaimed the moment it has been
    prepared: the account census orders its queue by games played, and this
    pass already holds every line it would otherwise reopen the archive to
    read.

    ``lines`` is kept only for a caller that frames games out of them, since a
    pass that never clears it would hold the whole archive in memory.
    """

    def __init__(
        self,
        handle: TextIO,
        counter: ArchiveAccountCounter,
        *,
        capture: bool,
    ) -> None:
        self._handle = handle
        self._counter = counter
        self._capture = capture
        self.lines: list[str] = []

    def readline(self) -> str:
        line = self._handle.readline()
        if line:
            self._counter.observe(line)
            if self._capture:
                self.lines.append(line)
        return line


def _framed_games(pgn_file: TextIO, counter: ArchiveAccountCounter) -> Iterator[str]:
    """Yield each game's raw text, framed but not parsed.

    Deciding where one game ends is the part that cannot be divided, so it runs
    on ``python-chess``'s own game-skipping scanner rather than on a second
    account of PGN framing. That scanner reads 16,800 games/s against the 576
    a full decode manages, which is what keeps one reader ahead of a machine
    full of workers.
    """

    reader = _ScanningReader(pgn_file, counter, capture=True)
    while True:
        reader.lines = []
        framed = chess.pgn.read_game(
            cast(TextIO, reader),
            Visitor=chess.pgn.SkipVisitor,
        )
        if framed is None:
            return
        yield "".join(reader.lines)


def _read_game_batches(
    pgn_file: TextIO, counter: ArchiveAccountCounter
) -> Iterator[str]:
    games = _framed_games(pgn_file, counter)
    while batch := list(islice(games, _GAMES_PER_JOB)):
        yield "".join(batch)


def _decode_batch(
    text: str,
    config: PrepareConfig,
    marked_accounts: MarkedAccounts | None,
) -> list[_ParsedGame]:
    return list(_decode_stream(StringIO(text), config, marked_accounts))


#: What a pooled worker decodes against, sent once when it starts rather than
#: with every job: a marked-account snapshot dwarfs the text a job carries.
_WORKER_CONTEXT: tuple[PrepareConfig, MarkedAccounts | None] | None = None


def _start_worker(
    config: PrepareConfig,
    marked_accounts: MarkedAccounts | None,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (config, marked_accounts)


def _decode_batch_in_worker(text: str) -> list[_ParsedGame]:
    assert _WORKER_CONTEXT is not None
    return _decode_batch(text, *_WORKER_CONTEXT)


def _decoded_games(
    pgn_file: TextIO,
    config: PrepareConfig,
    marked_accounts: MarkedAccounts | None,
    workers: int,
    counter: ArchiveAccountCounter,
) -> Generator[_ParsedGame]:
    """Yield every game's decode in source order, across ``workers`` processes.

    Order is what keeps the worker count out of the artifact. Acceptance,
    deduplication, the game bound and shard boundaries all read this one
    sequence, so a run on one process and a run on thirty write the same bytes.
    """

    if workers < 1:
        reader = cast(TextIO, _ScanningReader(pgn_file, counter, capture=False))
        yield from _decode_stream(reader, config, marked_accounts)
        return

    batches = _read_game_batches(pgn_file, counter)
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_start_worker,
        initargs=(config, marked_accounts),
    )
    inflight: deque[Future[list[_ParsedGame]]] = deque()
    try:
        while True:
            queued = workers + _JOB_PREFETCH - len(inflight)
            for text in islice(batches, queued):
                inflight.append(pool.submit(_decode_batch_in_worker, text))
            if not inflight:
                return
            yield from inflight.popleft().result()
    finally:
        # A caller that stopped at the game bound leaves jobs queued for games
        # past it, and the archive stays open until every worker is done.
        pool.shutdown(cancel_futures=True)


def _resolve_marked_accounts(
    resolved_config: ResolvedConfig[PrepareConfig],
    input_sha256: str,
) -> MarkedAccounts | None:
    """Load the configured marked-account snapshot and check it applies here."""

    configured = resolved_config.value.filters.marked_accounts
    if configured is None:
        return None
    try:
        return load_snapshot_covering(
            resolve_snapshot_path(configured, resolved_config.provenance.source),
            (input_sha256,),
        )
    except MarkedAccountError as error:
        raise DataPreparationError(str(error)) from error


class _RecordBuilder(chess.pgn.BaseVisitor[_ParsedGame]):
    """Build one normalized record during the parser's own walk of a game.

    ``read_game``'s default visitor allocates a node per ply to assemble a game
    tree that preparation then discards, replaying the mainline on a fresh
    board to read the same moves back. Collecting the record here costs one
    pass over a game rather than two.

    The comment rule is ``GameBuilder``'s and has to stay so. A comment belongs
    to the ply before it only while ``in_variation`` holds, which any move sets
    and only ``begin_variation`` clears — so a comment following a variation
    that contained no move belongs to no ply at all, and dropping it is what
    agrees with the tree this replaces.

    A game the headers alone reject is skipped at ``end_headers``, so its
    movetext is scanned for the end of the game rather than parsed. Nothing
    past that point can then name a different reason: see
    ``docs/decisions/0050-a-header-rejection-outranks-a-parse-error.md``.
    """

    def __init__(
        self,
        config: PrepareConfig,
        marked_accounts: MarkedAccounts | None,
    ) -> None:
        self._config = config
        self._marked_accounts = marked_accounts

    def begin_game(self) -> None:
        self._headers = dict(_TAG_ROSTER_DEFAULTS)
        self._moves: list[chess.Move] = []
        self._comments: list[str] = []
        self._board: chess.Board | None = None
        self._depth = 0
        self._in_variation = False
        self._failed = False

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        self._headers[tagname] = tagvalue

    def end_headers(self) -> chess.pgn.SkipType | None:
        self._screened: _ScreenedHeaders | str = _screen_headers(
            self._headers,
            self._config,
            self._marked_accounts,
        )
        return chess.pgn.SKIP if isinstance(self._screened, str) else None

    def visit_board(self, board: chess.Board) -> None:
        # Mainline moves are pushed onto this same board, so the first one is
        # also the final position once the walk is done.
        if self._board is None:
            self._board = board

    def begin_variation(self) -> None:
        self._depth += 1
        self._in_variation = False

    def end_variation(self) -> None:
        # Clamped because ``read_game`` also closes an error-skip through this
        # callback, with no ``begin_variation`` to match it.
        self._depth = max(self._depth - 1, 0)

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        self._in_variation = True
        if self._depth:
            return
        self._moves.append(move)
        self._comments.append("")

    def visit_comment(self, comment: str) -> None:
        if self._depth or not self._in_variation or not self._comments:
            return
        existing = self._comments[-1]
        if not existing:
            self._comments[-1] = comment
        elif comment:
            self._comments[-1] = f"{existing} {comment}"

    def visit_result(self, result: str) -> None:
        if self._headers.get("Result", "*") == "*":
            self._headers["Result"] = result

    def handle_error(self, error: Exception) -> None:
        self._failed = True
        logger.error("%s while parsing %s", error, self._headers.get("Site", "?"))

    def result(self) -> _ParsedGame:
        if isinstance(self._screened, str):
            return _ParsedGame(None, self._screened)
        if self._failed:
            return _ParsedGame(None, "pgn_parse_error")
        return _parse_game(
            self._headers,
            self._moves,
            self._comments,
            self._board,
            self._screened,
            config=self._config,
        )


def _screen_headers(
    headers: Mapping[str, str],
    config: PrepareConfig,
    marked_accounts: MarkedAccounts | None,
) -> _ScreenedHeaders | str:
    """Read the headers, returning why they rule the game out if they do."""

    variant = headers.get("Variant", "Standard")
    if variant.casefold() not in {"standard", "from position"}:
        return "unsupported_variant"
    if headers.get("SetUp") == "1" or headers.get("FEN"):
        return "nonstandard_initial_position"
    termination, termination_status = _parse_text(headers.get("Termination"))
    if termination == _RULES_INFRACTION_TERMINATION:
        # Ended by the platform on a client-side report rather than played to
        # a finish, so the record is a fragment of a game rather than a
        # completed human one. Rejected for that rather than as a cheating
        # filter, which 0041 measures this label as far too narrow to serve.
        return "rules_infraction"
    if config.filters.exclude_bots and _has_bot_player(headers):
        return "bot_game"
    if (
        config.filters.speed is not None
        and speed_from_time_control(headers.get("TimeControl")) != config.filters.speed
    ):
        return "speed_mismatch"
    if (
        config.filters.require_rated
        and "rated" not in headers.get("Event", "").casefold()
    ):
        return "unrated_game"
    if marked_accounts is not None and _has_marked_player(headers, marked_accounts):
        # Every game a marked account played is rejected rather than the moves
        # that were assisted, because no method separates the two per game and
        # a marked player's honest games are not what this corpus is for.
        #
        # Ordered after the speed, rated, and bot filters so the manifest's
        # count is a share of the corpus being built rather than of the whole
        # archive, which is the number 0041 is checked against.
        return "marked_account"

    source_game_key = _source_game_key(headers.get("Site"))
    if source_game_key is None:
        return "missing_source_game_key"

    white_rating = _parse_nonnegative_integer(headers.get("WhiteElo"))
    black_rating = _parse_nonnegative_integer(headers.get("BlackElo"))
    if config.filters.require_ratings and (
        white_rating.status != _STATUS_PRESENT or black_rating.status != _STATUS_PRESENT
    ):
        return "missing_or_invalid_rating"
    return _ScreenedHeaders(
        source_game_key,
        white_rating,
        black_rating,
        termination,
        termination_status,
    )


def _parse_game(
    headers: Mapping[str, str],
    moves: Sequence[chess.Move],
    comments: Sequence[str],
    final_board: chess.Board | None,
    screened: _ScreenedHeaders,
    *,
    config: PrepareConfig,
) -> _ParsedGame:
    source_game_key = screened.source_game_key
    white_rating = screened.white_rating
    black_rating = screened.black_rating
    termination = screened.termination
    termination_status = screened.termination_status
    time_initial, time_increment = _parse_time_control(headers.get("TimeControl"))
    if not all(moves):
        # ``parse_san`` promises a move that is legal or null and records an
        # error for anything else, so the null is the whole of what a legality
        # test over the replayed mainline used to catch.
        return _ParsedGame(None, "illegal_move")
    actions: list[int] = []
    clock_values: list[int | None] = []
    clock_statuses: list[FieldStatus] = []
    clock_precision_ms: int | None = None
    for move, comment in zip(moves, comments, strict=True):
        actions.append(encode_move(move))
        clock_value, clock_status, precision_ms = _parse_clock(comment)
        clock_values.append(clock_value)
        clock_statuses.append(clock_status)
        if precision_ms is not None:
            # Precision is inferred per ply from how the source printed the
            # clock, so an exporter that strips a trailing zero infers a coarser
            # tick for that ply alone. The finest tick describes every reading,
            # since a coarser one is representable in it.
            clock_precision_ms = (
                precision_ms
                if clock_precision_ms is None
                else min(clock_precision_ms, precision_ms)
            )

    if len(actions) < config.filters.minimum_plies:
        return _ParsedGame(None, "too_short")

    result = headers.get("Result")
    if result not in {"1-0", "0-1", "1/2-1/2", "*"}:
        return _ParsedGame(None, "invalid_result")

    if final_board is None:
        return _ParsedGame(None, "pgn_parse_error")
    derived_termination = derive_termination(
        result=result,
        source_termination=termination,
        final_board=final_board,
        clock_remaining_ms=clock_values,
        time_initial_ms=time_initial.value,
        abandonment_clock_share=config.termination.abandonment_clock_share,
    )
    ply_count = len(actions)
    terminal_action_id, terminal_action_status = terminal_action_for(
        derived_termination,
        final_board,
    )
    if terminal_action_id is not None:
        # The per-ply columns stay aligned with the action sequence, and no
        # source reports a clock for an ending, so the terminal action's
        # observation is explicitly unavailable rather than invented.
        actions.append(terminal_action_id)
        clock_values.append(None)
        clock_statuses.append(_STATUS_UNAVAILABLE)
    game_id = derive_game_id(config.source.id, source_game_key)
    split = split_name(
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
    source_date, source_date_status = _parse_utc_date(headers.get("UTCDate"))
    return _ParsedGame(
        {
            NormalizedColumn.SCHEMA_VERSION: SCHEMA_VERSION,
            NormalizedColumn.SOURCE_ID: config.source.id,
            NormalizedColumn.SOURCE_GAME_KEY: source_game_key,
            NormalizedColumn.SOURCE_DATE: source_date,
            NormalizedColumn.SOURCE_DATE_STATUS: source_date_status,
            NormalizedColumn.WHITE_PLAYER_DIGEST: _player_digest(headers.get("White")),
            NormalizedColumn.BLACK_PLAYER_DIGEST: _player_digest(headers.get("Black")),
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
            NormalizedColumn.SOURCE_RATING_NAMESPACE: rating_namespace_from_event(
                headers.get("Event"),
                prefix=config.source.rating_namespace_prefix,
            ),
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


def _has_marked_player(headers: Mapping[str, str], marked: MarkedAccounts) -> bool:
    return any(
        marked.contains(name)
        for name in (headers.get("White"), headers.get("Black"))
        if name
    )


def _player_digest(name: str | None) -> int | None:
    if name is None or not name.strip() or name.strip() == "?":
        return None
    return account_row_digest(name)


def _has_bot_player(headers: Mapping[str, str]) -> bool:
    return any(
        headers.get(header, "").casefold() == "bot"
        for header in ("WhiteTitle", "BlackTitle")
    )


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
    parsed = parse_time_control(value)
    if parsed is None:
        rejected = _OptionalInteger(None, _STATUS_REJECTED)
        return rejected, rejected
    initial_seconds, increment_seconds = parsed
    return (
        _OptionalInteger(initial_seconds * 1000, _STATUS_PRESENT),
        _OptionalInteger(increment_seconds * 1000, _STATUS_PRESENT),
    )


def _parse_clock(comment: str) -> _ParsedClock:
    clock_match = _CLOCK_RE.search(comment)
    if clock_match is not None:
        clock_text = clock_match.group(1)
        value = _clock_text_to_milliseconds(clock_text)
        return (
            (value, _STATUS_PRESENT, _clock_precision_ms(clock_text))
            if value is not None
            else _REJECTED_CLOCK
        )
    centiseconds_match = _CLOCK_CENTISECONDS_RE.search(comment)
    if centiseconds_match is not None:
        try:
            centiseconds = int(centiseconds_match.group(1))
        except ValueError:
            return _REJECTED_CLOCK
        if centiseconds < 0:
            return _REJECTED_CLOCK
        return (centiseconds * 10, _STATUS_PRESENT, 10)
    if _CLOCK_SHAPED_RE.search(comment):
        return _REJECTED_CLOCK
    return _UNAVAILABLE_CLOCK


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


def _parse_utc_date(value: str | None) -> tuple[date | None, FieldStatus]:
    if value is None or not value.strip():
        return None, _STATUS_UNAVAILABLE
    text = value.strip()
    if "?" in text and set(text) <= {"?", "."}:
        # PGN spells a date it does not know with question marks, so the
        # placeholder is the source reporting an absence rather than a value
        # that failed to parse. The question mark is required: a header of bare
        # dots is malformed rather than deliberately unknown, and falls through
        # to be rejected. A partly known date is rejected there too, since no
        # day is a day this column can carry.
        return None, _STATUS_UNAVAILABLE
    try:
        return datetime.strptime(text, "%Y.%m.%d").date(), _STATUS_PRESENT
    except ValueError:
        return None, _STATUS_REJECTED


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
    output_shards: list[dict[str, Any]],
    input_sha256: str,
) -> None:
    if not records:
        return
    records.sort(key=_record_game_id)
    normalized_directory.mkdir(parents=True, exist_ok=True)
    shard_key = input_sha256[:_SHARD_KEY_LENGTH]
    normalized_path = (
        normalized_directory / f"games-{shard_key}-{len(normalized_paths):05d}.parquet"
    )
    _write_parquet(records, normalized_path)
    split_counts = Counter(str(record[NormalizedColumn.SPLIT]) for record in records)
    normalized_paths.append(normalized_path)
    output_shards.append(
        {
            "path": normalized_path.relative_to(output_path).as_posix(),
            "sha256": file_sha256(normalized_path),
            # Recorded as well as named, so a shard's provenance can be checked
            # without parsing its file name.
            "input_sha256": input_sha256,
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


def _corpus_split_counts(inputs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = _summed_counts(entry["split_counts"] for entry in inputs)
    return {split_name: counts.get(split_name, 0) for split_name in SPLIT_NAMES}


def _corpus_games(inputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Roll every archive's counts up into one corpus-wide statement.

    Totals are derived from the per-archive records rather than carried forward
    from the previous run's totals, so a manifest cannot come to disagree with
    the parts it is made of.
    """

    blocks = [entry["games"] for entry in inputs]
    plies = [block["plies"] for block in blocks]
    return {
        "accepted": sum(block["accepted"] for block in blocks),
        "rejected": sum(block["rejected"] for block in blocks),
        "scanned": sum(block["scanned"] for block in blocks),
        "rejection_reasons": _summed_counts(
            block["rejection_reasons"] for block in blocks
        ),
        "plies": {
            "total": sum(block["total"] for block in plies),
            "minimum_per_game": _extreme(plies, "minimum_per_game", min),
            "maximum_per_game": _extreme(plies, "maximum_per_game", max),
        },
    }


def _corpus_coverage(inputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Roll every archive's coverage up into one corpus-wide statement.

    The abandonment threshold comes from the records like every other number
    rather than from the configuration, and every archive agrees on it because
    ``termination`` is part of the selection identity an append is refused for.
    """

    blocks = [entry["coverage"] for entry in inputs]
    clocks = [block["clock"] for block in blocks]
    ratings = [block["source_rating"] for block in blocks]
    terminations = [block["termination"] for block in blocks]
    return {
        "clock": {
            "status_plies": _summed_counts(clock["status_plies"] for clock in clocks),
            "precision_ms_games": _summed_counts(
                clock["precision_ms_games"] for clock in clocks
            ),
        },
        "time_initial_ms": _merged_integer_coverage(
            [block["time_initial_ms"] for block in blocks]
        ),
        "time_increment_ms": _merged_integer_coverage(
            [block["time_increment_ms"] for block in blocks]
        ),
        "source_rating": {
            "values_present": sum(rating["values_present"] for rating in ratings),
            "minimum": _extreme(ratings, "minimum", min),
            "maximum": _extreme(ratings, "maximum", max),
            "namespace_games": _summed_counts(
                rating["namespace_games"] for rating in ratings
            ),
        },
        "source_date": {
            "status_games": _summed_counts(
                block["source_date"]["status_games"] for block in blocks
            ),
        },
        # Over the axes the archives recorded rather than a list repeated here,
        # so an axis added to the per-archive block cannot go missing from the
        # corpus-wide one. The cost is that archives disagreeing about which
        # axes they carry raise here, after decoding, rather than in the
        # guarded roll-up, which only ever sees the recorded ones.
        "axes": {
            axis: _summed_counts_by_split(block["axes"][axis] for block in blocks)
            for axis in sorted({name for block in blocks for name in block["axes"]})
        },
        "termination": {
            "category_games": _summed_counts(
                termination["category_games"] for termination in terminations
            ),
            "attribution_games": _summed_counts(
                termination["attribution_games"] for termination in terminations
            ),
            "terminal_action_games": _summed_counts(
                termination["terminal_action_games"] for termination in terminations
            ),
            "abandonment": {
                "clock_share_threshold": terminations[0]["abandonment"][
                    "clock_share_threshold"
                ],
                "clock_share_judged_games": sum(
                    termination["abandonment"]["clock_share_judged_games"]
                    for termination in terminations
                ),
            },
        },
    }


def _merged_integer_coverage(
    blocks: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    return {
        "status_games": _summed_counts(block["status_games"] for block in blocks),
        "minimum": _extreme(blocks, "minimum", min),
        "maximum": _extreme(blocks, "maximum", max),
    }


def _summed_counts(blocks: Iterable[Mapping[str, int]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for block in blocks:
        total.update(block)
    return dict(sorted(total.items()))


def _rating_bucket(rating: object) -> str:
    """Name the bucket a player-slot's normalized rating falls in.

    A slot with no usable rating lands in a bucket spelled like the speed axis'
    but meaning something else on a different axis; the two are not the same
    fact and do not have to move together.
    """

    if not isinstance(rating, int):
        return "unclassified"
    floor = rating // _RATING_BUCKET_POINTS * _RATING_BUCKET_POINTS
    # Zero-padded so the manifest's sorted keys read in rating order; without
    # it a three-digit bucket lands between the two- and four-digit ones.
    return f"{floor:04d}_to_{floor + _RATING_BUCKET_POINTS - 1:04d}"


def _counts_by_split(
    counts: Mapping[tuple[str, str], int],
) -> dict[str, dict[str, int]]:
    """Turn ``(axis value, split)`` tallies into one split row per value.

    Every split is written for every observed value, so a zero is a measured
    absence rather than a key the reader has to notice is missing.
    """

    return {
        value: {split: counts.get((value, split), 0) for split in SPLIT_NAMES}
        for value in sorted({value for value, _ in counts})
    }


def _summed_counts_by_split(
    blocks: Iterable[Mapping[str, Mapping[str, int]]],
) -> dict[str, dict[str, int]]:
    total: Counter[tuple[str, str]] = Counter()
    for block in blocks:
        for value, splits in block.items():
            for split, count in splits.items():
                total[(value, split)] += count
    return _counts_by_split(total)


def _extreme(
    blocks: Sequence[Mapping[str, Any]],
    key: str,
    select: Callable[[Iterable[int]], int],
) -> int | None:
    """Combine one bound across archives, ignoring those that observed none."""

    observed = [block[key] for block in blocks if isinstance(block.get(key), int)]
    return select(observed) if observed else None
