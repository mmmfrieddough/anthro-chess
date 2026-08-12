"""The continuous account census a marked-account snapshot is cut from.

Account status is a live judgement the source publishes, and the only channel
that answers for it is one rate-limited bulk endpoint. Across the archives this
corpus is built from that answer arrives over weeks, so the census is a store
that accrues rather than a run with a finish line, and a snapshot cut from it
states the coverage it had reached. Decision record 0047 owns why, and what
that coverage does and does not claim.

Two consequences shape everything here. Accounts are asked about in descending
order of games played, because marked accounts play more than average, so
recall measured in player-slots runs far ahead of the share of accounts asked.
And every answer is stored against its account rather than as a position in a
list, because the account list grows with every archive acquired and an offset
into it means something else afterwards.

Everything this writes lives beneath the data root rather than in the
repository. The first census checkpointed itself beside its snapshot at a
gitignored path, and ``git worktree remove`` deletes ignored files, which is
how it was lost.

The counts an archive contributes are written by whichever pass reads it —
usually preparation, which is already reading every line — so the archive can be
reclaimed as soon as it is prepared. A counts file always speaks for a whole
archive: preparation writes one only when it reached the end, and
:func:`count_archive_accounts` is here for the archives that leaves.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from anthro_chess.data.accounts import MarkedAccounts, account_digest
from anthro_chess.data.artifacts import (
    SOURCE_USER_AGENT,
    file_sha256,
    open_pgn_text,
    write_text_atomically,
)

#: Bulk account lookup. One request answers for many accounts, which is what
#: makes covering an archive a matter of days rather than years.
LICHESS_USERS_ENDPOINT = "https://lichess.org/api/users"
#: The endpoint's documented maximum names per request. Names past it are
#: dropped rather than refused, so a larger batch loses accounts silently.
LICHESS_USERS_BATCH = 300
#: An access token for the source, which buys nothing but a larger allowance:
#: the endpoint needs no scope, so a token with none is enough. It is read from
#: the environment rather than any checked-in file because it is a credential.
LICHESS_TOKEN_VARIABLE = "LICHESS_TOKEN"
#: The source's own limiter, transcribed from where it enforces it rather than
#: inferred from refusals: ``lichess-org/lila``, ``Limiters.scala`` (the
#: ``apiUsers`` composite) and ``Api.scala`` (``usersByIds``). It charges the
#: calling address against a burst bucket and a daily one at once, and each
#: request costs ``len(batch) // divisor`` credits. Re-read those two files
#: before trusting these numbers again. Decision record 0047 owns why the pace
#: and the day's budget are derived from them rather than tuned against
#: refusals.
_BURST_CREDITS = 2_000
_BURST_WINDOW_SECONDS = 600.0
_DAILY_CREDITS = 30_000
#: What a token is worth: it halves the cost of every request.
_ANONYMOUS_DIVISOR = 3
_AUTHENTICATED_DIVISOR = 6
#: A refusal is a bucket emptying, and the burst one refills over the window
#: above, so one wait clears it. A second refusal is the daily bucket, which no
#: wait within a run clears, and that is an outcome rather than a fault.
_REFUSAL_ATTEMPTS = 2
_PLAYER_TAGS = ('[White "', '[Black "')
#: Both player tags are the same width, so the name starts at a fixed offset.
_PLAYER_TAG_LENGTH = len(_PLAYER_TAGS[0])
#: Where a census keeps what it accumulates, beneath the data root: an
#: archive's counts beside that archive, named after it because a selection may
#: acquire several into one artifact directory; the answers beside neither,
#: because account status belongs to the source rather than to any one
#: selection, and a second selection must inherit them rather than pay again.
CENSUS_DIRECTORY = "census"
ACCOUNT_GAMES_SUFFIX = ".accounts.tsv"
ANSWERS_FILE = "answers.tsv"
#: Version 2 drops the players that are not accounts, so a version 1 file
#: counts a third more player-slots than any account holds.
ACCOUNT_GAMES_FORMAT_VERSION = 2
_HEADER_PREFIX = "#"
logger = logging.getLogger(__name__)


class CensusError(ValueError):
    """Raised when the census cannot proceed."""


class SourceExhausted(CensusError):
    """Raised when the source's allowance is spent rather than something failing.

    Distinct from every other refusal because it is the expected end of a day's
    run: the accounts already answered are kept, and the next run continues.
    """


@dataclass(frozen=True)
class ArchiveAccounts:
    """How many games each account played in one archive.

    The digest is the archive the counts were taken from, so a re-pinned
    archive is counted again rather than silently described by the old file.
    """

    archive_sha256: str
    games_by_account: dict[str, int]


@dataclass(frozen=True)
class PinnedArchive:
    """One archive a selection pins, and where its counts are kept."""

    path: Path
    counts_path: Path
    sha256: str


def source_token() -> str | None:
    """Return the source access token this machine carries, if any."""

    return os.environ.get(LICHESS_TOKEN_VARIABLE, "").strip() or None


def is_account_name(name: str) -> bool:
    """Return whether a PGN player name could be an account at all.

    Not every player is one. A PGN prints ``?`` where the player was anonymous
    and ``AI level 3`` where the opponent was the source's own engine, and
    across this corpus those hold a third of every player-slot — enough to make
    a coverage figure that counted them meaningless, on top of the requests
    spent asking about names no account can carry. Both fail the source's
    username alphabet, which is what this tests rather than naming them.
    """

    return name.isascii() and name.replace("-", "").replace("_", "").isalnum()


class ArchiveAccountCounter:
    """Counts an account's games from the PGN lines it is shown.

    Preparation and a census pass both have every line of an archive in hand
    already, so both count through this rather than either parsing player names
    its own way: two producers of one counts file have to agree exactly.

    Names are folded to the source's account identity, because a PGN prints
    whatever capitalization the player typed and one account must not reach the
    queue twice.
    """

    def __init__(self) -> None:
        self._games_by_name: dict[str, int] = {}

    def observe(self, line: str) -> None:
        """Count one PGN line, which is a player tag or nothing of interest."""

        if line.startswith(_PLAYER_TAGS):
            name = line[_PLAYER_TAG_LENGTH:].partition('"')[0].strip().casefold()
            if name:
                self._games_by_name[name] = self._games_by_name.get(name, 0) + 1

    def account_games(self) -> dict[str, int]:
        """Return the counts, less the players that are not accounts.

        Sorting the non-accounts out here rather than in :meth:`observe` keeps
        the test off a path that runs twice per game across billions of them,
        and onto one that runs once per distinct name.
        """

        return {
            name: games
            for name, games in self._games_by_name.items()
            if is_account_name(name)
        }


def count_archive_accounts(archive_path: Path) -> dict[str, int]:
    """Return how many games each account appears in, across a whole archive.

    The whole archive is counted rather than the games a selection would
    accept, so raising a game bound or widening to another speed within a
    counted archive needs no new pass. Preparation writes the same counts as a
    by-product when it reads an archive to the end; this is for an archive
    nothing has prepared.
    """

    counter = ArchiveAccountCounter()
    with open_pgn_text(archive_path) as pgn_file:
        for line in pgn_file:
            counter.observe(line)
    return counter.account_games()


def write_account_games(path: Path, counted: ArchiveAccounts) -> None:
    """Write one archive's counts as a header line and sorted rows."""

    header = {
        "format_version": ACCOUNT_GAMES_FORMAT_VERSION,
        "archive_sha256": counted.archive_sha256,
        "accounts": len(counted.games_by_account),
        "slots": sum(counted.games_by_account.values()),
    }
    lines = [f"{_HEADER_PREFIX} {json.dumps(header, sort_keys=True)}"]
    lines.extend(
        f"{name}\t{games}" for name, games in sorted(counted.games_by_account.items())
    )
    write_text_atomically(path, "\n".join(lines) + "\n")


def read_account_games(path: Path) -> ArchiveAccounts:
    """Read one archive's counts."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CensusError(f"cannot read account counts {path}: {error}") from error
    header_line, _, body = text.partition("\n")
    if not header_line.startswith(_HEADER_PREFIX):
        raise CensusError(f"{path} has no header line")
    try:
        header = json.loads(header_line[len(_HEADER_PREFIX) :])
    except json.JSONDecodeError as error:
        raise CensusError(f"{path} has an unreadable header: {error}") from error
    if (
        not isinstance(header, dict)
        or header.get("format_version") != ACCOUNT_GAMES_FORMAT_VERSION
        or not isinstance(header.get("archive_sha256"), str)
    ):
        raise CensusError(
            f"{path} does not record the archive it counted in a format this code reads"
        )
    games_by_account: dict[str, int] = {}
    for line in body.splitlines():
        name, separator, games = line.partition("\t")
        if not separator or not games.isdecimal():
            raise CensusError(f"{path} holds a row that is not a count: {line!r}")
        games_by_account[name] = int(games)
    # The header's totals are what makes a short file loud. Nothing else would
    # notice one: rows that are gone are indistinguishable from accounts the
    # archive never held, and the census would ask about fewer accounts while
    # reporting full coverage of a smaller population.
    if (len(games_by_account), sum(games_by_account.values())) != (
        header.get("accounts"),
        header.get("slots"),
    ):
        raise CensusError(
            f"{path} holds {len(games_by_account)} account(s) over "
            f"{sum(games_by_account.values())} player-slot(s) but its header "
            f"claims {header.get('accounts')} over {header.get('slots')}"
        )
    return ArchiveAccounts(
        archive_sha256=str(header["archive_sha256"]),
        games_by_account=games_by_account,
    )


def account_games(archive: PinnedArchive) -> ArchiveAccounts:
    """Return an archive's counts, taking the pass over it at most once.

    The counts outlive the archive they came from, so a census keeps asking
    about an archive whose raw file was reclaimed after preparation. Only a
    first count, or one the selection has invalidated by pinning different
    bytes, needs the archive back.
    """

    if archive.counts_path.is_file():
        try:
            counted = read_account_games(archive.counts_path)
        except CensusError as error:
            # Recounting is the repair for every way a counts file can be
            # unusable — an older format, a short write — and it is available
            # whenever the archive still is. Raising instead would leave a
            # scheduled census failing nightly on something one pass fixes.
            logger.warning("%s; counting that archive again", error)
        else:
            if counted.archive_sha256 == archive.sha256:
                return counted
            logger.info(
                "%s counts an archive this selection no longer pins; counting again",
                archive.counts_path,
            )
    if not archive.path.is_file():
        raise CensusError(
            f"{archive.path} is not on disk, so the accounts this selection "
            "pins there cannot be counted; acquire it again"
        )
    logger.info("Counting the accounts in %s", archive.path.name)
    digest = file_sha256(archive.path)
    if digest != archive.sha256:
        raise CensusError(
            f"{archive.path} hashes to {digest}, not the {archive.sha256} this "
            "selection pins; acquire it again before counting it"
        )
    counted = ArchiveAccounts(
        archive_sha256=digest,
        games_by_account=count_archive_accounts(archive.path),
    )
    write_account_games(archive.counts_path, counted)
    return counted


def counted_archives(archives: Sequence[PinnedArchive]) -> list[PinnedArchive]:
    """Return the archives whose counts stand as they are.

    What a run can ask about is decided from these rather than from what it
    could count, so a day's allowance is spent before a backlog of counting
    rather than behind it.
    """

    return [archive for archive in archives if _counts_are_current(archive)]


def refresh_archive_counts(
    archives: Sequence[PinnedArchive],
    *,
    workers: int,
) -> None:
    """Count every archive whose counts are missing or superseded, at once.

    One pass decompresses tens of gigabytes, so a batch of newly acquired
    archives — or a format that supersedes every counts file at once — is hours
    of work that parallelizes exactly. Leaving it to :func:`account_games`
    instead would do the same work one archive at a time.
    """

    missing = [archive for archive in archives if not _counts_are_current(archive)]
    if not missing:
        return
    logger.info(
        "Counting %s archive(s) whose counts are missing or superseded, %s at a time",
        len(missing),
        workers,
    )
    if workers <= 1:
        for archive in missing:
            account_games(archive)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(_count_archive, missing):
            pass


def _counts_are_current(archive: PinnedArchive) -> bool:
    """Return whether an archive's counts are this format's, for these bytes.

    Reads the header alone. Deciding which of fifty archives need counting must
    not cost a parse of every row of the ones that do not.
    """

    try:
        with archive.counts_path.open(encoding="utf-8") as counts_file:
            header = json.loads(counts_file.readline().removeprefix(_HEADER_PREFIX))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(header, dict) and (
        header.get("format_version"),
        header.get("archive_sha256"),
    ) == (ACCOUNT_GAMES_FORMAT_VERSION, archive.sha256)


def _count_archive(archive: PinnedArchive) -> None:
    # Returning the counts would pickle a dictionary of every account in the
    # archive back to a caller that reads them from disk anyway.
    account_games(archive)


@dataclass(frozen=True)
class Census:
    """Every account the covered archives hold, and what the source said so far.

    ``answers`` is restricted to accounts the covered archives hold, so every
    quantity below is a share of this corpus rather than of everything the
    census has ever asked about.
    """

    archives: tuple[str, ...]
    games_by_account: dict[str, int]
    answers: dict[str, bool]

    @property
    def accounts_total(self) -> int:
        return len(self.games_by_account)

    @property
    def accounts_queried(self) -> int:
        return len(self.answers)

    @property
    def slots_total(self) -> int:
        return sum(self.games_by_account.values())

    @property
    def slots_queried(self) -> int:
        return sum(self.games_by_account[name] for name in self.answers)

    @property
    def marked(self) -> frozenset[str]:
        return frozenset(name for name, marked in self.answers.items() if marked)

    def queue(self, limit: int) -> list[str]:
        """Return the busiest accounts nobody has asked about yet."""

        unanswered = [
            name for name in self.games_by_account if name not in self.answers
        ]
        unanswered.sort(key=lambda name: (-self.games_by_account[name], name))
        return unanswered[:limit]


def read_census(archives_pinned: Sequence[PinnedArchive], answers_path: Path) -> Census:
    """Assemble the queue's inputs from the counted archives and the answers."""

    archives: list[str] = []
    games_by_account: dict[str, int] = {}
    for archive in archives_pinned:
        counted = account_games(archive)
        archives.append(counted.archive_sha256)
        for name, games in counted.games_by_account.items():
            games_by_account[name] = games_by_account.get(name, 0) + games
    return Census(
        archives=tuple(sorted(archives)),
        games_by_account=games_by_account,
        answers={
            name: marked
            for name, marked in read_answers(answers_path).items()
            if name in games_by_account
        },
    )


def read_answers(path: Path) -> dict[str, bool]:
    """Read every answer the census has recorded."""

    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and not text.endswith("\n"):
        # A run killed mid-append leaves a final line that was never a whole
        # answer. Dropping it costs one account being asked about again.
        lines.pop()
    answers: dict[str, bool] = {}
    for line in lines:
        name, separator, rest = line.partition("\t")
        marked = rest.partition("\t")[0]
        if not separator or marked not in {"0", "1"}:
            raise CensusError(f"{path} holds a line that is not an answer: {line!r}")
        answers[name] = marked == "1"
    return answers


def append_answers(path: Path, answers: Mapping[str, bool], queried_at: str) -> None:
    """Append one batch of answers, keeping every earlier one verbatim."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as store:
        store.write(
            "".join(
                f"{name}\t{int(marked)}\t{queried_at}\n"
                for name, marked in answers.items()
            )
        )


def sustainable_pause(batch_size: int, *, authenticated: bool) -> float:
    """Return the shortest pause between batches the burst allowance sustains.

    Two requests of the allowance are left unspent, and the pace is rounded up
    to a whole second, because the count of requests that lands in a window is
    a step function and a pace derived to the boundary sits on the step.

    A window holds one more request than its length divided by the spacing,
    since a request at each end falls inside it. Spending the allowance's
    average rate therefore puts 41 requests into a 40-request window, which is
    what the first scheduled census measured: refused on every fortieth batch,
    thirteen times, at intervals of 1186 seconds against the 600 it paced for,
    with half its running time spent asleep. Pacing one request back from that
    yields exactly 39.0 requests per window — the value at which the step
    changes, decided by rounding rather than by argument.

    Only the burst bucket is paced against. The daily one is spent as fast as
    the burst bucket allows, which is why a run ends either at the budget
    :func:`daily_account_allowance` predicts or at the refusal that proves the
    prediction was optimistic.
    """

    affordable = _BURST_CREDITS // _request_credits(batch_size, authenticated)
    return float(math.ceil(_BURST_WINDOW_SECONDS / (affordable - 2)))


def daily_account_allowance(batch_size: int, *, authenticated: bool) -> int:
    """Return how many accounts one day's credits buy at this batch size."""

    return _DAILY_CREDITS // _request_credits(batch_size, authenticated) * batch_size


def _request_credits(batch_size: int, authenticated: bool) -> int:
    divisor = _AUTHENTICATED_DIVISOR if authenticated else _ANONYMOUS_DIVISOR
    return max(batch_size // divisor, 1)


@dataclass(frozen=True)
class CensusRun:
    """What one day's asking achieved, and why it stopped.

    ``refused`` is not a failure. A run paced at the sustainable rate ends by
    emptying the daily bucket, and a scheduler that read that as a fault would
    report one every day the census worked as intended.

    ``asked`` names the accounts now answered for rather than how many, so what
    a run added to the census is read from the run rather than re-derived from
    how it consumed its queue.
    """

    asked: tuple[str, ...]
    accounts_marked: int
    refused: bool

    @property
    def accounts_asked(self) -> int:
        return len(self.asked)


def run_census(
    queue: Sequence[str],
    answers_path: Path,
    *,
    queried_at: str,
    batch_size: int = LICHESS_USERS_BATCH,
    pause_seconds: float | None = None,
) -> CensusRun:
    """Ask the source about a queue of accounts, recording answers as they come.

    Every account asked about is recorded, marked or not, so no account is ever
    paid for twice. An account the source does not answer for at all is
    recorded as unmarked: it has no status to disclose, which is the same thing
    preparation does with it.
    """

    if batch_size > LICHESS_USERS_BATCH:
        raise CensusError(
            f"a batch of {batch_size} exceeds the {LICHESS_USERS_BATCH} the source "
            "answers for; it drops the excess silently, and the answers would "
            "record the dropped accounts as asked about"
        )
    token = source_token()
    pause = (
        sustainable_pause(batch_size, authenticated=token is not None)
        if pause_seconds is None
        else pause_seconds
    )
    logger.info(
        "Asking %s about %s account(s), one batch of %s every %.1fs",
        "as an authenticated caller" if token else "anonymously",
        len(queue),
        batch_size,
        pause,
    )
    batches = [
        queue[start : start + batch_size] for start in range(0, len(queue), batch_size)
    ]
    asked: list[str] = []
    marked = 0
    for index, batch in enumerate(batches, start=1):
        started = time.monotonic()
        try:
            answers = _ask(batch, token)
        except SourceExhausted:
            logger.info(
                "The source's allowance is spent; %s account(s) asked about",
                len(asked),
            )
            return CensusRun(asked=tuple(asked), accounts_marked=marked, refused=True)
        append_answers(answers_path, answers, queried_at)
        asked.extend(batch)
        marked += sum(answers.values())
        if index % 25 == 0 or index == len(batches):
            logger.info(
                "Asked about %s/%s account(s); %s marked",
                len(asked),
                len(queue),
                marked,
            )
        if index < len(batches):
            # The limiter charges per request, so the pause is the interval
            # between them rather than time added to each.
            time.sleep(max(0.0, pause - (time.monotonic() - started)))
    return CensusRun(asked=tuple(asked), accounts_marked=marked, refused=False)


def snapshot_from_census(census: Census, *, queried_at: str) -> MarkedAccounts:
    """Cut a marked-account snapshot from the census as it stands."""

    return MarkedAccounts(
        covers_archives=census.archives,
        queried_at=queried_at,
        accounts_total=census.accounts_total,
        accounts_queried=census.accounts_queried,
        slots_total=census.slots_total,
        slots_queried=census.slots_queried,
        digests=frozenset(account_digest(name) for name in census.marked),
    )


def _ask(batch: Sequence[str], token: str | None) -> dict[str, bool]:
    answers = dict.fromkeys(batch, False)
    records = _post_usernames(list(batch), token)
    if not records:
        # An account the source omits has no status to disclose and is recorded
        # unmarked, which is right for an erased account and wrong for a whole
        # batch: nothing is ever re-asked, so a hiccup answering 200 with
        # nothing would write a batch of permanent clean verdicts. Stopping the
        # run costs a day of allowance and is visible; recording them is not.
        raise CensusError(
            f"the source answered for none of {len(batch)} account(s), which is "
            "a fault rather than a batch of accounts it has never heard of"
        )
    for record in records:
        identifier = record.get("id") or record.get("username")
        if isinstance(identifier, str) and identifier.casefold() in answers:
            answers[identifier.casefold()] = bool(record.get("tosViolation"))
    return answers


def _post_usernames(batch: list[str], token: str | None) -> list[dict[str, object]]:
    headers = {"User-Agent": SOURCE_USER_AGENT}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        LICHESS_USERS_ENDPOINT,
        data=",".join(batch).encode(),
        method="POST",
        headers=headers,
    )
    for attempt in range(1, _REFUSAL_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=60) as response:  # noqa: S310
                payload = json.loads(response.read())
        except HTTPError as error:
            if error.code != 429:
                raise CensusError(
                    f"cannot query account status from {LICHESS_USERS_ENDPOINT}: "
                    f"{error}"
                ) from error
            if attempt == _REFUSAL_ATTEMPTS:
                raise SourceExhausted(
                    "the source is still refusing after a full burst window, so "
                    "what is exhausted is the day's allowance"
                ) from error
            logger.warning(
                "Rate limited; waiting %ss for the allowance to refill "
                "(attempt %s of %s)",
                _BURST_WINDOW_SECONDS,
                attempt,
                _REFUSAL_ATTEMPTS,
            )
            time.sleep(_BURST_WINDOW_SECONDS)
        except (URLError, OSError, json.JSONDecodeError) as error:
            raise CensusError(
                f"cannot query account status from {LICHESS_USERS_ENDPOINT}: {error}"
            ) from error
        else:
            if not isinstance(payload, list):
                raise CensusError(
                    "account status endpoint returned an unexpected payload"
                )
            return [record for record in payload if isinstance(record, dict)]
    raise CensusError(  # pragma: no cover - the loop returns or raises
        "account status endpoint was never reached"
    )
