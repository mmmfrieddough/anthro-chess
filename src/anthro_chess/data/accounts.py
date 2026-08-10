"""Pinned snapshot of source accounts marked for terms-of-service violations.

Account status is a live judgement rather than a property of an archive.
Marks accumulate as moderation proceeds, so asking a source about the same
accounts twice returns different answers, and marks only ever accumulate, so
the second answer removes strictly more games than the first. Preparation
reading the source directly would therefore produce a different corpus on every
run and quietly shrink an evaluation pool that later generations are required
to contain.

So the answer is taken once, pinned to the archive it covers, and checked in.
Refreshing it is a deliberate act that starts a new corpus, exactly as changing
the archive digest does. A snapshot built against one archive refuses to
prepare another, which is what stops a widened corpus from silently keeping the
accounts nobody asked about.

Usernames are stored as truncated salted digests. Membership is all preparation
needs and a digest serves it as well as a name, so the checked-in file is not a
readable roster. The salt is public and the account space is the archive's, so
this obscures rather than protects — anyone holding the same archive can
recover the names from it. What it buys is that the repository does not itself
publish the list.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from anthro_chess.data.artifacts import (
    SOURCE_USER_AGENT,
    open_pgn_text,
    write_text_atomically,
)

#: Bulk account lookup. One request answers for many accounts, which is what
#: makes covering a whole archive a matter of hours rather than weeks.
LICHESS_USERS_ENDPOINT = "https://lichess.org/api/users"
#: The endpoint's documented maximum names per request.
LICHESS_USERS_BATCH = 300
#: Covering an archive takes hundreds of requests, so a refusal partway through
#: is ordinary rather than exceptional and waiting one out belongs here rather
#: than in whatever invokes the command.
#:
#: The length is chosen rather than measured, on thin evidence: a refusal was
#: once cleared by five minutes of silence and never by one minute, and asking
#: again sooner than that appeared to renew it rather than wait it out.
_REFUSAL_WAIT = 600.0
_REFUSAL_ATTEMPTS = 5
#: Between batches. The source documents no budget for this endpoint and
#: serves no ``Retry-After``, so this is chosen rather than derived, and the
#: choice is deliberately slow. What covering one archive established: a
#: refusal is a burst allowance running out rather than a rate being exceeded,
#: faster pauses spend it sooner and buy a long wait instead of throughput, and
#: a refusal compounds — once enough of them accumulate, pauses that had been
#: working are refused too. Nothing here was ever shown sustainable across
#: hundreds of batches, because the accumulated penalty spoiled every later
#: measurement. So this errs well past anything observed to fail rather than
#: near the fastest thing observed to work: a whole archive is a few hours
#: either way, and the alternative cost a day. Lower it with evidence.
_DEFAULT_PAUSE = 10.0
_PLAYER_TAGS = ('[White "', '[Black "')
#: Both player tags are the same width, so the name starts at a fixed offset.
_PLAYER_TAG_LENGTH = len(_PLAYER_TAGS[0])
logger = logging.getLogger(__name__)

SNAPSHOT_FORMAT_VERSION = 1
#: Fixed, checked-in, and part of the artifact's identity. Changing it
#: invalidates every stored digest, so it is versioned with the format rather
#: than configured per selection.
DIGEST_SALT = "anthro-marked-accounts-v1"
#: Half a SHA-256. Collisions across a source's whole account space stay far
#: below one game, and the file halves.
DIGEST_LENGTH = 32
_HEADER_PREFIX = "#"


class MarkedAccountError(ValueError):
    """Raised when a marked-account snapshot is unusable or does not apply."""


def account_digest(username: str) -> str:
    """Return the stored digest for one source username.

    Case is folded because a source's account identity is case-insensitive
    while the name a PGN prints preserves whatever the player typed.
    """

    normalized = username.strip().casefold()
    return sha256(f"{DIGEST_SALT}\0{normalized}".encode()).hexdigest()[:DIGEST_LENGTH]


#: Hex characters of :func:`account_digest` that fit a normalized row's fixed
#: 64-bit player column. Truncating a stored snapshot digest the same way
#: yields the same integer, so a corpus row and a snapshot can be matched
#: without the corpus storing usernames.
_ROW_DIGEST_LENGTH = 16


def account_row_digest(username: str) -> int:
    """Return the normalized-row player identifier for one source username."""

    return int(account_digest(username)[:_ROW_DIGEST_LENGTH], 16)


@dataclass(frozen=True)
class MarkedAccounts:
    """Marked accounts observed across the archive a snapshot covers.

    ``covers_archive`` is a coverage claim rather than provenance: the snapshot
    speaks for every account appearing anywhere in that archive, so preparation
    may read an unlisted account as unmarked instead of as unknown, and must
    refuse any other archive.

    Growing a snapshot to cover a second archive is deliberately absent: it
    would have to keep every earlier verdict verbatim and ask only about
    genuinely new accounts, since re-deciding a covered account applies a later
    moderation decision retroactively and drops games an earlier pool
    generation contains. Preparation appends one archive at a time, so a corpus
    spanning archives cannot set ``filters.marked_accounts`` at all until a
    snapshot can speak for more than one: ``require_archive`` refuses the second
    archive rather than preparing it unfiltered.
    """

    covers_archive: str
    queried_at: str
    accounts_queried: int
    digests: frozenset[str]

    @property
    def accounts_marked(self) -> int:
        """Return how many accounts the snapshot marks."""

        return len(self.digests)

    def contains(self, username: str) -> bool:
        """Return whether one source username is marked."""

        return account_digest(username) in self.digests

    def require_archive(self, archive_sha256: str) -> None:
        """Reject an archive this snapshot never asked about."""

        if archive_sha256 != self.covers_archive:
            raise MarkedAccountError(
                f"marked-account snapshot does not cover archive {archive_sha256} "
                f"(it covers {self.covers_archive}); build one for this archive "
                "with `uv run anthro data mark-accounts --config <selection> "
                "--output <path>` before preparing"
            )

    def write(self, path: str | Path) -> Path:
        """Write the snapshot as a header line and sorted digests."""

        output_path = Path(path)
        header = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "covers_archive": self.covers_archive,
            "queried_at": self.queried_at,
            "accounts_queried": self.accounts_queried,
            "accounts_marked": self.accounts_marked,
            "salt": DIGEST_SALT,
        }
        lines = [f"{_HEADER_PREFIX} {json.dumps(header, sort_keys=True)}"]
        lines.extend(sorted(self.digests))
        write_text_atomically(output_path, "\n".join(lines) + "\n")
        return output_path


def load_marked_accounts(path: str | Path) -> MarkedAccounts:
    """Read a checked-in marked-account snapshot."""

    snapshot_path = Path(path)
    try:
        text = snapshot_path.read_text(encoding="utf-8")
    except OSError as error:
        raise MarkedAccountError(
            f"cannot read marked-account snapshot {snapshot_path}: {error}"
        ) from error

    header_line, _, body = text.partition("\n")
    if not header_line.startswith(_HEADER_PREFIX):
        raise MarkedAccountError(
            f"marked-account snapshot {snapshot_path} has no header line"
        )
    try:
        header = json.loads(header_line[len(_HEADER_PREFIX) :])
    except json.JSONDecodeError as error:
        raise MarkedAccountError(
            f"marked-account snapshot {snapshot_path} has an unreadable header: {error}"
        ) from error
    if not isinstance(header, dict):
        raise MarkedAccountError(
            f"marked-account snapshot {snapshot_path} has a header that is not "
            "an object"
        )
    if header.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        raise MarkedAccountError(
            f"marked-account snapshot {snapshot_path} uses format version "
            f"{header.get('format_version')}; expected {SNAPSHOT_FORMAT_VERSION}"
        )
    if header.get("salt") != DIGEST_SALT:
        raise MarkedAccountError(
            f"marked-account snapshot {snapshot_path} was built with a different "
            "digest salt, so its digests cannot be matched"
        )

    digests = frozenset(line for line in body.splitlines() if line)
    if any(len(digest) != DIGEST_LENGTH for digest in digests):
        raise MarkedAccountError(
            f"marked-account snapshot {snapshot_path} holds a malformed digest"
        )
    if len(digests) != header.get("accounts_marked"):
        raise MarkedAccountError(
            f"marked-account snapshot {snapshot_path} holds {len(digests)} digest(s) "
            f"but its header claims {header.get('accounts_marked')}"
        )
    covers_archive = header.get("covers_archive")
    queried_at = header.get("queried_at")
    accounts_queried = header.get("accounts_queried")
    if (
        not isinstance(covers_archive, str)
        or not isinstance(queried_at, str)
        or not isinstance(accounts_queried, int)
    ):
        raise MarkedAccountError(
            f"marked-account snapshot {snapshot_path} does not record the archive "
            "it covers, when it was built, or how many accounts it asked about"
        )
    return MarkedAccounts(
        covers_archive=covers_archive,
        queried_at=queried_at,
        accounts_queried=accounts_queried,
        digests=digests,
    )


def resolve_snapshot_path(configured: Path, config_source: str | None) -> Path:
    """Locate a configured snapshot relative to the selection that names it.

    A snapshot is pinned beside its selection rather than under the artifact
    root, because it is a checked-in input rather than anything a run produces.
    Both the command that writes one and the preparation that reads one need
    that rule, so it is stated once.
    """

    if configured.is_absolute():
        return configured
    if config_source is None:
        raise MarkedAccountError(
            "filters.marked_accounts is relative but the configuration was not "
            "loaded from a file, so there is nothing to resolve it against"
        )
    return Path(config_source).parent / configured


def marked_accounts_from_usernames(
    marked: Iterable[str],
    *,
    archive_sha256: str,
    queried_at: str,
    accounts_queried: int,
) -> MarkedAccounts:
    """Build a first snapshot from the usernames a source reported as marked."""

    return MarkedAccounts(
        covers_archive=archive_sha256,
        queried_at=queried_at,
        accounts_queried=accounts_queried,
        digests=frozenset(account_digest(username) for username in marked),
    )


def scan_archive_accounts(archive_path: str | Path) -> list[str]:
    """Return every distinct account name appearing in one PGN archive.

    The whole archive is covered rather than the games a selection would
    accept, so raising a game bound or widening to another speed within the
    same archive needs no new snapshot.

    The order is deterministic because the query's resume file is an offset
    into this list.
    """

    accounts: set[str] = set()
    with open_pgn_text(Path(archive_path)) as pgn_file:
        for line in pgn_file:
            if line.startswith(_PLAYER_TAGS):
                name = line[_PLAYER_TAG_LENGTH:].partition('"')[0].strip()
                if name:
                    accounts.add(name)
    return sorted(accounts)


def query_marked_accounts(
    usernames: Sequence[str],
    *,
    batch_size: int = LICHESS_USERS_BATCH,
    pause_seconds: float | None = None,
    resume_path: str | Path | None = None,
) -> set[str]:
    """Return which of ``usernames`` the source reports as marked.

    A closed account reports no status at all, so it is left unmarked here
    rather than guessed at; the snapshot's coverage count records that it was
    asked about.

    Covering an archive takes hundreds of requests against a service that rate
    limits them, so progress is written to ``resume_path`` after every batch
    and a later call continues from it. Without that, any one refusal discards
    an hour of answers and the retry is likelier to be refused than the first
    attempt was.
    """

    pause = _DEFAULT_PAUSE if pause_seconds is None else pause_seconds
    progress_path = None if resume_path is None else Path(resume_path)
    fingerprint = _usernames_fingerprint(usernames)
    marked, completed = _load_progress(progress_path, fingerprint)
    if completed:
        logger.info(
            "Resuming after %s already-queried account(s); %s marked so far",
            completed,
            len(marked),
        )
    remaining = list(usernames[completed:])
    batches = [
        remaining[start : start + batch_size]
        for start in range(0, len(remaining), batch_size)
    ]
    for index, batch in enumerate(batches, start=1):
        for record in _post_usernames(batch):
            if record.get("tosViolation"):
                name = record.get("username") or record.get("id")
                if isinstance(name, str):
                    marked.add(name)
        completed += len(batch)
        _save_progress(progress_path, marked, completed, fingerprint)
        if index % 25 == 0 or index == len(batches):
            logger.info(
                "Queried %s/%s account(s); %s marked so far",
                completed,
                len(usernames),
                len(marked),
            )
        if index < len(batches):
            time.sleep(pause)
    return marked


def _usernames_fingerprint(usernames: Sequence[str]) -> str:
    digest = sha256()
    for name in usernames:
        digest.update(f"{name}\0".encode())
    return digest.hexdigest()


def _load_progress(path: Path | None, fingerprint: str) -> tuple[set[str], int]:
    if path is None or not path.is_file():
        return set(), 0
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        stored = record["usernames"]
        marked, completed = set(record["marked"]), int(record["completed"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise MarkedAccountError(
            f"cannot resume from {path}: {error}; delete it to start over"
        ) from error
    if stored != fingerprint:
        raise MarkedAccountError(
            f"{path} records progress through a different account list, so "
            "resuming from it would skip accounts and keep the other list's "
            "marks; delete it to start over"
        )
    return marked, completed


def _save_progress(
    path: Path | None, marked: set[str], completed: int, fingerprint: str
) -> None:
    if path is None:
        return
    # Unlike the snapshot it is written beside, this holds marked usernames in
    # the clear; ``.gitignore`` is what keeps it out of the repository.
    write_text_atomically(
        path,
        json.dumps(
            {
                "usernames": fingerprint,
                "completed": completed,
                "marked": sorted(marked),
            }
        ),
    )


def _post_usernames(batch: list[str]) -> list[dict[str, object]]:
    request = Request(
        LICHESS_USERS_ENDPOINT,
        data=",".join(batch).encode(),
        method="POST",
        headers={"User-Agent": SOURCE_USER_AGENT},
    )
    for attempt in range(1, _REFUSAL_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=60) as response:  # noqa: S310
                payload = json.loads(response.read())
        except HTTPError as error:
            if error.code != 429:
                raise MarkedAccountError(
                    f"cannot query account status from {LICHESS_USERS_ENDPOINT}: "
                    f"{error}"
                ) from error
            if attempt == _REFUSAL_ATTEMPTS:
                raise MarkedAccountError(
                    f"the source is still rate limiting after "
                    f"{_REFUSAL_ATTEMPTS} waits; leave it alone for a while and "
                    "run the command again. Progress is checkpointed, so a "
                    "later run resumes rather than repeating what is done"
                ) from error
            logger.warning(
                "Rate limited; waiting %ss for the allowance to refill "
                "(attempt %s of %s)",
                _REFUSAL_WAIT,
                attempt,
                _REFUSAL_ATTEMPTS,
            )
            time.sleep(_REFUSAL_WAIT)
        except (URLError, OSError, json.JSONDecodeError) as error:
            raise MarkedAccountError(
                f"cannot query account status from {LICHESS_USERS_ENDPOINT}: {error}"
            ) from error
        else:
            if not isinstance(payload, list):
                raise MarkedAccountError(
                    "account status endpoint returned an unexpected payload"
                )
            return [record for record in payload if isinstance(record, dict)]
    raise MarkedAccountError(  # pragma: no cover - the loop returns or raises
        "account status endpoint was never reached"
    )
