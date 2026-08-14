"""Pinned snapshot of source accounts marked for terms-of-service violations.

Account status is a live judgement rather than a property of an archive.
Marks accumulate as moderation proceeds, so asking a source about the same
accounts twice returns different answers, and marks only ever accumulate, so
the second answer removes strictly more games than the first. Preparation
reading the source directly would therefore produce a different corpus on every
run and quietly shrink an evaluation pool that later generations are required
to contain.

So the answer is taken once, pinned to the archives it covers, and checked in.
Refreshing it is a deliberate act that starts a new corpus, exactly as changing
an archive digest does. A snapshot refuses an archive it never counted, which
is what stops a widened corpus from silently keeping the accounts nobody asked
about.

What a snapshot does not claim is that everyone in those archives was asked
about. The census it is cut from asks the source about accounts continuously
and in descending order of games played, so a snapshot is a moment in that
rather than the end of it, and its header carries the coverage it had reached.
Decision record 0047 owns what that number is and is not.

Usernames are stored as truncated salted digests. Membership is all preparation
needs and a digest serves it as well as a name, so the checked-in file is not a
readable roster. The salt is public and the account space is the archives', so
this obscures rather than protects — anyone holding the same archives can
recover the names from it. What it buys is that the repository does not itself
publish the list.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from anthro_chess.data.artifacts import (
    file_sha256,
    manifest_archive_digests,
    write_text_atomically,
)
from anthro_chess.data.schema import NormalizedColumn

logger = logging.getLogger(__name__)

SNAPSHOT_FORMAT_VERSION = 2
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
    """Marked accounts found among the archives a snapshot covers.

    ``covers_archives`` is a coverage claim rather than provenance: the
    snapshot speaks for those archives and must refuse any other, so widening
    the corpus stops the run instead of quietly keeping the accounts nobody
    asked about.

    Within them it speaks partially, and the counts say how partially. An
    account that is listed is marked; an account that is not was either
    answered for and clean, or never asked about, and nothing here can tell
    the two apart, so its games are kept either way. ``slots_queried`` against
    ``slots_total`` is what that costs, in the player-slots the corpus is
    actually made of rather than in accounts, because marked accounts play more
    games than average.

    Growing a snapshot is deliberately absent. A later census answers for more
    accounts and re-answers for none, but re-cutting a snapshot from it applies
    later moderation decisions retroactively and drops games an earlier pool
    generation contains, which is why a refresh starts a new corpus rather than
    amending this one.
    """

    covers_archives: tuple[str, ...]
    queried_at: str
    accounts_total: int
    accounts_queried: int
    slots_total: int
    slots_queried: int
    digests: frozenset[str]

    @property
    def accounts_marked(self) -> int:
        """Return how many accounts the snapshot marks."""

        return len(self.digests)

    @property
    def slot_coverage(self) -> float:
        """Return the share of player-slots the census had answered for."""

        return self.slots_queried / self.slots_total if self.slots_total else 0.0

    def contains(self, username: str) -> bool:
        """Return whether one source username is marked."""

        return account_digest(username) in self.digests

    def row_digests(self) -> frozenset[int]:
        """Return the marks as the identifier a normalized row stores."""

        return frozenset(
            int(digest[:_ROW_DIGEST_LENGTH], 16) for digest in self.digests
        )

    def as_record(self) -> dict[str, object]:
        """Return what an artifact records about the snapshot it was cut under.

        A corpus and a pool both carry this, and they describe one snapshot, so
        a field added here reaches both rather than whichever writer was edited.
        """

        return {
            "covers_archives": list(self.covers_archives),
            "queried_at": self.queried_at,
            "accounts_total": self.accounts_total,
            "accounts_queried": self.accounts_queried,
            "accounts_marked": self.accounts_marked,
            "slots_total": self.slots_total,
            "slots_queried": self.slots_queried,
            "slot_coverage": self.slot_coverage,
        }

    def require_archive(self, archive_sha256: str) -> None:
        """Reject an archive this snapshot never counted."""

        if archive_sha256 not in self.covers_archives:
            raise MarkedAccountError(
                f"marked-account snapshot does not cover archive {archive_sha256} "
                f"(it covers {len(self.covers_archives)} other archive(s)); count "
                "that archive into the census with `uv run anthro data census "
                "--config <selection>` and cut a new snapshot that covers it"
            )

    def write(self, path: str | Path) -> Path:
        """Write the snapshot as a header line and sorted digests."""

        output_path = Path(path)
        header = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "covers_archives": sorted(self.covers_archives),
            "queried_at": self.queried_at,
            "accounts_total": self.accounts_total,
            "accounts_queried": self.accounts_queried,
            "slots_total": self.slots_total,
            "slots_queried": self.slots_queried,
            "accounts_marked": self.accounts_marked,
            "salt": DIGEST_SALT,
        }
        lines = [f"{_HEADER_PREFIX} {json.dumps(header, sort_keys=True)}"]
        lines.extend(sorted(self.digests))
        write_text_atomically(output_path, "\n".join(lines) + "\n")
        return output_path


def marks_a_player(
    row: Mapping[str, Any],
    marked_digests: frozenset[int] | None,
) -> bool:
    """Return whether either player of one normalized row is a marked account.

    Every reader that applies this rejection asks the same question of the same
    two columns, and one of them asks it negated, which is where a hand-written
    copy goes wrong. ``None`` and an empty set both answer ``False`` without
    reading either column, because a reader rejecting nobody does not project
    them; the two are distinguished by the caller, not here.
    """

    if not marked_digests:
        return False
    return (
        row[NormalizedColumn.WHITE_PLAYER_DIGEST] in marked_digests
        or row[NormalizedColumn.BLACK_PLAYER_DIGEST] in marked_digests
    )


@dataclass(frozen=True)
class CorpusSnapshot:
    """A snapshot one reader of a prepared corpus resolved, and its digest."""

    accounts: MarkedAccounts
    sha256: str

    def as_record(self) -> dict[str, object]:
        """Return what an artifact records about the rejection it applied.

        The file's own digest and not only the header counts: two snapshots
        can carry the same counts over different accounts, and an artifact
        naming a path alone cannot be checked once that path is overwritten.
        """

        return {**self.accounts.as_record(), "snapshot_sha256": self.sha256}


def snapshot_for_corpus(
    configured: Path | None,
    config_source: str | None,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> CorpusSnapshot | None:
    """Load the snapshot a selection names, if it covers the corpus it filters.

    Every reader of a prepared corpus that can apply this rejection reaches it
    the same way — a path beside the selection naming it, held to the archives
    the corpus manifest says it was prepared from — because a reader resolving
    either differently would reject a different set of games from the same
    corpus and the same snapshot.
    """

    if configured is None:
        return None
    path = resolve_snapshot_path(configured, config_source)
    accounts = load_snapshot_covering(
        path, manifest_archive_digests(manifest, manifest_path)
    )
    return CorpusSnapshot(accounts=accounts, sha256=file_sha256(path))


def load_snapshot_covering(
    path: str | Path,
    archive_sha256s: Iterable[str],
) -> MarkedAccounts:
    """Load a snapshot, refusing one that never counted an archive named here.

    A corpus applies this rejection while it is prepared, a pool applies it
    when its generation is cut, and a training selection applies it when it
    loads, so what a snapshot has to be before any of them may use it, and what
    it claims once they do, is stated once.
    """

    snapshot = load_marked_accounts(path)
    for archive_sha256 in archive_sha256s:
        snapshot.require_archive(archive_sha256)
    logger.info(
        "Rejecting the games of %s marked account(s), from a census cut %s that "
        "had answered for %.1f%% of the player-slots it counted",
        snapshot.accounts_marked,
        snapshot.queried_at,
        100 * snapshot.slot_coverage,
    )
    return snapshot


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
    covers_archives = header.get("covers_archives")
    queried_at = header.get("queried_at")
    accounts_total = header.get("accounts_total")
    accounts_queried = header.get("accounts_queried")
    slots_total = header.get("slots_total")
    slots_queried = header.get("slots_queried")
    if (
        not isinstance(covers_archives, list)
        or not covers_archives
        or not all(isinstance(archive, str) for archive in covers_archives)
        or not isinstance(queried_at, str)
        or not isinstance(accounts_total, int)
        or not isinstance(accounts_queried, int)
        or not isinstance(slots_total, int)
        or not isinstance(slots_queried, int)
    ):
        raise MarkedAccountError(
            f"marked-account snapshot {snapshot_path} does not record the archives "
            "it covers, when it was cut, or how much of them it asked about"
        )
    return MarkedAccounts(
        covers_archives=tuple(covers_archives),
        queried_at=queried_at,
        accounts_total=accounts_total,
        accounts_queried=accounts_queried,
        slots_total=slots_total,
        slots_queried=slots_queried,
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
            f"the marked-account snapshot path {configured} is relative but the "
            "configuration was not loaded from a file, so there is nothing to "
            "resolve it against"
        )
    return Path(config_source).parent / configured
