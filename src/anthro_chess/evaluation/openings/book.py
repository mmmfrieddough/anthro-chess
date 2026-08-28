"""The owned versioned opening book and its position index.

The book is checked into the package rather than read from a source export.
Source ``ECO`` and ``Opening`` headers are deliberately not captured into the
normalized schema: their granularity is fixed, databases disagree on assignment
through transpositions, and the name strings differ per source. Owning the book
and the matching procedure means a label means one thing regardless of where a
game came from.

Entries are indexed by position rather than by move sequence, so two move
orders reaching the same position resolve to the same opening.

Loading also derives a **continuation index**: for every named position, how
deep the book still runs past it and how many distinct labels remain reachable
from it. That second number is what separates a destination from a waypoint
without a curated list, since a position many named openings pass through has
many labels still reachable and one that has been committed to has one.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from importlib.resources import files
from typing import Any

import chess

from anthro_chess.evaluation.openings.names import OpeningLevel, opening_levels

BOOK_FILE_NAME = "openings.tsv"
BOOK_METADATA_FILE_NAME = "book.json"
_DATA_PACKAGE = "anthro_chess.evaluation.openings"


class OpeningBookError(ValueError):
    """Raised when the vendored opening book cannot be loaded or trusted."""


@dataclass(frozen=True)
class OpeningEntry:
    """One named opening, identified by the position its moves reach."""

    eco: str
    name: str
    uci: str
    epd: str
    ply: int


@dataclass(frozen=True)
class OpeningContinuation:
    """What the book still offers from one named position.

    Reachability is positional, exactly as matching is: an entry is reachable
    from a position when that position lies on the entry's own move path, no
    matter which move order got there.
    """

    #: Deepest book ply any reachable entry sits at, in book coordinates. Never
    #: below the position's own ply, since a position reaches itself.
    deepest_ply: int
    #: Distinct labels reachable onward, one count per granularity level. More
    #: than one means the position has not committed to a label at that level
    #: and is therefore a waypoint rather than a destination.
    reachable_family_labels: int
    reachable_variation_labels: int
    reachable_line_labels: int

    def reachable_labels(self, level: OpeningLevel) -> int:
        """Return how many distinct labels remain reachable at one level."""

        if level is OpeningLevel.FAMILY:
            return self.reachable_family_labels
        if level is OpeningLevel.VARIATION:
            return self.reachable_variation_labels
        return self.reachable_line_labels

    def waypoint(self, level: OpeningLevel) -> bool:
        """Return whether this position is a waypoint at one level.

        A position several labels still pass through has not been chosen; a
        game that stops there says something about how far it stayed in book
        rather than about which opening it preferred.
        """

        return self.reachable_labels(level) > 1

    def as_record(self) -> dict[str, object]:
        """Return the stored form of one position's continuations."""

        record: dict[str, object] = {"deepest_ply": self.deepest_ply}
        for level in OpeningLevel:
            record[f"reachable_{level.value}_labels"] = self.reachable_labels(level)
        return record


@dataclass(frozen=True)
class OpeningBook:
    """A loaded book plus the position index classification matches against."""

    name: str
    version: int
    content_sha256: str
    source: Mapping[str, Any]
    license: Mapping[str, Any]
    entries: tuple[OpeningEntry, ...]
    positions: Mapping[str, OpeningEntry]
    maximum_ply: int
    #: Derived at load, keyed by the same position strings as ``positions``.
    #: Only named positions need one: they are the only positions a game can
    #: be labeled with, and therefore the only ones a depth or waypoint
    #: reading is ever taken at.
    continuations: Mapping[str, OpeningContinuation]
    #: The same named positions under the library's own position key. An EPD is
    #: a string built by scanning all sixty-four squares, and classification
    #: takes one per ply of every game it reads, which made rendering positions
    #: the largest cost in the evaluation suite. The repetition counter keys
    #: positions the same way, so a board identical for a threefold claim is
    #: identical here.
    keyed: Mapping[Hashable, OpeningEntry]

    def identity(self) -> dict[str, object]:
        """Return the record artifacts carry alongside opening labels."""

        return {
            "name": self.name,
            "version": self.version,
            "entries": len(self.entries),
            "sha256": self.content_sha256,
        }

    def named_entry(self, board: chess.Board) -> OpeningEntry | None:
        """Return the entry naming this board, when the book names it.

        The lookup classification runs at every ply, which is why it takes the
        board rather than a rendering of it: a caller that never reaches a named
        position never builds a string at all.
        """

        return self.keyed.get(board._transposition_key())

    def continuation_for(self, epd: str) -> OpeningContinuation | None:
        """Return what the book still offers from a named position."""

        return self.continuations.get(epd)

    @classmethod
    def indexed(
        cls,
        *,
        name: str,
        version: int,
        content_sha256: str,
        source: Mapping[str, Any],
        license: Mapping[str, Any],
        entries: Sequence[OpeningEntry],
        paths: Sequence[Sequence[str]] | None = None,
    ) -> OpeningBook:
        """Build a book, deriving both indexes from its entries.

        The only constructor callers should use, so a book assembled in a test
        and the vendored one are indexed by the same code and cannot disagree
        about what a waypoint is.

        ``paths`` is the position each entry's moves passed through, which the
        loader already has from parsing. Omitting it replays the entries, which
        is what a caller building a book by hand wants and costs about a second
        over the vendored book.
        """

        if not entries:
            raise OpeningBookError("an opening book needs at least one opening")
        positions: dict[str, OpeningEntry] = {}
        for entry in entries:
            previous = positions.get(entry.epd)
            if previous is not None:
                raise OpeningBookError(
                    "the opening book names one position twice: "
                    f"{previous.name!r} and {entry.name!r}"
                )
            positions[entry.epd] = entry
        resolved = _entry_paths(entries) if paths is None else paths
        return cls(
            name=name,
            version=version,
            content_sha256=content_sha256,
            source=source,
            license=license,
            entries=tuple(entries),
            positions=positions,
            maximum_ply=max(entry.ply for entry in entries),
            continuations=_continuations(entries, resolved, positions),
            keyed=_keyed(positions),
        )


def _keyed(
    positions: Mapping[str, OpeningEntry],
) -> dict[Hashable, OpeningEntry]:
    """Key every named position the way classification will look one up.

    Through the same function on both sides, so the index cannot disagree with
    the position strings it stands for about which entry names a board.

    The collision this refuses is the one that makes the second index unsafe:
    two positions the book names separately reaching one key would silently give
    a game the other one's opening.
    """

    keyed: dict[Hashable, OpeningEntry] = {}
    for epd, entry in positions.items():
        board = chess.Board()
        board.set_epd(epd)
        key = board._transposition_key()
        previous = keyed.get(key)
        if previous is not None:
            raise OpeningBookError(
                "two named positions share one key, so the book cannot be "
                f"matched against: {previous.epd!r} and {epd!r}"
            )
        keyed[key] = entry
    return keyed


@cache
def load_book() -> OpeningBook:
    """Load and index the vendored book, once per process.

    Indexing replays every entry's moves to derive its position key, so the
    checked-in file stays a readable list of names and moves that can be
    audited against upstream instead of a table of opaque digests.

    The continuation index needs a key at every position an entry passes
    through rather than only at its last, which is most of the second or so
    this costs. Paid once per process behind the cache, against benchmark runs
    measured in minutes, so it buys the waypoint and depth readings for
    nothing that shows up in a measurement.
    """

    metadata = _load_metadata()
    content = _read_data_file(BOOK_FILE_NAME)
    observed = sha256(content.encode("utf-8")).hexdigest()
    expected = metadata.get("openings_sha256")
    if observed != expected:
        raise OpeningBookError(
            f"{BOOK_FILE_NAME} does not match the checksum recorded in "
            f"{BOOK_METADATA_FILE_NAME}: expected {expected}, observed {observed}"
        )

    parsed = list(_parse_entries(content))
    if not parsed:
        raise OpeningBookError(f"{BOOK_FILE_NAME} contains no openings")
    recorded = metadata.get("entries")
    if recorded != len(parsed):
        raise OpeningBookError(
            f"{BOOK_METADATA_FILE_NAME} records {recorded} opening(s) but "
            f"{BOOK_FILE_NAME} contains {len(parsed)}"
        )

    return OpeningBook.indexed(
        name=str(metadata["name"]),
        version=int(metadata["version"]),
        content_sha256=observed,
        source=dict(metadata["source"]),
        license=dict(metadata["license"]),
        entries=[entry for entry, _ in parsed],
        paths=[path for _, path in parsed],
    )


def _entry_paths(entries: Sequence[OpeningEntry]) -> list[tuple[str, ...]]:
    """Replay every entry to recover the positions its moves pass through."""

    paths: list[tuple[str, ...]] = []
    for entry in entries:
        board = chess.Board()
        path = [board.epd()]
        for token in entry.uci.split():
            board.push(chess.Move.from_uci(token))
            path.append(board.epd())
        paths.append(tuple(path))
    return paths


def _continuations(
    entries: Sequence[OpeningEntry],
    paths: Sequence[Sequence[str]],
    positions: Mapping[str, OpeningEntry],
) -> dict[str, OpeningContinuation]:
    """Index what the book still offers from every named position.

    One pass over the paths the parse already replayed, so the whole index
    costs a dictionary lookup per visited position — roughly thirty-seven
    thousand of them — rather than a second traversal of the book.

    Only named positions are kept. An unnamed position on the way to an entry
    can never be a game's label, so nothing would ever read its continuations,
    and dropping them keeps the index the size of the book rather than the size
    of its move tree.
    """

    labels: dict[str, tuple[set[str], set[str], set[str]]] = {}
    deepest: dict[str, int] = {}
    for entry, path in zip(entries, paths, strict=True):
        names = opening_levels(entry.name)
        for epd in path:
            if epd not in positions:
                continue
            reachable = labels.get(epd)
            if reachable is None:
                reachable = (set(), set(), set())
                labels[epd] = reachable
            for level_labels, name in zip(reachable, names, strict=True):
                level_labels.add(name)
            if entry.ply > deepest.get(epd, 0):
                deepest[epd] = entry.ply
    return {
        epd: OpeningContinuation(
            deepest_ply=deepest[epd],
            reachable_family_labels=len(families),
            reachable_variation_labels=len(variations),
            reachable_line_labels=len(lines),
        )
        for epd, (families, variations, lines) in labels.items()
    }


def opening_book_identity() -> dict[str, object]:
    """Return the book identity for artifacts that carry opening labels."""

    return load_book().identity()


def _parse_entries(content: str) -> Iterator[tuple[OpeningEntry, tuple[str, ...]]]:
    """Yield each entry together with every position its moves pass through.

    The path is yielded rather than stored on the entry: the continuation index
    is the only thing that needs it, and retaining thirty-seven thousand
    position strings for the life of the process to save one replay is a poor
    trade.
    """

    reader = csv.DictReader(content.splitlines(), delimiter="\t")
    if reader.fieldnames != ["eco", "name", "uci"]:
        raise OpeningBookError(
            f"{BOOK_FILE_NAME} must have eco, name, and uci columns; "
            f"found {reader.fieldnames}"
        )
    for number, row in enumerate(reader, start=2):
        name = (row["name"] or "").strip()
        uci = (row["uci"] or "").strip()
        if not name or not uci:
            raise OpeningBookError(
                f"{BOOK_FILE_NAME} line {number} has an empty name or move list"
            )
        board = chess.Board()
        path = [board.epd()]
        plies = 0
        for token in uci.split():
            try:
                move = chess.Move.from_uci(token)
            except ValueError as error:
                raise OpeningBookError(
                    f"{BOOK_FILE_NAME} line {number} has an unreadable move "
                    f"{token!r}: {error}"
                ) from error
            if not board.is_legal(move):
                raise OpeningBookError(
                    f"{BOOK_FILE_NAME} line {number} plays the illegal move {token!r}"
                )
            board.push(move)
            path.append(board.epd())
            plies += 1
        yield (
            OpeningEntry(
                eco=(row["eco"] or "").strip(),
                name=name,
                uci=uci,
                epd=board.epd(),
                ply=plies,
            ),
            tuple(path),
        )


def _load_metadata() -> Mapping[str, Any]:
    metadata = json.loads(_read_data_file(BOOK_METADATA_FILE_NAME))
    if not isinstance(metadata, dict):
        raise OpeningBookError(f"{BOOK_METADATA_FILE_NAME} must be a JSON object")
    missing = sorted(
        {"name", "version", "entries", "openings_sha256", "source", "license"}
        - set(metadata)
    )
    if missing:
        raise OpeningBookError(
            f"{BOOK_METADATA_FILE_NAME} is missing {', '.join(missing)}"
        )
    return metadata


def _read_data_file(name: str) -> str:
    resource = files(_DATA_PACKAGE).joinpath("data", name)
    try:
        return resource.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise OpeningBookError(
            f"the packaged opening book is missing {name}"
        ) from error
