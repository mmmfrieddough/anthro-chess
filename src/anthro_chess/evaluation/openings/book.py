"""The owned versioned opening book and its position index.

The book is checked into the package rather than read from a source export.
Source ``ECO`` and ``Opening`` headers are deliberately not captured into the
normalized schema: their granularity is fixed, databases disagree on assignment
through transpositions, and the name strings differ per source. Owning the book
and the matching procedure means a label means one thing regardless of where a
game came from.

Entries are indexed by position rather than by move sequence, so two move
orders reaching the same position resolve to the same opening.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from importlib.resources import files
from typing import Any

import chess

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

    def identity(self) -> dict[str, object]:
        """Return the record artifacts carry alongside opening labels."""

        return {
            "name": self.name,
            "version": self.version,
            "entries": len(self.entries),
            "sha256": self.content_sha256,
        }

    def entry_for(self, epd: str) -> OpeningEntry | None:
        """Return the book entry naming a position, if the book names it."""

        return self.positions.get(epd)


@cache
def load_book() -> OpeningBook:
    """Load and index the vendored book, once per process.

    Indexing replays every entry's moves to derive its position key, so the
    checked-in file stays a readable list of names and moves that can be
    audited against upstream instead of a table of opaque digests.
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

    entries: list[OpeningEntry] = []
    positions: dict[str, OpeningEntry] = {}
    for entry in _parse_entries(content):
        previous = positions.get(entry.epd)
        if previous is not None:
            raise OpeningBookError(
                "the opening book names one position twice: "
                f"{previous.name!r} and {entry.name!r}"
            )
        positions[entry.epd] = entry
        entries.append(entry)

    if not entries:
        raise OpeningBookError(f"{BOOK_FILE_NAME} contains no openings")
    recorded = metadata.get("entries")
    if recorded != len(entries):
        raise OpeningBookError(
            f"{BOOK_METADATA_FILE_NAME} records {recorded} opening(s) but "
            f"{BOOK_FILE_NAME} contains {len(entries)}"
        )

    return OpeningBook(
        name=str(metadata["name"]),
        version=int(metadata["version"]),
        content_sha256=observed,
        source=dict(metadata["source"]),
        license=dict(metadata["license"]),
        entries=tuple(entries),
        positions=positions,
        maximum_ply=max(entry.ply for entry in entries),
    )


def opening_book_identity() -> dict[str, object]:
    """Return the book identity for artifacts that carry opening labels."""

    return load_book().identity()


def _parse_entries(content: str) -> Iterator[OpeningEntry]:
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
            plies += 1
        yield OpeningEntry(
            eco=(row["eco"] or "").strip(),
            name=name,
            uci=uci,
            epd=board.epd(),
            ply=plies,
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
