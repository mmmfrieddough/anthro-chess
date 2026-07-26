#!/usr/bin/env python3
"""Regenerate the vendored opening book from its pinned upstream commit.

This is a maintenance script, not part of the runtime package. It runs only
when the book is deliberately updated, which is also when the book version is
bumped, so classification never changes underneath a recorded benchmark result.

Usage:

    scripts/vendor-opening-book.py --commit <upstream-sha> --version <n>

The output is the checked-in book data under the openings package: a canonical
TSV of opening names with their moves in UCI, plus the identity and license
record that artifacts carry alongside opening labels.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import urllib.request
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path

import chess

REPOSITORY = "https://github.com/lichess-org/chess-openings"
RAW_URL = "https://raw.githubusercontent.com/lichess-org/chess-openings/{commit}/{name}"
SOURCE_FILES = ("a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv")
LICENSE = {
    "spdx_id": "CC0-1.0",
    "name": "Creative Commons Zero v1.0 Universal",
    "url": "https://github.com/lichess-org/chess-openings/blob/master/COPYING.txt",
    "attribution": (
        "Opening names aggregated by the Lichess chess-openings project and "
        "dedicated to the public domain under CC0 1.0. Move sequences were "
        "converted from SAN to UCI; no names were changed."
    ),
}

DATA_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "anthro_chess"
    / "evaluation"
    / "openings"
    / "data"
)
MOVE_NUMBER = re.compile(r"^\d+\.+$")


def main(argv: list[str] | None = None) -> int:
    """Download, convert, and write the vendored opening book."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit",
        required=True,
        help="Upstream commit to pin, so a rebuild reproduces this book exactly.",
    )
    parser.add_argument(
        "--version",
        required=True,
        type=int,
        help="Book version to record. Bump it whenever the content changes.",
    )
    parser.add_argument(
        "--retrieved",
        required=True,
        help="ISO date the upstream content was retrieved.",
    )
    arguments = parser.parse_args(argv)

    rows = sorted(
        _converted_rows(arguments.commit),
        key=lambda row: (row[0], row[1], row[2]),
    )
    _reject_ambiguous_positions(rows)

    book_path = DATA_DIRECTORY / "openings.tsv"
    book_path.write_text(_render_tsv(rows), encoding="utf-8", newline="\n")
    metadata = {
        "name": "anthro-openings",
        "version": arguments.version,
        "entries": len(rows),
        "openings_sha256": sha256(book_path.read_bytes()).hexdigest(),
        "source": {
            "repository": REPOSITORY,
            "commit": arguments.commit,
            "retrieved": arguments.retrieved,
            "files": list(SOURCE_FILES),
        },
        "license": LICENSE,
    }
    metadata_path = DATA_DIRECTORY / "book.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {len(rows)} opening(s) to {book_path}")
    print(f"Wrote identity and license record to {metadata_path}")
    return 0


def _converted_rows(commit: str) -> Iterator[tuple[str, str, str]]:
    for name in SOURCE_FILES:
        url = RAW_URL.format(commit=commit, name=name)
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode("utf-8")
        for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
            yield row["eco"], row["name"], _uci_moves(row["name"], row["pgn"])


def _uci_moves(name: str, pgn: str) -> str:
    board = chess.Board()
    moves: list[str] = []
    for token in pgn.split():
        if MOVE_NUMBER.match(token):
            continue
        try:
            move = board.parse_san(token)
        except ValueError as error:
            raise SystemExit(f"{name}: cannot parse {token!r}: {error}") from error
        moves.append(move.uci())
        board.push(move)
    if not moves:
        raise SystemExit(f"{name}: has no moves")
    return " ".join(moves)


def _reject_ambiguous_positions(rows: list[tuple[str, str, str]]) -> None:
    """Fail rather than pick a winner when two names share one position.

    Classification matches positions, so two entries reaching the same position
    would make the label depend on iteration order.
    """

    seen: dict[str, str] = {}
    for _, name, uci in rows:
        board = chess.Board()
        for move in uci.split():
            board.push(chess.Move.from_uci(move))
        epd = board.epd()
        previous = seen.get(epd)
        if previous is not None:
            raise SystemExit(
                f"upstream book maps one position to both {previous!r} and {name!r}"
            )
        seen[epd] = name


def _render_tsv(rows: list[tuple[str, str, str]]) -> str:
    lines = ["\t".join(("eco", "name", "uci"))]
    lines.extend("\t".join(row) for row in rows)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
