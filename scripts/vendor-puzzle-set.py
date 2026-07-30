#!/usr/bin/env python3
"""Regenerate the owned puzzle set from a pinned Lichess database export.

The cumulative upstream export is intentionally not checked in. This script
verifies its digest, applies declared quality bounds, and keeps the lowest
SHA-256-ranked puzzle ids in each rating band. The resulting compact set is
stable across input order and auditable against the recorded source export.

Usage:

    scripts/vendor-puzzle-set.py \
      --source /path/to/lichess_db_puzzle.csv.zst \
      --source-sha256 <sha256> \
      --version 1 \
      --retrieved YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

import zstandard

SOURCE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
LICENSE = {
    "spdx_id": "CC0-1.0",
    "name": "Creative Commons Zero v1.0 Universal",
    "url": "https://database.lichess.org/",
    "attribution": (
        "Puzzles produced by Lichess and released as part of its open database "
        "under CC0 1.0."
    ),
}
DATA_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "anthro_chess"
    / "evaluation"
    / "puzzles"
    / "data"
)
OUTPUT_COLUMNS = (
    "puzzle_id",
    "initial_fen",
    "moves",
    "rating",
    "source_game_key",
)


@dataclass(frozen=True)
class SelectedPuzzle:
    """The fields retained from one upstream row."""

    rank: str
    puzzle_id: str
    initial_fen: str
    moves: str
    rating: int
    source_game_key: str

    def output_row(self) -> tuple[str, ...]:
        return (
            self.puzzle_id,
            self.initial_fen,
            self.moves,
            str(self.rating),
            self.source_game_key,
        )


def main(argv: list[str] | None = None) -> int:
    """Select, validate, and write the owned puzzle set."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--retrieved", required=True)
    parser.add_argument("--minimum-rating", type=int, default=800)
    parser.add_argument("--maximum-rating", type=int, default=2800)
    parser.add_argument("--band-width", type=int, default=400)
    parser.add_argument("--per-band", type=int, default=64)
    parser.add_argument("--minimum-plays", type=int, default=100)
    parser.add_argument("--minimum-popularity", type=int, default=0)
    arguments = parser.parse_args(argv)
    _validate_arguments(arguments)

    observed_source_sha256 = _file_sha256(arguments.source)
    if observed_source_sha256 != arguments.source_sha256.lower():
        raise SystemExit(
            "source checksum mismatch: expected "
            f"{arguments.source_sha256.lower()}, observed {observed_source_sha256}"
        )

    selected = _select(arguments)
    expected_bands = range(
        arguments.minimum_rating,
        arguments.maximum_rating,
        arguments.band_width,
    )
    short = [
        start
        for start in expected_bands
        if len(selected.get(start, ())) != arguments.per_band
    ]
    if short:
        raise SystemExit(
            "source did not fill rating band(s): "
            + ", ".join(str(start) for start in short)
        )
    rows = sorted(
        (puzzle for band in selected.values() for puzzle in band),
        key=lambda puzzle: puzzle.puzzle_id,
    )

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    puzzle_path = DATA_DIRECTORY / "puzzles.csv"
    puzzle_path.write_text(_render_csv(rows), encoding="utf-8", newline="\n")
    metadata = {
        "name": "anthro-lichess-puzzles",
        "version": arguments.version,
        "entries": len(rows),
        "puzzles_sha256": sha256(puzzle_path.read_bytes()).hexdigest(),
        "source": {
            "url": SOURCE_URL,
            "retrieved": arguments.retrieved,
            "sha256": observed_source_sha256,
            "format": "lichess-puzzle-csv-v1",
        },
        "license": LICENSE,
        "selection": {
            "algorithm": "sha256-rank-by-rating-band-v1",
            "minimum_rating": arguments.minimum_rating,
            "maximum_rating_exclusive": arguments.maximum_rating,
            "band_width": arguments.band_width,
            "per_band": arguments.per_band,
            "minimum_plays": arguments.minimum_plays,
            "minimum_popularity": arguments.minimum_popularity,
        },
    }
    metadata_path = DATA_DIRECTORY / "set.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {len(rows)} puzzle(s) to {puzzle_path}")
    print(f"Wrote identity and license record to {metadata_path}")
    return 0


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.version < 1:
        raise SystemExit("version must be positive")
    for name in ("band_width", "per_band", "minimum_plays"):
        if getattr(arguments, name) < 1:
            raise SystemExit(f"{name.replace('_', '-')} must be positive")
    if arguments.maximum_rating <= arguments.minimum_rating:
        raise SystemExit("maximum-rating must be greater than minimum-rating")
    if (arguments.maximum_rating - arguments.minimum_rating) % arguments.band_width:
        raise SystemExit("the rating range must divide evenly into bands")


def _select(arguments: argparse.Namespace) -> dict[int, list[SelectedPuzzle]]:
    bands: dict[int, list[SelectedPuzzle]] = {}
    for row in _source_rows(arguments.source):
        try:
            rating = int(row["Rating"])
            plays = int(row["NbPlays"])
            popularity = int(row["Popularity"])
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"invalid upstream puzzle row: {error}") from error
        if not (
            arguments.minimum_rating <= rating < arguments.maximum_rating
            and plays >= arguments.minimum_plays
            and popularity >= arguments.minimum_popularity
        ):
            continue
        puzzle_id = row["PuzzleId"].strip()
        source_game_key = _source_game_key(row["GameUrl"])
        if not puzzle_id or source_game_key is None:
            continue
        start = (
            (rating - arguments.minimum_rating) // arguments.band_width
        ) * arguments.band_width + arguments.minimum_rating
        candidate = SelectedPuzzle(
            rank=sha256(puzzle_id.encode()).hexdigest(),
            puzzle_id=puzzle_id,
            initial_fen=row["FEN"],
            moves=row["Moves"],
            rating=rating,
            source_game_key=source_game_key,
        )
        band = bands.setdefault(start, [])
        band.append(candidate)
        band.sort(key=lambda puzzle: (puzzle.rank, puzzle.puzzle_id))
        del band[arguments.per_band :]
    return bands


def _source_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
            yield from csv.DictReader(text)


def _source_game_key(url: str) -> str | None:
    parts = [part for part in urlparse(url).path.split("/") if part]
    if not parts:
        return None
    key = parts[0]
    return key if len(key) == 8 else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_csv(rows: list[SelectedPuzzle]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(OUTPUT_COLUMNS)
    writer.writerows(puzzle.output_row() for puzzle in rows)
    return buffer.getvalue()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
