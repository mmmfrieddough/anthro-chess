"""Build, load, and validate the external Lichess puzzle benchmark artifact."""

from __future__ import annotations

import csv
import heapq
import io
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import chess
from pydantic import Field, StrictInt, model_validator

from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import (
    ArchiveConfig,
    DataPreparationError,
    acquire_configured_archive,
)
from anthro_chess.data.artifacts import file_sha256

PUZZLE_FILE_NAME = "puzzles.csv"
PUZZLE_METADATA_FILE_NAME = "manifest.json"
PUZZLE_SELECTION_ALGORITHM = "sha256-rank-per-exact-rating-v1"
PUZZLE_SIZING_METHOD = "two-independent-proportions-worst-case-v1"
PUZZLE_DETECTION_CONFIDENCE = 0.95
PUZZLE_DETECTION_POWER = 0.80
PUZZLE_LICENSE = {
    "name": "Creative Commons Zero v1.0 Universal",
    "spdx_id": "CC0-1.0",
    "url": "https://database.lichess.org/",
    "attribution": (
        "Puzzles produced by Lichess and released as part of its open database "
        "under CC0 1.0."
    ),
}
_OUTPUT_COLUMNS = (
    "puzzle_id",
    "initial_fen",
    "moves",
    "rating",
    "source_game_key",
)
_NORMAL_95 = 1.959963984540054
_NORMAL_80_POWER = 0.8416212335729143


class PuzzleSetError(ValueError):
    """Raised when a puzzle artifact cannot be built, loaded, or trusted."""


class PuzzleSelectionConfig(ConfigModel):
    """The frozen population and uniform rating design of one puzzle set."""

    minimum_rating: StrictInt = Field(default=800, ge=0)
    maximum_rating_exclusive: StrictInt = Field(default=2800, ge=1)
    puzzles_per_rating: StrictInt = Field(default=10, ge=1)
    minimum_plays: StrictInt = Field(default=100, ge=1)
    minimum_popularity: StrictInt = Field(default=0)
    maximum_rating_deviation: StrictInt = Field(default=100, ge=1)
    local_precision_span: StrictInt = Field(default=400, ge=1)

    @model_validator(mode="after")
    def _validate_range(self) -> PuzzleSelectionConfig:
        if self.maximum_rating_exclusive <= self.minimum_rating:
            raise ValueError(
                "maximum_rating_exclusive must be greater than minimum_rating"
            )
        if self.local_precision_span > (
            self.maximum_rating_exclusive - self.minimum_rating
        ):
            raise ValueError("local_precision_span cannot exceed the rating range")
        return self


class PuzzleSetBuildConfig(ConfigModel):
    """Code-owned schema for ``anthro eval prepare-puzzles``."""

    artifact_name: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: StrictInt = Field(ge=1)
    source_retrieved: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    expected_entries: StrictInt = Field(ge=1)
    expected_puzzles_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    archive: ArchiveConfig
    selection: PuzzleSelectionConfig = PuzzleSelectionConfig()

    @model_validator(mode="after")
    def _validate_expected_size(self) -> PuzzleSetBuildConfig:
        selection = self.selection
        ratings = selection.maximum_rating_exclusive - selection.minimum_rating
        expected = ratings * selection.puzzles_per_rating
        if self.expected_entries != expected:
            raise ValueError(
                f"expected_entries must equal rating count times "
                f"puzzles_per_rating ({expected})"
            )
        return self


@dataclass(frozen=True)
class Puzzle:
    """One puzzle rooted before the opponent's setup move."""

    puzzle_id: str
    initial_fen: str
    moves: tuple[chess.Move, ...]
    rating: int
    source_game_key: str

    @property
    def solution_moves(self) -> tuple[chess.Move, ...]:
        """Return the player moves, excluding the opponent's forced replies."""

        return self.moves[1::2]

    def as_projection_record(self) -> dict[str, object]:
        """Return the stable content the benchmark series fingerprints."""

        return {
            "game_id": _puzzle_numeric_id(self.puzzle_id),
            "puzzle_id": self.puzzle_id,
            "initial_fen": self.initial_fen,
            "moves": [move.uci() for move in self.moves],
            "rating": self.rating,
            "source_game_key": self.source_game_key,
        }


@dataclass(frozen=True)
class PuzzleSet:
    """A generated puzzle artifact plus its external-source provenance."""

    name: str
    version: int
    content_sha256: str
    source: Mapping[str, Any]
    license: Mapping[str, Any]
    selection: Mapping[str, Any]
    sizing: Mapping[str, Any]
    coverage: Mapping[str, Any]
    puzzles: tuple[Puzzle, ...]

    def identity(self) -> dict[str, object]:
        """Return the set identity carried by benchmark artifacts."""

        return {
            "name": self.name,
            "version": self.version,
            "entries": len(self.puzzles),
            "sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class PuzzleSelection:
    """The puzzles one reading scores, and the spec that reproduces them."""

    puzzles: tuple[Puzzle, ...]
    puzzles_per_rating: int | None
    eligible_puzzles: int

    @property
    def selected_puzzles(self) -> int:
        """Return how many puzzles survived subsampling."""

        return len(self.puzzles)

    @property
    def subsampled(self) -> bool:
        """Return whether the dial actually dropped anything.

        A dial set at or above what every rating holds selected the whole set,
        so it is that reading rather than one of its own.
        """

        return self.selected_puzzles < self.eligible_puzzles

    @property
    def minimum_detectable_difference(self) -> float:
        """Return the smallest rate difference this many puzzles can resolve."""

        return conservative_detectable_difference(self.selected_puzzles)

    def as_record(self) -> dict[str, object]:
        """Return the stable spec record stored in benchmark artifacts."""

        return {
            "algorithm": PUZZLE_SELECTION_ALGORITHM,
            "puzzles_per_rating": self.puzzles_per_rating,
            "selected_puzzles": self.selected_puzzles,
            "eligible_puzzles": self.eligible_puzzles,
        }


@dataclass(frozen=True)
class PuzzleSetBuildResult:
    """The generated puzzle artifact and the verified raw source behind it."""

    artifact_path: Path
    puzzle_path: Path
    manifest_path: Path
    source_path: Path
    source_reused: bool
    entries: int
    puzzles_sha256: str


@dataclass(frozen=True)
class _SelectedPuzzle:
    rank: int
    puzzle_id: str
    initial_fen: str
    moves: str
    rating: int
    rating_deviation: int
    popularity: int
    plays: int
    source_game_key: str

    def output_row(self) -> tuple[str, ...]:
        return (
            self.puzzle_id,
            self.initial_fen,
            self.moves,
            str(self.rating),
            self.source_game_key,
        )


def prepare_puzzle_set(
    resolved_config: ResolvedConfig[PuzzleSetBuildConfig],
    output_directory: str | Path,
    *,
    source_path: str | Path | None = None,
) -> PuzzleSetBuildResult:
    """Build a deterministic external puzzle artifact from a pinned export."""

    config = resolved_config.value
    output = Path(output_directory)
    try:
        if source_path is None:
            acquired = acquire_configured_archive(output, config.archive)
            source = acquired.archive_path
            source_reused = acquired.reused
        else:
            source = Path(source_path)
            observed = file_sha256(source)
            if observed != config.archive.sha256:
                raise PuzzleSetError(
                    "input puzzle archive checksum mismatch: expected "
                    f"{config.archive.sha256}, observed {observed}"
                )
            source_reused = True
        selected, coverage = _select_puzzles(source, config.selection)
        _validate_selected(selected, config.selection)
        rendered = _render_csv(selected)
        observed_puzzles_sha256 = sha256(rendered.encode()).hexdigest()
        if observed_puzzles_sha256 != config.expected_puzzles_sha256:
            raise PuzzleSetError(
                "selected puzzle checksum mismatch: expected "
                f"{config.expected_puzzles_sha256}, "
                f"observed {observed_puzzles_sha256}"
            )

        output.mkdir(parents=True, exist_ok=True)
        puzzle_path = output / PUZZLE_FILE_NAME
        manifest_path = output / PUZZLE_METADATA_FILE_NAME
        _atomic_write_text(puzzle_path, rendered)
        metadata = {
            "name": config.name,
            "version": config.version,
            "entries": len(selected),
            "puzzles_sha256": observed_puzzles_sha256,
            "source": {
                "url": config.archive.url,
                "file_name": config.archive.file_name,
                "retrieved": config.source_retrieved,
                "sha256": config.archive.sha256,
                "format": "lichess-puzzle-csv-v1",
            },
            "license": PUZZLE_LICENSE,
            "selection": {
                "algorithm": PUZZLE_SELECTION_ALGORITHM,
                **config.selection.model_dump(mode="json"),
            },
            "sizing": _sizing_record(config.selection),
            "coverage": coverage,
        }
        _atomic_write_text(
            manifest_path,
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )
        loaded = load_puzzle_set(output)
        if len(loaded.puzzles) != config.expected_entries:
            raise PuzzleSetError(
                f"prepared {len(loaded.puzzles)} puzzles, "
                f"expected {config.expected_entries}"
            )
    except (DataPreparationError, OSError, ValueError) as error:
        if isinstance(error, PuzzleSetError):
            raise
        raise PuzzleSetError(str(error)) from error
    return PuzzleSetBuildResult(
        artifact_path=output,
        puzzle_path=puzzle_path,
        manifest_path=manifest_path,
        source_path=source,
        source_reused=source_reused,
        entries=len(selected),
        puzzles_sha256=observed_puzzles_sha256,
    )


def load_puzzle_set(path: str | Path) -> PuzzleSet:
    """Load a generated puzzle set, verifying its identity and chess lines."""

    return _load_puzzle_set(Path(path).expanduser().resolve())


@cache
def _load_puzzle_set(path: Path) -> PuzzleSet:
    metadata = _load_metadata(path / PUZZLE_METADATA_FILE_NAME)
    puzzle_path = path / PUZZLE_FILE_NAME
    try:
        content = puzzle_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise PuzzleSetError(f"the puzzle artifact is missing {puzzle_path}") from error
    observed = sha256(content.encode()).hexdigest()
    expected = metadata.get("puzzles_sha256")
    if observed != expected:
        raise PuzzleSetError(
            f"{puzzle_path} does not match the checksum recorded in "
            f"{PUZZLE_METADATA_FILE_NAME}: expected {expected}, observed {observed}"
        )

    puzzles = tuple(_parse_puzzles(content, puzzle_path))
    if not puzzles:
        raise PuzzleSetError(f"{puzzle_path} contains no puzzles")
    if metadata.get("entries") != len(puzzles):
        raise PuzzleSetError(
            f"{PUZZLE_METADATA_FILE_NAME} records {metadata.get('entries')} "
            f"puzzle(s) but {puzzle_path} contains {len(puzzles)}"
        )
    identifiers = [puzzle.puzzle_id for puzzle in puzzles]
    if identifiers != sorted(identifiers) or len(set(identifiers)) != len(identifiers):
        raise PuzzleSetError("puzzles must have unique identifiers in sorted order")
    source_games = [puzzle.source_game_key for puzzle in puzzles]
    if len(set(source_games)) != len(source_games):
        raise PuzzleSetError("puzzles must come from distinct source games")

    return PuzzleSet(
        name=str(metadata["name"]),
        version=int(metadata["version"]),
        content_sha256=observed,
        source=dict(metadata["source"]),
        license=dict(metadata["license"]),
        selection=dict(metadata["selection"]),
        sizing=dict(metadata["sizing"]),
        coverage=dict(metadata["coverage"]),
        puzzles=puzzles,
    )


def puzzle_set_identity(path: str | Path) -> dict[str, object]:
    """Return the identity for artifacts that carry puzzle results."""

    return load_puzzle_set(path).identity()


def select_puzzles(
    puzzle_set: PuzzleSet,
    puzzles_per_rating: int | None,
) -> PuzzleSelection:
    """Keep the lowest-ranked puzzles at each exact rating, as the build does.

    Ranking by the same hash within the same strata makes a subsample the set a
    build with this ``puzzles_per_rating`` would have written: the design stays
    uniform over exact ratings, and smaller readings nest inside larger ones on
    any machine.
    """

    kept = puzzle_set.puzzles
    if puzzles_per_rating is not None:
        by_rating: dict[int, list[Puzzle]] = {}
        for puzzle in puzzle_set.puzzles:
            by_rating.setdefault(puzzle.rating, []).append(puzzle)
        kept = tuple(
            sorted(
                (
                    puzzle
                    for group in by_rating.values()
                    for puzzle in sorted(
                        group, key=lambda item: _puzzle_rank(item.puzzle_id)
                    )[:puzzles_per_rating]
                ),
                key=lambda puzzle: puzzle.puzzle_id,
            )
        )
    return PuzzleSelection(
        puzzles=kept,
        puzzles_per_rating=puzzles_per_rating,
        eligible_puzzles=len(puzzle_set.puzzles),
    )


def conservative_detectable_difference(sample_size: int) -> float:
    """Return the 95%-confidence, 80%-power worst-case two-rate difference."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    standard_error = math.sqrt(2.0 * 0.25 / sample_size)
    return (_NORMAL_95 + _NORMAL_80_POWER) * standard_error


def _select_puzzles(
    source: Path,
    selection: PuzzleSelectionConfig,
) -> tuple[list[_SelectedPuzzle], dict[str, object]]:
    heaps: dict[int, list[tuple[int, str, _SelectedPuzzle]]] = {}
    candidate_counts: Counter[int] = Counter()
    accepted = 0
    try:
        import zstandard
    except ImportError as error:  # pragma: no cover - exercised by package extras
        raise PuzzleSetError(
            "preparing puzzle data requires the project's data dependencies"
        ) from error

    with source.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
            for row in csv.DictReader(text):
                candidate = _candidate(row, selection)
                if candidate is None:
                    continue
                accepted += 1
                candidate_counts[candidate.rating] += 1
                heap = heaps.setdefault(candidate.rating, [])
                item = (-candidate.rank, candidate.puzzle_id, candidate)
                if len(heap) < selection.puzzles_per_rating:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)

    selected = sorted(
        (
            item[2]
            for rating in range(
                selection.minimum_rating,
                selection.maximum_rating_exclusive,
            )
            for item in heaps.get(rating, ())
        ),
        key=lambda puzzle: puzzle.puzzle_id,
    )
    selected_ratings = Counter(puzzle.rating for puzzle in selected)
    selected_counts = list(selected_ratings.values())
    rating_counts = list(candidate_counts.values())
    coverage: dict[str, object] = {
        "eligible_candidates": accepted,
        "eligible_ratings": len(candidate_counts),
        "minimum_candidates_per_rating": min(rating_counts, default=0),
        "median_candidates_per_rating": (
            statistics.median(rating_counts) if rating_counts else 0
        ),
        "maximum_candidates_per_rating": max(rating_counts, default=0),
        "selected_source_games": len({puzzle.source_game_key for puzzle in selected}),
        "selected_ratings": len(selected_ratings),
        "minimum_selected_per_rating": min(selected_counts, default=0),
        "maximum_selected_per_rating": max(selected_counts, default=0),
    }
    return selected, coverage


def _candidate(
    row: Mapping[str, str],
    selection: PuzzleSelectionConfig,
) -> _SelectedPuzzle | None:
    try:
        rating = int(row["Rating"])
        rating_deviation = int(row["RatingDeviation"])
        popularity = int(row["Popularity"])
        plays = int(row["NbPlays"])
    except (KeyError, TypeError, ValueError) as error:
        raise PuzzleSetError(f"invalid upstream puzzle row: {error}") from error
    if not (
        selection.minimum_rating <= rating < selection.maximum_rating_exclusive
        and rating_deviation <= selection.maximum_rating_deviation
        and popularity >= selection.minimum_popularity
        and plays >= selection.minimum_plays
    ):
        return None
    puzzle_id = row["PuzzleId"].strip()
    source_game_key = _source_game_key(row["GameUrl"])
    if not puzzle_id or source_game_key is None:
        return None
    return _SelectedPuzzle(
        rank=_puzzle_rank(puzzle_id),
        puzzle_id=puzzle_id,
        initial_fen=row["FEN"],
        moves=row["Moves"],
        rating=rating,
        rating_deviation=rating_deviation,
        popularity=popularity,
        plays=plays,
        source_game_key=source_game_key,
    )


def _validate_selected(
    selected: list[_SelectedPuzzle],
    selection: PuzzleSelectionConfig,
) -> None:
    expected_ratings = range(
        selection.minimum_rating,
        selection.maximum_rating_exclusive,
    )
    counts = Counter(puzzle.rating for puzzle in selected)
    short = [
        rating
        for rating in expected_ratings
        if counts[rating] != selection.puzzles_per_rating
    ]
    if short:
        preview = ", ".join(str(rating) for rating in short[:10])
        raise PuzzleSetError(
            "source did not fill exact puzzle rating(s): "
            f"{preview}{' ...' if len(short) > 10 else ''}"
        )
    source_games = [puzzle.source_game_key for puzzle in selected]
    if len(set(source_games)) != len(source_games):
        raise PuzzleSetError("selected puzzles do not have distinct source games")


def _render_csv(puzzles: list[_SelectedPuzzle]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(_OUTPUT_COLUMNS)
    writer.writerows(puzzle.output_row() for puzzle in puzzles)
    return output.getvalue()


def _parse_puzzles(content: str, puzzle_path: Path) -> Iterator[Puzzle]:
    reader = csv.DictReader(content.splitlines())
    if tuple(reader.fieldnames or ()) != _OUTPUT_COLUMNS:
        raise PuzzleSetError(
            f"{puzzle_path} must have {', '.join(_OUTPUT_COLUMNS)} columns; "
            f"found {reader.fieldnames}"
        )
    for number, row in enumerate(reader, start=2):
        puzzle_id = (row["puzzle_id"] or "").strip()
        source_game_key = (row["source_game_key"] or "").strip()
        try:
            rating = int(row["rating"])
            board = chess.Board(row["initial_fen"])
        except (TypeError, ValueError) as error:
            raise PuzzleSetError(
                f"{puzzle_path} line {number} has invalid metadata: {error}"
            ) from error
        if not puzzle_id or not source_game_key or rating < 0:
            raise PuzzleSetError(
                f"{puzzle_path} line {number} has empty or invalid metadata"
            )
        moves: list[chess.Move] = []
        for ply, token in enumerate((row["moves"] or "").split()):
            try:
                move = chess.Move.from_uci(token)
            except ValueError as error:
                raise PuzzleSetError(
                    f"{puzzle_path} line {number} has unreadable move "
                    f"{token!r}: {error}"
                ) from error
            if not board.is_legal(move):
                raise PuzzleSetError(
                    f"{puzzle_path} line {number} plays illegal move "
                    f"{token!r} at ply {ply}"
                )
            moves.append(move)
            board.push(move)
        if len(moves) < 2 or len(moves) % 2 != 0:
            raise PuzzleSetError(
                f"{puzzle_path} line {number} needs an opponent setup move "
                "followed by one or more complete solution pairs"
            )
        yield Puzzle(
            puzzle_id=puzzle_id,
            initial_fen=row["initial_fen"],
            moves=tuple(moves),
            rating=rating,
            source_game_key=source_game_key,
        )


def _load_metadata(path: Path) -> Mapping[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PuzzleSetError(f"the puzzle artifact is missing {path}") from error
    except json.JSONDecodeError as error:
        raise PuzzleSetError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(metadata, dict):
        raise PuzzleSetError(f"{path} must be a JSON object")
    required = {
        "name",
        "version",
        "entries",
        "puzzles_sha256",
        "source",
        "license",
        "selection",
        "sizing",
        "coverage",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise PuzzleSetError(f"{path} is missing {', '.join(missing)}")
    return metadata


def _puzzle_rank(puzzle_id: str) -> int:
    """Rank uniformly by puzzle id so a subsample stays representative."""

    return int.from_bytes(sha256(puzzle_id.encode()).digest(), "big")


def _sizing_record(selection: PuzzleSelectionConfig) -> dict[str, object]:
    ratings = selection.maximum_rating_exclusive - selection.minimum_rating
    overall = ratings * selection.puzzles_per_rating
    local_ratings = selection.local_precision_span
    local = local_ratings * selection.puzzles_per_rating
    return {
        "method": PUZZLE_SIZING_METHOD,
        "confidence": PUZZLE_DETECTION_CONFIDENCE,
        "power": PUZZLE_DETECTION_POWER,
        "worst_case_probability": 0.5,
        "overall_puzzles": overall,
        "overall_minimum_detectable_difference": (
            conservative_detectable_difference(overall)
        ),
        "local_rating_span": local_ratings,
        "local_puzzles": local,
        "local_minimum_detectable_difference": (
            conservative_detectable_difference(local)
        ),
        "note": (
            "Conservative independent-checkpoint calculation; scoring the same "
            "puzzles across checkpoints is paired and generally more sensitive."
        ),
    }


def _source_game_key(url: str) -> str | None:
    parts = [part for part in urlparse(url).path.split("/") if part]
    if not parts:
        return None
    game_id = parts[0][:8]
    return game_id or None


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _puzzle_numeric_id(puzzle_id: str) -> int:
    """Map a source id onto the uint64 key the shared digest boundary expects."""

    return int.from_bytes(sha256(puzzle_id.encode()).digest()[:8], "big")
