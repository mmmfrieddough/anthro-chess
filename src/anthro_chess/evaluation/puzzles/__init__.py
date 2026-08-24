"""The owned puzzle set and its rating-response benchmark."""

from typing import TYPE_CHECKING, Any

from anthro_chess.evaluation.puzzles.dataset import (
    Puzzle,
    PuzzleSelection,
    PuzzleSelectionConfig,
    PuzzleSet,
    PuzzleSetBuildConfig,
    PuzzleSetBuildResult,
    PuzzleSetError,
    conservative_detectable_difference,
    load_puzzle_set,
    prepare_puzzle_set,
    puzzle_set_identity,
)

if TYPE_CHECKING:
    from anthro_chess.evaluation.puzzles.benchmark import (
        PUZZLE_BENCHMARK_VERSION,
        PuzzleBandResult,
        PuzzleBenchmarkConfig,
        PuzzleBenchmarkError,
        PuzzleBenchmarkResult,
        PuzzleRatingResult,
        PuzzleSpread,
        benchmark_puzzles,
        expected_score,
        fitted_rating,
    )

_LAZY_EXPORTS = frozenset(
    {
        "PUZZLE_BENCHMARK_VERSION",
        "PuzzleBandResult",
        "PuzzleBenchmarkConfig",
        "PuzzleBenchmarkError",
        "PuzzleBenchmarkResult",
        "PuzzleRatingResult",
        "PuzzleSpread",
        "benchmark_puzzles",
        "expected_score",
        "fitted_rating",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve torch-backed benchmark names only when they are used."""

    if name in _LAZY_EXPORTS:
        from importlib import import_module

        benchmark = import_module("anthro_chess.evaluation.puzzles.benchmark")
        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PUZZLE_BENCHMARK_VERSION",
    "Puzzle",
    "PuzzleSelectionConfig",
    "PuzzleBandResult",
    "PuzzleBenchmarkConfig",
    "PuzzleBenchmarkError",
    "PuzzleBenchmarkResult",
    "PuzzleRatingResult",
    "PuzzleSelection",
    "PuzzleSet",
    "PuzzleSetBuildConfig",
    "PuzzleSetBuildResult",
    "PuzzleSetError",
    "PuzzleSpread",
    "benchmark_puzzles",
    "conservative_detectable_difference",
    "expected_score",
    "fitted_rating",
    "load_puzzle_set",
    "prepare_puzzle_set",
    "puzzle_set_identity",
]
