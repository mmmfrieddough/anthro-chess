"""Coverage statistics summarizing a frozen evaluation pool.

These make a thin slice visible before a benchmark reports a number computed
from it. They stay to the cheap derived slices; rule-sensitive characteristics
cost far more per position and belong to the positions a benchmark scores.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from anthro_chess.data import GameEncodingInput, encode_game
from anthro_chess.data.artifacts import DataLoadingError
from anthro_chess.data.schema import NormalizedColumn
from anthro_chess.evaluation.slices import position_slices


def pool_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the pool so thin slices are visible before a benchmark runs."""

    phases: Counter[str] = Counter()
    colors: Counter[str] = Counter()
    legal_buckets: Counter[str] = Counter()
    rating_bands: Counter[str] = Counter()
    results: Counter[str] = Counter()
    clock_presence: Counter[str] = Counter()
    total_plies = 0
    minimum_plies: int | None = None
    maximum_plies: int | None = None
    unrated_positions = 0

    for row in rows:
        results[str(row[NormalizedColumn.RESULT])] += 1
        clock_statuses = row[NormalizedColumn.CLOCK_STATUS]
        clock_presence[
            "present"
            if any(status == "present" for status in clock_statuses)
            else "absent"
        ] += 1
        ply_count = int(row[NormalizedColumn.PLY_COUNT])
        total_plies += ply_count
        minimum_plies = (
            ply_count if minimum_plies is None else min(minimum_plies, ply_count)
        )
        maximum_plies = (
            ply_count if maximum_plies is None else max(maximum_plies, ply_count)
        )

        for ply in encode_game(_encoding_input(row)):
            slices = position_slices(ply)
            phases[str(slices.phase)] += 1
            colors[str(slices.color)] += 1
            legal_buckets[slices.legal_move_count_bucket] += 1
            if slices.rating_band is None:
                unrated_positions += 1
            else:
                rating_bands[slices.rating_band] += 1

    return {
        "games": len(rows),
        "plies": {
            "total": total_plies,
            "minimum_per_game": minimum_plies,
            "maximum_per_game": maximum_plies,
        },
        "results": dict(sorted(results.items())),
        "clock_presence_games": dict(sorted(clock_presence.items())),
        "phase_positions": dict(sorted(phases.items())),
        "color_positions": dict(sorted(colors.items())),
        "legal_move_count_positions": dict(sorted(legal_buckets.items())),
        "rating_band_positions": dict(sorted(rating_bands.items())),
        "positions_without_rating": unrated_positions,
    }


def _encoding_input(row: Mapping[str, Any]) -> GameEncodingInput:
    try:
        return GameEncodingInput(
            game_id=int(row[NormalizedColumn.GAME_ID]),
            ruleset=row[NormalizedColumn.RULESET],
            initial_position=row[NormalizedColumn.INITIAL_POSITION],
            action_ids=tuple(row[NormalizedColumn.ACTION_IDS]),
            white_normalized_rating=row[NormalizedColumn.WHITE_NORMALIZED_RATING],
            black_normalized_rating=row[NormalizedColumn.BLACK_NORMALIZED_RATING],
            time_initial_ms=row[NormalizedColumn.TIME_INITIAL_MS],
            time_increment_ms=row[NormalizedColumn.TIME_INCREMENT_MS],
            clock_remaining_ms=tuple(row[NormalizedColumn.CLOCK_REMAINING_MS]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DataLoadingError(f"invalid normalized game in pool: {error}") from error
