"""Tests for opening-family frequency and the tail reading built on it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from anthro_chess.data.artifacts import normalized_shard_paths
from anthro_chess.evaluation.aggregation import (
    OPENING_FAMILY_DIMENSION,
    SliceAggregator,
    SliceTable,
)
from anthro_chess.evaluation.opening_frequency import (
    UNCLASSIFIED_TIER,
    UNSEEN_TIER,
    OpeningFrequency,
    OpeningFrequencyError,
    count_opening_families,
    read_opening_tail,
)
from anthro_chess.evaluation.openings.names import UNCLASSIFIED
from anthro_chess.evaluation.policy import PositionPolicy
from anthro_chess.evaluation.slices import (
    GamePhase,
    PlayerColor,
    PositionSlices,
)

#: A second opening the book names. The shared fixture line is a Ruy Lopez, so
#: counting more than one family needs a game that leaves it rather than a
#: deeper line of the same one.
_SICILIAN = ("e2e4", "c7c5", "g1f3", "d7d6")

#: A game the book names nothing in. Matching is positional and every first
#: move from the standard start is catalogued, so a game only goes unclassified
#: by starting somewhere the book never reaches.
_UNNAMED_POSITION = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
_UNNAMED = ("e2e4", "e8d8")


def test_only_the_named_split_is_counted(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    rows = [
        normalized_row(1, split="train"),
        normalized_row(2, split="train"),
        normalized_row(3, split="train", moves=_SICILIAN),
        normalized_row(
            4,
            split="train",
            moves=_UNNAMED,
            initial_position=_UNNAMED_POSITION,
            result="1/2-1/2",
        ),
        normalized_row(5, split="test", moves=_SICILIAN),
    ]
    normalized, _ = write_corpus(tmp_path / "corpus", rows)

    frequency = count_opening_families(normalized_shard_paths(normalized), "train")

    assert frequency.games == 4
    assert frequency.classified_games == 3
    assert frequency.family_games == {"Ruy Lopez": 2, "Sicilian Defense": 1}
    assert frequency.share("Ruy Lopez") == pytest.approx(0.5)
    assert frequency.share("Old Indian Defense") == 0.0


def test_a_split_with_no_games_is_rejected(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    normalized, _ = write_corpus(
        tmp_path / "corpus",
        [normalized_row(1, split="test")],
    )

    with pytest.raises(OpeningFrequencyError, match="no train split games"):
        count_opening_families(normalized_shard_paths(normalized), "train")


def test_tiers_separate_scarcity_from_absence_and_from_being_unnamed() -> None:
    frequency = _frequency(
        {
            "Sicilian Defense": 600,
            "French Defense": 200,
            "Ruy Lopez": 50,
            "Bird Opening": 2,
        },
        games=10_000,
    )

    assert frequency.tier("Sicilian Defense") == "common_opening"
    assert frequency.tier("French Defense") == "uncommon_opening"
    assert frequency.tier("Ruy Lopez") == "rare_opening"
    assert frequency.tier("Bird Opening") == "very_rare_opening"
    assert frequency.tier("Grob Opening") == UNSEEN_TIER
    assert frequency.tier(UNCLASSIFIED) == UNCLASSIFIED_TIER


def test_the_tail_slope_reads_loss_still_falling_with_frequency() -> None:
    frequency = _frequency(
        {"Rare A": 10, "Rare B": 20, "Rare C": 40, "Sicilian Defense": 8_000},
        games=100_000,
    )
    slices = _slice_table(
        {"Rare A": 4.0, "Rare B": 3.0, "Rare C": 2.0, "Sicilian Defense": 1.0}
    )

    reading = read_opening_tail(slices, frequency)

    assert reading.tail_families == 3
    assert reading.tail_move_loss_slope is not None
    assert reading.tail_move_loss_slope < 0.0
    tiers = {row.family: row.tier for row in reading.families}
    assert tiers["Sicilian Defense"] == "common_opening"
    assert tiers["Rare A"] == "very_rare_opening"


def test_a_flat_tail_reports_a_slope_of_zero_rather_than_none() -> None:
    frequency = _frequency({"Rare A": 10, "Rare B": 40}, games=100_000)
    slices = _slice_table({"Rare A": 2.5, "Rare B": 2.5})

    reading = read_opening_tail(slices, frequency)

    assert reading.tail_move_loss_slope == pytest.approx(0.0)


def test_one_tail_family_supports_no_slope() -> None:
    frequency = _frequency({"Rare A": 10, "Sicilian Defense": 8_000}, games=100_000)
    slices = _slice_table({"Rare A": 4.0, "Sicilian Defense": 1.0})

    reading = read_opening_tail(slices, frequency)

    assert reading.tail_families == 1
    assert reading.tail_move_loss_slope is None


def test_unnamed_games_stay_off_the_frequency_axis() -> None:
    frequency = _frequency({"Sicilian Defense": 8_000}, games=100_000)
    slices = _slice_table({"Sicilian Defense": 1.0, UNCLASSIFIED: 5.0})

    reading = read_opening_tail(slices, frequency)

    assert [row.family for row in reading.families] == ["Sicilian Defense"]


def _frequency(family_games: dict[str, int], *, games: int) -> OpeningFrequency:
    return OpeningFrequency(
        split="train",
        games=games,
        family_games=family_games,
        paths=("normalized/games.parquet",),
    )


def _slice_table(move_losses: dict[str, float]) -> SliceTable:
    """Return a slice table holding one scored position per family."""

    aggregator = SliceAggregator()
    for index, (family, move_loss) in enumerate(sorted(move_losses.items())):
        aggregator.add(
            PositionPolicy(
                game_id=index,
                ply_index=0,
                target_action_id=7,
                legal_action_count=20,
                conditioned_rating=1500,
                move_nll=move_loss,
                legal_move_nll=move_loss,
                uniform_over_legal_move_nll=3.0,
                mask_penalty=0.25,
                legal_mass=0.8,
                legality_lift=1.5,
                legal_margin=0.5,
                top1_illegal=False,
                top_illegal_fraction=0.2,
                target_rank=1,
            ),
            PositionSlices(
                phase=GamePhase.OPENING,
                color=PlayerColor.WHITE,
                legal_move_count=20,
                legal_move_count_bucket="11_to_25",
                rating_band="1200_to_1599",
            ),
            (),
            opening_family=family,
        )
    table = aggregator.compute()
    assert set(table.dimensions[OPENING_FAMILY_DIMENSION]) == set(move_losses)
    return table
