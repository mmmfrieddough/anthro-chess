"""Tests for the human reference surface the rating ladder is read against."""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from anthro_chess.config import ConfigProvenance, ResolvedConfig, load_config
from anthro_chess.data.accounts import MarkedAccounts, account_digest
from anthro_chess.evaluation.ladder_reference import (
    LADDER_REFERENCE_FILE_NAME,
    LadderReferenceConfig,
    LadderReferenceError,
    build_ladder_reference,
    load_ladder_reference,
)

_SHIPPED = (
    Path(__file__).parents[2]
    / "configs/evaluation/lichess-univ-2018-01-2021-06-ladder-reference.toml"
)


def _resolved(
    normalized: Path,
    manifest: Path,
    **overrides: object,
) -> ResolvedConfig[LadderReferenceConfig]:
    return ResolvedConfig(
        value=LadderReferenceConfig.model_validate(
            {
                "reference_id": "fixture-reference",
                "normalized": str(normalized),
                "manifest": str(manifest),
                "speed": "blitz",
                "rating_namespace": "fixture_blitz",
                "target_ratings": [1500, 1650, 1800],
                "rating_window": 25,
                "excluded_terminations": ["clock_expiry", "abandonment"],
                "minimum_games": 2,
                **overrides,
            }
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )


@pytest.fixture
def corpus(
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> Callable[..., tuple[Path, Path]]:
    """Return a factory writing held-out games around a three-rating grid."""

    def build(tmp_path: Path, **layout: Any) -> tuple[Path, Path]:
        def game(
            index: int,
            white: int,
            black: int,
            result: str,
            **fields: Any,
        ) -> dict[str, Any]:
            termination = fields.pop("termination_category", None)
            namespace = fields.pop("source_rating_namespace", None)
            row = normalized_row(index, ratings=(white, black), result=result, **fields)
            if termination is not None:
                row["termination_category"] = termination
            if namespace is not None:
                row["source_rating_namespace"] = namespace
            return row

        rows = [
            # 1650 against 1500, the higher-rated player as white.
            game(1, 1650, 1500, "1-0"),
            game(2, 1660, 1490, "1-0"),
            game(3, 1625, 1525, "1/2-1/2"),
            # And as black.
            game(4, 1500, 1640, "0-1"),
            game(5, 1510, 1650, "1-0"),
            # 1800 against 1500, one colour only.
            game(6, 1800, 1500, "1-0"),
            # None of these reach the surface.
            game(7, 1650, 1500, "1-0", split="train"),
            game(8, 1650, 1530, "1-0"),
            game(9, 1650, 1500, "0-1", termination_category="clock_expiry"),
            game(10, 1650, 1500, "0-1", time_initial_ms=60_000),
            game(11, 1650, 1500, "0-1", source_rating_namespace="fixture_rapid"),
            game(12, 1500, 1510, "1-0"),
            game(13, 1650, 1500, "0-1", termination_category="abandonment"),
            game(14, 1650, 1500, "*"),
        ]
        return write_corpus(tmp_path / "corpus", rows, **layout)

    return build


def test_a_cell_averages_the_higher_rated_players_score_over_both_colours(
    tmp_path: Path,
    corpus: Callable[..., tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)

    result = build_ladder_reference(
        _resolved(normalized, manifest), tmp_path / "reference"
    )

    cell = load_ladder_reference(tmp_path / "reference").cell(1500, 1650)
    as_white = (2.5 / 3, math.sqrt(((2.25 / 3) - (2.5 / 3) ** 2) * 3 / 2 / 3))
    as_black = (0.5, math.sqrt(0.25 * 2 / 1 / 2))
    assert cell.games == 5
    assert cell.score == pytest.approx((as_white[0] + as_black[0]) / 2)
    assert cell.standard_error == pytest.approx(
        math.sqrt(as_white[1] ** 2 + as_black[1] ** 2) / 2
    )
    assert result.scanned == {
        "excluded_termination_games": 2,
        "marked_games": 0,
        "selected_games": 10,
        "split_games": 13,
        "surface_games": 6,
    }


def test_a_cell_short_of_either_colour_reports_its_count_and_no_score(
    tmp_path: Path,
    corpus: Callable[..., tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)

    build_ladder_reference(_resolved(normalized, manifest), tmp_path / "reference")

    reference = load_ladder_reference(tmp_path / "reference")
    one_colour = reference.cell(1800, 1500)
    empty = reference.cell(1800, 1650)
    assert (one_colour.thin, one_colour.games, one_colour.score) == (True, 1, None)
    assert (empty.thin, empty.games, empty.standard_error) == (True, 0, None)
    with pytest.raises(LadderReferenceError, match="no cell for 1500 against 1550"):
        reference.cell(1500, 1550)


def test_a_marked_account_takes_its_games_off_the_surface(
    tmp_path: Path,
    corpus: Callable[..., tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)
    snapshot = MarkedAccounts(
        covers_archives=("0" * 64,),
        queried_at="2026-08-13",
        accounts_total=40,
        accounts_queried=30,
        slots_total=200,
        slots_queried=180,
        digests=frozenset({account_digest("black5")}),
    ).write(tmp_path / "marked-accounts.txt")

    result = build_ladder_reference(
        _resolved(normalized, manifest, marked_accounts=str(snapshot)),
        tmp_path / "reference",
    )

    assert result.scanned["marked_games"] == 1
    assert load_ladder_reference(tmp_path / "reference").cell(1650, 1500).games == 4


def test_the_identity_is_the_surface_however_the_scan_was_divided(
    tmp_path: Path,
    corpus: Callable[..., tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path, games_per_shard=3, row_group_size=2)

    serial = build_ladder_reference(
        _resolved(normalized, manifest), tmp_path / "serial"
    )
    parallel = build_ladder_reference(
        _resolved(normalized, manifest), tmp_path / "parallel", workers=2
    )
    pinned = load_ladder_reference(
        tmp_path / "parallel", expected_surface_sha256=serial.reference.surface_sha256
    )

    assert parallel.reference.surface_sha256 == serial.reference.surface_sha256
    assert pinned.identity() == {
        "id": "fixture-reference",
        "surface_sha256": serial.reference.surface_sha256,
    }


def test_a_reference_that_is_not_the_pinned_one_is_refused(
    tmp_path: Path,
    corpus: Callable[..., tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)
    built = build_ladder_reference(
        _resolved(normalized, manifest), tmp_path / "reference"
    )

    with pytest.raises(LadderReferenceError, match="not the reference"):
        load_ladder_reference(tmp_path / "reference", expected_surface_sha256="0" * 64)
    with pytest.raises(LadderReferenceError, match="expected identity"):
        build_ladder_reference(
            _resolved(
                normalized,
                manifest,
                rating_window=20,
                expected_surface_sha256=built.reference.surface_sha256,
            ),
            tmp_path / "narrower",
        )


def test_an_edited_count_is_caught_on_load(
    tmp_path: Path,
    corpus: Callable[..., tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)
    build_ladder_reference(_resolved(normalized, manifest), tmp_path / "reference")
    path = tmp_path / "reference" / LADDER_REFERENCE_FILE_NAME
    record = json.loads(path.read_text())
    record["counts"]["white_wins"][1][0] += 1
    path.write_text(json.dumps(record))

    with pytest.raises(LadderReferenceError, match="does not hold the surface"):
        load_ladder_reference(tmp_path / "reference")


def test_a_window_reaching_two_grid_ratings_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="stand in for two grid ratings"):
        _resolved(tmp_path, tmp_path, rating_window=75)


def test_the_shipped_selection_is_valid() -> None:
    resolved = load_config(LadderReferenceConfig, path=_SHIPPED)

    assert resolved.value.expected_surface_sha256 is not None
