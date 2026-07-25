"""Tests for the fixed tricky-rule position suite."""

import json
from pathlib import Path

import pytest

from anthro_chess.evaluation import (
    PositionCharacteristic,
    PositionSuiteError,
    load_position_suite,
)
from anthro_chess.evaluation.slices import (
    LEGAL_MOVE_COUNT_BUCKETS,
    legal_move_count_bucket,
)


def test_packaged_suite_loads_and_verifies_every_declared_characteristic() -> None:
    suite = load_position_suite()

    assert suite.suite_id == "tricky-rules"
    assert suite.version == 1
    assert suite.identity()["positions"] == len(suite.positions)
    assert len({position.id for position in suite.positions}) == len(suite.positions)


def test_suite_covers_every_rule_sensitive_characteristic() -> None:
    """The suite is only useful if each rule case is actually represented."""

    suite = load_position_suite()

    missing = [
        str(characteristic)
        for characteristic in PositionCharacteristic
        if not suite.with_characteristic(characteristic)
    ]

    assert not missing


def test_suite_spans_every_legal_move_count_bucket() -> None:
    suite = load_position_suite()

    buckets = {
        legal_move_count_bucket(position.legal_move_count)
        for position in suite.scorable_positions()
    }

    assert buckets == {name for name, _, _ in LEGAL_MOVE_COUNT_BUCKETS}


def test_terminal_positions_are_kept_but_excluded_from_scoring() -> None:
    suite = load_position_suite()

    terminal = suite.with_characteristic(PositionCharacteristic.TERMINAL)

    assert terminal
    assert all(position.is_terminal for position in terminal)
    assert all(position.legal_move_count == 0 for position in terminal)
    assert all(not position.is_terminal for position in suite.scorable_positions())
    assert len(suite.scorable_positions()) + len(terminal) == len(suite.positions)


def test_scorable_positions_expose_sorted_unique_legal_actions() -> None:
    suite = load_position_suite()

    for position in suite.scorable_positions():
        actions = position.legal_action_ids
        assert actions
        assert tuple(sorted(set(actions))) == actions


def test_a_mislabeled_position_fails_to_load(tmp_path: Path) -> None:
    """A declared characteristic the position lacks must be an error."""

    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "suite_id": "broken",
                "version": 1,
                "positions": [
                    {
                        "id": "not-really-check",
                        "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                        "characteristics": ["check"],
                    }
                ],
            }
        )
    )

    with pytest.raises(PositionSuiteError, match="exact chess logic disagrees"):
        load_position_suite(path=path)


def test_undeclared_characteristics_are_allowed(tmp_path: Path) -> None:
    """Positions may hold properties they do not declare; only false claims fail."""

    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "suite_id": "partial",
                "version": 1,
                "positions": [
                    {
                        "id": "checkmate-without-declaring-check",
                        "fen": "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1",
                        "characteristics": ["terminal"],
                    }
                ],
            }
        )
    )

    suite = load_position_suite(path=path)

    assert suite.positions[0].is_terminal


def test_invalid_suite_material_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"

    path.write_text(json.dumps({"suite_id": "x", "version": 1, "positions": []}))
    with pytest.raises(PositionSuiteError, match="at least one position"):
        load_position_suite(path=path)

    path.write_text(
        json.dumps(
            {
                "suite_id": "x",
                "version": 1,
                "positions": [{"id": "bad-fen", "fen": "not-a-fen"}],
            }
        )
    )
    with pytest.raises(PositionSuiteError, match="invalid FEN"):
        load_position_suite(path=path)

    path.write_text(
        json.dumps(
            {
                "suite_id": "x",
                "version": 1,
                "positions": [
                    {
                        "id": "unknown-label",
                        "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                        "characteristics": ["zugzwang"],
                    }
                ],
            }
        )
    )
    with pytest.raises(PositionSuiteError, match="unknown characteristic"):
        load_position_suite(path=path)


def test_missing_packaged_suite_reports_its_name() -> None:
    with pytest.raises(PositionSuiteError, match="does-not-exist"):
        load_position_suite("does-not-exist")
