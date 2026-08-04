"""Tests for the shared path from normalized rows to scorable positions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chess
import pytest

from anthro_chess.evaluation import scoring
from anthro_chess.evaluation.scoring import build_scoring_inputs


def test_rule_labels_are_derived_only_where_a_reading_asks_for_them(
    normalized_row: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolving a position's predicates costs far more than encoding it.

    Two of the three benchmarks built on these inputs read the labels for a
    window of the positions they score or for none of them, so deriving every
    position's up front is most of what those readings pay for.
    """

    derived: list[chess.Board] = []
    original = scoring.position_labels

    def counting(board: chess.Board) -> scoring.PositionLabels:
        derived.append(board)
        return original(board)

    monkeypatch.setattr(scoring, "position_labels", counting)
    inputs = build_scoring_inputs(
        [normalized_row(51, split="test"), normalized_row(52, split="test")],
        split="test",
        batch_size=1,
        length_bucket_width=None,
        identity_sha256="c" * 64,
    )

    assert inputs.position_count > 2
    assert derived == []

    key = next(iter(inputs.plies))
    labels = inputs.labels(key)

    assert len(derived) == 1
    assert inputs.labels(key) is labels
