"""Tests for the shared path from normalized rows to scorable positions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chess
import pytest

from anthro_chess.evaluation import scoring
from anthro_chess.evaluation.results.metrics import (
    HELD_OUT_MOVE_LOSS_BY_OPENING_TIER,
)
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


def test_an_opening_family_is_classified_once_per_game(
    normalized_row: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every position of a game shares one label, so one replay covers them."""

    replays: list[str] = []
    original = scoring.classify_action_ids

    def counting(action_ids: Any, **kwargs: Any) -> Any:
        label = original(action_ids, **kwargs)
        replays.append(label.family)
        return label

    monkeypatch.setattr(scoring, "classify_action_ids", counting)
    inputs = build_scoring_inputs(
        [normalized_row(51, split="test"), normalized_row(52, split="test")],
        split="test",
        batch_size=1,
        length_bucket_width=None,
        identity_sha256="c" * 64,
    )

    assert replays == []
    assert inputs.opening_family(51) == "Ruy Lopez"
    assert inputs.opening_family(52) == "Ruy Lopez"
    assert inputs.opening_family(51) == "Ruy Lopez"
    assert replays == ["Ruy Lopez", "Ruy Lopez"]


def test_a_cadence_cannot_name_a_metric_it_could_never_produce() -> None:
    """A training cadence has no family count over the training selection."""

    supported = scoring.slice_metric_identifiers()

    assert "held_out.move_loss_opening" in supported
    assert not supported & {
        definition.identifier
        for definition in HELD_OUT_MOVE_LOSS_BY_OPENING_TIER.values()
    }
