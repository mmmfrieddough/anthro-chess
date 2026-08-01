"""Tests for the per-position policy quantities every benchmark shares."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
import torch

from anthro_chess.chess import ACTION_VOCABULARY_SIZE, MOVE_ACTION_COUNT
from anthro_chess.data import SequenceBatch
from anthro_chess.evaluation.policy import (
    TOP_ILLEGAL_ACTIONS,
    legal_policy_log_probabilities,
    policy_divergence,
    score_action_sets,
    score_positions,
    top_action,
)
from anthro_chess.models import MoveModelBatch


def test_policy_records_hand_computable_legality_and_rank(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    batch = MoveModelBatch.from_sequence_batch(sequence_batch((("e2e4",), 1500, None)))
    legal_actions = batch.legal_action_ids[0][0]
    target = int(batch.action_targets[0, 0].item())
    others = [action for action in legal_actions if action != target]
    illegal = _first_illegal_action(legal_actions)
    logits = torch.zeros((1, 1, ACTION_VOCABULARY_SIZE))
    logits[0, 0, target] = 2.0
    logits[0, 0, others[0]] = 3.0
    logits[0, 0, others[1]] = 1.5
    logits[0, 0, others[2]] = 1.2
    logits[0, 0, illegal] = 5.0

    (position,) = score_positions(logits, batch)

    probabilities = torch.softmax(logits[0, 0].double(), dim=-1)
    legal_mass = float(probabilities[list(legal_actions)].sum().item())
    uniform = len(legal_actions) / MOVE_ACTION_COUNT
    assert position.game_id == 100
    assert position.ply_index == 0
    assert position.conditioned_rating == 1500
    assert position.legal_action_count == len(legal_actions)
    assert position.move_nll == pytest.approx(
        -float(torch.log_softmax(logits[0, 0].double(), dim=-1)[target].item())
    )
    assert position.legal_mass == pytest.approx(legal_mass)
    assert position.illegal_mass == pytest.approx(1.0 - legal_mass)
    assert position.mask_penalty == pytest.approx(-math.log(legal_mass))
    assert position.legal_move_nll == pytest.approx(
        position.move_nll + math.log(legal_mass)
    )
    assert position.uniform_over_legal_move_nll == pytest.approx(
        math.log(len(legal_actions))
    )
    assert position.legal_margin == pytest.approx(3.0 - 5.0)
    assert position.legality_lift == pytest.approx(
        math.log(legal_mass / (1.0 - legal_mass)) - math.log(uniform / (1.0 - uniform))
    )
    assert position.top1_illegal is True
    assert position.top_illegal_fraction == pytest.approx(1 / TOP_ILLEGAL_ACTIONS)
    assert position.target_rank == 2
    assert position.within_top(2) is True
    assert position.within_top(1) is False


def test_target_rank_counts_only_stronger_legal_actions(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    batch = MoveModelBatch.from_sequence_batch(sequence_batch((("d2d4",), None, None)))
    legal_actions = batch.legal_action_ids[0][0]
    target = int(batch.action_targets[0, 0].item())
    logits = torch.zeros((1, 1, ACTION_VOCABULARY_SIZE))
    logits[0, 0, target] = 1.0
    logits[0, 0, _first_illegal_action(legal_actions)] = 9.0

    (position,) = score_positions(logits, batch)

    assert position.target_rank == 1
    assert position.within_top(1) is True


def test_named_action_sets_report_raw_mass_and_the_legal_greedy_choice(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    batch = MoveModelBatch.from_sequence_batch(sequence_batch((("e2e4",), 1500, None)))
    legal_actions = batch.legal_action_ids[0][0]
    target = int(batch.action_targets[0, 0].item())
    alternative = next(action for action in legal_actions if action != target)
    illegal = _first_illegal_action(legal_actions)
    logits = torch.zeros((1, 1, ACTION_VOCABULARY_SIZE))
    logits[0, 0, target] = 2.0
    logits[0, 0, alternative] = 3.0
    logits[0, 0, illegal] = 9.0

    (score,) = score_action_sets(
        logits,
        batch,
        {(100, 0): {"forced": {target}}},
    )

    probabilities = torch.softmax(logits[0, 0].double(), dim=-1)
    assert score.name == "forced"
    assert score.selected_action_id == alternative
    assert score.raw_probability_mass == pytest.approx(
        float(probabilities[target].item())
    )


def test_legal_policy_normalizes_over_legal_actions_only(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    batch = MoveModelBatch.from_sequence_batch(
        sequence_batch((("e2e4", "e7e5"), 1500, 1500))
    )
    legal_actions = batch.legal_action_ids[0][0]
    logits = torch.zeros((*batch.action_targets.shape, ACTION_VOCABULARY_SIZE))
    logits[0, 0, legal_actions[0]] = 4.0
    logits[0, 0, _first_illegal_action(legal_actions)] = 9.0

    first, second = legal_policy_log_probabilities(logits, batch)

    assert first.shape == (len(legal_actions),)
    assert float(torch.exp(first).sum().item()) == pytest.approx(1.0)
    assert top_action(first, legal_actions) == legal_actions[0]
    assert policy_divergence(first, first) == pytest.approx(0.0)
    assert policy_divergence(first, second) > 0.0


def test_scoring_rejects_legal_actions_that_do_not_align(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    batch = MoveModelBatch.from_sequence_batch(sequence_batch((("e2e4",), None, None)))
    misaligned = replace(
        batch,
        legal_action_ids=((tuple(reversed(batch.legal_action_ids[0][0])),),),
    )
    logits = torch.zeros((1, 1, ACTION_VOCABULARY_SIZE))

    with pytest.raises(ValueError, match="sorted and unique"):
        score_positions(logits, misaligned)


def test_scoring_preserves_a_game_id_past_the_signed_maximum(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    """Game ids are unsigned 64-bit hashes, and most of them exceed a long.

    Reading them through any signed representation wraps the upper half of the
    range onto negative values. Nothing downstream notices until aggregation
    fails to find a scored position's slice, so the fixture ids -- which are
    small -- cannot stand in for this.
    """

    batch = MoveModelBatch.from_sequence_batch(sequence_batch((("e2e4",), None, None)))
    identifier = 2**64 - 1234567
    batch = replace(
        batch,
        game_ids=torch.full_like(batch.game_ids, identifier, dtype=torch.uint64),
    )
    logits = torch.zeros((*batch.action_targets.shape, ACTION_VOCABULARY_SIZE))

    (position,) = score_positions(logits, batch)

    assert position.game_id == identifier


@pytest.mark.parametrize("corruption", [math.inf, -math.inf, math.nan])
def test_scoring_rejects_non_finite_logits_at_an_enabled_position(
    sequence_batch: Callable[..., SequenceBatch],
    corruption: float,
) -> None:
    batch = MoveModelBatch.from_sequence_batch(sequence_batch((("e2e4",), None, None)))
    logits = torch.zeros((*batch.action_targets.shape, ACTION_VOCABULARY_SIZE))
    logits[0, 0, 0] = corruption

    with pytest.raises(ValueError, match="enabled action logits must all be finite"):
        score_positions(logits, batch)


def test_scoring_ignores_non_finite_logits_where_no_action_is_scored(
    sequence_batch: Callable[..., SequenceBatch],
) -> None:
    """The check covers the rows the pass reads, which is where it can matter.

    Padding past the end of a shorter history carries whatever the model
    emitted there and is never scored, so holding it to the same standard
    would reject batches the benchmark handles correctly.
    """

    batch = MoveModelBatch.from_sequence_batch(
        sequence_batch((("e2e4",), None, None), (("d2d4", "d7d5"), None, None))
    )
    disabled = torch.nonzero(~batch.action_loss_mask, as_tuple=False).tolist()
    assert disabled
    logits = torch.zeros((*batch.action_targets.shape, ACTION_VOCABULARY_SIZE))
    for batch_index, sequence_index in disabled:
        logits[batch_index, sequence_index, 0] = math.inf

    scored = score_positions(logits, batch)

    assert len(scored) == int(batch.action_loss_mask.sum().item())


def test_scoring_never_asks_the_device_for_a_scalar(
    sequence_batch: Callable[..., SequenceBatch],
    device_read_trap: Callable[[Any], Any],
) -> None:
    """Every per-position quantity comes off one gather and one metadata read.

    Reading identity or conditioning a position at a time, or checking the
    logits on the device rather than on the copy already being made, blocks
    the queue once per position. On CPU that is free and invisible, so this
    asserts on the access pattern instead of on the numbers.
    """

    batch = MoveModelBatch.from_sequence_batch(
        sequence_batch((("e2e4", "e7e5"), 1500, 1600))
    )
    logits = torch.zeros((*batch.action_targets.shape, ACTION_VOCABULARY_SIZE))

    scored = score_positions(device_read_trap(logits), device_read_trap(batch))

    assert [position.as_record() for position in scored] == [
        position.as_record() for position in score_positions(logits, batch)
    ]


def test_the_trap_still_reports_non_finite_logits(
    sequence_batch: Callable[..., SequenceBatch],
    device_read_trap: Callable[[Any], Any],
) -> None:
    """The host-side check must reject what the device-side one rejected."""

    batch = MoveModelBatch.from_sequence_batch(sequence_batch((("e2e4",), None, None)))
    logits = torch.zeros((*batch.action_targets.shape, ACTION_VOCABULARY_SIZE))
    logits[0, 0, 0] = math.inf

    with pytest.raises(ValueError, match="finite"):
        score_positions(device_read_trap(logits), device_read_trap(batch))


def _first_illegal_action(legal_actions: tuple[int, ...]) -> int:
    legal = set(legal_actions)
    return next(
        action_id for action_id in range(MOVE_ACTION_COUNT) if action_id not in legal
    )
