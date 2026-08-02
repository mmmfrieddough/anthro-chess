from __future__ import annotations

import json
import math
from dataclasses import replace

import chess
import pytest
import torch
from torch import Tensor, nn

from anthro_chess.chess import ACTION_VOCABULARY_SIZE, encode_move
from anthro_chess.data import (
    GameEncodingInput,
    SequenceBatch,
    SequenceExample,
    collate_sequences,
    encode_game,
)
from anthro_chess.evaluation import (
    MoveValidationAccumulator,
    evaluate_move_model,
)
from anthro_chess.models import MoveModelBatch

from accelerators import accelerator_parameters


def test_metrics_use_explicit_masks_ratings_and_exact_legal_actions() -> None:
    sequence_batch = _sequence_batch(
        (("e2e4", "e7e5", "g1f3"), 1500, None),
        (("d2d4",), 2100, 2200),
    )
    batch = MoveModelBatch.from_sequence_batch(sequence_batch)
    batch = replace(
        batch,
        action_loss_mask=torch.tensor(
            [[True, True, False], [True, False, False]],
            dtype=torch.bool,
        ),
    )
    logits = torch.zeros(
        (*batch.action_targets.shape, ACTION_VOCABULARY_SIZE),
    )
    active_indices = torch.nonzero(batch.action_loss_mask, as_tuple=False)
    for offset, (batch_index, sequence_index) in enumerate(active_indices.tolist()):
        target = int(batch.action_targets[batch_index, sequence_index].item())
        logits[batch_index, sequence_index, target] = float(offset + 1)

    illegal_action = _first_illegal_action(batch.legal_action_ids[0][1])
    logits[0, 1, illegal_action] = 4.0
    logits[~batch.action_loss_mask] = torch.nan

    accumulator = MoveValidationAccumulator()
    accumulator.update(logits, batch)
    metrics = accumulator.compute()

    active_logits = logits[batch.action_loss_mask].double()
    active_targets = batch.action_targets[batch.action_loss_mask]
    expected_losses = (
        -torch.log_softmax(active_logits, dim=-1)
        .gather(
            1,
            active_targets.unsqueeze(1),
        )
        .squeeze(1)
    )
    expected_legal_masses = []
    expected_legal_losses = []
    for offset, (batch_index, sequence_index) in enumerate(active_indices.tolist()):
        legal_actions = batch.legal_action_ids[batch_index][sequence_index]
        probabilities = torch.softmax(active_logits[offset], dim=-1)
        expected_legal_masses.append(
            float(probabilities[list(legal_actions)].sum().item())
        )
        target = int(active_targets[offset].item())
        expected_legal_losses.append(
            float(
                (
                    torch.logsumexp(active_logits[offset, list(legal_actions)], dim=0)
                    - active_logits[offset, target]
                ).item()
            )
        )

    assert metrics.position_count == 3
    assert metrics.move_loss == pytest.approx(expected_losses.mean().item())
    assert metrics.legal_move_loss == pytest.approx(
        sum(expected_legal_losses) / len(expected_legal_losses)
    )
    assert metrics.legal_mass == pytest.approx(
        sum(expected_legal_masses) / len(expected_legal_masses)
    )
    assert metrics.mask_penalty == pytest.approx(
        sum(-math.log(value) for value in expected_legal_masses)
        / len(expected_legal_masses)
    )
    assert metrics.top1_illegal_rate == pytest.approx(1 / 3)
    assert metrics.rated_position_count == 2
    assert metrics.missing_rating_position_count == 1
    assert metrics.missing_rating_move_loss == pytest.approx(expected_losses[1].item())

    slices = {item.band.name: item for item in metrics.rating_slices}
    assert slices["under_1200"].position_count == 0
    assert slices["under_1200"].move_loss is None
    assert slices["1200_to_1599"].position_count == 1
    assert slices["1200_to_1599"].move_loss == pytest.approx(expected_losses[0].item())
    assert slices["1600_to_1999"].position_count == 0
    assert slices["2000_plus"].position_count == 1
    assert slices["2000_plus"].move_loss == pytest.approx(expected_losses[2].item())
    json.dumps(metrics.as_record(), sort_keys=True, allow_nan=False)


def test_simple_baselines_are_stable_across_batch_aggregation() -> None:
    first_sequence = _sequence_batch(
        (("e2e4", "e7e5"), None, None),
    )
    second_sequence = _sequence_batch(
        (("g1f3",), None, None),
    )
    first_batch = MoveModelBatch.from_sequence_batch(first_sequence)
    second_batch = MoveModelBatch.from_sequence_batch(second_sequence)
    accumulator = MoveValidationAccumulator()

    first_logits = torch.zeros(
        (*first_batch.action_targets.shape, ACTION_VOCABULARY_SIZE),
    )
    second_logits = torch.zeros(
        (*second_batch.action_targets.shape, ACTION_VOCABULARY_SIZE),
    )
    accumulator.update(first_logits, first_batch)
    accumulator.update(second_logits, second_batch)
    metrics = accumulator.compute()

    legal_counts = [
        len(legal_actions)
        for batch in (first_batch, second_batch)
        for row_index, row in enumerate(batch.legal_action_ids)
        for sequence_index, legal_actions in enumerate(row)
        if batch.action_loss_mask[row_index, sequence_index]
    ]
    expected_uniform_legal = sum(math.log(count) for count in legal_counts) / len(
        legal_counts
    )

    assert metrics.position_count == 3
    assert metrics.move_loss == pytest.approx(math.log(ACTION_VOCABULARY_SIZE))
    assert metrics.legal_move_loss == pytest.approx(expected_uniform_legal)
    assert metrics.uniform_over_legal_move_loss == pytest.approx(expected_uniform_legal)
    assert metrics.uniform_over_vocabulary_move_loss == pytest.approx(
        math.log(ACTION_VOCABULARY_SIZE)
    )


def test_metrics_reject_misaligned_legal_actions() -> None:
    batch = MoveModelBatch.from_sequence_batch(_sequence_batch((("e2e4",), None, None)))
    legal_actions = batch.legal_action_ids[0][0]
    misaligned = replace(
        batch,
        legal_action_ids=((tuple(reversed(legal_actions)),),),
    )
    logits = torch.zeros((1, 1, ACTION_VOCABULARY_SIZE))

    with pytest.raises(ValueError, match="sorted and unique"):
        MoveValidationAccumulator().update(logits, misaligned)


def test_evaluate_move_model_uses_loader_path_and_restores_training_mode() -> None:
    batches = (
        _sequence_batch((("e2e4", "e7e5"), 1199, 1200)),
        _sequence_batch((("d2d4",), None, None)),
    )
    model = _ConstantMoveModel()
    model.train()

    first = evaluate_move_model(model, batches)
    second = evaluate_move_model(model, batches)

    assert model.training
    assert first.position_count == 3
    assert first.as_record() == second.as_record()
    assert first.rated_position_count == 2
    assert first.missing_rating_position_count == 1


@pytest.mark.gpu
@pytest.mark.parametrize("accelerator", accelerator_parameters())
def test_accelerator_metrics_match_cpu_without_casting_logits_on_device(
    accelerator: str,
) -> None:
    """Ask the agreement of whichever accelerator this host actually has.

    Nothing here passes through a training or inference device selection, so
    this covers a backend the rest of the suite cannot yet request.
    """

    sequence_batch = _sequence_batch((("e2e4", "e7e5"), 1500, 1600))
    cpu_batch = MoveModelBatch.from_sequence_batch(sequence_batch)
    device_batch = MoveModelBatch.from_sequence_batch(
        sequence_batch,
        device=accelerator,
    )
    cpu_logits = torch.linspace(
        -2.0,
        2.0,
        steps=cpu_batch.action_targets.numel() * ACTION_VOCABULARY_SIZE,
        dtype=torch.float32,
    ).reshape(*cpu_batch.action_targets.shape, ACTION_VOCABULARY_SIZE)
    device_logits = cpu_logits.to(accelerator)

    cpu = MoveValidationAccumulator()
    cpu.update(cpu_logits, cpu_batch)
    on_device = MoveValidationAccumulator()
    on_device.update(device_logits, device_batch)

    cpu_metrics = cpu.compute()
    device_metrics = on_device.compute()
    assert device_metrics.position_count == cpu_metrics.position_count
    assert device_metrics.move_loss == pytest.approx(cpu_metrics.move_loss, rel=1e-6)
    assert device_metrics.legal_move_loss == pytest.approx(
        cpu_metrics.legal_move_loss,
        rel=1e-6,
    )
    assert device_metrics.mask_penalty == pytest.approx(
        cpu_metrics.mask_penalty,
        rel=1e-6,
    )
    assert device_metrics.legal_mass == pytest.approx(
        cpu_metrics.legal_mass,
        rel=1e-6,
    )


class _ConstantMoveModel(nn.Module):
    def forward(self, batch: MoveModelBatch) -> Tensor:
        return torch.zeros(
            (*batch.action_targets.shape, ACTION_VOCABULARY_SIZE),
            device=batch.action_targets.device,
        )


def _sequence_batch(
    *games: tuple[tuple[str, ...], int | None, int | None],
) -> SequenceBatch:
    examples = []
    for game_offset, (moves, white_rating, black_rating) in enumerate(games):
        board = chess.Board()
        action_ids = []
        for move_text in moves:
            move = chess.Move.from_uci(move_text)
            assert move in board.legal_moves
            action_ids.append(encode_move(move))
            board.push(move)
        plies = encode_game(
            GameEncodingInput(
                game_id=100 + game_offset,
                ruleset="standard",
                initial_position=chess.STARTING_FEN,
                action_ids=tuple(action_ids),
                white_normalized_rating=white_rating,
                black_normalized_rating=black_rating,
                time_initial_ms=None,
                time_increment_ms=None,
                clock_remaining_ms=tuple(None for _ in action_ids),
            )
        )
        examples.append(
            SequenceExample(
                shard_index=0,
                game_id=100 + game_offset,
                start_ply=0,
                plies=plies,
            )
        )
    return collate_sequences(examples)


def _first_illegal_action(legal_actions: tuple[int, ...]) -> int:
    legal = set(legal_actions)
    return next(
        action_id
        for action_id in range(ACTION_VOCABULARY_SIZE)
        if action_id not in legal
    )
