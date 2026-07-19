from __future__ import annotations

import copy

import chess
import pytest
import torch

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    action_vocabulary_identity,
    decode_move,
    encode_move,
)
from anthro_chess.data import (
    GameEncodingInput,
    SequenceBatch,
    SequenceExample,
    collate_sequences,
    encode_game,
)
from anthro_chess.models import (
    CausalMoveModel,
    MoveModelBatch,
    MoveModelConfig,
)
from anthro_chess.training import masked_action_cross_entropy


def test_tensor_boundary_preserves_board_target_context_and_padding() -> None:
    loader_batch = _sequence_batch(
        ("e2e4", "e7e5", "g1f3"),
        ("d2d4",),
    )

    batch = MoveModelBatch.from_sequence_batch(loader_batch)

    assert batch.inputs.piece_ids.shape == (2, 3, 64)
    assert batch.inputs.piece_ids.dtype == torch.long
    assert batch.inputs.piece_ids[0, 0].tolist() == list(
        loader_batch.inputs.piece_ids[0][0]
    )
    assert decode_move(int(batch.action_targets[0, 0].item())).uci() == "e2e4"
    assert batch.action_targets[0, 0].item() in batch.legal_action_ids[0][0]
    assert batch.inputs.previous_action_id.present[0].tolist() == [
        False,
        True,
        True,
    ]
    assert batch.inputs.player_rating.present[0].tolist() == [True, False, True]
    assert batch.inputs.opponent_rating.present[0].tolist() == [
        False,
        True,
        False,
    ]
    assert batch.inputs.time_initial_ms.present[0].tolist() == [
        True,
        True,
        True,
    ]
    assert batch.attention_mask[1].tolist() == [True, False, False]
    assert batch.action_loss_mask.tolist() == batch.attention_mask.tolist()


def test_forward_is_cpu_only_action_vocabulary_compatible_and_masks_padding() -> None:
    batch = MoveModelBatch.from_sequence_batch(
        _sequence_batch(
            ("e2e4", "e7e5", "g1f3"),
            ("d2d4",),
        )
    )
    model = CausalMoveModel(_tiny_config())

    logits = model(batch)

    assert logits.shape == (2, 3, ACTION_VOCABULARY_SIZE)
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()
    assert torch.count_nonzero(logits[1, 1:]).item() == 0
    assert model.identity()["action_vocabulary"] == action_vocabulary_identity()
    assert model.identity()["timing_head"] is False


def test_future_context_does_not_change_earlier_predictions() -> None:
    torch.manual_seed(3)
    original = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("e2e4", "e7e5", "g1f3"))
    )
    changed = copy.deepcopy(original)
    changed.inputs.piece_ids[0, 2].zero_()
    changed.inputs.previous_action_id.values[0, 2] = encode_move(
        chess.Move.from_uci("d2d4")
    )
    changed.inputs.player_rating.values[0, 2] = 9999
    model = CausalMoveModel(_tiny_config()).eval()

    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed)

    torch.testing.assert_close(
        original_logits[:, :2],
        changed_logits[:, :2],
        rtol=0.0,
        atol=1e-6,
    )
    assert not torch.equal(original_logits[:, 2], changed_logits[:, 2])


@pytest.mark.parametrize(
    "moves",
    [
        ("e2e4",),
        ("e2e4", "e7e5", "g1f3"),
    ],
    ids=["single-timestep", "short-causal-sequence"],
)
def test_ordinary_model_and_loss_path_overfits_a_fixed_tiny_sample(
    moves: tuple[str, ...],
) -> None:
    torch.manual_seed(11)
    batch = MoveModelBatch.from_sequence_batch(_sequence_batch(moves))
    model = CausalMoveModel(_tiny_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    with torch.no_grad():
        initial_loss = masked_action_cross_entropy(
            model(batch),
            batch.action_targets,
            batch.action_loss_mask,
        ).item()

    for _ in range(60):
        optimizer.zero_grad()
        loss = masked_action_cross_entropy(
            model(batch),
            batch.action_targets,
            batch.action_loss_mask,
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = masked_action_cross_entropy(
            model(batch),
            batch.action_targets,
            batch.action_loss_mask,
        ).item()

    assert initial_loss > 1.0
    assert final_loss < initial_loss * 0.2


def test_model_config_rejects_incompatible_attention_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MoveModelConfig(model_dim=18, attention_heads=4)
    with pytest.raises(ValueError, match="even"):
        MoveModelConfig(model_dim=15, attention_heads=3)


def _sequence_batch(*move_lines: tuple[str, ...]) -> SequenceBatch:
    examples = []
    for game_offset, moves in enumerate(move_lines):
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
                white_normalized_rating=1500,
                black_normalized_rating=None,
                time_initial_ms=60_000,
                time_increment_ms=0,
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


def _tiny_config() -> MoveModelConfig:
    return MoveModelConfig(
        piece_embedding_dim=2,
        action_embedding_dim=4,
        model_dim=16,
        attention_heads=2,
        transformer_layers=1,
        feedforward_dim=24,
        dropout=0.0,
    )
