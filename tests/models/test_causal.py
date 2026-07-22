from __future__ import annotations

import math

import chess
import pytest
import torch

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    action_vocabulary_identity,
    decode_move,
    encode_move,
    legal_action_ids,
)
from anthro_chess.data import (
    GameEncodingInput,
    SequenceBatch,
    SequenceExample,
    collate_sequences,
    encode_game,
    encoding_identity,
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
        white_rating=1500,
    )

    batch = MoveModelBatch.from_sequence_batch(loader_batch)
    initial_board = chess.Board()
    first_action = encode_move(chess.Move.from_uci("e2e4"))
    second_action = encode_move(chess.Move.from_uci("e7e5"))

    assert batch.inputs.piece_ids.shape == (2, 3, 64)
    assert batch.inputs.piece_ids.dtype == torch.long
    assert batch.inputs.piece_ids[0, 0].tolist() == list(
        loader_batch.inputs.piece_ids[0][0]
    )
    assert batch.inputs.side_to_move[0].tolist() == [0, 1, 0]
    assert batch.inputs.castling_rights[0].tolist() == [15, 15, 15]
    assert batch.inputs.en_passant_square.present[0].tolist() == [
        False,
        True,
        True,
    ]
    assert batch.inputs.en_passant_square.values[0].tolist() == [
        0,
        chess.E3,
        chess.E6,
    ]
    assert batch.inputs.halfmove_clock[0].tolist() == [0, 0, 0]
    assert batch.inputs.fullmove_number[0].tolist() == [1, 1, 2]
    assert batch.inputs.previous_action_id.present[0].tolist() == [
        False,
        True,
        True,
    ]
    assert batch.inputs.previous_action_id.values[0].tolist() == [
        0,
        first_action,
        second_action,
    ]
    assert batch.inputs.target_rating.present[0].tolist() == [True, False, True]
    assert batch.inputs.target_rating.values[0].tolist() == [1500, 0, 1500]
    assert decode_move(int(batch.action_targets[0, 0].item())).uci() == "e2e4"
    assert tuple(batch.legal_action_ids[0][0]) == legal_action_ids(initial_board)
    assert batch.game_ids[0].tolist() == [100, 100, 100]
    assert batch.ply_indices[0].tolist() == [0, 1, 2]
    assert batch.attention_mask[1].tolist() == [True, False, False]
    assert batch.action_loss_mask.tolist() == [
        [True, True, True],
        [True, False, False],
    ]


def test_tensor_boundary_preserves_unsigned_normalized_game_ids() -> None:
    game_id = 2**63 + 7

    batch = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("e2e4",), game_id_base=game_id)
    )

    assert batch.game_ids.dtype == torch.uint64
    assert int(batch.game_ids[0, 0].item()) == game_id


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
    assert model.identity()["version"] == 3
    assert model.identity()["action_vocabulary"] == action_vocabulary_identity()
    assert model.identity()["encoding"] == encoding_identity()
    assert (
        model.identity()["rating_conditioning"] == "post-transformer-feature-modulation"
    )
    assert model.identity()["timing_inputs"] is False
    assert model.identity()["timing_head"] is False


def test_future_context_does_not_change_earlier_predictions() -> None:
    torch.manual_seed(3)
    original = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("e2e4", "e7e5", "g1f3"))
    )
    changed = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("e2e4", "e7e5", "d2d4"))
    )
    changed.inputs.piece_ids[0, 2].zero_()
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


def test_rating_changes_decision_without_changing_encoded_history() -> None:
    torch.manual_seed(7)
    unrated = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("e2e4", "e7e5", "g1f3"))
    )
    rated = MoveModelBatch.from_sequence_batch(
        _sequence_batch(
            ("e2e4", "e7e5", "g1f3"),
            white_rating=2000,
            black_rating=1800,
        )
    )
    model = CausalMoveModel(_tiny_config()).eval()

    with torch.no_grad():
        unrated_history = model.encode_history(unrated)
        rated_history = model.encode_history(rated)
        unrated_logits = model(unrated)
        rated_logits = model(rated)

    torch.testing.assert_close(unrated_history, rated_history, rtol=0.0, atol=0.0)
    assert not torch.equal(unrated_logits, rated_logits)


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
    assert not torch.any(batch.inputs.target_rating.present)
    model = CausalMoveModel(_tiny_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    with torch.no_grad():
        initial_loss_tensor = masked_action_cross_entropy(
            model(batch),
            batch.action_targets,
            batch.action_loss_mask,
        )
        initial_loss = initial_loss_tensor.item()

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

    expected_untrained_loss = math.log(ACTION_VOCABULARY_SIZE)
    assert torch.isfinite(initial_loss_tensor)
    assert abs(initial_loss - expected_untrained_loss) < 2.0
    assert final_loss < initial_loss * 0.2


def test_model_config_rejects_incompatible_attention_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MoveModelConfig(model_dim=18, attention_heads=4)
    with pytest.raises(ValueError, match="even"):
        MoveModelConfig(model_dim=15, attention_heads=3)


def _sequence_batch(
    *move_lines: tuple[str, ...],
    white_rating: int | None = None,
    black_rating: int | None = None,
    game_id_base: int = 100,
) -> SequenceBatch:
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
                game_id=game_id_base + game_offset,
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
                game_id=game_id_base + game_offset,
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
