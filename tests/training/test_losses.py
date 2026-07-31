import chess
import pytest
import torch
from torch.nn import functional as F

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    RESIGNATION_ACTION_ID,
    encode_move,
)
from anthro_chess.data import (
    GameEncodingInput,
    SequenceExample,
    collate_sequences,
    encode_game,
)
from anthro_chess.models import MoveModelBatch
from anthro_chess.training import masked_action_cross_entropy


def test_action_loss_uses_only_explicitly_enabled_targets() -> None:
    logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [100.0, -100.0, -100.0]],
            [[0.0, 2.0, -1.0], [-100.0, 100.0, -100.0]],
        ]
    )
    targets = torch.tensor([[0, 2], [1, 2]])
    mask = torch.tensor([[True, False], [True, False]])

    loss = masked_action_cross_entropy(logits, targets, mask)

    expected = F.cross_entropy(
        torch.stack((logits[0, 0], logits[1, 0])),
        torch.tensor([0, 1]),
    )
    torch.testing.assert_close(loss, expected)


def test_padded_logits_and_targets_cannot_change_action_loss() -> None:
    torch.manual_seed(5)
    logits = torch.randn((2, 3, 7))
    targets = torch.tensor([[1, 2, 3], [4, 0, 0]])
    mask = torch.tensor([[True, True, True], [True, False, False]])
    expected = masked_action_cross_entropy(logits, targets, mask)

    changed_logits = logits.clone()
    changed_logits[~mask] = torch.tensor(
        [1000.0, -1000.0, 500.0, -500.0, 250.0, -250.0, 0.0]
    )
    changed_targets = targets.clone()
    changed_targets[~mask] = 6

    actual = masked_action_cross_entropy(changed_logits, changed_targets, mask)

    torch.testing.assert_close(actual, expected)


def test_a_terminal_target_is_supervised_like_any_other_action() -> None:
    """A game ending in a decision must reach the loss, mask included.

    The batch boundary rejects a target its timestep does not enable, so this
    also pins that the training mask offers the terminal action at the step it
    was taken.
    """

    moves = ("e2e4", "e7e5", "g1f3")
    action_ids = tuple(encode_move(chess.Move.from_uci(move)) for move in moves)
    plies = encode_game(
        GameEncodingInput(
            game_id=1,
            ruleset="standard",
            initial_position=chess.STARTING_FEN,
            action_ids=(*action_ids, RESIGNATION_ACTION_ID),
            white_normalized_rating=1500,
            black_normalized_rating=1500,
            time_initial_ms=None,
            time_increment_ms=None,
            clock_remaining_ms=(None,) * (len(moves) + 1),
        )
    )
    batch = MoveModelBatch.from_sequence_batch(
        collate_sequences([SequenceExample(0, 1, 0, plies)])
    )

    assert batch.action_targets[0, -1].item() == RESIGNATION_ACTION_ID
    assert bool(batch.action_loss_mask[0, -1])
    assert RESIGNATION_ACTION_ID in batch.legal_action_ids[0][-1]

    logits = torch.zeros((1, len(plies), ACTION_VOCABULARY_SIZE), requires_grad=True)
    loss = masked_action_cross_entropy(
        logits, batch.action_targets, batch.action_loss_mask
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    # The terminal step contributes a gradient, so it is trained rather than
    # merely tolerated.
    assert torch.any(logits.grad[0, -1] != 0.0)


def test_action_loss_rejects_empty_or_misaligned_masks() -> None:
    logits = torch.zeros((1, 2, 3))
    targets = torch.zeros((1, 2), dtype=torch.long)

    with pytest.raises(ValueError, match="at least one"):
        masked_action_cross_entropy(
            logits,
            targets,
            torch.zeros((1, 2), dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="align"):
        masked_action_cross_entropy(
            logits,
            targets,
            torch.ones((1, 1), dtype=torch.bool),
        )
