import pytest
import torch
from torch.nn import functional as F

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
