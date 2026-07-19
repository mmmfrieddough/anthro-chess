"""Masked objectives shared by correctness checks and ordinary training."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_action_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    loss_mask: Tensor,
) -> Tensor:
    """Average action cross-entropy over explicitly enabled timesteps."""

    if logits.ndim != 3:
        raise ValueError("action logits must be batch by sequence by vocabulary")
    if targets.shape != logits.shape[:2] or loss_mask.shape != targets.shape:
        raise ValueError("action logits, targets, and loss mask must align")
    if targets.dtype != torch.long:
        raise ValueError("action targets must use torch.long")
    if loss_mask.dtype != torch.bool:
        raise ValueError("action loss mask must use torch.bool")
    if not torch.any(loss_mask):
        raise ValueError("action loss mask must enable at least one target")
    return F.cross_entropy(logits[loss_mask], targets[loss_mask])
