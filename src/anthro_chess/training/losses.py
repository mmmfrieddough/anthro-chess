"""Masked objectives shared by correctness checks and ordinary training."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

#: What :func:`torch.nn.functional.cross_entropy` drops, and its own default.
_IGNORE_INDEX = -100


def masked_action_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    loss_mask: Tensor,
) -> Tensor:
    """Average action cross-entropy over explicitly enabled timesteps.

    Disabled timesteps are dropped by target rather than by gathering the
    enabled rows. Indexing with the mask produces a shape that depends on the
    mask's contents, which is the second of two data-dependent shapes standing
    between this model and a compiled step, and it is also the slower of the
    two ways to say the same thing.

    A batch that enables nothing yields a non-finite loss rather than an error.
    The loader cannot produce one — every example holds at least one ply, and
    the loss mask is the attention mask — and a training run rejects a
    non-finite loss on its own.
    """

    if logits.ndim != 3:
        raise ValueError("action logits must be batch by sequence by vocabulary")
    if targets.shape != logits.shape[:2] or loss_mask.shape != targets.shape:
        raise ValueError("action logits, targets, and loss mask must align")
    if targets.dtype != torch.long:
        raise ValueError("action targets must use torch.long")
    if loss_mask.dtype != torch.bool:
        raise ValueError("action loss mask must use torch.bool")
    supervised = torch.where(loss_mask, targets, _IGNORE_INDEX)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        supervised.reshape(-1),
        ignore_index=_IGNORE_INDEX,
    )
