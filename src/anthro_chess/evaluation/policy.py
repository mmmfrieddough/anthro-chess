"""Per-position policy quantities every offline benchmark reads.

Move prediction, legality diagnostics, dependency tests, and later decision
decomposition all need the same few numbers about one scored position. They
are computed once here so those benchmarks share a code path instead of each
re-deriving a policy from raw logits and drifting apart.

Everything is computed in float64 on the host. The quantities are small and
compared across checkpoints and machines, so reproducibility matters more than
the negligible cost of moving one batch of active rows to the CPU.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    DRAW_CLAIM_ACTION_ID,
    MOVE_ACTION_COUNT,
    RESIGNATION_ACTION_ID,
)
from anthro_chess.models import MoveModelBatch

POLICY_SCORING_VERSION = 1

#: How many raw top actions the illegal-fraction diagnostic inspects.
TOP_ILLEGAL_ACTIONS = 5

#: Keeps ``legality_lift`` finite when a model places essentially all mass on
#: legal moves. The clamp is far below any difference worth reporting.
_PROBABILITY_EPSILON = 1e-12


@dataclass(frozen=True)
class PositionPolicy:
    """What one scored held-out decision says about the model's policy.

    ``conditioned_rating`` is the value the model actually saw, which is not
    the position's true rating when a dependency test corrupts conditioning.
    """

    game_id: int
    ply_index: int
    target_action_id: int
    legal_action_count: int
    conditioned_rating: int | None
    move_nll: float
    legal_move_nll: float
    uniform_over_legal_move_nll: float
    mask_penalty: float
    legal_mass: float
    legality_lift: float
    legal_margin: float
    top1_illegal: bool
    top_illegal_fraction: float
    target_rank: int

    @property
    def illegal_mass(self) -> float:
        """Return the raw probability mass the model places on illegal moves."""

        return 1.0 - self.legal_mass

    def within_top(self, k: int) -> bool:
        """Return whether the human move is in the legal-masked top ``k``."""

        if type(k) is not int or k < 1:
            raise ValueError("top-k accuracy needs a positive integer k")
        return self.target_rank <= k

    def as_record(self) -> dict[str, object]:
        """Return the detail-tier record for one scored position."""

        return {
            "game_id": self.game_id,
            "ply_index": self.ply_index,
            "target_action_id": self.target_action_id,
            "legal_action_count": self.legal_action_count,
            "conditioned_rating": self.conditioned_rating,
            "move_nll": self.move_nll,
            "legal_move_nll": self.legal_move_nll,
            "uniform_over_legal_move_nll": self.uniform_over_legal_move_nll,
            "mask_penalty": self.mask_penalty,
            "legal_mass": self.legal_mass,
            "legality_lift": self.legality_lift,
            "legal_margin": self.legal_margin,
            "top1_illegal": self.top1_illegal,
            "top_illegal_fraction": self.top_illegal_fraction,
            "target_rank": self.target_rank,
        }


@dataclass(frozen=True)
class ActionSetPolicy:
    """What the model says about one named subset of legal actions.

    ``best_rank`` is where the set's strongest member sits in the legal-masked
    ordering the runtime samples from. Mass alone cannot tell a near miss from
    an absence: a set holding the second choice and one holding the twentieth
    can carry the same small probability. It is ``None`` for an empty set,
    which is a real state rather than a defect — a threatened mate no legal
    move prevents has no successful action to rank.
    """

    game_id: int
    ply_index: int
    name: str
    selected_action_id: int
    raw_probability_mass: float
    best_rank: int | None


@dataclass(frozen=True)
class TerminalActionPolicy:
    """What the model says about ending the game at one scored position.

    Terminal actions are enabled at every ply the encoder produces, so this is
    defined everywhere rather than only where a game actually ended. That is
    the point: the interesting half of a resignation reading is the mass the
    policy spends on resigning at the thousands of plies where nobody did.

    ``draw_claim_mass`` is ``None`` where exact chess logic offered no claim,
    which is not the same as a zero. A claim the rules never made available is
    absent from the decision rather than an option the model declined.
    """

    game_id: int
    ply_index: int
    target_action_id: int
    resignation_mass: float
    draw_claim_mass: float | None

    @property
    def target_is_terminal(self) -> bool:
        """Return whether the action actually taken here ended the game."""

        return self.target_action_id >= MOVE_ACTION_COUNT

    def as_record(self) -> dict[str, object]:
        """Return the detail-tier record for one scored decision."""

        return {
            "game_id": self.game_id,
            "ply_index": self.ply_index,
            "target_action_id": self.target_action_id,
            "resignation_mass": self.resignation_mass,
            "draw_claim_mass": self.draw_claim_mass,
        }


@dataclass(frozen=True)
class _ActiveBatch:
    """The enabled rows of one batch, aligned and validated once."""

    logits: Tensor
    legal_mask: Tensor
    legal_rows: tuple[tuple[int, ...], ...]
    targets: tuple[int, ...]
    game_ids: tuple[int, ...]
    ply_indices: tuple[int, ...]
    ratings: tuple[int | None, ...]


def score_positions(
    logits: Tensor,
    batch: MoveModelBatch,
) -> tuple[PositionPolicy, ...]:
    """Return one policy record per enabled action target in a batch."""

    active = _active_batch(logits, batch)
    if not active.legal_rows:
        return ()

    log_probabilities = torch.log_softmax(active.logits, dim=-1)
    target_index = torch.tensor(active.targets, dtype=torch.long).unsqueeze(1)
    move_nll = -log_probabilities.gather(1, target_index).squeeze(1)

    masked = active.logits.masked_fill(~active.legal_mask, -torch.inf)
    log_legal_mass = torch.logsumexp(masked, dim=-1) - torch.logsumexp(
        active.logits, dim=-1
    )
    legal_mass = torch.exp(log_legal_mass)
    legal_move_nll = move_nll + log_legal_mass

    top_actions = torch.topk(active.logits, TOP_ILLEGAL_ACTIONS, dim=-1).indices
    top_legal = active.legal_mask.gather(1, top_actions)
    top1_illegal = ~top_legal[:, 0]
    top_illegal_fraction = 1.0 - top_legal.to(dtype=torch.float64).mean(dim=-1)

    maximum_legal = masked.amax(dim=-1)
    maximum_illegal = active.logits.masked_fill(active.legal_mask, -torch.inf).amax(
        dim=-1
    )
    legal_margin = maximum_legal - maximum_illegal

    target_logits = active.logits.gather(1, target_index)
    better = (active.logits > target_logits) & active.legal_mask
    target_rank = better.sum(dim=-1) + 1

    move_nll_values = move_nll.tolist()
    legal_move_nll_values = legal_move_nll.tolist()
    mask_penalty_values = (-log_legal_mass).tolist()
    legal_mass_values = legal_mass.tolist()
    legal_margin_values = legal_margin.tolist()
    top_illegal_values = top_illegal_fraction.tolist()
    top1_illegal_values = top1_illegal.tolist()
    target_rank_values = target_rank.tolist()

    return tuple(
        PositionPolicy(
            game_id=active.game_ids[offset],
            ply_index=active.ply_indices[offset],
            target_action_id=active.targets[offset],
            legal_action_count=len(legal_actions),
            conditioned_rating=active.ratings[offset],
            move_nll=move_nll_values[offset],
            legal_move_nll=legal_move_nll_values[offset],
            uniform_over_legal_move_nll=math.log(len(legal_actions)),
            mask_penalty=mask_penalty_values[offset],
            legal_mass=legal_mass_values[offset],
            legality_lift=_legality_lift(
                legal_mass_values[offset],
                len(legal_actions),
            ),
            legal_margin=legal_margin_values[offset],
            top1_illegal=bool(top1_illegal_values[offset]),
            top_illegal_fraction=top_illegal_values[offset],
            target_rank=int(target_rank_values[offset]),
        )
        for offset, legal_actions in enumerate(active.legal_rows)
    )


def score_action_sets(
    logits: Tensor,
    batch: MoveModelBatch,
    action_sets: Mapping[tuple[int, int], Mapping[str, Collection[int]]],
) -> tuple[ActionSetPolicy, ...]:
    """Score named legal-action subsets without retaining whole policies."""

    active = _active_batch(logits, batch)
    if not active.legal_rows:
        return ()

    probabilities = torch.softmax(active.logits, dim=-1)
    masked = active.logits.masked_fill(~active.legal_mask, -torch.inf)
    selected = torch.argmax(masked, dim=-1).tolist()
    scored: list[ActionSetPolicy] = []
    for offset, legal_actions in enumerate(active.legal_rows):
        key = (active.game_ids[offset], active.ply_indices[offset])
        named_sets = action_sets.get(key)
        if not named_sets:
            continue
        legal = frozenset(legal_actions)
        for name, action_ids in sorted(named_sets.items()):
            actions = tuple(sorted(set(action_ids)))
            if any(action not in legal for action in actions):
                raise ValueError(
                    f"action set {name!r} contains an action that is not legal at {key}"
                )
            mass = 0.0
            best_rank: int | None = None
            if actions:
                indices = torch.tensor(actions, dtype=torch.long)
                mass = float(probabilities[offset, indices].sum().item())
                best_logit = masked[offset, indices].amax()
                best_rank = int((masked[offset] > best_logit).sum().item()) + 1
            scored.append(
                ActionSetPolicy(
                    game_id=key[0],
                    ply_index=key[1],
                    name=name,
                    selected_action_id=int(selected[offset]),
                    raw_probability_mass=mass,
                    best_rank=best_rank,
                )
            )
    return tuple(scored)


def score_terminal_actions(
    logits: Tensor,
    batch: MoveModelBatch,
) -> tuple[TerminalActionPolicy, ...]:
    """Return the raw terminal-action mass at every scored decision.

    Raw rather than legal-masked, deliberately. The runtime samples from the
    masked policy, but a resignation reading is about how much the model wants
    to resign rather than about what the mask left it, and masking would rescale
    the quantity by whatever legality problem the checkpoint happens to have.
    """

    active = _active_batch(logits, batch)
    if not active.legal_rows:
        return ()

    probabilities = torch.softmax(active.logits, dim=-1)
    scored: list[TerminalActionPolicy] = []
    for offset, legal_actions in enumerate(active.legal_rows):
        legal = frozenset(legal_actions)
        scored.append(
            TerminalActionPolicy(
                game_id=active.game_ids[offset],
                ply_index=active.ply_indices[offset],
                target_action_id=active.targets[offset],
                resignation_mass=float(
                    probabilities[offset, RESIGNATION_ACTION_ID].item()
                ),
                draw_claim_mass=(
                    float(probabilities[offset, DRAW_CLAIM_ACTION_ID].item())
                    if DRAW_CLAIM_ACTION_ID in legal
                    else None
                ),
            )
        )
    return tuple(scored)


def legal_policy_log_probabilities(
    logits: Tensor,
    batch: MoveModelBatch,
) -> tuple[Tensor, ...]:
    """Return each position's log policy over its own legal actions.

    Comparing two conditioning values needs the distribution the runtime would
    sample from, so this is the legal-masked policy rather than the raw one.
    Each tensor is ordered by the position's sorted legal action ids.
    """

    active = _active_batch(logits, batch)
    masked = active.logits.masked_fill(~active.legal_mask, -torch.inf)
    normalized = torch.log_softmax(masked, dim=-1)
    return tuple(
        normalized[offset, torch.tensor(legal_actions, dtype=torch.long)].clone()
        for offset, legal_actions in enumerate(active.legal_rows)
    )


def policy_divergence(reference: Tensor, candidate: Tensor) -> float:
    """Return the Kullback-Leibler divergence between two legal policies."""

    if reference.shape != candidate.shape:
        raise ValueError("policy divergence needs two distributions over one position")
    probabilities = torch.exp(reference)
    return float((probabilities * (reference - candidate)).sum().item())


def top_action(log_probabilities: Tensor, legal_actions: Sequence[int]) -> int:
    """Return the action a greedy runtime would play from a legal policy."""

    if log_probabilities.shape != (len(legal_actions),):
        raise ValueError("a legal policy must align with its position's actions")
    return int(legal_actions[int(torch.argmax(log_probabilities).item())])


def _legality_lift(legal_mass: float, legal_action_count: int) -> float:
    """Return legal mass relative to uniform over the move vocabulary."""

    uniform = legal_action_count / MOVE_ACTION_COUNT
    return _logit(legal_mass) - _logit(uniform)


def _logit(probability: float) -> float:
    clamped = min(max(probability, _PROBABILITY_EPSILON), 1.0 - _PROBABILITY_EPSILON)
    return math.log(clamped / (1.0 - clamped))


def _active_batch(logits: Tensor, batch: MoveModelBatch) -> _ActiveBatch:
    """Gather the enabled rows onto the host in two device reads.

    Every quantity below is derived from the same batch, so asking the device
    for each one separately would drain its command queue once per question
    while the answers all arrive from one pass. The active logits come across
    as a block, and the alignment metadata comes across stacked, which is also
    what lets the finite check read the copy that was already being made
    instead of paying for a synchronization of its own.
    """

    batch.validate()
    _validate_logit_shape(logits, batch)
    legal_action_ids = batch.legal_action_ids
    if legal_action_ids is None:
        raise ValueError("scoring a policy needs the batch's legal actions")

    active_logits = (
        logits[batch.action_loss_mask].detach().cpu().to(dtype=torch.float64)
    )
    if not torch.all(torch.isfinite(active_logits)):
        raise ValueError("enabled action logits must all be finite")

    # Game ids stay in their own dtype. They are unsigned 64-bit hashes, and
    # folding them into the stack below would wrap every id past the signed
    # maximum onto a negative one that matches no scored position.
    game_id_rows = batch.game_ids.detach().cpu().tolist()
    enabled, target_rows, ply_index_rows, rating_rows, present_rows = (
        torch.stack(
            (
                batch.action_loss_mask.to(dtype=torch.long),
                batch.action_targets.to(dtype=torch.long),
                batch.ply_indices.to(dtype=torch.long),
                batch.inputs.target_rating.values.to(dtype=torch.long),
                batch.inputs.target_rating.present.to(dtype=torch.long),
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    active_indices = tuple(
        (batch_index, sequence_index)
        for batch_index, row in enumerate(enabled)
        for sequence_index, active in enumerate(row)
        if active
    )
    targets = tuple(
        target_rows[batch_index][sequence_index]
        for batch_index, sequence_index in active_indices
    )
    rating_values = [
        rating_rows[batch_index][sequence_index]
        for batch_index, sequence_index in active_indices
    ]
    rating_present = [
        bool(present_rows[batch_index][sequence_index])
        for batch_index, sequence_index in active_indices
    ]
    if any(
        present and value < 0
        for value, present in zip(rating_values, rating_present, strict=True)
    ):
        raise ValueError("present player ratings must be nonnegative")

    legal_rows: list[tuple[int, ...]] = []
    game_ids: list[int] = []
    ply_indices: list[int] = []
    for (batch_index, sequence_index), target in zip(
        active_indices,
        targets,
        strict=True,
    ):
        legal_actions = legal_action_ids[batch_index][sequence_index]
        _validate_legal_actions(legal_actions, target)
        legal_rows.append(legal_actions)
        game_ids.append(game_id_rows[batch_index][sequence_index])
        ply_indices.append(ply_index_rows[batch_index][sequence_index])

    legal_mask = torch.zeros_like(active_logits, dtype=torch.bool)
    for offset, legal_actions in enumerate(legal_rows):
        legal_mask[offset, torch.tensor(legal_actions, dtype=torch.long)] = True

    return _ActiveBatch(
        logits=active_logits,
        legal_mask=legal_mask,
        legal_rows=tuple(legal_rows),
        targets=targets,
        game_ids=tuple(game_ids),
        ply_indices=tuple(ply_indices),
        ratings=tuple(
            value if present else None
            for value, present in zip(rating_values, rating_present, strict=True)
        ),
    )


def _validate_logit_shape(logits: Tensor, batch: MoveModelBatch) -> None:
    expected_shape = (*batch.action_targets.shape, ACTION_VOCABULARY_SIZE)
    if logits.shape != expected_shape:
        raise ValueError(
            "action logits must align with model targets and the action vocabulary"
        )
    if not logits.is_floating_point():
        raise ValueError("action logits must use a floating-point dtype")


def _validate_legal_actions(legal_actions: tuple[int, ...], target: int) -> None:
    if not legal_actions:
        raise ValueError("enabled validation position has no legal actions")
    if tuple(sorted(set(legal_actions))) != legal_actions:
        raise ValueError("legal actions must be sorted and unique")
    if legal_actions[0] < 0 or legal_actions[-1] >= ACTION_VOCABULARY_SIZE:
        raise ValueError("legal action is outside the action vocabulary")
    if target not in legal_actions:
        raise ValueError("enabled validation target is not legal")
