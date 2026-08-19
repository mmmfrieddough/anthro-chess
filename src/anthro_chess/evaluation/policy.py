"""Per-position policy quantities every offline benchmark reads.

Move prediction, legality diagnostics, dependency tests, and later decision
decomposition all need the same few numbers about one scored position. They
are computed once here so those benchmarks share a code path instead of each
re-deriving a policy from raw logits and drifting apart.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
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


@dataclass(frozen=True, eq=False)
class ActiveBatch:
    """The enabled rows of one batch, aligned and validated once.

    Building this is what scoring a batch costs the host: a device read, a
    validation of every enabled row, and a legal mask the width of the action
    vocabulary. It is a value rather than a step inside each scorer so that a
    caller reading several quantities off one forward pass pays for it once.

    Equality is identity: a generated one would ask the logits and the mask for
    a truth value they have none of. Comparing two of these means ``torch.equal``
    per tensor, and the device read that costs.
    """

    logits: Tensor
    legal_mask: Tensor
    #: Each enabled row's target, as a column to gather through. Built once
    #: because every pass over a batch gathers through the same one.
    target_index: Tensor
    legal_rows: tuple[tuple[int, ...], ...]
    targets: tuple[int, ...]
    game_ids: tuple[int, ...]
    ply_indices: tuple[int, ...]
    ratings: tuple[int | None, ...]


@dataclass(frozen=True)
class PositionColumns:
    """One batch's scored positions, held as a column per quantity.

    Aggregation sums these and nothing else, so a reading scoring millions of
    positions reads them as columns rather than as a record each.
    :class:`PositionPolicy` is the same numbers for a caller that wants one
    position at a time, and is built out of these rather than beside them.
    """

    game_ids: tuple[int, ...]
    ply_indices: tuple[int, ...]
    target_action_ids: tuple[int, ...]
    conditioned_ratings: tuple[int | None, ...]
    legal_action_counts: NDArray[np.int64]
    move_nll: NDArray[np.float64]
    legal_move_nll: NDArray[np.float64]
    uniform_over_legal_move_nll: NDArray[np.float64]
    mask_penalty: NDArray[np.float64]
    legal_mass: NDArray[np.float64]
    legality_lift: NDArray[np.float64]
    legal_margin: NDArray[np.float64]
    top_illegal_fraction: NDArray[np.float64]
    top1_illegal: NDArray[np.bool_]
    target_rank: NDArray[np.int64]

    def __len__(self) -> int:
        """Return how many positions these columns cover."""

        return len(self.game_ids)

    def select(self, rows: NDArray[np.int64]) -> PositionColumns:
        """Return the same columns restricted to some of their positions."""

        return PositionColumns(
            game_ids=tuple(self.game_ids[row] for row in rows),
            ply_indices=tuple(self.ply_indices[row] for row in rows),
            target_action_ids=tuple(self.target_action_ids[row] for row in rows),
            conditioned_ratings=tuple(self.conditioned_ratings[row] for row in rows),
            legal_action_counts=self.legal_action_counts[rows],
            move_nll=self.move_nll[rows],
            legal_move_nll=self.legal_move_nll[rows],
            uniform_over_legal_move_nll=self.uniform_over_legal_move_nll[rows],
            mask_penalty=self.mask_penalty[rows],
            legal_mass=self.legal_mass[rows],
            legality_lift=self.legality_lift[rows],
            legal_margin=self.legal_margin[rows],
            top_illegal_fraction=self.top_illegal_fraction[rows],
            top1_illegal=self.top1_illegal[rows],
            target_rank=self.target_rank[rows],
        )

    @classmethod
    def from_records(
        cls,
        positions: Sequence[PositionPolicy],
    ) -> PositionColumns:
        """Return the same columns for positions already scored as records.

        A benchmark that holds records rather than a batch aggregates through
        the same columns, so a slice is summed one way rather than two.
        """

        def column(attribute: str, dtype: Any) -> NDArray[Any]:
            return np.fromiter(
                (getattr(position, attribute) for position in positions),
                dtype=dtype,
                count=len(positions),
            )

        return cls(
            game_ids=tuple(position.game_id for position in positions),
            ply_indices=tuple(position.ply_index for position in positions),
            target_action_ids=tuple(
                position.target_action_id for position in positions
            ),
            conditioned_ratings=tuple(
                position.conditioned_rating for position in positions
            ),
            legal_action_counts=column("legal_action_count", np.int64),
            move_nll=column("move_nll", np.float64),
            legal_move_nll=column("legal_move_nll", np.float64),
            uniform_over_legal_move_nll=column(
                "uniform_over_legal_move_nll", np.float64
            ),
            mask_penalty=column("mask_penalty", np.float64),
            legal_mass=column("legal_mass", np.float64),
            legality_lift=column("legality_lift", np.float64),
            legal_margin=column("legal_margin", np.float64),
            top_illegal_fraction=column("top_illegal_fraction", np.float64),
            top1_illegal=column("top1_illegal", np.bool_),
            target_rank=column("target_rank", np.int64),
        )


def score_columns(active: ActiveBatch) -> PositionColumns:
    """Return every reported quantity of one batch, reduced on its device.

    Each is a reduction over one position's row of the action vocabulary, so
    all of them are taken where the logits are and cross to the host together.
    """

    log_probabilities = torch.log_softmax(active.logits, dim=-1)
    move_nll = -log_probabilities.gather(1, active.target_index).squeeze(1)

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

    target_logits = active.logits.gather(1, active.target_index)
    better = (active.logits > target_logits) & active.legal_mask
    target_rank = better.sum(dim=-1) + 1

    legal_counts = active.legal_mask.sum(dim=-1)
    counted = legal_counts.to(dtype=torch.float64)
    legality_lift = _logit(legal_mass) - _logit(counted / MOVE_ACTION_COUNT)

    # One read rather than one per column: every column is the same length and
    # comes off the same pass, and reading each separately is a separate round
    # trip to the device the reduction ran on.
    values = (
        torch.stack(
            (
                move_nll,
                legal_move_nll,
                torch.log(counted),
                -log_legal_mass,
                legal_mass,
                legality_lift,
                legal_margin,
                top_illegal_fraction,
            )
        )
        .cpu()
        .numpy()
    )
    counts, ranks, illegal = (
        torch.stack((legal_counts, target_rank, top1_illegal.to(dtype=torch.long)))
        .cpu()
        .numpy()
    )
    return PositionColumns(
        game_ids=active.game_ids,
        ply_indices=active.ply_indices,
        target_action_ids=active.targets,
        conditioned_ratings=active.ratings,
        legal_action_counts=counts,
        move_nll=values[0],
        legal_move_nll=values[1],
        uniform_over_legal_move_nll=values[2],
        mask_penalty=values[3],
        legal_mass=values[4],
        legality_lift=values[5],
        legal_margin=values[6],
        top_illegal_fraction=values[7],
        top1_illegal=illegal.astype(np.bool_),
        target_rank=ranks,
    )


def score_positions(active: ActiveBatch) -> tuple[PositionPolicy, ...]:
    """Return one policy record per enabled action target in a batch."""

    if not active.legal_rows:
        return ()
    return position_records(score_columns(active))


def position_records(columns: PositionColumns) -> tuple[PositionPolicy, ...]:
    """Return one record per position of a scored batch.

    The reported quantities are summed as columns; a record is for a reader
    that wants one position, which is the detail tier and the benchmarks that
    hold a scored set rather than a batch.
    """

    move_nll = columns.move_nll.tolist()
    legal_move_nll = columns.legal_move_nll.tolist()
    uniform = columns.uniform_over_legal_move_nll.tolist()
    mask_penalty = columns.mask_penalty.tolist()
    legal_mass = columns.legal_mass.tolist()
    legality_lift = columns.legality_lift.tolist()
    legal_margin = columns.legal_margin.tolist()
    top_illegal_fraction = columns.top_illegal_fraction.tolist()
    top1_illegal = columns.top1_illegal.tolist()
    target_rank = columns.target_rank.tolist()
    legal_action_counts = columns.legal_action_counts.tolist()

    return tuple(
        PositionPolicy(
            game_id=columns.game_ids[offset],
            ply_index=columns.ply_indices[offset],
            target_action_id=columns.target_action_ids[offset],
            legal_action_count=legal_action_counts[offset],
            conditioned_rating=columns.conditioned_ratings[offset],
            move_nll=move_nll[offset],
            legal_move_nll=legal_move_nll[offset],
            uniform_over_legal_move_nll=uniform[offset],
            mask_penalty=mask_penalty[offset],
            legal_mass=legal_mass[offset],
            legality_lift=legality_lift[offset],
            legal_margin=legal_margin[offset],
            top1_illegal=top1_illegal[offset],
            top_illegal_fraction=top_illegal_fraction[offset],
            target_rank=target_rank[offset],
        )
        for offset in range(len(columns))
    )


def score_action_sets(
    active: ActiveBatch,
    action_sets: Mapping[tuple[int, int], Mapping[str, Collection[int]]],
) -> tuple[ActionSetPolicy, ...]:
    """Score named legal-action subsets without retaining whole policies.

    Every set a batch realizes is reduced in one pass. A set holds a handful of
    actions and a batch realizes hundreds of them, so reducing one at a time
    spends more on dispatch than on the arithmetic.
    """

    if not active.legal_rows:
        return ()

    probabilities = torch.softmax(active.logits, dim=-1)
    masked = active.logits.masked_fill(~active.legal_mask, -torch.inf)
    selected = torch.argmax(masked, dim=-1).tolist()

    named: list[tuple[int, str, int]] = []
    rows: list[int] = []
    members: list[int] = []
    owners: list[int] = []
    for offset, legal_actions in enumerate(active.legal_rows):
        key = (active.game_ids[offset], active.ply_indices[offset])
        candidates = action_sets.get(key)
        if not candidates:
            continue
        legal = frozenset(legal_actions)
        for name, action_ids in sorted(candidates.items()):
            actions = tuple(sorted(set(action_ids)))
            if any(action not in legal for action in actions):
                raise ValueError(
                    f"action set {name!r} contains an action that is not legal at {key}"
                )
            owners.extend([len(named)] * len(actions))
            named.append((offset, name, len(actions)))
            rows.extend([offset] * len(actions))
            members.extend(actions)
    if not named:
        return ()

    device = active.logits.device
    row_index = torch.tensor(rows, dtype=torch.long, device=device)
    member_index = torch.tensor(members, dtype=torch.long, device=device)
    owner_index = torch.tensor(owners, dtype=torch.long, device=device)
    mass = torch.zeros(len(named), dtype=probabilities.dtype, device=device).index_add_(
        0, owner_index, probabilities[row_index, member_index]
    )
    best = torch.full(
        (len(named),), -torch.inf, dtype=masked.dtype, device=device
    ).scatter_reduce_(0, owner_index, masked[row_index, member_index], reduce="amax")
    set_rows = torch.tensor(
        [offset for offset, _, _ in named], dtype=torch.long, device=device
    )
    ranks = (masked[set_rows] > best.unsqueeze(1)).sum(dim=-1) + 1

    mass_values = mass.tolist()
    rank_values = ranks.tolist()
    return tuple(
        ActionSetPolicy(
            game_id=active.game_ids[offset],
            ply_index=active.ply_indices[offset],
            name=name,
            selected_action_id=selected[offset],
            raw_probability_mass=mass_values[index],
            best_rank=rank_values[index] if size else None,
        )
        for index, (offset, name, size) in enumerate(named)
    )


def score_terminal_actions(active: ActiveBatch) -> tuple[TerminalActionPolicy, ...]:
    """Return the raw terminal-action mass at every scored decision.

    Raw rather than legal-masked, deliberately. The runtime samples from the
    masked policy, but a resignation reading is about how much the model wants
    to resign rather than about what the mask left it, and masking would rescale
    the quantity by whatever legality problem the checkpoint happens to have.
    """

    if not active.legal_rows:
        return ()

    probabilities = torch.softmax(active.logits, dim=-1)
    resignation, draw_claim = probabilities[
        :, [RESIGNATION_ACTION_ID, DRAW_CLAIM_ACTION_ID]
    ].T.tolist()
    return tuple(
        TerminalActionPolicy(
            game_id=active.game_ids[offset],
            ply_index=active.ply_indices[offset],
            target_action_id=active.targets[offset],
            resignation_mass=resignation[offset],
            draw_claim_mass=(
                draw_claim[offset] if DRAW_CLAIM_ACTION_ID in legal_actions else None
            ),
        )
        for offset, legal_actions in enumerate(active.legal_rows)
    )


@dataclass(frozen=True)
class TrajectoryColumns:
    """What two anchor conditionings say about every position in one batch.

    ``strength_signal`` is positive where the strong-rating conditioning
    explains the move actually played better than the weak one, which is the
    available proxy for how strong the play looked. ``alignment`` is positive
    where the policy at the position's true rating sits closer to the
    strong-conditioned policy than to the weak-conditioned one.
    """

    strength_signal: tuple[float, ...]
    alignment: tuple[float, ...]
    anchor_divergence: tuple[float, ...]
    anchor_agreement: tuple[bool, ...]


def treatment_move_losses(
    logits: Tensor,
    batch: MoveModelBatch,
    active: ActiveBatch,
) -> tuple[float, ...]:
    """Return each enabled position's move loss under one treatment.

    A conditioning treatment contributes one number per position to the
    dependency columns, so it is reduced where the logits already are and only
    that number is read back. Reading the whole action vocabulary across for
    each treatment moves about a thousand times what the reduction keeps.
    """

    enabled = _enabled_logits(logits, batch)
    losses = (
        -torch.log_softmax(enabled, dim=-1).gather(1, active.target_index).squeeze(1)
    ).cpu()
    if not torch.all(torch.isfinite(losses)):
        raise ValueError("a conditioning treatment produced a non-finite move loss")
    return tuple(losses.tolist())


def trajectory_columns(
    true_logits: Tensor,
    low_logits: Tensor,
    high_logits: Tensor,
    batch: MoveModelBatch,
    active: ActiveBatch,
) -> TrajectoryColumns:
    """Compare every position's policy at two anchor conditioning ratings.

    Each quantity is a reduction over one position's legal actions, so all four
    are taken on the device and only the four numbers per position come back.
    """

    true = _legal_log_policy(true_logits, batch, active)
    low = _legal_log_policy(low_logits, batch, active)
    high = _legal_log_policy(high_logits, batch, active)
    strength, alignment, divergence = torch.stack(
        (
            (high - low).gather(1, active.target_index).squeeze(1),
            _divergence(true, low, active) - _divergence(true, high, active),
            _divergence(low, high, active),
        )
    ).tolist()
    agreement = (torch.argmax(low, dim=-1) == torch.argmax(high, dim=-1)).tolist()
    return TrajectoryColumns(
        strength_signal=tuple(strength),
        alignment=tuple(alignment),
        anchor_divergence=tuple(divergence),
        anchor_agreement=tuple(agreement),
    )


def _enabled_logits(logits: Tensor, batch: MoveModelBatch) -> Tensor:
    """Return the enabled rows in float64, left on the device they came off."""

    return logits[batch.action_loss_mask].to(dtype=torch.float64)


def _legal_log_policy(
    logits: Tensor,
    batch: MoveModelBatch,
    active: ActiveBatch,
) -> Tensor:
    """Return each position's log policy over its own legal actions."""

    enabled = _enabled_logits(logits, batch)
    return torch.log_softmax(
        enabled.masked_fill(~active.legal_mask, -torch.inf), dim=-1
    )


def _divergence(
    reference: Tensor,
    candidate: Tensor,
    active: ActiveBatch,
) -> Tensor:
    """Return each position's Kullback-Leibler divergence between two policies.

    An illegal action carries no mass and a log probability of negative
    infinity in both policies, so its term is the zero the mask writes rather
    than the difference of two infinities.
    """

    terms = torch.exp(reference) * (reference - candidate)
    return torch.where(active.legal_mask, terms, 0.0).sum(dim=-1)


def top_action(log_probabilities: Tensor, legal_actions: Sequence[int]) -> int:
    """Return the action a greedy runtime would play from a legal policy."""

    if log_probabilities.shape != (len(legal_actions),):
        raise ValueError("a legal policy must align with its position's actions")
    return int(legal_actions[int(torch.argmax(log_probabilities).item())])


def _logit(probability: Tensor) -> Tensor:
    clamped = probability.clamp(_PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    return torch.log(clamped / (1.0 - clamped))


def active_batch(logits: Tensor, batch: MoveModelBatch) -> ActiveBatch:
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

    active_logits = _active_logits(logits, batch)

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

    return ActiveBatch(
        logits=active_logits,
        legal_mask=_legal_mask(legal_rows, active_logits),
        target_index=torch.tensor(
            targets, dtype=torch.long, device=active_logits.device
        ).unsqueeze(1),
        legal_rows=tuple(legal_rows),
        targets=targets,
        game_ids=tuple(game_ids),
        ply_indices=tuple(ply_indices),
        ratings=_ratings(rating_values, rating_present),
    )


def _legal_mask(legal_rows: Sequence[Sequence[int]], active_logits: Tensor) -> Tensor:
    """Return one row per enabled position, marking its legal actions.

    Written in a single indexed assignment rather than a row at a time. A row
    holds a few dozen legal actions and a batch holds hundreds of rows -- the
    evaluation defaults reach 610 enabled rows at the median -- so a per-row
    write spends more of the host building index tensors and dispatching
    kernels than on the mask itself, about four times more at that shape.
    """

    device = active_logits.device
    legal_mask = torch.zeros_like(active_logits, dtype=torch.bool)
    rows = torch.repeat_interleave(
        torch.arange(len(legal_rows), dtype=torch.long, device=device),
        torch.tensor([len(row) for row in legal_rows], dtype=torch.long, device=device),
    )
    columns = torch.tensor(
        [action for row in legal_rows for action in row],
        dtype=torch.long,
        device=device,
    )
    legal_mask[rows, columns] = True
    return legal_mask


def _active_logits(logits: Tensor, batch: MoveModelBatch) -> Tensor:
    """Return the enabled rows in float64, on the device they came off.

    Every quantity read off them is a reduction over the action vocabulary, so
    they are reduced where they already are and only the results come back. A
    batch of held-out positions carries about two thousand logits each and
    yields ten numbers, so moving the vocabulary instead is a thousandfold of
    the traffic.
    """

    active_logits = logits[batch.action_loss_mask].detach().to(dtype=torch.float64)
    # One read for the whole batch, rather than the per-position reads the
    # quantities below are reduced on the device to avoid.
    if not bool(torch.all(torch.isfinite(active_logits)).cpu()):
        raise ValueError("enabled action logits must all be finite")
    return active_logits


def _active_ratings(batch: MoveModelBatch) -> tuple[int | None, ...]:
    """Return the rating each enabled row carries, in one device read."""

    rating = batch.inputs.target_rating
    values, present = (
        torch.stack(
            (
                rating.values.to(dtype=torch.long),
                rating.present.to(dtype=torch.long),
            )
        )[:, batch.action_loss_mask]
        .detach()
        .cpu()
        .tolist()
    )
    return _ratings(values, [bool(value) for value in present])


def _ratings(
    values: Sequence[int],
    present: Sequence[bool],
) -> tuple[int | None, ...]:
    """Return the rating the model saw at each enabled row, absent as ``None``."""

    if any(
        is_present and value < 0
        for value, is_present in zip(values, present, strict=True)
    ):
        raise ValueError("present player ratings must be nonnegative")
    return tuple(
        value if is_present else None
        for value, is_present in zip(values, present, strict=True)
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
