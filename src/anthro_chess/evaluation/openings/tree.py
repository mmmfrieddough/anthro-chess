"""The shallow repertoire, computed exactly rather than sampled.

A model's policy at a fixed position is one forward pass, so the distribution
over openings it would play is a property of the policy rather than something
that has to be estimated from games. Walking the opening tree and keeping every
line above a cumulative-probability threshold produces that distribution with no
sampling noise at all, plus a bound on how much of it the pruning could have
moved.

This is why repertoire and book depth are separate readings rather than one.
They differ in meaning — which opening was chosen against how far into it the
game stayed — and they differ in computational character too: the shallow end is
exactly computable this way, while a deep reading has a branching factor no
threshold tames and must still be sampled from played games.

The walk asks a caller for policies rather than owning a model. Batching several
positions into one forward pass, holding sessions, and applying the legal mask
all belong to the runtime; what belongs here is the tree, the pruning, and the
labeling. It also makes the walk testable against a policy that is written down
rather than trained.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import chess

from anthro_chess.evaluation.openings.book import OpeningBook, load_book
from anthro_chess.evaluation.openings.classification import (
    UNCLASSIFIED_LABEL,
    classify_moves,
)
from anthro_chess.evaluation.openings.names import OpeningLevel

OPENING_TREE_VERSION = 1


class OpeningTreeError(ValueError):
    """Raised when an opening-tree walk cannot be carried out as asked."""


class ActionPolicy(Protocol):
    """Supply the move distribution each of several prefixes continues with."""

    def __call__(
        self,
        prefixes: Sequence[tuple[chess.Move, ...]],
    ) -> Sequence[Mapping[chess.Move, float]]: ...


@dataclass(frozen=True)
class RepertoireWalk:
    """One exact repertoire reading, and the bound its pruning implies."""

    plies: int
    threshold: float
    level: OpeningLevel
    #: Probability mass per destination label, and the mass that stopped on a
    #: waypoint. The two sum to one: a line that ran out of depth, terminated,
    #: or was pruned still carries the label it had reached.
    destinations: Mapping[str, float]
    waypoint_mass: float
    #: Mass that stopped being expanded because it fell below the threshold.
    #: The assumption-free bound: if pruned lines could go anywhere, no label's
    #: share can be wrong by more than this.
    pruned_mass: float
    #: The bound that is actually worth reading. Pruned mass whose label had not
    #: been committed yet — it sat on a waypoint, or off book entirely — so
    #: continuing it could still move the distribution. Mass pruned on a
    #: destination cannot: a destination is a position with one reachable label
    #: at this level, so playing on keeps the label it already has.
    #:
    #: This is far tighter than ``pruned_mass`` and it matters. Measured on a
    #: real checkpoint, the assumption-free bound sits near one: probability
    #: disperses across dozens of legal moves per ply, so most individual lines
    #: fall below any affordable threshold even while the dominant ones reach
    #: full depth. Most of that pruned mass has already picked a family, which
    #: is why the two numbers are so far apart — 0.96 against 0.38 at one
    #: measured setting.
    #:
    #: Reachability here is the book's own canonical-path notion, the same one
    #: that separates a waypoint from a destination. A game that transposed out
    #: of a destination into another family by a route no book entry takes would
    #: escape it; that is a narrow case, and treating it otherwise would mean
    #: two different definitions of reachability in one reading.
    unsettled_mass: float
    #: Lines that ended before the ply limit because the model put mass on a
    #: non-move action. Reported apart from pruning, which is an approximation,
    #: because this is behavior.
    terminal_mass: float
    positions_evaluated: int
    #: Deepest ply the walk actually expanded to. Normally equal to ``plies``,
    #: since the dominant line usually stays above the threshold the whole way;
    #: it falls short when the policy is flat enough that even the best line
    #: disperses first. Reported because nothing else in the record would say
    #: that a walk declaring eight plies had stopped at three.
    deepest_expanded_ply: int
    lines: int

    def repertoire(self) -> dict[str, float]:
        """Return the destination distribution, renormalized over choices.

        The human side of this comparison counts only games that reached a
        destination, so the model side has to be conditional on the same thing;
        comparing a distribution that keeps its waypoint mass against one that
        never had any would report the waypoint rate as a repertoire
        difference.
        """

        total = sum(self.destinations.values())
        if total <= 0.0:
            return {}
        return {
            label: mass / total for label, mass in sorted(self.destinations.items())
        }

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one exact walk."""

        return {
            "version": OPENING_TREE_VERSION,
            "plies": self.plies,
            "threshold": self.threshold,
            "level": self.level.value,
            "destination_mass": dict(sorted(self.destinations.items())),
            "repertoire": self.repertoire(),
            "waypoint_mass": self.waypoint_mass,
            "pruned_mass": self.pruned_mass,
            "unsettled_mass": self.unsettled_mass,
            "terminal_mass": self.terminal_mass,
            "positions_evaluated": self.positions_evaluated,
            "deepest_expanded_ply": self.deepest_expanded_ply,
            "lines": self.lines,
        }


def walk_repertoire(
    policy: ActionPolicy,
    *,
    plies: int,
    threshold: float,
    level: OpeningLevel = OpeningLevel.FAMILY,
    book: OpeningBook | None = None,
) -> RepertoireWalk:
    """Enumerate the openings a policy would play, exactly above a threshold.

    Lines are expanded breadth-first so every position at one depth can be
    handed to the caller in a single batch, which is the difference between one
    forward pass per position and one per position with no batching at all.

    Prefixes are never merged across transpositions. The policy conditions on
    the trajectory rather than on the position alone, so two move orders that
    reach the same board are two different questions to ask it.
    """

    if plies < 1:
        raise OpeningTreeError("an opening-tree walk needs at least one ply")
    if not 0.0 < threshold <= 1.0:
        raise OpeningTreeError(
            "an opening-tree walk needs a threshold above zero and at most one"
        )
    resolved = load_book() if book is None else book

    destinations: dict[str, float] = {}
    waypoint_mass = 0.0
    pruned_mass = 0.0
    unsettled_mass = 0.0
    terminal_mass = 0.0
    evaluated = 0
    deepest = 0
    lines = 0

    def settle(
        prefix: tuple[chess.Move, ...],
        mass: float,
        *,
        pruned: bool = False,
    ) -> None:
        nonlocal waypoint_mass, unsettled_mass, lines
        lines += 1
        label = (
            UNCLASSIFIED_LABEL
            if not prefix
            else classify_moves(prefix, book=resolved, plies=plies)
        )
        destination = label.destination(level)
        # A line stopped short can still move the distribution unless the book
        # has already committed its label. Off book counts as uncommitted: it
        # can transpose back into a named position later.
        if pruned and (destination is None or not label.classified):
            unsettled_mass += mass
        if destination is None:
            waypoint_mass += mass
            return
        destinations[destination] = destinations.get(destination, 0.0) + mass

    frontier: dict[tuple[chess.Move, ...], float] = {(): 1.0}
    for depth in range(plies):
        survivors: list[tuple[tuple[chess.Move, ...], float]] = []
        for prefix, mass in frontier.items():
            if mass < threshold:
                pruned_mass += mass
                settle(prefix, mass, pruned=True)
                continue
            survivors.append((prefix, mass))
        if not survivors:
            # Everything left was pruned and settled just above, so the frontier
            # is empty rather than pending: leaving it populated would settle
            # the same mass a second time in the trailing pass.
            frontier = {}
            break
        deepest = depth + 1
        evaluated += len(survivors)
        distributions = policy([prefix for prefix, _ in survivors])
        if len(distributions) != len(survivors):
            raise OpeningTreeError(
                "the policy returned "
                f"{len(distributions)} distribution(s) for {len(survivors)} "
                "position(s)"
            )
        expanded: dict[tuple[chess.Move, ...], float] = {}
        for (prefix, mass), distribution in zip(survivors, distributions, strict=True):
            moved = 0.0
            for move, probability in distribution.items():
                if probability <= 0.0:
                    continue
                moved += probability
                child = (*prefix, move)
                expanded[child] = expanded.get(child, 0.0) + mass * probability
            if moved > 1.0 + _MASS_TOLERANCE:
                raise OpeningTreeError(
                    "a policy distribution put more than unit mass on moves"
                )
            # Whatever the policy did not spend on a move ended the line: a
            # resignation, or any other non-move action the position enabled.
            remainder = max(0.0, 1.0 - moved)
            if remainder > 0.0:
                terminal_mass += mass * remainder
                settle(prefix, mass * remainder)
        frontier = expanded

    for prefix, mass in frontier.items():
        settle(prefix, mass)

    # These are probabilities by construction, but a walk settles thousands of
    # tiny masses and float addition is not associative, so the last bit can
    # land just outside the unit interval. Clamping keeps a reported bound from
    # reading as more than all of the mass.
    return RepertoireWalk(
        plies=plies,
        threshold=threshold,
        level=level,
        destinations=dict(sorted(destinations.items())),
        waypoint_mass=_unit(waypoint_mass),
        pruned_mass=_unit(pruned_mass),
        unsettled_mass=_unit(unsettled_mass),
        terminal_mass=_unit(terminal_mass),
        positions_evaluated=evaluated,
        deepest_expanded_ply=deepest,
        lines=lines,
    )


def _unit(mass: float) -> float:
    """Return one accumulated probability mass, held inside the unit interval."""

    return min(1.0, max(0.0, mass))


#: Slack allowed when checking that a policy's move probabilities are a
#: sub-distribution. Softmax over float32 logits does not sum to exactly one.
_MASS_TOLERANCE = 1e-6


__all__ = [
    "OPENING_TREE_VERSION",
    "ActionPolicy",
    "OpeningTreeError",
    "RepertoireWalk",
    "walk_repertoire",
]
