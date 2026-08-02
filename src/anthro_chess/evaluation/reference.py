"""The human side of a generated-play comparison, and the quantities compared.

A rollout scalar says what a checkpoint did. It does not say whether that is
*human*, and it cannot: the project has deliberately declined to hardcode a
target draw rate or a target game length, because human games are the reference
for all of them. This module supplies that reference and the quantity
definitions both sides are read through.

One definition per quantity, used for both sides. That is the point of the
module: a game length measured one way on generated games and another way on
human games would produce a distance that is partly an artifact of the two
implementations. Every quantity here is a pure function of a trajectory, plus
the result, which both sides carry.

The human side is read straight from the frozen pool rather than reconstructed
into game records. A human game has no seat configuration, no seed, and no
ending this project's harness vocabulary can express — "lost on time" is not a
rule outcome — so building a record would mean inventing all three. The shared
trajectory analysis is what makes that unnecessary.

Rating is the curve's axis, so a reference game needs one. Games are placed at
the mean of the two players' normalized ratings, and a game whose players are
far apart is excluded rather than averaged into the middle: it is a mismatch
rather than a game at the average of the two, and its behavior belongs to
neither rating.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any

from pydantic import Field, StrictBool, StrictInt

from anthro_chess.config import ConfigModel
from anthro_chess.data.schema import NormalizedColumn
from anthro_chess.evaluation.curves import (
    BandwidthSelection,
    CurveQuantity,
    CurveSpec,
    Observation,
    select_neighbours,
)
from anthro_chess.evaluation.games import (
    GameFeatures,
    TrajectoryFeatures,
    analyze_trajectory,
)
from anthro_chess.evaluation.openings import OpeningBook, OpeningLevel

REFERENCE_VERSION = 1

logger = logging.getLogger(__name__)


class ReferenceError(ValueError):
    """Raised when the human reference cannot be built as configured."""


class ComparedQuantity(StrEnum):
    """What a generated-play curve comparison measures.

    Deliberately not everything a rollout reports. A quantity belongs here when
    a human game and a generated game both have it and the two are comparable:
    the harness's ply limit has no human counterpart, and how a game *ended* is
    a richer vocabulary on the human side than the model can produce, so
    termination is measured separately rather than folded in here.
    """

    #: Moves per game. The most legible human-likeness reading there is, and
    #: censored by the ply limit: a suite that adjudicates most of its games
    #: reports a lower bound rather than an estimate, so read this beside the
    #: unfinished rate.
    GAME_LENGTH = "game-length"
    #: The result distribution. A model that draws far more or far less than
    #: humans of its rating is not playing like them, whichever way it errs.
    #: Also gated by the unfinished rate: an adjudicated game contributes a
    #: result humans essentially never produce, so a suite that does not finish
    #: its games measures the ply limit here rather than the model. Measured on
    #: the proof run, 72% of generated games hit the limit against 0.075% of
    #: human games in the same pool.
    RESULT = "result"
    #: Share of plies that revisited a position. Humans repeat, so the target
    #: is the human rate rather than zero.
    REPETITION = "repetition"
    #: How much of the play from first recurrence onward stayed in the cycle.
    CYCLE = "cycle"
    #: Which openings get chosen, grouped by classified family. Destinations
    #: only: a game that left the book on a position several openings still
    #: pass through has not chosen anything, and counting it here would let a
    #: depth effect into a distribution that is supposed to measure preference.
    #: Games with no destination contribute nothing rather than a category, so
    #: this quantity's observation counts are below the game count by the
    #: waypoint rate.
    REPERTOIRE = "repertoire"
    #: Whether the game stopped on a waypoint. The behavior the repertoire
    #: distribution deliberately excludes, reported on its own because it is
    #: real and strongly rating-sensitive — just not a choice.
    WAYPOINT = "waypoint"
    #: How deep into catalogued theory the game got, in book plies.
    BOOK_DEPTH = "book-depth"
    #: How deep the book still ran onward from where the game left it. Raw
    #: depth conflates choosing a well-analyzed line with knowing it, and this
    #: is the half that says which line was chosen.
    BOOK_AVAILABLE_DEPTH = "book-available-depth"
    #: The share of that available theory the game actually consumed.
    BOOK_CONSUMED_FRACTION = "book-consumed-fraction"
    #: Distinct moves as a share of moves played.
    MOVE_DIVERSITY = "move-diversity"

    @property
    def kind(self) -> CurveQuantity:
        """Return whether this quantity is a scalar or a category per game."""

        return _QUANTITY_KINDS[self]


_QUANTITY_KINDS: Mapping[ComparedQuantity, CurveQuantity] = {
    ComparedQuantity.GAME_LENGTH: CurveQuantity.SCALAR,
    ComparedQuantity.RESULT: CurveQuantity.CATEGORICAL,
    ComparedQuantity.REPETITION: CurveQuantity.SCALAR,
    ComparedQuantity.CYCLE: CurveQuantity.SCALAR,
    ComparedQuantity.REPERTOIRE: CurveQuantity.CATEGORICAL,
    ComparedQuantity.WAYPOINT: CurveQuantity.SCALAR,
    ComparedQuantity.BOOK_DEPTH: CurveQuantity.SCALAR,
    ComparedQuantity.BOOK_AVAILABLE_DEPTH: CurveQuantity.SCALAR,
    ComparedQuantity.BOOK_CONSUMED_FRACTION: CurveQuantity.SCALAR,
    ComparedQuantity.MOVE_DIVERSITY: CurveQuantity.SCALAR,
}


@dataclass(frozen=True)
class ComparableGame:
    """One game reduced to what a curve comparison reads from it.

    Both sides produce this, which is what keeps a quantity one quantity. The
    generated side fills ``rating`` from the configured conditioning rating and
    the human side from the players' own.
    """

    rating: float
    result: str
    trajectory: TrajectoryFeatures

    def value(
        self,
        quantity: ComparedQuantity,
        *,
        level: OpeningLevel = OpeningLevel.FAMILY,
    ) -> float | str | None:
        """Return this game's contribution to one compared quantity.

        ``None`` means the game has nothing to say about this quantity, which
        is not the same as a zero: a game that stopped on a waypoint made no
        repertoire choice, and one the book never named has no depth. Those
        games are left out of the quantity rather than given a value that would
        be read as one.
        """

        repetition = self.trajectory.repetition
        opening = self.trajectory.opening
        if quantity is ComparedQuantity.GAME_LENGTH:
            return float(self.trajectory.ply_count)
        if quantity is ComparedQuantity.RESULT:
            return self.result
        if quantity is ComparedQuantity.REPETITION:
            return repetition.repeated_ply_fraction
        if quantity is ComparedQuantity.CYCLE:
            return repetition.cycle_ply_fraction
        if quantity is ComparedQuantity.REPERTOIRE:
            return opening.destination(level)
        if quantity is ComparedQuantity.WAYPOINT:
            return float(opening.waypoint(level))
        if quantity is ComparedQuantity.BOOK_DEPTH:
            return float(opening.book_ply) if opening.classified else None
        if quantity is ComparedQuantity.BOOK_AVAILABLE_DEPTH:
            return float(opening.available_ply) if opening.classified else None
        if quantity is ComparedQuantity.BOOK_CONSUMED_FRACTION:
            return opening.consumed_fraction if opening.classified else None
        return self.trajectory.distinct_move_fraction

    def observation(
        self,
        quantity: ComparedQuantity,
        *,
        level: OpeningLevel = OpeningLevel.FAMILY,
    ) -> Observation | None:
        """Return this game as one observation of a curve, when it has one."""

        value = self.value(quantity, level=level)
        return None if value is None else Observation(rating=self.rating, value=value)


class ReferenceConfig(ConfigModel):
    """Which human games a comparison is measured against."""

    #: Compare generated play against human play at all. Off leaves the rollout
    #: scalars, which are still worth recording; on is what turns them into a
    #: statement about human-likeness.
    enabled: StrictBool = True
    #: Widest gap between the two players' ratings a reference game may have.
    #: A lopsided game is a mismatch rather than a game at the average of its
    #: two ratings, and its length and result belong to neither player's level.
    maximum_rating_gap: Annotated[StrictInt, Field(ge=0)] = 200
    #: Resamples behind the floors and reference levels every distance is read
    #: against. A sample count, so it is provenance rather than identity.
    resamples: Annotated[StrictInt, Field(ge=0)] = 200
    seed: Annotated[StrictInt, Field(ge=0)] = 0


@dataclass(frozen=True)
class HumanReference:
    """The human games one suite is compared against, and what excluded the rest."""

    games: tuple[ComparableGame, ...]
    excluded: dict[str, int]

    def observations(
        self,
        quantity: ComparedQuantity,
        *,
        level: OpeningLevel = OpeningLevel.FAMILY,
    ) -> tuple[Observation, ...]:
        """Return the reference observations for one quantity.

        Shorter than the reference for a quantity some games do not have. That
        is the intended shape: both sides drop the same games for the same
        reason, so the comparison stays like for like.
        """

        return observations_for(self.games, quantity, level=level)

    def as_record(self) -> dict[str, Any]:
        """Return the provenance record stored with a comparison."""

        ratings = sorted(game.rating for game in self.games)
        return {
            "version": REFERENCE_VERSION,
            "games": len(self.games),
            "excluded_games": dict(sorted(self.excluded.items())),
            "rating_range": (None if not ratings else [ratings[0], ratings[-1]]),
        }


def human_reference(
    rows: Sequence[Mapping[str, Any]],
    config: ReferenceConfig,
    *,
    book: OpeningBook | None = None,
) -> HumanReference:
    """Reduce frozen pool rows to the reference a curve comparison reads.

    Exclusions are counted rather than dropped silently, because a reference
    that quietly lost half its games to a missing rating field would shift
    every curve without saying so.
    """

    games: list[ComparableGame] = []
    excluded: dict[str, int] = {}
    for row in rows:
        game, reason = _comparable_game(row, config, book=book)
        if game is None:
            excluded[reason or "unusable"] = excluded.get(reason or "unusable", 0) + 1
            continue
        games.append(game)
    if not games:
        raise ReferenceError(
            "no human game in the reference view carries the ratings and moves "
            "a curve comparison needs"
        )
    logger.info(
        "Human reference: %s game(s), %s excluded",
        len(games),
        sum(excluded.values()),
    )
    return HumanReference(games=tuple(games), excluded=excluded)


def observations_for(
    games: Sequence[ComparableGame],
    quantity: ComparedQuantity,
    *,
    level: OpeningLevel = OpeningLevel.FAMILY,
) -> tuple[Observation, ...]:
    """Return one quantity's observations, skipping games that lack it."""

    collected = (game.observation(quantity, level=level) for game in games)
    return tuple(observation for observation in collected if observation is not None)


def generated_games(
    features: Sequence[GameFeatures],
    *,
    rating: int,
) -> tuple[ComparableGame, ...]:
    """Return generated games as comparable ones at their conditioning rating.

    The model side's rating is the rating it was *told* to play at, which is
    what makes the comparison a test of the dial rather than of an estimate.
    """

    return tuple(
        ComparableGame(
            rating=float(rating),
            result=str(feature.result),
            trajectory=feature.trajectory,
        )
        for feature in features
    )


def _comparable_game(
    row: Mapping[str, Any],
    config: ReferenceConfig,
    *,
    book: OpeningBook | None,
) -> tuple[ComparableGame | None, str | None]:
    white = row.get(NormalizedColumn.WHITE_NORMALIZED_RATING.value)
    black = row.get(NormalizedColumn.BLACK_NORMALIZED_RATING.value)
    if white is None or black is None:
        return None, "missing_ratings"
    if abs(int(white) - int(black)) > config.maximum_rating_gap:
        return None, "rating_gap"
    action_ids = row.get(NormalizedColumn.ACTION_IDS.value)
    if not action_ids:
        return None, "no_moves"
    trajectory = analyze_trajectory(
        str(row[NormalizedColumn.INITIAL_POSITION.value]),
        [int(value) for value in action_ids],
        book=book,
    )
    return (
        ComparableGame(
            rating=(float(white) + float(black)) / 2.0,
            result=str(row[NormalizedColumn.RESULT.value]),
            trajectory=trajectory,
        ),
        None,
    )


#: Version of the declared curve shape. Bumping it ends every generated-play
#: curve series, so it changes only when the bandwidth does.
CURVE_SPEC_VERSION = 1

#: Bandwidth selected once from the human reference by cross-validation and then
#: frozen, per `docs/evaluation.md`. Re-selecting per run would mean two
#: checkpoints were measured differently, so this is code rather than
#: configuration and changing it is a benchmark version bump. Reproduce with
#: `anthro eval curve-bandwidth`.
#:
#: One bandwidth for every quantity rather than six, chosen deliberately. Run
#: over 6,804 matched-rating games of the frozen blitz test pool, only three
#: quantities have an interior optimum — game length at 1024, opening and move
#: diversity at 2048 — while result and cycle improve monotonically out to the
#: largest candidate, because human result and cycle behavior barely varies with
#: rating at all. Freezing those at the boundary would make their neighbourhood
#: most of the reference at every point, which is a global average wearing the
#: shape of a curve, and would collapse the conditional reading into the pooled
#: one it is supposed to be read against.
#:
#: 1024 is game length's own optimum and costs every other quantity at most
#: 0.20% against its own best cross-validation error (result 0.20%, repetition
#: 0.15%, cycle 0.15%, move diversity 0.09%, opening 0.04%). Their error curves
#: are flat enough over this range that the choice sits inside the noise of
#: their optima, so one explainable number beats six boundary artifacts.
#:
#: The repertoire, waypoint, and book-depth quantities inherit the same number
#: rather than one selected for them. Selection needs the real pool, which this
#: change had no access to; inheriting a declared bandwidth is what the single
#: shared value was for, and re-running `anthro eval curve-bandwidth` against
#: the pool is what would replace it. A later selection that moves the number
#: is a spec version bump, which is the mechanism that keeps the two readings
#: from being silently merged.
DECLARED_NEIGHBOURS: Mapping[ComparedQuantity, int] = {
    quantity: 1024 for quantity in ComparedQuantity
}


def minimum_reference_games(ratings: Sequence[float], neighbours: int) -> int:
    """Return the smallest reference a grid of this size can be estimated on.

    One full bandwidth per grid point. At a neighbour-count bandwidth the
    reference size *is* the smoothing radius: a reference only a little larger
    than the bandwidth puts most of itself inside every point's neighbourhood,
    so adjacent points are estimated from largely the same games and the
    conditional curve resolves fewer points than it plots. Below this the
    neighbourhoods cannot be disjoint however the ratings fall.

    A floor rather than a target. A reference exactly this size still overlaps
    wherever the corpus is thin, which is why the realized bandwidth is reported
    per grid point instead of being assumed from the count.
    """

    return neighbours * len({float(rating) for rating in ratings})


def curve_spec(
    quantity: ComparedQuantity,
    ratings: Sequence[float],
) -> CurveSpec:
    """Return one quantity's comparison shape, evaluated where the model played.

    The evaluation points are the conditioning ratings the suite actually
    generated games at, rather than a separately declared grid. Generated games
    are produced on demand and none of them is committed, so there is no reason
    to estimate the comparison anywhere the model was not asked to play: such a
    point has a human curve and nothing to compare it against, and it would
    drop out of the reading anyway. A declared grid would also be a second list
    that has to agree with the configured ratings and nothing forcing it to.

    The points are still part of series identity, because the distance is a
    mean over them. They reach the fingerprint through the reading's declared
    workload, which already carries the rating grid, so nothing extra is needed
    here. The **bandwidth** stays declared and frozen: it is the smoothing
    rather than the points, and re-selecting it per run would mean two
    checkpoints were measured differently.
    """

    try:
        neighbours = DECLARED_NEIGHBOURS[quantity]
    except KeyError:  # pragma: no cover - the mapping covers the enum
        raise ReferenceError(f"no bandwidth is declared for {quantity}") from None
    points = tuple(sorted(float(rating) for rating in dict.fromkeys(ratings)))
    if len(points) < 2:
        raise ReferenceError(
            "a curve needs at least two conditioning ratings to be estimated "
            "over; a suite that played one rating has a point, not a curve"
        )
    return CurveSpec(
        name=f"generated-play-{quantity.value}",
        version=CURVE_SPEC_VERSION,
        quantity=quantity.kind,
        neighbours=neighbours,
        grid=points,
    )


def select_bandwidths(
    reference: HumanReference,
    *,
    level: OpeningLevel = OpeningLevel.FAMILY,
    candidates: Sequence[int] | None = None,
) -> dict[ComparedQuantity, BandwidthSelection]:
    """Choose each quantity's bandwidth from the human reference.

    The offline step behind ``DECLARED_NEIGHBOURS``. Its output is quoted into
    code and frozen rather than consumed at run time; a benchmark that selected
    its own bandwidth would measure two checkpoints differently.
    """

    selections: dict[ComparedQuantity, BandwidthSelection] = {}
    for quantity in ComparedQuantity:
        observations = reference.observations(quantity, level=level)
        selections[quantity] = select_neighbours(
            observations,
            quantity=quantity.kind,
            **({} if candidates is None else {"candidates": candidates}),
        )
    return selections


def iter_quantities() -> Iterator[ComparedQuantity]:
    """Yield every compared quantity in reporting order."""

    yield from ComparedQuantity


__all__ = [
    "CURVE_SPEC_VERSION",
    "DECLARED_NEIGHBOURS",
    "REFERENCE_VERSION",
    "ComparableGame",
    "ComparedQuantity",
    "HumanReference",
    "ReferenceConfig",
    "ReferenceError",
    "curve_spec",
    "generated_games",
    "human_reference",
    "iter_quantities",
    "minimum_reference_games",
    "observations_for",
    "select_bandwidths",
]
