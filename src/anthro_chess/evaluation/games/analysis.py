"""Distribution features computed from game records rather than from games.

Rollout distribution tests, human-likeness feature matching, and later
preference-control evaluation all want the same list of things about a set of
games. Computing them from retained records is what makes a new feature cheap:
it is recomputed over games already on disk instead of by regenerating them,
which for a model rollout is the difference between seconds and hours.

Everything here replays moves, which is exact and cheap. Nothing here consults
a model, so a benchmark's analysis pass has no checkpoint dependency and human
reference games can be run through the identical code path.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import chess

from anthro_chess.chess import decode_move
from anthro_chess.evaluation.games.records import (
    GameRecord,
    GameResult,
    GameTermination,
)
from anthro_chess.evaluation.openings import (
    OpeningBook,
    OpeningLabel,
    OpeningLevel,
    classify_action_ids,
    opening_book_identity,
    opening_distribution,
)

GAME_ANALYSIS_VERSION = 1


@dataclass(frozen=True)
class RepetitionDiagnostics:
    """How much of a game revisited positions it had already reached.

    Reported for every game rather than only for games that ended in a
    repetition draw, because the interesting failure is a model that shuffles
    for twenty plies and then finds something else: it never terminates by
    repetition, and an aggregate draw rate cannot see it.
    """

    #: Ply index at which some position first occurred a second time.
    first_repetition_ply: int | None
    #: Ply index at which some position first occurred a third time, which is
    #: when a draw first became claimable.
    threefold_ply: int | None
    #: The largest number of times any one position occurred.
    maximum_occurrences: int
    #: Share of plies whose resulting position had been reached before.
    repeated_ply_fraction: float
    #: Share of the plies from the first recurrence onward that were themselves
    #: recurrences. This separates a game that touched a position twice and
    #: moved on from one that stayed in a cycle, which the whole-game fraction
    #: cannot: a long game that shuffles at the end dilutes to a small number
    #: there while reading near one here.
    cycle_ply_fraction: float

    @property
    def repeated(self) -> bool:
        """Return whether any position occurred more than once."""

        return self.first_repetition_ply is not None

    @property
    def threefold_claimable(self) -> bool:
        """Return whether a threefold draw ever became claimable."""

        return self.threefold_ply is not None


@dataclass(frozen=True)
class GameFeatures:
    """The distribution features one game contributes."""

    game_id: str
    ply_count: int
    generated_plies: int
    result: GameResult
    termination: GameTermination
    adjudicated: bool
    opening: OpeningLabel
    repetition: RepetitionDiagnostics
    #: Distinct moves as a share of moves played. A game that shuffles two
    #: pieces and one that develops sixteen differ here well before either
    #: reaches a repetition threshold.
    distinct_move_fraction: float

    def as_record(self) -> dict[str, Any]:
        """Return the per-game record stored in the detail tier."""

        return {
            "game_id": self.game_id,
            "ply_count": self.ply_count,
            "generated_plies": self.generated_plies,
            "result": self.result,
            "termination": self.termination.value,
            "adjudicated": self.adjudicated,
            "opening": self.opening.as_record(),
            "repetition": {
                "first_repetition_ply": self.repetition.first_repetition_ply,
                "threefold_ply": self.repetition.threefold_ply,
                "maximum_occurrences": self.repetition.maximum_occurrences,
                "repeated_ply_fraction": self.repetition.repeated_ply_fraction,
                "cycle_ply_fraction": self.repetition.cycle_ply_fraction,
            },
            "distinct_move_fraction": self.distinct_move_fraction,
        }


@dataclass(frozen=True)
class GameDistribution:
    """What a set of games looks like, in the shape a benchmark reports."""

    version: int
    games: int
    mean_ply_count: float
    mean_generated_plies: float
    result_counts: dict[str, int]
    termination_counts: dict[str, int]
    adjudicated_games: int
    opening_level: OpeningLevel
    opening_counts: dict[str, int]
    opening_book: dict[str, object]
    repeated_games: int
    threefold_claimable_games: int
    mean_repeated_ply_fraction: float
    #: Averaged over the games that repeated at all, so it says how deep the
    #: cycles ran rather than how often they happened. Averaging it over every
    #: game would fold those two questions into one number that answers
    #: neither. Zero when no game repeated.
    mean_cycle_ply_fraction: float
    mean_distinct_move_fraction: float
    #: Distinct move sequences as a share of games. A suite whose games are all
    #: the same trajectory reports one here, which is the signature of a seed
    #: that did not vary or a temperature that collapsed the policy.
    distinct_game_fraction: float

    def as_record(self) -> dict[str, Any]:
        """Return the summary record a benchmark stores beside its metrics."""

        return {
            "version": self.version,
            "games": self.games,
            "mean_ply_count": self.mean_ply_count,
            "mean_generated_plies": self.mean_generated_plies,
            "result_counts": dict(self.result_counts),
            "termination_counts": dict(self.termination_counts),
            "adjudicated_games": self.adjudicated_games,
            "opening_level": self.opening_level.value,
            "opening_counts": dict(self.opening_counts),
            "opening_book": dict(self.opening_book),
            "repeated_games": self.repeated_games,
            "threefold_claimable_games": self.threefold_claimable_games,
            "mean_repeated_ply_fraction": self.mean_repeated_ply_fraction,
            "mean_cycle_ply_fraction": self.mean_cycle_ply_fraction,
            "mean_distinct_move_fraction": self.mean_distinct_move_fraction,
            "distinct_game_fraction": self.distinct_game_fraction,
        }


def analyze_game(
    record: GameRecord,
    *,
    book: OpeningBook | None = None,
) -> GameFeatures:
    """Return the distribution features of one recorded game."""

    moves = record.move_action_ids
    return GameFeatures(
        game_id=record.game_id,
        ply_count=record.ply_count,
        generated_plies=record.generated_plies,
        result=record.outcome.result,
        termination=record.outcome.termination,
        adjudicated=record.outcome.adjudicated,
        opening=classify_action_ids(
            record.action_ids,
            initial_position=record.initial_position,
            book=book,
        ),
        repetition=_repetition_diagnostics(record),
        distinct_move_fraction=_fraction(len(set(moves)), len(moves)),
    )


def analyze_games(
    records: Iterable[GameRecord],
    *,
    book: OpeningBook | None = None,
) -> tuple[GameFeatures, ...]:
    """Return the features of many games, classifying against one book load."""

    return tuple(analyze_game(record, book=book) for record in records)


def summarize_games(
    features: Sequence[GameFeatures],
    *,
    level: OpeningLevel = OpeningLevel.FAMILY,
) -> GameDistribution:
    """Aggregate per-game features into one distribution summary.

    Opening mass is aggregated by classified family rather than by literal move
    prefix, so transpositions land together and a broad opening does not
    fragment across buckets.
    """

    if not features:
        raise ValueError("a distribution summary needs at least one game")
    games = len(features)
    return GameDistribution(
        version=GAME_ANALYSIS_VERSION,
        games=games,
        mean_ply_count=_mean(feature.ply_count for feature in features),
        mean_generated_plies=_mean(feature.generated_plies for feature in features),
        result_counts=_counts(feature.result for feature in features),
        termination_counts=_counts(feature.termination.value for feature in features),
        adjudicated_games=sum(1 for feature in features if feature.adjudicated),
        opening_level=level,
        opening_counts=opening_distribution(
            (feature.opening for feature in features),
            level,
        ),
        opening_book=opening_book_identity(),
        repeated_games=sum(1 for feature in features if feature.repetition.repeated),
        threefold_claimable_games=sum(
            1 for feature in features if feature.repetition.threefold_claimable
        ),
        mean_repeated_ply_fraction=_mean(
            feature.repetition.repeated_ply_fraction for feature in features
        ),
        mean_cycle_ply_fraction=_mean(
            feature.repetition.cycle_ply_fraction
            for feature in features
            if feature.repetition.repeated
        ),
        mean_distinct_move_fraction=_mean(
            feature.distinct_move_fraction for feature in features
        ),
        distinct_game_fraction=_fraction(
            len({feature.game_id for feature in features}),
            games,
        ),
    )


def _repetition_diagnostics(record: GameRecord) -> RepetitionDiagnostics:
    """Count position recurrences by replaying the game once.

    Positions are keyed by EPD, which is what the repetition rules compare:
    placement, side to move, castling rights, and a legal en passant square,
    without the move counters. Keying on the full FEN instead would report zero
    repetitions for a game that drew by one, since the halfmove clock differs
    at every occurrence.
    """

    board = chess.Board(record.initial_position)
    occurrences: dict[str, int] = {board.epd(): 1}
    first_repetition: int | None = None
    threefold: int | None = None
    repeated_plies = 0
    moves = record.move_action_ids
    for ply_index, action_id in enumerate(moves):
        board.push(decode_move(action_id))
        key = board.epd()
        seen = occurrences.get(key, 0) + 1
        occurrences[key] = seen
        if seen > 1:
            repeated_plies += 1
            if first_repetition is None:
                first_repetition = ply_index
        if seen > 2 and threefold is None:
            threefold = ply_index
    cycle_plies = 0 if first_repetition is None else len(moves) - first_repetition
    return RepetitionDiagnostics(
        first_repetition_ply=first_repetition,
        threefold_ply=threefold,
        maximum_occurrences=max(occurrences.values()),
        repeated_ply_fraction=_fraction(repeated_plies, len(moves)),
        cycle_ply_fraction=_fraction(repeated_plies, cycle_plies),
    )


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _fraction(part: int, whole: int) -> float:
    return part / whole if whole else 0.0
