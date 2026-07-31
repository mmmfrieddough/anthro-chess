"""Tests for the exactly enumerated shallow repertoire."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import chess
import pytest

from anthro_chess.evaluation.openings import (
    ActionPolicy,
    OpeningBook,
    OpeningEntry,
    OpeningLevel,
    OpeningTreeError,
    walk_repertoire,
)


def moves(sans: str) -> list[chess.Move]:
    """Return the moves of a SAN line played from the standard start."""

    board = chess.Board()
    played: list[chess.Move] = []
    for san in sans.split():
        move = board.parse_san(san)
        played.append(move)
        board.push(move)
    return played


def build_book(entries: Sequence[tuple[str, str, str]]) -> OpeningBook:
    """Build a book small enough that a walk's arithmetic is checkable by hand."""

    built: list[OpeningEntry] = []
    for eco, name, sans in entries:
        played = moves(sans)
        board = chess.Board()
        for move in played:
            board.push(move)
        built.append(
            OpeningEntry(
                eco=eco,
                name=name,
                uci=" ".join(move.uci() for move in played),
                epd=board.epd(),
                ply=len(played),
            )
        )
    return OpeningBook.indexed(
        name="test-book",
        version=1,
        content_sha256="0" * 64,
        source={},
        license={},
        entries=built,
    )


#: One waypoint at the first move and two destinations under it, so a walk has
#: both a choice to report and a partway stop to exclude.
BOOK = build_book(
    [
        ("A00", "Waypoint Test", "e4"),
        ("B20", "Sicilian Test", "e4 c5"),
        ("C20", "Open Test", "e4 e5"),
    ]
)


def scripted(
    script: Mapping[tuple[str, ...], Mapping[str, float]],
    *,
    default: Mapping[str, float] | None = None,
) -> ActionPolicy:
    """Return a policy written down move by move, keyed by UCI prefix."""

    def policy(
        prefixes: Sequence[tuple[chess.Move, ...]],
    ) -> tuple[dict[chess.Move, float], ...]:
        resolved = []
        for prefix in prefixes:
            key = tuple(move.uci() for move in prefix)
            distribution = script.get(key, default or {})
            resolved.append(
                {
                    chess.Move.from_uci(uci): probability
                    for uci, probability in distribution.items()
                }
            )
        return tuple(resolved)

    return policy


def test_a_walk_reports_the_distribution_its_policy_implies() -> None:
    """Two plies of a written-down policy have an arithmetic answer."""

    walk = walk_repertoire(
        scripted(
            {
                (): {"e2e4": 1.0},
                ("e2e4",): {"c7c5": 0.75, "e7e5": 0.25},
            }
        ),
        plies=2,
        threshold=0.01,
        book=BOOK,
    )

    assert walk.repertoire() == {
        "Open Test": pytest.approx(0.25),
        "Sicilian Test": pytest.approx(0.75),
    }
    assert walk.pruned_mass == pytest.approx(0.0)
    assert walk.waypoint_mass == pytest.approx(0.0)


def test_mass_that_stops_on_a_waypoint_is_reported_apart_from_the_choices() -> None:
    """A line that left the book partway chose nothing to put in a repertoire."""

    walk = walk_repertoire(
        scripted(
            {
                (): {"e2e4": 1.0},
                ("e2e4",): {"c7c5": 0.5, "a7a6": 0.5},
            }
        ),
        plies=2,
        threshold=0.01,
        book=BOOK,
    )

    # 1.e4 a6 leaves the book at a position two openings still pass through.
    assert walk.waypoint_mass == pytest.approx(0.5)
    assert walk.destinations == {"Sicilian Test": pytest.approx(0.5)}
    # Renormalized, so the reading is conditional on having chosen at all —
    # the same condition the human side of the comparison is under.
    assert walk.repertoire() == {"Sicilian Test": pytest.approx(1.0)}


def test_pruned_mass_is_reported_as_the_bound_it_is() -> None:
    """The walk is exact above its threshold and says how much sat below it."""

    walk = walk_repertoire(
        scripted(
            {
                (): {"e2e4": 1.0},
                ("e2e4",): {"c7c5": 0.98, "e7e5": 0.02},
            }
        ),
        plies=3,
        threshold=0.05,
        book=BOOK,
    )

    # The Open Test line falls below the threshold at ply two, so it is settled
    # where it stood rather than expanded — and its mass is the error bound.
    assert walk.pruned_mass == pytest.approx(0.02)
    assert walk.repertoire()["Open Test"] == pytest.approx(0.02)


def test_mass_the_policy_does_not_spend_on_a_move_ends_the_line() -> None:
    """A resignation is behavior, and is reported apart from pruning."""

    walk = walk_repertoire(
        scripted(
            {
                (): {"e2e4": 1.0},
                # A quarter of the mass went to a non-move action.
                ("e2e4",): {"c7c5": 0.75},
            }
        ),
        plies=2,
        threshold=0.01,
        book=BOOK,
    )

    assert walk.terminal_mass == pytest.approx(0.25)
    assert walk.pruned_mass == pytest.approx(0.0)
    # The terminated quarter stopped on the waypoint it had reached.
    assert walk.waypoint_mass == pytest.approx(0.25)


def test_every_line_lands_somewhere() -> None:
    """Mass that vanished would be a distribution nobody could read."""

    walk = walk_repertoire(
        scripted(
            {
                (): {"e2e4": 0.6, "d2d4": 0.4},
                ("e2e4",): {"c7c5": 0.5, "e7e5": 0.5},
            }
        ),
        plies=3,
        threshold=0.01,
        book=BOOK,
    )

    assert sum(walk.destinations.values()) + walk.waypoint_mass == pytest.approx(1.0)


def test_the_walk_stops_at_the_declared_depth() -> None:
    """Depth is the reading's own dial rather than the policy's patience."""

    walk = walk_repertoire(
        scripted({}, default={"e2e4": 1.0}),
        plies=1,
        threshold=0.01,
        book=BOOK,
    )

    assert walk.positions_evaluated == 1
    assert walk.waypoint_mass == pytest.approx(1.0)


def test_transpositions_stay_separate_questions_for_the_policy() -> None:
    """The policy conditions on the trajectory, not on the board alone."""

    seen: list[tuple[str, ...]] = []

    def policy(
        prefixes: Sequence[tuple[chess.Move, ...]],
    ) -> tuple[dict[chess.Move, float], ...]:
        seen.extend(tuple(move.uci() for move in prefix) for prefix in prefixes)
        return tuple({} for _ in prefixes)

    walk_repertoire(policy, plies=1, threshold=0.5, book=BOOK)

    assert seen == [()]


@pytest.mark.parametrize(
    ("plies", "threshold"),
    [(0, 0.1), (2, 0.0), (2, 1.5)],
)
def test_an_unusable_walk_is_rejected(plies: int, threshold: float) -> None:
    with pytest.raises(OpeningTreeError):
        walk_repertoire(scripted({}), plies=plies, threshold=threshold, book=BOOK)


def test_a_policy_returning_the_wrong_count_is_rejected() -> None:
    """Misaligned distributions would attribute one line's mass to another."""

    def policy(
        prefixes: Sequence[tuple[chess.Move, ...]],
    ) -> tuple[dict[chess.Move, float], ...]:
        return ()

    with pytest.raises(OpeningTreeError, match="distribution"):
        walk_repertoire(policy, plies=2, threshold=0.5, book=BOOK)


def test_a_policy_spending_more_than_unit_mass_is_rejected() -> None:
    with pytest.raises(OpeningTreeError, match="unit mass"):
        walk_repertoire(
            scripted({(): {"e2e4": 0.8, "d2d4": 0.8}}),
            plies=2,
            threshold=0.1,
            book=BOOK,
        )


def test_the_walk_record_carries_its_bound_and_its_cost() -> None:
    walk = walk_repertoire(
        scripted({(): {"e2e4": 1.0}, ("e2e4",): {"c7c5": 1.0}}),
        plies=2,
        threshold=0.01,
        level=OpeningLevel.FAMILY,
        book=BOOK,
    )
    record = walk.as_record()

    assert record["plies"] == 2
    assert record["level"] == OpeningLevel.FAMILY.value
    assert record["pruned_mass"] == pytest.approx(0.0)
    assert record["positions_evaluated"] == 2
    assert record["repertoire"] == {"Sicilian Test": pytest.approx(1.0)}


def test_the_walk_labels_by_the_level_it_was_asked_for() -> None:
    """A finer level splits what the family level keeps together."""

    book = build_book(
        [
            ("C20", "Open Test", "e4 e5"),
            ("C60", "Open Test: Deep Line", "e4 e5 Nf3 Nc6 Bb5"),
        ]
    )
    line = [move.uci() for move in moves("e4 e5 Nf3 Nc6 Bb5")]
    script = {tuple(line[:index]): {line[index]: 1.0} for index in range(len(line))}

    family = walk_repertoire(scripted(script), plies=5, threshold=0.01, book=book)
    deepest = walk_repertoire(
        scripted(script),
        plies=5,
        threshold=0.01,
        level=OpeningLevel.LINE,
        book=book,
    )

    assert set(family.repertoire()) == {"Open Test"}
    assert set(deepest.repertoire()) == {"Open Test: Deep Line"}
