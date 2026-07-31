"""Tests for game-level opening classification from the owned versioned book."""

from __future__ import annotations

import socket
import urllib.request
from collections.abc import Sequence
from hashlib import sha256
from importlib.resources import files

import chess
import pytest

from anthro_chess.chess import RESIGNATION_ACTION_ID, encode_move
from anthro_chess.evaluation.openings import (
    OPENING_CLASSIFICATION_VERSION,
    UNCLASSIFIED,
    OpeningBook,
    OpeningClassificationError,
    OpeningEntry,
    OpeningLevel,
    classify_action_ids,
    classify_moves,
    classify_progression,
    load_book,
    opening_book_identity,
    opening_distribution,
    opening_levels,
    repertoire_distribution,
)
from anthro_chess.evaluation.openings.book import BOOK_FILE_NAME

#: A named line and one transposition into the same position by another order.
NIMZO_INDIAN = "d4 Nf6 c4 e6 Nc3 Bb4"
NIMZO_INDIAN_TRANSPOSED = "c4 e6 Nc3 Bb4 d4 Nf6"

#: A book name that carries all three granularity levels.
NAJDORF_ENGLISH_ATTACK = "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3"


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
    """Build a small in-memory book so semantics do not depend on the vendored one."""

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


def test_book_identity_matches_its_recorded_metadata() -> None:
    book = load_book()
    packaged = (
        files("anthro_chess.evaluation.openings")
        .joinpath("data", BOOK_FILE_NAME)
        .read_text(encoding="utf-8")
    )

    assert opening_book_identity() == {
        "name": book.name,
        "version": book.version,
        "entries": len(book.entries),
        "sha256": sha256(packaged.encode("utf-8")).hexdigest(),
    }


def test_book_records_an_explicit_license_and_source() -> None:
    book = load_book()

    assert book.license["spdx_id"]
    assert book.license["url"]
    assert book.source["repository"]
    assert book.source["commit"]


def test_book_lines_replay_to_their_indexed_positions() -> None:
    """Re-derive every key so the index cannot drift from the checked-in moves."""

    book = load_book()

    for entry in book.entries:
        board = chess.Board()
        for uci in entry.uci.split():
            board.push(chess.Move.from_uci(uci))
        assert board.epd() == entry.epd
        assert entry.ply == len(entry.uci.split())
        assert book.entry_for(entry.epd) is entry


def test_book_never_names_an_opening_unclassified() -> None:
    """The unclassified label must stay distinguishable from a real family."""

    for entry in load_book().entries:
        family, _, _ = opening_levels(entry.name)
        assert family.lower() != UNCLASSIFIED


def test_transposition_lands_in_the_same_opening() -> None:
    direct = classify_moves(moves(NIMZO_INDIAN))
    transposed = classify_moves(moves(NIMZO_INDIAN_TRANSPOSED))

    assert direct.family == "Nimzo-Indian Defense"
    assert transposed == direct


def test_classification_emits_every_granularity_level() -> None:
    label = classify_moves(moves(NAJDORF_ENGLISH_ATTACK))

    assert label.family == "Sicilian Defense"
    assert label.variation == "Sicilian Defense: Najdorf Variation"
    assert label.line == "Sicilian Defense: Najdorf Variation, English Attack"
    assert label.label(OpeningLevel.FAMILY) == label.family
    assert label.label(OpeningLevel.VARIATION) == label.variation
    assert label.label(OpeningLevel.LINE) == label.line


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Ruy Lopez", ("Ruy Lopez", "Ruy Lopez", "Ruy Lopez")),
        (
            "Polish Opening, with d5",
            ("Polish Opening", "Polish Opening, with d5", "Polish Opening, with d5"),
        ),
        (
            "Slav Defense: Modern Line, Bf5",
            (
                "Slav Defense",
                "Slav Defense: Modern Line",
                "Slav Defense: Modern Line, Bf5",
            ),
        ),
    ],
)
def test_levels_truncate_the_name_at_its_separators(
    name: str,
    expected: tuple[str, str, str],
) -> None:
    assert opening_levels(name) == expected


def test_the_deepest_named_position_wins() -> None:
    book = build_book(
        [
            ("B20", "Sicilian Defense", "e4 c5"),
            ("B27", "Sicilian Defense: Hyperaccelerated Dragon", "e4 c5 Nf3 g6"),
        ]
    )

    label = classify_moves(moves("e4 c5 Nf3 g6"), book=book)

    assert label.variation == "Sicilian Defense: Hyperaccelerated Dragon"
    assert label.matched_ply == 4
    assert label.eco == "B27"


def test_leaving_the_book_keeps_the_deepest_match() -> None:
    book = build_book([("B20", "Sicilian Defense", "e4 c5")])

    label = classify_moves(moves("e4 c5 Nh3 a6 Ng1 h6"), book=book)

    assert label.line == "Sicilian Defense"
    assert label.matched_ply == 2


def test_a_game_the_book_does_not_name_is_explicitly_unclassified() -> None:
    book = build_book([("B20", "Sicilian Defense", "e4 c5")])

    label = classify_moves(moves("d4 d5"), book=book)

    assert not label.classified
    assert label.family == UNCLASSIFIED
    assert label.variation == UNCLASSIFIED
    assert label.line == UNCLASSIFIED
    assert label.eco is None
    assert label.matched_ply == 0


def test_a_game_with_no_moves_is_unclassified() -> None:
    assert not classify_moves([]).classified


def test_an_explicit_standard_start_matches_the_default() -> None:
    played = moves(NIMZO_INDIAN)

    assert classify_moves(
        played, initial_position=chess.STARTING_FEN
    ) == classify_moves(played)


def test_a_game_starting_away_from_the_opening_is_unclassified() -> None:
    position = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    board = chess.Board(position)

    label = classify_moves([board.parse_san("e4")], initial_position=position)

    assert not label.classified


def test_action_ids_classify_like_moves() -> None:
    played = moves(NIMZO_INDIAN)

    assert classify_action_ids(
        [encode_move(move) for move in played]
    ) == classify_moves(played)


def test_a_non_move_action_ends_the_scan() -> None:
    played = moves(NIMZO_INDIAN)
    action_ids = [encode_move(move) for move in played[:2]]
    action_ids.append(RESIGNATION_ACTION_ID)
    action_ids.extend(encode_move(move) for move in played[2:])

    label = classify_action_ids(action_ids)

    assert label.matched_ply == 2
    assert label.family != "Nimzo-Indian Defense"


def test_an_illegal_move_is_rejected_rather_than_silently_unclassified() -> None:
    with pytest.raises(OpeningClassificationError, match="illegal"):
        classify_moves([chess.Move.from_uci("e2e4"), chess.Move.from_uci("e2e4")])


def test_an_unreadable_initial_position_is_rejected() -> None:
    with pytest.raises(OpeningClassificationError, match="cannot replay"):
        classify_moves([], initial_position="not a position")


def test_the_label_record_is_stable() -> None:
    record = classify_moves(moves(NAJDORF_ENGLISH_ATTACK)).as_record()

    assert record == {
        "version": OPENING_CLASSIFICATION_VERSION,
        "classified": True,
        "eco": "B90",
        "matched_ply": 11,
        "book_ply": 11,
        "available_ply": 17,
        "consumed_fraction": pytest.approx(11 / 17),
        # The English Attack is the family's destination — no other family is
        # reachable from it — while deeper names split it further, so the same
        # position is a waypoint at the two finer levels.
        "waypoint": {"family": False, "variation": True, "line": True},
        "family": "Sicilian Defense",
        "variation": "Sicilian Defense: Najdorf Variation",
        "line": "Sicilian Defense: Najdorf Variation, English Attack",
    }


def test_distributions_aggregate_at_the_requested_level() -> None:
    labels = [
        classify_moves(moves(NAJDORF_ENGLISH_ATTACK)),
        classify_moves(moves("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4")),
        classify_moves(moves(NIMZO_INDIAN)),
        classify_moves([]),
    ]

    assert opening_distribution(labels) == {
        "Nimzo-Indian Defense": 1,
        "Sicilian Defense": 2,
        UNCLASSIFIED: 1,
    }
    assert opening_distribution(labels, OpeningLevel.VARIATION) == {
        "Nimzo-Indian Defense": 1,
        "Sicilian Defense: Najdorf Variation": 2,
        UNCLASSIFIED: 1,
    }
    assert len(opening_distribution(labels, OpeningLevel.LINE)) == 4


def test_classification_is_deterministic_and_needs_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The book ships with the package, so a rebuild of the index cannot dial out."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("classification must not open a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    load_book.cache_clear()
    try:
        first = classify_moves(moves(NAJDORF_ENGLISH_ATTACK))
        load_book.cache_clear()
        second = classify_moves(moves(NAJDORF_ENGLISH_ATTACK))
    except BaseException:
        # Only on the failing path: a book left half-built by an exception is
        # not one later tests should inherit. On the passing path the cache
        # holds the book this test just proved is the same one, and clearing it
        # would charge the next caller a full rebuild to arrive back here.
        load_book.cache_clear()
        raise

    assert first == second


def test_the_continuation_index_covers_every_named_position() -> None:
    """A named position with no continuations could never be judged."""

    book = load_book()

    assert set(book.continuations) == set(book.positions)
    for entry in book.entries:
        continuation = book.continuation_for(entry.epd)
        assert continuation is not None
        # A position reaches itself, so the deepest theory onward can never be
        # shallower than the position's own depth.
        assert continuation.deepest_ply >= entry.ply
        assert continuation.reachable_labels(OpeningLevel.FAMILY) >= 1


def test_a_position_many_openings_pass_through_is_a_waypoint() -> None:
    """The rule is structural: more than one label still reachable onward."""

    book = build_book(
        [
            ("A00", "Test Waypoint", "e4"),
            ("B20", "Sicilian Test", "e4 c5"),
            ("C20", "Open Test", "e4 e5"),
        ]
    )

    waypoint = classify_moves(moves("e4"), book=book)
    chosen = classify_moves(moves("e4 c5"), book=book)

    assert waypoint.waypoint(OpeningLevel.FAMILY)
    assert waypoint.destination(OpeningLevel.FAMILY) is None
    assert not chosen.waypoint(OpeningLevel.FAMILY)
    assert chosen.destination(OpeningLevel.FAMILY) == "Sicilian Test"


def test_book_depth_separates_how_far_from_how_far_it_could_have_gone() -> None:
    """Raw depth conflates choosing a deep line with knowing one."""

    book = build_book(
        [
            ("C20", "Open Test", "e4 e5"),
            ("C60", "Deep Test", "e4 e5 Nf3 Nc6 Bb5"),
        ]
    )

    shallow = classify_moves(moves("e4 e5"), book=book)
    deep = classify_moves(moves("e4 e5 Nf3 Nc6 Bb5"), book=book)

    assert (shallow.book_ply, shallow.available_ply) == (2, 5)
    assert shallow.consumed_fraction == pytest.approx(0.4)
    assert (deep.book_ply, deep.available_ply) == (5, 5)
    assert deep.consumed_fraction == pytest.approx(1.0)


def test_an_unnamed_game_has_no_depth_and_is_not_a_waypoint() -> None:
    """Off book is a statement about what was played, not about how far."""

    label = classify_moves(moves("a3"), book=build_book([("C20", "Open", "e4 e5")]))

    assert not label.classified
    assert (label.book_ply, label.available_ply) == (0, 0)
    assert label.consumed_fraction == 0.0
    assert not label.waypoint(OpeningLevel.FAMILY)
    assert label.destination(OpeningLevel.FAMILY) == UNCLASSIFIED


def test_book_depth_is_in_book_coordinates_rather_than_game_plies() -> None:
    """A transposition reaches the same theory however long the order took."""

    direct = classify_moves(moves(NIMZO_INDIAN))
    transposed = classify_moves(moves(NIMZO_INDIAN_TRANSPOSED))

    assert direct.book_ply == transposed.book_ply
    assert direct.available_ply == transposed.available_ply


def test_truncated_classification_only_ever_removes_matches() -> None:
    label = classify_moves(moves(NAJDORF_ENGLISH_ATTACK), plies=6)

    assert label.matched_ply <= 6
    assert label.family == "Sicilian Defense"
    assert label.line != "Sicilian Defense: Najdorf Variation, English Attack"


def test_a_progression_matches_classifying_each_depth_separately() -> None:
    """One replay has to agree with the obvious slow way, or it is not the same."""

    played = moves(NAJDORF_ENGLISH_ATTACK)
    progression = classify_progression(played, plies=12)

    assert len(progression) == 12
    for ply, label in enumerate(progression, start=1):
        assert label == classify_moves(played, plies=ply)


def test_a_progression_pads_a_short_game_with_the_label_it_kept() -> None:
    progression = classify_progression(moves("e4 e5"), plies=5)

    assert len(progression) == 5
    assert progression[-1] == progression[1]


def test_the_repertoire_distribution_leaves_waypoints_out() -> None:
    """Counting a game that chose nothing turns depth into a preference."""

    book = build_book(
        [
            ("A00", "Waypoint Test", "e4"),
            ("B20", "Sicilian Test", "e4 c5"),
            ("C20", "Open Test", "e4 e5"),
        ]
    )
    labels = [
        classify_moves(moves("e4"), book=book),
        classify_moves(moves("e4 c5"), book=book),
        classify_moves(moves("e4 e5"), book=book),
    ]

    assert opening_distribution(labels) == {
        "Open Test": 1,
        "Sicilian Test": 1,
        "Waypoint Test": 1,
    }
    assert repertoire_distribution(labels) == {"Open Test": 1, "Sicilian Test": 1}
