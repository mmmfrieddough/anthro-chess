import pytest

from anthro_chess.data import rating_namespace_from_event


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ("Rated UltraBullet game", "lichess_ultrabullet"),
        ("Rated Bullet game", "lichess_bullet"),
        ("Rated Blitz game", "lichess_blitz"),
        ("Rated Rapid game", "lichess_rapid"),
        ("Rated Classical game", "lichess_classical"),
        ("Rated Correspondence game", "lichess_correspondence"),
    ],
)
def test_a_label_names_the_pool_that_rated_the_game(
    event: str,
    expected: str,
) -> None:
    assert rating_namespace_from_event(event, prefix="lichess") == expected


def test_a_tournament_label_names_its_pool_beside_the_tournament() -> None:
    """The label a source writes is free text with the pool word inside it."""

    event = "Rated Bullet tournament https://lichess.org/tournament/yc1WW2Ox"

    assert rating_namespace_from_event(event, prefix="lichess") == "lichess_bullet"


def test_the_ultrabullet_pool_is_not_read_as_the_bullet_one() -> None:
    """The pools are separate ladders, and one word contains the other."""

    assert (
        rating_namespace_from_event("Rated UltraBullet game", prefix="lichess")
        != "lichess_bullet"
    )


@pytest.mark.parametrize("event", [None, "?", "Rated game", "Rated Antichess game"])
def test_a_label_naming_no_pool_yields_no_namespace(event: str | None) -> None:
    """A guessed pool is undetectable downstream where an absent one is not."""

    assert rating_namespace_from_event(event, prefix="lichess") is None


def test_a_source_declaring_no_prefix_yields_no_namespace() -> None:
    assert rating_namespace_from_event("Rated Blitz game", prefix=None) is None
