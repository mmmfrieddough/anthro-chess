import pytest

from anthro_chess.data import Speed, speed_from_time_control


@pytest.mark.parametrize(
    ("time_control", "expected"),
    [
        ("15+0", Speed.ULTRABULLET),
        ("29+0", Speed.ULTRABULLET),
        ("30+0", Speed.BULLET),
        ("179+0", Speed.BULLET),
        ("180+0", Speed.BLITZ),
        ("479+0", Speed.BLITZ),
        ("480+0", Speed.RAPID),
        ("1499+0", Speed.RAPID),
        ("1500+0", Speed.CLASSICAL),
        ("21599+0", Speed.CLASSICAL),
        ("21600+0", Speed.CORRESPONDENCE),
    ],
)
def test_a_clock_falls_in_the_band_its_length_estimate_lands_in(
    time_control: str,
    expected: Speed,
) -> None:
    assert speed_from_time_control(time_control) is expected


@pytest.mark.parametrize(
    ("time_control", "expected"),
    [
        # An increment is worth forty moves of it, so a control the initial
        # clock alone would call ultrabullet is bullet and one it would call
        # blitz is rapid.
        ("0+2", Speed.BULLET),
        ("300+8", Speed.RAPID),
    ],
)
def test_an_increment_counts_forty_times_toward_the_estimate(
    time_control: str,
    expected: Speed,
) -> None:
    assert speed_from_time_control(time_control) is expected


def test_a_game_played_without_a_clock_is_correspondence() -> None:
    assert speed_from_time_control("-") is Speed.CORRESPONDENCE


@pytest.mark.parametrize("time_control", [None, "", "?", "300", "1/86400"])
def test_a_time_control_naming_no_clock_bands_into_nothing(
    time_control: str | None,
) -> None:
    """An unreadable control yields no class rather than a plausible one."""

    assert speed_from_time_control(time_control) is None
