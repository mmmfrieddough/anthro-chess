"""Derive a game's speed class from the time control it was played at.

A source's own speed label is a fact about the source at the moment it wrote
the archive, not about the game. Lichess had no Rapid category in 2017-04, so
its `Event` header calls a 10+0 game Classical there and Rapid a year later,
and a corpus spanning both carries an axis meaning two things at once. The
time control does not move, which is why the axis is derived from it here.

The bands and the forty-move length estimate are the source's own, so this
reproduces the archive's labels wherever the archive's vocabulary was already
the current one. See
``0055-the-speed-axis-is-derived-from-the-time-control.md``.
"""

from __future__ import annotations

import re
from enum import StrEnum

#: PGN ``TimeControl`` for a game played without a clock.
_UNLIMITED = "-"

_TIME_CONTROL_RE = re.compile(r"(\d+)\+(\d+)")

#: A game is assumed to last this many moves per side when its length is
#: estimated from the clock, which is how the source bands its own speeds.
_ESTIMATED_MOVES = 40


class Speed(StrEnum):
    """How fast a game was played, banded by its estimated total length."""

    ULTRABULLET = "ultrabullet"
    BULLET = "bullet"
    BLITZ = "blitz"
    RAPID = "rapid"
    CLASSICAL = "classical"
    CORRESPONDENCE = "correspondence"


#: Inclusive upper bound in estimated seconds for every band below
#: correspondence, which takes everything above the last of them.
_UPPER_BOUND_SECONDS: tuple[tuple[int, Speed], ...] = (
    (29, Speed.ULTRABULLET),
    (179, Speed.BULLET),
    (479, Speed.BLITZ),
    (1499, Speed.RAPID),
    (21599, Speed.CLASSICAL),
)


def parse_time_control(value: str | None) -> tuple[int, int] | None:
    """Return the initial clock and increment in seconds, or ``None``.

    ``None`` means the value names no clocked control: it is absent, unknown,
    unlimited, or in a form this does not read.
    """

    if value is None:
        return None
    match = _TIME_CONTROL_RE.fullmatch(value.strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def speed_from_time_control(value: str | None) -> Speed | None:
    """Band a PGN ``TimeControl`` value, or ``None`` when it says nothing."""

    if value is not None and value.strip() == _UNLIMITED:
        return Speed.CORRESPONDENCE
    parsed = parse_time_control(value)
    if parsed is None:
        return None
    initial_seconds, increment_seconds = parsed
    estimate = initial_seconds + _ESTIMATED_MOVES * increment_seconds
    for upper_bound, speed in _UPPER_BOUND_SECONDS:
        if estimate <= upper_bound:
            return speed
    return Speed.CORRESPONDENCE
