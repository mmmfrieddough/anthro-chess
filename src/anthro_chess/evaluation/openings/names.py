"""Naming granularity, derived from the structure of book names themselves.

Book names nest from broad to specific and separate their levels with a colon
or a comma, so the levels are the name truncated at its first and second
separator rather than a second table that has to be kept in step with the book.

This lives apart from classification because the book itself needs it: the
continuation index counts how many distinct labels remain reachable from a
position, and that count is what tells a waypoint from a destination.
"""

from __future__ import annotations

from enum import StrEnum

#: Label used at every level for a game the book does not name. Games that
#: match nothing are reported explicitly rather than forced into a nearest
#: family, so an unnamed line stays visible as an unnamed line.
UNCLASSIFIED = "unclassified"

#: Characters that separate one naming level from the next in book names.
_LEVEL_SEPARATORS = ":,"


class OpeningLevel(StrEnum):
    """Naming granularity a benchmark can aggregate at.

    One classification pass emits all three, so a benchmark picks the level it
    needs instead of the book choosing for it: broad distribution comparisons
    want families, a regression in one line wants the deepest name.
    """

    FAMILY = "family"
    VARIATION = "variation"
    LINE = "line"


def opening_levels(name: str) -> tuple[str, str, str]:
    """Return the family, variation, and line forms of one book name."""

    cuts = [index for index, char in enumerate(name) if char in _LEVEL_SEPARATORS]
    family = name[: cuts[0]].strip() if cuts else name
    variation = name[: cuts[1]].strip() if len(cuts) > 1 else name
    return family, variation, name


def opening_level(name: str, level: OpeningLevel) -> str:
    """Return one book name reduced to a single granularity level."""

    family, variation, line = opening_levels(name)
    if level is OpeningLevel.FAMILY:
        return family
    if level is OpeningLevel.VARIATION:
        return variation
    return line


__all__ = [
    "UNCLASSIFIED",
    "OpeningLevel",
    "opening_level",
    "opening_levels",
]
