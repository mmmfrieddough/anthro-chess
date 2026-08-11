"""Derive which rating pool a game's source ratings came from.

The pool is read from the source's own ``Event`` label rather than from the
time control the speed axis is derived from, because the two legitimately
disagree and only the label answers this question. See
``0057-the-rating-namespace-is-derived-per-game.md``.
"""

from __future__ import annotations

import re

from anthro_chess.data.speed import Speed

#: The pool words a source may name. A pool exists per speed class and the
#: source names both with one vocabulary, so the classes are that vocabulary
#: rather than a second copy of it kept in step by hand.
_POOL_WORDS = frozenset(speed.value for speed in Speed)
#: ``\b`` is what keeps an ultrabullet game off the bullet ladder. One pool word
#: contains the other and the alternation is built from an unordered set, so the
#: boundaries are the only thing deciding which of the two a label matches.
_POOL_RE = re.compile(r"\b(" + "|".join(_POOL_WORDS) + r")\b", re.IGNORECASE)


def names_a_pool(identifier: str) -> bool:
    """Whether an identifier's own words name one of the rating pools."""

    return not _POOL_WORDS.isdisjoint(re.split(r"[-_]", identifier))


def rating_namespace_from_event(
    event: str | None,
    *,
    prefix: str | None,
) -> str | None:
    """Return the ``<prefix>_<pool>`` namespace an ``Event`` label names.

    ``None`` where the source declares no prefix or its label names no pool.
    That is the honest record of a rating whose pool is unknown: an absent
    namespace is detectable by every later reader, while a guessed one pools
    two scales as one and nothing downstream can tell.
    """

    if prefix is None or event is None:
        return None
    match = _POOL_RE.search(event)
    if match is None:
        return None
    return f"{prefix}_{match.group(1).casefold()}"
