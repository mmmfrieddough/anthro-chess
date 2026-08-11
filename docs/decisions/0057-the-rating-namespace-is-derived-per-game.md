# 0057: The Rating Namespace Is Derived Per Game

Date: 2026-08-11

## Status

Accepted. Extends `0056-the-speed-axis-is-derived-from-the-time-control.md`,
which settled that the source's own label is what answers the pool question and
left the answer declared once per selection.

## Context

Preparation wrote `source_rating_namespace` from one configured constant, so
every game of a selection carried the same value. That was true only because
the selection isolated one speed and every accepted game came from one pool.

The isolation is gone. A selection may keep every speed, and even a single-speed
one can accept games another pool rated: `speed = "rapid"` against the pinned
2017-04 archive accepts 2,584,704 games whose ratings came from the classical
pool, because no rapid pool existed to rate them in. Under a declared namespace,
each of those rows claims whatever the selection said, and nothing fails.

A rating is meaningless without its pool. One Lichess player holds a bullet, a
blitz, a rapid and a classical rating at once, and they are different numbers.
Everything conditioning on rating — the ladder, rating-conditioned training, the
transfer from a configured rating to played strength — would read four scales as
one. No later pass can repair the column either: the correct value is not
recoverable from a row that already claims the wrong one.

## Decision

**The namespace is derived per game from the source's own `Event` label.**
Configuration declares the source's prefix — the family of pools it rates in —
and the pool comes from the label, so the column is `<prefix>_<pool>`. The label
is free text and the pool word is read out of it leftmost first, which is where
a source writes it and before whatever tournament reference follows.

**A label naming no pool records no namespace, and the game is kept.** Its
ratings are still what the source said, and the row states that the pool is
unknown rather than guessing one. An absent value is detectable by every later
reader; a wrong one is not.

**The configured field is renamed rather than reinterpreted.** A selection
carrying the old whole-namespace key fails to load instead of composing
`lichess_blitz_blitz` out of a value that was already complete.

Preparation reports the pools it stamped in the manifest coverage, because a
corpus whose labels named nothing recognizable is otherwise indistinguishable
from one whose pools were all recorded — and the run that would notice costs
hours.

## Consequences

What the pinned baseline selection prepares is unchanged. Every game it accepts
is one the archive labels Blitz, since the cross-tabulation in `0056` shows
label and derived class agreeing everywhere except the classical/rapid split, so
each row is stamped the same value the constant wrote. The recorded selection
changes with the renamed key, so a corpus prepared before this is rebuilt rather
than appended to — the same refusal that already applied to every corpus
predating `0056`, and no multi-archive corpus has been started.

A multi-speed corpus is now describable, which is what `#89` needs before it
prepares anything.

The prefix is not a general mapping. It assumes a source writes its pool into
the label with the same vocabulary it names speeds with, which is Lichess's
habit rather than a rule. A source that names its pools some other way needs its
own reader, and until one exists it declares no prefix and its rows say the pool
is unknown — which keeps its games usable for everything that does not condition
on rating.

**A label answers the pool question only as well as the export that wrote it
does.** Measured over the first 300,000 games of each: the standard 2017-04
archive labels 68,844 games Classical whose derived class is rapid, the
disagreement `0056` tabulated; the universal export of that same month labels
67,606 of them Rapid and disagrees with the derived class on nothing, as it also
does in 2019-04. That export was written after Lichess split the rapid pool and
carries today's vocabulary throughout, so for it the label is the class
re-derived rather than the pool that rated the game, and a 2017-04 row is
stamped with a pool that did not exist to produce its number. Nothing about this
rule caused that or would be repaired by reversing it — deriving from the time
control is wrong on the same rows for the same reason, because the export
carries no record of the pool at all. `#442` holds the evidence and the
decision.
