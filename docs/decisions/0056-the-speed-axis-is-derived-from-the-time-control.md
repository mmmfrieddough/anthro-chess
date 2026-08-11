# 0056: The Speed Axis Is Derived From The Time Control

Date: 2026-08-11

## Status

Accepted. Extends `0045-centisecond-clocks-from-a-closed-export.md`, whose span
is what makes the source's own speed label ambiguous.

## Context

Preparation filtered on the speed word in the PGN `Event` header, which was
correct while the corpus was one month of one export. `0045` widened it to 51
months, and the label does not hold still across them.

Lichess had no Rapid category in 2017-04. It calls a 10+0 game Classical there
and Rapid in any later month, because the category was introduced in between.
Cross-tabulating the archive's own label against the class derived from
`TimeControl` over every game in the pinned 2017-04 standard archive:

| archive label | derived | games |
| --- | --- | --- |
| ultrabullet | ultrabullet | 371,052 |
| bullet | bullet | 2,844,321 |
| blitz | blitz | 5,287,959 |
| classical | **rapid** | **2,584,704** |
| classical | classical | 218,979 |
| correspondence | correspondence | 41,491 |

11,348,506 games, and the classes disagree nowhere except that split. Twelve
parts of the archive's Classical bucket are games any later month calls Rapid,
against one part still called Classical.

A corpus spanning 2017-04 and any later month would therefore carry a speed
axis meaning two different things depending on which archive a game came from.
The evaluation core is designated across that axis and its per-axis power is
fixed permanently at designation, so an axis that means two things is not
something a later pass can repair.

## Decision

**The speed axis is derived from `TimeControl`, not from the `Event` header.**
The initial clock plus forty increments, banded — the source's own estimate and
its own bands, which is why the derivation reproduces the source's labels
wherever the source's vocabulary had already reached its current form.

**The rating namespace stays on the `Event` label.** The two derivations
legitimately disagree, and for 2017-04 the disagreement is the point: those
2,584,704 rapid-shaped games were played by people carrying a *classical*
rating, because no rapid pool existed to rate them in. The time control says
how fast the game was; only the source's own label says which pool the rating
came from.

**A time control that names no clock yields no class**, and a configured speed
filter rejects such a game rather than falling back to the label. A speed word
in the event text is not evidence about the game.

## Consequences

The class is derivable from a normalized row's time fields for every game with
a clock, so slicing a benchmark by speed needs no schema column — which is what
`docs/data.md` (Corpus Expansion) assumes. Correspondence is the exception: an
unlimited control and an absent one both record the initial clock as
unavailable, so a row alone cannot tell the two apart. A control that was
present and unreadable is separable, being recorded as rejected instead.

Two names changed with the derivation. The filter is `filters.speed` rather
than `filters.event_speed`, and preparation reports a rejection under it as
`speed_mismatch` rather than `rating_namespace_mismatch` — a name that was only
ever accurate because isolating one speed was how the corpus guaranteed one
rating namespace. A selection carrying the old key fails to load rather than
being silently reinterpreted.

`ultrabullet` becomes selectable. The event-label pattern never matched it, so
that class was previously unnameable and its games were rejected by any speed
filter.

What the pinned baseline selection prepares is unchanged: blitz membership is
identical under both derivations for all 11,348,506 games of its archive, so
its shards, and any model trained on them, are the same. What it records is
not, and the recorded selection is the corpus's identity — so a corpus prepared
before this cannot be appended to, and is rebuilt rather than extended. That
refusal is loud and says so. No multi-archive corpus had been started when this
landed, and the baseline had already reached its bound.
