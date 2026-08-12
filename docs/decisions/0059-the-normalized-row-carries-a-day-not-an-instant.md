# 0059: The Normalized Row Carries A Day, Not An Instant

Date: 2026-08-12

## Status

Accepted. Closes the gap
`0045-centisecond-clocks-from-a-closed-export.md` names in its own
Consequences: that record fixes a corpus spanning three populations and observes
that nothing in the row can separate them.

## Context

The corpus is the Lichess universal export, and its span contains two population
step changes. Monthly volume runs 43.8M before the pandemic, 71.1M after
(+62.3%), and 96.4M after the Queen's Gambit window (+35.6%). Both survive a
seasonal control and both are influxes of newer and therefore weaker players.
`#89` lists temporal spread among the axes the corpus should be able to measure,
and before this the schema's 33 columns held no date, month, era, or archive;
`source_game_key` is the Lichess id and is not usefully ordered.

Shard names are digest-derived per archive, so provenance exists at the file
level. That is not enough. `0012-derived-evaluation-views.md` makes a benchmark a
view over the materialized pool, and a benchmark needing what the view layer
cannot derive is exactly the signal that a field belongs in the schema.

This binds hardest on Milestone 6. Move-time distributions under a fixed clock
are the behavior that shifts most with population, so if the move-time head
reads a timing mismatch against the human reference, "the corpus blends three
populations" is not a testable explanation without a date on the row.

The source carries `UTCDate` and `UTCTime` as separate headers, so nothing has
to be inferred. Measured on one real shard of 50,000 games — 10,538,575 bytes,
zstd, one row group — appending a column and rewriting:

| column | delta | per game | over ~2B games |
| --- | --- | --- | --- |
| source month, constant per shard | +294 B | 0.0059 B | ~12 MB |
| `UTCDate` as `date32` | +456 B | 0.0091 B | ~18 MB |
| `UTCDate` + `UTCTime` to the second | +303,268 B | 6.07 B | ~12 GB |

Those shards run ~211 bytes a game, putting the corpus near 420 GB. Day
granularity is cheap because a shard holds roughly thirty distinct values, rows
sit in source order, and the dictionary runs are long. Second precision defeats
all of it: cardinality approaches one value per row, and 12 GB is 2.9% of the
corpus.

## Decision

**The row carries the source's `UTCDate` as a `date32`, with a status column
beside it.** A game the source dates is dated; one whose header is the PGN
unknown-date placeholder, or absent, records `unavailable`; one whose header is
present but names no day — a partly known date, an impossible one — records
`rejected`. Nothing is defaulted, per the rule `docs/data.md` (Missing Fields)
already states for every optional field.

**`UTCTime` is not taken, and that is a refusal rather than a deferral.**
`#90` fixes the core's axes at designation, so a field left out now is left out
permanently. The population question is month-scale and a day answers it. What
time of day does to move choice or move time is a research question rather than
a product one, and this is a product project.

**The `Date` header is not a fallback.** It is a different quantity in a
timezone the header does not state, and a column blending the two would be worse
than one that admits the absence.

**A view can bound the date, and a game with none is excluded by either bound.**
An era reading that quietly counted games of unknown era would not be an era
reading.

## Consequences

Normalized schema version 4, preprocessing version 8. Every corpus and frozen
pool prepared before this is rebuilt rather than appended to. Nothing is frozen
yet — no evaluation core is designated — so this is the cheap moment, and it is
the last one.

The pool cut carries the column without being asked to: a freeze reads the whole
schema and writes it back, so what the corpus holds the pool holds. Reaching it
from a view cost the pool's projection one column.

**A reviewer who expects the time-of-day question to be asked should overrule
this before `#90`.** After designation there is no second chance at it, the
export is closed, and no later pass can recover from the corpus what was never
written into it.

## References

- `#89`, the breadth pass whose temporal axis this supplies
- `#90`, which fixes the core's axes at designation
- `docs/decisions/0045-centisecond-clocks-from-a-closed-export.md`
- `docs/decisions/0012-derived-evaluation-views.md`
- `docs/data.md` (Identifiers And Provenance, Missing Fields)
