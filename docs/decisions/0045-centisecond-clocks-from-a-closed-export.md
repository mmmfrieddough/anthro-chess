# 0045: The Corpus Comes From A Closed Export, For Its Centisecond Clocks

Date: 2026-08-10

## Status

Accepted as initial design direction. Names the source, precision and span the
corpus in `#89` is built from, and the limitations that choice accepts
permanently. Extended by
`0056-the-speed-axis-is-derived-from-the-time-control.md`, which takes the
speed axis off the source's own label because this span is one the label does
not hold still across. Narrowed by
`0058-the-corpus-starts-after-the-rapid-pool-split.md`, which drops the nine
months this span opens with: the export names a rating pool for them that did
not exist when they were played. The gap the Consequences below name is closed
by `0059-the-normalized-row-carries-a-day-not-an-instant.md`, which puts the
date on the row and says why the time of day stays off it.

## Context

Lichess publishes the same games twice. The **standard export** runs 2013-01
through the present month — 163 months, 8.04B rated standard games — and carries
`%clk` comments at **one-second** precision from 2017-04. The **universal
export** (`db-univ`) runs 2013-01 through 2021-06 and carries `%clkc` comments
at **centisecond** precision.

Measured while choosing between them:

| | standard | universal |
| --- | --- | --- |
| clock precision | 1 s | **10 ms** |
| span | 2013-01 → present | 2013-01 → **2021-06** |
| games in span | 8,038,784,095 | 3,734,068,694 |
| compression | zstd | bzip2 |
| still growing | yes | **no, last modified 2021-07** |

Two measurements settled the shape of the decision.

**Centisecond clocks begin exactly at 2017-04.** Samples of 2013-06, 2015-06 and
2016-06 contain no `%clkc` at all; 2017-04 onward carries it on 85-93% of games.
So the universal export's advantage covers 51 of its 102 months, and the earlier
half is dominated by casual, anonymous and engine games this corpus would reject
anyway — 80% casual and 74% anonymous in the 2013-06 sample.

**The two exports share game identity.** 39 of 40 rated game ids sampled from
`db-univ` 2017-04 appear in the standard export for the same month. The internal
game id derives from the source id and the source game key, so choosing between
them is a precision choice on the same games rather than a choice of corpus.

## Decision

**The corpus is the universal export, 2017-04 through 2021-06** — 51 archives,
3.35B games as published, ~2.21B after filtering to rated standard games with
both players named and rated.

**Clock precision is the reason.** One-second quantization on a three-minute
blitz game is coarse against move times that are frequently under a second, and
the move-time head in Milestone 6 reads exactly that signal. Precision is fixed
permanently at core designation, so it is not deferrable in the way volume is.

**The span is accepted as final rather than as a starting point.** `db-univ` has
not been extended since 2021-07 and there is no reason to expect it will be.
This is a closed dataset, which is an argument for taking all of it rather than
against: there is no coming back for more centisecond data.

## What this costs

**The corpus ends in mid-2021.** The standard export's 2021-07 through 2026-07
months hold 5,672,489,880 games — **71% of everything Lichess has published**,
and the most recent five years of it. A model trained here imitates 2017-2021
players, and the evaluation core can never measure whether it matches
contemporary ones.

**Rating semantics are pinned to that era.** A Lichess 1500 in 2019 is not a
1500 in 2026, so a product promising "play like a 1500" means a 2017-2021 1500.

**The population changes twice inside the span, and both changes are permanent
features of the corpus.** Monthly volume runs 43.8M before the pandemic, 71.1M
after (+62.3%), and 96.4M after the Queen's Gambit window (+35.6%) — both step
changes survive a seasonal control, and both are influxes of new and therefore
weaker players. The corpus is a blend across those populations, and the corpus
carries no date field with which to separate them unless one is added.

**Preparation is slower.** Bzip2 decompresses at roughly 52 MB/s per core here
against Zstandard's hundreds. That is well clear of the ~235 games/s decode rate
it feeds, so it does not bind, but it removes a margin.

None of these is reversible after `#90` designates the core.

## Alternatives considered

**The standard export for the whole span.** Keeps the recent five years and
compresses better, at one-second clocks forever. Rejected because precision is
the irreversible half and volume is not: 2.21B games is already far beyond what
this project will train on.

**Both, with precision recorded per game.** The schema already carries
`clock_precision_ms`, so a corpus could hold 10 ms games from one export and 1 s
games from the other and slice on it. Not rejected on the merits — it remains
available, and adding modern months later breaks nothing, because adding games
is a superset operation. It is not done now because it doubles the corpus for
data the project cannot yet train on, and because a modern slice is more useful
as an out-of-distribution check outside the core than inside it.

## Consequences

- `configs/data/lichess-univ-2017-04-2021-06.toml` names the 51 archives, each
  pinned by Lichess's published digest.
- The selection reuses the split seed the standard-export selection already
  froze rather than taking a fresh one. The two exports publish the same games
  under the same ids, so a new seed would reassign every one of them: 94.8% of
  the existing frozen pool would leave the held-out split, about nine tenths of
  that into training.
- `ArchiveConfig` accepts `bzip2`, and PGN reading decompresses it in the
  stream, because the chosen source publishes nothing else.
- A modern out-of-distribution slice stays available later, outside the core.
- If the corpus is ever to distinguish its own eras — and the two population
  breaks above are the reason it might want to — the normalized row needs a
  date or source-month field. It carries none today, and the core's axes are
  fixed at designation.

## References

- `#89`, the breadth pass this supplies
- `docs/decisions/0046-a-corpus-is-appended-one-archive-at-a-time.md`, which
  says how one corpus is built from these 51 archives
- `docs/data.md` (Corpus Expansion, Primary Source)
- `docs/decisions/0011-held-out-test-partition.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
