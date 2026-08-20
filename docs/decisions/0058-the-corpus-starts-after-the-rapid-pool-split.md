# 0058: The Corpus Starts After The Rapid Pool Split

Date: 2026-08-12

## Status

Accepted. Narrows the span set by
`0045-centisecond-clocks-from-a-closed-export.md`, which chose every month that
carries centisecond clocks, and removes the months on which
`0057-the-rating-namespace-is-derived-per-game.md` cannot name the pool
correctly. `0072-the-clock-says-which-pool-a-rating-came-from.md` rests on this
cut: with those months gone, the namespace and the class derived from the time
control agree on every game, which is what lets one rating input stand without a
pool beside it.

## Context

Lichess rated ten-minute games on its classical ladder until it split Rapid out
of Classical. The universal export was written afterwards and re-derived every
`Event` label under the bands current at export time, so its pre-split months
label those games Rapid while the ratings beside them came off the classical
ladder. Both signals a row could be built from say rapid; only the
contemporaneous standard export still says classical.

Three measurements over the archives settle what happened. Matching games by id
across the two 2017-04 exports, 210,232 games appear in both: **every rating is
identical** and **48,607 labels differ**, so the later export re-banded names
and left the numbers alone. Chaining each player's games by the rating carried
between them — a pair sits on one ladder when `elo + ratingdiff` of one game is
the rating entering the next — shows rapid-shaped and classical-shaped games
chaining at 92.7% in 2017-04 against a 93.3% same-class baseline, while every
genuinely separate pair chains at 0.0%. Sweeping that test by month puts the
change between 2017-12 and 2018-01, and within 2017-12 the hourly rates fall
from 97% at 03:00 UTC on the 2nd to 40% at 04:00 and 5% at 05:00.

**The boundary does not resolve to a game.** Lichess seeded each player's new
rapid rating from their classical one at their first post-split rapid game, so
pairs keep chaining coincidentally for hours and days afterwards, one player at
a time. A test immune to that — whether a rapid game still moved a classical
rating — finds both verdicts scattered across the whole of 2017-12-02 rather
than stepping. Whatever the rollout was, the export does not record an instant.

Affected are 2017-04 through 2017-12: nine months, 215,665,936 published games,
6.4% of the span. Only the rapid-shaped games in them are misnamed, since the
classical-shaped ones already carry the ladder's own name — roughly 1% of the
corpus once filtering is applied.

## Decision

**The corpus is the universal export from 2018-01 through 2021-06** — 42
archives, 3,136,069,223 games as published.

The alternative was to keep those months and correct the label from a hardcoded
split instant. It was rejected because there is no instant to hardcode. Any
constant misfiles some hours of 2017-12-02 in one direction or the other, and
the correction would be permanent surface — a date threaded into a derivation
that is otherwise a pure function of the label, plus a record explaining a fuzzy
boundary and a seeding tail forever, to keep 1% of the corpus. Reading the
contemporaneous standard exports instead was rejected as 51 more archives and a
join by game id, to recover what dropping nine months settles for free.

**The namespace stays a pure function of the source's label**, which is what
this buys: no era of the corpus needs a footnote, and `0057` holds without
exception across every month in it.

## Consequences

The corpus loses its nine oldest and smallest months. That costs volume the
project is not short of, and it narrows temporal spread from 51 months to 42,
still three and a half years. Clock precision, which is `0045`'s entire reason
for this source, is unaffected: every remaining month carries centisecond
clocks.

The span is now bounded at both ends by something real rather than by the
source's extent alone — the pool split at the start and the export's end at
2021-06 — so the "take all of a closed dataset" argument in `0045` no longer
reaches the first nine months.

**Coverage of the check is partial.** The month sweep ran over the archives
present locally: 2018-01 through 2020-04, 28 of the corpus's 42 months, every
one of them showing rapid and classical as separate ladders and no other pair
sharing one. The 14 months from 2020-05 are unchecked and would need acquiring
before a generation is cut against them; the same chaining test is what would
establish them.
