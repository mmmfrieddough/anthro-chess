# 0047: Account Status Is Censused Continuously And Claims A Partial Recall

Date: 2026-08-10

## Status

Accepted. Settles `#391` and decides the shape of `#358`. Refines
`0041-games-of-marked-accounts-leave-the-corpus.md`, which decided that marked
accounts' games leave the corpus but assumed the label could be had for every
account; this records what it actually costs and what the filter may therefore
claim.

## Context

`0041` settled that every game a marked account played is rejected, and left
acquiring the label as a matter of waiting. The corpus has since grown from one
archive to the 51 that `0045` names, and the account universe with it: a full
pass over the 37 archives on disk found **3,776,811 distinct accounts** across
2.076B player-slots, and the 51-archive union extrapolates to **7.7-9.6M**.

At that size the acquisition question stops being clerical. The only channel
that answers it is rate limited, its ceiling is capped below totality by
something no channel reaches, and the corpus therefore has to say what recall
it claims rather than imply it caught everything.

## There is no second channel

Three independent checks, each against a source that can be re-read rather than
against inference:

- `lichess-org/api` defines exactly two endpoints returning the projection that
  carries the mark: the bulk lookup, and one restricted to bots. The real-time
  status endpoint takes more ids per request and is documented as cheap to
  poll, but returns presence only.
- `database.lichess.org` offers games, variants, puzzles, evaluations,
  broadcasts and openings. It publishes nothing about accounts, and the bulk
  endpoint's own text says a full user download is not provided.
- `lichess-org/lila` names the mark in three files, of which the only API
  projection is `modules/user/src/main/JsonView.scala`.

So the bulk endpoint is the channel, and the remaining questions are how fast
it can be asked and how much of the answer exists.

## The limiter, read rather than inferred

The endpoint charges the **calling address** against two buckets at once,
2,000 credits per ten minutes and 30,000 per day, and each request costs
`len(batch) // divisor` credits. The divisor is 3 for an anonymous caller and 6
for an authenticated one; a third tier exists for accounts the source has
verified, which is an identity badge granted at its discretion and not
something to plan against. This is transcribed from `lila`'s
`modules/web/src/main/Limiters.scala` and `app/controllers/Api.scala`, which is
where to re-read it. The published API documentation states a flat allowance
that no longer matches either tier.

Multiplied out, at the maximum 300 names per request:

| caller | per ten minutes | per day |
| --- | --- | --- |
| anonymous | 6,000 | 90,000 |
| authenticated | 12,000 | 180,000 |

Two things follow, and both contradict what the machinery previously assumed.

**The budget is spent in accounts, not requests.** Batch size does not change
what a name costs, so there is no throughput to win by tuning it, and a batch
over the maximum loses the excess names silently rather than being refused.

**Sending a token doubles the allowance**, and the endpoint requires no scope,
so a token carrying none is enough. This is the source's own mechanism rather
than a way around one.

The reading is confirmed twice over. The refusals the first census recorded,
before any of this was known, were a burst allowance of roughly 20 requests
refilled by about 600 seconds and a longer cap near 300 batches per address per
day — exactly what the limiter yields for an anonymous caller at 300 names per
batch.

And the tier was measured rather than assumed: sending 300-name batches back to
back with a token, the source refused on the 41st. Anonymous predicts the 21st,
since the same bucket buys half as many requests at twice the cost. So the token
is applied, and the authenticated row above is the one that holds.

## Decision

**Account status comes from the authenticated bulk endpoint, paced from the
limiter rather than tuned against refusals.** A refusal is the only feedback the
source gives, and one already spent leaves nothing to measure the next attempt
against, which is why the previous constants settled at three times the
sustainable rate and read the resulting penalty as evidence that refusals
compound.

**The census will run continuously rather than being re-run.** It is to ask
about accounts in descending order of games played, record each answer against
the account, and spend the day's allowance whenever the machine is available.
Nothing about it is a campaign with a finish line. `#358` builds this; today's
command asks about one archive's accounts in alphabetical order and checkpoints
an offset, which is why it cannot serve.

**The corpus claims the recall the census had reached when its snapshot was
cut, as a number.** Which number that is, is not decided here. It is read off
the census at the moment the evaluation core is designated, by whoever
designates it.

**Nothing waits on the census.** It does not block corpus preparation, pool
generation, or core designation.

## Why ordering by game count replaces choosing a coverage target

Marked accounts play more than average — prevalence rises from 0.67% at 1-2
games to 6.00% at 401-1000, and the slot-weighted rate of 4.26% exceeds the
account-weighted 3.19%. Asking about the busiest accounts first therefore
tracks contamination rather than working against it, and recall accrues far
faster than cost:

| accounts asked | share | share of marked player-slots caught |
| --- | --- | --- |
| 94,860 | 2.5% | 50.0% |
| 473,456 | 12.5% | 80.0% |
| 751,585 | 19.9% | 91.6% |
| 1,117,936 | 29.6% | 96.8% |
| 3,776,811 | 100% | 100% |

Measured over the 37 archives on disk; the union's shares will differ, but the
shape will not.

This is what makes a stopping rule unnecessary. Under any fixed target the
project would have to choose a coverage figure in advance, defend it, and wait
for it before anything downstream could proceed. Under a descending-order
census every target is satisfied on the way to the next one, coverage is
readable at any moment as the queried set's share of player-slots, and the only
decision left is which moment to read it at.

It also removes the reason to keep asking. The last 3.2 points of recall cost
2.4 times what the first 96.8 did, and they buy less than the error bar
described below.

## What the number is not

**It is a share of what the source will disclose, not of what it knows.** A
closed account's projection is short-circuited to its id, username and a closed
flag before the mark is ever added, so a closed account reads as unmarked and
is indistinguishable from an honest one. Closure runs at **15.97% of accounts
and 13.15% of player-slots**, and none of 464 sampled closed accounts, nor any
of 41 in a later live batch, exposed a mark. If closure is independent of
marking, then catching 91.6% of visible marks is catching about 79.5% of the
marks the source actually holds.

**And what the source holds is not what happened.** `0041` already accepts that
the undetected fraction is unknowable, because no platform publishes its own
detection rate. That residual is larger than the difference between any two
rows of the table above, which is the strongest argument against spending
months to close the gap between them.

So the corpus claims a stated share of a bounded disclosure of an unbounded
population, and says so in those terms rather than reporting a single figure
that reads as completeness.

## What was declined

**Querying from several addresses.** The limiter is per-address, so this
multiplies the rate linearly. It is circumventing a documented limit, and this
project identifies itself in a header precisely so that a source that wants to
refuse it can. Named here so that it is decided rather than drifted into.

**Asking the source to change what a closed account discloses.** The
short-circuit strips rating, profile, play time and the mark together, and the
same change that introduced it deliberately kept closed accounts in the
response. It is a decision about people who closed their accounts, not an
oversight, and the project has no standing to ask for it to be reversed.

**Waiting on a reply before deciding.** Asking the source for a bulk artifact
or a raised allowance costs nothing and may yet improve on all of this. It
cannot be the plan: a reply that never arrives would block core designation
indefinitely, and this decision has to hold whether or not one comes. Anything
the source grants starts a better snapshot later, which `0041` already treats
as a new corpus rather than an amendment to this one.

## Consequences

- The census stops being a blocker. `#358` produces an artifact that improves
  while everything downstream proceeds, and `#90` is no longer sequenced behind
  it.
- That transfers a real cost to whoever designates the core. `0041` records
  that removal after designation is impossible, because expansion must preserve
  containment, so the recall readable on that day is the recall the evaluation
  reference carries permanently. This decision makes that a deliberate reading
  rather than a deadline.
- The snapshot format has to change. It currently claims to speak for every
  account in the archives it covers, which lets preparation read an unlisted
  account as unmarked; under a partial census an unlisted account is genuinely
  unknown, so the snapshot has to record what it asked about and what coverage
  that reached. `#358` owns that change.
- The progress file has to become a store keyed by account rather than an
  offset into a fixed list, since the account list grows with every archive
  acquired and the current file refuses to resume when it changes. It also has
  to move out of the repository tree, which `#358` already requires: it sits
  beside the snapshot at a gitignored path today, and `git worktree remove`
  deletes ignored files, which is how the first census was lost. `#358` owns
  both.
- `LICHESS_TOKEN` becomes part of the machine's setup rather than the
  repository's, as a credential. It carries no `ANTHRO_CHESS_` prefix because
  it is a credential for an external service rather than a setting of this
  project, and that is the name the wider Lichess tooling already uses.

## References

- `#391`, `#358`, `#90`
- `docs/data.md` (Sampling And Weighting)
- `configs/data/marked-accounts/README.md`
- `docs/decisions/0041-games-of-marked-accounts-leave-the-corpus.md`
- `docs/decisions/0045-centisecond-clocks-from-a-closed-export.md`
