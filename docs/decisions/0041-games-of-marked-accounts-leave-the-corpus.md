# 0041: Games Of Marked Accounts Leave The Corpus

Date: 2026-08-08

## Status

Accepted. Settles `#105`, and supersedes the prevalence premise in its body.
It is the recorded exception to
`0016-sampling-axes-versus-measured-distributions.md`, which otherwise keeps
preparation filters to validity and sends editorial choices to training
selection.

## Context

`#105` deferred this question on the grounds that "prevalence is low and
openings are the least affected part of a cheated game". The second half holds.
The first half is wrong, and measuring it is what reopened the question.

Against the live Lichess `tosViolation` flag, on the 2017-04 standard rated
blitz selection:

| Quantity | Estimate |
| --- | --- |
| Accounts since marked | 4.27% (3.36-5.41%, n=1500) |
| Games with at least one since-marked player | 8-10.4% |
| Same, engine marks rather than boosting | ~6-8% |
| Assisted **moves**, inferred | 0.2-2%, unmeasured |

The source does not help. Lichess's monthly exports select on creation month,
rated flag, and variant, with no filter on marks or account status, and the
exported user projection carries only username and title, so account status
cannot reach the PGN even in principle. No regeneration has ever applied a
removal. Any belief that the database arrives pre-cleaned is false.

Two caveats run in opposite directions. `tosViolation` is *ever-marked as of
the query date*, undated, and conflates engine assistance with boosting and
sandbagging. Against that, 13.2% of these accounts are now closed and a closed
account exposes no flag at all, so the figures are lower bounds.

## Decision

**Every game an account marked by its source played is rejected during
preparation.** The games leave every split, not only the training one.

**The filter acts on accounts, not on moves or games.** No detector is built.

**The account label is a snapshot, pinned and checked in, never queried while
preparing.**

## Why per-account

Per-game detection does not exist, at any price, and this is the finding that
decides the shape of the filter rather than a caveat on it.

Single-game ROC-AUC for an aggregate match-rate and move-quality detector, with
full engine analysis, is 0.53 at 15% assisted moves, 0.62 at 50%, and 0.75
against a cheater who preserves the aggregates. It is at chance in exactly the
sparse-assistance regime a rational cheater occupies, and one well-chosen
engine move already lifts expected score from 0.51 to 0.71. Regan's own
variance model needs an 89-95% top-1 engine match within a single game to clear
the threshold FIDE uses; a Regan-style system applied to 120,000 pre-2005
games, when engine assistance was near-impossible, flagged at least 92 players.

Rating dependence compounds it: baseline top-1 engine match runs from 39% at
1400 to 61% at 2800, so reaching equal confidence needs roughly eight times
more moves at 2700 than at 1500, and a fixed threshold would filter unevenly
across an axis the model conditions on. Engine analysis at corpus scale was
measured on the project's own machine at 1.9 days for depth 12 and 7.0 days for
a 100k-node budget over ~70M positions. The runtime is affordable; the
discrimination is not, and it would add a non-PyPI binary dependency and a
second artifact family nobody would ever regenerate to verify.

Two labels the PGN carries directly were measured and are far too narrow to
serve. `Termination "Rules infraction"` is a client-side self-report rather
than a moderation event, at 27 per million eligible games and a median 15 plies
against 63 for ordinary blitz. Absent `RatingDiff` tags are much sharper — 81.9%
of those players are marked (76.3-86.4%, n=221) against 4.0% of a control drawn
from ordinary games, a 18.6-fold enrichment — but they occur in 234 games per
million, so between them the two reach a recall near 0.2%.

The source's own account-level judgement has none of these problems. It costs
one bulk request per 300 accounts, needs no inference, and covers the whole
population rather than a structurally selected sliver.

Its cost is precision of a different kind: the label is undated and covers
boosting as well as engine use, so roughly a tenth of a month's games are
removed to eliminate an assisted subset that is plausibly one or two percent.
Most of what goes is honest play by somebody who cheated at some point, perhaps
years later. That is accepted deliberately. A marked player's honest games are
not what this corpus is for, no method distinguishes them, and the project has
far more source games than it takes.

## Why the corpus and not training selection

`0016-sampling-axes-versus-measured-distributions.md` says to prefer training
selection for every editorial choice and to keep preparation filters to
validity. This is the deliberate exception to that rule, and it is recorded as
one because a preparation filter is a definitional statement about what counts
as human play for every later benchmark.

Three things make the exception right here.

The project's entire subject is human-like play. An evaluation reference
containing engine-assisted games measures the wrong target: a model that
matched real humans exactly would read as deficient against it, and the suite
would reward engine-likeness in the one direction the project must not drift.

The human-versus-engine classifier draws its human class from the pool, so
contamination there is not a level shift that cancels in a comparison but
training data for the instrument, and its mandated dataset freeze would make
that permanent for the instrument's life.

And the window is open. Nothing is protected before the evaluation core is
designated, the roadmap sequences this decision ahead of the breadth pass
precisely so it can remove games, and removal after designation is impossible
because expansion has to preserve containment.

What the exception costs is real and is accepted. It is irreversible: the
games are gone from every split, so the effect of removing them can never be
measured against a reference that still contains them, and no future checkpoint
can be scored on them. A benchmark cannot detect a bias it shares with its own
reference, and this filter is now part of both.

## Why the label is pinned rather than queried

Everything else the pipeline reads is reproducible from a digest. The archive
is fetched by URL and verified against a published checksum, and the same bytes
give the same corpus forever.

Account status has no fixed input. It is a live judgement that only ever
accumulates, so asking the source again returns a different and larger answer.
A preparation that queried it would produce a different corpus on every run of
an identical configuration, and each run would remove games the last one kept —
silently shrinking any evaluation pool built on it and breaking the superset
property `0013-benchmark-result-comparability.md` depends on.

So the answer is taken once, pinned to the archives it covers, and checked in.
Refreshing it is a deliberate act that starts a new corpus, exactly as changing
an archive digest is. A snapshot covers the one archive it was built for, and
widening one to a second is left to whatever first prepares from two: it has to
carry every earlier verdict over untouched and query only genuinely new
accounts, because re-deciding an account an earlier snapshot already spoke for
applies a later moderation decision retroactively.

Preparation refuses an archive the snapshot does not cover rather than
preparing it unfiltered, so once a selection names a snapshot, widening the
corpus and forgetting to refresh it stops the run instead of quietly keeping
every account nobody asked about. Naming the snapshot in the first place is not
enforced that way — an unset `filters.marked_accounts` prepares unfiltered, and
is indistinguishable from a deliberate choice not to filter. That is why the
setting lands with the artifact rather than ahead of it, and why `#358` blocks
`#90` rather than being left as a step to remember.

Usernames are stored as truncated salted digests, because membership is all
preparation needs and this repository is public. What that achieves is bounded
and worth stating: the salt is checked in and the account space is the covered
archive's, so anyone holding that archive can recover the names. The mark is
the source's own published judgement rather than a finding of this project's,
and the digests keep the repository from republishing it as a list anyone can
read or search — which is the sense in which `docs/evaluation.md` says this
project makes no claims about real player cheating.

## The residual, which is smaller but not zero

Filtering marked accounts removes the contamination the source caught. It does
not remove the contamination nobody caught, and there is no way to estimate how
much that is: the fraction of cheaters a platform detects is unpublished by
every platform.

So the reference is cleaner rather than clean, and the residual bias runs in
the same direction as before — the human reference still reads very slightly
stronger and more engine-like than real human play, most sharply in adjudicated
decisions, where contamination concentrates in the mate-available and
material-gain populations that family scores, and in held-out move prediction.
Opening repertoire is least affected, since assistance begins after the book
phase. The puzzle families, the self-play ladder, the engine anchor, legality,
novelty, and the efficiency families are structurally immune.

The magnitude of what remains is below the seed-to-seed variance
`0029-model-change-control-arm.md` records, which is also why this change is
not accompanied by a control-arm reading: at this prevalence such a reading
would be null by construction, and a null reading is grounds for removing a
change rather than keeping it.

## Keeping the opponent's half was considered

The obvious economy is to keep each rejected game and mask only the marked
player's plies, since the opponent did nothing wrong. It is mechanically
straightforward: loss masks exist for exactly this, and the marked player's
moves would stay as context, which is what the opponent actually faced.

It is declined because the corpus is not data-limited. The selection takes its
configured bound from a month holding several times that many games in this
speed alone, so rejecting roughly a tenth of them costs a slightly deeper scan
rather than any data — the same bound is reached with games nobody has to
reason about. The plies recovered would be a few percent of the total, bought
with a schema field, a loss-mask path, and their tests.

The evaluation half is the stronger objection. A game one player cheated in has
a result, a length, and a termination shaped by that, and those are exactly what
the game-level benchmarks read from the reference. Such a game would have to be
excluded from the pool while being half-included in training, which is a second
mechanism to keep consistent with the first.

Nothing about this is foreclosed. The archive is pinned and the snapshot is the
label that finds these games again, so a corpus of humans playing against engine
assistance — of interest to the human-versus-engine classifier — can be selected
from the same source whenever it is wanted.

## Consequences

The corpus loses the games of marked accounts, so the evaluation pool becomes a
new generation rather than a wider one, and readings recorded against the
previous generation do not compare across the seam. `#90` cuts the next
generation after this one and its superset check applies from there.

None of that has happened yet. The decision is settled and the mechanism is in
place, but the baseline selection still prepares unfiltered because the snapshot
it would name is incomplete: the source rate limits the lookup hard enough that
covering one archive takes several sessions spread over days, and two thirds of
one is built. The filter turns on when that artifact exists, and the pool
generation moves then rather than now.

`#89` widens the corpus and must refresh the snapshot as part of that work; the
run fails until it does.

Revisiting this is worth it only on new evidence rather than on reflection.
The two that would matter: a published estimate of what fraction of cheating a
platform detects, which would say how much residual remains; and a per-game
detector with a validated operating point, which would let the filter act on
games instead of on people and give back the honest games this one discards.

## References

- `docs/data.md`
- `docs/evaluation.md`
- `configs/data/marked-accounts/README.md`
- `docs/decisions/0011-held-out-test-partition.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/decisions/0016-sampling-axes-versus-measured-distributions.md`
- `docs/decisions/0029-model-change-control-arm.md`
