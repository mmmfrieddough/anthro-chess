# 0034: A Ladder Reading Is Qualified By Refitting Resampled Games

Date: 2026-08-04

## Status

Accepted. Extends `0026-conservative-dispersion-bounds.md`, composes with
`0032-a-replayed-reading-has-no-evaluation-noise.md`, and settles the gap
`0027-settled-rating-ladder-grid.md` left open.

## Context

The first full-size ladder reading put every seat within a 10 to 17 Elo span
across a 900-Elo configured range, with pairwise ordering at chance, at both
checkpoints of `training-blitz-30k-v4`. That was written up as a flat rating
transfer. It may well be one. Nothing in the output established that it was
above what the ladder can resolve, because the ladder reported no floor of any
kind beside any number it produced.

That is not a small omission on a benchmark whose deliverable is a null. A seat
is placed from `(seats - 1) x games per pairing` games — 1,344 at the declared
grid — of which roughly half reach the ply limit and inform no comparison. A
Bradley-Terry rating estimated from a few hundred effective games has a standard
error, and ordering, slope, span, ladder error and both temperature responses
are all functions of those ratings, so each inherits it.

Decision 0027 declines to trade per-seat sample for wall-clock time on exactly
this ground: "the transfer is flat" and "the sample is too thin to see one" are
not distinguishable from the output.

The gap is structural rather than an oversight. The data-sampling bootstrap the
rest of the suite uses resamples per-game additive contributions, and a fitted
rating is not one — no game carries a share of it. The paired estimator decision
0033 keeps cannot serve here either, and 0033 says why: a ladder generates its
own games, so two checkpoints share no sample to pair on.

## Decision

**A ladder redraws the games its pairings played, refits, and reads the spread
of everything the fit yields.** One resample is one whole reading.

### Refit Rather Than Propagate

The alternative the issue named is the fit's own curvature: a maximum-likelihood
Bradley-Terry fit yields standard errors from its Hessian at no extra games.
That reaches the ratings and stops there.

Ordering is a step function of rating differences and the ladder error is a mean
of absolute ones. Neither has a derivative for a standard error to be propagated
through, so the delta method has nothing to say about the two quantities the
benchmark is most often read for. A curvature estimate would also describe a
Bernoulli outcome the ladder does not observe, since draws enter the fit as half
a point to each side; and it counts a pairing of greedy seats as evidence,
though re-running reproduces that pairing exactly.

Refitting has none of those problems because it is not an approximation of the
reduction — it *is* the reduction, run again. Every quantity is reached by the
route the reading itself took, which is also what keeps a floor from describing
a slightly different quantity from the number it sits beside.

It is affordable. A refit of the declared 15-seat, 105-pairing grid costs about
2.9 ms in this checkout, so the shipped thousand resamples cost roughly three
seconds against a ladder that has run in about two hours.

### The Floor Is Evaluation Noise, And Travels On The Measurement

`docs/evaluation.md` already settles the kind: a rollout has no fixed data to
re-measure on, so bootstrapping the generated games and re-running under another
seed estimate the same quantity. The floor is `evaluation`, which is the kind
that qualifies a delta between two checkpoints.

It is attached to each measurement rather than recorded as a characterization
against the series. Seeds, openings and games per pairing are deliberately
outside a ladder's series identity — decision 0022 keeps them out because more
games estimate the same ladder more precisely — so a floor filed against the
series would later be looked up beside a reading taken at a different sample
size and be wrong by whatever the two sizes differ by. A floor that is a
function of the reading's own sample can only be right where the reading is.

### A Pairing That Replays Is Held Fixed

Decision 0032 says a comparison whose model side cannot vary states a floor of
zero rather than estimating one. A ladder is the mixed case that decision did
not have to face: at the declared grid ten of the 105 pairings are between two
greedy seats and replay move for move, while the other 95 redraw.

Those ten are held at the results they produced and only the rest are resampled,
which is the same rule applied where it bites. The predicate is
`replicates_vary`, the one the harness already uses to decide how many
replicates a pairing is worth playing, so the two cannot disagree about whether
a suite's games are forced. A ladder whose every seat is greedy states zero
throughout and records that it stated rather than estimated.

The games behind the bound's degrees of freedom are therefore the redrawn ones.
Games that replay contribute no spread, so counting them would buy certainty
about a dispersion they say nothing about.

### The Two Degenerate States

Both states 0022 reports as results rather than errors get an answer, and the
answers differ because the states differ.

**A fit that did not converge** is still qualified. The spread of an estimator
that is struggling is still the spread of the number the benchmark reports, and
suppressing the floor would leave a reader with a bare number in exactly the
case where they most need to know what it can bear. What is reported beside the
floors is how many resamples also failed to converge, which is the thing that
says how much to trust them.

**A seat the fit clamped is not qualified, and the output says so.** Its number
is the declared spread rather than an estimate, and no resample can improve on
that: every resample of a seat that won every game wins every game again, so its
spread is exactly zero and a floor built from it would license any delta at all.
That is the `0.0000` ambiguity #175 raises, arriving through the one mechanism
where the answer is knowable before a resample is drawn — so it is settled
structurally, from `RatingFit.clamped`, rather than by inspecting the output of
the bootstrap.

The reading names each such quantity and why, in its own result and in `anthro
eval ladder`. `no_sampling_floor_reason` is not the vehicle: that declaration is
a property of a metric in the registry and refuses only the data-sampling kind,
where this is a property of one seat in one reading and refuses the evaluation
kind.

### The Scored-Game Count Joins The Headline

`ladder.scored_game_rate` is now a metric per seat, floor and all. It was
previously in a log line, and on the one full-size reading taken it is the
quantity that discriminated between the checkpoints — scored games rose from
3,405 to 5,273 of 10,080 — while every headline number sat inside what the
ladder could resolve.

A rate rather than a count, because sample size is deliberately outside series
identity: a count would move when the seeds did, and stop being comparable
across two readings of the same ladder.

It is directional where the rest of the family is not. Decision 0030 puts the
ply limit past the longest game in the corpus, so a seat playing the way its
corpus does reaches it essentially never, and the share rising is the model
learning to finish games rather than a dial being turned.

## Consequences

Nothing the ladder measures changes value, so no ladder series ends and the
reading taken before this can be compared against one taken after. What changes
is that the second one says what it can resolve.

**The estimate errs wide by the spread between openings.** The draw is over a
pairing's games without regard to which opening each came from, which is the
estimator the generated-play family already uses. The openings are frozen, so a
fresh seed replays the same set and the between-opening component is not
something a re-run redraws — `docs/evaluation.md` states the general rule that a
floor qualifying a delta must exclude what the two sides share, and this floor
does not.

Stratifying the draw by an opening and a colour is the correction, and it is
filed rather than taken here for two reasons. Its size is unmeasured, and it has
a failure mode in the direction that matters: a stratum holding one game shows
no spread at all, so a reduced grid run at one seed would report a floor of zero
where the unstratified draw reports one that is merely too wide. A floor that
understates is the failure floors exist to prevent, and choosing between them
needs the measurement rather than an argument.

**The error profile is not qualified.** A seat's preferred-selection rate,
policy regret and selected rank are means over decisions rather than outputs of
the fit, so the refit does not reach them and their deltas report the noise as
unknown. That is the correct verdict — it is a floor somebody could still
produce, by retaining per-game decision contributions the way the dependency
family does — rather than the `unqualifiable` one.

Two readings of one checkpoint at different sample sizes now report different
floors, which is the point and is worth stating because nothing else in the
suite behaves that way: every other floor is characterized against a series and
looked up. Raising seeds, openings or games per position is the lever decision
0027 names for buying precision, and this is where that purchase becomes
visible.

## References

- #190 — the gap this closes; #146, `docs/planning/first-full-suite-reading.md`
- #177 — the flat-transfer finding this exists to qualify
- `docs/decisions/0022-one-joint-rating-ladder-fit.md`
- `docs/decisions/0026-conservative-dispersion-bounds.md`
- `docs/decisions/0027-settled-rating-ladder-grid.md`
- `docs/decisions/0030-ladder-ply-limit-at-the-trained-bound.md`
- `docs/decisions/0032-a-replayed-reading-has-no-evaluation-noise.md`
- `docs/decisions/0033-pairing-is-a-correctness-fix-not-a-resolution-lever.md`
- `docs/evaluation.md` — "Noise Characterization", "The Implemented Ladder"
