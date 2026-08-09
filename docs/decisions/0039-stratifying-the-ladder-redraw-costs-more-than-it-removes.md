# 0039: Stratifying The Ladder Redraw Costs More Than It Removes

Date: 2026-08-08

## Status

Accepted. Settles the between-opening consequence
`0034-qualifying-a-rating-ladder-reading.md` filed, and answers the same
question for the generated-play curve family. Its rescaled draw is taken by
`0042-the-puzzle-response-is-qualified-within-its-reading.md`, which applies it
to the puzzle family and to the paired estimator.

## Context

A ladder qualifies its reading by redrawing each pairing's games and refitting.
The draw is over a pairing's games without regard to which opening each came
from, and 0034 recorded that as a known defect: the openings are frozen, so a
fresh seed replays the same sixteen and the spread *between* them is common-mode
across two checkpoints read on the same ladder. `docs/evaluation.md` states the
general rule that a floor qualifying a delta must exclude what the two sides
share, so every ladder floor was expected to be wider than the delta's real
run-to-run variability, by a factor nobody had measured.

0034 named the correction — stratify the draw so the only thing a resample
moves is what a seed decides — and declined to take it for two reasons. Its size
was unmeasured, and it has a failure mode in the direction that matters: a
stratum holding one game shows no spread at all.

`#302` asked for the measurement first, and to settle the small-stratum problem
rather than argue it.

## The measurement

Three readings on `issue-203-treatment` step 8000, the strongest checkpoint this
machine can load. A stratum throughout is one opening and one colour assignment,
which is the pair of inputs a re-run holds fixed.

### The component itself, at the declared grid

The shipped selection, unchanged apart from the checkpoint: 15 seats, 105
pairings, 9,440 games, of which 95 pairings redraw and 10 replay. Each redrawn
pairing holds 32 strata of 3 games. A one-way random-effects decomposition of
each outcome indicator over those 380 pairing-by-category cells gives the design
effect — the factor by which the unstratified draw's variance exceeds the
stratified one, which is `1 + between/within`:

| quantity | design effect | floor width |
| --- | ---: | ---: |
| first wins | 1.004 | x1.002 |
| draws | 0.997 | x0.999 |
| first losses | 0.992 | x0.996 |
| unfinished | 1.012 | x1.006 |
| all four pooled | 1.004 | x1.002 |
| game length in plies | 1.011 | x1.006 |

A between-stratum variance cannot be negative, so the design effect cannot truly
sit below one and the three rows that do are what an estimate of zero looks like
at this sample. The component `#302` expected to bite hardest, the unfinished
share behind `ladder.scored_game_rate`, is the largest of them and is worth six
parts in a thousand of floor width.

### Whether a deeper opening would change that

Eight plies of a human blitz game may simply be too shallow to separate
anything. Replayed at three depths on a reduced six-seat grid over the same
sixteen openings:

| prefix plies | outcome deff | game-length deff |
| ---: | ---: | ---: |
| 8 (declared) | 1.011 | 0.989 |
| 24 | 0.986 | 1.009 |
| 40 | 1.008 | 1.047 |

No trend, and the largest figure in the table is a 2.3% widening at five times
the declared depth.

### What the estimators say against a real re-run

The decisive reading, because it needs no decomposition to be believed. Twelve
independent seed triples of the reduced grid over the same frozen openings —
which is exactly what a re-run is — give the true run-to-run standard deviation
of every quantity the ladder reports. Each replicate also reports what three
redraw schemes estimate that spread to be. Over 31 qualified quantities:

| redraw | ratio to the true spread |
| --- | ---: |
| over a pairing's games, as shipped | x0.99 |
| within each stratum, plug-in | x0.83 |
| within each stratum, rescaled | x1.00 |

The shipped estimator is calibrated. The stratified one is 17% narrow.

## The stratified draw is biased low, and the small stratum is that bias at its limit

The third row is what settles it. A plug-in bootstrap draws from the observed
proportions, and the variance of a stratum of `n` games drawn that way is
`(n-1)/n` of the variance a fresh stratum of `n` games actually has. At the
declared grid a stratum is three games, so a stratified draw understates every
dispersion by `sqrt(2/3) = 0.816` before it removes anything. Measured, the
stratified-to-unstratified ratio is 0.837 against the 0.816 the bias alone
predicts, which leaves a between-opening component of nothing.

The rescaled row is the same draw with that bias removed — draw `n-1` of each
stratum's `n` games and scale the counts back up, which is the standard
correction and is exact for a mean. It lands at x1.00, on top of where the
unstratified draw already sits. So the correction that makes stratifying honest
also makes it pointless: both estimators reproduce the true spread, because there
is no common-mode component between them.

This reframes the small-stratum problem 0034 filed. The one-game stratum is
where `(n-1)/n` reaches zero, the plug-in draw loses all of the spread rather
than a share of it, and the rescaled correction is undefined — `n-1` games is
none. So a stratified draw of either kind needs two games in every stratum
before it means anything, which is the guard the puzzle family already carries
in `puzzles_per_rating`. What the measurement adds is that two is not where the
problem stops: at three games the plug-in draw is still the understatement above,
which nothing in the output would have shown, and a floor that understates is
the failure floors exist to prevent.

## Decision

**The ladder keeps drawing over a pairing's games.** `LadderPairing` keeps its
aggregate outcome counts, no per-stratum counts are retained, and the reading
gains no second estimator to choose between — which is also what keeps `#294`'s
defect from being reintroduced here in a new family.

The reason is not that the correction is hard. It is that the thing it corrects
is not there.

## The curve family is the same decision

`#302` left open whether the generated-play family needed its own record. It does
not.

Where that family fields a human-prefix arm at all — its selection file owns
which arms are fielded, and says why — the arm continues prefixes drawn from the
same frozen pool through the same generation harness. Its view and its prefix
depth are its own, but those are a count of openings and a depth rather than a
different kind of position source, and the sensitivity table above brackets the
depth and finds neither lever moving the component. The measurement also covers
the shape of quantity a curve comparison reduces, since game length is a
per-game continuous feature rather than an outcome. One decision covers both.

## Consequences

Nothing changes value, no series ends, and `#328` can take the definitive
ladder reading against the estimator as it stands.

What 0034 recorded as a known defect is withdrawn rather than deferred, and its
whole consequence goes with it: the component, the stratified correction it
named, and the sizing it said that correction still needed.

**What would reopen it.** The measurement is one checkpoint on one machine, and
that checkpoint is 8,000 steps into a run rather than a converged model. A
stronger model is the case where an opening plausibly starts to decide a game,
and the deeper-prefix rows are the closest available proxy rather than a
substitute for reading it again on a converged checkpoint. Reading it again is
the decomposition above run over a ladder's retained games, and needs no new
play while that ladder's detail tier survives: group its games by
`source_game_id` and the seat that held white, and read the between-stratum
component out of the two mean squares. The trigger is a design effect whose
floor-width factor — its square root — is worth having; the table above is the
conversion.

A reopening that finds one has to take the rescaled draw rather than the plug-in
one, which is where the `(n-1)/n` understatement stops being a footnote. It is a
property of every plug-in bootstrap in the suite and negligible wherever the
resampling unit count is large, but stratifying cuts that count to the games in
one stratum, and there the plug-in draw only beats the shipped estimator once the
design effect exceeds `n/(n-1)` — 1.5 at a three-game stratum. Below that its own
bias is the larger error: at a design effect of 1.1 it would trade a floor 5%
wide for one 22% narrow.

## References

- `#302` — the measurement this records; `#190`, where the gap was found
- `docs/decisions/0034-qualifying-a-rating-ladder-reading.md` — the consequence
  this settles
- `docs/decisions/0026-conservative-dispersion-bounds.md` — why a floor is built
  from a bounded spread rather than a measured one
- `docs/decisions/0032-a-replayed-reading-has-no-evaluation-noise.md` — the
  pairings this draw holds fixed
- `docs/decisions/0033-pairing-is-a-correctness-fix-not-a-resolution-lever.md` —
  a floor wrong by a known factor is a defect, whichever direction it errs
- `docs/evaluation.md` — "Noise Characterization", "The Implemented Ladder"
