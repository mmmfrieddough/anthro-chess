# 0072: The Clock Says Which Pool A Rating Came From

Date: 2026-08-19

## Status

Accepted. Refines `0005-lichess-default-rating-scale.md`, which chose a rating
scale without naming which of a source's pools that scale is. Extends
`0056-the-speed-axis-is-derived-from-the-time-control.md`, which keeps the
namespace and the speed class as separate derivations because the two can
disagree: this record measures that in the corpus as
`0058-the-corpus-starts-after-the-rapid-pool-split.md` cut it, they never do.

## Context

The runtime carries one target rating and the corpus carries six pools. Bullet,
blitz, rapid, classical, ultrabullet, and correspondence all reach the
normalized rating column as one number on one axis, and a 1600 in one of them is
not a 1600 in another. 0005 named the scale Lichess-like and said nothing about
which pool, which was complete while the corpus was a single month of a single
speed and is not complete now.

The field states the problem and answers it by splitting. Maia-2 says ratings
across game types "are not comparable (e.g. a rating of 1800 in Rapid is
significantly weaker than a rating of 1800 in Blitz on Lichess)", names Maia-1
for mixing them, and ships one model per pool. Chessformer trains on blitz
alone and Allie on blitz alone.

So the question is whether this project needs a namespace input beside the
rating, a rating-system input beside that, or neither.

## What Was Measured

**The namespace is a deterministic function of the time control in this
corpus.** Cross-tabulating `source_rating_namespace` against the class derived
from the normalized clock, over 300,000 games drawn from six random shards:
every game agrees, and nothing lands off the diagonal. The corpus manifest
carries the same result at full size, where the per-namespace game counts equal
the per-class counts on all six categories across 2,087,063,655 accepted games.
Correspondence is included, reaching the derivation as a clockless game with no
class at all.

The agreement is structural rather than lucky. Lichess rates a game in exactly
one pool and picks that pool by banding the time control, which is the banding
`anthro_chess.data.speed` reproduces. The two derivations can part only where
the source's own vocabulary has moved, which is the 2017 relabelling 0056
documents, and 0058 cut those months out of the corpus.

## Decision

**One rating input, and the clock says which pool it is denominated in.** The
model reads the mover's rating and the game's time control. The pool is
recoverable from the second, so the rating is not asked to carry it, and the
dial means the player's rating in whichever pool the chosen control lands in.

**No namespace input.** It would be collinear with the time control by the
measurement above, so nothing in training could teach the model to use it as
anything but a relabelling of the clock. The combinations that would make it
worth having, a blitz rating attached to a bullet game, do not occur in the data
and cannot: a source that rates per pool never produces one. Adding the input
would put an axis in front of a caller whose off-diagonal settings were never
trained and cannot be, which is worse than not offering it.

**No rating-system input.** Every rating in the corpus comes from one system, so
such an input would be constant wherever it could be observed and would have no
contrast to learn from. The information needed to design it arrives with a
second system and does not exist before then.

**A foreign rating system masks its rating rather than converting it.** A source
that does not declare its ratings normalized contributes an absent rating, which
reaches the model as the unrated embedding. Those games still train move choice,
terminal behavior, and timing without moving a dial denominated in someone
else's units.

**The model reads the raw control, not the class.** The class is a function of
the initial clock and the increment, so giving both is strictly more informative
than giving the band, and the bands are wide: two controls inside one of them
are not played alike.

## Why The Field's Split Does Not Transfer

Every project that trains one pool shares a property that this one will not.
None of them models the clock. Maia-2 has no time head and drops positions under
thirty seconds, and Chessformer's released checkpoints predict no time.

Without a clock input the pool is genuinely unrecoverable, a 1600 bullet game
and a 1600 classical game are the same input with different distributions, and
splitting the corpus is the only defense available. That defense is the input
this project is committed to adding, so the precedent argues for a workaround to
a constraint this project does not share.

Keeping every pool is also what the product asks for. `docs/vision.md` commits
to clocked play and a realistic time to move across controls, and a model
trained on one band delivers that band with a disclaimer. Dropping the others is
not a smaller sample of the same behavior either: it removes the region where
longer thought and deeper preparation live, which is the argument
`0062-the-breadth-corpus-filters-for-validity-alone.md` already made against
editorial filtering, applied at the loader instead of at preparation.

## What This Gives Up

**The dial is not denominated absolutely.** The same number at two controls is
two different strengths, because that is what the source's own pools mean. A
caller who wants comparable strength across speeds does not get it from one
setting, and no convention available here would provide it.

**No published result backs a multi-pool dial.** The single-pool papers are the
prior art, and this leaves it deliberately. What the measurement establishes is
that the information the split protects is present in an input this project has
committed to, not that a model trained across pools reads well.

**A rating-conditioned reading taken before the clock is an input is taken over
a mixture.** The model cannot separate the pools until it can see the control,
so a dial read on a widened selection before then is reading a blend and will
understate what conditioning is worth.

## Consequences

Rating-conditioned comparisons are read per speed rather than in aggregate,
which is what an unconditioned arm regressing toward the dominant speed would
otherwise hide in the blend.

Whether the self-play rating ladder needs a speed axis of its own is left open.
Self-play games carry no source row to read a namespace from, so if the ladder
separates pools at all it separates them by the configured control's class,
which is the axis benchmark slices already report on.

## References

- `0057-the-rating-namespace-is-derived-per-game.md` for why the namespace is
  read per game and why an absent one beats a guessed one
- `0009-decision-only-rating-conditioning.md` for the mover-only rule this
  leaves untouched
- `0062-the-breadth-corpus-filters-for-validity-alone.md` for the axes the
  corpus keeps whole
- `docs/data.md` (Rating Scale) for what the dial is denominated in
- `docs/research.md` (Human-Like Chess Modeling) for the single-pool readings
  this argues against transferring
