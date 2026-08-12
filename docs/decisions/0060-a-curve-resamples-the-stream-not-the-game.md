# 0060: A Curve Comparison Resamples The Stream, Not The Game

Date: 2026-08-12

## Status

Accepted. Settles what
`0032-a-replayed-reading-has-no-evaluation-noise.md` left open, and refines
`0026-conservative-dispersion-bounds.md` on what counts as a replicate here.
Takes the rescaled draw `0039-stratifying-the-ladder-redraw-costs-more-than-it-removes.md`
named as the one a reopening has to use.

## Context

A generated-play curve comparison estimates its own spread by resampling the
games it generated. 0032 measured that estimator at temperature zero, where the
right answer is knowable exactly, and found it reporting a floor of 23 plies
against a true run-to-run movement of zero. The mechanism it identified was
support dropout: at the declared bandwidth a grid point is estimated from the
games at that rating alone, so a resample that misses them all drops the point
out of the conditional mean, and the spread of *which points survived* is what
the floor reported.

0032 fixed the deterministic case by stating a zero, and deliberately left the
general one open: any model side thin enough for a resample to lose a grid point
often reports that fragility as measurement noise, and the reduced sweep reads
every temperature row at exactly that size. `#279` asked which of the two the
number is at realistic sizes, and to settle it by measurement rather than by
argument.

## The measurement

`issue-203-treatment` step 8000 under the shipped rollout selection, with the
reduced sweep's own sample counts — one game per position, so twelve generated
games across the six-point rating grid — played independently at 32 seeds, at
temperatures 0.7 and 1.0. A seed is a whole re-run of the reading, so the spread
of a distance across the 32 is the run-to-run movement a floor claims to bound,
measured rather than estimated. Every seed also reports what resampling its own
games says that spread is. Twenty quantity-by-temperature rows throughout;
figures are the median ratio to the true spread.

### The floor is about half the movement it claims to bound

Readings are assembled from `k` seeds each, so the same sample prices both
resampling units at three reading sizes. A seed contributes two streams, being
the two colour assignments.

| reading | streams | draw over games | draw over streams | rescaled |
| --- | ---: | ---: | ---: | ---: |
| 1 seed — the shipped reduced sweep | 2 | x0.53 | x0.48 | x0.72 |
| 2 seeds | 4 | x0.70 | x0.85 | x0.99 |
| 4 seeds | 8 | x0.66 | x0.98 | x1.04 |

The pooled distance behaves the same way: x0.56, x0.71, x0.65 for the draw over
games, against x0.66, x1.00, x1.02 for the rescaled draw over streams. Both are
the metrics decision 0020 names as the ones that rank two checkpoints.

So the shipped estimator is not inflated by support churn. It is *narrow*, by
about a factor of two, and it stays narrow as the reading grows — the direction
a floor exists to prevent, because a floor that understates licenses noise as a
finding.

### The games in one reading are not independent

The harness derives a game's seed from the position, the colour assignment and
the replicate index, and **not** from the conditioning rating. One stream
therefore plays a game at every point of the rating grid, and twelve games are
two draws rather than twelve.

Rebuilding readings whose ratings come from *different* seeds — the same games
with that shared stream broken — costs x1.41 of spread (geometric mean x1.50,
range x1.00 to x2.34). The mean correlation between two ratings' estimates
within a seed runs from 0.03 on cycle share to 0.73 on the share of available
theory a game consumed, which is what one expects: a stream that chose the
French Defense at 1300 chooses it again at 2100.

The rest of the shortfall is the plug-in bias 0039 already priced. A draw from
the units in hand has `(n-1)/n` of a fresh draw's variance, which at two games
per rating is `sqrt(1/2) = 0.71`. The two together are `1.41 x 1.41 = 2.0`,
which is the whole of the gap.

### Support churn is real and is not what the floor was reporting

At twelve games, 53% of resamples lost at least one grid point outright — 69%
for the repertoire quantity, whose model side is thinner still because a game
that stopped on a waypoint has no opening to contribute. So the mechanism 0032
identified is present at full strength.

It is not what the number is made of. Restricting the estimate to the resamples
that kept the point reading's support moves it from x0.53 to x0.48: removing the
churn makes the floor **narrower**, against a truth it was already half of. The
churn was covering a fraction of a much larger error in the other direction.

## Decision

**The resampling unit is the stream a game was drawn from, not the game.** An
observation names it, a side whose observations name none is drawn game by game
as before, and the human reference is that case — a pool game was played once by
two people, so it is its own unit.

**The degrees of freedom behind the bound are the streams, not the games.** This
is 0026's rule applied to the right noun: a rating grid multiplies games without
multiplying the draws behind them, so counting games claims a precision the
suite did not buy.

**The draw is rescaled.** Its spread is multiplied by `sqrt(m / (m - 1))` for
`m` streams, which removes the plug-in understatement exactly for a mean and is
what the third column above measures. The null levels take the same correction
on each side's own deviation, since they read it off the same replicates.
Negligible at the counts a full sweep plays and worth 22% at the three this
family will not go below.

**A model side that varies and holds fewer than three streams estimates
nothing** — no spread, and no null level either. Two streams leave three
distinct resamples, and reading a spread off them fails in both directions at
once: the table above has the estimate ranging over a factor of four there, and
the shakedown reading taken for this change found several quantities where the
two streams agreed and the estimate came out at exactly zero, which is a floor
that clears every delta. `SeedSpread` already withholds below three replicates,
and this is the same rule about the same kind of number. The levels go with it
because their model half is read off those same three outcomes; keeping them
would move the failure from a number a reader distrusts to a verdict they do
not.

**A replayed side is the exception**, and needs only a model side a resample can
move at all. Its floor is stated rather than estimated, and its own half of the
null is zero by construction rather than badly estimated, so 0032's consequence
holds as written: a greedy row states its zero and keeps its levels.

Fixing the unit disposes of support churn as a side effect rather than as a
second mechanism, wherever a stream reached every rating: it then contributes one
game per point, a draw over streams holds the grid's allocation exactly where a
re-run holds it, and a resample can no longer empty a rating the suite always
plays. The four quantities that drop games — a game that left the book on a
waypoint chose no opening — keep unbalanced streams and can still lose a point.
That residue is not the artifact this removes: a fresh seed's stream may fail to
reach the book at that rating too, so it is variation a floor should carry.

## Consequences

**The reduced sweep's sampled curve rows qualify nothing.** Its rollout step
plays one seed at one game per position, which is two streams, so every row
above temperature zero now reports an unknown floor, no null level, and an
unknown rating-response verdict — where it used to report a confident floor at
about half the true width. That is the honest reading of twelve games from two
draws, and the lever it points at is real: `generation.games_per_position` and
`grid.seeds` both multiply streams, while the rating grid does not.

So the reduced overrides now play four games per position rather than one, which
is eight streams and the count the table above measures as calibrated. It
triples the games the step generates and costs about a fourteenth of its wall
clock — 149 seconds against 157, one Linux CUDA host, `issue-203-treatment`
step 8000 — because the step is dominated by replaying and classifying the
12,000-game human reference rather than by generating anything. The count is
close to free against this step's fixed cost, which is the fact a budgeting
argument should start from.

**Eight streams qualify a reading; they do not rank two checkpoints with it.**
Read across steps 1000 and 8000 of one run, one of twenty comparisons cleared its
combined floor — which is what twenty comparisons give by chance at the declared
coverage, so it is not a finding. Game length's delta was 22 plies against a
floor of 77. Scaling the same figures, a delta that size wants something near the
full sweep's forty streams. The reduced sweep's curve reading is therefore a
smoke test that now says so, and ranking two checkpoints belongs at full scale.
Where the count should sit is a budgeting question, and this is the reading it
should be argued from.

The temperature-zero row is unaffected and keeps both, which is the one place a
reduced sweep still qualifies a generated-play distance.

**Nothing committed moves.** The store holds no generated-play or termination
result, and no distance changes value: the point reading is untouched and only
the number beside it moves. The stored dispersion records the stream count, and
its estimator name changes, so a reading taken before this change is
distinguishable from one taken after rather than silently comparable.

**The termination mix takes the same unit**, because it plays the same rating
grid through the same harness and so has the same defect for the same reason.
The measurement above is on the rollout family; what carries to termination is
the mechanism — a seed derived without the rating — rather than a second
reading.

**The temperature-zero row is unaffected.** It states a zero under 0032 and goes
on stating it, and a greedy row's two colour assignments are two streams, so its
null levels survive as 0032 said they should.

## What This Leaves, And Why It Is Not A Measurement

Two things sit next to this decision and are settled by argument rather than by
another reading.

**A zero is read off the curve, not off the reduction.** Decision 0042 keeps this
family's estimated zero where the shared arithmetic refuses one. A draw that
cannot move the model side does not reduce to exactly zero, though — it reduces
to the last bits of the curve arithmetic, and the shakedown below found
`6.34e-16` where the resample had moved nothing at all. That value takes neither
0042's exemption nor the refusal, and bounds into a floor that clears every delta
while reading as an ordinary estimate. So the question is asked of the model
curve — did any replicate differ from the point reading — which needs no
tolerance to answer, and the answer routes to the zero 0042 already prescribes.

**The flatness null is the wrong shape, and no permutation fixes it.** The two
distance levels now draw streams; `_flat_variation` still permutes games, and
switching it to permute streams would not be right either. A model whose policy
ignored the rating input meets the *same stream* at every grid point, so it plays
one game across the whole grid and its curve is exactly flat — at any sample
size, because every point is then estimated from the same games. The level a
no-response model reads at is therefore zero, and anything a permutation reports
above that makes `AVERAGE_HUMAN` fire more readily than it should.

That is derived from the seed derivation rather than measured, and the mechanism
is observable: at temperature zero on this checkpoint one stream played six
*different* games at the six ratings, so the rating input does reach the policy
and the degenerate case is the hypothetical rather than the observation. The
caveat is batching, which is not bit-for-bit identical across batch
compositions — measured elsewhere as not changing any move.

What it implies is larger than a permutation swap: the verdict as written — "as
flat as one with no rating response at all" — is answerable exactly and needs no
null. Whether that is the useful question, against "is the response large enough
to matter", is a design question about the verdict. The comparison already
computes `human_variation` over the same grid, which is the yardstick that
question would want.

## Alternatives Considered

`#279` proposed three, all of them ways to report the churn rather than to
remove it. The measurement rules out all three by finding that the churn is a
minority of a number that is wrong in the opposite direction.

**Report the share of resamples that lost a grid point, beside the floor.** This
makes a number legible that is not the number a reader needs. A floor at half the
true spread annotated with its churn share is still a floor at half the true
spread.

**Withhold the floor where that share passes a threshold.** Worse than
annotating, for 0035's reason: it discards the reading in the cases where the
estimate happens to be least bad, since the resamples that churn are the ones
carrying the extra width.

**Hold the supported point set fixed at the point reading's.** This has no
definition where it is needed. A replicate that lost a rating has no estimate at
that point to average, so fixing the point set leaves the conditional mean
undefined rather than stable — and the measurement shows the variant it
approximates, restricting to the replicates that kept support, is narrower still.

**Derive a game's seed from its conditioning rating**, making the games
independent and the old estimator nearly right. This changes what is measured
rather than what is claimed about it. The shared stream is a paired design across
the rating axis, which is what the flatness reading rests on, and unpicking it to
make a floor easier to estimate is the wrong way round.

## What Would Reopen It

The measurement is one checkpoint at one reading size on one machine, and its
32 seeds put roughly 13% of relative uncertainty on each true spread. The
figures that would matter are a rescaled draw landing away from x1.00 at four
streams and above, which is a claim about the resampling unit rather than about
the model, so any suite with three or more seeds can check it: the per-seed
distances the reading already records are the truth, and the floor beside them is
the estimate.

The correlation between two ratings within one stream is what makes the unit
necessary, and it is a property of the policy rather than of the harness — so the
checkpoint it was measured on matters, and this one barely responds to its rating
input. Across the whole 1100-to-2100 grid each quantity's mean moves by 0.15 to
0.88 of what one rating point moves between seeds, and over half the streams
played a single opening at every rating they reached.

Two ratings whose policies are nearly identical are exactly the case where one
stream produces nearly the same game at both, so the correlation measured here is
plausibly an upper bound. A model that genuinely responds to rating would hold
different policies at 1100 and 2100, and the same draws would produce more
different games. Sharpness pushes the other way — a concentrated policy sends the
same draw to the same move more often — so the net direction on a converged model
is not predictable from here and is worth re-reading rather than assuming.

What that would cost if the correlation goes away is bounded, and in the safe
direction. A draw over blocks estimates the same spread whether or not the games
inside a block move together; it reads it from fewer units, so the bound above it
is wider rather than wrong. Where it would bite is the three-stream minimum:
twelve genuinely independent games would carry eleven degrees of freedom and a
usable floor, where two streams carry one and none. That is the figure to
re-measure on a converged checkpoint, and it is the same figure the reduced
sweep's sample counts have to be set against.

## References

- `#279` — the question this answers; `#257`, where the mechanism was first
  measured
- `docs/decisions/0032-a-replayed-reading-has-no-evaluation-noise.md` — the case
  this leaves settled, and the one it left open
- `docs/decisions/0026-conservative-dispersion-bounds.md` — why a floor rests on
  a bounded spread, and what counts as a replicate
- `docs/decisions/0039-stratifying-the-ladder-redraw-costs-more-than-it-removes.md`
  — the plug-in bias and the rescaled draw that removes it
- `docs/decisions/0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`
  — how the spread this estimates becomes a floor
- `docs/evaluation.md` — "Noise Characterization", "Human-Reference Curve
  Comparisons"
