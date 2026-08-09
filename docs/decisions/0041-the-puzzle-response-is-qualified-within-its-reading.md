# 0041: The Puzzle Response Is Qualified Within Its Own Reading

Date: 2026-08-08

## Status

Accepted. Applies `0033-pairing-is-a-correctness-fix-not-a-resolution-lever.md` to
the puzzle family, takes the estimator shape
`0034-qualifying-a-rating-ladder-reading.md` settled for the ladder, takes the
rescaled stratified draw
`0039-stratifying-the-ladder-redraw-costs-more-than-it-removes.md` requires, and
rests on `0026-conservative-dispersion-bounds.md` for the bound.

## Context

The first full suite reading reported a puzzle rating response of about 0.5
percentage points across configured ratings 1000 to 2200, with a negative slope,
and it was written up as a finding about the model. It is not one; it is below
what the benchmark can resolve. Nothing in the output said so.

Half of that gap is closed. The command now prints the realized sample size and
the independent-sample difference it can detect, and the four solve-rate metrics
carry a cross-checkpoint floor from the per-puzzle contributions each reading
retains. What stayed bare is the response itself — the fitted puzzle rating at
each configured rating, the slope through them, and the pairwise ordering. Those
are the numbers the benchmark exists to produce, and they were the numbers the
finding was written from.

Two candidate qualifiers were already present and neither is the answer.

**The detectable difference is a planning bound.** It sizes the set before any
checkpoint pair exists, at the worst-case rate of one half, for two
*independently sampled* readings. A checkpoint solving two percent of lines is
nowhere near that variance, the two readings compared are not independent
samples, and the ordering is not a rate at all.

**An independent-input characterized floor would be the wrong estimator.**
Decision 0033 measures what that substitution costs: dropping the covariance two
checkpoints scored on one sample share reports a width about 1.9x wider than the
delta's real sampling variability, which turns real improvements into noise. The
puzzle family is the extreme case for that argument, since every checkpoint is
scored on the identical frozen set. So this family does not get the
`_characterize_noise` treatment the checkpoint runner gives its own pass, and
that is a decision rather than an omission.

## Decision

**A puzzle reading refits resampled puzzles and reads the spread of everything
the fit yields.** One resample is one whole response grid.

### The Response Is A Comparison Inside One Reading

Every configured rating is scored on the same puzzles. The rating response is
therefore not four readings placed side by side; it is one draw of puzzles asked
the same question at four conditionings, and the peculiarities of that draw hit
every configured rating together.

That is what a replicate has to preserve, so a replicate redraws the puzzles
once and refits every configured rating from that one draw. The slope and the
ordering are then computed from the replicate's own fitted ratings by the same
reductions the reading ran, which is what keeps the spread attached to the
number it sits beside rather than to a slightly different quantity.

It answers the question the reading is read for — whether this set resolves a
rating response at all — which is exactly the question the first full reading got
wrong.

### The Draw Is Stratified By Exact Puzzle Rating

The selection is uniform over exact integer puzzle ratings, and the retained
paired contributions already stratify by that rating to preserve it. The refit
draws the same way, for the same reason: a draw that ignored the design would
inject rating-composition variance the design excludes.

It also makes the refit nearly free. Holding each stratum at its own size holds
the scored rating composition fixed, so the expected-score sum the fit bisects is
one increasing curve every replicate shares. The curve is tabulated once and each
replicate's fit is read off it, instead of a thousand bisections per configured
rating. The whole estimate measured 0.48 s over 20,000 puzzles against a step
that runs in about 45 s, and 0.36 s over the reduced sweep's 4,000 against 15 s.

**The draw is rescaled, not plug-in.** Decision 0039 measures what a plug-in
stratified draw costs — the variance of a stratum of `n` drawn from its own
observed proportions is `(n-1)/n` of the truth — and names the correction: take
`n-1` of each stratum's units and scale the counts back up. The reduced sweep
scores two puzzles per rating, where that bias is a factor of two in variance and
the largest it can be short of the one-puzzle stratum where it takes everything.
Measured here before the correction, the fitted rating's spread grew by only
1.69x to 1.75x between 4,000 puzzles and 20,000 where `1/√n` predicts 2.24x;
after it, by 2.26x to 2.29x. A stratum of one leaves nothing to draw, so a set
built at one puzzle per rating gets no resolution and the reading says so —
`puzzles_per_rating` already refuses that as a reading dial.

### A Quantity No Redraw Moved Reports No Spread

Decision 0034 refuses a bootstrap dispersion of exactly zero for the ladder and
files the question for every other family. This is that question answered here,
and the answer is the same: a spread of zero says every delta is a finding, which
is the failure a resolution exists to prevent.

It is not a corner case for this benchmark. Pairwise ordering is a step function
that saturates at one as soon as the fit separates the configured ratings, and a
checkpoint that solves nothing at a low configured rating pins that fit at the
bottom of the search range in every replicate. Both report the spread as unknown
and the output says so.

### It Qualifies The Reading, Not A Delta

The spreads are printed with the reading and retained in its detail payload.
They are not attached to the stored measurements as noise floors.

A floor on a measurement is looked up to qualify a *delta between two
checkpoints*, and this spread is not that quantity: it is estimated from one
reading's own draw without the covariance the two checkpoints share, so a report
that used it would judge every puzzle delta by the estimator 0033 rejects. The
paired estimator is the correct one, and it cannot reach these quantities,
because it retains per-unit values and reduces them by a mean while a fitted
rating is a nonlinear functional of the whole draw.

**So a report renders these metrics' deltas as noise `unknown`, and that is
where they stay.** It is the honest verdict — a floor nobody has produced rather
than one that cannot exist — and `no_sampling_floor_reason` is deliberately not
set on them, because resampling the scored puzzles plainly can estimate their
dispersion; this reading does it.

Building the cross-checkpoint version was considered and declined rather than
deferred. It would need the report to re-run a benchmark-specific reduction per
replicate — invert the expected-score sum, refit, then slope and ordering — and
no mechanism exists for a benchmark to hand the report a reduction. Adding one
couples the results layer to benchmark code or rewrites the reporting contract,
and it has to serve the ladder too, which refits the same way and cannot pair at
all. That is a large change for rows that read `unknown` and gate nothing: the
slope is `INFORMATIONAL`, and the four solve-rate metrics beside it already
carry paired floors and are what moves first.

If it is ever wanted, the cheap form is the one to build and not the one above:
attach these spreads to the measurements with `bounded_floor` — the `sqrt(2)`
variant, since a delta between two readings is what a floor covers — and declare
`paired_sampling_floor` on the metrics, so `0035`'s degraded-floor path annotates
the substitution automatically. It errs wide, which is the direction that costs
findings rather than invents them.

## Consequences

Nothing the benchmark measures changes value, so no puzzle series ends and a
reading taken before this is comparable with one taken after. What changes is
that the second one says what it can resolve.

**The estimate errs wide as a delta qualifier and narrow as nothing.** It carries
the full sampling variability of one reading rather than of a difference, so a
response whose slope sits inside this spread is unresolved by any estimator; a
slope that clears it is not thereby established as a checkpoint delta, because it
was never asked that question.

**The reduced sweep resolves about half as finely, and now says so.** The dial
takes the set from 20,000 puzzles to 4,000, and the corrected spreads track
`1/√n` across that step, so the reduced sweep's response is the reading most at
risk of being written up as a finding.

**The retained paired contributions take the same correction.** The existing
paired estimator stratifies by exact rating with the same two-puzzle strata, so
a reduced sweep's puzzle floors carried the understatement this reading's
estimator removes, and every other stratified paired floor in the suite carried
it too. It now draws rescaled as well. Nothing recorded changes, because a
paired floor is derived at report time from machine-local contributions and
never stored.

**A stratum of one withholds the paired floor rather than reporting zero.** The
rescaled draw takes one fewer than a stratum holds, so a stratum of one leaves
nothing to draw — and the plug-in draw it replaces did not fail there, it
returned a confident zero that cleared every delta. The comparison now reports
why it has no floor. That is the same rule `0034` states for the ladder,
arriving at the paired estimator by the route `#304` predicted.

**An estimated dispersion of zero is refused where a bootstrap produces one.**
`evaluation.noise.bootstrap_floors` now omits such a metric, as it already
omitted one whose dispersion could not be estimated at all. The rule is `0034`'s,
applied where `#304` found the same defect: a resample observed that it could not
move this number, not that a wider draw could not, and a quantity identical in
every game scored reads that way at any sample size.

`#304` also asks whether the rule belongs beside `bounded_floor` rather than at
each call site. It belongs at the call site. The predicate is one comparison,
while what to *do* about a zero differs by caller — omit one metric, refuse a
whole comparison — and pushing it into the shared arithmetic would also catch
`0032`'s *stated* zero, which is the one that is correct.

**The curve family keeps its estimated zero**, which is the other half of the
per-family question `#304` asks, and it is a different answer for a reason
rather than for convenience.

A curve's floor is attached to its measurement because it is a function of that
reading's own configuration, and for generated play `0032` already establishes
that evaluation and data-sampling noise coincide — the games *are* the draw. So a
distance that no redraw moves is a statement about the play this reading
generated, not about a sample that happened to come out flat, and the reading
already exposes it directly: a `CurveComparison` carries `model_variation`
beside the distance, so a reader sees that the model side did not move without
having to infer it from the floor.

The measured consequence of the alternative decided this. Refusing there is the
same two lines and in production would never fire — a curve over hundreds of
varied games always moves — but at fixture scale it removes floors from
whichever quantities that fixture cannot exercise, and *which* quantities those
are shifts with the configuration. Measured on the generated-play fixtures:
with one game per cell, all ten; with a stub whose games differ, resignation
enabled and four games per cell, four of ten (repetition, cycle, book depth,
move diversity); adding a third seed made repertoire degenerate too, and the
depth sweep lost its floor at the plies the games no longer reached. Six tests
across `test_rollout.py` and `test_termination_benchmark.py` assert that every
quantity carries a floor, and no fixture short of production-scale play lets
them keep saying so. Narrowing them to whichever subset a given fixture happens
to move would leave
`test_a_distance_carries_the_floor_it_has_to_clear` asserting less than its name
claims, which is a worse outcome than a zero the reading already qualifies with
`model_variation`.

**The fitted rating is the quantity this most affects.** A bisection over
expected score is pinned near the bottom of its search range at a checkpoint
solving under two percent of lines, where a handful of solved lines move it by
about a hundred rating points. That is a property of the fit at the floor rather
than of the resolution, and it is why the fit's own spread is printed per
configured rating instead of only the slope's.

## References

- #173 — the gap this closes; #146, where the reading was taken
- #168 / #321 — the realized sample size and detectable difference in the output
- #356 — the cross-checkpoint version, considered and closed; #304 — the
  estimated zero, part fixed here and part still open
- `docs/decisions/0026-conservative-dispersion-bounds.md`
- `docs/decisions/0032-a-replayed-reading-has-no-evaluation-noise.md`
- `docs/decisions/0033-pairing-is-a-correctness-fix-not-a-resolution-lever.md`
- `docs/decisions/0034-qualifying-a-rating-ladder-reading.md`
- `docs/decisions/0039-stratifying-the-ladder-redraw-costs-more-than-it-removes.md`
- `docs/evaluation.md` — "Noise Characterization", "Puzzle Rating Response"
