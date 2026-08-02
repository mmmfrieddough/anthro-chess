# 0028: The Rating Dependency Family Is Qualified By A Weighted Paired Bootstrap

Date: 2026-08-02

## Status

Accepted. Extends `0026-conservative-dispersion-bounds.md`.

## Context

Every metric in the rating-dependency family rendered `noise unknown` in `anthro
eval report`. The committed data-sampling characterizations from the first full
suite reading carried 54 floors covering `held_out`, `legality`, and
`adjudicated`, and not one of them was a `dependency.*` metric. That family is
scored inside `anthro eval run`, the one command whose records are committed, so
the gap sat beside floors rather than in a corner of the suite nobody reads.

`docs/planning/first-full-suite-reading.md` reports the dependency movement —
`absent` degradation rising from +0.315239 to +0.824292 between two checkpoints
— as having moved the way a model learning to use its conditioning should. That
reading may well be right. Nothing in the output established that it was above
resolution.

The gap was structural rather than a missed call. `_characterize_noise` in
`anthro_chess.evaluation.checkpoint` bootstraps the merge of the held-out and
adjudication per-game totals; the dependency tests run in their own path and
their results were never passed in. Behind that, every dependency metric is
registered `MetricCost.REPEATED_PASS`: a degradation is the difference between
two full scoring passes under different conditioning, which is not obviously the
per-game additive contribution `GameTotals` carries.

## Decision

**The dependency family is qualified by a weighted paired bootstrap over the
games it scored, retained in the detail tier and joined at report time.**

### Why Paired Rather Than Characterized

The family scores one frozen pool view repeatedly. Two comparable checkpoints
therefore score the *same games*, and what a report needs is the uncertainty in
their paired difference, not how far each reading would move on a fresh draw
from the population. `docs/evaluation.md` already draws that line and gives the
mechanism: a deterministic fixed-input benchmark retains aligned per-unit
contributions in the detail tier, and reporting bootstraps the delta from them.
Such a floor belongs to the comparison and cannot correctly be attached to
either reading alone, which is why it is not stored as a characterization.

The machinery existed for the puzzle family, in
`anthro_chess.evaluation.results.paired`. This is the second family to use it.

### Why Weighted

A degradation is a mean over *positions*, and the resampling unit is the *game* —
positions inside one game are far from independent, and resampling them would
report a floor several times too narrow. Those two facts do not compose without a
weight: games hold different numbers of rated positions, so a plain mean over
per-game values is not the reported metric.

The metric is a ratio of sums, so a game contributes its own mean weighted by the
positions behind it, and the bootstrap recomputes both numerator and denominator
per draw. `PairedContributions` gained an optional per-unit weight vector for
this, which is a version bump rather than a new mechanism; an absent weight
vector is the unweighted case earlier versions carried, and the puzzle family's
retention is unchanged.

The alternative was to redefine the degradations as means over games. That would
have ended the series, changed what the benchmark reports, and made the metric
harder to explain, all to avoid one vector.

### Five Of Seven

Five metrics are covered: the three corruption degradations, the anchor policy
divergence, and the anchor top-one agreement. Each is a mean over the same rated
positions, so each has a per-game share.

**Two are not, and they are not waiting on further work.** Neither has a
per-game share to retain, so neither can carry a sampling floor.
`dependency.rating_cross_conditioning_match_rate` counts rating slices rather
than games, so resampling games estimates the dispersion of a different
quantity. `dependency.rating_within_game_response` splits each rating slice at
that slice's own median prefix strength, so a game's contribution is not fixed
under resampling and there is no per-unit contribution to bootstrap.

### Saying So In The Output

`noise unknown` was one word for two situations: a floor nobody has produced
yet, and a floor that cannot exist. Only the first is worth waiting for.

A metric may now declare `no_sampling_floor_reason` in the registry, and a report
renders `unqualifiable` rather than `unknown` for it. The declaration annotates
the metric rather than redefining it, so it stays out of series identity and
needs no `definition_version` bump. `anthro eval metrics` prints the reason,
which is where the verdict points.

**The declaration is scoped to data-sampling, not to floors in general**, and the
name says so. Both reasons above are arguments about resampling the units a
reading scored. Evaluation noise and training noise are read from repeated
measurements rather than from per-unit contributions, so either would qualify
these two metrics perfectly well; a report refuses only the sampling floor and
judges the delta against any other kind it has. Suppressing every kind would
have meant that a repeat-run characterization landing later was computed,
matched to the series, and then silently discarded, with the report still
claiming no floor could exist.

## Consequences

The floor arrives with the reading, at the cost of one bootstrap over numbers the
run already computed. Nothing is added to the committed tier: the retained values
live in the machine-local detail payload beside the dependency tables, and a
comparison whose two detail payloads are not both available reports the noise as
unknown rather than substituting an independent-input estimate.

A dependency reading taken before this change has no retained contributions, so
its deltas keep reporting `unknown`. That is correct — the inputs were not kept —
and it means the family's floors begin at the next reading rather than
retroactively.

`no_sampling_floor_reason` is a general facility introduced for two specific
metrics. Other families carry quantities with the same shape — #175 raises the
same ambiguity about floors rendering as exactly `0.0000` — and adopting it there
is a separate decision about each family, not a consequence of this one.

Two adjacent gaps are untouched. The rating ladder reports ordering, slope, and
span with no resolution beside them (#190), and its fitted ratings are not
per-game additive contributions either, so it needs its own answer rather than
this one. The puzzle family's floor question (#173) is likewise separate.

## References

- #179 — the gap this closes
- #146 — where it was found; `docs/planning/first-full-suite-reading.md`
- #173, #175, #190 — the same class of reading-surface gap elsewhere
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/decisions/0026-conservative-dispersion-bounds.md`
- `docs/evaluation.md` — "Noise Characterization"
