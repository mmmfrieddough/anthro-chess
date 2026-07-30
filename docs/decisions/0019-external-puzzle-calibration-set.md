# 0019: External, Uniform-Rating Puzzle Calibration Set

Date: 2026-07-29

## Status

Accepted.

## Context

The puzzle-rating benchmark needs enough positions to resolve modest checkpoint
movement, stable inputs over years, coverage across the full difficulty range,
and a runtime low enough for canonical post-training evaluation.

Sampling directly from the Lichess puzzle population would reproduce its uneven
rating density. A few wide strata prevent gross undercoverage but leave
within-stratum density bias and make a band boundary look more meaningful than
it is. Keeping a tiny selected CSV in the package makes the command convenient,
but differs from the project's established boundary for canonical evaluation
pools and encourages choosing size from repository aesthetics instead of
measurement precision.

## Decision

Size the canonical set with a conservative worst-case comparison of two
independent proportions at 95 percent confidence and 80 percent power. Use a
target that resolves roughly a one-and-a-half percentage-point overall change
and roughly a three-point change in a broad local rating region. Because every
checkpoint scores the same puzzles, retain source-game-aligned metric
contributions in the machine-local detail tier and bootstrap their paired
differences within exact-rating strata for actual checkpoint comparisons. The
independent calculation is a conservative planning bound, not the floor used to
judge those deltas.

Use a uniform exact-rating design: select the same fixed count at every integer
puzzle rating in the declared interval. Within each rating, choose by stable
hash rank after applying loose participation and popularity filters plus a
direct rating-deviation ceiling. Preserve wide bands only as presentation
drill-downs.

Keep the canonical puzzle records and raw Lichess export in the ignored data
root. Commit the pinned source configuration, deterministic build recipe,
quality and statistical design, and expected selected-content digest. Tests use
small local artifacts. Benchmark results fingerprint the realized selected
content.

Treat the continuous response curve as primary. Use the shared frozen
nearest-neighbour estimator so the analytic human reference and model readings
take the same local bandwidth, and retain band summaries for interpretation.

## Consequences

The set size follows a declared precision target rather than an arbitrary
package budget. Uniform exact-rating sampling gives every part of the range
equal designed support and makes an overall metric an average over that design,
not an estimate of the natural Lichess puzzle distribution.

The first build needs network access unless the pinned raw archive is supplied,
then both preparation and evaluation work offline from the data root. A moving
upstream puzzle URL can make an old raw snapshot hard to reacquire; the source
digest detects that failure rather than silently rebuilding a different set.
Publishing preserved raw or derived artifacts can improve availability without
changing the benchmark identity.

Canonical evaluation costs more than the original smoke-sized set, but remains
a bounded sequence of batched forward passes. A future frequent cadence may use
a deterministic uniform subsample with its own fingerprint; it must not shrink
the canonical artifact.

Source-game keys remain in the artifact for overlap provenance and resampling.
The pinned source currently contributes at most one eligible puzzle per source
game, so game-clustered and puzzle-level support are identical; validation
rejects a selected set that violates that property.

A paired sampling floor belongs to two results rather than either result alone.
Reports compute it from matching detail payloads. If those machine-local
payloads are unavailable, the paired floor is unknown; an independent-input
floor must not be substituted.

## References

- `docs/evaluation.md`
- `docs/data.md`
- `docs/decisions/0012-derived-evaluation-views.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/decisions/0014-evaluation-result-storage.md`
