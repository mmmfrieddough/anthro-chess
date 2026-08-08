# 0016: Sampling Axes Versus Measured Distributions

Date: 2026-07-26

## Status

Accepted. `0040-games-of-marked-accounts-leave-the-corpus.md` records the one
exception taken to the preparation-filter rule below, and says what that costs.

## Context

Training selection is a dial over one corpus, and the obvious use for it is to
even out overrepresented groups so training is not dominated by the most common
examples. `docs/data.md` lists the axes that dial could act on, including
opening family, game length, and clock-data availability.

Several of those axes are also distributions the project measures. Rollout
benchmarks compare generated opening distribution, game length, results,
termination, and repetition against a human reference. Reweighting an axis
moves the very distribution its benchmark reads, so the benchmark ends up
reporting the sampling recipe rather than the model.

Measurement on the baseline corpus showed how large the stake is for openings.
Restricting to games where both players sit in the same rating band, the total
variation distance between the opening distribution at 1200-1600 and at
1800-2200 is roughly 0.28 at family level. The Sicilian rises from under 5% to
over 17% across the rating range while generic unnamed lines collapse from over
12% to under 1%. Rating-conditional opening choice is not an incidental
statistic; it is one of the most visible human-likeness properties the model
has, and it is visible before the bot plays a single good middlegame move.

## Decision

Resampling is safe on an axis the model is explicitly conditioned on. It is
unsafe on an axis the model must reproduce unconditionally.

Rating is a conditioning input, so reweighting rating changes the marginal
distribution of the input rather than the conditional behavior the benchmarks
read. That is why `docs/evaluation.md` can prescribe reweighting as a response
to a rating-calibration finding. Opening family is not a conditioning input, so
reweighting it changes the output distribution directly.

Consequently:

- **Do not resample or reweight training selection by opening family.** Learn
  the rating-conditional opening distribution from human games. Loss weighting
  by family is the same operation as resampling by family and is covered by the
  same decision.
- **Do not resample by game length.** The efficiency motive is served by
  length-bucketed batching, which groups similar lengths within a batch without
  changing how often any game is seen.
- **Filter for validity, never for balance.** Corrupt games, unsupported
  variants, and identifiable bot or engine games are validity filters. Dropping
  valid games because their result, termination, length, or repetition pattern
  is an inconvenient shape is not.
- **Before adding a resampling axis, check whether a benchmark measures it.** If
  one does and the model does not condition on it, the axis is closed until
  conditioning exists.

Compensating for a distorted training distribution by adding a conditioning
input to steer it back was considered and rejected. It makes a core behavioral
property depend on a later, more experimental feature, and it is a convoluted
way to recover something that costs nothing to preserve.

## Corpus Filters And Training Selection Are Not Equivalent

Two operations look similar and differ in an important way.

A **corpus-level filter** runs during preparation, before split assignment, so
it removes games from every split. It shifts the training data and the
evaluation reference in the same direction, and a benchmark cannot detect a
bias it shares with its own reference: the model matches the reference because
both were shaped by the same filter. More measurement does not fix this.

**Training selection** runs after the split and touches only the train split. It
shifts what the model learned while the reference stays clean, so a distortion
shows up as a mismatch rather than hiding. It is also reversible.

Prefer training selection for every editorial choice, and keep preparation
filters to validity. A corpus-level filter is a definitional statement about
what "human" means for every future benchmark, and it must be recorded as one.

Corpus-level filters also interact with pool generations. Removing games or
rejecting previously accepted games breaks the superset property that
`docs/decisions/0013-benchmark-result-comparability.md` depends on, which ends
the affected series permanently. A filter that must remove games is far cheaper
before the evaluation core is designated than after it.

## Admitting A New Data Source

A source may be mixed into the human corpus when its games are exchangeable with
existing games conditional on what the model conditions on. Any source-linked
variable that affects behavior and is not a conditioning input becomes a
confound the model silently averages over.

A different site is usually fine on this test. The project is not trying to
imitate players of one platform, so the site itself is not a behavior axis worth
preserving. Rating scale differences are handled by normalization, per
`docs/decisions/0005-lichess-default-rating-scale.md`.

Time control is not fine on this test until the policy conditions on it, because
time control genuinely changes how people open and how long games run.

Curated historical and master collections fail it on two axes that conditioning
on rating and time control does not reach. **Era** is one: opening theory moved
across a century, so a pre-war master corpus has almost no Najdorf. **Curation
selection** is the other: collections keep notable and decisive games while
online exports are exhaustive, so result and length distributions differ by
construction of the collection rather than by any variable available to
condition on.

The weighting concern for such sources is conditional rather than marginal. A
collection can be a negligible share of the whole corpus and still be a
majority of the games in the high-rating, long-time-control region, which is
exactly where the reference is thinnest and the distortion is least visible in
an aggregate.

Source admission is testable rather than assumed. Classify a candidate source's
games and compare its rating-conditional opening distribution against the
existing corpus. Agreement within the measured noise floor is evidence the
source is exchangeable on that axis; a systematic offset is evidence it is not.
The same comparison diagnoses rating normalization: if a source's 1600 has the
repertoire of the existing corpus's 1400, the mapping is wrong rather than the
players.

## Consequences

The training loop keeps the human opening distribution, which is what makes the
rating-conditional opening benchmark a measurement of the model.

The cost is accepting whatever sample-efficiency loss comes from a long tail of
rare openings. That cost is believed small: openings are the most repetitive
part of chess, rare lines transpose into common structures within a few plies,
and at corpus scale even a 0.5% family has thousands of games. Over-learning
popular theory is also correct for human-likeness, since humans over-know those
lines too.

That belief is measurable rather than assumed. Held-out move loss sliced by
opening family, plotted against each family's training frequency, shows whether
loss is still falling with frequency at the tail. Revisiting this decision
requires two signals together: a loss-versus-frequency curve still rising at
the tail, and a generated opening distribution that underrepresents the tail
relative to humans beyond the noise floor. Either alone is weak, since rare
openings are genuinely harder to predict and matching the human curve means the
behavior is already right. Even then the first response is more data rather
than reweighting.

## References

- `docs/data.md`
- `docs/evaluation.md`
- `docs/decisions/0005-lichess-default-rating-scale.md`
- `docs/decisions/0009-decision-only-rating-conditioning.md`
- `docs/decisions/0011-held-out-test-partition.md`
- `docs/decisions/0012-derived-evaluation-views.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/decisions/0015-owned-opening-book.md`
