# 0077: The Puzzle Scale Is Not The Game Scale

Date: 2026-08-23

## Status

Accepted. Withdraws the human reference `0019-external-puzzle-calibration-set.md`
gave the puzzle benchmark, and the two metrics computed from it. The rest of that
record stands: the external set, its uniform exact-rating design, and the
ordering the benchmark exists to read.

## Context

The benchmark scores an owned Lichess puzzle set at several configured game
ratings and reports how solve rate tracks them. Beside that it reported a human
reference, computed as

```python
expected_score(target_rating, puzzle.rating)
```

where `target_rating` is a configured Lichess **game** rating and `puzzle.rating`
is a Lichess **puzzle** rating. Those are separate Glicko pools. An
expected-score formula applied across them states the difference between two
scales rather than the difference between a player and a puzzle, and nothing in
the project establishes that the pools coincide. The benchmark's own selection
file already said the opposite, that the reading is ordering and slope rather
than absolute strength.

Two registered metrics measured distance from that reference,
`puzzle.greedy_curve_distance` and `puzzle.sampled_curve_distance`, both
`LOWER_IS_BETTER`. That direction is the part that misleads: it asserts a
checkpoint closer to the reference is better, when the reference is a curve
placed by an assumption no reading tested.

What it produced on a real checkpoint, at 20,000 puzzles:

| configured | reference solve rate | model line completion | fitted puzzle rating |
| ---: | ---: | ---: | ---: |
| 1000 | 12.4% | 35.3% | 1503 |
| 2200 | 69.8% | 43.5% | 1669 |

The reference says the model is three times too strong at the bottom of the grid
and far too weak at the top. Some of that is the model, whose fitted rating
spans 166 points across a 1200-point configured range. The reference cannot
separate the two, because its own placement is the untested part.

## Decision

**Drop the human reference and both distances from it.** `human_expected_score`
leaves the per-rating results and the rating bands, `greedy_curve_distance` and
`sampled_curve_distance` leave the registry, and the nearest-neighbour curve the
distances were read off leaves with them.

What remains is either scale-free or stated in puzzle points, and every part of
it answers the question the benchmark was built for:

- Solve rates, greedy and sampled, first move and full line.
- The fitted puzzle rating each configured rating produces, on the puzzle scale.
- The slope through those fits, and their pairwise ordering.

### Calibrating The Two Scales Is Not Preferred To Dropping Them

A measured offset between the pools would rescue the reference. Deriving one
needs paired player and puzzle ratings for the same accounts, which is not in
the corpus and is not in the puzzle export. Buying that to restore two metrics
whose question the slope and ordering already answer is not worth its cost, and
a project that prefers direct product choices over research questions should not
acquire a dataset to keep a diagnostic.

### The Delta Was Not The Defence Either

Both arms of a comparison meet the same misplaced reference, so a delta in curve
distance does carry information. That is an argument for a metric named after
what it measures, which this one is not: a reader takes `curve_distance` with a
`LOWER_IS_BETTER` direction for distance from human play, and the summary tier
gives them nothing that says otherwise.

## Consequences

**The registry loses two metrics**, leaving nine; the join
`0078-puzzle-training-overlap-is-measured-and-not-corrected.md` removes takes
the benchmark to eight.

**The benchmark version becomes 2**, matching what a ladder bump already
records: the detail payload's keys changed. It does not separate the series.
A quality metric's fingerprint is its identifier, its definition version, and
its data component, so readings on either side of this bump sit on one history.
That is correct here, because the surviving eight are computed exactly as
before, and it is worth stating plainly because a bump is easy to mistake for a
seam that nothing in the fingerprint would produce.

**`_continuous_curve` and `PuzzleCurvePoint` are removed.** The rating bands
were already a second, independent drill-down and remain the only one.

**`expected_score` stays.** It inverts observed solve rates into a fitted puzzle
rating, where both of its arguments are on the puzzle scale, which is the use it
was correct for all along.

## References

- `docs/decisions/0019-external-puzzle-calibration-set.md`
- `docs/decisions/0078-puzzle-training-overlap-is-measured-and-not-corrected.md`
- `#329`, the suite walkthrough that read the reference against the fits
- `docs/evaluation.md`, "Puzzle Rating Response"
