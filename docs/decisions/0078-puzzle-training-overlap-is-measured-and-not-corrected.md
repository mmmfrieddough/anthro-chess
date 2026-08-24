# 0078: Puzzle Training Overlap Is Measured And Not Corrected

Date: 2026-08-23

## Status

Accepted. Removes the provenance join
`0019-external-puzzle-calibration-set.md` added and the metric it fed, and
settles the contamination question that join was standing in for. Composes with
`0044-the-puzzle-selection-is-vendored-not-refetched.md`, whose no-network,
no-archive build is what a corpus-dependent selection would have cost.

## Context

Lichess puzzles are cut from Lichess games, and the corpus is Lichess games, so
some puzzles come from games a checkpoint trained on. The benchmark measured
that on every invocation: for each scored puzzle it joined the source game key
against every non-test row of the corpus and reported
`puzzle.training_overlap_rate`.

Three things were wrong with it.

**It was the whole benchmark.** The step measured 3572.5 s against a full sweep,
of which everything except the join was 53.2 s. The join walked 41,763 shards
and about two billion rows to produce one number.

**The number was not about the checkpoint.** It is a function of the puzzle
selection and the corpus generation, both pinned in configuration. Every
checkpoint recomputed the identical 0.2022, and nothing read it.

**Nothing acted on it.** The prior reasoning was that one exposure among
millions does not produce recall, which is an assertion the reading never
tested. Either the contamination matters, in which case reporting it is not the
response, or it does not, in which case the reading is decoration.

## Decision

**Measure the contamination once, and act on the answer.** The measurement is
below. It shows no effect, so the join, the metric, and the corpus path in the
benchmark's selection are all removed, and no filter is applied.

### What The Measurement Found

Of 20,000 scored puzzles, 4,044 came from games in the corpus outside its test
partition. Comparing solve rates on those against the rest, raw, looks like
contamination. It is not: the overlapping puzzles are easier, mean rating 1755
against 1810.8. Holding difficulty fixed by comparing within each exact puzzle
rating, over 13,806 usable strata:

| metric | raw | within rating | 95% interval |
| --- | ---: | ---: | ---: |
| greedy first move | -0.00120 | -0.01116 | [-0.02764, +0.00549] |
| greedy line | +0.03800 | +0.01106 | [-0.00225, +0.02423] |
| sampled first move | -0.00672 | -0.00955 | [-0.01951, +0.00034] |
| sampled line | +0.01854 | +0.00588 | [-0.00073, +0.01247] |

Every interval covers zero, and the signs disagree: both first-move metrics go
negative. Recall would move all four together, and would move the first move of
a line most of all, since that is the move a memorized game supplies first.
Inconsistent signs across that many strata is noise.

A second argument holds independently of the measurement. Both arms of a
comparison train on the same corpus, so a constant overlap bias is identical on
each side and cancels in the delta. It could only reach an absolute claim about
puzzle solving, or a change that alters how much a model memorizes.

### Filtering Would Have Cost The Portable Build

Excluding overlapping puzzles at selection time makes the vendored selection a
function of a corpus generation. Decision 0044 exists so the puzzle artifact
builds with no network and no archive, on a machine holding neither; a selection
filtered against a 412 GB prepared corpus cannot be rebuilt on that machine at
all. Detecting staleness when the corpus is re-cut is further work again, and
all of it buys a correction the measurement says is not needed.

### Re-measuring Is Cheap Now, And Is Not Scheduled

The join read every row into Python dictionaries, which is what made it take an
hour. Vectorized over Arrow and spread across cores it reproduces the same
0.2022 in 22.1 seconds. That is what made settling this affordable, and it is
what makes re-settling it affordable if the corpus widens materially. It is
deliberately not a standing obligation on anybody: no reading depends on the
answer, so nothing has to remember to refresh it.

## Consequences

**The puzzle step loses its dominant cost.** What is left is bounded by the
puzzles scored rather than by the corpus.

**`puzzle.training_overlap_rate` leaves the registry**, and `training_normalized`
leaves the benchmark's selection schema and its artifact-root rebasing.

**The benchmark no longer reads the corpus at all.** It needs the puzzle
artifact and a checkpoint, which is the whole of its inputs.

**A widened corpus raises the overlap silently.** Accepted: the measurement
found no effect to grow, and a reading nobody consults would not have surfaced
one either.

## References

- `docs/decisions/0019-external-puzzle-calibration-set.md`
- `docs/decisions/0044-the-puzzle-selection-is-vendored-not-refetched.md`
- `docs/decisions/0077-the-puzzle-scale-is-not-the-game-scale.md`
- `#329` — the suite walkthrough the measurement was taken under
- `docs/evaluation.md` — "Puzzle Rating Response"
