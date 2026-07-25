# 0011: Held-Out Test Partition And Stable Split Hashing

Date: 2026-07-25

## Status

Accepted.

## Context

The normalized data pipeline originally produced a two-way `train` and
`validation` split. That was never a considered choice: it arrived with the
first reproducible sample path because the training loop needed two splits, and
the bounded baseline corpus reused it.

Validation is consumed during training. It reports validation move loss, rating
slices, and the numbers used to choose checkpoints. If checkpoint comparison and
regression reporting also draw from validation, the project reports benchmark
results on data it has been selecting against, and over many checkpoints that
becomes fitting to validation.

Evaluation also needs its inputs to stay comparable as the corpus grows. The
corpus is expected to widen over time: more months, possibly more speeds, larger
game bounds. A held-out set that silently absorbed games a later training run
consumed would be worse than useless, because the resulting metrics would look
fine.

## Decision

Split normalized games three ways into `train`, `validation`, and `test`.

Assignment stays a pure function of the split seed and the internal game id.
`test` claims the lowest hash range, `validation` the next, and everything else
is `train`.

Training must never consume `test`. The training configuration rejects a `test`
selection for both its train and validation data.

The split seed is frozen once a benchmark pool has been built from a selection.

## Consequences

Because assignment depends only on the seed and the game id, growing a corpus,
changing its filters, or raising its game bound never moves an existing game
between splits. A game held out today cannot appear in a later training
selection, so a frozen benchmark stays safe as the corpus widens.

Ordering the ranges with `test` first also keeps its membership stable when
`validation_fraction` later changes. The cost is a one-time reassignment
relative to the previous two-way split, which the preprocessing version bump
already forces.

Changing the split seed breaks the guarantee and can move a held-out test game
into training. That constraint is not enforceable from inside a single
preparation run, since the run cannot see selections made under other seeds. It
is therefore documented at the configuration boundary and recorded in pool
manifests, and the leakage check in the offline evaluation runner compares
recorded pool game ids against the training identity of the checkpoint being
evaluated.

Reserving a third partition slightly reduces training data. At the scales this
project targets, that cost is far smaller than the cost of unfalsifiable
benchmark numbers.

Comparing checkpoints on the test partition does apply mild selection pressure
to it over time. That is accepted rather than designed away: a partition the
training loop never consumes removes the large and continuous leak, and a
stricter never-touched holdout would not pay for its own ceremony at this stage.

## References

- `docs/data.md`
- `docs/evaluation.md`
- `docs/decisions/0004-source-agnostic-normalized-data.md`
- `docs/decisions/0012-derived-evaluation-views.md`
