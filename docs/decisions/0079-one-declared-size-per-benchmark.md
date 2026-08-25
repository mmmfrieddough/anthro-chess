# 0079: One Declared Size Per Benchmark

Date: 2026-08-24

## Status

Accepted. Supersedes `0051-every-suite-step-declares-both-scales.md`, which put
the ladder into the reduced sweep and named `#329` as where every reduction in
the file would be sized against what it resolves. This is that assessment, and
it withdraws the scale rather than sizing it.

Refines `0063-the-full-sweep-decides-a-change-and-the-canonical-line-is-its-byproduct.md`
by removing the middle term of its three-way separation; what a claim is decided
on is unchanged. Refines
`0027-settled-rating-ladder-grid.md` in one place only: the reduced ladder its
status describes no longer exists, so the sweep runs the declared grid.

## Context

The suite carried two scales. A reduced sweep was the default, a full sweep was
`--full`, and each step declared a list of sample-count overrides the reduced
scale applied. The reduced scale was justified on cost: a sweep measured in
hours is not a default anyone will run on a new checkpoint.

Two facts about it were established while walking the suite.

**A smaller reading lands in a series nothing accumulates.** The scored units
are the data component, so a smaller reading carries a different fingerprint. It
can never be read against the readings at the declared size that are the
project's record, which means it answers only against other reduced readings,
and that parallel history has no consumer. This is the argument that reaches
every dial the scale moved: view sizes, seed lists, games per position, and
resample counts alike.

**Where the dial is a view size, the smaller reading is also a strict prefix.**
`apply_view` sorts eligible games by `rank_key(seed, game_id)` and takes the
first `maximum_games`, so at one seed every smaller view is nested inside every
larger one over the same pool. Scoring both scores the smaller games twice and
learns nothing the first pass did not already hold more precisely. That is the
sharpest form of the point but not the general one, since a seed list or a
resample count is not a prefix of anything.

Together those make the second scale a second complete sizing problem with
nothing to size against.

Two of the withdrawn reductions were not sample counts at all. `inference` had
`latency.sweep_plies=[0, 40, 80]` against a declared ten points and
`throughput.sweep_batch_sizes=[1, 8]` against six, which are grids: the file's
own rule said a reduction never moves one, because a sweep that changed a grid
reports a different quantity rather than the same one less precisely. That the
shipped lists had drifted past their own rule is part of what a second set of
numbers with no consumer costs. `0051` said as much about the one step it examined: at
eight games a pairing, `#328` measured the reduced ladder's pairwise ordering at
0.667 +/- 0.442, slope 0.065 +/- 0.080, and span 72 +/- 65 Elo, every width
covering its own estimate.

## Decision

**Each benchmark declares one size, in its own selection file, and the sweep
applies no scale of its own.** `SuiteScale`, the per-step `reduced` and `scales`
fields, the validator that kept an unreachable reduction out of the schema, and
the `--full` flag are all removed.

### A Cheaper Reading Is An Override, Not A Shipped Size

Every benchmark command takes `--set` against its own schema, so a single
benchmark is read smaller directly:

```console
uv run anthro eval run --config <selection> --set view.maximum_games=2000
```

Through the sweep it goes per step, because `--set` there resolves against
`SuiteConfig` rather than against the selection a step names:

```console
uv run anthro eval suite --config <suite> \
  --set 'benchmarks.run.overrides=["view.maximum_games=2000"]'
```

That is deliberately not one flag for the whole sweep. The dial is not the same
key at every step, and there is no size that means the same thing across all
nine of them: `puzzles` counts puzzles per exact rating, `ladder` shrinks
openings under a seat grid that is the measurement, and `inference` has no view
at all. A single flag would have to pick one meaning per step, which is what the
withdrawn `reduced` lists were.

What neither form does is accumulate a history, which is correct, because a
history of coarse readings was the thing nothing consumed.

### A Monitoring Series Would Need The Opposite Property

The one real case for a second fixed size is a cadence whose history is itself
the reading. That needs the size frozen, because changing it breaks the series.
A reduced scale is explicitly a budget dial, retuned whenever the budget is
reassessed. A frozen monitoring size and a budget dial cannot be the same
number, so the reduced scale could not have become one without ceasing to be
what it was for.

Preview cadences during training already subsample in their own view, which is
where that need is assigned.

### The Cost Argument Had Reversed

The justification was a sweep measured in hours. Composing the last full sweep
with what `#536` measured for the puzzle step it replaced puts the sweep at
about forty-seven minutes, of which the ladder is half. That figure is a
derivation from two readings rather than one sweep timed end to end, and it is
nowhere near the hours a second scale was justified against.

## Consequences

**The suite runs one way**, so `--plan` and a sweep no longer disagree about
which sizes they describe, and a resumed sweep's ledger no longer has to refuse
a different scale because there is only one.

**Every step's size is now a decision with one owner**, its own selection file,
rather than a pair of numbers split between that file and the suite.

**The ladder's reduction is withdrawn rather than resized.** `0051` predicted
this outcome for it. What the ladder should cost at its declared grid is a
separate question and `#329` is still where it is asked.

**Readings taken at a reduced scale stay where they are.** They were always
their own series, so nothing merges and nothing is invalidated; they simply have
no successors.

## References

- `docs/decisions/0051-every-suite-step-declares-both-scales.md`
- `docs/decisions/0063-the-full-sweep-decides-a-change-and-the-canonical-line-is-its-byproduct.md`
- `docs/decisions/0027-settled-rating-ladder-grid.md`
- `docs/decisions/0012-derived-evaluation-views.md`, the nested hash-rank view
- `#328`, the widths the reduced ladder was measured at
- `#329`, the suite walkthrough this assessment belongs to
- `docs/evaluation.md`, "The Benchmark Suite"
