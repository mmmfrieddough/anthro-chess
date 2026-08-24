# 0079: One Declared Size Per Benchmark

Date: 2026-08-24

## Status

Accepted. Supersedes `0051-every-suite-step-declares-both-scales.md`, which put
the ladder into the reduced sweep and named `#329` as where every reduction in
the file would be sized against what it resolves. This is that assessment, and
it withdraws the scale rather than sizing it.

## Context

The suite carried two scales. A reduced sweep was the default, a full sweep was
`--full`, and each step declared a list of sample-count overrides the reduced
scale applied. The reduced scale was justified on cost: a sweep measured in
hours is not a default anyone will run on a new checkpoint.

Two facts about it were established while walking the suite.

**A reduced reading is a prefix of a full one rather than a second
measurement.** `apply_view` sorts eligible games by `rank_key(seed, game_id)`
and takes the first `maximum_games` of them. At one seed, every smaller view is
therefore a nested prefix of every larger one over the same pool. Scoring both
scores the smaller games twice and learns nothing from the second pass that the
first did not already contain more precisely.

**The prefix lands in a series nothing accumulates.** The scored games are the
data component, so a smaller view carries a different fingerprint. A reduced
reading can never be read against the full readings that are the project's
record, which means it answers only against other reduced readings, and that
parallel history has no consumer.

Together those make the second scale a second complete sizing problem with
nothing to size against. `0051` said as much about the one step it examined: at
eight games a pairing, `#328` measured the reduced ladder's pairwise ordering at
0.667 +/- 0.442, slope 0.065 +/- 0.080, and span 72 +/- 65 Elo, every width
covering its own estimate.

## Decision

**Each benchmark declares one size, in its own selection file, and the sweep
applies no scale of its own.** `SuiteScale`, the per-step `reduced` and `scales`
fields, the validator that kept an unreachable reduction out of the schema, and
the `--full` flag are all removed.

### A Cheaper Reading Is An Override, Not A Shipped Size

`--set view.maximum_games=N` already reaches every benchmark and the sweep. That
covers the real use, which is one person wanting one quick look, and it is
honest about what it produces: a reading in the series that size names, chosen
at the moment of running rather than shipped as a second answer.

What it does not do is accumulate a history, which is correct, because a history
of coarse readings was the thing nothing consumed.

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

The justification was a sweep measured in hours. Measured after `#535` and
`#536`, the sweep is about forty-seven minutes, of which the ladder is half.
Nothing about that needs a second scale to be runnable.

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
- `docs/decisions/0012-derived-evaluation-views.md` — the nested hash-rank view
- `#328` — the widths the reduced ladder was measured at
- `#329` — the suite walkthrough this assessment belongs to
- `docs/evaluation.md` — "The Benchmark Suite"
