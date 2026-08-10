# 0050: Every Suite Step Declares Both Scales

Date: 2026-08-10

## Status

Accepted. Withdraws a consequence of `0027-settled-rating-ladder-grid.md`,
rests on `0034-qualifying-a-rating-ladder-reading.md` for what makes a thin
reading legible, and composes with `0031-committed-benchmark-cost.md`.

## Context

The rating ladder is the only benchmark that plays seats against each other, so
it is the only strength reading the project has. It was excluded from the
reduced sweep — the default way every new checkpoint is read — which left that
sweep with no strength reading at all.

The exclusion was argued on cost, and the cost was wrong. Decision 0031 records
what happened to every figure the suite argued from: the generated-game fix took
a game from fifteen-odd seconds to well under one while the comments stood, and
the ladder's was among the worst, at roughly eighteen seconds a game against a
measured 0.469. The grid the suite had declared for a reduced ladder — one seed
and four openings, 840 games — was defended as four hours. Measured on one RTX
4090 it is **409.95 s**.

The file also carried that reduction beside `scales = ["full"]`. Neither scale
could apply it: a reduced sweep drops the step before overrides are considered,
and a full sweep never applies a reduction. So it was the one override list in
the suite that reached no schema and was therefore validated by nothing, while
telling a reader that a reduced ladder existed.

## Decision

**Every step in the shipped sweep declares both scales.** The ladder joins the
reduced sweep at the reduction already written for it, and no step now narrows
`scales` at all.

### A Reduced Sweep Without A Strength Reading Is The Worse Default

The reduced sweep is what a new checkpoint is actually read by. A default that
omits the only head-to-head measurement answers every question about a
checkpoint except the first one anybody asks.

Cost is no longer the objection. Against the 291.57 s the rest of the reduced
sweep measured, the ladder takes it to roughly twelve minutes and becomes the
majority of it. That is a large share of a small number, and the sweep is still
a thing somebody runs on a new checkpoint without planning their day around it.

### The Thin Reading Is Qualified, Not Withheld

This reduction is the one in the file whose reading is not known to resolve
anything. At eight games a pairing, `#328` measured pairwise ordering
0.667 ±0.442, slope 0.065 ±0.080, and span 72 ±65 Elo — every width covering its
own estimate. The declared grid gives each seat 1,344 games where this gives
112, so its standard error is roughly 3.5x wider on everything the fit yields.

Shipping it anyway is a bet on decision 0034 rather than against it. Since that
record the ladder states a floor beside every quantity it reports, so a reading
that cannot discriminate says so in its own output. The failure this would have
been before 0034 — three bare numbers a reader takes for a finding — is the one
0034 exists to prevent, and a floor that swallows its estimate is the instrument
working.

What is not claimed is that the reduced ladder is *useful* yet. `#328` takes the
full-grid reading that gives these widths something to be judged against, and
`#329` walks the whole suite with the maintainer at the end of the milestone,
which is where every reduction in this file is sized against what it resolves.
This record puts the ladder in front of that review rather than deciding its
size ahead of it.

### The Seat Grid Is Still Not A Dial

Nothing here reopens 0027. Seeds and openings are sample sizes and stay outside
series identity; the seat grid is the measurement, and one joint fit places
every seat on a single internal scale, so a ladder fitted on different seats
cannot be read against this one at all. The reduction moves only the two dials
0027 names as the lever.

### A Reduction No Sweep Would Apply Is Refused

`SuiteBenchmarkConfig` rejects a `reduced` list on a step that excludes the
reduced scale.

An override is validated against its target schema at the moment it is applied,
so a typo fails in the first second. One that no scale applies reaches no schema
and is checked by nothing — the suite's own dead list proved it, planning
cleanly at both scales with a nonsense key inside it. Forbidding the combination
is preferred to validating an unreachable list at plan time: the state stops
being representable rather than being checked, and the declaration a reader
lands on is true without a comment explaining that it is inert.

The same rule reaches the other route to an unread override. A step that reads
another step's output has no selection of its own, so overrides declared on it
are refused rather than discarded and then reported in the plan record as
applied.

## Consequences

**0027's affordability consequence is withdrawn.** It concluded that the ladder
"has no honest reduction to offer"; it does, and the sweep now runs it.

**The reduced sweep roughly doubles**, and the ladder is the majority of what it
costs. Each invocation commits its own wall clock under 0031, so the figure that
matters at the milestone review is measured rather than argued.

**Reduced and full ladder readings are separate series**, as at every other
step, so the thin reading cannot contaminate the full one and no ladder series
ends here.

**Sizing is deferred, not settled.** `#329` and `#330` are where this reduction
is judged against what it resolves, with every step's cost and resolution
visible at once. A finding that it resolves nothing worth twelve minutes is an
expected outcome of that review rather than a failure of this record.

## References

- `#209` — the dead list and the stale defence
- `#328` — the full-grid reading these widths are judged against
- `#329`, `#330` — the suite walkthrough and budget this reduction is sized at
- `docs/decisions/0027-settled-rating-ladder-grid.md`
- `docs/decisions/0031-committed-benchmark-cost.md`
- `docs/decisions/0034-qualifying-a-rating-ladder-reading.md`
- `docs/evaluation.md` — "The Benchmark Suite", "The Implemented Ladder"
