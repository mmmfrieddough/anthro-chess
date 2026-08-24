# 0027: The Rating Ladder's Grid Is Settled At Its Declared Size

Date: 2026-08-02

## Status

Accepted. Extends `0022-one-joint-rating-ladder-fit.md`. Extended by
`0030-ladder-ply-limit-at-the-trained-bound.md`,
`0034-qualifying-a-rating-ladder-reading.md` settles a gap it left open, and
`0051-every-suite-step-declares-both-scales.md` withdraws the
affordability consequence below, having measured the ladder in minutes rather
than the hours argued from here;
`0079-one-declared-size-per-benchmark.md` then withdrew the reduced ladder
itself, so the sweep runs the declared grid.
`0064-the-complete-round-robin-is-the-optimal-ladder-design.md` answers the
cost question this record left to the pairing structure, measures the precision
lever the section below points at, and narrows it: openings are a sample size as
that section says, but they reach the workload fingerprint through their game-id
digest, so raising them does end a series where raising seeds does not.

## Context

Decision 0022 fixed the ladder's shape — one round robin, one joint fit — and
recorded what that shape costs: quadratic in seat count, and any grid change
ends every series the ladder writes. It deliberately did not choose the grid.
Nothing had yet run at declared size, so there was no cost to weigh against.

The shipped selection declares four ratings, three temperatures, and an ablated
arm. That is fifteen seats and 105 pairings, and at sixteen openings played from
both sides across three seeds it was 10,080 games per checkpoint when this was
written. It is 9,440 since the ten pairings between two temperature-zero seats
stopped replaying one game at every seed; every figure below was measured at the
larger count.

That size was first estimated at 42–50 hours per checkpoint, which would have
made the ladder a constraint on the project rather than a benchmark in it. The
estimate came from a per-game cost that was mostly a defect: a device
synchronization on every generated move, since removed. Measured afterwards at
declared size on two checkpoints, the ladder cost **2.01 h and 1.83 h**, about
0.7 s per game. `#146` records the reading.

A grid cut was attractive while the number was 42 hours, and it was attractive
*now* rather than later for a second reason: decision 0013 protects nothing
before the evaluation core is designated, so a cut is free until then and
permanently expensive after. The core designation is also the seam where every
ladder series ends regardless, because the opening pool is part of the declared
workload. Whatever the grid is going to be, it should be chosen once, at that
seam.

So the question is no longer what the ladder costs. It is whether the declared
grid is the right thing to keep measuring.

## Decision

**The grid stays as declared: four ratings, three temperatures, and the ablated
arm.** The values live in `configs/evaluation/rating-ladder.toml`; this record
owns why they are not going to change.

### A Cheaper Ladder Is Also A Noisier One

The cost argument for cutting seats is real and quadratic. The argument against
is easy to miss, because it runs the other way on the same structure: in a round
robin a seat's *own* sample is linear in the seat count. Each seat plays every
other, so its games are `(seats - 1) × games per pairing`, and the declared
selection puts 96 games in a pairing — 32 in one whose seats are both greedy,
for the reason the precision section below gives.

| grid | seats | pairings | games | games per seat |
| --- | --- | --- | --- | --- |
| declared: 4 ratings × 3 temperatures + ablation | 15 | 105 | 10,080 | 1,344 |
| 4 ratings × 2 temperatures + ablation | 10 | 45 | 4,320 | 864 |
| 3 ratings × 3 temperatures + ablation | 12 | 66 | 6,336 | 1,056 |

The counts are each grid's full replicate arithmetic, which is the like-for-like
comparison between grids. A temperature-zero seat realizes 1,088 of its 1,344.

Dropping a temperature buys back 57% of the cost and takes 36% of every seat's
sample with it. So a cut is not only incomparable with what came before, as 0022
established; it is also a worse measurement of every seat that survives it.

That lands on the quantities that are already smallest. The first full-size
reading put every seat within a 10 to 17 Elo span across a 900-Elo configured
range, with ordering at chance. Whether that is a genuinely flat transfer or a
sample too thin to resolve one is a distinction the ladder cannot currently
draw, because it reports no floor beside its reading. Until it can, spending
per-seat sample to buy time is spending the wrong resource: two hours is
affordable and resolution is what the reading lacks.

### Each Axis Is Load-Bearing

No axis can be cut without giving up a quantity the benchmark is declared to
report.

The **rating axis** is the primary deliverable. `docs/evaluation.md` asks for the
transfer function with enough shape to be actionable — ordering, slope, and where
the relationship degrades — and four points is already close to the fewest that
can show shape rather than a slope through two. It is also the axis that returned
the reading the project most needs to re-examine, which is an argument for
keeping it rather than for trimming it.

The **temperature axis** is at the floor 0022 set for it: three points is the
fewest that shows a temperature response as a shape instead of a line through two
points, and zero is in the grid deliberately as the degenerate sampling case.

The **ablated arm** is what makes the attenuation a measurement rather than a
ratio across two unrelated scales. It is also, on the first reading, the arm that
discriminated — removing conditioning cost about 38 Elo while moving the dial
across its whole range cost about 10 — so it is carrying the comparison that gave
the rating axis its meaning.

### Precision Has A Lever That Does Not Reopen This

Seeds, games per position, and openings are sample sizes rather than
measurement settings, and 0022 keeps them out of series identity for exactly
this reason. A ladder that needs to be more precise is made more precise by
raising them, at linear cost and without ending a series. That is the lever to
reach for, and it means this decision does not have to be revisited to buy
resolution.

The lever is inert on the ten pairings whose seats are both at temperature zero.
Two greedy seats replay one game per opening, so those pairings play one
replicate and record the seeds they played; raising the count there would enter
the same result into the joint fit again rather than more evidence.

## Consequences

A full ladder is a **scheduled reading rather than a routine one**. Two hours per
checkpoint is affordable deliberately and not by habit, which is why the ladder
stays out of the reduced sweep and belongs to the full one — its cost is a grid
rather than a sample size, so it has no honest reduction to offer.

Once the evaluation core is designated, a grid change ends a protected series.
From that point the grid is effectively permanent, and widening it — another
rating, another temperature — is a new benchmark generation rather than a tuning
step, to be taken at a seam if it is taken at all.

Two adjacent questions were left open here, and neither is a grid question.

The **ply limit** is in the declared workload and the evidence questioned it:
between 47% and 66% of games reached it, contributed no result, and consumed most
of the benchmark's cost. Whether 300 plies is the right value was a measurement
this record did not have.
`docs/decisions/0030-ladder-ply-limit-at-the-trained-bound.md` has since taken it
and settled the limit where it stands, inside the same free-until-the-core
window as the grid.

The **missing floor** was the more consequential of the two. The ladder stated
ordering, slope, and span with nothing beside them saying what it could resolve,
which is what made "the transfer is flat" and "the sample is thin"
indistinguishable from the output, and which is the reason this record declines
to trade sample for time.
`docs/decisions/0034-qualifying-a-rating-ladder-reading.md` has since taken it.
That does not reopen the grid: it makes the precision lever this record points
at legible, since raising seeds or openings now moves a number the reading
prints rather than an unstated quantity.

## References

- `docs/decisions/0022-one-joint-rating-ladder-fit.md`
- `docs/decisions/0030-ladder-ply-limit-at-the-trained-bound.md`
- `docs/decisions/0034-qualifying-a-rating-ladder-reading.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/decisions/0020-declared-settings-scope-generated-series.md`
- `docs/evaluation.md`, "The Implemented Ladder"
- `#146` — the reading this rests on
- `configs/evaluation/rating-ladder.toml`
