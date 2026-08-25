# 0064: The Complete Round Robin Is The Ladder's Optimal Design

Date: 2026-08-15

## Status

Accepted. Settles the subset question `0022-one-joint-rating-ladder-fit.md`
declined on an implementation ground and `0027-settled-rating-ladder-grid.md`
did not reach, and replaces 0022's reason with a stronger one.

Answers the pressure
`0063-the-full-sweep-decides-a-change-and-the-canonical-line-is-its-byproduct.md`
put on the ladder by making the full sweep the routine cost of deciding a
change.

The qualification and sweep-membership records are read rather than changed:
nothing here alters what a ladder reading is qualified by or which sweeps run
one.

Extended by `0080-the-ladder-widens-and-openings-replace-seeds.md`, which
resizes the grid this pairing structure is complete over and acts on the
opening lever measured here, leaving the structure itself unchanged.

## Context

The ladder is the most expensive step of the suite at both scales, and its cost
is quadratic because every seat plays every other. Two records have already
declined to cut it, and neither declined what `#473` proposes.

0022 named a scheduled subset and did not adopt it, for one reason: "a round
robin keeps the comparison graph connected without a connectivity check, which
is what the fit needs to place every seat on one scale." That is a statement
about implementation cost. It says the complete graph is the cheapest way to
guarantee connectivity, not that an incomplete connected design would measure
anything worse.

0027 declined to cut **seats**, and that argument is untouched here: in a round
robin a seat's own sample is linear in the seat count, so a cheaper ladder is
also a noisier one. That record owns the arithmetic.

Cutting **pairings while holding the seat grid fixed** avoids both objections,
and the arithmetic for it is genuinely appealing. Hold the total game budget
fixed at the reduced sweep's 840 and each seat plays the same number of games
either way:

| design | pairings | games/pairing |
| --- | --- | --- |
| complete | 105 | 8 |
| incomplete | 40 | 21 |

Eight games is a score rate with a standard error near 0.18, so the reduced
ladder spends its whole budget on a complete graph whose every edge is too noisy
to say much on its own. Concentrating that budget into fewer, better-estimated
edges is the obvious next thought, and nothing in the repository had asked
whether it works.

The question is being asked now rather than at leisure because 0063 changed what
the answer is worth. 0027 settled the grid while a full ladder was a rare,
deliberately-scheduled reading; under 0063 the full sweep is what every
adopt-or-drop decision is read at, so the ladder's cost is paid per accepted
change rather than per milestone. That raises the stakes on the answer without
touching the arithmetic, which is why this record measures rather than argues.

## Decision

**The ladder keeps its complete round robin, because at a fixed game budget it
is the optimal design rather than the convenient one.** No incomplete design was
found that resolves any published quantity better than replication noise, and
several resolve some of them much worse.

### Information Adds Like Conductance, So Concentrating It Buys Nothing

The appealing arithmetic treats a pairing's precision as the thing being bought.
It is not. What the fit spends its budget on is the precision of *rating
differences*, and those do not come from single pairings.

Bradley-Terry Fisher information is a weighted graph Laplacian: a pairing
contributes `games x p(1-p)` along the contrast between its two seats. The
variance of any rating difference is then the effective resistance between those
two seats in the network where each pairing is a conductance. Two consequences
follow, and both cut against concentration.

A pair with no direct pairing is still measured, through every path that joins
them. And a pair with a direct pairing is measured mostly by the *other*
pairings — in a complete graph on 15 seats the thirteen two-step paths carry 87%
of the contrast between two seats and the direct edge carries 13%. Moving games
off an edge does not delete that edge's information so much as redistribute it,
and moving games onto one does not concentrate information so much as strand it.

The total conductance is fixed by the budget. Foster's theorem — that the
conductance-weighted effective resistances of a graph's edges sum to
`seats - 1` — then says a design's own edges **average** `seats - 1` divided by
that total conductance, whatever the design is. A sparse design's own edges
therefore average exactly the variance the complete graph gives all of its
edges. What differs is only what happens to the pairs the design left out, and
those get worse.

### The Complete Graph Is Exactly The Optimum

Fix the total conductance `C`. The Laplacian's trace is then `2C`, so its
non-zero eigenvalues sum to `2C`, and the average variance over all pairwise
contrasts is proportional to the sum of their reciprocals. Subject to a fixed
sum, a sum of reciprocals is smallest when the eigenvalues are all equal — and
the complete graph with equal games per pairing is the design that makes them
equal.

So the round robin is not an approximation of the best design available at this
budget. It **is** it, and it is optimal for the maximum contrast variance and
the average one at the same time.

The exactness has one boundary worth stating. It assumes every pairing carries
the same Bernoulli weight, which fails once seats are far enough apart that
`p(1-p)` differs across pairings; there the optimum tilts toward the closer
pairs. The measurement below fields that tilt as an explicit design and it loses
anyway.

### The Measurement

Simulated from known seat strengths at the reduced sweep's budget of 840 played
games, run through this repository's own `fit_ratings` and reading reductions,
600 replications per design. The truth is the flat transfer `#146` and `#177`
actually measured — a 900-Elo configured range compressing into the teens, with
the ablated arm 38 Elo down — since that is the state the reading has to
discriminate in. Games reach the ply limit at the 439-in-840 rate `#328`
measured and are dropped unscored, so each design scores roughly 400 of its 840,
as the shipped reading does. Each column is the spread of one published quantity
across replications; the worst-contrast column is the largest variance the design
gives any seat pair, in the units of the paragraphs above.

| design | pairings | games/pr | worst contrast | error sd | slope sd | span sd | response sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **complete** | **105** | **8** | **0.135** | **19.2** | **0.056** | **35.8** | **26.5** |
| circulant, degree 8 | 60 | 14 | 0.160 | 19.7 | 0.060 | 36.8 | 34.2 |
| circulant, degree 6 | 45 | 18–19 | 0.198 | 20.5 | 0.060 | 36.1 | 40.4 |
| rows complete, temperature rungs | 40 | 21 | 0.271 | 20.4 | 0.059 | 34.6 | 57.6 |
| random connected | 40 | 21 | 0.307 | 29.1 | 0.077 | 46.5 | 29.5 |
| closest-gap, oracle | 40 | 21 | 1.042 | 25.5 | 0.084 | 42.3 | 30.5 |
| spanning tree | 14 | 60 | 0.976 | 35.3 | 0.120 | 65.0 | 47.7 |
| complete, three seeds | 105 | 8–24 | 0.062 | 11.5 | 0.034 | 20.4 | 15.0 |

The complete design is the only row whose worst contrast equals its best; every
other design has a seat pair it resolves several times worse, and the ladder
reports quantities over every pair. No incomplete design improves any column by
more than replication noise — the one that comes closest reads 34.6 against 35.8
on span, inside the sampling error of a spread estimated from 600 draws — while
three designs are half again as wide or worse somewhere. That is Foster's
theorem showing up as data: concentration does not buy precision on the rating
row, and it costs elsewhere.

The spanning tree — the extreme of concentration, 60 games in each of 14
pairings — failed to converge in 476 of its 600 fits. The design that
concentrates most is the one that breaks the estimator.

A cross-check on the arithmetic: the complete design's measured contrast
variance of 0.1337 matches the closed form `(seats - 1) / C` at 0.1333.

Repeated against a checkpoint whose transfer works — a 300-Elo fitted span, 100
Elo of temperature cost — every ordering above holds and the oracle design is
the worst of all, at 115 Elo of span spread against the complete design's 53.

### A Ladder Reads Two Axes, So No Sparse Design Is Dense Where It Is Read

The pattern in the response column is the practical form of the theorem, and it
is what makes this benchmark a bad candidate for a subset even where one is
conditioned well.

The issue anticipated that adjacent-rating pairs "carry the ordering metric this
benchmark is most read for and probably have to stay dense." They do — and so do
the same-rating cross-temperature pairs, which carry the temperature response,
and the conditioned-to-ablated pairs, which carry the attenuation. A ladder is
one fit over a two-dimensional surface, and every design that is dense along one
axis spends its budget on pairings that say nothing about the other. The
row-complete design above is the clearest case: it resolves the rating row as
well as anything on the table and is the worst design tested for the temperature
response, because its 30 within-row pairings carry no temperature contrast at
all.

The complete graph is the only design that is dense on both axes at once. That
is not a coincidence of this grid; it is what "optimal for the maximum contrast"
means when the quantities read span the whole surface.

### Nothing New Joins The Declared Workload

The issue asked whether a pairing set would have to be declared, and warned that
getting it wrong would either fracture the ladder's series or let two
incomparable designs share a line. The structure does not change, so no new
dimension is declared. 0022's rule stands unaltered: the grid is identity, and
seed count and games per pairing are not.

The second half of that warning is worth recording as more than hypothetical.
The workload's pairing field is a constant naming the only design rather than
anything derived from the pairings actually played, so it would not have caught a
subset joining the existing series — the failure the issue named. Under this
decision it is a correct name; it was never the guard.

The connectivity guarantee 0022 named as the blocker likewise stays where it is.
It is discharged by construction rather than checked, which is the cheapest place
for it to live, and there is now a second reason not to move it.

### The Thin Reduced Reading Has A Lever, And It Is Not This One

`#473`'s premise is right that eight games a pairing resolves little. The remedy
is the one 0027 already names, and the last row of the table sizes it: three
seeds at the complete design cuts every published width by about 40%.

That row plays 2,360 games rather than 2,520, because `collapse_replicates`
holds the ten pairings between two greedy seats at one replicate however many
seeds are configured. So the lever is a little weaker than a plain tripling, and
weakest on the row the temperature response is read from. The table's figures are
measured with that collapse in place rather than assumed away.

Seeds and openings are not equivalent for this, and only one of them is free:
three seeds buy 24 games a pairing at no cost to the series, and eleven openings
buy 22 and a permanent seam, because the opening selection reaches the workload
fingerprint through its game-id digest and the seed count does not. `0039`
measured the between-opening spread at nothing, which is what makes the two close
enough on precision for that difference to decide it.

**Sizing that lever is not settled here**, and `0051` says where it is. What this
record settles is that the choice there is between sample sizes, with no
structural option worth putting beside them.

## Consequences

**Nothing changes in what the ladder plays, so no series ends.** The reading
`#328` takes is unaffected, and the sequencing concern that motivated deciding
before it — that a later pairing change would end or force a retake of that
series — is discharged by the answer being no.

**The reduced ladder's thinness is now a sizing question alone.** It has one
lever, that lever is linear, and `#329` and `#330` are where it is pulled or
not. A future session reading "eight games a pairing is too few" should reach
for seeds rather than for the pairing structure.

**Reallocating games across a complete graph is a different lever, and is not
adopted.** Giving near-even pairings more games than lopsided ones while keeping
every pairing is the one form of concentration the theorem above does not rule
out, and it is untested. It is not filed as work: the ladder's measured state is
one where every seat is within a 10 to 17 Elo span, so every pairing is already
near-even and there is nothing to tilt. A checkpoint whose seats separate enough
for the weights to differ would be the first evidence that the question is real.

**This question is closed rather than deferred.** Reopening it needs a defect in
the conductance argument or a measurement that beats the table above, not an
appeal to the budget arithmetic.

## References

- `#473` — this decision, and the design comparison the table reports
- `#146`, `#177` — the flat transfer the simulated truth is taken from
- `#328` — the definitive reading this precedes
- `#329`, `#330` — where the reduced ladder's sample size is settled
- `docs/decisions/0022-one-joint-rating-ladder-fit.md`
- `docs/decisions/0027-settled-rating-ladder-grid.md`
- `docs/decisions/0034-qualifying-a-rating-ladder-reading.md`
- `docs/decisions/0039-stratifying-the-ladder-redraw-costs-more-than-it-removes.md`
- `docs/decisions/0051-every-suite-step-declares-both-scales.md`
- `docs/decisions/0063-the-full-sweep-decides-a-change-and-the-canonical-line-is-its-byproduct.md`
- `docs/evaluation.md` — "The Implemented Ladder"
- `configs/evaluation/rating-ladder.toml`
