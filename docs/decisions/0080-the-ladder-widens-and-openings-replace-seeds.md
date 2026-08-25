# 0080: The Ladder Widens, And Openings Replace Seeds As Its Precision Lever

Date: 2026-08-25

## Status

Accepted. Supersedes `0027-settled-rating-ladder-grid.md`, which settled the
grid at four ratings and named seeds, games per position, and openings together
as the precision lever. Both halves are replaced: the grid widens to eight
ratings, and openings alone carry precision. Extends
`0022-one-joint-rating-ladder-fit.md`, whose shape is unchanged, and
`0064-the-complete-round-robin-is-the-optimal-ladder-design.md`, whose pairing
structure is unchanged.

## Context

0027 settled the grid while the open question was whether the ladder was
affordable. It is: the round robin now plays across worker processes, and the
declared reading runs in minutes. What the walkthrough behind `#329` asked
instead was whether the reading discriminates, and two of its quantities do not.

**Ordering was saturated by the spacing, not by the model.** At four ratings
three hundred points apart, `ladder.rating_order_accuracy` and
`ladder.adjacent_rating_order_accuracy` both read exactly 1.000 at every
temperature, against floors of 0.045 and 0.090. A metric pinned at its ceiling
reports nothing about a change. Refitting the same checkpoint over six ratings
two hundred points apart returned 0.933 and 0.800, with a named inversion
between neighbours whose fitted ratings sat 3.5 points apart against a per-seat
floor near ten. The ladder had been ordering pairs no model would fail to
order.

Adjacent ordering is also quantized by the grid rather than by its sample: with
`n` configured ratings it takes values in steps of `1 / (n - 1)`, so four
ratings can only report thirds. More games narrow a floor the metric cannot
cross.

**Opening choice moves a reading about as much as sampling does.** Four fully
disjoint draws of sixteen openings, on one checkpoint at the declared grid,
moved the reported quantities by these multiples of the floor beside them:

| quantity | between-draw spread, as a multiple of the stated floor |
| --- | ---: |
| `ladder.rating_ladder_error` at the reference row | 1.94 |
| `ladder.fitted_rating_slope` at the reference row | 1.42 |
| `ladder.fitted_rating_span` at the reference row | 0.91 |
| the same three at temperature zero | 1.28, 1.46, 1.42 |
| slope at temperature 0.7 | 1.73 |

The floor is not wrong. It answers what re-running this benchmark would move,
and `0039-stratifying-the-ladder-redraw-costs-more-than-it-removes.md` keeps the
openings out of the redraw deliberately, because they are declared workload
rather than a sampled population. But the deliverable includes the transfer
function's slope as a value and not only as a delta, and a slope good to one
floor width is good to rather less than that across a different sixteen.

## Decision

**Eight ratings, evenly spaced, from 1050 to 2100.** That spans the sixth to the
ninetieth percentile of the corpus's blitz ratings, which is where it has the
volume to have taught the model anything, and it gives adjacent ordering seven
pairs instead of three.

Even spacing is the load-bearing part rather than the count. A pairwise
comparison carries the most information near an even score and almost none near
a shutout, so as the model improves the extremes will stop informing the fit;
the round robin carries the scale through the chain of adjacent pairs, and that
chain is what the spacing keeps alive.

**One seed, and openings are the precision lever.** A seed redraws the sampling
within games the openings already chose, so it narrows one of the two spreads
above and leaves the other untouched. Openings narrow both, because more
openings are also more games. At a fixed game count the opening-heavy split
therefore dominates: forty-eight openings at one seed and sixteen at three both
play ninety-six games in a pairing, over three times as many distinct positions.

It is also the only lever that reaches a pairing of two greedy seats.
`collapse_replicates` already cuts those to one seed, because a second replicate
would replay the same game, so under the old split they played thirty-two games
where every other pairing played ninety-six. They now play ninety-six too.

**The temperature axis and the ablated arm are unchanged.** 0027's argument for
both stands and nothing measured here touches it.

## Consequences

The declared grid is twenty-seven seats and 351 pairings, and 33,696 games at
forty-eight openings played from both sides. Measured on this machine at eight
workers: **296.6 s per checkpoint**.

Raising the openings is also what feeds the model. A pairing offers
`openings x colours` games, which is the ceiling on how many decisions reach one
forward pass, so the split above took the nominal batch from sixteen to
forty-eight and measured 4,185 decisions per second against 7,905. The
throughput dial and the precision dial are the same dial.

Every ladder series ends here, which 0022 already required of any grid change.
