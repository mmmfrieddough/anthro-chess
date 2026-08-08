# 0037: The Human Reference Is Bandwidth, Not Sample Size

Date: 2026-08-02

## Status

Accepted. Refines `0020-declared-settings-scope-generated-series.md`.

## Context

Decision 0020 separated what a generated-play reading measures from how
precisely it measures it. Seed count, games per position, and concurrency are
sample counts: more of them estimates the same distribution more finely, so they
stay provenance rather than identity. That rule is what lets the reduced sweep
exist at all — `configs/evaluation/checkpoint-suite.toml` shrinks sample counts
and nothing else, so a reduced reading is the same quantity read less precisely.

The human reference the curve comparisons are smoothed against looked like one
more sample count, and was treated as one. The reduced sweep capped it at 2,000
games in both benchmarks that read it.

It is not a sample count, and the reason is in the bandwidth. The declared
smoothing is a **neighbour count** — 1,024 — chosen so the span adapts to how
densely the corpus covers the rating range. That makes the reference size the
radius: the same 1,024 neighbours occupy whatever rating span the reference's
density puts them in. Measured on the frozen blitz test pool, against the shipped
six-point grid spaced 200 rating points apart:

| reference view | usable games | widest neighbourhood | human game-length response |
| --- | --- | --- | --- |
| 2,000 | 1,701 | ±620 | 3.55 |
| 8,000 | 6,804 | ±252 | 5.51 |
| 12,000 | 10,206 | ±180 | 6.00 |
| whole pool | 42,610 | ±45 | 6.13 |

At the reduced size a single neighbourhood spanned more than three grid steps, so
the six points were largely one estimate repeated, and the human reference's own
rating response — the movement the model's response is supposed to be read
against — read 42% below what the whole pool gives. That is not a noisier
reading of the same quantity. It is a different quantity, and nothing said so:
the reference reached the recorded envelope as provenance, so two readings taken
against 1,701 and 10,206 human games produced the same series fingerprint. #217
recorded the collapse; the shared fingerprint was found while fixing it.

The size also sat 1.66x above a hard floor. Below 1,024 usable games
`compare_curves` raises rather than degrades, and how much of a view survives the
rating-gap filter depends on the pool's rating composition rather than on
configuration, so a corpus change could have turned a shipped sweep from wide
floors into an exception.

## Decision

**The human reference is declared once, at a size the rating grid can resolve,
and is part of series identity.**

### Declared Rather Than Reduced

Both benchmarks that read it — `configs/evaluation/generated-play-rollout.toml`
and `configs/evaluation/game-termination.toml` — declare
`reference.view.maximum_games` themselves. The suite's reduced sweep does not
override it, for the same reason it does not move a rating grid or a ply limit.

That resolves both ends at once. The reduced sweep stops re-smoothing the curve,
and the full sweep stops paying for human sample the estimator cannot use: past
the point where the neighbourhoods are already disjoint, more reference narrows
nothing, and on the rollout side every reference game is replayed and classified,
so the whole pool costs almost five minutes against seventy seconds.

### One Bandwidth Per Grid Point Is The Floor

A curve over `n` evaluation points at a bandwidth of `k` neighbours needs at
least `n × k` reference games for its neighbourhoods to be disjoint at all. That
is a floor rather than a target — a reference exactly that size still overlaps
wherever the corpus is thin — and it is checked twice:

- on the configuration, so a suite plan rejects a too-small declared cap in the
  first second rather than after the generation it precedes;
- on the realized reference after the pool pass and before the first generated
  game, because a cap cannot promise how much of a view survives the rating-gap
  filter.

The rollout raises on the second check, which is what it did anyway once
`compare_curves` was reached — only hours later. The termination benchmark warns
instead: its guardrails, deficit, and held-out readings need no curve, and a
single time-control class holding too few games is a statement about that class.

### The Reference Joins The Workload

A curve reading's declared workload carries the reference view's name, the digest
over the games it selected, and the rating gap that decides which of them
survive. Those three determine the realized reference exactly, so a reading
smoothed against a different reference lands on a different series rather than on
the same line.

This is 0013's own rule rather than an exception to it — a fingerprint covers
what a measurement consumed and how it was computed, and the reference is both.
What 0020 got right and this record narrows is that *generating* more games is a
sample count. Reading more of the fixed human side is not, because it is the
smoothing.

### The Realized Bandwidth Is What Gets Reported

A reading prints the rating span its smoother actually reached at each
evaluation point, not the declared neighbour count. The count is identical at
every reference size and says nothing about a particular reading; the span read
against the grid spacing is what tells a reader whether the points are points.

## Consequences

Every generated-play curve series and every termination-mix series ends with this
change, twice over: the reference size moved and the workload gained a field.
That is deliberate and it is timed. Decision 0013 protects nothing before the
evaluation core is designated at #90, so ending these series is free now and
permanently expensive afterwards — and #90 is also the change most likely to move
the excluded fraction this record's floor is measured against.

The reduced sweep's rollout step gets about seventy seconds longer and its
termination step about a second, while a full sweep's rollout step gets about
three and a half minutes shorter. A reduced curve reading now says something
about rating response, which it could not before.

The declared size is a property of this corpus. It was chosen where the shipped
six-point grid resolves and the human curve reads within 2% of the whole pool;
a corpus with a different rating distribution would land somewhere else, and the
per-point span a reading prints is what would show it. The bandwidth selection
behind the 1,024 is unchanged and still reproducible with
`anthro eval curve-bandwidth`.

Two things are deliberately left alone. The **held-out and prefix views** are
sized by scoring and generation cost and remain ordinary sample counts. And a
quantity whose observations fall below the bandwidth while the reference as a
whole clears it still fails the rollout rather than reporting that one quantity
unavailable, the way the termination mix does — a real asymmetry between the two
benchmarks, and not one this record settles.

## References

- `docs/decisions/0020-declared-settings-scope-generated-series.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/evaluation.md`, "Human-Reference Curve Comparisons"
- `configs/evaluation/generated-play-rollout.toml`
- `configs/evaluation/game-termination.toml`
- `src/anthro_chess/evaluation/curves.py`, `reference.py`
