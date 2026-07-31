# 0022: One Joint Rating-Ladder Fit, With Ablation In It

Date: 2026-07-31

## Status

Accepted.

## Context

The rating ladder has to answer three questions at once: where each configured
rating lands, what temperature costs in strength, and how much of that cost
rating conditioning resists. The first is a statement about one row of a grid;
the other two are statements about differences *across* rows and arms.

The obvious construction is one ladder per temperature, plus a separate ladder
for the ablated arm. It reads naturally — each row is a self-contained
tournament — and it is wrong for a reason that is easy to miss until the numbers
are in hand.

A Bradley-Terry or logistic rating fit is invariant to adding a constant to
every rating in it. That is not a defect; it is what the model says, because
only rating *differences* are identified by match results. A fit is therefore
pinned by an arbitrary anchor, and two fits performed over disjoint sets of
games are pinned by two unrelated anchors.

So a temperature response computed from independent per-temperature fits is the
difference of two numbers that were each free to be shifted by any amount. It
has the shape of a measurement and none of the content. The same applies with
more force to the attenuation, which divides one such difference by another.

## Decision

**A ladder is one round robin and one fit over the whole surface.** The unit
that competes is a *seat*: a conditioning and a temperature. Every seat plays
every other seat, and one fit places them all on one internal scale.

The rating response is then read along one axis of that surface, the temperature
response along the other, and both are differences within a single fit rather
than across two.

### Ablated Seats Are Seats

The control arm — the same model with no target rating supplied — is fielded as
additional seats in the same round robin rather than as a second ladder. The
reasoning is identical: an attenuation is a ratio of two responses, and a ratio
across two unrelated scales is arithmetic rather than a measurement.

Ablation is expressed as **absent** conditioning, which is the treatment the
dependency tests already define and the runtime already supports, rather than as
a new mode. It inherits that treatment's caveat, recorded here so it is not
rediscovered as a surprise: when the training corpus never contained
rating-absent positions, the ablated arm partly measures the model's reaction to
the input being *missing* rather than to its value.

There is one ablated seat per temperature rather than one per cell. At a fixed
temperature every ablated seat would be the same configuration, and pairing a
configuration against itself measures the color balance and nothing else.

### The Anchor Is The Only Absolute

The fit is shifted so the conditioned seats at the declared reference
temperature average to the mean of their configured ratings. Nothing else the
benchmark reports depends on that choice: ordering, slope, span, and both
temperature responses are invariant to it, and the reference row's ladder error
is measured against the configured scale by construction.

Rows at other temperatures carry the offset temperature imposes on them. That is
the temperature response showing up in the ladder error rather than a defect in
the anchoring.

### The Whole Grid Is In Every Workload

Because a seat's fitted rating is an output of the joint fit, it depends on the
population that seat was placed against. Adding a temperature, widening the
rating grid, or fielding the ablated arm changes what a seat's number means, so
the whole grid joins the declared workload of every result the ladder writes —
including a single seat's. Seed count and games per pairing stay out, since more
games estimate the same ladder more precisely.

## Consequences

The cost is quadratic. A full round robin over `R` ratings and `T` temperatures
with an ablated arm fields `T(R + 1)` seats and plays `T(R+1)(T(R+1)-1)/2`
pairings, so a grid is widened deliberately rather than by habit. A scheduled
subset would cut that, and is not adopted here: a round robin keeps the
comparison graph connected without a connectivity check, which is what the fit
needs to place every seat on one scale.

Any change to the grid ends every series the ladder writes. That is correct
under `0013-benchmark-result-comparability.md` — the numbers genuinely are not
comparable — but it means a grid should be chosen to last rather than tuned per
run.

Unfinished games are excluded rather than adjudicated. A game that reaches the
ply limit has no result and informs no pairwise comparison; scoring it as a draw
would report the ply limit as a level of play. A suite where nothing finished is
a generation failure and fails loudly, which is a different thing from a
degenerate fit and is reported differently.

## References

- `docs/decisions/0008-rating-temperature-independence.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/decisions/0020-declared-settings-scope-generated-series.md`
- `docs/evaluation.md`, "Rating Calibration"
