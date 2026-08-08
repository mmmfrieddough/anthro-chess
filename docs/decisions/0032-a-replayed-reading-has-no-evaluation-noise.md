# 0032: A Reading That Replays Has No Evaluation Noise To Estimate

Date: 2026-08-04

## Status

Accepted. Refines `0026-conservative-dispersion-bounds.md`.
`0034-qualifying-a-rating-ladder-reading.md` composes with it.

## Context

A generated-play curve comparison estimates its own floor by bootstrapping the
games it generated, and `docs/evaluation.md` says why that is evaluation noise
rather than data-sampling noise: a rollout has no fixed data to re-measure on,
so a fresh draw of games is exactly what another seed produces.

That equivalence fails wherever the seats are greedy. Selection at temperature
zero is an argmax, so a fresh seed draws nothing and replays the reading move for
move. The evaluation noise of such a reading is not small — it is exactly zero,
and this is the case the noise taxonomy's first bullet already names when it says
deterministic offline metrics over a frozen pool have none.

What the bootstrap reports there is not zero, and it is not stable either.
Measured on `issue-216-shakedown` step 100 with the shipped
`configs/evaluation/generated-play-rollout.toml`, two runs of the same reading
differing only in how many copies of each forced game were played — 240 model
games before #216's replicate collapse, 12 after:

| quantity | conditional distance | floor at 240 games | floor at 12 games |
| --- | --- | --- | --- |
| game-length | 52.843917 (identical) | 4.0e-14 | 23.035 |
| book-depth | 2.071816 (identical) | 7.6e-16 | 0.566 |
| book-available-depth | 3.840025 (identical) | 3.5e-15 | 0.238 |
| cycle | 0.543089 (identical) | 3.3e-16 | 0.198 |
| move-diversity | 0.393770 (identical) | 7.8e-17 | 0.025 |

Every distance is bit-identical, which is the point: only the number claiming
what the reading can resolve moved, and it moved by fourteen orders of magnitude.
Step 500 of the same run shows the same pattern against distances that again do
not move. The floors at temperature 0.7 and 1.0 are identical across both runs,
so it is specific to the row whose model side cannot vary.

The mechanism is support dropout rather than any property of the metric. Each
grid point is estimated inside a radius the human reference forces, and at the
declared bandwidth that radius is narrower than the rating grid's spacing, so a
point is estimated from the games at that rating alone. Resampling 240 rows that
are twenty copies of twelve games essentially never loses a rating's support, and
the dispersion is float noise; resampling the twelve distinct games drops a
rating outright often, the conditional mean is taken over a different set of
points each time, and the spread of *that* is what the floor reported. The
pre-collapse number was right by accident at the price of 228 redundant games per
rollout; the post-collapse number is wrong and cheap.

This matters because `generated_play.*_curve_distance` and `*_pooled_distance`
are the metrics decision 0020 names as the ones that rank two checkpoints, and a
floor is what qualifies a delta between two of them. A game-length floor of 23
plies on a reading whose true run-to-run movement is zero means the
temperature-zero row resolves nothing below 23 plies that it can in fact resolve
exactly.

## Decision

**A comparison whose model side cannot vary states a floor of zero rather than
estimating one.** The caller declares it — the rollout and termination
benchmarks read it from the same predicate that decides how many replicates a
cell is worth playing, so the two cannot disagree about whether a suite's games
are forced. The floor keeps the `evaluation` kind, because zero is the correct
value of that quantity and not an admission that some other quantity was
measured instead.

**The artifact records how the floor was arrived at.** A bootstrap over
plentiful games also lands near zero, so the value alone does not distinguish a
stated floor from an estimated one, and a reader qualifying a delta needs to know
which they are holding. The comparison's stored floors carry the method and no
resample count, and each floor's source names it.

A stated floor does not depend on resampling, so it survives a model side too
thin to bootstrap. Nothing about being replayed is a sample-size question.

## Consequences

The temperature-zero row resolves every delta it can resolve. Two checkpoints
read at temperature zero against the same human reference and the same start
positions differ only in their weights, so any difference at all between their
distances is attributable to the weights, and the report says so instead of
burying it under a floor built from games neither run was going to redraw.

That holds for a delta whose two sides were both read this way. A floor travels
on the measurement rather than in the series identity, so a comparison against a
result recorded before this change still binds at whichever side's floor is
wider — which is the old bootstrap's. Nothing committed is affected, since the
store holds no generated-play result yet, but a temperature-zero series that
spans the change resolves nothing below its own oldest floor until both sides
have been re-read.

Nothing changes at nonzero temperature, which is where the shipped suite takes
its headline readings. The bootstrap remains the estimator there, and remains
checked against the per-seed spread recorded beside it.

A zero floor is not a claim that the reading generalizes. It says the delta is
real for these games; whether the same ordering holds on a different draw of
start positions is data-sampling, which decision-relevant comparisons in this
project are explicitly not qualified against, for the reason
`docs/evaluation.md` gives — both sides of the delta share the draw, so it is
common-mode.

## Alternatives Considered

**Report no floor at all**, as `SeedSpread` does below three replicates. This
understates what is known. A missing floor renders as `unknown`, which the
reporting layer defines as a floor that could exist and was not found — work
somebody could still do. Here the floor exists, is exactly known, and no amount
of further work would improve it, so `unknown` is the one verdict that is
definitely wrong.

**Keep the bootstrap and declare it `data-sampling`.** This is honest about what
the number is and does not fix what it does. A data-sampling floor attached to a
measurement still binds the delta between two checkpoints, since the reporting
layer takes the widest applicable floor of any kind, so the 23-ply figure would
go on suppressing exactly the findings it suppresses now, merely relabelled. It
also asserts the bootstrap is a sound estimate of data-sampling noise, which the
support-dropout mechanism above makes doubtful at that sample size.

**Detect determinism from the games rather than declaring it.** After #216's
replicate collapse the model side holds one copy of each forced game, so there is
no duplication left to detect, and identical games would in any case be evidence
of a coincidence rather than of a point mass. The seats' temperatures are what
make the reading deterministic, and that is knowable before a game is played.

## What Remains Open

Support dropout is not specific to temperature zero. Any model side thin enough
for a resample to lose a grid point's support often will report that fragility as
measurement noise, and nothing beside the floor says which of the two a reader is
looking at. This decision removes the case where the answer is knowable exactly;
it does not decide what a floor should claim when the resample is unstable but
the games are genuinely redrawn. Tracked separately.
