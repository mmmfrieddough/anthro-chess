# 0087: Hyperparameter Rules Are Fitted Along The Regime Ray, And Refuse Outside It

Date: 2026-09-02

## Status

Accepted. Fixes step 5 of the order in `docs/scaling.md`.

Rests on `0076-the-vehicle-is-width-128-at-the-target-regime.md`, whose rate
sweep is the top rung of the size fit here and whose recipe every arm below
copies, and on
`0071-the-target-is-the-size-the-published-ladder-flattens-at.md`, which fixes
the target these rules have to reach and the counting convention they are
expressed in.

`0067-a-horizon-is-a-branch-not-a-restart.md` owns the schedule family the
warmup and cooldown rules sit inside, and is the reason the cooldown fraction is
carried rather than fitted.

`0086-the-training-run-is-the-binding-floor-at-vehicle-scale.md` supplies the
dispersion every near-optimal band here is read against, and this record extends
that measurement to two smaller widths.

## Context

`docs/scaling.md` requires a scale-dependent setting to be recorded as the rule
that produces it rather than as the number the rule produced, because a value is
correct at one point and silently wrong everywhere else with no goodness-of-fit
statistic to say so. Until now the project had values.

`0076` swept the rate once, at one width and one horizon, and said in as many
words that it was a rung of this fit rather than the fit. The other four
settings had never been compared against anything: warmup, the decay timescale
and the second-moment decay only became configurable in `#493`, and the batch
had been chosen for throughput rather than for learning.

The order matters. `docs/scaling.md` marks the peak learning rate hard against
six of eight columns of its coupling table, so a ladder whose rungs are unequally
tuned measures the tuning rather than the model. `docs/research.md` (Resolving
Discrepancies In Compute-Optimal Scaling) records a published exponent moving
substantially once the batch, the rate and the second-moment decay were re-tuned
at each size, and records that every intermediate fit was statistically
well-behaved while it was wrong.

## What Was Measured

Every arm copies the vehicle's recipe and changes one thing. Relaxed
determinism, `bfloat16-mixed` with `high` matmul precision, compilation at the
shipped default, the widened corpus through the shard-backed loader, no
selection filters, and the constant trunk with a square-root cooldown over the
final fifth.

**The response is the mean training loss over the final three logged
intervals**, with the logging interval set to a hundred and twentieth of each
run's horizon so that the window is the same share of every arm. That is the
statistic `0076` used, and reproducing its three published figures to four
decimal places is what lets its sweep serve as this fit's top rung. Training
loss rather than validation for the reason `0076` gives: at these horizons
against a corpus of 138.7e9 plies nothing repeats, so a training position is a
held-out position.

**Sizes are the multiples of 32 the head dimension allows.** Widths 32, 64 and
128 carry the size fit and **width 96 is held out entirely**, so the validation
run is a prediction tested rather than a curve interpolated through its own
answer.

<!-- FILL: the measurement tables, the fitted exponents, and the held-out result -->

## Decision

<!-- FILL -->

## What This Gives Up, Deliberately

<!-- FILL -->

## Consequences

<!-- FILL -->

## References

- `0076-the-vehicle-is-width-128-at-the-target-regime.md`
- `0071-the-target-is-the-size-the-published-ladder-flattens-at.md`
- `0067-a-horizon-is-a-branch-not-a-restart.md`
- `0086-the-training-run-is-the-binding-floor-at-vehicle-scale.md`
- `docs/scaling.md`: the program, the coupling table, and the rules-not-values rule
- `#489`: the issue these rules were fitted for
