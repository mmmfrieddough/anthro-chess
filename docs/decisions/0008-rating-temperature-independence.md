# 0008: Rating And Temperature Independence

Date: 2026-07-21

## Status

Accepted as initial design direction.

## Context

Anthro Chess exposes target rating and temperature as separate user controls.
The project has held that these should stay independent, but that phrase was
carrying two different claims.

The first is a constraint on how the controls are built: temperature should not
be used as a mechanism for reaching a target rating, and rating should not
silently adjust temperature. This exists so the rating dial means one thing and
so strength is produced by the learned conditional distribution rather than by
distorting it.

The second is a claim about behavior: that changing temperature leaves played
strength roughly unchanged. This is not obviously true. Raising temperature is
expected to cost strength, because chess value is asymmetric — sampling a weak
move loses more than sampling a strong one gains, and a lost position cannot be
recovered by later play at the same rating ceiling. Temperature also changes the
shape of mistakes, spreading them across the policy instead of concentrating
them where a player of that rating would actually err.

A rating-conditioned causal model may resist part of that drift. It observes the
moves played so far, so a model that has learned how a player of a given rating
performs across a game can shift its distribution to compensate. That mechanism
is real but bounded: it can regress downward at any time, while regressing
upward is limited by what the model has learned, and histories containing
sampling noise are off-distribution relative to human play at that rating.

## Decision

Treat the two claims separately.

Keep control independence as a design constraint. Temperature must never be
tuned or derived to hit a rating target, and rating must never adjust
temperature internally.

Treat behavioral independence as a measured quantity rather than a requirement
or an abandoned goal. Report rating calibration against a declared reference
temperature so the figure means one thing, and measure the temperature response
of fitted empirical rating separately.

Measure how much rating conditioning attenuates that response by comparing
against the same measurement with rating conditioning ablated. The ablated model
is the null baseline, so the difference between the two responses is the
compensation attributable to conditioning. Report it as an attenuation rather
than as an independence claim.

## Consequences

Rating figures are only meaningful with their reference temperature attached,
and comparisons across different temperatures are not valid without it.

The project makes no promise that temperature preserves strength. If the
measured response turns out to be small, that is a finding about the model, not
a guarantee to users or a property later changes must preserve.

Strength and error-profile metrics must be read together. A temperature setting
that preserves average score while changing the distribution of mistakes is not
playing at the same rating in the sense this project cares about, and a
strength-only metric would hide that.

The ablated-baseline comparison reuses the dependency-test machinery described
in `docs/evaluation.md`, so the two share inputs and infrastructure rather than
being separate benchmark paths.
