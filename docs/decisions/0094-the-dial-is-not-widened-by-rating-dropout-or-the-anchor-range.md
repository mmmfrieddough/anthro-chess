# 0094: The Dial Is Not Widened By Rating Dropout Or The Anchor Range

Date: 2026-10-09

## Status

Accepted. Answers `#496`.

## Context

On the ablation vehicle the rating dial is ordered perfectly and compressed
uniformly: `ladder.fitted_rating_slope` reads about 0.30 Elo of played strength
per Elo asked, at every temperature and across the whole range the ladder plays.
`#496` listed five candidates for widening it.

Two were answered elsewhere. Scale is the only lever measured to move the slope
by much, and `0090` carries its readings up to the target's width. Balancing the
training selection's rating axis bought +0.005 to +0.038 and is adopted
(`0092`).

Of the remaining three, rating dropout and the anchor range are read here as
vehicle arms. The third, an auxiliary head predicting the mover's rating, is
closed without one.

## Decision

**Neither rating dropout nor a narrower anchor range is adopted, and neither
setting is kept.**

**Classifier-free guidance is not taken either.** It would extrapolate between
the rated and unrated predictions at serving, which targets the compression
directly, but it buys span by sharpening the move distribution, costs a second
forward pass per decision, and treats the symptom. The maintainer declined it
when the training-side candidates came back null.

The dial's remaining levers are therefore the two already decided for the
target run: its width, and the rating balance.

## What Was Measured

Every arm is the vehicle at seed 17 on the vehicle's loader seed, at the vehicle's
peak rate, scored on the checkpoint suite and read against the mean of the six
`#488` replicates, which share that draw. Deviations are in replicate standard
deviations.

| `ladder.fitted_rating_slope` | T=0 | T=0.7 | T=1.0 |
| --- | ---: | ---: | ---: |
| replicates | 0.298 ± 0.017 | 0.303 ± 0.010 | 0.303 ± 0.008 |
| rating dropout 0.1 | 0.296 (-0.1) | 0.322 (+1.9) | 0.304 (+0.2) |
| rating dropout 0.3 | 0.280 (-1.0) | 0.300 (-0.4) | 0.323 (+2.4) |
| anchors 400 and 3400 | 0.302 (+0.2) | 0.317 (+1.4) | 0.293 (-1.2) |
| anchors 800 and 2600 | 0.304 (+0.4) | 0.327 (+2.5) | 0.320 (+2.1) |

`anthro eval report` reads every slope delta it applies the seed floor to as
inside it. Each arm's claim, written before it ran, was a rise at all three
temperatures that grew with the dose. No pair of arms shows that ordering.

**Rating dropout trains the unrated embedding and makes the rating matter
less.** Withholding the rating on a fraction of training decisions turns the
`unrated` anchor, which a fully rated corpus never trains, into a real marginal:
`dependency.rating_absent_degradation` falls from 0.090 to 0.025 and 0.022. But
`dependency.rating_anchor_top1_agreement` rises with the dose, 0.666 to 0.673 to
0.683, so the extreme anchors change fewer top-1 moves. At 0.3 held-out loss
leans the same way, worse in both tail bands and better in the middle, though
inside the seed floor. The premise was a
conditioned path the loss is indifferent to; the path was already in use, and
dropout weakens it.

**The anchor range is close to a reparameterization, as the code says.** 800 and
2600 reads above the replicates at all three temperatures, by roughly half of
what the rating balance bought at T=0.7 and T=1.0 and a fifth at T=0, but
inside the seed floor, and at T=1.0 the midpoint dose reads below the vehicle.

## What This Does Not Claim

**A small anchor effect is not ruled out.** 800 and 2600 is positive at every
temperature, and more seeds of it might separate it from the floor. It is not
pursued because it could not be adopted: the engine accepts settings below 800,
and a weak anchor there makes every one of them play identically.

**Temperature attenuation moving toward zero on the dropout arms is not the dial
improving.** The ablated ladder seat plays the unrated embedding, which dropout
trained, so the ratio's denominator changed meaning. Conditioned strength loses
more to temperature at dropout 0.3, not less.

**These are vehicle readings.** Whether either candidate behaves differently at
the target's width is not measured.

## Consequences

A slope still far below one at the target is a finding about its width and the
rating balance rather than a reason to reopen these candidates.

The dropout arms are retained and hold a trained unconditional path, so
reconsidering guidance would need no training. They load only with the setting
restored.

## Alternatives Considered

**A rating-prediction head.** With the rating always supplied, the head reads
its own input, so it was only meaningful combined with dropout, and closes with
it.

**Rescaling the configured rating against the measured transfer.** It cannot
reach outside the delivered range, so it would advertise a narrower span rather
than widen the dial.

## References

- `#496`, `#498`, `#54`, `#556`, `#577`
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0066-the-trunk-sees-the-rating-and-the-board-keeps-its-shape.md`
- `0090-the-size-is-not-fitted-here-and-the-rate-is-measured-at-the-target.md`
- `0092-the-rating-axis-is-balanced-by-thinning-at-four.md`
- `docs/evaluation.md` (Rating Calibration)
