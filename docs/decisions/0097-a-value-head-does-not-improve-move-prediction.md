# 0097: A Value Head Does Not Improve Move Prediction

Date: 2026-10-13

## Status

Accepted. Answers `#502`.

## Context

Chessformer (Maia-3) and Maia-2 both train a value head beside the policy.
Neither publishes an ablation of what it buys for human move prediction. `0066`
left auxiliary heads out of the architecture as ordinary candidates, to be read
against the vehicle, since the vehicle survives adding one.

The argument for it was representational: game outcome is a label every
training game carries for free, and a trunk made to encode who is winning gives
the policy more to work with. `#502` added two hopes: that a sense of losing
would place resignation better, and that outcome is closer to strength than
move identity is, so the rating dial might widen.

## Decision

**No value head is adopted, and the setting is not kept.** The model trains on
the move loss alone, as before.

## What Was Measured

The arm is the vehicle at seed 17 on the vehicle's loader seed, trained across
both cards, with Chessformer's head added. The final square tokens are
mean-pooled, layer-normalized, projected to 128 with a ReLU, and read out as
loss, draw and win for the mover. The target is the game's recorded result from
the mover's side at every decision, weighted at 0.1 against a move loss of 1.0,
which is Chessformer's weight. Both arms train at 0.003, which `anthro scale`
gives at this width and horizon; the head adds 17,155 parameters to 1,422,662
and changes neither. The weight and the claim were posted on `#502` before the
arm ran.

It is read on the checkpoint suite against the mean of the six `#488`
replicates, with deviations in their standard deviations, and against the
vehicle trained across both cards at seed 17, which matches it in everything
but the head.

| metric | replicates | arm | deviation | two-card vehicle |
| --- | ---: | ---: | ---: | ---: |
| `held_out.move_loss` | 1.4292 | 1.4287 | -0.5 | 1.4284 |
| `held_out.move_loss_under_1200` | 1.6181 | 1.6181 | +0.2 | 1.6180 |
| `held_out.move_loss_1200_to_1599` | 1.4763 | 1.4761 | -0.3 | 1.4756 |
| `held_out.move_loss_1600_to_1999` | 1.3870 | 1.3865 | -0.5 | 1.3863 |
| `held_out.move_loss_2000_plus` | 1.3268 | 1.3259 | -0.8 | 1.3255 |
| `held_out.top1_accuracy` | 0.5376 | 0.5376 | +0.2 | 0.5378 |
| `termination.resignation_mass_separation` | 0.1304 | 0.1308 | +0.3 | 0.1284 |

`ladder.fitted_rating_slope` reads 0.301, 0.302 and 0.302 at T=0, 0.7 and 1.0,
against replicate means of 0.298, 0.303 and 0.303: +0.2, -0.1 and -0.1 standard
deviations. Legality and every rating-dependency metric also read within 1.5
standard deviations.

**Every reading above is a null.** The carrying metric moves by half a
replicate standard deviation and reads 0.0003 worse than the matched two-card
vehicle. No rating band separates from the rest. Neither of `#502`'s secondary
hopes shows: the dial is no wider, and resignation separation sits at the
replicates' mean. It reads 0.0024 above the two-card vehicle only because that
vehicle sits 1.5 standard deviations below the mean.

Elsewhere in the suite, ultrabullet move loss reads 4.1 standard deviations
above the replicates, but the two-card vehicle departs the same way (+2.8), so
that follows the two-card training rather than the head. Against the two-card
vehicle, a few legality readings clear the bootstrap floor as worse, each inside
the replicates' spread.

**The head learned what it was asked to.** On the first 2,000 games of the
frozen test pool, it predicts the outcome at a cross-entropy of 0.765 against
0.879 for the class priors. It names the outcome correctly on 61.2% of
decisions, where always naming the commonest outcome gets 47.1%, and on 79.1%
of those in the last tenth of a game. The trunk encodes who is winning, and the
move head gains nothing from it.

`anthro eval report` withholds the seed floor for this arm, because
`training_health.gradient_norm` is taken over the whole objective and the
outcome term moves it outside the vehicle's band (0.155 against 0.145 for the
two-card vehicle; the clip rate stayed zero). That does not bear on the
conclusion: nothing here is large enough for a floor to matter.

## What This Does Not Claim

**Other weights are not ruled out.** Only 0.1 was read, and it was fixed in
advance rather than tuned. A larger weight would spend more capacity on the
outcome; nothing here suggests the move head would gain from that.

**This is a vehicle reading.** Whether a value head behaves differently at the
target's width is not measured, though `docs/scaling.md` expects most loss
changes to transfer as a constant offset, and this one has none.

**An evaluation output is still not part of the product.** Should one ever be
wanted for its own sake, it would be argued from `docs/vision.md`, not from
move prediction.

## Consequences

The implementation the arm trained with is kept on branch `502-value-head`, so
the retained run `arm-502-value-head` stays loadable. The run loads only with
that code.

`#503` read Maia-2's other auxiliary head, over move components, the same way
and found it worse (`#585`). Neither of the field's auxiliary heads is carried here.

## References

- `#502`, `#503`, `#585`, `#488`, `#495`, `#496`
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0066-the-trunk-sees-the-rating-and-the-board-keeps-its-shape.md`
- `0017-derived-termination-and-terminal-actions.md`
- `docs/research.md` (Human-Like Chess Modeling)
