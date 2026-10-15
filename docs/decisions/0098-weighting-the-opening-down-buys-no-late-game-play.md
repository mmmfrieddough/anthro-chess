# 0098: Weighting The Opening Down Buys No Late-Game Play

Date: 2026-10-14

## Status

Accepted. Answers `#554`. Measures the tail cost that
`0016-sampling-axes-versus-measured-distributions.md` accepts, and the question
`docs/scaling.md` listed as open, whether state-frequency skew costs capacity.

## Context

Human archives repeat themselves most at the start: an opening position recurs
across millions of games while nearly every middlegame position appears once.
Outside work reports capacity flowing to frequent positions at the expense of
rarer decisive ones in some board games. If that happened here, late positions
would be underfit relative to their difficulty, and weighting the loss down
through the opening would move effort toward them.

`0016` closes reweighting by opening family, because it distorts a distribution
the model reproduces unconditionally. Weighting by ply index does not: every
game at a given ply carries the same weight, so the openings' shape at each ply
is untouched and only the opening's total share moves. Ply index stands in for
position repetition. This record does not measure repetition itself, and
weighting by position frequency was not tried, since it down-weights popular
lines specifically and is family reweighting by another route.

## Decision

**The move loss stays unweighted by ply.** The weighting is not kept, not even
as a disabled setting.

**The opening-frequency axis is read on every canonical checkpoint reading.**
That is the instrument this reading needed and found switched off.

## What Was Measured

### Before any arm

The per-family reading is only meaningful against training frequency, and
counting it over the whole train split costs about six processor-days. It now
counts a fixed sample of the training shards instead, and was read on the
vehicle-width horizon sweep (`h490-hz-0p003`, 100 to 1,600 positions per
parameter) and on the `#488` replicates.

**Rare openings are not data-starved.** Every frequency tier gains about the
same amount as training doubles:

| move-loss drop, 800 to 1,600 | common | uncommon | rare | very rare |
| --- | ---: | ---: | ---: | ---: |
| `held_out.move_loss_*_opening` | 0.0081 | 0.0068 | 0.0073 | 0.0083 |

`tail_move_loss_slope` reads -0.043 at 100 and -0.037 at 1,600, and the
replicates read -0.0378, -0.0371 and -0.0369. Sixteen times the training
removes about a seventh of it, so the tail's higher loss is mostly the openings
being harder rather than scarcer.

**Late phases have somewhat more left to learn.** The same doubling improves the
opening by 0.0054, the middlegame by 0.0085 and the endgame by 0.0078. Widening
from 128 to 512 at a matched number of positions shows it at about twice the
ratio: opening -0.029, middlegame -0.060, endgame -0.054. A power law fitted per
phase, with phase weights chosen to minimize total loss, gave weights of 0.82,
1.11 and 1.05. It predicted -0.0002 aggregate, against a replicate standard
deviation of 0.0009, from opening +0.002, middlegame -0.0015 and endgame
-0.0007.

### The arms

Each decision's loss was weighted by `1 - s * (1 - min(ply, 24) / 24)`, averaged
over the weights so the learning rate meant the same thing at every strength.
The weight rose from `1 - s` at the first ply to one at the twelfth full move,
where the evaluation's opening phase ends. Two arms, at `s` = 0.5 near the
fitted optimum and 0.9 past it, were each the vehicle at seed 17 on its loader
seed, trained across both cards at the 0.003 `anthro scale` gives. They were read
on the checkpoint suite against the six replicates and the two-card vehicle.
The claim was posted on `#554` before either ran.

| metric | replicates | s = 0.5 | s = 0.9 | two-card vehicle |
| --- | ---: | ---: | ---: | ---: |
| `held_out.move_loss` | 1.4292 ± 0.0009 | 1.4291 (-0.0) | 1.4292 (+0.0) | 1.4284 |
| `held_out.move_loss_opening` | 1.5143 ± 0.0002 | 1.5163 (+11.8) | 1.5189 (+27.3) | 1.5142 |
| `held_out.move_loss_middlegame` | 1.4328 ± 0.0015 | 1.4317 (-0.8) | 1.4302 (-1.8) | 1.4316 |
| `held_out.move_loss_endgame` | 1.1754 ± 0.0010 | 1.1742 (-1.1) | 1.1734 (-1.9) | 1.1747 |
| `puzzle.greedy_first_move_accuracy` | 0.5782 ± 0.0032 | 0.5861 (+2.5) | 0.5828 (+1.5) | 0.5797 |
| `puzzle.greedy_line_completion` | 0.4025 ± 0.0028 | 0.4085 (+2.1) | 0.4074 (+1.7) | 0.4041 |
| `puzzle.sampled_first_move_solve_rate` | 0.4288 ± 0.0010 | 0.4320 (+3.0) | 0.4289 (+0.1) | 0.4284 |

Deviations are in replicate standard deviations.

**The phase reading came out as the fit predicted, and that is the problem.**
The opening lost 0.0020 and 0.0046, in proportion to the dose and far outside a
spread that is unusually tight on that slice. The middlegame and endgame gained
0.0011 to 0.0026, every one inside two standard deviations, and against the
two-card vehicle, which matches the arms in everything but the weighting, the
milder arm's middlegame gain is gone. The aggregate did not move at either dose.
The weighting moves loss from late positions to early ones and buys nothing net.

**The puzzle gains are the one thing that leans the other way, and they do not
hold up.** The milder arm reads about 0.6 points better on greedy first-move
accuracy, and `anthro eval report` against the two-card vehicle finds every
puzzle delta clears benchmark noise and none clears the seed floor. The stronger
arm, which moved more loss into the late phases, reads less, so the movement
has no dose response behind it.

**None of the guards on generated play regressed.** The rollout's
`generated_play.exact_repertoire_*` distances sat inside the replicates' spread
at both doses: the opening's move loss worsened without the bot's openings
looking less human. Termination and game-length distances sat inside it too,
except one termination cell that read 3.3 standard deviations better at 0.9.

**The stronger dose costs elsewhere.** At 0.9, `ladder.fitted_rating_slope`
reads 2.6 standard deviations narrower at two of its three temperatures and
`held_out.move_loss_ultrabullet` 5.3 worse. `training_health.gradient_norm` reads
0.165 against a replicate band about 0.138, so the report withholds the seed
floor for that arm. `adjudicated.*_best_rank` moved inside two standard
deviations at both doses except mate-available at 0.9 (-2.3).

## What This Does Not Claim

**Repetition was not measured.** Ply index is a proxy for it, chosen because the
direct version is the one `0016` prohibits.

**Other ramp shapes were not read.** Only a linear ramp ending at ply 24 was
tried. The fit says the best possible phase allocation gains less than the seed
floor whatever its shape, which is why a third dose was not spent.

**This is a vehicle reading.** At the target's width the late phases may have
more capacity to use, but the width readings above say the opening benefits from
width too, at half the rate, so the trade is not expected to reverse.

## Consequences

The implementation the arms trained with is kept on branch `554-ply-weight`.
The weighting touches only the training objective, so `arm-554-ply-weight-0p5`
and `arm-554-ply-weight-0p9` load with ordinary code.

Every canonical reading now carries `held_out.move_loss_*_opening` tier series and
the tail slope in its detail, and pays about five minutes for the count.

## References

- `#554`, `#498`, `#488`, `#495`
- `0016-sampling-axes-versus-measured-distributions.md`
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0088-the-horizon-has-a-ceiling-and-it-is-counted-in-steps.md`
- `0092-the-rating-axis-is-balanced-by-thinning-at-four.md`
- `docs/evaluation.md` (Opening Family And The Rare-Opening Tail)
