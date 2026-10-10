# 0095: The Clock Enters Beside The Rating, And Absence Is Trained

Date: 2026-10-10

## Status

Accepted. Answers `#497`.

## Context

The normalized row already carries each game's time control and a clock reading
after every move, and the per-ply encoding already turns those into the clock
each side held before the move. Nothing reached the model: the move head was
blind to the clock, so a hurried move could only be attributed to the position
or the rating.

The vehicle trains on every speed unconditioned, so it is the control and the
question costs one arm.

## Decision

**A model can read the time context, behind `model.clock_inputs`.** Each
decision sees six values in the mover's frame: the initial clock, the
increment, both clocks before the move, and the time each player spent on their
last move. Each is read as the log of its seconds beside a presence flag, passed
through a small two-layer projection, and added to every square token beside
the rating, before the first layer. The clock describes the decision being made
rather than one square, which is the same reason the rating enters there.

**The last move times are differenced inside the model, from per-ply columns
the encoding already has.** The opponent's clock before their move is the
side-to-move clock one ply back, and the mover's is two plies back; a recorded
clock already includes the increment the move earned. Adding the times to the
encoding instead would have moved the encoding identity, which every model
identity carries, and with it the vehicle's digest and the seed floor stored
under it.

**Every absence a served decision can present is trained.** Three absences are
drawn independently on 5% of training decisions each: the whole time context,
the clock state with the control kept, and the initial clock alone. They stand
for a game served with no clock, a benchmark that declares a control but cannot
run a clock before the model can write time, and a protocol that joins a game
partway and never learns the initial clock. The corpus presents almost none of
these on its own: 99.8% of games carry clocks, and the 0.2% that do not are
untimed games of their own kind rather than games whose clock is unknown.

**Held-out move loss is committed per speed class and per bucket of the mover's
clock.** Comparable work drops or filters every decision under thirty seconds,
so the buckets split there and again at ten seconds, where the scramble starts.

## What Was Measured

The arm is the vehicle at seed 17 on the vehicle's loader seed with the setting
on, trained across both cards. It is read against the mean of the six `#488`
replicates, with deltas in their standard deviations, and against the vehicle
trained across both cards at seed 17, which matches the arm in everything but
the change. The claim and the dependency predictions were posted on `#497`
before the arm ran.

Held-out move loss on the canonical view, 14.2M decisions:

| slice | decisions | replicates | arm | delta | sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 14,161,038 | 1.4292 | 1.4019 | -0.0273 | -29 |
| ultrabullet | 161,507 | 1.9493 | 1.7310 | -0.2183 | -342 |
| bullet | 4,726,569 | 1.4738 | 1.4363 | -0.0376 | -50 |
| blitz | 7,063,441 | 1.3858 | 1.3689 | -0.0169 | -16 |
| rapid | 1,930,963 | 1.4262 | 1.4024 | -0.0238 | -23 |
| classical | 253,782 | 1.4782 | 1.4441 | -0.0341 | -38 |
| untimed | 24,776 | 1.6137 | 1.6178 | +0.0040 | +3 |
| under 10 s left | 892,960 | 1.5058 | 1.4026 | -0.1032 | -75 |
| 10 to 30 s left | 1,661,577 | 1.4071 | 1.3651 | -0.0420 | -39 |
| over 30 s left | 11,581,725 | 1.4260 | 1.4066 | -0.0194 | -22 |

Against the two-card vehicle every delta is within 0.002 of these. Top-1
accuracy rises from 0.5376 to 0.5434. Every timed slice clears both floors.

The ordering mostly matches the claim and departs from it in one place: the
fastest speeds and the shortest clocks gain most, but classical gains more than
rapid or blitz. Blitz, the commonest speed, gains least. That is what a model
regressing toward the dominant speed predicts. The scramble, the regime the
comparable work never measured, gains five times what decisions with time to
spare do.

**The model reads the rating harder, as predicted.** Against the replicates,
`dependency.rating_shuffled_degradation` rises from 0.0634 to 0.0803,
`dependency.rating_cross_conditioning_penalty` from 0.1066 to 0.1341, and
`dependency.rating_anchor_policy_divergence` from 0.384 to 0.493, all clearing
the seed floor. With the clock available, a hurried move no longer has to be
absorbed into the rating.

**Serving without a clock costs more than the clock buys.** On the dependency
view the arm reads 1.3989 with its full time context, 1.4080 with the control
alone, and 1.4332 with nothing, where the vehicle at seed 17 reads 1.4258. The control alone
keeps two thirds of the gain. With no time context at all the arm is 0.0074
worse than the vehicle. That path is an unconditioned model trained on a
twentieth of the decisions, and it is what generated play, the ladder, and the
UCI engine currently serve.

**Little played untimed moved past the seed floor.** The ladder's fitted slope,
the puzzle response, and every generated-play distance to the human reference
read inside it; four of 418 generated-play cells clear it, none of them a
distance. On human games, resignation separation rises from 0.1325 to 0.1736 and
clears the seed floor. Every adjudicated human gap that clears a floor closes
toward the human rate, for example a mate-available gap of 0.138 falling to
0.126, and the two best-rank readings that report worse are the same movement
seen from the other side: the model now ranks the best move lower more often, as
a human under the clock does.

## What This Gives Up

**The gain is not served yet.** Until a running clock reaches the model, and
until the benchmarks that play declare their control, the engine and every
self-play reading present the untimed path, which is worse than the vehicle.
`#582` feeds the UCI clock and `#583` seats the rating benchmarks at their
games' control.

**Correspondence and unknown share one representation.** The corpus's untimed
games are read as absent, and so is a game whose clock was not supplied. The
untimed slice reads 0.004 worse, inside its floor, on 24,776 decisions.

**The held-out projection does not cover the time columns.** A series sliced by
speed or clock depends on them, but the move-prediction projection digests
moves and ratings only. While the pool is frozen nothing can move them, and
adding them would have broken every held-out series and the seed floor stored
against them.

## Consequences

The setting joins the accepted set `#495` runs together before anything reaches
the canonical line. The vehicle does not change: with the setting off, a model's
identity is the one every earlier run recorded.

The vehicle's stored seed dispersion was regenerated from its six arms rescored
at this code, so the new series carry a seed floor.

## References

- `#497`, `#487`, `#488`, `#495`, `#582`, `#583`
- `0045-centisecond-clocks-from-a-closed-export.md`
- `0056-the-speed-axis-is-derived-from-the-time-control.md`
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0066-the-trunk-sees-the-rating-and-the-board-keeps-its-shape.md`
- `0072-the-clock-says-which-pool-a-rating-came-from.md`
- `docs/architecture.md` (Decision, Static, And Dynamic Metadata)
- `docs/evaluation.md` (Dependency Tests)
