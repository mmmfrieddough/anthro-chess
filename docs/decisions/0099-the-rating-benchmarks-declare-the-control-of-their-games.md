# 0099: The Rating Benchmarks Declare The Control Of Their Games

Date: 2026-10-15

## Status

Accepted. Answers `#583`. Settles for the ladder what
`0072-the-clock-says-which-pool-a-rating-came-from.md` left open, and serves
the declared-control context
`0095-the-clock-enters-beside-the-rating-and-absence-is-trained.md` trained.
Refined by `0104-the-accepted-set-composes-and-the-ladder-reads-two-speeds.md`,
which adds the ladder's second speed this record left open.

## Context

A normalized rating names no pool; the time control does. A model that reads
the control recovers the pool from it, so a rating asked of it with no control
is a rating in no particular pool, and what it plays there is a blend across
speeds.

The ladder and the generated-play rollout draw their openings and their human
reference at blitz, and until now seated every model untimed. `0095` measured
what that costs a clock-reading model on human games: on the dependency view the
control alone keeps two thirds of what the full clock buys, and nothing at all is
worse than the vehicle. Nothing in self-play can run a clock before the model
writes time, so the clocks themselves cannot be supplied. The control can.

The puzzle reading has no game clock and no speed, and its ratings are their
own scale (`0077-the-puzzle-scale-is-not-the-game-scale.md`).

## Decision

**The ladder and the rollout declare one control, and every seat is shown it
with both clocks absent.** The rollout's opening walk is shown it too, since it
enumerates what the games would play. The control is part of the declared
workload, so the change starts new series rather than continuing old ones. A
declared control must band into the speed class the benchmark's pool views are
drawn at, and a configuration where the two disagree is refused.

**The control is 3+0, the commonest blitz control in the pool.** Of the pool's
102,698 games rated in the blitz pool, 39% are 3+0, 33% 5+0, 15% 3+2 and 10%
5+3. One control rather than each opening's own: the standard-start rollout arm
and the walk have no source game to take one from, and every ladder pairing
plays the same roots, so one control keeps their inputs identical.

**The puzzle reading declares no control.** A puzzle is solved without a clock,
its position may come from a game at any speed, and a puzzle rating is not a
number in any game pool. A control would name a pool the reading is not asked
about, and the blitz one in particular would assert a speed the puzzle set does
not have. A clock-reading model is asked from its untimed input there.

**A model that reads no clock plays exactly what it played untimed.** Its
readings move only in their series identity.

## What Was Measured

The six vehicle arms and both `#497` clock arms were read at the declared size
with the control in place, and the vehicle's seed dispersion was regenerated
from its arms so the new series carry a seed floor.

**The vehicle reproduces itself exactly.** A vehicle arm's ladder slopes and
rollout cells under 3+0 equal its untimed readings to the last bit; only the
series identity moved.

**The dial widens for a clock-reading model.** `ladder.fitted_rating_slope` at
T=0 / 0.7 / 1.0, against the vehicle's 0.289 / 0.316 / 0.291 at seed 17:

| arm | untimed | 3+0 declared |
| --- | --- | --- |
| clock, seed 17 | 0.289 / 0.301 / 0.314 | 0.365 / 0.371 / 0.410 |
| clock, seed 29 | 0.299 / 0.309 / 0.289 | 0.387 / 0.399 / 0.368 |

Every declared delta clears the benchmark's own floor, and three of the six
clear the seed floor as well; no untimed one does. Order stays perfect
throughout. The seed-17 reading equals the one the issue took through a patch
before this change, to the digit.

**Generated play does not move toward the human reference.** Of the 48 distance
cells per arm, none clears both floors untimed, and with the control four do on
seed 17 and none on seed 29. Those four are the result and repetition distances
at T=0.7, and all four move away from the reference. An unrecorded shakedown on
two checkpoints of the seed-17 arm read game length farther from the blitz
reference at 3+0 than untimed, at both checkpoints and both sampled
temperatures, inside the seed floor.

## What This Gives Up

**The reference is a blitz mixture and the seats play 3+0.** The rollout's human
reference spans every blitz control at the shares above. Narrowing it to 3+0
would re-smooth every curve at the declared bandwidth, which is sized on the
whole class.

**One speed.** A dial measured at blitz says nothing about the dial at bullet,
and the ladder's cost doubles with every speed it is read at. Whether to read a
second one is decided when a clock-reading model is adopted (`#495`).

## References

- `#583`, `#497`, `#495`
- `0072-the-clock-says-which-pool-a-rating-came-from.md`
- `0077-the-puzzle-scale-is-not-the-game-scale.md`
- `0080-the-ladder-widens-and-openings-replace-seeds.md`
- `0095-the-clock-enters-beside-the-rating-and-absence-is-trained.md`
- `docs/evaluation.md` (The Implemented Ladder, Puzzle Rating Response, Rollout
  Distribution Tests)
