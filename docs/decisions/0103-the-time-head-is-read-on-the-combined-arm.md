# 0103: The Time Head Is Read On The Combined Arm

Date: 2026-10-21

## Status

Accepted. Applies
`0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md` to the
last candidates before the target run. The vehicle stays frozen and every other
candidate is still read against it.

## Context

The plan read the move-time head (`#593`) as an arm of the vehicle with clock
inputs, then read the accepted set again with the head in it (`#595`). That is
three vehicle-scale arms for one change, and the second combined arm differs from
`#495`'s treatment by the head alone.

`0065` keeps the base frozen because a base that moves cannot carry a seed floor.
For the head the floor buys little. Its carrying metrics are timing metrics,
which have no earlier reading to compare against, and the comparison that says
whether it costs move prediction is against the clock arm, which has no floor
either. `0065` also notes that a change read on top of others gives their
interaction and its own effect together, with no way to separate them. Here that
is the question: the head is adopted only into the recipe the target trains, so
its effect inside that recipe is the one that matters.

## Decision

**The time head is read as an arm of `#495`'s combined configuration, against
`#495`'s treatment arm, which differs from it by the head alone.** That arm is
also the combined arm with the head in it, so `#595` is not run.

**The stability package is read the same way once.** Beside its range test,
`#499` runs one arm of the combined configuration with the package, against
`#495`'s treatment arm, so a package that changes quality shows at vehicle scale
rather than first at `#491`.

**A new candidate is still an arm of the vehicle.** A change that has not yet
been read on its own, such as an input `#555` turns up, is read against the
vehicle and its floor, and a new set is read as one further arm as before.

## What This Gives Up

**The head's main effect on its own.** Read on the combined base, the head's
delta includes whatever it does with the set. Nothing downstream asks for the
main effect alone.

**A seed floor on the head's reading.** Neither plan had one where it mattered.

## References

- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `#593`, `#595`, `#495`, `#499`, `#491`, `#555`
