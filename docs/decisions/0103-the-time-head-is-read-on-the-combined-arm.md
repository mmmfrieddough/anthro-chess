# 0103: The Time Head Is Read On The Combined Arm

Date: 2026-10-21

## Status

Accepted. Applies
`0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md` to the
last candidates before the target run. The vehicle stays frozen and every other
candidate is still read against it.

`0105-the-target-trains-with-query-key-normalization-and-a-log-partition-penalty.md` takes the stability package's reading this arranges.

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

**The time head is read as an arm of the set `#495` adopts, against the `#495`
arm that carries exactly that set.** That is `#495`'s treatment arm, or, where
its leave-one-out pass drops a member, the arm without it. The head's arm is
then also the combined arm with the head in it, so `#595` is not run.

**The stability package is read the same way once.** Beside its range test,
`#499` runs one arm of that set with the package, against the same control, so a
package that changes quality shows at vehicle scale rather than first at `#491`.

## References

- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `#593`, `#595`, `#495`, `#499`, `#491`
