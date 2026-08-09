# 0036: A Floor Only One Side Of A Delta Offers Does Not Qualify It

Date: 2026-08-08

## Status

Accepted. Settles the narrowing filed under "Withholding a floor protects a
reading, not a comparison" in
`0034-qualifying-a-rating-ladder-reading.md`, and states the boundary against
`0035-a-degraded-floor-is-annotated-rather-than-withheld.md`. Superseded by
`0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`, under which both readings always
supply their own dispersion.

## Context

The reporting layer takes the widest floor of each kind offered by either side
of a delta. That rule was written for two descriptions of the same quantity: a
floor attached to the baseline's measurement and one attached to the current's
answer the same question, and keeping the wider is the conservative pick.

Decision 0034 introduced the first measurements that carry no floor on purpose.
A ladder seat that scored nothing or scored everything has no finite
maximum-likelihood rating, so its number is the declared spread rather than an
estimate, and every resample of it reproduces the same bound; the reading
withholds the floor and says why. The protection that withholding buys held only
where both sides withheld. Where checkpoint A pinned such a seat and checkpoint B
placed it normally, the union took B's floor and judged the delta against it — so
a difference of a few hundred points from a number that was never an estimate
read as a finding, in exactly the transition the ladder exists to catch.

It is not the ladder's problem. Any family that withholds a floor per reading
rather than per metric has the same hole. `no_sampling_floor_reason` does not,
because it is a property of a metric in the registry and therefore applies to
both sides by construction.

## Decision

**Withhold the kind, and say which one.** A floor kind offered by one side of a
delta and not the other does not qualify that delta. The row names the withheld
kind in its note and carries the kinds as their own field in the
machine-readable record.

The rule is scoped to floors attached to a measurement, because those are the
only ones that can be one-sided. A characterized floor is a property of the
series and a paired floor is a property of the comparison; each describes both
operands however few of them attached anything, so a kind either of them supplies
is never withheld.

Dropping the floor is not enough on its own, because a delta is judged against
every kind that applies and the kinds are not interchangeable. A row whose
evaluation floor is withheld would otherwise be decided by a narrower training
floor and read as `cleared` — the same false finding through a different door.
So `cleared` is withdrawn wherever a kind was withheld and the row reads
`unknown`, while `within` stands: a delta inside any one floor is not a finding
whatever else went unmeasured. `unknown` outranks `unqualifiable` for the same
reason it is chosen at all — a withheld kind is work a re-recorded baseline
finishes, where `unqualifiable` tells a reader to stop waiting.

## Why This Is Withheld Where 0035 Annotates

Decision 0035 kept an unpaired floor standing in for a paired one, and annotated
the row rather than withholding the number. The two cases differ in what is known
about the error.

An unpaired floor is wrong by a measured factor in a known direction: it drops
the covariance the two checkpoints share and comes out about 1.9x too wide, so
`cleared` survives it intact and only `within` is weakened. Half the instrument
still works, and withholding would discard it.

A one-sided floor has no such guarantee. It describes one operand, and nothing
observed says the other is quieter — the side that offered nothing may be
noisier, and where a benchmark withholds per reading it is saying that side is
not an estimate at all. The layer's own stated bias is that a floor understating
the noise is worse than one overstating it, and this is the understating
direction. The layer already refuses a borrowed floor on exactly these grounds:
decision 0025 gives a delta whose sides ran on different machines no execution
floor rather than one machine's.

## Consequences

**A delta across the arrival of a floor estimator becomes unknown.** Where a
baseline was recorded before a benchmark produced floors and the current reading
after, the kind is one-sided and the delta is no longer qualified. That is the
honest reading and it is work somebody can do — re-record the baseline — which
is what `unknown` is for. The withheld kind is named, so the row says a floor was
declined rather than never found.

**A verdict changes rather than only a note.** Automation keys on the noise
verdict, so the honesty has to reach it; the field beside it exists to keep the
two absences apart. `unknown` alone would send a reader to characterize a series
that is already characterized.

**The ladder's withholding now protects comparisons as well as readings.** A seat
that is a bound at one checkpoint and an estimate at the other is unqualified in
the delta, which is what 0034 wanted and could not get from within the ladder.

## References

- #303 — the gap this closes; #190, #175
- `docs/decisions/0025-machine-scoped-execution-noise-floors.md`
- `docs/decisions/0033-pairing-is-a-correctness-fix-not-a-resolution-lever.md`
- `docs/decisions/0034-qualifying-a-rating-ladder-reading.md`
- `docs/decisions/0035-a-degraded-floor-is-annotated-rather-than-withheld.md`
- `docs/evaluation.md` — "Noise Characterization"
