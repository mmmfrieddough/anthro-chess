# 0068: A Pool Re-Cut Breaks Benchmark History, And That Is Accepted

Date: 2026-08-16

## Status

Accepted. Supersedes the "Pool Generations, Core, And Current" section of
`0013-benchmark-result-comparability.md`, and with it the per-axis sizing
argument `0052-a-bounded-pool-is-a-fixed-admission-fraction.md` attaches to
designation. Everything else in both records stands.

## Context

`0013` answered "is the model better than it was a year ago" with two views of
the pool. One generation would be designated the **core** and scored by every
later reading, giving a line that survives a pool re-cut; the whole pool would
be scored alongside it as **current**, giving precision and a check on
overfitting the core. The machinery was specified in `#90` and half built.

Two facts about this project, neither visible when `0013` was written, decide
against it.

**The pool is unlikely to be re-cut.** The corpus is built: 42 archives,
2.09 billion games, spanning the axes the project intends to measure. The pool
cut from it is a fixed fraction of the held-out split, so it moves only when the
corpus does, and no further acquisition is planned. A mechanism whose entire
value is realized at the second cut has no value if there is no second cut.

**Until then the two views are the same set of games.** At the cut, core and
current select identically, so they produce one number twice, land on one
series, and every reading carries two envelopes per result kind that a reader
must tell apart. That cost is certain and immediate; the benefit is contingent
on an event that may never arrive.

The alternative `0013` also named, **anchor checkpoints** re-scored on both
sides of a seam, is separately unavailable: `#90` established that no
corpus-scale checkpoint still loads after the model identity moved, and there is
no migration path.

## Decision

**There is no evaluation core.** A benchmark scores the pool it is pinned to and
reports one number. Nothing is designated, nothing is carried forward, and no
reading is taken twice.

**A pool re-cut breaks the affected series, and the break is accepted rather
than bridged.** Fingerprints already detect it and reports already render a seam
as a seam, so a re-cut re-baselines. What is given up is the ability to compare a
reading taken before a re-cut against one taken after. The project accepts
losing that in exchange for one number per reading.

**Containment still binds.** A generation names its predecessor and is refused if
it drops a game the predecessor holds. That check costs nothing when it never
runs, and a pool that silently loses games is unrecoverable in a way a
re-baseline is not.

**Per-axis coverage is reported when a pool is cut**, not at a designation that
no longer exists. What a reading resolves on an axis follows from the games the
pool holds on it, and that is worth reading whenever a pool is made.

## Consequences

The landmark that other records lean on moves. Several argue that a change is
free "before the evaluation core is designated": `0016`, `0030`, `0037`, `0041`,
`0045`, `0056`, `0059`, and `0062`. Read them as naming the first pool cut of
`#90` instead. The window they describe is the same window; it was only ever
called after the designation because the two were one event.

`0052` sizes the pool from what each metric needs to resolve an effect, and that
survives. What does not is the claim that per-axis power is fixed *permanently*
at designation. A later cut may raise the admission fraction freely, so a thin
axis is a reason to re-cut rather than a loss that outlives the decision. The
sizing question becomes ordinary rather than irreversible.

The selection pressure `0012` and `0013` worried about is unchanged in kind and
now unmeasured. Comparing many checkpoints against one pool applies it either
way; the growing current view was the instrument that would have made it
visible, and there is no longer one. A re-cut is what relieves it, and nothing
reports when relief is due.

Reversing this is possible and gets more expensive with time. Designating a core
later means either accepting that the line starts then, or re-scoring retained
checkpoints against the older pool, which needs those checkpoints to still load.

## References

- `docs/evaluation.md`
- `docs/decisions/0012-derived-evaluation-views.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/decisions/0052-a-bounded-pool-is-a-fixed-admission-fraction.md`
