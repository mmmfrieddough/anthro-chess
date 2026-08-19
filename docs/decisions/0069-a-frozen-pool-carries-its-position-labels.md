# 0069: A Frozen Pool Carries Its Position Labels

Date: 2026-08-18

## Status

Accepted. Refines `0012-derived-evaluation-views.md`, which said derived
benchmark inputs are never stored data. That still holds for views, which select
games; it no longer holds for the rule-sensitive labels of the games a frozen
pool already fixed.

## Context

The canonical pool reading took four hours, and 63.5% of it resolved the
forward-looking predicates the adjudicated-decisions family reads: pushing every
legal move to test for mate, the null-move probe for a threatened mate, and the
exchange on every capture. Measured on this host over 960 games spread across
the batch plan, that is 43 ms per game against 3.8 ms to encode the same game.

The predicates and characteristics of a position are a pure function of the
position. A pool is frozen, pinned by a generation digest, and refused when
superseded, so every reading of every checkpoint and every ablation arm resolved
the same answer again. Measured over the canonical pool, the whole of that
answer is 14,161,038 positions and 26.7 MB, which is smaller than the pool's own
`games.parquet`.

`0012` ruled out stored derivations for a reason that still applies to views: a
benchmark needing a field the layer cannot derive is a schema problem rather
than a reason to grow a cache. Position labels are not that case. They are
derivable, they are derived by the same code either way, and what is stored is
the output of that code against a frozen input rather than a field nothing owns.

## Decision

**A pool's position labels are derived once per generation and read back after
that.** The artifact sits beside the pool, sharing its lifetime and going away
with it.

**Nothing has to remember to build it.** The first reading of a pool that has no
matching artifact derives one and saves it; every later reading finds it. There
is no build step to run, and no state that is correct only because somebody ran
one.

**Staleness is a miss rather than a hazard.** The artifact carries a key over
the pool identity, the slice scheme version, and the names of every
characteristic and predicate. A re-cut pool, a changed scheme, a reordered
enum, and an added predicate all fail to match, and a failure to match derives
the labels again rather than reading what was written under other rules.

**Live derivation stays.** Anything scoring positions no pool holds — a
perturbed continuation, a generated game — resolves its labels as before. The
artifact is a store of answers for a fixed set of questions, not a replacement
for the ability to ask them.

## Consequences

A canonical reading no longer pays for the predicates at all, and the first
reading of a new pool generation pays about twelve minutes across this machine's
cores. The saving is per reading rather than once: it lands again on every
checkpoint, every ablation arm, and every in-training preview over the same
pool.

The first in-training preview after a pool is cut stalls for that build. It is
the same work the reading would have done anyway, taken once instead of at every
cadence.

A predicate change now has a cost that a reading pays rather than a maintainer:
the key stops matching and the next reading rebuilds. That is the intended
behavior and it is why the key covers the names rather than only the version.

The artifact is a machine-local derived file, so a machine that has never read a
pool has to derive it and a machine that has can be handed nothing. That is the
same shape every other artifact root here already has.

## References

- `src/anthro_chess/evaluation/labels.py`
- `docs/evaluation.md` (Benchmark Data Layers, Adjudicated Decisions)
- `docs/decisions/0012-derived-evaluation-views.md`
- `#519`
