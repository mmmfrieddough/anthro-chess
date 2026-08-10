# 0048: A Pinned Pool Generation Is Not Part Of A Cost Workload

Date: 2026-08-10

## Status

Accepted. Refines `0031-committed-benchmark-cost.md`.

## Context

Decision 0031 made the declared cost workload a digest of a benchmark's whole
resolved configuration, with two normalizations: the model selection comes out,
and every path drops its machine prefix. It then drew one deliberate limit,
arguing that a pool joins the digest "as the artifact it names, not as the
realized dataset identity — `pool_id`, `pool_version`, `game_ids_sha256` — that
every reading carries beside it." Re-freezing a pool at the same path therefore
keeps both readings on one cost line, which is what a reader wants: a bigger
pool costs more to read, and that is a fact about the work rather than a
different kind of work.

That argument rested on an accident. Realized data identity stayed out of the
digest because it was not in the configuration — it lived in the pool's own
manifest and in each reading's dataset reference. Nothing enforced it.

A benchmark selection now pins the pool generation it is defined over, so a
superseded pool left at the configured path is refused rather than scored. That
pin is realized data identity, it is in the configuration, and it moves at every
generation cut. Left in the digest it would have inverted 0031's guarantee
exactly: each cut would start a fresh `benchmark.wall_clock_seconds` line for
every benchmark, by way of the pin the cut forces rather than by any change in
how much work there is.

## Decision

**A configured field naming realized data identity is excluded from the cost
workload.** The pin is dropped wherever it sits in a selection, including
nested under a benchmark's own table.

The exclusion is derived from the type that declares the pin rather than from a
field-name literal. The five benchmark schemas carrying a pin inherit it from
one model, and the normalizer drops exactly that model's fields, so a sixth
benchmark gets the exclusion by inheriting the pin and cannot get it wrong by
spelling the field differently.

This makes 0031's third normalization explicit. Two of them are about what a
cost line varies along; this one is about what a cost line must not fracture on.

## Consequences

The workload digest is unchanged by this feature: a selection that pins a
generation and one that pins none digest identically, and both match what the
same selection digested before the pin existed. No cost series breaks, and a
generation cut still leaves both sides of it on one line.

The rule now has an owner. A future field that records which data was realized
rather than how much work was done belongs on the same model, and a future
reader asking why a re-freeze does not start a new cost line is answered here
rather than by a comment in the normalizer.

The limit 0031 drew is unchanged and still deliberate: a re-freeze that grows
the pool makes the reading genuinely more expensive, and that shows up as a
movement on one line rather than as a new one. The readings themselves record
which pool version they scored, which is where a reader confirms two cost
figures are comparable.

## References

- `docs/decisions/0031-committed-benchmark-cost.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/decisions/0018-workload-scoped-efficiency-series.md`
- `src/anthro_chess/evaluation/cost.py`
- `src/anthro_chess/evaluation/pool.py`
