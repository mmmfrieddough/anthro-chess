# 0049: A Bounded Pool Is A Fixed Admission Fraction

Date: 2026-08-10

## Status

Accepted. Refines `0012-derived-evaluation-views.md`.

## Context

A pool selection names a corpus, a manifest, and a split, and nothing else. The
pool is therefore the whole test split: 50,073 games today, and roughly 110
million once the corpus `#89` widens is prepared, since the split holds 5% of
2.21 billion games.

Benchmark work does not grow with that. Every benchmark declares a bounded view,
and the largest declared one takes 12,000 games. What grows is everything a
process pays before a view is applied. Measured on the frozen 50,073-game pool,
and extrapolated from that per-game cost:

| per benchmark process | at 50,073 games | at 110 million |
| --- | --- | --- |
| `load_pool` wall clock | 1.59 s | ~58 min |
| resident `PoolGame` tuple | 37 MB | ~81 GB |
| peak while loading | 60 MB | ~132 GB |
| artifact re-checksummed on every load | 10.4 MB | ~23 GB, ~14 s |

So the shape is bounded work behind unbounded startup, and the largest view
would use about 0.01% of what every process materializes.

Shrinking `split.test_fraction` fixes the arithmetic and is the wrong lever.
Games leaving `test` is what containment forbids once a core is designated, so a
fraction chosen tight now could never be loosened, and one chosen generous
reproduces this.

## Decision

**The pool is a bounded uniform sample of the test split, and the bound is a
fixed admission fraction rather than a game count.**

A test game is admitted when a seeded rank of its game id falls below the
threshold the fraction implies. Admission is a pure function of the game id and
a constant seed — the property decision 0011 already rests on for split
assignment — so growing the corpus only ever adds games to the pool.

A count cannot hold. The next generation would rank a larger split and keep the
lowest N of it, and the newly available games take ranks among the old ones and
push some of them past N. Every generation must contain the last, so a bound
that evicts is not a bound this project can have. Bounded and contained together
force a fixed fraction; there is no third form.

**The bound is sized from evaluation power, and is at least 100,000 games at
designation.** The number comes from the dispersions the committed readings in
`results/records/` measured over 400 games, extrapolated by `uv run anthro eval
noise plan`:

| to resolve | binding metric | games |
| --- | --- | --- |
| 1% of value, held-out family | `held_out.move_loss_under_1200` | 15,282 |
| 1% of value, legality family | `legality.mask_penalty_castling_rights` | 64,277 |
| 0.5% of value, held-out family | `held_out.move_loss_under_1200` | 61,126 |
| 1% of value, en passant slice | `legality.mask_penalty_en_passant` | 645,623 |
| 1% of value, stalemate availability | `adjudicated.stalemate_available_best_rank` | 1,727,807 |

100,000 games therefore resolves a 1% relative effect on every held-out and
legality metric except the rare-rule slices, and a 0.5% effect across the whole
held-out family, with headroom for the view filters that discard part of a pool
before ranking it. It costs roughly 74 MB resident and 3.2 s per process at the
rates above. It does not buy the rare-rule tail, and nothing cuttable buys the
stalemate-availability family, whose noise at 400 games is the same size as the
quantity it measures.

**Raising the bound later is available; lowering it never is.** A threshold that
only rises admits a superset, which is what a generation cut already is. The
number to choose is therefore the smallest defensible one rather than the
largest affordable one, and the rare-rule tail is what a later raise would be
for.

## Consequences

Generation one is untouched. With no fraction configured the freeze admits the
whole split, so the existing pool's identity digest and the five selections
pinned to it are unchanged. `#90` sets the fraction that realizes the target
against the widened corpus, and later generations reuse that fraction rather
than re-deriving a count. Where a realized test split is smaller than the
target, no fraction is configured and the pool is the split.

The pool still tracks corpus composition. Admission is uniform and unstratified,
so 0012's representativeness property is unchanged and a view over a sampled
pool remains a uniform sample of the corpus.

0012 controlled runtime at the view layer rather than by shrinking the pool.
That still holds, at one remove: views are still where a benchmark buys speed,
inside a pool that is now itself capped relative to the split.

The pool grows with the corpus, by the same multiple. A fixed fraction is the
only bounded rule containment allows, so a corpus that doubles doubles the pool.
That growth is the growth `current` exists to have, and it is nothing like the
2000x the unbounded split imposes at the widened corpus.

Freezing becomes bounded too. Both the rows a freeze accumulates and the train
ids it holds for the overlap check are filtered by admission, so the pass over
the corpus no longer has to hold the split in memory to write a pool from it.
The recorded count of compared train games still covers the whole split.

The sizing rests on one dispersion per metric, measured at proof scale, and
`games_to_resolve` deliberately errs high. A later reading that measures a
different spread moves the number a future generation is cut at, not the
mechanism this record decides.

## References

- `docs/evaluation.md`
- `docs/data.md`
- `docs/decisions/0011-held-out-test-partition.md`
- `docs/decisions/0012-derived-evaluation-views.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `src/anthro_chess/evaluation/pool.py`
