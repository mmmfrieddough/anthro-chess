# 0053: The Pool Is Sized To The Reader It Waits On

Date: 2026-08-11

## Status

Accepted. Refines `0049-one-reader-frames-a-pool-decodes.md`, whose account of
what constrains a run no longer holds, and records what a second preparation
running beside the first is worth. The idle half of the machine it leaves is
taken up by
`0054-archives-are-prepared-together-and-recorded-once.md`.

## Context

`0049` divided preparation into one reader that frames and a pool that decodes,
and measured the pool as the constraint: throughput rose to 24 workers, the
reader could frame three times faster than the pool consumed, and a machine
would have to grow threefold before the reader mattered. The default pool size
was set from that — one decoder per core the process may run on, less one for
the reader.

`0050` and the one-pass decode landed since made a decode about 2.3x cheaper per
game. The pool consumes that much faster, and the reader was not made faster at
all, so the threefold headroom `0049` measured is gone. Re-measured on the same
archive, on an otherwise idle machine, at a pinned revision:

| Workers | Scanned/s | Reader CPU | Pool CPU |
| --- | --- | --- | --- |
| 4 | 5,001 | 51% | 3.7 cores |
| 6 | 6,906 | 72% | 5.2 cores |
| 8 | 8,335 | 88% | 6.6 cores |
| 12 | 8,459 | 96% | 8.3 cores |
| 16 | 8,451 | 100% | 8.1 cores |
| 24 | 7,890 | 99% | 7.7 cores |
| 31 | 7,852 | 101% | 8.0 cores |

The reader holds a full core from twelve workers on, and the pool never draws
more than about eight of the machine's sixteen. Throughput is flat from eight
to sixteen and falls slowly after, so the default this machine computed — 31 —
sat 7% below a plateau it had already passed, while spending 30% more CPU per
game on dispatching to decoders that were waiting.

Profiling the reader says the core is spent on framing (62%) and decompression
(21%). Writing shards is 4.3% of it.

## Decision

**The default pool size is capped at what one reader can keep fed.** It stays
`affinity - 1` on a small machine and stops at twelve on a large one.

**The cap is a property of the pipeline, not of the machine.** A reader spends
about 113 microseconds of CPU per game and a decode about 981, so a reader
saturates at roughly nine decoders. Both scale with a core's speed, so the
ratio is what carries across machines, where a core count does not. Twelve
rather than nine because the plateau is wide and the penalty is asymmetric:
four workers cost 41% of throughput, four too many cost nothing measurable.

**An explicit `--workers` is still obeyed exactly**, including values past the
cap, because a caller running several preparations at once is sizing each one
against the others rather than against the machine.

## What it bought

7% against the default this machine previously computed, and about 30% less
CPU burned per game. Nothing about the artifact changes; `0049`'s guarantee
that the worker count cannot reach it is untouched.

## What this costs

**A larger machine no longer buys a larger pool.** That is the finding rather
than a regression — past the cap the extra decoders were idle — but it does
mean preparation now leaves half of this machine unused, which the next record
to touch this should take up rather than treat as settled.

Measured, in the same conditions, several preparations running side by side
rather than one with a bigger pool:

| Arrangement | Aggregate scanned/s | Against one |
| --- | --- | --- |
| 1 preparation, 12 workers | 8,614 | 1.00x |
| 2 preparations, 8 workers each | 13,753 | 1.60x |
| 4 preparations, 4 workers each | 16,334 | 1.90x |
| 8 preparations, 3 workers each | 18,417 | **2.14x** |

The reader is a per-preparation ceiling, not a per-machine one, and archives
are independent inputs. Shard names already carry their input's digest, so the
shards of concurrent runs do not collide; it is the manifest's read-modify-write
that `docs/data.md` says two runs cannot both do. Lifting that is the work this
record does not do, and it is worth more than everything measured under `#389`
put together.

## Why not the obvious alternatives

**A formula over physical cores, or half the thread count**, gives sixteen here,
which measures the same as twelve. It is rejected because it agrees with the
answer only at this size: on a machine with twice the threads it says 32, which
this table shows is well into the declining tail. The quantity that generalizes
is the reader-to-decode work ratio, and no core count expresses it.

**Making the reader faster instead** is the real repair and is not attempted
here. Framing is 62% of its core and runs on `python-chess`'s own scanner,
which `0049` chose deliberately over a second account of where a game ends;
decompression is a further 21% and is bzip2 because the universal export
publishes nothing else.

**Moving the shard write off the reader** was built and measured three times
and is not merged. It is 4.3% of the reader's core, and handing records to
another process costs more than it saves at the shard size the selection pins:
0.973x at 50,000 games per shard, 1.008x at 10,000.

## References

- `#389`
- `docs/data.md` (Preparing At Corpus Scale)
- `docs/decisions/0049-one-reader-frames-a-pool-decodes.md`
- `docs/decisions/0050-a-header-rejection-outranks-a-parse-error.md`
