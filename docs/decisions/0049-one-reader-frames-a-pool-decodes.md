# 0049: One Reader Frames, A Pool Decodes

Date: 2026-08-10

## Status

Accepted. Defines how preparation divides work across processes, and what a
run may assume about the artifact when it does.

## Context

`0045` names the corpus as 51 monthly archives and `0046` prepares them one at
a time. Preparation decoded games in one loop on one core, which is what makes
the corpus a schedule problem rather than a build.

Measured on the pinned 2017-04 archive, an unfiltered run scanned 576 games/s.
The selection publishes 3.35B games, so one loop is 67 days of decoding, and
every later data, training and evaluation task waits behind it.

Profiling says where that time goes and, more usefully, where it does not:

| Stage | Rate on one core |
| --- | --- |
| decompressing and reading lines | ~27,000 games/s |
| framing games without parsing them | 16,800 games/s |
| full decode: parse, replay, encode, derive | 576 games/s |

Within a decode, `chess.pgn.read_game` is 74% and the replay and encoding this
project adds is 26%, so parallelism has to cover the parse to be worth
anything, and work can only be handed out as unparsed text.

## Decision

**One process frames, many decode.** The reader walks the decompressed stream,
decides where each game ends, and hands batches of raw PGN text to a process
pool. Workers parse, replay, encode and classify; the reader consumes their
results in submission order and does the accepting, deduplication, counting and
shard writing exactly as before.

**Framing runs on the parser's own scanner.** `chess.pgn.read_game` with
`SkipVisitor` walks a game without parsing it, and a reader that records the
lines it consumed has that game's text. Where a game ends is then one
implementation rather than two that can disagree.

**Results are consumed in source order.** Acceptance, the duplicate check, the
accepted-game bound and shard boundaries all read one sequence, so the number
of processes cannot reach the artifact.

**Worker count is a runtime argument, not configuration.** `prepare_pgn` takes
it and `anthro data prepare --workers` supplies it. It is absent from
`PrepareConfig`, so it is absent from the manifest's resolved configuration and
from the selection identity an append is checked against.

## What it bought

Same archive, same 113,580-game prefix, on a 16-core machine with two threads
per core:

| Workers | Games/s | Against one |
| --- | --- | --- |
| 0 | 576 | 1.0x |
| 4 | 2,249 | 3.9x |
| 8 | 3,911 | 6.8x |
| 16 | 5,090 | 8.8x |
| 24 | 5,627 | 9.8x |
| 31 | 5,600 | 9.7x |

The corpus falls from 67 days of decoding to about 7. The shards and manifest a
31-process run wrote were byte-identical to the ones a single-process run wrote
over the same input, including across job boundaries and across the bound that
stops a run partway through a batch.

## Why not the obvious alternatives

**Splitting the archive by byte offset** and giving each worker its own range is
how this is usually done, and bzip2 is why it is not done here. The universal
export publishes bzip2 and nothing else, and a single stream has no offset a
worker can start decompressing from without the blocks before it.

**A second implementation of PGN framing** — split on blank lines, or on a line
starting with `[Event` — is several times faster than the parser's scanner and
was rejected. Both rules are wrong on a comment holding a blank line, the
parser's rule for ending a game is neither of them, and a framing disagreement
does not fail: it decodes a fragment of a game as a game. The scanner is 29x
faster than a decode, which is enough headroom for far more workers than this
machine has.

**Accumulating each batch's statistics in its worker** would take the last
non-framing work out of the reader. It is not done because a worker cannot know
which of its games the reader will drop as a duplicate or cut at the bound, so
the counts it returned would have to be corrected per game anyway.

## What this costs

**The scaling is the machine's, not the pool's.** Past 16 workers each one runs
at roughly half its solo rate, and throughput flattens near 5,600 games/s
whatever the pool size. The reader is not the constraint — it can frame three
times faster than the pool consumes — so tuning it further buys nothing on this
machine. A machine with more physical cores will go faster without any change
here.

**Shard writing stalls the pipeline.** Writing a 50,000-game shard and digesting
it takes 1.2s in the reader, during which nothing is framed and every worker
drains. That is about 7% at the shard size the selection pins, and moving it
off the reader is a separate change.

**A decoding failure now crosses a process boundary.** Anything raised while
parsing a game surfaces from the pool rather than from the loop, and the
archive stays open until the pool has shut down.

## References

- `#389`, and `#89` which it unblocks
- `docs/data.md` (Preparing At Corpus Scale)
- `docs/decisions/0045-centisecond-clocks-from-a-closed-export.md`
- `docs/decisions/0046-a-corpus-is-appended-one-archive-at-a-time.md`
