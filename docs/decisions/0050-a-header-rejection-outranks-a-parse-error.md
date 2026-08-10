# 0050: A Header Rejection Outranks A Parse Error

Date: 2026-08-10

## Status

Accepted. Refines `0049-one-reader-frames-a-pool-decodes.md`, and changes what
preparation records as a rejected game's reason.

## Context

`0049` put decoding on a pool and found the machine rather than the reader is
what binds: past 16 workers each one runs at roughly half its solo rate. Work
moved between processes therefore buys nothing at the pool sizes this runs at.
Only work that stops being done at all does.

About half of a real archive is never built into anything. Over the first
200,192 games of the pinned 2017-04 universal export, 95,869 are rejected, and
95,559 of those are settled by the headers alone — the filters reading `Event`,
`Variant`, `SetUp`, `Termination`, `Site` and the rating tags. Every one of
them was parsed in full first: the movetext walked, each SAN resolved against a
board, each move pushed, and a record assembled and discarded. The filters ran
afterwards, on headers that had been sitting in memory the whole time.

## Decision

**Filters the headers settle run at `end_headers`, and a game they reject is
skipped there.** `chess.pgn.read_game` takes `SKIP` from that callback and
switches to the scanner it uses to find the end of a game — the same scanner
`0049` frames on, so no third account of where a game ends is introduced.

**One function owns the header stage.** `_screen_headers` holds every filter
that needs no move. It returns the reason when the headers rule the game out,
and otherwise returns what it read, so the move stage does not read the same
tags a second time.

**A header rejection is final.** Nothing after `end_headers` runs for such a
game, so nothing later can name a different reason for it.

## What it bought

Same fixed 10,240-game slice at a 200,000-game offset into the pinned 2017-04
archive, and the same bounded 24-worker run, alternating arms so both saw the
same machine:

| | Before | After | |
| --- | --- | --- | --- |
| Decode stage, one core, CPU time | 864 games/s | 1,390 games/s | 1.61x |
| End to end, 24 workers, wall clock | 6,311 scanned/s | 7,091 scanned/s | 1.12x |

Medians of three pairs, and ratios rather than rates: another tenant held about
a quarter of the machine throughout, which both arms saw equally and which
holds the absolute end-to-end numbers below what a quiet machine would give.

Unlike a change that moves work between processes, this one converts. It
removes CPU per game rather than relocating it, which is the only thing that
helps once the machine rather than any one process is the constraint.

Measured the same way against the state `0049` left, which also carries the
one-pass decode landed since, the whole is 2.33x on the decode stage and 1.39x
end to end. The corpus that record put at about seven days of decoding is now
about five.

## What this costs

**A rejected game's recorded reason can change.** A game the headers rule out
is no longer given the chance to fail to parse first, so where it qualified for
both, the manifest now names the header reason instead of `pgn_parse_error`.
Nothing becomes false — each new label is true of the game and strictly more
specific — and no accepted game moves.

Measured over those 200,192 games: the accepted set, and every accepted
record's bytes, are identical; the rejected set is identical; two games change
reason, both from `pgn_parse_error` to `unrated_game`.

**This has to land before a corpus build starts, not during one.**
`rejection_reasons` is not part of the selection identity an append is refused
for, so preparing some archives before this change and some after leaves one
manifest whose per-archive censuses were computed two ways with nothing
detecting it.

**An archive cannot demonstrate what it does not contain.** The pinned export
carries no unparseable `FEN` and no unresolvable `Variant`, both of which
`read_game` reports before it reads a move and which this change also moves to
the header reason. Only the tests say what those games now get.

## Why not the obvious alternatives

**Keeping the old reasons exactly**, by re-parsing a header-rejected game to
find out whether it would also have failed, was measured at 0.950x — slower
than doing nothing, because it parses precisely the games the skip exists to
avoid. There is no formulation of this change with a zero artifact delta.

**Screening in the reader**, so that only survivors are framed out to the pool,
was rejected on `0049`'s own finding. The reader is already near 100% of a core
at 24 workers, and the pool it feeds has spare workers rather than spare
reader; per-game work belongs on the side that scales.

## References

- `#389`
- `docs/data.md` (Preparing At Corpus Scale)
- `docs/decisions/0049-one-reader-frames-a-pool-decodes.md`
- `docs/decisions/0041-games-of-marked-accounts-leave-the-corpus.md`
