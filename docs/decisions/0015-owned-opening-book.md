# 0015: An Owned Opening Book Matched By Position

Date: 2026-07-26

## Status

Accepted.

## Context

Rollout benchmarks compare generated games against human games on opening
distribution. That comparison needs an aggregation level, and the level decides
whether the comparison can support a statement like "this checkpoint plays too
few Sicilians."

Three sources of an opening label were available.

Source exports carry `ECO` and `Opening` headers. Decision 0004 keeps the
normalized schema source-agnostic, and these headers are the clearest case
against capturing source metadata: ECO is five hundred fixed buckets whose
granularity is historical rather than principled, the codes were defined by a
book of variations rather than by an algorithm so databases disagree on
assignment through transpositions, and the name strings differ per source. A
label built from them would mean something different depending on where a game
came from, and generated games have no header at all.

The literal first N plies need no book and no vendored data. They also fragment
broad openings across many buckets while narrow ones keep their mass in one, so
statistical power drains into buckets nobody cares about individually, and they
split transpositions that reach the same position by different move orders, so a
model with a systematic move-order preference looks wildly off while playing the
same chess.

Owning the book means owning the matching procedure, which is the property that
makes a label mean one thing across sources and across generated games.

## Decision

Vendor an opening book into the package, index it by position, and classify each
game by the deepest book position it reaches.

The vendored content is the Lichess `chess-openings` aggregate, pinned to an
upstream commit and dedicated to the public domain under CC0. What is checked in
is a canonical list of names with their moves in UCI plus an identity and
license record; the position index is derived at load time, so the checked-in
file stays auditable against upstream rather than being a table of opaque keys.
A maintenance script regenerates both from the pinned commit.

Matching is on positions, not move sequences, so transpositions land in the same
opening. The label is the deepest named position the game reaches, which is the
forward-scan equivalent of walking backward from the end of book depth until a
named position appears.

One classification pass emits three granularity levels — family, variation, and
line — so each benchmark picks the level it needs. Book names already nest from
broad to specific and separate their levels with a colon or a comma, so the
levels are the name truncated at its first and second separator rather than a
second table that has to be kept in step with the book.

Games matching nothing carry an explicit unclassified label at every level
rather than being forced into a nearest family.

Labels are derived in the evaluation view layer, per decision 0012, so changing
the book or the granularity never regenerates the corpus.

## Consequences

Updating the book is a deliberate act: it bumps the book version and changes the
book identity that benchmark artifacts carry, so results computed under two
books are distinguishable rather than silently merged.

Because the vendored book names every legal first move, a game played from the
standard starting position is effectively always classified at family level. The
unclassified label is therefore reached mainly by games that start away from the
opening, which is the honest outcome for a benchmark continuing from a mid-game
prefix.

Granularity derived from name structure inherits upstream naming conventions.
That is acceptable while the book is one aggregate with a consistent convention,
and it is the first thing to revisit if a second book is ever merged in.

Two book entries reaching the same position would make a label depend on
iteration order. Both the vendoring script and the loader reject that rather
than pick a winner.

Per-ply multi-label opening metadata for preference conditioning is separate
later work. It should extend this book rather than introduce a second one.

## References

- `docs/evaluation.md`
- `docs/preference-controls.md`
- `docs/decisions/0004-source-agnostic-normalized-data.md`
- `docs/decisions/0012-derived-evaluation-views.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
