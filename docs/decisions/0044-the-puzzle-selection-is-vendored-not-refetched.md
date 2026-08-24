# 0044: The Puzzle Selection Is Vendored Rather Than Refetched

Date: 2026-08-09

## Status

Accepted. Replaces the artifact-boundary consequence of
`0019-external-puzzle-calibration-set.md`, which kept the selected records out
of the repository, and leaves the rest of that record — the uniform exact-rating
design, the sizing calculation, and the identity a reading carries — untouched.

Refined by `0078-puzzle-training-overlap-is-measured-and-not-corrected.md`, which
rests on the no-network, no-archive build below to refuse a selection filtered
against the corpus. The two megabytes this record weighed are now twenty: the
selection was re-derived at a hundred puzzles per rating, and the reasoning
about what may be committed is unchanged by the size.

## Context

Decision 0019 committed the recipe and the expected identity and left the
selected records under the data root, on the reasoning that a pinned source
digest plus a deterministic selection makes a rebuild elsewhere checkable.

That holds only while the pinned source is fetchable. Upstream serves puzzles at
one rolling URL with no dated snapshot beside it and no history, so a pin stops
resolving the moment upstream regenerates. The first pin lasted three days. Its
replacement was already stale by the time a second machine would have needed it,
and no revision of that file is addressable once it has been overwritten.

The consequence is not a slow rebuild but an impossible one. The build correctly
refuses a source whose digest does not match, so a machine that never downloaded
the archive cannot reach the pinned identity by any route. What existed instead
was a single uncommitted copy on one host, roughly two megabytes, holding the
only surviving instance of the rows every puzzle reading would be compared
against. Losing that host would have ended the comparability of the project's
one external yardstick, and re-pinning is not a repair: it selects different
puzzles, so it starts a new set version rather than restoring the old one.

The archive itself cannot be committed — it is roughly 300 MB and its licence
is not the constraint. The selection cut from it can: it is about 2 MB, is CC0,
and is the part a reading actually consumes.

## Decision

Vendor the selected rows into the package, beside a record of the upstream
revision and population they were drawn from, and build the artifact from that
pair rather than from the archive.

`anthro eval prepare-puzzles` reads the vendored pair, checks it against the
configuration, and installs the artifact under the data root exactly as before.
It needs no network and no archive. The pinned archive stays in the
configuration as provenance and as the input a re-derivation reads;
`scripts/vendor-puzzle-selection.py` is the only thing that reads it, following
the pattern `0015-owned-opening-book.md` already uses for the opening book.

The configuration states the pin and the design; the vendored record states what
was selected under them. A build compares the two and refuses when they
disagree, so re-pinning without re-deriving fails rather than producing an
artifact whose manifest describes rows it did not produce.

The vendored record carries what the archive pass observed and nothing else: the
upstream revision, the design applied to it, and the coverage of the population
it drew from. The manifest takes those three from the record and regenerates the
rest — licence and the sizing calculation — from the configuration and the code,
so a build stores a fact twice only where the second copy is the thing being
compared.

## Consequences

The canonical artifact is reachable from a clone alone. A machine that has never
downloaded the archive reproduces `expected_puzzles_sha256` byte for byte, which
is what makes a reading taken on one machine comparable to a reading taken on
another. The rebuild recipe loses its dependence on a URL that has already
failed twice, and the network is needed only for the game corpus.

The wheel grows by about 2 MB. That is the price of the artifact surviving the
loss of any single machine, and it is paid once rather than per rebuild.

The identity did not change. The vendored rows are the rows the pinned archive
selected, so no reading is invalidated and no set version is bumped by this
change.

Re-pinning is now two steps that fail loudly when only one is taken: edit the
configuration, then run the vendoring script and record the identity it prints.
What the comparison cannot catch is a filter edited in the configuration and
re-derived in the same pass — that is an intentional new selection, and the
identity it produces is what distinguishes it.

The vendored selection is package data rather than an evaluation pool, which is
a boundary 0019 deliberately drew the other way. It is drawn here on
availability rather than convenience: an evaluation pool is regenerable from a
corpus that is itself pinned and fetchable, and this one is not.

## References

- `docs/evaluation.md`
- `docs/decisions/0015-owned-opening-book.md`
- `docs/decisions/0019-external-puzzle-calibration-set.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
