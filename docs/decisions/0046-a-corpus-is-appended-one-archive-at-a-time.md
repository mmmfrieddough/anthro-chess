# 0046: A Corpus Is Appended One Archive At A Time

Date: 2026-08-10

## Status

Accepted as initial design direction. Defines how a corpus spanning many source
archives is built, and what a run may assume when it is interrupted.

## Context

`0045` names the corpus as 51 monthly archives. Preparation built a corpus from
one: it wrote one manifest recording one input digest, numbered shards from zero
per run, and so replaced whatever was already in the output directory.

The constraint that settles the shape is storage rather than convenience. The 51
archives total 0.97 TB and the corpus lands near 465 GB, against 787 GB of disk
on the machine that has to build it. They cannot both exist, so the only
workable order is fetch one month, prepare it, delete the archive, continue.
Preparation therefore has to accept archives one at a time, and a pass that
takes many hours across many archives has to survive being interrupted in the
middle.

Doing that by hand — renaming shards, merging manifests — discards exactly what
makes the corpus reproducible. A manifest that cannot say which archive a shard
came from cannot be checked against the source's published digests at all.

## Decision

**One run prepares one archive and appends it.** A run reads one input, writes
that archive's shards beside whatever is already in `normalized/`, and rewrites
the manifest to span every archive prepared so far. Preparation never replaces
a corpus; rebuilding means removing the artifact directory.

**Shard names carry a prefix of the input's digest.** A name is then unique
across archives without consulting the manifest, which means a retried archive
overwrites its own shards and nothing else. The shard record repeats the full
digest so a reader checks provenance from the manifest rather than by parsing
file names.

**Corpus totals are derived from the per-archive records**, not carried forward
from the previous run's totals. Each archive's entry holds its own digest,
counts, rejection reasons, split counts and coverage; the corpus-wide `games`,
`coverage` and `split.counts` blocks are recomputed from that list on every
write.

**An archive the manifest already records is skipped.** Re-running a pass from
its beginning costs one digest of the input and changes nothing, which is what
makes the pass resumable.

**A selection that would change what a game becomes is refused.** The `source`,
`split`, `filters` and `termination` sections, the schema and preprocessing
versions, and the action vocabulary must match what the manifest recorded.

**The accepted-game bound counts the corpus**, not each archive.

## Why these rather than the obvious alternatives

**Sequential shard numbering continued from the manifest** is the smaller change
and was rejected. It makes the next shard's name a function of manifest state,
so a manifest that is missing or stale sends the next run's writes over another
archive's shards. It also leaves the stale-shard sweep unable to tell an
interrupted attempt's orphans from another month's data without trusting that
same state. Digest-derived names remove both failure modes by construction.

**Merging each archive's numbers into a running total** needs the same amount of
merge code as deriving them, because combining N blocks and combining two are
the same operation. Deriving from the parts buys something the running tally
does not: the totals cannot come to disagree with the records they summarize,
and there is no state whose loss silently corrupts the next write.

**Refusing a raised game bound** would have been the consistent reading of "one
corpus, one selection", and is deliberately not done. Raising a bound keeps
every game already accepted and admits more, which is the expansion
`docs/data.md` describes; refusing it would force a full rebuild of 465 GB to
add games. `split.require_nonempty` is exempt for the weaker reason that it
checks a result rather than shaping one. Every other field fails closed, so a
configuration section gaining a field refuses an append until someone decides
the field is safe.

## What this costs

**Source order becomes run order.** The recipe takes accepted games in source
order until the bound, and across archives that order is the order the runs
happened rather than anything the configuration states. A corpus built months
1..51 and one built 51..1 under the same bound are different corpora. The
manifest's `inputs` list records the order that was used.

**Re-preparing one archive of a built corpus is not supported.** An archive
already recorded is skipped, so correcting one month means rebuilding the
corpus. Supporting it would mean subtracting a set of games from a bound applied
in run order, and nothing needs it yet.

**A marked-account snapshot covers the archives its census counted**, and
`require_archive` refuses any other. Appending an archive the snapshot does not
cover stops the run rather than preparing it unfiltered, and this rule is why a
snapshot is cut only once the census has counted every archive a selection pins:
a corpus that dies on its fortieth append cannot be repaired incrementally.
`0041` and `0047` carry that end.

**Duplicate detection is per archive, not per corpus.** A run rejects a game id
it has already seen in the archive it is reading, and holding every id of a
2.2B-game corpus in memory to extend that across archives is not affordable, so
two archives publishing the same game contribute it twice. The split-boundary
guarantee survives — assignment is a pure function of the id, so both copies
land in the same split — but a game can appear twice in the corpus and twice in
a pool frozen from it. The pinned selection is 51 distinct months, which do not
overlap; a source that re-cuts or re-publishes archives would.

**The manifest grows with the corpus**, to tens of thousands of shard records at
full size, and every consumer reads it whole. That is a scale problem the corpus
already had and this does not fix.

**One corpus directory takes one writer at a time.** An append is a
read-modify-write of the manifest with no lock, so two runs against the same
directory can each write a manifest that omits the other's archive, and the
loser's shards are then swept as orphans. `docs/data.md` already asked for no
concurrent writers against one corpus; appending makes the cost of ignoring it
an archive rather than a run.

## Consequences

- `anthro data prepare` reports what it did to the corpus, not only what it
  read: an archive prepared, an archive already in, or a corpus at its bound.
- A selection pinning many archives requires an explicit `--input` per run,
  because it has no single default archive.
- The manifest records `inputs` rather than `input`, and a corpus prepared
  before that is refused rather than migrated.
- A frozen pool's source provenance carries the corpus's whole `inputs` list.

## References

- `#388`, and `#89` which it unblocks
- `docs/data.md` (Building One Corpus From Many Archives, Corpus Expansion)
- `docs/decisions/0011-held-out-test-partition.md`, for why appending cannot
  move an existing game between splits
- `docs/decisions/0041-games-of-marked-accounts-leave-the-corpus.md`
- `docs/decisions/0045-centisecond-clocks-from-a-closed-export.md`
