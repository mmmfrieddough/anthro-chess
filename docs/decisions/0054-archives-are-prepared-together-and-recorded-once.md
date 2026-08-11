# 0054: Archives Are Prepared Together And Recorded Once

Date: 2026-08-11

## Status

Accepted. Refines `0046-a-corpus-is-appended-one-archive-at-a-time.md`, which
this leaves true of the manifest and no longer true of the decoding, and takes
up the unused half of the machine that
`0053-the-pool-is-sized-to-the-reader-it-waits-on.md` left behind.

## Context

`0053` capped the decoding pool because one preparation cannot use a whole
machine: the reader that frames its games runs in a single process, and past
about a dozen decoders it is what the pool waits on. Throughput saturates near
8,450 scanned/s on this machine with roughly half of it idle, and no worker
count changes that.

`0046` prepares one archive per run, which is what lets a selection larger than
the disk be built at all. Nothing about that requires the *decoding* to be one
at a time — archives are independent inputs, and shard names already carry
their input's digest so two archives cannot collide. What cannot be shared is
the manifest, and `0046` says so: "two runs against the same directory can each
write one that omits the other's archive, and the loser's shards are then swept
as orphans."

That understates it. A run does not only rewrite the manifest; it deletes every
shard the manifest it is about to write does not claim. Two runs at once means
one deleting the other's shards while they are still being written, which is
lost data rather than a lost update. A lock around the manifest would not have
fixed it.

## Decision

**One run prepares several archives and writes their manifest once.** Decoding,
shard writing and the per-archive census counts happen per archive and in
parallel; the corpus totals, the split check, the sweep of unclaimed shards and
the manifest rewrite happen once, after every archive is done.

**Per-archive work is kept free of corpus state**, which is what allows it to
run elsewhere. `_prepare_archive` returns the shards it wrote and the manifest
entry describing them, and reads nothing about the archives beside it.

**The inputs decide the order, not the finishing.** The manifest a concurrent
run writes over some archives is byte for byte the manifest a run over the same
archives one at a time writes.

**A selection with a corpus-wide `maximum_games` is still prepared one archive
at a time**, whatever `--concurrency` asks for. What an archive may admit is the
bound less what every archive before it contributed, so deciding that for two at
once would overshoot by whatever the second one took.

**Archives prepared at once divide the machine rather than each taking it.**
`0053`'s cap is what one reader can be fed, which is a bound per archive; the
pool is the smaller of that and the machine's cores divided between the archives
sharing it. Without the second bound the default forks a full pool per archive,
and eight archives ask for 96 decoders on 32 threads.

**How many archives is derived, not asked for.** A selection pinning many has no
single default input, so naming none of them prepares all of them, at the fewest
that fill the machine — the fewest because each archive in flight is one more
that has to be on disk and one more marked-account snapshot held. Naming one
input still prepares exactly that one.

## What it bought

Two real archives of 250,000 games each, on an idle machine:

| Arrangement | Wall | Rate | |
| --- | --- | --- | --- |
| One at a time, 12 workers | 55.4s | 9,027 scanned/s | 1.00x |
| Both at once, 4 workers each | 49.9s | 10,017 scanned/s | 1.11x |
| Both at once, 6 workers each | 37.6s | 13,302 scanned/s | 1.47x |
| Both at once, 8 workers each | 35.7s | 13,998 scanned/s | 1.55x |
| Both at once, 12 workers each | 33.4s | 14,988 scanned/s | **1.66x** |
| Both at once, 15 workers each | 33.7s | 14,854 scanned/s | 1.65x |

The output was byte-identical between them, over every shard and the manifest.

Two archives are enough to show that the gain is not a tail effect: alone they
take 27.6s and 28.7s, so the slower one is 1.04x the faster and the wall clock
of a concurrent pair is not being set by an imbalance between them.

How far it goes, over twelve distinct archives of 40,000 games so that every
arrangement divides them evenly and none ends on a short last wave:

| Archives at once | Decoders each | Processes | Rate | Against one |
| --- | --- | --- | --- | --- |
| 1 | 12 | 13 | 8,812 | 1.00x |
| 2 | 12 | 26 | 14,976 | 1.70x |
| 3 | 9 | 30 | 17,568 | 2.00x |
| 4 | 7 | 32 | 18,538 | **2.11x** |
| 6 | 4 | 30 | 18,481 | 2.10x |
| 12 | 1 | 24 | 14,576 | 1.66x |

Throughput peaks where the processes come to the machine's own count, and both
sides of that are worse: two archives leave six threads unused, twelve give
each reader a pool too small to keep it fed. Four is the fewest arrangement
reaching the peak here, and is what the default picks. For the pinned
51-archive selection this is roughly four and a half days of decoding against
about two.

`0053` measured the ceiling this approaches with more archives in flight: eight
at three workers each reach 18,417 scanned/s, **2.14x** one preparation with the
machine to itself. For the pinned 51-archive selection that is roughly four and
a half days of decoding against about two.

## What this costs

**Several archives have to be on disk at once.** `0046`'s reason for one at a
time was a selection larger than the disk, and preparing eight together needs
eight of them fetched. At this selection's sizes that is about 50 GB against a
machine with 313 GB free, but it is a real constraint on how far the arrangement
scales and it is the caller's to manage.

**The marked-account snapshot is loaded once per archive rather than once.**
Every decoder already holds a copy, so the multiplier is unchanged per decoder,
but the total is what the concurrency is.

**A crash now loses more.** One archive at a time recorded each archive as it
finished; a batch records nothing until all of them are done, so an interrupted
batch leaves shards that the next run sweeps and redoes. `0046`'s property that
re-running from the beginning costs nothing still holds — it just costs the
batch rather than the archive.

## Why not the obvious alternatives

**A lock around the manifest**, leaving runs otherwise independent, does not
work: the sweep of unclaimed shards is destructive against a concurrent run's
output, and it happens before the manifest is written rather than under whatever
lock protects it.

**Recording each archive as it finishes**, under that lock, would keep the
crash behaviour `0046` has. It is rejected because the sweep would still have to
be suppressed while any archive is in flight, which means the batch has to be
known anyway — at which point the single rewrite is simpler and writes the
manifest fewer times.

**Widening the pool instead** is what `0053` measured and capped. Past about a
dozen decoders they wait on the reader rather than work.

## References

- `#389`
- `docs/data.md` (Preparing At Corpus Scale, Building One Corpus From Many Archives)
- `docs/decisions/0046-a-corpus-is-appended-one-archive-at-a-time.md`
- `docs/decisions/0053-the-pool-is-sized-to-the-reader-it-waits-on.md`
