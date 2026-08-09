# Marked-Account Snapshots

Each file here records which accounts a source had marked for breaking its
rules at one moment, for one archive. Preparation rejects every game those
accounts played. `uv run anthro data mark-accounts` builds one;
`docs/decisions/0041-games-of-marked-accounts-leave-the-corpus.md` owns why the
games go rather than the moves.

These are pinned inputs rather than derived artifacts, which is why they are
checked in beside the selection that names them. Everything else the pipeline
reads is reproducible from a digest — the archive is fetched by URL and
verified against a checksum, and the same bytes give the same corpus forever.
Account status has no such fixed input. It is a live judgement that only ever
accumulates, so asking the source again returns a different answer, and a
larger one: games an earlier run accepted would disappear from a later run of
the identical configuration, and from any evaluation pool built on it.

So the answer is taken once and pinned. Refreshing a snapshot is a deliberate
act that starts a new corpus, exactly as changing an archive digest is.

A snapshot names the archives it covers and speaks for every account appearing
anywhere in them, so raising a game bound or selecting another speed within a
covered archive needs no new snapshot. Preparation refuses an archive outside
that set rather than preparing it unfiltered, so widening the corpus fails
loudly instead of silently keeping the accounts nobody asked about.

Each snapshot is built for one archive, and nothing here widens one to a second.
Whatever does will have to carry every earlier verdict over untouched and query
only genuinely new accounts, because re-deciding an account an earlier snapshot
already spoke for applies a later moderation decision retroactively and drops
games a previous pool generation contains. Until then a second archive gets a
second snapshot, and the command refuses to write one over the other.

Usernames are stored as truncated salted digests. Membership is all preparation
needs, and a digest serves it as well as a name does, so this repository does
not carry a readable list of real people labelled as rule breakers. It is worth
being exact about what that buys: the salt is public and the account space is
the covered archive's, so anyone holding that archive can recover the names.
The mark is the source's own published judgement rather than a finding of this
project's, and the digests keep the repository from republishing it in a form
anyone can read or search.
