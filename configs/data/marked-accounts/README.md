# Marked-Account Snapshots

Each file here records which accounts a source had marked for breaking its
rules at one moment. Every game those accounts played is rejected: by
preparation when a source selection names the snapshot, and otherwise when the
evaluation pool's generation is cut, which is where a corpus prepared without
one applies it.
`uv run anthro data mark-accounts` cuts one from the census;
`docs/decisions/0041-games-of-marked-accounts-leave-the-corpus.md` owns why the
games go rather than the moves, and why the label is pinned rather than queried.

These are pinned inputs rather than derived artifacts, which is why they are
checked in beside the selection that names them. The answer is taken once and
pinned: refreshing a snapshot is a deliberate act that starts a new corpus,
exactly as changing an archive digest is.

A snapshot names the archives it covers and refuses any other, so raising a game
bound or selecting another speed within a covered archive needs no new snapshot
while widening the corpus past them fails loudly instead of silently keeping the
accounts nobody asked about.

Within those archives it speaks partially, and its header says how partially. A
listed account is marked; an unlisted one was either answered for and clean or
never asked about, and its games are kept either way. The census behind
a snapshot asks in descending order of games played and has no finish line, so
the header carries the share of accounts and of player-slots it had reached, and
`docs/decisions/0047-account-status-is-censused-continuously-and-claims-a-partial-recall.md`
owns what that share does and does not claim.

Nothing widens a snapshot in place. A later census answers for more accounts and
re-answers for none, but a snapshot cut from it rejects games this one keeps, so
it starts a new corpus rather than amending this one — `mark-accounts` refuses
to write over an existing snapshot for that reason.

Usernames are stored as truncated salted digests. Membership is all preparation
needs, and a digest serves it as well as a name does, so this repository does
not carry a readable list of real people labelled as rule breakers. It is worth
being exact about what that buys: the salt is public and the account space is
the covered archives', so anyone holding them can recover the names.
The mark is the source's own published judgement rather than a finding of this
project's, and the digests keep the repository from republishing it in a form
anyone can read or search.
