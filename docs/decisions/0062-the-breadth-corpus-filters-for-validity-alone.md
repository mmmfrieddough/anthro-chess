# 0062: The Breadth Corpus Filters For Validity Alone

Date: 2026-08-13

## Status

Accepted. Settles the axes half of `#89` and names what the corpus `#90`
designates its core from is filtered by. Applies
`0016-sampling-axes-versus-measured-distributions.md` to preparation rather than
to sampling, and leaves the one exception
`0041-games-of-marked-accounts-leave-the-corpus.md` carved for it untouched.

## Context

`#90` fixes the evaluation core's per-axis statistical power permanently. An
axis the corpus holds no games for can never be measured afterwards, and an axis
it holds a *distorted* population for is worse than one it is missing, because
the distortion is not visible from inside the measurement.

That second half is the reason this record exists. A preparation filter runs
before split assignment, so it removes games from every split at once. It shifts
the training data and the evaluation reference in the same direction, and a
benchmark cannot detect a bias it shares with its own reference — the model
matches because both were shaped by the same filter. A training-selection filter
touches only the train split, so the same distortion reads as a mismatch against
a clean reference and stays reversible.

A preparation filter is therefore a definitional claim about what counts as
human play for every benchmark this project will ever run.

## The axes

Four, and the first two are required rather than weighed:

- **Speed**, which
  `0056-the-speed-axis-is-derived-from-the-time-control.md` takes off the
  source's own label and onto the control it was played at. Time control
  genuinely changes opening choice and game length, so it is a conditioning axis
  for benchmarks and not only a coverage one.
- **Timing-data presence.** The move-time head has no reference to be measured
  against unless the corpus carries clocked games across speeds.
- **Rating**, which the policy already conditions on.
- **Temporal spread**, which is the 42 months the span holds.

## Decision

**Preparation filters for validity and nothing else.** The selection accepts
rated standard games with both players rated and neither a bot, and rejects
nothing else. Every axis above is left whole and narrowed downstream: by the
training-selection dial `#88` built, and by the pool's own admission fraction in
`0052-a-bounded-pool-is-a-fixed-admission-fraction.md`.

Three settings follow from that and are worth stating because each was
available and declined.

**No speed filter.** The corpus carries every class. Training selection isolates
one until the policy reads the clock, which is a load-time property of a run
rather than of the artifact.

**No accepted-game bound.** A bound counts accepted games in source order, so it
would truncate the span from its most recent end and take the temporal axis with
it. The pool is bounded instead, as a fraction of the split rather than a count
of the corpus.

**The ply floor is left at the schema's own default**, rather than carried over
from the baseline selection in `configs/data/lichess-blitz-2017-04.toml`, which
sets a higher one. Nothing about this selection changes — it declared no filters
before — so what is decided here is the declining. Game
length is an axis benchmarks measure the model's own distribution over, and
`0016` closes exactly that class of axis to reshaping — so a floor here is the
blind spot above rather than a validity rule. Measured on 2018-01 over 400,000
rated standard games with both ratings, a floor of eight drops **0.851%** of
accepted games and a floor of one drops **0.007%**. Benchmarks that need depth
declare it in their own view, where the rejection is counted and visible.

The remaining 0.007% is not an arbitrary residue. All 27 zero-move games in that
sample are `1-0` with a `Normal` termination, which is Black resigning before
White has moved; none is `0-1`, consistent with the player on move being able to
abort instead. `0017-derived-termination-and-terminal-actions.md` excludes a
resignation made on the opponent's clock from the action sequence, having no
decision point to attach it to. A zero-move game therefore normalizes to an
empty action sequence — no plies, no actions, nothing to train or score on. One
ply is the boundary below which a record carries no action at all, not a
judgement about which games are worth keeping.

**The marked-account rejection is not applied here, and is not thereby
abandoned.** `0041` still binds before the core is designated. What changed is
that it no longer has to be answered while the PGN is being read: the normalized
row carries both players' account digests, truncated from the same salted hash a
snapshot stores. Setting `marked_accounts` on this run would pin whatever recall
the census had reached on the day the corpus was built, into a run that cannot
be revisited without re-parsing 42 archives —
`0047-account-status-is-censused-continuously-and-claims-a-partial-recall.md`
puts that reading at designation, by whoever designates it, and this keeps it
there.

## Consequences

**Coverage is reported per axis and per split.** The manifest counts accepted
games by speed and by clock presence, and player-slots by rating, each split
three ways. What a held-out reading can resolve on an axis is then readable
before a pool is cut from it rather than after the generation that fixes it.

**The rating buckets are finer than any band a benchmark slices on**, and named
by their range, so a reader re-adds them into whatever banding it uses;
`anthro_chess.data.prepare` owns the width. The manifest deliberately does not
pin one benchmark's current banding into every corpus ever prepared, and the
layering would not allow it anyway: `evaluation` imports `data`, never the
reverse.

They count player-slots rather than games, because the readings this axis has to
size are per-slot: a sliced move loss is realized in the plies whose mover falls
in the band, not in games where both players do.

**Speed is banded off the normalized clock**, which is how a pool and a training
selection read it, so the manifest reports the axis its consumers will see. That
costs one distinction: an unlimited control reaches those columns as an
unavailable clock, indistinguishable from a control the source never gave, so
correspondence games are counted as unclassified rather than as their own class.
The header derivation `0056` uses does separate them, and the two disagree on
exactly that population.

**Temporal coverage gets no axis of its own.** The manifest already records one
entry per archive with its own split counts, and an archive is a month, so the
temporal crosstab is already there. Adding a second one would be the same
numbers at a different address.

**A row-level marked-account filter has to exist before `#90`.** Nothing reads
the digest columns today — preparation writes them and there is no consumer. If
that filter is not built, the rejection `0041` requires never happens at all,
and after designation it cannot. This is the one obligation this record creates,
and `#466` carries it as a blocker of `#90` rather than as a sentence here,
because a prose obligation is one nothing schedules.

**Games this corpus will not hold, it will never hold.** Casual games, unrated
players, bots, and variants are out permanently for the core, since containment
forbids a later generation from being anything but a superset.

## References

- `#89`, `#90`
- `docs/data.md` (Corpus Expansion, Sampling And Weighting)
- `docs/decisions/0016-sampling-axes-versus-measured-distributions.md`
- `docs/decisions/0041-games-of-marked-accounts-leave-the-corpus.md`
- `docs/decisions/0045-centisecond-clocks-from-a-closed-export.md`
- `docs/decisions/0052-a-bounded-pool-is-a-fixed-admission-fraction.md`
- `configs/data/lichess-univ-2018-01-2021-06.toml`
