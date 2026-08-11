# 0055: A Missing Artifact Fails Its Step, Not The Plan

Date: 2026-08-11

## Status

Accepted.

## Context

Two places said the sweep refuses a pool that is not there before it spends any
time: `docs/evaluation.md` under "The Benchmark Suite", and the module docstring
of `anthro_chess.evaluation.suite`. Neither was true. `resolve_suite` validates
the selection files, the benchmark names, the overrides against their schemas,
the ordering, and the game-dependency constraint, and touches the filesystem for
none of the artifacts a step will measure over; `roots.resolve_artifact_roots`
only rewrites paths. A plan built with `pool = /absolutely/not/here` resolves
every step and starts.

The claim was load-bearing rather than decorative. `#234` — a missing puzzle
artifact ending the whole sweep instead of failing one step — proposed resolving
itself by checking that artifact "the way it already checks a pool", a symmetry
that had never existed. A false statement of behavior had become the precedent
for a design.

So the repair was a choice rather than a typo fix: either the two statements go,
or the check they describe gets built.

## Decision

**Plan resolution reads the selections, not the artifacts they name.** An
artifact that is not on this host is found by the step that reads it, fails that
step, and leaves the rest of the sweep running. What resolution keeps is what
the configuration alone settles, and each of those makes a sweep *wrong* rather
than partial, so there is no reading to lose by refusing it.

### A Sweep Delivers What It Can

`#234` already settled this for the puzzle artifact, and `#255` implemented it:
the load moved inside the try that converts that benchmark's failures, so a host
without the pinned artifact gets one failed step and every other reading instead
of nothing.

Checking artifacts at plan time reverses that resolution and generalizes the
reversal. The hosts this decides are real ones: every artifact a step reads is
built rather than checked in, and lives outside the worktree beneath a root a
machine may not have at all, while the reduced sweep is the default way a new
checkpoint is read. Refusing a sweep that would return seven readings of eight
is the more expensive failure, and the ledger, the skip rule for dependent
steps, and per-step failure conversion are already the machinery that makes the
cheaper one work.

### A Plan-Time Check Is Weaker Than It Sounds

Existence when the plan resolved is not readability when the step runs. A pool
that was there an hour ago can be truncated, unreadable, or a different
generation than the one a selection pinned, and the step's own loader is the
real check in every case. The plan-time check would therefore not remove a
failure mode; it would add a second, earlier, and weaker place artifacts are
validated, whose agreement with the first nothing enforces.

## Consequences

**`#254` closes without new machinery.** A test pins that a plan resolves against
an artifact that is not there, so building the check later means arguing with
this record rather than leaving a doc to go quietly stale again.

**One artifact still escapes, and `#252` is where it is fixed.** The novelty
step's leakage check raises `LeakageError`, which that benchmark's registry entry
does not list, so a host missing the training corpus that check reads still ends
the whole sweep. This record is what makes that a defect rather than a second
policy, and until it closes the behavior described here is the pool's and the
puzzle set's rather than every artifact's.

## References

- `#254` — the claim this record answers
- `#234`, `#255` — the puzzle artifact, and the resolution this follows
- `#252` — the remaining escape
- `docs/evaluation.md` — "The Benchmark Suite"
- `src/anthro_chess/evaluation/suite.py` — `resolve_suite`, `_run_step`
- `src/anthro_chess/evaluation/roots.py` — the declared artifact fields
