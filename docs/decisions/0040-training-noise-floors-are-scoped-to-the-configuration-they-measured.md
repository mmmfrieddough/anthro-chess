# 0040: A Training Noise Floor Is Scoped To The Configuration Its Replicates Shared

Date: 2026-08-08

## Status

Accepted. Applies the shape of
`0025-machine-scoped-execution-noise-floors.md` to a second noise kind, and
lifts the prohibition in `0029-model-change-control-arm.md` that stood in for
it.

## Context

Two rules that are each correct met in a place neither of them was written for.

Decision 0013 keeps how a measurement was produced out of series identity, and
decisions 0018 and 0021 keep it out on purpose for the axes a report is supposed
to attribute a delta to rather than refuse it. A delta between models of
different size on one pool is interpretable, so nothing about the training run
reaches the fingerprint.

A noise floor is stored under that same fingerprint, which is what makes it
invalidate when the pool moves instead of lingering as a stale constant. For a
training floor the intersection of those two rules is a floor with no scope:
seed variance measured on four short arms of a 276k-parameter configuration is
resolved for **every** later delta on the same pool and view, including one
between a converged run of a model thirty times larger and its own baseline.

This is the defect decision 0025 fixed for execution floors, in a second kind,
and its sentence transfers unchanged: an execution floor without its execution
would be applied to every machine that ever recorded the series, which is the
one thing that kind must not do.

Decision 0029 met it as a live obstacle rather than as a hypothetical. It
defines how a model change shows it improved something, and a training floor is
what that process needs when a delta is too narrow to attribute to the change
rather than to seed luck; the floor it measured on those four arms came out 2.1
to 7.3x the data-sampling floor of the same metrics, which is the difference
between qualifying a claim and not. What it could not do was store one, so it
instructed that a characterized training floor is read beside its comparison and
not committed. That is a workaround whose enforcement is a reader's memory, and
it costs the floor exactly where a stored one is worth most: on the next change
tested against the same base.

`#223` settled that the floor system survives, so this had to be answered rather
than closed with the estimator.

## Decision

A training noise floor records the training configuration its replicates shared,
and it is valid only within it.

### The Scope Is The Configuration, Not The Run

The identity is the checkpoint's compatibility record — the split
`docs/training-and-runtime.md` already draws between the settings a resumed run
must match and the ones that merely describe a run — digested without the
initialization seed. Everything that decides what the optimizer does to the
weights is in it: model, corpus, arithmetic, encoding, action vocabulary. The
seed is out because it is the axis the floor measures; leaving it in would give
every replicate its own scope and no floor could ever resolve.

The split is reused rather than reinvented because it is already maintained. A
setting added to what a continuation must match becomes part of what a floor
describes on the same commit, which is the only version of this that stays true
without anybody remembering to keep two lists aligned.

### A Reading Carries It, And A Report Matches Both Sides

Every result records the training identity of the checkpoint it scored, beside
the parameter digest that names the exact weights. A report resolves a training
floor only where that identity matches the characterization on **both** operands
of the delta.

A reading that carries no identity — recorded before this existed, or taken
through a runner supplied rather than loaded — matches nothing, so its noise is
reported as unknown rather than qualified by a borrowed floor. The same answer
covers a delta whose two sides were trained under different configurations,
which is the case the process in 0029 is most likely to produce by accident.

Characterization refuses rather than guesses: replicates that do not all record
one identity are not one configuration's seed variance, and the command says so
instead of storing a floor whose scope nobody could state.

### The Prohibition In 0029 Is Lifted

A characterized training floor is committed to the store like any other. The
reason it was withheld was that a stored one had no scope; it has one now, and
the recorded floor is resolved for the configuration it measured and for nothing
else.

## Consequences

A training floor is now worth characterizing once per configuration rather than
once per comparison, which is what makes 0029's control arm cheaper the second
time it is used. The reading in that record — four arms, half an hour of one
host at that scale — becomes a stored asset instead of a number in a pull
request body.

The cost is that a floor does not travel, and travels less than an execution
floor does. Every change to the model, the corpus, or the arithmetic starts a
new scope with no floor in it, and a converged baseline will need its own. That
is honest, and it is the same trade 0025 accepted: a new machine reports unknown
noise until it characterizes its own.

Two axes are deliberately outside the scope, and a reader qualifying a delta
should know which.

**Where it ran.** Backend, device, and thread count are provenance rather than
compatibility, so a floor characterized on one host resolves on another. Seed
variance is a property of what was optimized rather than of the accelerator that
optimized it, and the arithmetic settings that would change the answer —
precision, matmul precision, determinism — are inside the identity because a
continuation must match them.

**How far it trained.** The configured step budget is not in the identity,
because with a constant learning rate the weights at a given step do not depend
on how many more were configured. What the scope therefore does not distinguish
is the *maturity* of the arms: a floor characterized at 8,000 steps resolves for
a delta between checkpoints near a plateau, where seed variance is smaller.
Nothing here detects that, the same way nothing detects an execution floor going
stale as a machine's state moves; the characterization records what its
replicates were and re-characterizing is a matter of naming different
checkpoints.

Results recorded before this identity existed keep their floors of every other
kind and lose only the training one, which no committed record carried.

## References

- `docs/evaluation.md`
- `docs/training-and-runtime.md`
- `docs/decisions/0018-workload-scoped-efficiency-series.md`
- `docs/decisions/0021-efficiency-identity-excludes-compared-conditions.md`
- `docs/decisions/0025-machine-scoped-execution-noise-floors.md`
- `docs/decisions/0029-model-change-control-arm.md`
