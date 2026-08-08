# Roadmap

This document is for implementation planning. It describes how the project might
move toward the end state, not what the end state is.

The main project docs describe the intended end state:

- `docs/vision.md`
- `docs/design-principles.md`
- `docs/architecture.md`
- `docs/engine-behavior.md`
- `docs/data.md`
- `docs/training-and-runtime.md`
- `docs/interfaces.md`
- `docs/evaluation.md`

Roadmap items are allowed to change as the project develops. When the roadmap
and the end-state docs disagree, the end-state docs are authoritative unless we
explicitly decide to update them.

## Planning Principles

- Keep roadmap stages broad enough to survive implementation changes.
- Use the roadmap to organize work, not to define product requirements.
- Move detailed task lists into GitHub issues when they become actionable.
- Update the roadmap when the rough order changes.

## Milestone Shape

The roadmap should be organized around proofs of progress. Each milestone
should leave the project in a state that is more usable or more measurable than
before.

### 0. Project Setup

Make the repository ready for implementation work.

Closed. Package structure, entry points, test runner, tooling, configuration
conventions, and the issue workflow.

### 1. Minimal Training Loop

Prove that the project can train a model correctly from real chess data.

Closed. Fixtures, a bounded reproducible Lichess corpus, deterministic
board-state reconstruction, model-facing encodings, dataloading, a model,
move-prediction loss, and checkpoint/resume support.
`docs/training-and-runtime.md` owns the staged training-correctness protocol
this established and the acceptance form the result was read against.

Two sequencing decisions from this stage still hold. Timing data is preserved
in the pipeline wherever a source carries it, but timing inputs and outputs stay
out of the model and the objective until stage 6, after the move-only path is
useful and measurable. And independent midgame chunks remain a scalability
option that has to be compared against full-history training before becoming a
default, rather than a default nobody compared.

### 2. Playable Proof

Connect model output to an actual playable chess game.

Closed. The inference runtime — exact game-state updates, legal move
generation, illegal-move masking, action sampling — and a UCI process a real
GUI plays against. `docs/interfaces.md` and
`docs/decisions/0006-direct-uci-invocation.md` own what that path is and why it
is invoked directly.

### 3. Evaluation Harness

Turn evaluation into a first-class project system early, so model versions can
be compared without relying on subjective playtesting.

Closed. The evaluation-data contract, the results store and metric registry,
comparability fingerprints, live training observability, and the first
generation of benchmarks. `docs/evaluation.md` owns the current set and the
shape several of them share;
`docs/decisions/0013-benchmark-result-comparability.md` and
`docs/decisions/0017-derived-termination-and-terminal-actions.md` own the two
choices that changed an identity.

**Rating measurement belongs with the harness rather than with the scaling
work.** This argument is kept because it answers a position that was actually
held, and a re-sequencing proposal would otherwise re-litigate it from nothing.
That position was that rating measurement needs a model worth measuring, which
is circular: this benchmark is the instrument that establishes whether a model
is worth measuring, and deferring it means the first reading lands on an
already-scaled checkpoint with no baseline to read it against. A degenerate
result on a weak checkpoint is a reportable outcome rather than an error, and it
is not a calibration verdict. Anchoring the resulting scale against an external
engine is a different question needing an external binary, and stays later.

Three things were deferred out of this stage and have not arrived.
Human-likeness evaluation beyond distribution metrics — a compact
human-vs-engine classifier — becomes useful once the model generates coherent games, but should not block the
basic bot or turn into a separate anti-cheat project. Timing diagnostics arrive
with timing itself, in the stage that adds it. And the per-ply multi-label form
of opening classification stays in stage 6; the game-level form landed here only
because rollout distribution comparison needs family aggregation.

### 4. Efficient Loop And Trusted Readings

Make every experiment cheaper and every reading interpretable, before scaling
starts spending against them.

Stage 3 built the instrument. Reading it on real checkpoints, and profiling it
on the CUDA host, showed that it is not yet affordable enough or resolved enough
to guide the decisions stage 5 wants to make. None of that is scaling work, and
all of it sits upstream of scaling work: a capacity comparison read from a suite
whose cost is unknown, and whose readings mostly state no resolution, produces a
number nobody can act on.

Some of stage 3's freedom runs through this stage and ends inside it. No
benchmark history is protected and no checkpoint is worth preserving until the
evaluation core is designated, so breaking a comparability series, bumping the
preprocessing version, changing the action vocabulary, and regenerating the
corpus all stay cheap, and work should not be deferred or resequenced to avoid
them. Batching an expensive corpus regeneration is still worthwhile, but that is
an argument about compute rather than about history. Designation fixes the
core's per-axis statistical power permanently, which is why the corpus and pool
work that breaks containment belongs before it. See
`docs/decisions/0013-benchmark-result-comparability.md`.

This stage has two halves. The first makes an experiment cheaper. The training
step should stop spending longer building a batch than the device spends
computing it, which is a property of the loader's representation rather than of
model size and so is worth fixing at any capacity later selected. The evaluation
suite should stop paying for repeated work — pools materialized once per
benchmark, scoring passes it already holds the answer to, replicates that are
provably one game. Training moves to a multi-GPU CUDA host here, and using every
card on a long run is the ordinary execution path rather than an optimization to
justify later. Single-device CUDA lands first, because it is the foundation the
distributed path shares and the baseline its scaling efficiency is read against,
but distribution does not wait for a demonstrated single-device limit, and it is
measured at a width where the reading means something. Corpus-scale training
reads through bounded-memory shard-backed loading rather than fixture-oriented
eager materialization, which is what makes a pass over the prepared selection
possible at all. Both loaders stay, because a fixture or a proof slice is
cheaper read eagerly; `docs/data.md` owns which applies where.

The second half makes a reading interpretable. A benchmark should record what it
cost where it records what it measured, so a cost claim is reviewable the way a
metric delta is; cost figures asserted in configuration files drift silently and
have. The families the suite actually spends its time in should state a
resolution, or record that they cannot discriminate and what that means for
reading them. The shape of the noise-floor system is settled here against
measured resolution rather than extended by default.

Sample size, rather than the estimator, turned out to be the binding constraint
on resolution, and sample size is an efficiency problem. That is why the two
halves are one stage rather than two: the cheap lever on resolution is a view
size nobody could afford, and affording it is the first half's job.

This stage also defines how a change that alters the model shows it improved
something. The test suite proves a change did not break anything; nothing yet
says how to tell an improvement from run-to-run noise, so every model-affecting
change otherwise sets its own evidence standard. That definition should compose
the machinery this stage and stage 3 already built rather than add tooling.

This stage ends by freezing the evaluation reference, because that is the event
the disposability window closes at rather than a stage boundary. Deciding
engine-assisted filtering, widening the corpus across the measurement axes,
cutting the second pool generation, and designating the core all land here, in
that order: each breaks containment or ends a benchmark series, and all of them
are free until the designation and permanent after it.

Buying capacity is what belongs to stage 5 and is blocked on this one.

### 5. Scale And Improve

Use the working loop and evaluation harness to improve the bot.

The first work in this stage is the move-only model itself: expand data scale,
tune sampling and weighting, improve model capacity, strengthen checkpointing
and reproducible runs, and improve runtime reliability. Iterating here is the
point of having built the harness first, and it should continue until the
benchmark surfaces stop moving.

Distribution replicates the model rather than sharding it, so it buys throughput
and not capacity, and the per-card memory ceiling still bounds how large a model
this stage can select.

Before capacity is scaled, the current architecture should be trained until its
held-out metrics plateau. Every checkpoint the project has read so far stopped
on a step bound while still improving, so which resource is binding — capacity,
data, or budget — is not yet known. A budget answer costs one run at the current
size; a capacity answer costs several runs at larger sizes against a fixed
memory ceiling. Establishing the plateau first is both the cheaper question and
the control that makes the capacity comparison readable.

The data work that establishes the reference happens in stage 4, in an order
later comparisons depend on: training selection becomes a filterable dial over
one broad corpus, the corpus widens across the axes the project intends to keep
measuring, and cutting the resulting pool generation is the point at which the
long-lived evaluation core is designated and benchmarks begin reporting against
both it and the growing current pool. What remains here is scaling volume within
those axes, with the earlier baseline re-scored against the reference so the
whole arc stays on one comparable scale.

That ordering is why the core is not frozen during stage 3. A reference
designated against the narrow first corpus could never measure the axes added
later, and there is almost no benchmark history to protect before it exists.

Decisions that would remove games from the corpus belong before the widening
rather than after it. Expansion has to preserve containment, so a rejection
filter introduced later cannot be applied to a corpus an evaluation core has
already been designated from. Whether engine-assisted filtering acts on the
corpus or only on training selection is therefore settled ahead of the breadth
pass, even though implementing it may not be.

The transfer function from configured rating to played strength and its
temperature response are measured in stage 3, so what remains here is anchoring
that scale against a fixed external engine reference. It follows rather than
leads because it needs an external binary and because an anchor is only useful
once ordering already exists.

Timing is not one piece of work, and its parts have different deadlines. They
are split across this stage and the next rather than staged together.

**Timed-game breadth belongs to the stage-4 breadth pass and cannot slip.** Time
control and timing-data presence are among the axes the corpus widens across,
and the evaluation core is designated from the result. A core holding no timed
games across speeds could never measure timing behavior, for the life of the
project, so this half is irreversible and lands with the breadth pass rather
than with the feature it serves. Benchmarks slice by speed from that point on.

**Conditioning the policy on time control comes last in this stage**, after the
move-only path is useful on its own. It is what releases training selection to
widen across speeds, and staging it after the corpus, the pool, and benchmark
slicing have already widened gives the change a before-and-after reading instead
of bundling it with the timing feature. See `docs/data.md` and
`docs/design-principles.md`, which uses this case as its worked example of the
pattern.

**The move-time head itself moves to stage 6.** It is a second output head, a
second masked loss, its own diagnostics, and clock handling at the UCI boundary
— a feature rather than a scaling step. `docs/data.md` also places useful timing
conditioning at a corpus scale beyond where this stage's depth pass lands.
Timing diagnostics arrive with it.

Evaluation should guide this work. New data, model changes, and runtime changes
should be judged by the same benchmark surfaces whenever possible, with deeper
diagnostics available when a regression or surprising result appears.

### 6. Late Controllability

Add the move-time head and learned preference controls after the model, runtime,
and evaluation stack are clearly working.

Move-time prediction arrives here rather than with the scaling work. By this
point the corpus, the evaluation core, and benchmark slicing already cover
timing, and the policy already conditions on time control, so what remains is
the feature itself: an action-conditioned time head, a masked timing objective
for the games that carry usable clocks, the timing diagnostics, and clock
handling at the UCI boundary. See `docs/architecture.md` and
`docs/decisions/0003-action-conditioned-timing.md`. It precedes timing-style
preference controls, which have nothing to steer until it exists.

Preference controls for openings, style, timing style, player-inspired
tendencies, and other human-play concepts are important to the end state, but
they should not block the basic bot. This work should start once there is a
model whose behavior can be measured and played.

Possible late-stage preference-control work includes:

- deriving multi-label position metadata for opening families, structures,
  aggression, sacrifices, development, and other concepts;
- labeling positions or conservative position windows rather than blindly
  labeling whole games;
- using known-opening position matching and walking backward to the nearest
  useful opening or structure category;
- building matched contrast sets by rating, color, move number, and other
  relevant context;
- inspecting or steering model activations from those labeled sets;
- calibrating preference sliders so rating, legality, and position coherence
  remain intact.

Suggested order for the first learned preference controls:

1. Define a small opening-family taxonomy.
2. Build known-opening position matching and backward lookup.
3. Generate conservative per-ply opening-family labels.
4. Train or extract activation-difference steering directions for a few
   opening families.
5. Evaluate whether sliders increase the intended opening tendency without
   breaking rating calibration.
6. Add a few non-opening labels, starting with easy structural concepts.
7. Compare activation-difference steering against supervised vectors and sparse
   autoencoder features.
8. Add application sliders once the controls are stable enough to be useful.

See `docs/preference-controls.md` for the preference-control subsystem design.
See `docs/evaluation.md` for the evaluation design.
