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

This stage should establish the package structure, entry points, test runner,
development tooling, configuration conventions, and issue-tracking workflow. It
should also keep the docs and decisions easy for future agents to follow.

Detailed setup tasks should move into GitHub issues once they are actionable.

### 1. Minimal Training Loop

Prove that the project can train a model correctly from real chess data.

This stage should include small checked-in fixtures, a bounded reproducible
many-game Lichess corpus, deterministic board-state reconstruction,
model-facing encodings, dataloading, a basic model, move-prediction loss,
training configuration, validation metrics, and practical checkpoint/resume
support.

Start with the simplest training target that proves the loop works. Timing data
should be preserved in the data pipeline where available, but Milestone 1 keeps
timing inputs and outputs out of the model and training objective. Timing
behavior is added after the move-only path is useful and measurable.

Before scaling the first run, establish the staged training-correctness
protocol in `docs/training-and-runtime.md`: inspect fixed model inputs, prove a
stripped-down deterministic path can overfit a tiny sample, exercise causal
sequence behavior, and then demonstrate held-out signal above simple
baselines. Add context and optimization features from that trusted baseline
rather than introducing several unverified changes at once.

Prefer full-game, length-bucketed batches for the first proof so training
matches full-history inference while limiting padding waste. Tune
bucket-specific batch sizes, gradient accumulation, and GPU headroom from
measured model memory and throughput rather than hardcoding a speculative
device budget in the data layer. Treat independent midgame chunks as a
scalability option that must be compared with full-history training before it
becomes a default.

The output of this milestone does not need to be a strong chess opponent or
provide the final playable interface. It does need to demonstrate learned move
structure beyond randomness made legal by masking, including held-out
performance above appropriate simple move-selection baselines. That checkpoint
should be credible input to the playable proof rather than only evidence that
the training command runs.

### 2. Playable Proof

Connect model output to an actual playable chess game.

This stage should build the basic inference runtime: exact game-state updates,
legal move generation, illegal-move masking, action sampling, configuration,
and local runtime APIs. It should also connect the runtime to a real chessboard
experience.

UCI is the preferred compatibility path for local chess GUIs. The first UCI
implementation only needs enough polish to play games reliably, keep stdout
reserved for protocol messages, and expose the core options that already exist.

The goal is a proof that the model can choose legal moves in a real game loop
and begin to play somewhat sensible chess. It is acceptable for the model to be
weak at this stage.

### 3. Evaluation Harness

Turn evaluation into a first-class project system early.

Basic validation metrics should exist during the minimal training loop, but
this stage should make evaluation coherent enough to compare model versions
without relying on subjective playtesting.

This stage builds the instrument; it does not yet take readings anyone acts on.
Model iteration starts in stage 4, so no benchmark history accumulated here is
protected and no checkpoint produced here is worth preserving. Breaking a
comparability series, bumping the preprocessing version, changing the action
vocabulary, and regenerating the corpus are all free during this stage, and
work should not be deferred or resequenced to avoid them. Batching an expensive
corpus regeneration is still worthwhile, but that is an argument about compute
rather than about history. The comparability machinery itself is built to full
strength anyway, because it has to be trustworthy before the first reading that
matters. See `docs/decisions/0013-benchmark-result-comparability.md`.

This stage also establishes the evaluation-data contract the later benchmarks
share: a `test` partition training never consumes, one frozen pool drawn from
it, and derived views each benchmark selects through. Game-level opening
classification from an owned versioned book lands here too, because rollout
distribution comparison needs family aggregation; the per-ply multi-label form
for preference conditioning stays in stage 5.

Alongside the data contract, this stage builds the result infrastructure the
benchmarks share: a durable results store benchmarks append to, a metric
registry with stable identities and declared directions, an artifact envelope
carrying provenance, and per-series comparability fingerprints. Reports and
charts are views over the store rather than one-off comparisons of files, which
is what makes checkpoint history queryable rather than reconstructed. Noise
characterization belongs here too, and training-noise characterization
specifically should happen while runs are still short, because it only becomes
harder to afford later. This work should land before the individual benchmarks
so they do not each invent an incompatible result shape.

Live training observability arrives in this stage as well, so a run in progress
can be followed without waiting for it to finish.

The initial harness should emphasize reusable benchmarks that can run against
future models:

- held-out move prediction metrics;
- rating-sliced validation metrics;
- dependency tests showing whether conditioning inputs change behavior in the
  intended direction;
- illegal-move probability and legal-mask diagnostics;
- controlled training-efficiency comparisons using active positions per second,
  memory, and quality versus processed positions and wall-clock time;
- batch-one end-to-end move latency and declared-batch inference throughput,
  with cold-start time reported separately;
- generated-game rollout checks once the runtime can play.

Several of these benchmarks share one shape: measure a quantity on generated
games and on human games, and compare the two against rating. Opening
repertoire, book depth, game length, results, repetition, and termination all
fit it, so the comparison machinery is built once rather than four times.
`docs/evaluation.md` owns that shape and the estimation constraints it carries.

How games end becomes measurable in this stage. Deriving termination categories
during preprocessing turns resignation from an unreachable vocabulary slot into
a learned action with real labels, and adds a draw-claim action so untimed games
have a terminator that is not a hardcoded move limit. Both change the action
vocabulary identity, so they land as one bump rather than two, to pay the
corpus regeneration once. See
`docs/decisions/0017-derived-termination-and-terminal-actions.md`.

This stage should establish whether the model uses its conditioning inputs,
whether behavior shifts across settings, and what strength those settings
actually produce: the transfer function from configured rating to fitted
empirical rating, and its temperature response.

Rating measurement belongs here rather than with the scaling work. The earlier
argument was that it needs a model worth measuring, but that is circular: this
benchmark is the instrument that establishes whether a model is worth
measuring, and deferring it means the first reading lands on an already-scaled
checkpoint with no baseline to compare against. A degenerate result on a weak
checkpoint is a reportable outcome rather than an error, and it is not a
calibration verdict. Anchoring the resulting scale against an external engine
stays in the next stage, since it needs an external binary and answers a
different question.

Human-likeness evaluation beyond simple distribution metrics belongs later in
the process. A compact human-vs-engine classifier can be useful once the model
can generate coherent games, but it should not block the basic bot or become a
separate anti-cheat project.

Timing diagnostics arrive with timing itself, in the stage that adds it.

### 4. Scale And Improve

Use the working loop and evaluation harness to improve the bot.

The first work in this stage is the move-only model itself: expand data scale,
tune sampling and weighting, improve model capacity, strengthen checkpointing
and reproducible runs, and improve runtime reliability. Iterating here is the
point of having built the harness first, and it should continue until the
benchmark surfaces stop moving.

Training moves to a multi-GPU CUDA host in this stage, and using every card on a
long run is the ordinary execution path rather than an optimization to justify
later. Single-device CUDA still lands first, because it is the foundation the
distributed path shares and the baseline its scaling efficiency is read against,
but distribution does not wait for a demonstrated single-device limit. Short
independent jobs may still take one card each; that is a scheduling choice.
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

The data work has an order that later comparisons depend on. Training selection
becomes a filterable dial over one broad corpus first, so the value of adding a
data source can be measured against a single evaluation reference instead of
against two incomparable ones. The corpus then widens across the axes the
project intends to keep measuring, sized by evaluation power rather than
training volume. Cutting the resulting pool generation is the point at which the
long-lived evaluation core is designated and benchmarks begin reporting against
both it and the growing current pool. Scaling volume within those axes follows,
with the earlier baseline re-scored against the new reference so the whole arc
stays on one comparable scale.

This ordering is why the core is not frozen during stage 3. A reference
designated against the narrow first corpus could never measure the axes added
later, and there is almost no benchmark history to protect before this stage
begins.

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

**Timed-game breadth belongs to this stage and cannot slip.** Time control and
timing-data presence are among the axes the corpus widens across, and the
evaluation core is designated from the result. A core holding no timed games
across speeds could never measure timing behavior, for the life of the project,
so this half is irreversible and lands with the breadth pass rather than with
the feature it serves. Benchmarks slice by speed from that point on.

**Conditioning the policy on time control comes last in this stage**, after the
move-only path is useful on its own. It is what releases training selection to
widen across speeds, and staging it after the corpus, the pool, and benchmark
slicing have already widened gives the change a before-and-after reading instead
of bundling it with the timing feature. See `docs/data.md` and
`docs/design-principles.md`, which uses this case as its worked example of the
pattern.

**The move-time head itself moves to stage 5.** It is a second output head, a
second masked loss, its own diagnostics, and clock handling at the UCI boundary
— a feature rather than a scaling step. `docs/data.md` also places useful timing
conditioning at a corpus scale beyond where this stage's depth pass lands.
Timing diagnostics arrive with it.

Corpus-scale training reads through bounded-memory shard-backed loading rather
than the fixture-oriented eager per-ply materialization, which is what makes a
pass over the prepared million-game selection possible at all. Both loaders
stay, because a fixture or a proof slice is cheaper read eagerly; `docs/data.md`
owns which applies where.

Evaluation should guide this work. New data, model changes, and runtime changes
should be judged by the same benchmark surfaces whenever possible, with deeper
diagnostics available when a regression or surprising result appears.

### 5. Late Controllability

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
