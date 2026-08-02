# Training And Runtime

The training approach is supervised learning from human games.

The model should learn move choice from data. When timing is enabled, it should
also learn move timing from clock data. Exact chess logic handles state
reconstruction and legal-output filtering.

## Training Data

See `docs/data.md` for the data philosophy, source survey, normalized schema,
storage format, scale estimates, and sampling strategy. This section focuses on
how that data becomes training examples.

Useful game records should include:

- moves;
- game termination information, including resignation when available;
- normalized player ratings when available on the project rating scale;
- source ratings for provenance and possible later normalization;
- player identity when intentionally modeling individual or famous-player style;
- starting clock time and increment when timing is available;
- clock states or move times when timing is available;
- enough metadata to derive bot color and game phase.

Clock precision below one second is preferred for realistic fast-game behavior.
Centisecond or millisecond precision would be better if available.

Opening and preference labels are not required for the base move model, but
derived labels may be useful for optional preference controls. These labels
should be treated as training metadata rather than fixed product requirements.

## Preprocessing

For each game:

1. Parse the move list.
2. Reconstruct exact board state before each ply.
3. Generate legal moves for each position.
4. Encode the previous move.
5. Select the side-to-move player's optional normalized rating as the decision
   target rating for that ply, without adding it to historical timestep inputs.
6. Encode dynamic clock features when available, plus phase features.
7. Build one training timestep per ply.

Board generation should use exact chess rules. The neural model should not need
to infer the current board from raw notation.

The typed per-ply encoding contract implements this boundary for
standard chess and exposes a stable serialized identity for downstream
compatibility checks. It preserves exact pre-move state, prior and target
actions, legal-action alignment, per-decision target rating, and timing
missingness without making PGN or UCI text model inputs. Normalized data keeps
both source-player ratings for provenance and selects only the mover's rating
at each supervised ply. Historical timestep inputs contain neither player's
rating. The dataloading layer encodes each game once, retains both players'
moves, and enables action loss on every valid ply. It then packs these values
into framework-neutral numeric sequence batches so model code can make the
final tensor/device conversion without reopening normalized data or
reconstructing alignment.

Optional preference labels should be allowed to be multi-label. A single ply may
belong to several useful concepts, such as an opening family, a pawn structure,
an attacking pattern, or a material-sacrifice pattern. These labels should be
attached at the ply or position-window level when possible rather than copied
blindly across an entire game.

Opening-family labels should usually apply only while the opening or related
structure is active.

Other preference labels should be designed case by case. Some concepts may be
mostly structural, such as fianchetto setups or closed centers. Others may need
event-style definitions, such as sacrifices or early piece development. Timing
style labels may come from clock usage and time-pressure context. Player-style
labels may come from games by a specific well-represented player. The goal is to
create useful contrast sets for learning or discovering behavior controls, not
to build a perfect chess taxonomy.

## Sequence Training

Training should use full game sequences, or fixed-length chunks of game
sequences, whenever practical. The transformer should receive one timestep per
ply and use a causal attention mask so every ply prediction can be trained in a
single parallel forward pass while still preventing access to future moves.

The loader supports full games and contiguous chunks, groups sequences into
configurable length buckets, pads only within the current batch, and emits
separate padding, action-loss, nullable-context, and causal-attention masks.
Deterministic ordering is derived from an explicit seed and epoch. A
serializable dataset identity plus next-batch cursor permits exact continuation
without preserving opaque worker state.

A training selection chooses between an eager loader and a bounded-memory
shard-backed one, and the choice belongs to the selection rather than to the
run: a corpus-scale train split and a small validation split can sit in one
configuration and read through different loaders. `docs/data.md` owns what each
holds, how the shard-backed epoch is ordered, and why the two are not
interchangeable mid-run. What matters here is that the model, loss, runner,
checkpoint format, and resume contract are the same either way, and that the
run record names which loader read each selection, because a training curve is
not comparable across two different epoch orders.

The initial action-only model consumes that ordinary loader boundary through
`anthro_chess.models`, preserving its explicit targets, legal actions,
rating missingness, padding, and causal masks during tensor conversion. Timing
fields remain preserved in the loader output but are deliberately outside the
Milestone 1 model boundary and objective. The shared masked action objective
lives in `anthro_chess.training`; deterministic structural and tiny-overfit
checks exercise those same model and loss APIs.

The shared training runner also lives in `anthro_chess.training` and uses those
ordinary loader, model, loss, and validation boundaries. It resolves explicit
`cpu`, `mps`, or `auto` device selection at one boundary and moves the model,
batches, loss, validation, and optimizer state through that resolved device.
Explicit MPS selection never silently falls back. Strict determinism remains
available for the stripped-down CPU correctness path, while ordinary
accelerator work can select the performance-oriented mode. Run artifacts own
the exact device, precision, determinism, accumulation, throughput, phase
timing, and available memory measurements rather than duplicating those values
in prose. The same runner writes atomic optimizer-step checkpoints and can
continue from the latest checkpoint in a run or an explicitly selected
checkpoint across supported backends.

This is compatible with exact board reconstruction because the board state for
each ply can be computed before training and included in that ply's input
embedding. It is also compatible with teacher forcing: previous moves are known
from the human game record during training, while the model predicts the current
move and optional move time at each timestep.

When timing is enabled, the time target should be trained conditionally on the
observed human action for that ply:

```text
action_loss = -log p(human_action | context)
time_loss   = -log p(human_time | context, human_action)
```

At inference, the runtime samples a legal action first, then samples move time
conditioned on that selected action. This is a small structured-output
factorization. It avoids treating move choice and timing as independent outputs
while preserving efficient full-sequence training with teacher forcing.

This creates the ordinary train/inference distinction that the time head sees
human actions during training and sampled model actions during inference. That
is acceptable for the initial design because the dataset only gives a true time
target for the action the human actually played. Evaluation should watch for
move-time incoherence in generated games rather than trying to invent timing
targets for actions humans did not choose.

## Losses

The core losses are:

- action cross-entropy over the fixed action vocabulary;
- sampling cross-entropy for the sampleable move-time distribution when timing
  is enabled.

Each target should have an explicit loss mask. Missing metadata should not be
turned into fake training targets. For example, games without clock data can
still train the action head, but they should contribute no time-loss term. Games
with clock data should feed that clock context into the shared model and action
head as well as the time head, because available time can affect human move
choice.

Illegal move masking is required at inference. Training may also use legal-move
awareness depending on the final implementation.

Other useful losses may include:

- auxiliary value or phase prediction;
- calibration losses for timing, when timing is enabled, or skill control.

Mask penalty can be considered as an auxiliary legality loss if evaluation shows
the model gives too much probability to illegal moves before runtime masking.
It should not be part of the initial core loss.

Optional preference-control mechanisms should be learned or derived from data,
not implemented as hardcoded post-processing rules over move logits.

## Training Correctness Protocol

Neural training can continue without an obvious failure even when examples,
targets, masks, dependencies, or optimization are wired incorrectly. Establish
the first training path in stages so each layer has a trusted result before
more complexity is introduced.

The initial model and training runner should support one small deterministic
correctness path through the same package APIs used for ordinary training. This
is a debug selection of the real pipeline, not a parallel model or training
implementation.

Use this progression when establishing the first training loop:

1. Decode and inspect a fixed batch at the model boundary. Confirm that boards,
   target actions, previous actions, legal actions, nullable context, padding,
   and loss masks describe the intended positions.
2. Use a fixed seed and the simplest supported optimization setup. Keep timing,
   auxiliary losses, regularization, dropout, learning-rate schedules,
   mixed-precision behavior, and concurrent loading disabled unless the item is
   the subject of the check.
3. Check structural invariants before relying on optimization. Initial losses
   should be finite and plausible for the output space, padding should
   contribute no loss, and perturbing future timesteps should not affect
   earlier predictions.
4. Overfit a fixed tiny sample. Begin with single-timestep examples when useful
   for isolation, then repeat with short causal sequences so the attention and
   sequence-loss path is exercised.
5. Evaluate on frozen held-out examples against uniform-over-legal and other
   appropriate simple move-selection baselines. Memorizing a tiny sample is
   necessary evidence that the optimization path works, but it is not evidence
   that the inputs contain transferable chess signal.
6. Introduce additional context or training features in coherent steps. Compare
   each candidate with the last trusted configuration. Confirm its intended
   effect, and retain complexity only when its behavior is understood and its
   product value or measured benefit justifies its cost.

Exact board reconstruction, the shared action codec, legal-action alignment,
and explicit masks remain enabled throughout this progression. They are
correctness boundaries, not optional model features.

Record enough information to reproduce each result, including the resolved
configuration, seed, data and encoding identities, and relevant metrics. Exact
debug values belong in code, tests, or checked-in configuration rather than in
this document.

The first completed progression, its reproducible commands, and measured
evidence are recorded in `docs/planning/minimal-training-proof.md`.

The complete progression is a gate for the first training implementation and
for later changes to foundational data, encoding, alignment, model, or loss
contracts. Routine model changes should run the relevant deterministic checks
and compare frozen validation metrics. Changes outside the training path do not
need to repeat the full progression.

## Checkpointing And Resume

Training should be designed for practical fine-grained resume. Long runs should
not depend on reaching an epoch boundary before useful state is saved.

The goal is bounded lost work, usually measured in minutes or a small number of
optimizer steps, not perfect bit-exact replay in every circumstance. Exact
replay is useful where it comes naturally, but the project should not distort
the data pipeline or training architecture just to reproduce every random
choice after interruption.

Checkpoints should be keyed by optimizer step rather than epoch. A complete
training checkpoint should include, where applicable:

- model weights;
- optimizer state;
- scheduler state;
- mixed-precision scaler state;
- global step;
- consumed example, token, or ply counts when useful;
- training config and code/data version metadata;
- random number generator state where practical;
- dataloader or sampler position where practical.

Checkpoint cadence should support both step-based and time-based policies, such
as every fixed number of optimizer steps or every fixed number of minutes. The
latest checkpoint should be enough to resume normal training without waiting
for epoch-sized units of work.

The dataloader should prefer restartable sampling designs. Avoid relying on one
large opaque shuffle whose exact internal state is hard to recover. Better
patterns include deterministic shard ordering from a seed and pass id,
deterministic within-shard shuffle, explicit shard and offset cursors, or
sampling recipes that can be reconstructed from global step, worker id, and
seed.

Resume should restore training behavior well enough that validation curves,
learning-rate schedules, and data coverage remain meaningful. If exact sample
order cannot be restored without significant complexity, resuming from a recent
checkpoint and continuing with an equivalent sampling recipe is acceptable.

The current action-only runner restores model and optimizer state, global
progress counters, Python and Torch random-number-generator state, and the
loader's exact deterministic next-batch cursor. Each checkpoint retains the
resolved configuration and code, data, model, action-vocabulary, encoding, and
execution provenance. Tensor state is loaded through CPU storage before the
model and optimizer restore it onto the selected CPU or MPS backend. Resume
compares code-owned compatibility identities before loading state and rejects
unsafe changes clearly.

The initial runner uses full precision. CPU supports strict or relaxed
determinism. The current MPS Transformer backward path requires relaxed
determinism because the locked Torch build lacks a deterministic implementation
for one required gradient operation; selecting strict MPS training fails before
optimization with a clear error. An MPS checkpoint preserves MPS RNG state.
When a CPU checkpoint starts an MPS continuation, the target-backend RNG is
initialized reproducibly from the run seed because the source checkpoint has no
MPS RNG state. Loader continuation remains exact, but stochastic model behavior
across different backends is not promised to be bit-identical.

The checkpoint configuration schema and artifact version in
`anthro_chess.training` are the source of truth for exact fields and selection
syntax.

### Shared Machine-Local Runs

Before a checkpoint is ready for public model hosting, complete training runs
may live in a shared machine-local directory outside repository worktrees.
Use `ANTHRO_CHESS_RUN_ROOT` for that location. The CLI maps configured relative
run paths beneath this root and maps configured relative dataset paths beneath
`ANTHRO_CHESS_DATA_ROOT`. An explicit absolute path or command-line path
override takes precedence. Resolved paths are retained in the run artifact.

Both roots are optional, because a fresh clone resolves checked-in relative
paths inside the worktree and several commands depend on that. What is not
optional is that the failure stays legible: a command needing a root it cannot
resolve names the variable and what it would have to hold, so a misconfigured
machine reports itself rather than presenting as one holding nothing.
`anthro machine` answers the same question ahead of the failure, and
`docs/interfaces.md` describes it.

A typical layout keeps datasets and runs as siblings:

```text
~/.local/share/anthro-chess/
  datasets/
  runs/
```

For example:

```console
export ANTHRO_CHESS_DATA_ROOT="$HOME/.local/share/anthro-chess/datasets"
export ANTHRO_CHESS_RUN_ROOT="$HOME/.local/share/anthro-chess/runs"
```

Retain the complete run directory as one artifact:

```text
<run-name>/
  run.json
  metrics.jsonl
  tensorboard/
    events.out.tfevents...
  checkpoints/
    latest.json
    step-........pt
```

Do not preserve only a copied weight file. The run record, metrics, interval
checkpoints, compatibility identities, and latest pointer are needed for
comparison, resume, provenance, and later runtime selection. A model runner can
accept an explicit compatible checkpoint path beneath this root without copying
it or requiring an external model registry.

The run record carries the training selection the run resolved as well as the
corpus it read, because two runs over one corpus can differ only in what they
selected from it. `docs/data.md` owns what that selection is and why it is a
load-time dial.

The metrics stream carries more than one kind of record, each labelled, because
a cadence reading and an optimizer step do not arrive on the same schedule and
should not be forced into one row. A reader selects the kind it wants rather
than inferring it from which fields happen to be present.

Reported throughput describes training and excludes the time a cadence spent
measuring, which is reported beside it. Wall-clock elapsed time is reported
unchanged. Without that separation a run would appear several times slower for
no reason other than that it evaluated itself on the way past.

Throughput is a **steady-state** figure over active non-padding positions, so
it is absent until the run leaves warmup rather than reported as a number that
drifts for the whole run. Startup, checkpoint writes, cadence evaluation, final
validation, and the run's own instrumentation are each timed and each excluded
from it. `anthro_chess.training.efficiency` owns the measurement and its
configuration; `docs/evaluation.md` owns how the results are read.

### Deferred Read-Back

A training step's reported quantities — loss, active positions, whether the
loss is finite — are accumulated **on device** and read back once per logging
interval, the way `anthro_chess.training.health` already handles optimizer
statistics. Reading any of them per micro-batch forces a device
synchronization, which stops the host from queueing the next step while the
device works on this one.

Two consequences are worth stating because neither is obvious.

A non-finite loss is detected at the next logging boundary rather than at the
exact step, so the failure names the interval it occurred in and up to one
interval of updates is applied from a corrupted gradient. The run fails either
way; what changes is the precision of the report, and `log_every_steps` sets
it.

An individual step no longer has a meaningful duration. Between two
synchronizations the host runs ahead, so a step returns when its work is queued
and the queue is paid off by whichever step synchronizes next. Only a
drain-bracketed **interval** is honestly timed, which is why the measurement's
unit is the interval and why the reported step-time spread is per interval.

How much the deferral is worth depends on which side is the bottleneck, so the
run measures it rather than assuming: a configured share of intervals runs the
synchronizing arm, interleaved with the deferred one, and the difference is
reported. On one Apple Silicon MPS run it was worth about 1.5 ms of a 39.8 ms
step without gradient accumulation, and nothing measurable at four accumulation
micro-batches, where the device is busy enough that the host never gets ahead.

The same rule governs the offline scoring pass and the per-move inference path,
and there it is not a tradeoff at all. Batch validation, the finite-logit
checks, and each position's identity and conditioning are all questions whose
answers arrive in a transfer those paths were already making, so asking the
device separately buys nothing and blocks the queue once per check, per
position, or per generated move. Every one of them is read across in a batched
transfer instead, with no check weakened. This is invisible on CPU, where a
synchronization costs nothing, so it is enforced by tests that fail when a
device tensor is evaluated in boolean context rather than left to review.

Bulk benchmark diagnostics are machine-local for the same reason and default
beneath this root. `ANTHRO_CHESS_RESULT_DETAIL_ROOT` overrides that location
when detail should live elsewhere. Committed benchmark summaries are separate
and live in the repository; see `docs/evaluation.md`.

Some checkpoints are **anchors**: the ones a long-running comparison starts from,
and the ones re-scored when a new evaluation pool generation is cut. Anchors are
retained rather than cleaned up, because losing one removes the left edge of
every chart that begins there. Anchors are a handful per milestone rather than
every interval checkpoint, and they stay where runs already live; retention is a
policy, not a separate storage system. See
`docs/decisions/0014-evaluation-result-storage.md`.

Retention is affordable partly because a lost checkpoint costs a re-run rather
than an unreconstructable result. That holds only while runs remain reproducible
from configuration, seed, and a corpus regenerable from its pinned source digest,
which makes run reproducibility a property worth actively protecting.

The current model runner resolves either an authoritative explicit checkpoint,
an explicit retained run plus checkpoint selector, or an intentionally
maintained default selection beneath the run root. Resolution never guesses from
file names or modification times. The strict configuration and selection-record
schemas in `anthro_chess.inference` own the exact names and precedence. Every
selection retains and reports the absolute run record and checkpoint paths.

That default selection is deliberate, so a command maintains it rather than a
process inferring it: `anthro model select` writes the record, and the same
validation a runner performs runs before it is written, so a selection that
would fail to load is refused where it is made instead of at the next process
start.

The runner resolves its own device selection, separate from the training one
because the two accept different backends: inference and every benchmark
resolving through it accept CUDA, and training does not yet. `auto` takes an
available accelerator and otherwise CPU, while an explicit accelerator fails
rather than falling back, and the failure distinguishes a Torch build without
that backend from a host without such a device. The two are different problems
with different fixes, and a message conflating them would send a reader to the
wrong one. `anthro_chess.inference` owns the accepted set.

Because the whole evaluation suite is inference-bound rather than
training-bound, that selection is what decides what a sweep costs on a given
machine. What a benchmark measured is reported with the device it ran on, so a
reading taken on an accelerator is never silently compared against one taken on
CPU; `docs/evaluation.md` owns those comparability rules.

Loading validates the run and checkpoint model configuration, action
vocabulary, model-facing encoding, decision-only rating contract, and supported
timing shape before restoring weights. The runner loads checkpoints through CPU
storage, places the model on the resolved device, recomputes the full
target-free history for each request, and exposes only the current decision's
detached CPU action logits. It does not import training-loop orchestration,
invent an opponent rating, add a cache, mask actions, sample moves, or expose a
timing output.

Machine-local retention is not a release. Publishing selected inference
artifacts through Hugging Face or another registry remains a later decision
once checkpoint quality, packaging, and public compatibility expectations are
stable.

## Training Evaluation

Training should report a compact default set of validation metrics and preserve
deeper diagnostics for regressions.

Default validation metrics should include move loss, timing loss when timing is
enabled, rating-sliced validation metrics, and illegal-move mask penalty.

Evaluation is not only a post-training step. Measurements run at declared
cadences during a run so a bad run is visible early: cheap health signals every
optimizer step, held-out previews at validation intervals, fuller diagnostics
less often, and the canonical suite at the end. In-training previews use the
`validation` split, never the held-out test pool, so early readings cannot
influence checkpoint selection. `docs/evaluation.md` owns the cadence model and
the rules that keep previews interpretable against the canonical reading.

A run declares its schedule in its own configuration, and the whole schedule is
resolved before the first optimizer step so an unaffordable or impossible entry
fails immediately. Readings go to the results store as ordinary benchmark
results and to the metrics stream beside the training curve; the canonical
end-of-run reading over the frozen test pool remains a separate command against
a retained checkpoint. A cadence changes what a run reports and not what it
trains, so it stays out of the checkpoint compatibility record and a resumed run
may schedule differently than the run it continues.

### Training Observability

Every training run emits TensorBoard output by default. Following a run should
require nothing beyond opening TensorBoard against the run root: no flag to
enable, no separate configuration step, and no need to ask for it when
dispatching work. Because runs already live in per-run directories, pointing
TensorBoard at the root overlays a running experiment against previous runs at
the same step, which is the main thing it is there for.

With `ANTHRO_CHESS_RUN_ROOT` set to the shared runs directory, launch the view
from the project environment:

```console
uv run tensorboard --logdir "$ANTHRO_CHESS_RUN_ROOT"
```

Each run writes beneath its own `tensorboard/` directory. TensorBoard discovers
those directories recursively, so one process follows the current run and
overlays earlier runs by optimizer step.

Event files are a derived view rather than a source of truth. The metrics stream
and the evaluation results store stay authoritative, and event files remain
regenerable and safe to delete. Training curves and in-training evaluation
readings belong there; cross-version project history does not, because
TensorBoard has no notion of which results are comparable and would happily
overlay series that are not. See
`docs/decisions/0014-evaluation-result-storage.md`.

See `docs/evaluation.md` for the full evaluation design, including legality,
rating calibration, timing rollouts, human-likeness metrics, and
preference-control evaluation.

## Operational Logging

Process entry points configure Python's standard logging once, and package
modules use module-named loggers for lifecycle and failure context. Native
commands send operational logs to standard error while preserving requested
command results on standard output. Their global `--log-level` option controls
verbosity.

Training progress now uses logging, but metrics, run records, checkpoints,
manifests, evaluation results, and future complete generated-game traces remain
versioned artifacts rather than log records. Default logging records aggregate
counts, selected paths, device and lifecycle state; it excludes raw corpus
records, player identities, complete move histories, full model distributions,
and secrets.

UCI uses bounded rotating file logs with a standard-error fallback. Its
destination and verbosity controls are described in `docs/interfaces.md`; the
logging module remains the source of truth for exact destination and rotation
defaults. UCI's DEBUG tier is the narrow exception to the general move-history
exclusion: it records versioned accepted-position snapshots and engine
decisions so live games can be reconstructed after replacements and takebacks.
These diagnostic events are not benchmark artifacts and do not change default
logging volume.

## Inference Loop

When it is the bot's turn:

1. Update exact game state from all moves so far.
2. Build target-free full-history features through the shared model-facing
   context builder, ending with the current timestep:
   - board before the bot move;
   - previous move;
   - current clocks and previous move times when timing is enabled.
3. Run the rating-neutral causal history encoder, then condition the current
   decision feature on Anthro's one configured target rating. Add a causal KV
   cache later only if measured runtime performance requires it. `anthro eval
   inference` is the measurement that answers that, and it reports latency
   against history depth, which is where recomputation shows up.
4. Mask illegal moves while preserving the enabled terminal actions the
   position allows.
5. Sample a valid action using temperature.
6. If timing is enabled, condition the time head on the sampled action and
   sample move time from that action-conditional time distribution.
7. If timing is enabled, set `submit_at = received_at + sampled_time_ms`.
8. If timing is enabled, wait until the current time is at or after `submit_at`.
9. Submit the move or game-ending action and update local game state.

## Runtime Requirements

The runtime must:

- never submit illegal moves;
- support resignation and draw claims as valid game-ending actions when
  enabled;
- support untimed play;
- respect clocks when timing is enabled;
- sample timing plausibly rather than always moving at fixed intervals when
  timing is enabled;
- expose configuration for target rating, temperature, optional preferences, and
  optional clock settings.

The current `anthro_chess.runtime` implementation provides the untimed core of
this boundary. A game session owns exact board state and full move history,
accepts observed legal moves from either player, applies Anthro's target rating
only to its current decision context, masks and samples model-runner logits,
and updates the game with the selected valid action. Strict code-owned settings
define rating, temperature, the random-seed policy, and whether each terminal
action is enabled. Timing and preference settings remain future additions
rather than placeholder inputs or fabricated clock values.

A session can also score an enabled action without deciding anything, leaving the
board and the random stream untouched. That is one call rather than a second
selection path on purpose: the decisions of a game the runtime did not originate
are recovered through the same code that would have made them, so a re-scored
decision and a live one are the same measurement. `docs/evaluation.md` describes
what reads it.

A session separates advancing the game from establishing randomness. Only a new
game — session construction, `reset`, or a `ucinewgame` boundary — begins a new
random stream from the seed policy; synchronizing to a new position never
reseeds or rewinds the active stream. An unset seed draws fresh operating-system
entropy per game, so ordinary nonzero-temperature play varies and independent
processes stay independent, while an explicit non-negative seed reproduces a game
for debugging and benchmarks. Temperature zero stays greedy and
seed-independent. See `docs/decisions/0010-separate-position-sync-from-randomness.md`.

## Runtime Interfaces

The Anthro runtime should have a direct internal API that can be used by several
frontends. UCI should be the default compatibility interface for local chess
GUIs and engine tools, but UCI should remain an interface layer over the runtime
rather than a model input format.

In UCI mode, the engine process should load the runtime and model directly.
Keep standard output reserved for UCI messages and send ordinary logs to the
bounded application-log destination or its standard-error fallback.

See `docs/interfaces.md` for the UCI and native-interface design.

## Hardware Assumptions

The project should favor architectures that can be trained on high-end consumer
GPUs. Main experiments should ideally be practical in days, with a larger final
run possibly taking weeks.
