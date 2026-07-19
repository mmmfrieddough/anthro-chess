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
5. Encode static game settings.
6. Encode dynamic clock features when available, plus phase features.
7. Build one training timestep per ply.

Board generation should use exact chess rules. The neural model should not need
to infer the current board from raw notation.

The initial typed per-ply encoding contract implements this boundary for
standard chess and exposes a stable serialized identity for downstream
compatibility checks. It preserves exact pre-move state, prior and target
actions, legal-action alignment, rating context, and timing missingness without
making PGN or UCI text model inputs. The dataloading layer packs these values
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

The initial loader supports full games and contiguous chunks, groups sequences
into configurable length buckets, pads only within the current batch, and emits
separate padding, action-loss, nullable-context, and causal-attention masks.
Deterministic ordering is derived from an explicit seed and epoch. A
serializable dataset identity plus next-batch cursor permits exact continuation
without preserving opaque worker state.

The initial action-only model consumes that ordinary loader boundary through
`anthro_chess.models`, preserving its explicit targets, legal actions,
missingness, padding, and causal masks during tensor conversion. The shared
masked action objective lives in `anthro_chess.training`; deterministic
structural and tiny-overfit checks exercise those same model and loss APIs.

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

## Training Evaluation

Training should report a compact default set of validation metrics and preserve
deeper diagnostics for regressions.

Default validation metrics should include move loss, timing loss when timing is
enabled, rating-sliced validation metrics, and illegal-move mask penalty.

See `docs/evaluation.md` for the full evaluation design, including legality,
rating calibration, timing rollouts, human-likeness metrics, and
preference-control evaluation.

## Inference Loop

When it is the bot's turn:

1. Update exact game state from all moves so far.
2. Build current timestep features:
   - board before the bot move;
   - previous move;
   - current clocks and previous move times when timing is enabled;
   - static game settings.
3. Run the model with the causal KV cache.
4. Mask illegal moves while preserving enabled non-move actions such as
   resignation.
5. Sample a valid action using temperature.
6. If timing is enabled, condition the time head on the sampled action and
   sample move time from that action-conditional time distribution.
7. If timing is enabled, set `submit_at = received_at + sampled_time_ms`.
8. If timing is enabled, wait until the current time is at or after `submit_at`.
9. Submit the move or game-ending action and update local game state.

## Runtime Requirements

The runtime must:

- never submit illegal moves;
- support resignation as a valid game-ending action when enabled;
- support untimed play;
- respect clocks when timing is enabled;
- sample timing plausibly rather than always moving at fixed intervals when
  timing is enabled;
- expose configuration for target rating, temperature, optional preferences, and
  optional clock settings.

## Runtime Interfaces

The Anthro runtime should have a direct internal API that can be used by several
frontends. UCI should be the default compatibility interface for local chess
GUIs and engine tools, but UCI should remain an interface layer over the runtime
rather than a model input format.

In UCI mode, the engine process should load the runtime and model directly.
Keep standard output reserved for UCI messages and send ordinary logs to files
or standard error.

See `docs/interfaces.md` for the UCI and native-interface design.

## Hardware Assumptions

The project should favor architectures that can be trained on high-end consumer
GPUs. Main experiments should ideally be practical in days, with a larger final
run possibly taking weeks.
