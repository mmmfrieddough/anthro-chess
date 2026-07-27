# Architecture

The preferred architecture is a hybrid symbolic-neural chess model.

Symbolic chess logic computes exact board state and legal moves. The neural
model learns human move preference, time usage, and controllable behavior from
encoded state and game context.

## High-Level Shape

For each ply `t`:

```text
exact board before move_t = ChessRules(moves_0 ... moves_t-1)
board_embedding_t         = BoardEncoder(exact board before move_t)
previous_move_embedding   = MoveEmbedding(move_t-1)
dynamic_embedding_t       = Phase features, plus clock features when available
trajectory_settings       = Preferences and clock settings when enabled

x_t = combine(
  board_embedding_t,
  previous_move_embedding,
  dynamic_embedding_t,
  trajectory_settings
)

CausalTransformer(x_0 ... x_t) -> h_t
decision_conditioner(h_t, target_rating_t) -> d_t
action_head(d_t) -> action_t
time_head(d_t, action_embedding_t) -> optional move_time_t
```

The board state is not learned from the move sequence. It is computed exactly
from the game history and then embedded by a learned board encoder.

When timing is enabled, action and move time should be treated as a dependent
structured output rather than two unrelated samples. The preferred
factorization is:

```text
p(action_t, move_time_t | context_t)
  = p(action_t | context_t) * p(move_time_t | context_t, action_t)
```

This keeps the shared sequence representation responsible for the chess
context, while letting the time distribution depend on the actual action that
will be submitted.

## Runtime Layering

The implementation should keep a small number of conceptual layers distinct.
These layers do not need to be separate processes. In the normal case they
should be modules in one application process, with clear APIs between them.

Recommended layers:

- model definition: neural network modules and learned parameters;
- model runner: checkpoint loading, device placement, tensor construction,
  cached inference, and conversion from raw model outputs into runtime-facing
  distributions;
- chess logic: exact board reconstruction, rule bookkeeping, legal move
  generation, move encoding, and move-string conversion;
- decision runtime: game-session state, configuration, legal masking, action
  sampling, optional timing behavior, and local game updates;
- interfaces: UCI, native CLI, web UI, benchmark harnesses, or any other way
  outside callers interact with the runtime.

The decision runtime is the core application boundary. Interfaces should call
the runtime. The runtime may call the model runner and chess logic. The model
itself should remain protocol-agnostic and should not know whether a request
came from UCI, a native UI, or an evaluation benchmark.

Chess logic is worth treating as its own conceptual layer even if runtime code
uses it constantly. It is deterministic infrastructure shared by training,
inference, evaluation, and interfaces. Keeping it separate avoids mixing exact
rule enforcement with learned behavior or protocol handling.

## Board State

The system should compute exact board state before each ply. The model input may
represent that state compactly rather than exposing every rule field as a named
feature.

The computed state must contain enough information for legal move generation,
including:

- pieces and side to move;
- castling rights;
- en-passant state;
- draw-rule counters if used;
- any other state required for complete move generation.

A simple board encoder may use square and piece embeddings followed by pooling,
an MLP, or another small learned encoder. Larger encoders can be considered if
simple encodings are insufficient.

## Sequence Model

The preferred sequence model is a causal transformer with one timestep per ply.

The transformer sees:

- compact board embeddings;
- previous move embeddings;
- current clock and phase features when timing is enabled;
- trajectory metadata that genuinely helps interpret the history;
- historical context through causal attention.

The current action model deliberately keeps target rating outside that causal
history encoder. After the transformer has produced a rating-neutral feature
for each position, a small nonlinear feature-modulation network combines that
feature with the rating of the player choosing that move. The action head still
has learned computation after rating enters; rating is not merely added to the
final probabilities.

The current checkpoint-backed model runner implements that correctness
baseline. It converts one typed target-free decision trajectory into the shared
model tensor boundary, applies the controlled player's rating only to the final
decision, recomputes the complete causal history, and returns detached raw
action logits to the decision runtime.

The current game-session runtime owns the canonical board and complete observed
move history, builds that shared context, masks actions against exact legal
moves, and applies the selected move. Its strict settings keep target rating
and temperature independent and leave resignation disabled unless a caller
deliberately enables it.

Selection also reports what the policy said about the action it applied. This
is the single place an action is chosen, so exposing those quantities here is
what lets generated benchmark games and games reconstructed from live play be
analyzed by one code path, and it keeps a rollout benchmark measuring the
policy the engine actually plays rather than a second implementation of it.
Interfaces that only need the move keep using the thin call.

Runtime state has several different lifetimes. Checkpoint loading, device
placement, and other expensive model-runner initialization should survive for
the process lifetime. Exact board and move history should advance
incrementally when a caller supplies an extension of the known game, while an
unrelated position, takeback, or divergent history should invalidate only what
can no longer be reused and fall back to atomic exact replacement. Encoded
history and future transformer key-value caches should follow the same
common-prefix boundary. Caching is not allowed to weaken support for arbitrary
valid positions and should be implemented only where its correctness and
measured value justify the complexity.

The sampling generator has a game lifetime and must not be reset as a side
effect of synchronizing a position. Greedy temperature-zero selection is
deterministic. Nonzero-temperature interactive play uses fresh game randomness
by default, while an explicit seed makes games and benchmarks reproducible.
See
[`0010-separate-position-sync-from-randomness.md`](decisions/0010-separate-position-sync-from-randomness.md).

Training should make full use of the causal attention mask. A complete game, or
a chunk of a game, can be fed to the transformer at once so all ply predictions
are trained in parallel while each timestep only attends to prior timesteps.

The current initial implementation follows this shape with a learned compact
board encoder, a per-ply context combiner, an action-only causal transformer,
and a head over the shared action vocabulary. Its tensor boundary,
hyperparameter schema, compatibility identity, and model definition live in
`anthro_chess.models`.

## Decision, Static, And Dynamic Metadata

Decision controls include:

- target rating on the project's internal rating scale for the player choosing
  the current action;
- temperature, which remains a runtime sampling control rather than a learned
  history feature.

Static game settings may include:

- starting clock time when timing is enabled;
- increment when timing is enabled;
- optional preference settings.

The current implementation applies the optional target rating only at the
decision layer for the corresponding supervised ply. It does not place either
player's rating in historical timestep features and does not need a controlled
color input: the exact board already identifies whose turn it is. Training can
therefore encode each game once and learn from every valid ply, selecting the
mover's rating for that ply when it is available. At runtime the caller supplies
only Anthro's chosen target rating when Anthro is making a decision; no opponent
rating is required.

The rationale and accepted tradeoffs are recorded in
`docs/decisions/0009-decision-only-rating-conditioning.md`.

This placement is intentionally specific to rating. Other sequence-wide values
should not be moved mechanically. Clock settings can change how every observed
time should be interpreted, and a future preference control may describe the
whole requested trajectory, so their placement should follow their semantics.
Missing values must remain explicit wherever a feature is placed.

Dynamic features include:

- bot remaining clock time when timing is enabled;
- opponent remaining clock time when timing is enabled;
- previous move times when available;
- move number or game phase;
- explicit markers for untimed play or missing dynamic fields.

Dynamic metadata must be represented per ply because it changes throughout the
game.

### Possible Rating-Aware History Reader

Late decision conditioning means the causal transformer analyzes the position
and history without knowing the requested strength. That is the simplest
contract and preserves a single rating-neutral history pass, but it may prove
too shallow if rating needs to change which historical patterns receive
attention rather than only how the completed representation is used.

If evaluation shows that limitation, test a small rating-conditioned query or
cross-attention reader over the rating-neutral causal states before the action
head. This would add rating-aware learned processing without putting ratings on
past moves, exposing an opponent rating, or duplicating the transformer pass.
It is an experiment to measure during performance tuning, not part of the
current architecture.

## Action Output

The action head should output logits over a fixed action vocabulary. Most
actions are UCI-style chess moves. The vocabulary should also include a
resignation action.

Before sampling, the runtime must mask all illegal moves using exact chess
logic. If the sampled action is a move, it must always be legal in the current
position.

Resignation and draw claims are not board moves and are not produced by legal
move generation. Runtime should treat them as valid game-ending actions when
enabled by the application or benchmark context.

The two are masked differently. Resignation is always available to the side to
move, so enabling it is purely a runtime policy. A draw claim is available only
when the repetition or fifty-move condition already holds, which exact chess
logic computes from the board and history. Neither introduces game state outside
the board, which is why offering and accepting draws is excluded: a pending
offer would.

Outside protocols may support only part of the action vocabulary. For example,
standard UCI requires a `bestmove` response and provides no portable
engine-to-GUI response for resigning or claiming a draw. Protocol adapters
should translate the subset they can represent without changing the model's
native action space.

## Optional Time Output

When timing is enabled, the time head should predict a distribution over move
time rather than a single average scalar. Untimed games should not require
timing inputs or a model-sampled move delay.

The time head should condition on the chosen action as well as the shared
context state. This avoids forcing the runtime to sample action and time
independently from marginal distributions that may not agree. For example, the
same position may support both an automatic recapture and a difficult quiet
move, and those actions may have different plausible timing distributions.

The time distribution should be directly sampleable and trained with the
sampling cross-entropy approach against the observed move time. Do not model
move time as fixed bucket classification. Runtime samples from the learned
distribution and converts the sample into an exact millisecond delay.
