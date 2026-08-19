# Architecture

The preferred architecture is a hybrid symbolic-neural chess model.

Symbolic chess logic computes exact board state and legal moves. The neural
model learns human move preference, time usage, and controllable behavior from
encoded state and game context.

## High-Level Shape

For each ply `t`:

```text
exact board before move_t = ChessRules(moves_0 ... moves_t-1)
history_t                 = the n boards ending at that one, each from the
                            side to move when it was current
position_features_t       = Rule state, repetition, phase features, the colour
                            to move, plus clock features when available
trajectory_settings       = Preferences and clock settings when enabled

squares_t = SquareTokens(
  history_t,
  position_features_t,
  target_rating_t,
  trajectory_settings
)                                             -> 64 tokens

SpatialLayers(squares_t) -> decision_t        -> 64 tokens

action_head(decision_t) -> action_t
time_head(decision_t, action_embedding_t) -> optional move_time_t
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

The board is presented to the model as one token per square rather than as a
single pooled vector, so board geometry survives into the network instead of
being flattened away before the first layer. Attention between those tokens
carries a learned geometric bias generated from the position itself, which is
what lets the relations that matter in a given position be attended along. Its
bank of square-relation templates belongs to the whole model rather than to any
one layer.
`docs/decisions/0066-the-trunk-sees-the-rating-and-the-board-keeps-its-shape.md`
records why the board keeps its shape, and
`docs/decisions/0070-one-decision-per-pass-and-history-in-the-token-depth.md`
why the model reads one decision at a time.

Every board is presented from the side to move, mirrored so the player choosing
is always the one playing up the board. Which colour that player is survives as
its own input, because a repertoire as white is not the mirror of a repertoire
as black.

## Decision Model

One forward pass is one decision. History is the depth of each square token
rather than an axis of its own: the last several boards are stacked into the
token, and differencing consecutive boards recovers what moved, so no separate
previous-move input is carried.

The model sees, for the decision it is asked about:

- that position's 64 square tokens, each carrying its own square across the
  stacked boards;
- how often each of those boards had already occurred, which is what a draw
  claim turns on;
- castling rights, the en-passant square, and the halfmove clock;
- the colour the deciding player is playing;
- the target rating, which is part of the representation before layer zero;
- current clock and phase features when timing is enabled.

Target rating is embedded into the square tokens before any layer runs, so
every stage of the network computes with it. It is placed by interpolating
between a learned weak anchor and a learned strong anchor, which makes the
representation monotone in the rating by construction rather than leaving the
ordering to be discovered.

Each decision reads one rating, its own mover's, in training and at runtime
alike. Nothing is required from the caller but Anthro's own target rating, and
no controlled-color input is needed because the exact board already identifies
the side to move.

The current checkpoint-backed model runner implements that correctness
baseline. It converts one typed target-free decision trajectory into the shared
model tensor boundary, names the ply being decided, and returns detached raw
action logits to the decision runtime. What that costs does not grow with the
game: the decision reads the boards stacked behind it and nothing earlier.

It also accepts several pending decisions at once, padding the shorter
histories past their end so each reads its own decision at its own length. A
live game has one decision to make and never uses this; a caller holding many
independent games, such as a generated-game benchmark, is the reason it exists.
The batched surface is declared separately from the single-decision one, so a
runner that offers only the latter stays a valid runner.

The current game-session runtime owns the canonical board and complete observed
move history, holds the context encoded for it, masks actions against exact
legal moves, and applies the selected move. Its strict settings keep target
rating and temperature independent and leave resignation disabled unless a
caller deliberately enables it.

An encoded timestep depends only on the root position and the moves before it,
so the session keeps the encoding it already has and builds only the timesteps
a position update actually added. That makes per-decision encoding a function
of new plies rather than of game length, and it is why validating an incoming
history and building the context the model reads are one pass rather than two.

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
can no longer be reused and fall back to atomic exact replacement. Caching is
not allowed to weaken support for arbitrary valid positions and should be
implemented only where its correctness and measured value justify the
complexity.

Encoded history reuses that boundary today. Caching model state across
decisions does not arise: a decision reads a fixed window of boards rather than
the game so far, so there is no growing prefix to hold.

The sampling generator has a game lifetime and must not be reset as a side
effect of synchronizing a position. Greedy temperature-zero selection is
deterministic. Nonzero-temperature interactive play uses fresh game randomness
by default, while an explicit seed makes games and benchmarks reproducible.
See
[`0010-separate-position-sync-from-randomness.md`](decisions/0010-separate-position-sync-from-randomness.md).

A complete game, or a chunk of one, is still fed to the model at once, because
every ply of it is a supervised decision and batching them costs one pass. What
travels between those decisions is nothing: each reads its own stacked boards.

The current implementation follows this shape with a square-token encoder over
the stacked history, a stack of spatial layers reading the geometric bias, and a
source-destination head over the shared action vocabulary. Its tensor boundary,
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

The current implementation supplies the optional target rating for the
supervised ply's own mover, and reads it from the first layer onward. Training
encodes each game once and learns from every valid ply, selecting the mover's
rating for that ply when it is available. At runtime the caller supplies only
Anthro's chosen target rating when Anthro is making a decision; no opponent
rating is required.

The rationale and accepted tradeoffs are recorded in
`docs/decisions/0009-decision-only-rating-conditioning.md`, and where the rating
enters is settled by
`docs/decisions/0066-the-trunk-sees-the-rating-and-the-board-keeps-its-shape.md`.

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

## Action Output

The action head should output logits over a fixed action vocabulary. Most
actions are UCI-style chess moves. The vocabulary also includes two terminal
actions, resignation and a draw claim, which end a game instead of changing the
position.

A move's logit is produced as an attention from its source square against its
destination square, with a per-destination bias selecting the promotion piece,
so the head is expressed in the same board geometry the encoder produced. The
terminal actions carry no move and are scored from the history feature instead.
That is the head's internal structure; the fixed vocabulary above remains the
contract every caller sees, and a gather built from the vocabulary itself joins
the two.

Before sampling, the runtime must mask all illegal moves using exact chess
logic. If the sampled action is a move, it must always be legal in the current
position.

Resignation and draw claims are not board moves and are not produced by legal
move generation. Runtime should treat them as valid game-ending actions when
enabled by the application or benchmark context.

The two are masked differently. Resignation is always available to the side to
move, so enabling it is purely a runtime policy. A draw claim is available only
when the repetition or fifty-move condition already holds in the position
itself, which exact chess logic computes from the board and history. The rules
also allow claiming alongside an announced move that would create the
condition; that form is deliberately outside the action, which carries no move,
so enabling a claim never offers one the action could not honestly express.
Neither introduces game state outside the board, which is why offering and
accepting draws is excluded: a pending offer would.

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
