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
static_game_embedding     = Rating/preference settings, plus clock settings when enabled

x_t = combine(
  board_embedding_t,
  previous_move_embedding,
  dynamic_embedding_t,
  static_game_embedding
)

CausalTransformer(x_0 ... x_t) -> move_t, optional time_t
```

The board state is not learned from the move sequence. It is computed exactly
from the game history and then embedded by a learned board encoder.

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
- static game metadata;
- historical context through causal attention.

Live inference should use a key-value cache so only the newest timestep needs to
be processed after each move.

Training should make full use of the causal attention mask. A complete game, or
a chunk of a game, can be fed to the transformer at once so all ply predictions
are trained in parallel while each timestep only attends to prior timesteps.

## Static And Dynamic Metadata

Static game settings include:

- target rating;
- starting clock time when timing is enabled;
- increment when timing is enabled;
- bot color;
- optional preference settings.

The current preference is to broadcast a small `static_game_embedding` at every
timestep. This is cheap and more reliable than depending only on a prefix token.

Dynamic features include:

- bot remaining clock time when timing is enabled;
- opponent remaining clock time when timing is enabled;
- previous move times when available;
- move number or game phase;
- any explicit marker needed to indicate untimed play.

Dynamic metadata must be represented per ply because it changes throughout the
game.

## Move Output

The move head should output logits over a fixed move vocabulary, likely based on
UCI-style move tokens.

Before sampling, the runtime must mask all illegal moves using exact chess
logic. The sampled move must always be legal in the current position.

## Optional Time Output

When timing is enabled, the time head should predict a distribution over move
time rather than a single average scalar. Untimed games should not require
timing inputs or a model-sampled move delay.

The time distribution should be directly sampleable and trained with the
sampling cross-entropy approach against the observed move time. Do not model
move time as fixed bucket classification. Runtime samples from the learned
distribution and converts the sample into an exact millisecond delay.
