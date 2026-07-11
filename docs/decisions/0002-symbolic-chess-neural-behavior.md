# 0002: Symbolic Chess Logic With Neural Behavior

Date: 2026-07-11

## Status

Accepted as initial design direction.

## Context

Anthro Chess needs to play legal chess while modeling human-like move choice,
timing, rating, and optional preferences. Some parts of chess are exact rules:
board reconstruction, legal move generation, check, castling rights,
en passant, promotion, move counters, and game-ending conditions. Other parts
are behavioral: which legal action a player would choose, how long they might
think, and how those choices vary by rating and settings.

The model should not spend capacity relearning deterministic chess state that
the runtime can compute exactly. At the same time, the model should learn
human-like behavior from human games rather than relying on hard-coded mistake
injection.

## Decision

Use a hybrid symbolic-neural architecture:

- deterministic chess logic computes exact game state, rule bookkeeping, and
  legal actions;
- model-facing encoders represent the current state and trajectory;
- a learned causal model predicts behavior from that state and history;
- runtime logic masks illegal model outputs before sampling.

The neural model owns behavioral prediction. The symbolic layer owns chess-rule
correctness.

## Consequences

This makes legality a runtime guarantee instead of a learned hope. It also
keeps model inputs grounded in exact state instead of raw interface protocols or
fragile text histories.

The model may still be evaluated on how much probability it assigns to illegal
moves before masking, but illegal moves should never be submitted by the
runtime.

Chess-rule changes, encodings, and legal masking need focused tests because
bugs in the symbolic layer can corrupt both training examples and runtime play.
