# Design Principles

These principles describe the current design bias for Anthro Chess. They should
guide implementation decisions, but they can evolve as testing reveals what
actually works.

## Compute The Board State Exactly

The model should not have to reconstruct the board from raw move history.
Deterministic chess logic should build the exact board state before each model
prediction.

The model still learns how humans behave from the encoded state it receives.
Some rule-sensitive concepts may be represented implicitly in that encoding
rather than passed as separate named features.

The runtime should also compute legal moves and mask illegal model outputs
before sampling. That legal-output failsafe is separate from how the model
learns move preferences.

## Learn Human Patterns From Human Games

The model should learn human move choice, timing, and imperfection from human
games. Avoid designing separate systems whose purpose is to force particular
classes of human mistakes.

Randomness should come from explicit sampling controls and learned human
variation, not from bypassing legality or injecting arbitrary noise.

## Build The Bot

This is not primarily a research project. The documentation should support
building a cool, playable, human-like chess bot.

Favor straightforward engineering choices that make the product work. Avoid
framing work as experiments unless the measurement directly helps build or tune
the bot.

## Keep Controls Independent

Important user-facing controls should remain distinct:

- Target rating controls the skill and style level to imitate.
- Temperature controls sampling variety.
- Optional style settings should be explicit.

Game-derived context such as clocks, move history, and phase should be modeled
as inputs, but it is not the same kind of thing as a user-facing behavior dial.

Temperature should not be hardwired as a hidden function of rating or clock
state.

## Prefer Compact Per-Ply Context

The sequence model should operate one timestep per ply. Each timestep should
receive a compact embedding of the exact board state, previous move, dynamic
clock features when available, and static game settings.

Training should preserve efficient causal-transformer behavior: feed full game
sequences or sequence chunks in parallel with a causal mask instead of training
one ply at a time in an autoregressive loop.

Avoid expanding every board into many sequence tokens unless testing proves that
the compact approach is insufficient.
