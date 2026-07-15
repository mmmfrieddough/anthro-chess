# 0007: Python-Chess Rules With An Anthro-Owned Action Contract

Date: 2026-07-14

## Status

Accepted for the initial standard-chess implementation.

## Context

Board reconstruction, legal move generation, and special-rule bookkeeping are
correctness-critical infrastructure shared by data preparation, training,
evaluation, and runtime. Reimplementing those rules would create a second chess
engine to maintain. At the same time, third-party objects and notation formats
should not become the project's model-facing compatibility contract.

The initial move model also needs one fixed action vocabulary. That vocabulary
must represent every standard board move, promotions, and resignation while
remaining stable across data manifests, checkpoints, evaluation, and runtime.

## Decision

Use `python-chess` for standard-chess rules and notation parsing. Keep it behind
the typed adapters in `anthro_chess.chess`, which own immutable positions,
moves, replay errors, and the public boundary used by the rest of the project.

Use one generated, versioned standard-chess action vocabulary. Board moves are
identified by their from-square, to-square, and optional promotion. Its ordered
tokens include every geometrically possible queen-like or knight-like square
pair, all standard promotion variants, and a distinct resignation action.
Castling uses the standard king move. The exact size and digest are exposed by
the codec identity and protected by tests.

The action ids, not UCI strings, are the model-facing and normalized-data
contract. Coordinate and SAN parsing remain boundary and debugging tools.

## Consequences

Anthro-owned code stays focused on stable adapters and compatibility rather
than chess-rule implementation. Consumers share the same move validation,
legal action ids, and vocabulary identity.

The first codec is deliberately standard-chess-only. A future variant or
Chess960 codec must have a different versioned identity rather than silently
changing this vocabulary.

`python-chess` is distributed under GPL-3.0-or-later. Anthro Chess uses the
same GPL-3.0-or-later terms so its source distribution and package metadata
state the combined program's licensing clearly.
