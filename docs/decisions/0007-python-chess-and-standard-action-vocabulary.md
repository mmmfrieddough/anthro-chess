# 0007: Python-Chess Rules With An Anthro-Owned Action Contract

Date: 2026-07-14

## Status

Accepted for the initial standard-chess implementation.

## Context

Board reconstruction, legal move generation, and special-rule bookkeeping are
correctness-critical infrastructure shared by data preparation, training,
evaluation, and runtime. Reimplementing those rules, or wrapping the library's
domain model in parallel Anthro-owned types, would add code without improving
the model-facing contract.

The initial move model also needs one fixed action vocabulary. That vocabulary
must represent every standard board move, promotions, and resignation while
remaining stable across data manifests, checkpoints, evaluation, and runtime.

## Decision

Use `python-chess` directly for standard-chess boards, moves, rule bookkeeping,
notation parsing, and errors. Training, evaluation, data, and runtime code may
work with `chess.Board` and `chess.Move` rather than translating them into
parallel project types.

Anthro Chess owns one generated, versioned standard-chess action vocabulary.
Board moves are identified by their `python-chess` move value: from-square,
to-square, and optional promotion. The ordered vocabulary includes every
geometrically possible queen-like or knight-like square pair, all standard
promotion variants, and a distinct resignation id. Castling uses the standard
king move. The exact size and digest are exposed by the codec identity and
protected by tests.

The action ids, not Python objects or UCI strings, are the persisted
model-facing and normalized-data contract. UCI strings are used only to define
and hash the deterministic vocabulary order; coordinate and SAN parsing remain
interface and debugging tools.

## Consequences

Anthro-owned chess code stays limited to the compatibility boundary the model
actually needs: move/action-id conversion, legal action ids, the separate
resignation id, and vocabulary identity. Consumers use the mature rules
library directly instead of maintaining duplicate representations and
translations.

The first codec is deliberately standard-chess-only. A future variant or
Chess960 codec must have a different versioned identity rather than silently
changing this vocabulary.

`python-chess` is distributed under GPL-3.0-or-later. Anthro Chess uses the
same GPL-3.0-or-later terms so its source distribution and package metadata
state the combined program's licensing clearly.
