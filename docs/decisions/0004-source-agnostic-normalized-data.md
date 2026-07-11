# 0004: Source-Agnostic Normalized Data Pipeline

Date: 2026-07-11

## Status

Accepted as initial design direction.

## Context

The project is expected to use mostly Lichess data at first, especially sources
with useful clock information. Other sources may later contribute high-level
human games, different rating systems, player-specific games, engine games for
diagnostics, or out-of-distribution evaluation.

The project should avoid a pile of hand-managed data files whose provenance,
schema, and preprocessing are hard to reproduce. It should also avoid locking
the internal schema to one provider's exact export format.

## Decision

Build reproducible data pipelines that ingest source-specific exports and write
compact, source-agnostic normalized records.

The normalized format should:

- use generic field names rather than provider-specific names;
- distinguish game-level fields from ply-level fields;
- store observed facts and compact labels;
- explicitly represent missing fields instead of overloading zero or another
  valid value;
- keep raw or source-specific material outside the training schema unless it is
  needed for provenance or reproducibility;
- recompute deterministic chess state and legal masks unless profiling shows a
  strong reason to materialize them.

Lichess can heavily shape the first implementation, but the schema should be a
superset of useful training concepts rather than a copy of Lichess PGN.

## Consequences

This preserves the ability to mix data sources later without changing model
code every time a new source is added.

Pipeline scripts become part of the project contract: data used for training
should be reproducible from documented sources and transformations.

The normalized format must balance storage size against dataloader cost. Board
state and legal masks should start as computed-on-read unless they become a
measured bottleneck.
