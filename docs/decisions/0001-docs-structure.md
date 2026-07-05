# 0001: Initial Documentation Structure

Date: 2026-07-04

## Status

Accepted as the initial project documentation structure.

## Context

The repo is starting from a minimal state with a README and license. The project
already has a rich planning document describing a controllable human-like chess
model, but that information needs to be split into durable, navigable documents
for both humans and AI agents.

## Decision

Use root-level `AGENTS.md` as the entry point for AI agents and keep durable
design material under `docs/`.

Initial docs:

- `docs/vision.md`
- `docs/design-principles.md`
- `docs/architecture.md`
- `docs/engine-behavior.md`
- `docs/training-and-runtime.md`
- `docs/planning/roadmap.md`
- `docs/decisions/`

## Consequences

Future agents should have enough context to make design-aligned changes without
requiring the original planning conversation.

The docs are intentionally living documents. Implementation changes that alter
the intended architecture or product direction should update the relevant docs
and, when appropriate, add a decision record.

Planning docs are separated from end-state design docs so implementation order
can evolve without making the roadmap look like part of the product definition.
