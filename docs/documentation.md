# Documentation Guide

This repo uses documentation as shared project context for humans and AI agents.
The docs should make the intended direction easy to recover without rereading
old chats, but they should not duplicate every exact value that already has a
clear source of truth in code, schemas, or config files.

## How To Read The Docs

`AGENTS.md` is the agent entry point. It should stay concise and route agents to
the project workflow and relevant topic docs instead of repeating their detail.

The main topic docs under `docs/` describe the intended end state of the
project. They are living design docs: treat them as the current best intent,
not as a claim that every described feature is already implemented.

`docs/issue-workflow.md` describes how actionable work is tracked and organized
in GitHub. `docs/planning/` is for implementation order, staged plans, and
tradeoffs about how to reach the intended end state. When planning docs and
end-state docs disagree, the end-state docs win unless the project direction is
explicitly changed.

`docs/decisions/` records durable choices and why they were made.

`docs/research.md` records relevant outside work. It is supporting context, not
the project specification.

## What Belongs Where

- `README.md`: short public front door, current capabilities, intended use, and
  links to deeper docs.
- `CONTRIBUTING.md`: concise human-facing development and contribution guide.
- `AGENTS.md`: concise agent routing and durable project guardrails.
- `configs/README.md`: checked-in configuration conventions and ownership.
- `results/README.md`: the committed benchmark results store and the boundary
  with machine-local diagnostics.
- `docs/issue-workflow.md`: issue organization, labels, GPU routing, claiming,
  and the tracking conventions specific to this repository.
- `docs/vision.md`: high-level product intent and boundaries.
- `docs/design-principles.md`: principles that guide implementation choices.
- `docs/architecture.md`: system shape, major layers, and module boundaries.
- `docs/engine-behavior.md`: user-visible engine behavior.
- `docs/data.md`: data philosophy, source handling, schema direction, and
  training-data constraints.
- `docs/training-and-runtime.md`: model training and runtime behavior.
- `docs/interfaces.md`: UCI and other external or native interfaces.
- `docs/evaluation.md`: evaluation philosophy, metrics, and benchmark shape.
- `docs/preference-controls.md`: preference-control subsystem design.
- `docs/research.md`: papers, related projects, and source links.
- `docs/planning/roadmap.md`: broad implementation order.
- `docs/decisions/`: accepted architectural or project-shaping decisions.

If a new topic does not fit an existing doc, add a focused doc under `docs/`
and link it from `README.md` and, when agents need it routinely, from
`AGENTS.md`.

## Updating Docs During Implementation

Update docs in the same change when implementation alters project intent,
architecture, data format, runtime behavior, interfaces, or evaluation.

Do not rewrite design docs to pretend future work already exists. Prefer clear
wording such as:

- "The intended design is..."
- "The current implementation..."
- "This remains open..."

As code appears, docs should become maps to the real sources of truth. Exact
names, thresholds, defaults, schema fields, command flags, and config values
should usually live in code, schemas, or config files. In prose, describe the
concept and link to the source of truth instead of copying every value by hand.

It is fine to document exact details when they are stable external contracts,
hard to discover from code, generated from the source of truth, or necessary
for humans to use the project correctly.

## Adding Detail

Add more detail when it:

- guides imminent implementation;
- records something learned while implementing;
- prevents future agents from making a likely wrong assumption;
- explains a non-obvious tradeoff;
- documents a public or stable interface;
- points to the real source of truth for exact values.

Avoid fake precision. Do not invent final package names, tensor shapes, storage
layouts, thresholds, benchmark gates, or config schemas before the project has
enough evidence to choose them.

## Decision Records

Add a decision record when a choice is durable, architectural, hard to infer
from code, or likely to be re-litigated later.

Good decision-record candidates include:

- process or layer boundaries;
- model architecture choices;
- data format and rating-scale choices;
- interface strategy;
- evaluation gates that become part of the project contract;
- accepted tradeoffs where plausible alternatives existed.

Do not add a decision record for every evolving thought. Normal design intent
belongs in the topic docs. Build order belongs in `docs/planning/roadmap.md`.
Small implementation facts belong in code, tests, or nearby comments.

Use the existing records in `docs/decisions/` as the format. Early records may
say "Accepted as initial design direction" when the choice is strong enough to
guide implementation but still open to being superseded by later evidence.

If a later choice reverses an old one, add a new decision record that supersedes
it instead of rewriting history.

## Research Notes

Use `docs/research.md` for outside work that materially informs the project.
Each entry should say what the source is, what matters, which part of Anthro
Chess it applies to, and how this project differs.

Do not let research notes turn the project into research for its own sake. The
project goal remains a usable human-like chess opponent.
