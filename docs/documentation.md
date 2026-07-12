# Documentation Guide

This repo uses documentation as shared project context for humans and AI agents.
The docs should make the intended direction easy to recover without rereading
old chats, but they should not duplicate every exact value that already has a
clear source of truth in code, schemas, or config files.

## How To Read The Docs

`AGENTS.md` is the agent entry point. It should stay concise and point agents to
the right project docs instead of repeating all design details.

The main docs under `docs/` describe the intended end state of the project.
They are living design docs: treat them as the current best intent, not as a
claim that every described feature is already implemented.

`docs/planning/` is for implementation order, staged plans, and tradeoffs about
how to get there. Planning docs should not redefine the product's end state.
When planning docs and end-state docs disagree, the end-state docs win unless
the project direction is explicitly changed.

`docs/decisions/` records durable choices and why they were made.

`docs/research.md` records relevant outside work. It is supporting context, not
the project specification.

## What Belongs Where

- `README.md`: short public front door, intended use, and links to deeper docs.
- `AGENTS.md`: concise agent guidance, reading order, and durable guardrails.
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

If a new topic does not fit an existing doc, add a new focused doc under
`docs/` and link it from `README.md` and, when agents need it routinely, from
`AGENTS.md`.

## Starting Implementation Work

The user should not need to paste a long boilerplate prompt at the start of each
implementation session. `AGENTS.md`, this guide, topic docs, and decision
records are the durable handoff.

When given an implementation request, agents should:

1. Start from `AGENTS.md`.
2. Use the request to identify the relevant topic docs.
3. Read the relevant decision records.
4. Inspect the current repo before choosing edits.
5. If the request is broad, choose or propose a cohesive first slice.
6. Implement within that scope.
7. Add or update tests appropriate to the change.
8. Update docs only when the implementation changes durable intent, behavior,
   interfaces, data shape, evaluation, or source-of-truth ownership.

The user may still start a session with a short goal, such as "start the data
pipeline" or "implement the UCI wrapper." The agent is responsible for turning
that goal into a repo-aware work plan by reading the local context.

If an explicit current-work file, issue, or roadmap item exists, use it as
planning context. Do not require one to begin a well-scoped task.

## GitHub Issue Workflow

Use GitHub issues as the default tracker for actionable implementation work
when the repo has GitHub access available. The roadmap should stay broad;
issues hold the concrete task queue, discussion, links to pull requests, and
per-task status.

Use GitHub's built-in planning structure instead of encoding workflow state in
issue titles. Roadmap milestones should map to real GitHub milestones. Tracker
issues should group work for a milestone or workstream. Child implementation
issues should be attached as sub-issues when GitHub supports it for the repo.
Use issue dependencies for true blockers and sequencing constraints. Use labels
for filtering and routing.

The user's recurring manual actions should stay small:

1. Ask an agent to refine the task roadmap.
2. Ask an agent to create or update issues for a roadmap milestone.
3. Ask an agent to implement a specific issue or choose the next appropriate
   issue.

The user should be able to do any of these with a short natural-language
request. The agent is responsible for reading the repo context, finding
relevant docs, checking issue state, and turning the request into a scoped
plan.

### Issue Organization

Use this structure for milestone work:

- a real GitHub milestone named for the roadmap milestone, such as
  `0. Project Setup`;
- one tracker issue for the milestone, labeled `type: tracker`;
- focused actionable child issues, labeled `type: task`;
- real GitHub sub-issue relationships from the tracker to the child issues when
  available;
- real GitHub issue dependencies for child issues that must happen before other
  child issues can be started or finished;
- milestone assignment on both the tracker and its child issues.

Do not put milestone names in ordinary issue titles when the issue is already
assigned to a GitHub milestone. Tracker issues may include the milestone name
because their purpose is to be a visible front door for that milestone.

Use a small label taxonomy:

- `area: setup`
- `area: docs`
- `area: chess-logic`
- `area: data`
- `area: training`
- `area: runtime`
- `area: interfaces`
- `area: evaluation`
- `type: task`
- `type: tracker`
- `type: decision`
- `status: blocked`

Add new labels only when they solve a recurring filtering or routing problem.
Prefer assigning one primary area plus any genuinely relevant secondary area.
Use `status: blocked` only when the issue cannot progress without another
issue, decision, or external input.

Use dependencies to express actual order constraints, not every preferred
implementation sequence. A useful dependency is one where starting or finishing
one issue would be confusing, wasteful, or impossible before another issue is
done. For softer ordering, write the suggested order in the tracker issue
instead of creating blocker relationships.

### Refining The Task Roadmap

When asked to build or refine the task roadmap, an agent should:

- read `AGENTS.md`, this guide, `docs/planning/roadmap.md`, and relevant design
  docs;
- inspect existing open issues, milestones, tracker issues, labels, and
  sub-issue relationships before creating new ones;
- create or update the real GitHub milestone when needed;
- create or update a tracker issue for the milestone when needed;
- create or update near-term actionable child issues and attach them as
  sub-issues of the tracker when GitHub supports it;
- add GitHub issue dependencies for true blockers between child issues;
- apply the repo's `area:`, `type:`, and `status:` labels consistently;
- keep issue bodies focused on intent, acceptance criteria, relevant docs, and
  likely test/doc updates;
- avoid creating a detailed issue tree for the entire project before evidence
  exists;
- prefer a short runway of ready issues plus broader later placeholders;
- update `docs/planning/roadmap.md` only when the broad implementation order
  changes.

### Implementing An Issue

When asked to implement an issue, an agent should:

- read the issue, its parent tracker, sub-issues if any, labels, milestone, and
  any linked docs or decisions;
- inspect the current repo state before editing;
- clarify only if the issue cannot be scoped safely from existing context;
- implement one cohesive slice;
- add or update tests appropriate to the change;
- update docs only when durable intent, behavior, interfaces, data shape,
  evaluation, or source-of-truth ownership changes;
- keep issue metadata current when GitHub tooling is available, including
  labels, milestone, dependencies, blockers, and sub-issue state;
- comment on the issue with important findings, verification results,
  follow-ups, and completion status;
- close the issue only after the implementation is merged or otherwise clearly
  completed in the repo state the user wants tracked.

If the user asks an agent to choose the next issue, the agent should prefer
ready, unblocked work in the current milestone that matches the current project
stage and avoids uncoordinated edits to shared foundations. Check tracker
issues, sub-issue progress, dependencies, labels, and recent issue activity
before choosing.

When finishing issue work, leave GitHub in a useful state for the next agent:

- link or mention the pull request or commit when applicable;
- summarize what changed and what verification ran;
- record follow-up issues instead of hiding unfinished work in a closing
  comment;
- update the tracker issue if milestone progress or ordering changed;
- close completed child issues, and close the tracker only when its milestone
  work is complete.

Parallel sessions are useful once boundaries are clear. Prefer parallel work
across separate areas such as chess logic, data ingestion, evaluation, and UCI.
Avoid running multiple sessions that modify the same shared architecture,
schema, package layout, or decision record unless the user explicitly
coordinates that work.

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
hard to discover from code, generated from the source of truth, or necessary for
humans to use the project correctly.

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

## Agent Checklist

Before substantive changes:

1. Read `AGENTS.md`.
2. Read this guide.
3. Read the relevant topic docs.
4. Check relevant decision records.
5. Keep roadmap/build-order edits in `docs/planning/`.
6. Update affected docs when the change alters durable intent.
7. Add a decision record only when the rationale has lasting value.
