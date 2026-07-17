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

## Configuration Conventions

Configuration schemas, validation, and defaults are owned by command-specific
Pydantic models. Shared loading and provenance mechanics live under
`anthro_chess.config`. Checked-in TOML files under `configs/` are reproducible
selections for implemented commands; they do not define or duplicate schemas.

Commands should use code-owned defaults or an explicit configuration path and
must not depend on discovery from a repository working directory. Unknown
fields and invalid values should fail through the shared strict loader. Runs
and artifacts should retain the resolved configuration record together with
the relevant code, model, encoding, and data-provenance versions.

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
for stable filtering and routing, not to duplicate dependency state.

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

#### Implementation Queue And Public Intake

An issue is eligible for automatic selection as the next implementation task
only when all of the following are true in live GitHub metadata:

- it has the `type: task` label;
- it belongs to the active milestone;
- it is attached as a sub-issue of that milestone's tracker; and
- it has no open blockers; and
- it is not already claimed by another active implementation session.

The active milestone is the current roadmap stage represented by the live
milestone and tracker state. Among eligible issues, prefer work that matches the
current project stage and avoids conflicts with other active work.

Public issue forms are intake, not an implementation queue. They must not
automatically assign `type: task`, a milestone, a tracker relationship, or a
dependency relationship. An issue missing any eligibility metadata remains
intake awaiting maintainer triage. A maintainer promotes it into the queue only
by deliberately assigning all required metadata and any genuine blockers.

Contributor blank issues are disabled so public intake uses the focused forms.
Repository maintainers can still open a blank issue when GitHub permissions
allow it.

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

Keep tracker bodies focused on the milestone's goal, intended outcome, scope
boundaries, completion criteria, and relevant documentation or decisions. Do
not copy child-issue lists, checklists, status summaries, dependency maps,
labels, or other GitHub metadata into the body. Native milestone, sub-issue,
and dependency metadata is the source of truth.

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

Add new labels only when they solve a recurring filtering or routing problem.
Prefer assigning one primary area plus any genuinely relevant secondary area.

Use dependencies to express actual order constraints, not every preferred
implementation sequence. A useful dependency is one where starting or finishing
one issue would be confusing, wasteful, or impossible before another issue is
done. For softer ordering, write the suggested order in the tracker issue
instead of creating blocker relationships.

Do not add a status label such as `status: blocked` to repeat dependency state.
Parallel status metadata can become stale when a blocker closes. GitHub issue
dependencies are the source of truth for whether another issue blocks the work.
Represent an external blocker with a focused issue or decision when practical;
otherwise describe it clearly in the affected issue.

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
- apply the repo's `area:` and `type:` labels consistently;
- keep issue bodies focused on intent, acceptance criteria, relevant docs, and
  likely test/doc updates;
- avoid creating a detailed issue tree for the entire project before evidence
  exists;
- prefer a short runway of ready issues plus broader later placeholders;
- update `docs/planning/roadmap.md` only when the broad implementation order
  changes.

### Implementing An Issue

Treat a request to work on or implement an issue as authorization to carry the
work through the complete pre-merge workflow. The agent may create a branch and
worktree, edit files, run checks, commit, push, update in-scope issue metadata,
and open a pull request without asking separately at each step. This does not
authorize merging the pull request or making unrelated repository changes.

Before starting, the agent should:

- read the issue, its parent tracker, sub-issues if any, labels, milestone, and
  any linked docs or decisions;
- inspect the current repo and GitHub state before editing;
- confirm that formal dependencies do not block the issue;
- clarify only if the issue cannot be scoped safely from existing context;
- choose one cohesive, reviewable slice when the issue is broad;
- identify likely shared files or foundations before deciding whether the work
  can safely run in parallel with other sessions.

#### Branches And Worktrees

Use one dedicated branch and one isolated Git worktree per issue. A normal
branch name is `issue-<number>-<short-slug>`. If the session already runs in an
appropriate isolated worktree, reuse it instead of nesting another worktree.

Create new issue work from current `origin/main`. The base checkout open in an
editor does not need to remain perfectly current, and a session should not
switch branches or update files in that shared checkout merely to start its
work. Fetch the remote state and create the issue branch and worktree directly
from `origin/main`.

Never run multiple implementation sessions in the same worktree. Uncommitted
files are shared by every process using that directory even when branches are
different. Separate worktrees provide each session with its own branch and
filesystem while sharing the repository's Git object database, so separate
full clones are not normally needed.

Before editing, inspect the worktree for existing changes. Preserve unrelated
user changes and do not stage them into the issue commit.

#### Claiming Active Work

After the isolated issue worktree is ready and before substantive editing,
claim the issue in GitHub so another agent does not select the same task. Assign
the issue to the configured GitHub identity and add one concise comment that
identifies the agent and its stable session identifier:

```text
Claimed for implementation by <agent-name>.

Session: `<session-id>`
```

Use the actual agent or execution-surface name and the stable session identifier
provided by its environment. GitHub already records the comment author and
timestamp.

Do not include a hostname, absolute worktree path, local username or directory
layout, or a restatement of the issue scope in the public claim comment. Do not
include the local branch unless it communicates something that cannot be
recovered from the eventual pull request.

An assignment with a session claim, or an active pull request linked to the
issue, means the issue is already in progress and must be excluded when another
agent chooses work. Do not silently take over a claim that might be stale.

The claim comment is the only routine progress comment needed before
publication. Add further issue comments only for material findings, blockers,
coordination, scope changes, or handoff information that should remain visible
independently of the pull request.

If work is abandoned before a pull request is opened, remove the assignee and
add a brief comment that the claim was released. Include handoff information
only when unfinished work or a material finding remains useful. Once a linked
pull request exists, it becomes the primary status signal; leave the issue
assigned and let the merge close it normally.

#### Implementation And Verification

During implementation, the agent should:

- ensure required machine-level bootstrap tools are available, installing a
  missing tool or reporting a genuine installation blocker;
- initialize the current issue worktree with `uv sync --locked` before ordinary
  implementation work, using an unlocked dependency command only when the task
  intentionally changes dependency metadata;
- run development and final verification commands through the current
  worktree's project environment, never another worktree's virtual environment;
- implement one cohesive slice;
- add or update tests appropriate to the change;
- update docs only when durable intent, behavior, interfaces, data shape,
  evaluation, or source-of-truth ownership changes;
- keep issue metadata current when GitHub tooling is available, including
  labels, milestone, dependencies, and sub-issue state;
- run focused checks while iterating and all reasonable final verification
  before publishing;
- inspect the final diff for scope, accidental files, generated-file drift,
  secrets, and missing tests or documentation;
- sync with current `origin/main` before publishing and resolve conflicts in the
  issue worktree.

Pull requests and pushes to `main` run the canonical CPU-only verification and
built-package smoke test through `.github/workflows/ci.yml`. The workflow's
`Required CI` job is the stable required merge check. If that job is renamed,
update the repository merge rule in the same change so the protected branch
does not point at a stale check name.

If the user asks an agent to choose the next issue, the agent should prefer
eligible work from the active milestone's implementation queue. Check tracker
issues, sub-issue progress, dependencies, labels, assignments, claim comments,
linked pull requests, and recent issue activity before choosing. Treat claimed
issues as active work and issues outside the queue as intake awaiting
maintainer triage.

#### Publishing For Review

Commit only the issue's scoped files, push the issue branch, and open a pull
request when the implementation and verification are complete. Open it ready
for review by default. Use a draft pull request only when the user asks to see
work in progress, the work is intentionally being handed off incomplete, or an
external condition prevents it from being reviewable.

The pull request should:

- have a concise title describing the change;
- explain what changed and why;
- list the verification that ran and its result;
- identify material limitations, decisions, or reviewer attention points;
- link any genuine follow-up issues instead of hiding unfinished work in prose;
- include `Closes #<issue-number>` so merging into the default branch closes
  the implementation issue automatically.

The pull request is the normal human approval boundary. The agent must not
merge it or enable auto-merge unless the user explicitly asks. The user reviews
the diff and merges it when satisfied; a separate formal approval is not
required when repository rules allow the pull-request author to merge their
own work.

Use the configured Git and GitHub identities for commits and publication. Do
not invent a bot identity or add AI-attribution trailers or footers unless the
user requests them or the repository adopts an explicit attribution policy.

Before ending the session, leave GitHub in a useful review state:

- ensure the pull request links the issue and contains the implementation and
  verification summary;
- record follow-up issues instead of hiding unfinished work in a closing
  comment;
- update the tracker before publishing if the work changed milestone scope or
  ordering;
- add an issue comment only when an important finding needs to remain visible
  independently of the pull request.

Do not close the implementation issue merely because the pull request is ready.
The merge should close it. Close a tracker only when all of its milestone work
is actually complete.

#### After Publication And Merge

A complete, verified, ready-for-review pull request is the normal terminal
deliverable for an implementation session. The user should be able to review
and merge it, then call the task finished without returning to that session.

Configure GitHub to delete remote issue branches after merge. Local issue
branches and worktrees may remain; they are harmless apart from disk use and
listing clutter. Remove them during occasional housekeeping only after
confirming that their pull requests merged and their worktrees contain no
uncommitted changes.

The user should not need to return to an implementation session after merging
its pull request for routine post-merge cleanup or tracker commentary.

Keep milestone state aligned with the work it represents. When the milestone's
completion criteria are satisfied and all included work is complete or
explicitly tracked elsewhere, correct stale metadata, close the tracker, and
then close the GitHub milestone. Otherwise, leave the tracker and milestone
open and make the remaining work discoverable. Formal dependencies remain the
source of truth, so there is no duplicate blocked-status label to clear after a
merge.

#### Parallel Sessions

Parallel sessions are useful once boundaries are clear. Prefer parallel work
across separate areas such as chess logic, data ingestion, evaluation, and UCI.
Avoid running multiple sessions that modify the same shared architecture,
schema, package layout, or decision record unless the user explicitly
coordinates that work. Every parallel session must have its own issue branch
and worktree. When overlap becomes apparent, stop one slice or establish an
explicit dependency instead of letting two agents independently rewrite the
same foundation.

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
