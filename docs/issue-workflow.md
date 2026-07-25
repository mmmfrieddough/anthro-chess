# Issue Workflow

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

## Starting Implementation Work

The user should not need to paste a long boilerplate prompt at the start of each
implementation session. `AGENTS.md`, `docs/documentation.md`, topic docs, and
decision records are the durable handoff.

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

## Issue Organization

### Implementation Queue And Public Intake

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

Selection is also constrained by the execution capabilities declared for the
session. A session without the GPU environment specified by an issue must
exclude issues labeled `execution: gpu-required`. Issues labeled
`verification: gpu-required` remain eligible because their implementation can
be completed without a GPU. The dispatcher or execution environment should
declare the session's surface and hardware capabilities explicitly; the
implementation session should treat that declaration as authoritative rather
than inferring capabilities from incidental host details.

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
- `execution: gpu-required`
- `verification: gpu-required`

Add new labels only when they solve a recurring filtering or routing problem.
Prefer assigning one primary area plus any genuinely relevant secondary area.

The GPU labels have different routing meanings:

- `execution: gpu-required` means meaningful implementation, measurement, or
  iteration requires the GPU environment specified in the issue. Sessions
  without that environment must not select the issue.
- `verification: gpu-required` means the implementation can be completed and
  reviewed without a GPU, but a required acceptance check still needs the GPU
  environment specified in the issue. This label does not remove the issue
  from a CPU-only or cloud implementation queue.

Use `execution: gpu-required` when work without the GPU cannot produce a
coherent, independently reviewable implementation that materially satisfies
the issue. Examples include finding a maximum multi-GPU batch size, debugging
an NCCL hang, comparing GPU-kernel throughput, or training several variants to
select a checkpoint.

Use `verification: gpu-required` only when the remaining GPU work is a bounded
acceptance check rather than substantial implementation or experimental
iteration. Do not add it for optional confidence-building. When a useful
non-GPU slice can stand alone but substantial GPU work remains, prefer separate
issues with a dependency between the cloud-suitable preparation and the
GPU-required task instead of planning an incomplete implementation handoff.

The issue body should state the required environment precisely enough to act
on, such as Apple Silicon MPS, one CUDA GPU with a minimum memory capacity, or
two specified CUDA GPUs with relevant distributed-runtime constraints. The
label expresses routing; the issue body owns the exact hardware, software,
commands, expected evidence, and acceptance criteria.

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

## Refining The Task Roadmap

When asked to build or refine the task roadmap, an agent should:

- read `AGENTS.md`, this workflow, `docs/planning/roadmap.md`, and relevant
  design docs;
- inspect existing open issues, milestones, tracker issues, labels, and
  sub-issue relationships before creating new ones;
- create or update the real GitHub milestone when needed;
- create or update a tracker issue for the milestone when needed;
- create or update near-term actionable child issues and attach them as
  sub-issues of the tracker when GitHub supports it;
- add GitHub issue dependencies for true blockers between child issues;
- apply the repo's `area:`, `type:`, `execution:`, and `verification:` labels
  consistently;
- classify concrete GPU requirements using the routing test above and record
  the exact required environment and acceptance evidence in the issue body;
- keep issue bodies focused on intent, acceptance criteria, relevant docs, and
  likely test/doc updates;
- avoid creating a detailed issue tree for the entire project before evidence
  exists;
- prefer a short runway of ready issues plus broader later placeholders;
- update `docs/planning/roadmap.md` only when the broad implementation order
  changes.

## Implementing An Issue

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
- confirm that the declared execution capabilities are compatible with any
  `execution:` requirement;
- clarify only if the issue cannot be scoped safely from existing context;
- choose one cohesive, reviewable slice when the issue is broad;
- identify likely shared files or foundations before deciding whether the work
  can safely run in parallel with other sessions.

### Branches And Worktrees

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

### Claiming Active Work

After the isolated issue worktree is ready and before substantive editing,
claim the issue in GitHub so another agent does not select the same task. Assign
the issue to the configured GitHub identity and add one concise comment that
identifies the agent and, when available, its stable session link or identifier:

```text
Claimed for implementation by <agent-name>.

Session: <session-link-or-id>
```

Use the actual agent or execution-surface name and the stable session identifier
provided by its environment. Prefer a direct session link when the execution
surface exposes one. If it exposes neither a link nor a stable identifier, omit
the `Session` line rather than delaying the claim or inventing a value. GitHub
already records the comment author and timestamp.

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

The implementation session owns issue and pull-request updates after dispatch.
The dispatcher does not need to claim the issue, publish progress, or perform
handoff bookkeeping. Before claiming, the session should recheck eligibility
and stop if another active claim or pull request appeared after dispatch.

If work is abandoned before a pull request is opened, remove the assignee and
add a brief comment that the claim was released. Include handoff information
only when unfinished work or a material finding remains useful. Once a linked
pull request exists, it becomes the primary status signal; leave the issue
assigned and let the merge close it normally.

### Implementation And Verification

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
  labels, milestone, dependencies, sub-issue state, and newly discovered
  execution or verification requirements;
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

If a session discovers that meaningful progress actually requires an
unavailable GPU environment, correct the issue metadata before stopping. Apply
`execution: gpu-required`, record the exact requirement and evidence needed,
release the claim if no pull request will remain active, and leave one concise
finding or handoff comment. Do not publish a token partial change merely to
avoid releasing the issue.

If the user asks an agent to choose the next issue, the agent should prefer
eligible work from the active milestone's implementation queue. Check tracker
issues, sub-issue progress, dependencies, labels, assignments, claim comments,
linked pull requests, and recent issue activity before choosing. Treat claimed
issues as active work and issues outside the queue as intake awaiting
maintainer triage.

### Offering A Real GUI Check

Some changes are only convincing in a real chess GUI. Automated coverage proves
protocol behavior, but command ordering, option presentation, and how a game
actually feels belong to the maintainer.

Offer a GUI check without being asked when a change alters what a GUI observes:
move selection, sampling or seeding, rating or temperature controls, position
synchronization, advertised options, protocol command handling, or engine
lifecycle. Skip it for changes a GUI cannot see, such as data pipeline,
training, evaluation, or documentation work.

To offer one, initialize the worktree environment, point the GUI at the
worktree with `scripts/anthro-gui-target .`, and tell the maintainer the engine
is ready to test, which behavior to look at, and what a good result looks like.
Do not reconfigure the GUI itself and do not edit anything under `.venv`. The
mechanism is documented in `playable-uci.md`.

Leave the pointer aimed at the worktree while the pull request is open. Clear it
with `scripts/anthro-gui-target --clear` when abandoning the branch, and note in
the issue that the maintainer should clear it after merge if the worktree is
removed. A removed target fails loudly rather than silently serving stale code.

Maintainer GUI observations belong in the issue as findings. Treat a GUI session
as acceptance evidence, not as a substitute for automated coverage: whenever a
GUI check finds a defect, add the regression test that would have caught it.

### Publishing For Review

Commit only the issue's scoped files, push the issue branch, and open a pull
request when the implementation and verification are complete. Open it ready
for review by default. Use a draft pull request only when the user asks to see
work in progress, the work is intentionally being handed off incomplete, or an
external condition prevents it from being reviewable.

For an issue labeled `verification: gpu-required`, a session without the
specified GPU may still open a ready-for-review pull request after the
implementation and all available checks are complete. The pull request must
prominently identify the pending GPU verification, including the required
environment, exact command or procedure, expected evidence, and any artifacts
to preserve. The pending check becomes part of local review, and the issue
label remains as the durable routing signal.

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

### Local Handoff And Ownership

When substantial local or GPU implementation remains, publish a draft pull
request instead of leaving only a pushed branch. The draft provides a durable
diff, CI results, discussion surface, and ownership boundary. Include a
`Local handoff` section covering:

- the precise required environment;
- completed work and verification;
- remaining implementation or investigation;
- exact commands or procedures to run;
- expected evidence and artifacts;
- the prior session link or identifier when available.

Do not use a draft handoff for work that should have been split into an
independently reviewable cloud task and a separate GPU-required issue. Prefer
that issue split during planning whenever the boundary is known in advance.

A new local session taking over an issue and pull request becomes the sole
owner of the task from that point forward. It should read the issue, pull
request, diff, CI, and handoff; add one concise takeover comment; continue on
the existing pull-request branch in its own worktree; complete the required
local work and final verification; update the issue and pull request; and mark
the draft ready for review when complete. The user should not need to return to
the prior cloud session.

### After Publication And Merge

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

### Parallel Sessions

Parallel sessions are useful once boundaries are clear. Prefer parallel work
across separate areas such as chess logic, data ingestion, evaluation, and UCI.
Avoid running multiple sessions that modify the same shared architecture,
schema, package layout, or decision record unless the user explicitly
coordinates that work. Every parallel session must have its own issue branch
and worktree. When overlap becomes apparent, stop one slice or establish an
explicit dependency instead of letting two agents independently rewrite the
same foundation.

## Agent Checklist

Before substantive changes:

1. Read `AGENTS.md`.
2. Read `docs/documentation.md`.
3. Read the relevant topic docs.
4. Check relevant decision records.
5. Keep roadmap/build-order edits in `docs/planning/`.
6. Update affected docs when the change alters durable intent.
7. Add a decision record only when the rationale has lasting value.
8. Offer a real GUI check when the change alters what a GUI observes.
