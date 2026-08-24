# Issue Workflow

This document holds the conventions specific to this repository: how issues are
organized, which labels mean what, what has to be true before an issue is ready
to pick up, and which checks and tools this project uses.

It deliberately says nothing about how to carry out the work. Loading context,
isolating a branch, setting up an environment, scoping a slice, and publishing
are the agent's own business, and local setup is documented in
`CONTRIBUTING.md`. What follows is only what cannot be inferred from the
repository itself.

Use GitHub issues as the tracker for actionable implementation work. The roadmap
stays broad; issues hold the concrete task queue, discussion, links to pull
requests, and per-task status.

Use GitHub's built-in planning structure instead of encoding workflow state in
issue titles. Roadmap milestones map to real GitHub milestones. Tracker issues
group work for a milestone or workstream. Child implementation issues are
attached as sub-issues. Issue dependencies express true blockers and sequencing
constraints. Labels are for stable filtering and routing, not for duplicating
dependency state.

Issue branches are named `issue-<number>-<short-slug>`. Sessions frequently run
in parallel against this repository, so an issue is claimed in GitHub while it
is being worked on, and the pull request that closes it is the point at which
the maintainer reviews and merges.

## Issue Organization

### Implementation Queue And Public Intake

An issue is ready to be picked up as an implementation task only when all of the
following are true in live GitHub metadata:

- it has the `type: task` label;
- it belongs to the active milestone;
- it is attached as a sub-issue of that milestone's tracker; and
- it has no open blockers; and
- it is not already claimed by another active implementation session.

The active milestone is the lowest-numbered open milestone. Milestones carry a
numeric prefix so that ordering is readable from the list itself rather than
inferred from tracker state. `Parked` deliberately carries none, so it can never
read as the stage after the current one.

Selection is also constrained by the hardware an issue requires. An issue
labeled `execution: gpu-required` is not available where the GPU environment its
body specifies is not. An issue labeled `verification: gpu-required` stays
available, because its implementation can be completed without a GPU.

Public issue forms are intake, not an implementation queue. They must not
automatically assign `type: task`, a milestone, a tracker relationship, or a
dependency relationship: a submission from outside the project has not yet been
judged real, in scope, or correctly placed, and a form cannot assert that it
has been.

Making that judgement is project-side work, and working in this repository is
being on that side. An issue filed from work done here is therefore a finding
rather than a submission, and is filed with its metadata already on it — the
`area:` and `type:` labels, the milestone it belongs to, its tracker, and any
genuine blocker. Leaving it bare defers nothing: it moves the work to someone
holding less context than whoever wrote the evidence into the body, and leaves
the finding unfindable until they get to it.

Two things keep that honest. The milestone is where the work belongs rather
than the one that would make it selectable, so a finding outside the current
stage goes in a later numbered milestone or in `Parked`; and a blocker is a real
order constraint rather than a preference, by the rule below. What stays with
the maintainer is what it is everywhere else here — reviewing and merging — and
an issue still has to carry every field to be picked up, so one filed bare is
intake either way.

`Parked` holds real findings that improve something already working. A reading
that is wrong, a run that loses data, or a defect a player would observe belongs
in the active milestone whatever it costs to fix. Everything else names the
decision it changes and who is waiting on that decision; where there is none, it
is parked. Until this project has completed a real training run, an efficiency
or precision improvement to evaluation machinery that already produces a correct
reading is parked unless it unblocks the active milestone's exit — say which.

An issue leaves `Parked` by being moved into a numbered milestone deliberately.
Nothing sweeps it and no review is scheduled, so the finding keeps its evidence
and stays searchable until someone decides it is next.

An issue whose acceptance is that the model got better names the metric that
carries the claim, before the work starts. Give the metric as
`uv run anthro eval metrics` reports it, the benchmark that produces it, and the
shape the movement is expected to take:

- **one metric**, where the change targets a quantity and the rest should hold;
- **several named metrics**, where the claim is a conjunction and partial
  movement is a finding rather than a pass;
- **across the board**, where the change is a general capability one and the
  interesting outcome is anything that moved the wrong way.

Written after the run, a claim is not a claim: whichever metric moved becomes
the one that was meant, and a change that improved one thing while quietly
degrading two reads as a success. Written first, the reading can fail.

This applies to changes claiming a model improvement, not to every issue. A bug
fix, a refactor, or a documentation change has no carrying metric and should not
invent one.

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

### Labels

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

### GPU Routing

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

If a session discovers that meaningful progress actually requires an
unavailable GPU environment, correct the issue metadata before stopping. Apply
`execution: gpu-required`, record the exact requirement and evidence needed,
release the claim if no pull request will remain active, and leave one concise
finding or handoff comment. Do not publish a token partial change merely to
avoid releasing the issue.

### Dependencies

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
- write acceptance criteria that decide new surface rather than only describing
  it: when an issue may add an option, a mode, a dependency, or a layer, the
  criterion says what evidence justifies keeping it and that an unproven
  addition is removed rather than shipped disabled by default, per
  `docs/design-principles.md` "Earn New Surface";
- avoid creating a detailed issue tree for the entire project before evidence
  exists;
- prefer a short runway of ready issues plus broader later placeholders;
- update `docs/planning/roadmap.md` only when the broad implementation order
  changes.

## Claiming

Assign the issue to the configured GitHub identity and add one concise comment
that identifies the agent and, when available, its stable session link or
identifier:

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
issue, means the issue is already in progress. Do not silently take over a claim
that might be stale.

The claim comment is the only routine progress comment needed before
publication. Add further issue comments only for material findings, blockers,
coordination, scope changes, or handoff information that should remain visible
independently of the pull request.

If work is abandoned before a pull request is opened, remove the assignee and
add a brief comment that the claim was released.

## Verification

Pull requests and pushes to `main` run the canonical CPU-only verification and
built-package smoke test through `.github/workflows/ci.yml`. The workflow's
`Required CI` job is the stable required merge check. If that job is renamed,
update the repository merge rule in the same change so the protected branch
does not point at a stale check name.

Keep issue metadata current when GitHub tooling is available, including labels,
milestone, dependencies, sub-issue state, and newly discovered execution or
verification requirements.

An unreproduced flake gets an issue, not an argument. A test that failed once
and passes in isolation is an unknown until something explains it, and reasoning
that this change could not have caused it disposes of the question without
answering it. File the failure with the invocation that produced it, then carry
on.

## Taking A Shakedown Reading

A new or materially changed benchmark is verified against real checkpoints
before its pull request is ready, not only against fixtures.
`docs/evaluation.md` owns what a shakedown reading is and why; this section
owns when one is required and how a session without checkpoints routes it.

It applies to a change that adds a benchmark, changes how an existing metric is
computed, or changes what a benchmark selects to measure over. It does not apply
to reporting, storage, or CLI plumbing that moves no measured quantity, and a
change whose whole effect is on the store's layering is verified against the
store instead.

Run the benchmark on two checkpoints far apart in one training run, with a
capped view and `--no-record`, and report in the pull request what was
expected, what was read, and whether they agreed. A disagreement is a finding to
write up rather than a run to repeat until it looks better.

Establish whether this machine has checkpoints before concluding that it has
none. They live outside the worktree, beneath `ANTHRO_CHESS_RUN_ROOT` and
`ANTHRO_CHESS_DATA_ROOT`, so a checkout with no `artifacts/` or `runs/`
directory is not evidence either way and neither is an empty search of the
repository and its worktrees. `CONTRIBUTING.md` gives the commands that answer
it. Deferring a reading that could have been taken is the more costly error,
because it moves work to the maintainer and lands the change on fixtures alone.

A session that genuinely has neither checkpoints nor a normalized corpus cannot
take the reading, and this does not block implementation. Route it the way a
pending GPU check is routed: complete the implementation, open the pull request
with the pending reading named prominently, including the exact command and what
the expected direction is, and leave the reading as part of local review.

Shakedown readings are never appended to the committed results store. Committing
a benchmark result is a separate decision about project history and is made
where that history is designated, not as a side effect of landing a benchmark.

## Showing A Model Change Improved Something

A shakedown reading asks whether an instrument measures anything. This asks the
other question, about the model rather than the instrument: whether a change
that alters what the model learns improved it. It applies to a change under
`src/anthro_chess/models/`, `src/anthro_chess/training/`,
`src/anthro_chess/data/`, `src/anthro_chess/chess/actions.py`,
`configs/training/`, or `configs/data/`, because those are the paths a training
run reads. `docs/evaluation.md` owns what the reading is; this section owns when
it is required and how a session that cannot take it routes it.

**What the control is depends on which question the change asks.** A candidate
being evaluated for adoption is read as an arm against the ablation vehicle — the
frozen configuration
`docs/decisions/0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
designates — because it is the only base in this project carrying a characterized
seed dispersion. A change to the canonical line itself is read against the prior
checkpoint whose training identity matches, which is what `0063` settles. Neither
substitutes for the other, and the pull request says which it took.

An arm is the vehicle's own configuration with the one change applied and its
own run name, so what separates the two arms is visible in the command:

```console
uv run anthro train --config configs/training/ablation-vehicle.toml \
  --set 'run_name="arm-<issue>-control"'

uv run anthro train --config configs/training/ablation-vehicle.toml \
  --set 'run_name="arm-<issue>-treatment"' --set <the one change>
```

The control arm is only trained where no reading of the vehicle at this horizon
is already on the machine. `training_sha256` is what settles that: a recorded
arm whose identity matches the treatment's in everything but the change is a
control, and `0063` owns why a second run buys nothing the identity check does
not. Both arms are then scored the same way, which is the block below.

Arms against a frozen base give main effects and no interaction terms, so a set
of separately accepted candidates runs together as one further arm before any of
it reaches the canonical line. That arm is what says the set composes; without
it, a stack of individually justified changes is an assumption.

**Both arms get their own learning rate, or the reading is about the tuning.**
The peak learning rate is coupled to nearly every other choice a candidate might
change, so a candidate compared against a baseline whose rate suits the baseline
and not the candidate measures which arm was better tuned. Where the project's
fitted rule covers the candidate, applying it to both arms discharges this; where
the candidate is the kind of change the rule does not cover, the pull request
says how each arm's rate was set. `docs/scaling.md` owns which changes those are.
A candidate discarded on a negative reading is only discarded where this was
done.

Not every change under those paths trains a different model. A change meant to
leave the weights alone — a loader representation, an instrumentation path, a
refactor — says so in the pull request and shows it instead: two short runs at
one seed under strict determinism, one on each side of the change, whose
readings agree exactly.
That costs minutes rather than hours and claims something a training comparison
cannot, so establish it before assuming the expensive reading is owed.

The reading is complete before the pull request is ready, and the claim it
tests — which metric moves, in which direction — is written down before either
arm runs. It belongs in the pull request beside the other commands that were
run, and the pull request says which benchmarks were read and which were not.

Two arms establish that two models differ. What establishes that the change is
why depends on whether the comparison has a seed floor available.

**Read against the vehicle, it does.** The vehicle's dispersion is stored against
its identity digest, so a delta clearing both it and the evaluation floor is
qualified on the term that otherwise fakes narrow results. The exception is an
arm whose training-health readings depart from the vehicle's: instability widens
an arm's spread beyond what a floor measured on stable baselines allows, so the
floor reads too narrow and the comparison says so rather than quoting it.

**Read anywhere else, it does not.** No floor sees training-seed noise, so a
`cleared` verdict says the delta survived a different draw of evaluation units and
nothing more. The pull request says that rather than reporting the verdict as a
result, and carries the claim only where the delta is far enough outside seed
variance that nothing else explains it, or where arms were read at several seeds.

`docs/evaluation.md` (Regression Comparisons) owns the rule, and
`docs/decisions/0065-...md` owns why the vehicle is the one place this is
affordable.

A session with neither the training hardware nor the corpus cannot produce an
arm. Route it the way a pending GPU check is routed: complete the
implementation, open the pull request with the pending reading named
prominently, including both arms' exact commands and the expected direction, and
correct the issue's routing label — `execution: gpu-required` where what the
reading says would change the implementation, `verification: gpu-required` where
the implementation is settled and only the reading remains.

**A null or negative reading is a result rather than a failed attempt.** The
pull request lands with the reading it got, and the issue closes having answered
its question. An arm is never re-run in the hope of a better number; a re-run is
legitimate only when the first was faulty for a stated reason, and then both
readings are reported. Where a change was worth having only if it improved the
model, a null reading removes it rather than merging it disabled, and the
recorded finding is what stops the next session from trying it again blindly.

**A change being adopted carries its treatment arm's records into the committed
store, in its own pull request.** Benchmarks write machine-local, so nothing
reaches `results/` on its own; the copy is one command against the label the
records name:

```console
uv run anthro eval promote --checkpoint <treatment-arm-label>
```

Commit what it wrote alongside the change. Merging is the acceptance, so an
unmerged pull request promotes nothing and a candidate that is dropped costs
nothing to unwind. The control arm is not promoted: it is the checkpoint the
comparison was read against, and either it is already in the store or it is a
reading of something nobody adopted.

Promote the reading the comparison was decided on, which is the sweep at its
declared sizes,
`docs/decisions/0063-the-full-sweep-decides-a-change-and-the-canonical-line-is-its-byproduct.md`
requires for that decision. Where the pull request carries something else, a
capped reading or a change adopted for a reason no benchmark measured, say so
beside the copy and leave it to review; a capped view names its realized size in
the record, so the diff already shows which it is.

## Offering A Real GUI Check

Some changes are only convincing in a real chess GUI. Automated coverage proves
protocol behavior, but command ordering, option presentation, and how a game
actually feels belong to the maintainer.

Offer a GUI check without being asked when a change alters what a GUI observes:
move selection, sampling or seeding, rating or temperature controls, position
synchronization, advertised options, protocol command handling, or engine
lifecycle. Skip it for changes a GUI cannot see, such as data pipeline,
training, evaluation, or documentation work.

To offer one, point the GUI at the working checkout with
`scripts/anthro-gui-target .`, and tell the maintainer the engine is ready to
test, which behavior to look at, and what a good result looks like. The GUI
itself is configured once and is never reconfigured per issue, so do not change
its settings or edit anything under `.venv`. The mechanism is documented in
`playable-uci.md`.

Leave the pointer aimed at that checkout while the pull request is open. Clear
it with `scripts/anthro-gui-target --clear` when abandoning the branch, and note
in the issue that the maintainer should clear it after merge if the checkout is
removed. A removed target fails loudly rather than silently serving stale code.

Maintainer GUI observations belong in the issue as findings. Treat a GUI session
as acceptance evidence, not as a substitute for automated coverage: whenever a
GUI check finds a defect, add the regression test that would have caught it.

## Publishing

Before marking a pull request ready, account for its size against the request
that produced it. Read the diff statistics with `git diff --stat main...HEAD` and
answer in the pull request body from what the diff contains rather than from
memory: the new surface it adds, and which part would go if the change had to be
a third smaller. An addition with no answer is a candidate for removal, not for a
better explanation.

This is a required answer rather than a size limit on purpose. A threshold only
teaches everyone to sit just under it, and the diffs worth questioning here are
not the ones that cross a line but the ones whose size nobody was asked to
justify.

Include `Closes #<issue-number>` so merging into the default branch closes the
implementation issue automatically. Do not close the implementation issue
manually because the pull request is ready; the merge should close it.

A task is expected to leave nothing behind, and one that does has a scope result
to explain rather than a queue to fill: the pull request says what was left and
why the issue still closes. What this change made wrong or newly reachable is
never that — it is the change's own defect and is fixed here, whatever the issue
asked for.

Remaining work that is genuinely separable and clears the filing bar becomes its
own issue before the merge, with the dependency attached. Never a comment on the
issue this pull request closes: no later session reads a closed issue, so work
recorded there is indistinguishable from work nobody found. Everything real but
below the bar goes in the pull request body under `## Findings not filed`, where
the maintainer sees it while reviewing and decides; the global maintenance burden
rule owns which is which.

For an issue labeled `verification: gpu-required`, a session without the
specified GPU may still open a ready-for-review pull request after the
implementation and all available checks are complete. The pull request must
prominently identify the pending GPU verification, including the required
environment, exact command or procedure, expected evidence, and any artifacts
to preserve. The pending check becomes part of local review, and the issue
label remains as the durable routing signal.

When substantial GPU implementation remains, a draft pull request carrying a
`Local handoff` section is how that work is passed on: the precise required
environment, completed work and verification, remaining implementation, the
exact commands to run, and expected evidence. A draft handoff is not a
substitute for splitting the issue — when the cloud-suitable and GPU-required
boundary is known during planning, prefer two issues with a dependency.

Use the configured Git and GitHub identities for commits and publication, and do
not invent a bot identity. Authorship stays with the maintainer.

AI-attribution trailers and footers are acceptable. Several agent harnesses add
them automatically, so a rule against them would be contradicted on every commit
and leaves the agent resolving a conflict it cannot win. Leave whatever the
harness adds in place.

## After Merge

Remote issue branches are deleted automatically after merge.

Keep milestone state aligned with the work it represents. When the milestone's
completion criteria are satisfied and all included work is complete or
explicitly tracked elsewhere, correct stale metadata, close the tracker, and
then close the GitHub milestone. Otherwise, leave the tracker and milestone
open and make the remaining work discoverable. Close a tracker only when all of
its milestone work is actually complete.

## Parallel Sessions

Several sessions often run against this repository at once, which is why claims
and dependencies matter here more than issue count alone would suggest.

The areas that parallelize cleanly are chess logic, data ingestion, evaluation,
and UCI. The shared architecture, data schema, package layout, and decision
records do not: two issues touching those at the same time will conflict, so
they are sequenced with a dependency rather than run concurrently.

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
9. Take a shakedown reading when the change adds or alters a benchmark.
10. Read a model change against a control arm when the change decides what a
    training run learns.
11. Promote the treatment arm's records in the pull request adopting the model
    change that produced them, when a control-arm reading decided it.
12. Account for the diff's size and new surface before marking the pull request
    ready.
13. Account for anything the task left behind before the merge closes the issue:
    fixed here, filed as its own issue, or explained in the pull request.
