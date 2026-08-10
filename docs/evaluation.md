# Evaluation

Evaluation is a core part of Anthro Chess. The project should make model
quality visible through repeatable benchmarks rather than relying on subjective
playtesting or manual chess judgment.

The goal is not to collapse all quality into one number. Different benchmarks
should answer different questions: whether the code is correct, whether the
model matches human data, whether the selected rating is calibrated, whether
timing is plausible, whether legal masking is only a guardrail, and whether the
bot remains more human-like than a conventional engine.

## Presentation

Evaluation should be presented as metric families with a small default view and
deeper diagnostics available when something regresses.

Avoid a single overall Anthro Chess score. A single aggregate would hide
important tradeoffs, such as improving move loss while hurting rating
calibration or timing.

Families cover training health, held-out prediction, legality, correctness,
rating behavior, generated play, game termination, decision decomposition,
timing, training efficiency, inference efficiency, and later additions such as
move-time coherence, human-likeness, and preference controls. The metric registry in
`anthro_chess.evaluation.results` owns the exact family and metric identifiers,
their declared directions, and their definition versions; `anthro eval metrics`
prints them.

Efficiency is deliberately separate from training health rather than folded
into it. The two are read on different terms: a training-health metric is a
statement about the model alone, while an efficiency metric is a statement
about the model on a machine under a workload, so a delta has to say which of
those moved before it means anything. Training efficiency is scoped to a run
and is not part of the end-of-run checkpoint suite; inference efficiency is
scoped to a checkpoint and is, because an opponent too slow to play against is
a product failure regardless of its other numbers.

Each family should have one to three headline metrics. Detailed slices and
diagnostics should remain available, but they should not be required reading for
every checkpoint comparison.

Example comparison table:

```text
Model   Move Loss   Rating Error   Mask Penalty   Timing Error   Human-Like
v0.3    2.41        180            0.083          0.31           0.62
v0.4    2.32        140            0.045          0.28           0.66
v0.5    2.29        260            0.041          0.27           0.59
```

The values and scales are examples. The important idea is that model comparison
should stay compact by default while preserving enough detail to explain why a
model changed.

Comparison is not only pairwise. The project should be able to see improvement
across its whole history, not just before and after one change, so results
accumulate as a durable history rather than as one-off artifacts compared on
demand.

## Benchmark Infrastructure

Benchmarks append to a **results store**; reports, comparisons, and charts are
views over that store. This inverts the obvious arrangement, where each
benchmark writes an artifact and comparison is a separate operation over files,
and it is what makes "compare against a checkpoint from a year ago" a query
rather than a task.

Results are layered like the diagnostics they contain. A small summary tier is
committed to the repository, so history is versioned with the code, metric
movement shows up as a reviewable diff, and agents read results with ordinary
file tools rather than through a service. Bulk diagnostics stay machine-local.
`docs/decisions/0014-evaluation-result-storage.md` owns that split and the
reasoning behind not adopting an experiment-tracking platform.

Every result carries a **fingerprint** identifying the series it belongs to.
Results with matching fingerprints are comparable; results without are not, and
a report should say so rather than present the difference as a change in model
quality. `docs/decisions/0013-benchmark-result-comparability.md` owns the
fingerprint contract, the rules for bridging a broken series, and how pool
generations relate to long-running comparisons.

Metric identity is a contract, not an implementation detail. Each metric has a
stable identifier, a declared direction of improvement, a family, and a
definition version, so a metric can be named in an issue or a report without
consulting a schema, and no reader has to infer whether lower is better. A
changed definition means a new identity rather than a quietly redefined series.

Artifacts should record enough provenance to recompute their own fingerprint,
and reporting should accept new artifact kinds as later benchmarks land without
restructuring. Metrics with no data dependency, such as optimizer and parameter
statistics, carry no data component in their fingerprint and are therefore
immune to changes in evaluation inputs.

Efficiency and generated-play metrics carry one further component: the
**declared workload**, the settings whose change would make a delta
meaningless. Change the ply depth a latency figure is taken at, or the
temperature a rollout is played at, and the number measures a different
quantity, so that belongs in identity just as scored content does. Sample
counts do not: generating more games, like scoring more, estimates the same
quantity more precisely. Benchmark cost is the one metric where that last
sentence inverts, and it is scoped accordingly; see "What A Benchmark Cost".

Everything else a result was measured under is a **coordinate**: recorded,
diffed, and named by a report, but never digested. The machine is one, and so
is anything a reader might want to subtract across — a training run's model
architecture, batch, and corpus are coordinates for exactly that reason. A
delta across them is interpretable rather than meaningless, so a report
attributes it instead of ending a series.

`docs/decisions/0018-workload-scoped-efficiency-series.md` owns the rule,
`0020-declared-settings-scope-generated-series.md` extends it to generated
play, including why a rollout's human prefixes are provenance rather than a
data component, and
`0021-efficiency-identity-excludes-compared-conditions.md` draws the line
between identity and coordinates.

### Where The Store Lives

`anthro_chess.evaluation.results` implements this layer and owns the exact
record schema, metric registry, fingerprint algorithm, and size budget.

The committed summary tier is one small JSON file per result under the store
root, beside the bridges that rejoin a broken series and the characterized
noise floors that qualify one. One file per record is what keeps concurrent
appends and Git merges additive; a concurrent write into the same store fails
on an exclusive lock rather than producing a partial record. The store root
defaults to `results/` in the repository and can be pointed elsewhere with
`ANTHRO_CHESS_RESULTS_ROOT`.

The detail tier is machine-local and holds per-position diagnostics, slice
tables, and generated game records. A summary record references a detail
payload by path and digest rather than embedding it, and the store refuses a
record whose payload belongs on the other side of that boundary. The detail
root resolves from `ANTHRO_CHESS_RESULT_DETAIL_ROOT`, or beneath
`ANTHRO_CHESS_RUN_ROOT` when that is where runs already live.

`anthro eval report` is the reading surface: a compact delta view by default,
with slices, provenance, per-series history, and machine-readable output behind
explicit options.

A report's row is one **series**, not one metric. A benchmark that varies a
dial across a matrix writes one result per cell, so a checkpoint holds several
readings of one metric that differ only by declared workload; showing the most
recent would present one arbitrary cell as the checkpoint's value and hide the
rest. Rows are grouped by workload and labelled with the fields that actually
tell the cells apart. The exception is a family measured under one workload
before a change and one after, which pairs, so the report names the workload
change instead of showing two half-rows that never say why. A metric declaring
no workload can still land on more than one series when the inputs underneath
it move, and there the most recent reading is shown together with how many
series stand behind it. `anthro eval bridge` records, lists, and revokes bridges.
`anthro eval noise` characterizes floors, lists them, and answers how many
games an axis needs. `anthro eval inference` measures what a checkpoint costs
to play with; see inference efficiency below. `anthro eval decisions` separates
model error from sampling error over a payload of generated games or a played
session's log; see decision decomposition below. `anthro eval puzzles` measures
the external puzzle-rating response described in the rating section, and
`anthro eval ladder` measures the self-play rating ladder and its temperature
response described beside it. `anthro eval termination` measures how a
checkpoint ends games against the human termination mix; see game termination
below. `anthro eval budget` reports held-out quality
against the training budget that bought it, joining two families rather than
defining a third; see training efficiency below. Training efficiency itself has no command, because it is
measured by `anthro train` while the run happens. `anthro eval suite` runs all
of the checkpoint-scoped benchmarks above in one sweep; see the benchmark suite
below.

`anthro eval tensorboard OUTPUT` regenerates a disposable chart view of the
store's checkpoint history. Checkpoint ordinal is its step axis, and each raw
series fingerprint is emitted as a separate TensorBoard run, so a comparability
break becomes another line rather than a continuous line through the seam.
Metric families and identifiers organize the tags and run names.

This view is deliberately less expressive than `anthro eval report`. It has no
checkpoint labels on the x-axis, no noise-floor error bars, no explicit absent
families, and no bridge semantics; even bridged fingerprints stay on separate
lines. Deleting the output loses nothing, and the output must live outside the
committed results store. Decision 0023 owns this constrained projection and
refines decision 0014's earlier prohibition on cross-version TensorBoard
history.

### The Benchmark Suite

`anthro eval suite` evaluates one checkpoint across every benchmark, and is the
default way a new checkpoint is read. It **composes** the per-benchmark
selections rather than restating any of them: a suite selection names each
benchmark's own file, the checkpoint, and nothing about how that benchmark
measures. A second copy of a declared workload would be a second thing to keep
correct, and the first one to drift would be silent.

It is a driver rather than a shell script because a sweep is long enough for
four things to matter that a script cannot supply.

**The whole plan resolves before any of it runs.** An unreadable selection, a
pool that does not exist, a dependency on a benchmark the sweep does not
include, or a step configured to discard output another step reads all fail in
the first second. This is the same rule training cadences follow: a run that
will fail should fail before it spends time, not after.

**Ordering is enforced rather than documented.** Decision decomposition reads
the games the rollout generated, so the suite orders it after the rollout and
refuses a sweep where it could find nothing to read. The games are handed over
as a payload in the sweep directory rather than in memory, which is what lets a
resumed sweep retry the decomposition without replaying the rollout.

**Recording is decided per benchmark within one sweep.** A sweep that commits
one baseline reading and leaves the rest as evidence about the instrument is
the normal case, not an exception, so the decision belongs to each step rather
than to the sweep.

**A sweep that dies late keeps what it already read.** Each step's outcome is
written to a machine-local ledger as it finishes, and a failed step does not
end the sweep: the independent benchmarks after it still run, and only the
steps that read its output are skipped. A resumed sweep refuses a ledger
belonging to a different plan, so a reduced sweep can never continue a full one
or another checkpoint's.

The **reduced sweep is the default and the full sweep is opt-in**, because a
sweep measured in hours is not a default anyone will run on a new checkpoint.
Reductions are confined to sample counts — scored view sizes, seeds, games per
position, resamples, puzzles per rating, measured decisions — and never touch a
grid, a dose, a temperature, or a ply limit, since those decide *what* is
measured and shrinking one would report a different quantity rather than the
same one less precisely. A smaller view is still its own data component, so a
reduced sweep is a separate series rather than a partial down payment on a full
one, and the scale is named in the output and in the ledger for that reason.

Not every view is a sample count, and the exception is the human reference the
curve comparisons are smoothed against: at a neighbour-count bandwidth its size
is the smoothing radius rather than a sample size, so it is declared by the
benchmarks that read it and left alone by both scales.

That rule has a consequence worth stating, because it looks like an omission
otherwise: **a benchmark whose cost is a grid rather than a sample size has no
reduction at all**, and belongs to the full sweep alone. The rating ladder is
the case. Its cost is quadratic in its seat count, the seats are the
measurement, and its only sample dials — seeds and openings — are already at
their floor while it is still playing hundreds of games. Shrinking the seat
grid instead would not merely be less precise: one joint fit places every seat
on a single internal scale, so a ladder fitted on a different set of seats
cannot be read against this one at all. A step declares which scales include
it rather than being nominally reduced and still unaffordable, since a reduced
sweep nobody can run is worse than one that says what it left out.

The suite adds no measurement of its own and registers no metric. Decision
decomposition is the one step it cannot commit, because that family has no
result kind: a decomposition is read per cell of the dials it was made under,
and which cell a committed series would follow is not yet decided.

`anthro_chess.evaluation.benchmarks` owns what a benchmark declares, the
registry of them, and the one path that resolves a selection, runs one, and
assembles what it recorded. The suite and every command that runs a benchmark
from a selection file are callers of it, so neither holds its own idea of what
a benchmark takes, roots, or raises. A benchmark measures and adds its readings
through `anthro_chess.evaluation.recording`; where those land, and whether they
are committed at all, is the driver's rather than the benchmark's.
`anthro_chess.evaluation.suite` owns the suite schema, the ordering rules, and
the ledger format; the shipped sweep lives beside the selections it composes
under `configs/evaluation/`. Every benchmark selection inherits
`anthro_chess.evaluation.selection.CheckpointSelection`, so which checkpoint a
sweep replaces, and what a reading calls it, are declared once rather than
repeated by each schema.

### What A Benchmark Cost

Every benchmark that records a reading also records what the invocation cost,
as a single wall-clock measurement in its own committed record. This exists
because scope decisions are made on these numbers — which benchmarks a reduced
sweep can include, whether a step has an affordable reduction — and before it
existed the numbers lived only in comments that nothing could contradict, by
which point they had drifted by an order of magnitude or more.

The driver records it rather than the sweep, because a single-benchmark
invocation is how most readings are taken and a cost that only existed inside a
sweep would miss them. The measured window is therefore the invocation in
process, from the driver's first statement to the moment the recording has
assembled everything it will commit, and a sweep's total is the sum of its
steps plus what the driver pays.

Series identity is where this metric departs from every other workload-scoped
one. The declared workload digests the benchmark's whole resolved
configuration, because a sample count decides how much work was done and the
work is the quantity being measured — the reverse of the rule that keeps sample
counts out of a latency series. Two things are taken out of it: the model
selection and its label, since the checkpoint is the coordinate a cost line
varies along, and the machine prefix of every artifact path, since the artifact
is the same work wherever it is rooted.

Read a cost delta against a characterized execution floor. A shared machine
moves these numbers much further than a model change does, and nothing in a
record says the machine was busy; decision 0031 carries the measurements.

`anthro_chess.evaluation.cost` owns the record and the workload normalization;
`docs/decisions/0031-committed-benchmark-cost.md` owns the reasoning.

### The Checkpoint Evaluation Runner

`anthro eval run` scores one compatible checkpoint over a deterministic view of
the frozen pool and appends the result. It is the canonical end-of-run reading,
and it is a library before it is a command, so in-training evaluation at
declared cadences calls the same entry point over a smaller view instead of
growing a second implementation that has to be kept consistent with this one.

A **leakage check runs before any scoring**, so a checkpoint that trained on
these games fails loudly rather than producing a plausible number nobody
re-examines. When the checkpoint trained on the same normalized corpus the pool
was drawn from, internal game ids are comparable and the check reads two
columns. When the corpora differ an id means nothing, so games are compared by
what they contain. A training corpus this machine cannot read is an error with
a configurable override, not a skipped check.

Which sliced series are **committed** is a deliberate, bounded choice, because
only a committed series can be compared over the life of the project. Overall
prediction and legality headlines are committed; so are move loss and mask
penalty per phase, move loss per default rating band, and mask penalty per rule
case. Phase is committed on the evidence that held-out mask penalty varies
severalfold between opening, middlegame, and endgame positions: a pool-wide
average sits between those populations, and a comparison that does not hold
phase fixed reads a shift in game-length or phase composition as a legality
change. Everything else — color, legal-move-count buckets, cross-conditioning
tables, per-position records — stays in the machine-local detail tier.

Rating-band series are committed only when the run uses the default bands. A
changed band boundary is a different measurement rather than movement in an
existing series, so it reports into the detail tier instead of quietly
continuing a line.

### Shakedown Readings

A benchmark is not finished when its tests pass. Fixtures prove the code runs;
they cannot prove the benchmark measures anything, because a fixture returns
whatever it was built to return. So a new benchmark takes a **shakedown
reading** on real checkpoints before it lands, and the reading is reported
where the change is reviewed.

The reading is on **two checkpoints far apart in one training run**, with the
expected direction stated before the run rather than after it. One reading
only shows the benchmark completing. Two show whether it *discriminates*, which
is the failure that matters: a benchmark returning the same number early and
late in training is broken, and nothing in a fixture suite will say so. Stating
the expectation first is what makes the reading falsifiable; a number
interpreted afterwards can always be made to sound reasonable.

A shakedown is deliberately cheap. Sample counts are not part of series
identity, so a small view reads the same metric the headline configuration
does, and a benchmark whose full grid is expensive should be shaken down on a
reduced one. Cost measured at realistic scale is a useful thing to report
alongside it, since a benchmark nobody can afford to run is a different kind of
defect.

A shakedown reading **is not recorded**. Every benchmark command that appends
to the store takes `--no-record` for exactly this, and the reading is evidence
about the instrument rather than about the model. It is not quoted as model
quality, does not satisfy an evidence gate stated anywhere in these docs, and
does not become a baseline. Readings that disagree with the expectation are the
valuable ones and are reported rather than quietly re-run; the outcome of a
shakedown can legitimately be that the benchmark is wrong.

`docs/issue-workflow.md` owns when a change needs one and how a session without
checkpoints routes it.

## Human-Reference Curve Comparisons

Several benchmarks turn out to have one shape: measure something on generated
games, measure the same thing on human games, and compare the two as a function
of rating. Opening repertoire, book depth, game length, result and termination
patterns, and repetition all fit it. Building it once avoids four near-duplicate
implementations that drift apart.

The shape is: a human reference curve estimated from the pool, a model curve
measured across a configured-rating grid, a distance between them reduced to one
scalar for the headline and history, and the curves themselves retained for
drill-down.

Rating is treated as continuous rather than banded. Binning is only forced when
estimating a whole categorical distribution at a point; a single quantity as a
function of rating needs no bins, because each game contributes one observation
at its own rating. Curves are estimated with a nearest-neighbour smoother, whose
span adapts to the human data's uneven density across the rating range.

Two constraints keep the comparison honest.

Smoothing biases a curve toward flatness in proportion to its bandwidth. The
model side has uniform density by construction and the human side does not, so
smoothing each at its own bandwidth would put an artifact into their difference
near every peak. **Apply the same local bandwidth to both curves at each
evaluation point**, taking the one the human data forces.

The bandwidth is part of the measurement, not a per-run choice. Selecting it
per run would mean two checkpoints were measured differently and their series
would not be comparable. Choose it once from the human reference by
cross-validation, then freeze and declare it; changing it is a benchmark version
bump.

Because the bandwidth is a neighbour count, **the reference size is part of
it**. The same declared count spans a wider rating range on a smaller reference,
so shrinking the reference does not sample the curve more coarsely — it smooths
it more heavily, until every evaluation point is estimated from the same games
and the grid resolves fewer points than it plots. The reference is therefore
declared at a size the grid can resolve, is neither shrunk by a reduced sweep nor
left uncapped at full scale, and joins the declared workload so two readings
smoothed differently cannot share a series.
`docs/decisions/0037-the-human-reference-is-bandwidth-not-sample-size.md` owns
that rule.

Report the effective local sample size alongside the curve, so a difference
where the human reference is thin is not read as the same strength of claim as
one where it is dense. Report the **realized** bandwidth per evaluation point
too, rather than the declared neighbour count: the count is the same at every
reference size and says nothing about a particular reading, while the rating
span it reached is what tells a reader whether the grid resolved its points.

Both a rating-conditional and a rating-free reading come from the same pass,
and they answer different questions. The pooled reading asks whether the model
behaves like humans at all. The curve asks whether it responds to rating the
way humans do. They dissociate in a specific and likely way: a model that
matches the pooled human distribution with a flat curve has learned the average
human and is ignoring its rating input. That is the behavioral form of a rating
dependency test, and unlike the corrupted-conditioning form it survives
discrete sampling into whole games.

The pooled reading averages each side's own curve across the rating grid rather
than pooling raw games. The two sides are different rating designs — a human
corpus concentrated in the middle of the range against generated games spread
evenly across it — so pooling raw games would report that design difference as
a distance. Averaging the curves keeps both sides on one rating mixture and
keeps the smoothing bias cancelling, for the same reason the shared bandwidth
does point by point.

A distance also needs a reference level rather than only a floor, and the two
are not interchangeable. A floor says how far a number moves between two runs
when nothing changed, which is what qualifies a delta between checkpoints. A
level says what the number reads at when there is nothing to find: two finite
samples of one population never agree exactly, so a distance is never zero and
a curve is never exactly flat. Both come from the comparison's own resampling,
and both travel with the result, since neither is a property of a series that
could be characterized once and looked up.

The conditional and the pooled reading each have their own level and their own
floor, so a report shows both per reading rather than sharing one across the
pair. A level or a floor rendered beside the other reading's distance is read as
though it belonged to it, which is how the first full suite reading drew a wrong
conclusion from a verdict that was correct.

`anthro_chess.evaluation.curves` implements this shape and owns the estimator,
the declared bandwidth and its offline selection, the distance reduction, and
the artifact each benchmark writes: curve points as data in the detail tier,
and the scalar distances, their floors, and their reference levels in the
committed summary tier.

## Noise Characterization

> **Superseded in design, partly in code.**
> `docs/decisions/0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`
> replaces the four kinds, the scope rules and the stored characterizations
> described below with one dispersion per reading, combined at comparison time.
> The combining has landed: a reading carries its own dispersion and a delta is
> floored by the two in front of it. The kinds, the scope rules and the
> characterization store are still here for the sources a reading cannot
> measure, and this section still describes those. It is rewritten as the rest
> of that migration lands, and the tracker issue holds the order.

A delta is not a finding until it is larger than the noise in the measurement.
Reports should annotate every change with the noise floor it did or did not
clear, and a delta inside the floor should be visible but marked rather than
hidden, so a consistent small regression is not lost.

Four sources of noise are distinct, and conflating them is the usual mistake.
Each one licenses a different comparison, so the useful way to name them is by
the question they answer rather than by how they are computed:

- **evaluation noise**: the same checkpoint re-measured. This is the floor that
  qualifies a **delta between two checkpoints**, which is the project's central
  comparison. Deterministic offline metrics over a frozen pool have none;
  rollout metrics have a lot, driven by seeds.
- **data-sampling noise**: how much the metric would move on a different draw of
  the same size from the same population. It sizes a **view or a pool** and does
  *not* qualify a checkpoint delta. Estimable by bootstrapping from a single
  run, so it costs nothing.
- **training noise**: the same configuration trained from a different seed. This
  qualifies a **configuration change** rather than a checkpoint delta, and it is
  the expensive one, since it needs several training runs.
- **execution noise**: the same checkpoint timed again on the same machine. This
  qualifies a **delta between two efficiency readings**, and it is the one
  source that cannot be estimated from numbers already computed, because what
  varies is the machine rather than the sample.

Which of these a reading needs follows from what is being claimed, and the two
questions are easy to conflate. Whether one run improved between two of its own
steps is a checkpoint delta. Whether a change to the model, the data, or the
training setup improved anything is a configuration change, and clearing a
sampling or evaluation floor does not establish it: the two arms differ by their
initialization seeds as well as by the change. **Regression Comparisons** below
holds what a claim rests on, and decision 0029 holds the measurement that settled
it.

The first two coincide for generated play, and that is worth stating plainly
because the definitions above read as if they never could. A rollout has no
fixed data to re-measure on — the games *are* the draw — so bootstrapping the
generated games and re-running under another seed estimate the same quantity.
That is why a generated-play floor is evaluation noise even though it is
computed by a bootstrap, and why rollout metrics need no separate expensive
characterization.

They coincide only where re-running redraws the games. Greedy seats replay
theirs, so a temperature-zero reading is the deterministic case the first bullet
already names: another seed reproduces it exactly, and its evaluation noise is
zero rather than small. A bootstrap over those games still returns a number, and
that number is data-sampling noise wearing an evaluation floor's label — it says
how far a different draw of games would have landed, which two checkpoints read
on the identical games are not exposed to. Such a reading therefore states a
floor of zero rather than estimating one, and records that it was stated.
`docs/decisions/0032-a-replayed-reading-has-no-evaluation-noise.md` owns the
rule and why reporting no floor at all would understate what is known.

A floor that qualifies a delta must exclude anything the two sides of that delta
share. Two checkpoints are compared against the *same fixed* human reference, so
the reference's own sampling error is common-mode and cancels; including it can
only inflate the floor and hide real movement. Only the side being compared is
resampled. How much this matters depends on how thin the reference is relative
to the bandwidth: on a reference of a few hundred games it widened floors
noticeably, while at the declared bandwidth over the frozen blitz pool the
difference was under a percent. It is excluded because it is not part of the
question, rather than because it is always large.

All four reduce to one reportable quantity: the spread of the metric across
replicates of that noise source. A **floor** is that spread expressed as a
delta, because a delta is what a report shows and a standard deviation is not
directly comparable to one.

A reading stores the spread and never a floor built from it. The variance of a
difference is the sum of the two variances, so the floor of a delta is combined
from the dispersion each of the two readings carries, each bounded first. That
matters because the two readings of one metric do not share a spread: the two
committed to this repository differ by up to two orders of magnitude on the same
metric, and a floor computed inside either one assumes the other matched it.
Where the two do agree the arithmetic reduces to the familiar `sqrt(2)`. A
characterization is the case where they agree by construction, since its
replicates are draws of one quantity, so a stored floor keeps that factor.
Coverage is applied at comparison time rather than stored on a reading, because
a floor is a claim the comparison makes; `anthro_chess.evaluation.results` owns
the arithmetic, stored inputs, and lookup.

### The Spread A Floor Is Built From

A measured spread is an estimate, and a floor built directly on it is too
narrow about half the time — which is exactly when being too narrow does
damage, because a floor exists to stop noise from reading as a finding. So the
spread is not used as measured. Every floor is built from a conservative upper
limit on it, the chi-squared bound for the degrees of freedom actually behind
the estimate, and both quantities are stored: the measured spread describes the
machine or the sample, and the bound is what qualifies a delta.

That makes a floor a tolerance bound, carrying two declared numbers that answer
two different questions. **Coverage** says what proportion of same-weights
deltas the floor covers if the spread were known exactly. **Confidence** says
how sure the bound is that the spread is no larger than assumed. They multiply.

What counts as a degree of freedom is the independent replicate, not the
number of values in hand, and the distinction decides whether the bound means
anything. Bootstrap resamples are drawn from one sample, so the **games** are
the replicates and the resample count is not; readings repeated inside one
process share a warm allocator and a compiled kernel, so the **processes** are
the replicates and the readings are not. Counting either of the cheap numbers
would buy apparent certainty about the spread for free, which is the failure
the bound exists to remove. `docs/decisions/0026-conservative-dispersion-bounds.md`
owns this rule.

The bound is severe at small replicate counts and that is the honest reading:
three or four replicates say very little about a spread. It also means adding
replicates is the only way to narrow a floor without weakening what it claims,
which is what the characterization defaults are chosen against.

One thing the bound does not cover, and it is the larger term. The bound
describes how well the spread *within* a characterization is known. A report
compares readings taken later, when the machine is in a different thermal and
contention state, and no arithmetic on the characterization's own replicates can
reach that drift.

Measured rather than assumed: one configuration characterized and then read back
on a quiet machine had 3.8% of its same-weights deltas clear their floors, and
the identical configuration after an hour of sustained benchmarking had 15.9%.
Machine state moved the result four times further than replacing the point
estimate with its bound did, and it moved the median and mean latency series as
much as the tail. So an execution floor is re-characterized when conditions have
plainly moved rather than treated as a constant of the hardware, and whether
storing one for later lookup is the right shape at all is an open question
rather than a settled design.
`docs/decisions/0026-conservative-dispersion-bounds.md` holds the evidence.

The estimators differ even though the reported quantity does not. Data-sampling
noise is bootstrapped by resampling the **games** a run scored, since positions
within one game are far from independent and resampling them would report a
floor several times too narrow. Evaluation and training noise are read from
repeated measurements the store already holds, which is what keeps the
expensive kind a matter of recording several short runs rather than building a
second harness.

Noise characterizations are stored in the results store under the same
fingerprint rules as any other measurement, so they invalidate on the same
terms rather than lingering as stale constants. A floor characterized on a pool
that has since been regenerated stops matching and the report says the floor is
unknown, which is the honest answer.

### Training Noise

Training noise should be characterized early, while runs are short. It is the
most valuable of the three and the only one that becomes harder to obtain over
time: once runs are long and expensive, several repeat runs stop being
affordable, and the project loses the ability to distinguish a small improvement
from seed luck for the rest of its life.

The floor is a property of the training configuration its replicates shared
rather than of the pool they were scored on, and the series fingerprint carries
nothing about the training run by design — decisions 0018 and 0021 keep it out
so that a delta across model size stays interpretable. A training
characterization therefore records the training identity it was measured under,
and a report applies it only to a delta that identity describes.

What that takes is not what an execution floor takes, because the two scopes are
different sorts of thing. A machine is a condition a reading was taken under, so
a delta spanning two machines is described by neither. A training configuration
is a **null distribution** — the spread a different seed of it would have
produced — and the question a delta asks is whether its other operand falls
outside that spread. That is the control-arm comparison, whose two sides differ
in configuration by construction, so requiring both to match would refuse the
one comparison the floor exists for. One operand carrying the characterized
configuration is therefore what makes the floor apply; a delta describing
neither is reported as unknown, and where both operands carry characterized
configurations the widest of the two floors binds. A reading recorded without an
identity carries no configuration to match, and replicates that do not all share
one are refused rather than characterized.
`docs/decisions/0040-training-noise-floors-are-scoped-to-the-configuration-they-measured.md`
owns that rule and what the scope deliberately leaves out.

### Execution Noise

Timing is where a floor matters most and where the usual estimators do not
apply. Two readings of one checkpoint taken minutes apart differ, and nothing
about the model, the data, or the seed moved; a report with no floor can only
say the number changed, so sub-percent jitter renders as a regression.

The noise source here is the machine — scheduler contention, thermal state,
other processes, allocator and kernel warmth — so it cannot be bootstrapped out
of an already-measured latency. It is characterized by **measuring again**, and
by measuring in more than one process. A reading a report compares was produced
by its own invocation, which paid its own model load and its own lazy kernel
compilation, so a floor built only from repeats inside one process omits the
component most likely to dominate. Repeats within a process are still taken, and
what they add is the answer to whether that cheaper form of replication would
have sufficed on this device; that share is reported beside the floor rather
than folded into it. Cold-start metrics take one reading per process, because a
reload inside a warm process is not a second cold start.

The process count is therefore what the floor's dispersion bound rests on, and
it is the one setting that trades measurement time for resolving power. Cutting
it does not produce a narrower floor, only a less certain one. The default is
set where the bound stops improving quickly enough to be worth another model
load, and the code owns the exact value.

The floor is a property of a machine and a workload rather than of a checkpoint.
Decision 0018 deliberately keeps the machine out of an efficiency series so that
a latency history stays continuous across a hardware change, which means the
series fingerprint alone cannot stop one machine's floor from qualifying
another's delta. An execution characterization therefore carries the execution
it was measured under, and a report applies it only where that environment
matches on both sides of the delta; anywhere else the noise is reported as
unknown. `docs/decisions/0025-machine-scoped-execution-noise-floors.md` owns
that rule.

Nothing measured during a characterization is appended to the results store.
The readings are evidence about the machine rather than about the model, and
recording them would let a checkpoint's history depend on how often its noise
was characterized. `anthro eval noise sample` takes one process's readings and
`anthro eval noise characterize --kind execution` spreads them across processes
and records the resulting floor;
`anthro_chess.evaluation.execution_noise` owns the procedure.

Because a data-sampling spread costs only a resampling of numbers a run already
computed, the checkpoint evaluation runner produces its own and attaches it to
each measurement it records. Attaching it is what lets two readings of one
series each carry their own: a spread filed against the series would be one
number where the comparison needs two.

Two checkpoints scored on one frozen set share their draw, so the variance of
their difference is smaller than the sum of the two variances by the covariance
between them. The combined floor drops that term and is about 1.9x wider than an
estimator that keeps it, which costs real improvements rather than inventing
them.
`docs/decisions/0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`
records that width as the accepted price of a bar that is always available and
always means one thing, and supersedes the records that measured it.

A delta is judged against the widest floor that applies to it, since a finding
has to clear every noise source, and the report names which one that was. A
delta inside its floor is still shown with its value, so a small regression that
repeats across checkpoints stays visible instead of being filtered away.

**A floor has to describe both operands.** Widest-of-the-two chooses between two
descriptions of the same delta, so it needs two. Where a floor is attached to a
measurement it belongs to that reading, and a kind one side attached and the
other did not qualifies one operand rather than the difference — the report
withholds it and names the kind it declined. Nothing says the missing side is
quieter, and a benchmark that withholds a floor per reading is saying that side
is not an estimate at all. A characterized floor is a property of the series, so
it cannot be one-sided either.

Withholding reaches the verdict, not only the note. A delta cannot be reported
as clearing every noise source while one of them is a kind this comparison could
not size, so such a row is unknown however comfortably it clears the floors that
remain. A delta *within* one of them is still within it, since a delta inside
any floor is not a finding whatever else went unmeasured.
`docs/decisions/0036-a-one-sided-floor-does-not-qualify-a-delta.md` owns the
rule.

**No floor at all is two situations, not one.** A floor may be missing because
nobody has characterized it yet, which is work somebody could do, or because the
metric counts something resampling cannot estimate, which is not. A metric of
the second kind declares why in the registry, and a report renders it
`unqualifiable` rather than `unknown`. Reporting both as unknown sets a reader to
work that cannot be done, and it is the same ambiguity as a floor rendering as
exactly zero.

That second ambiguity is why a bootstrap that could not move a quantity reports
no floor rather than a floor of zero. What the resample observed is that *this*
draw could not move the number, which is not the observation that nothing could:
a quantity identical in every unit scored reads that way at any sample size, and
the wider draw that would move it is exactly the work `unknown` points at. A zero
would instead clear every later delta, which is the failure a floor exists to
prevent. The genuine zero is the *stated* one, where re-measuring replays the
same games, and a reading records that it stated rather than estimated.

That declaration rules out one estimator rather than every floor. It says
resampling the units a reading scored cannot estimate this metric's dispersion,
so no data-sampling floor can exist for it — but evaluation and training noise
are read from repeated measurements instead, and either still describes such a
metric. A report refuses only the sampling floor and judges the delta against
any other kind it has.

Sampling-noise estimates are also what size the evaluation inputs. A
conservative independent-input estimate is what a benchmark is sized from before
any checkpoint has been read, and a reading's own measured spread replaces it as
soon as one exists. Either shrinks with the square root of the units behind it,
so how many games an axis needs in order to resolve an effect of a given size is
computable rather than guessed, and `anthro eval
noise plan` computes it from the newest reading that measured its own spread
over a counted sample. No benchmark-level resolution constant is declared or
kept current for it.

## Benchmark Data Layers

Benchmark inputs are layered as partition, pool, and views. Keeping them
separate is what lets many benchmarks with different needs share one set of
evaluation inputs instead of accumulating a tailored dataset each.

The **partition** decides what a game may be used for. `test` is held back from
training entirely; `docs/data.md` owns the split contract.

The **pool** is the `test` partition materialized as one versioned, checksummed
artifact with its own manifest and coverage statistics. It carries no
per-benchmark tailoring, and it is a regenerable pipeline output rather than
committed data. Its manifest records source, split recipe, schema,
preprocessing, action, encoding, and benchmark versions, the selected game ids
and their content hashes, and a build-time overlap check against the train
split. Coverage statistics report ply counts, results, clock presence, and
position counts by phase, color, legal-move-count bucket, and rating band, so a
thin slice is visible before a benchmark reports a number computed from it.

**Views** are per-benchmark deterministic selections over the pool: filtering by
ply count or rating presence; projecting to prefixes; subsampling by hash rank.
Each benchmark records its resolved view spec,
including the digest of the selected game ids, in its own artifact. Views are
derivations, never new stored data. A benchmark needing something the view layer
cannot derive is a signal that the field belongs in the normalized schema.

Benchmarks that must run quickly subsample in their own view rather than forcing
a smaller pool, so evaluation cost does not grow as the corpus does.

Representativeness and frozenness belong to different things. The pool recipe is
uniform and unstratified, so its composition tracks corpus composition
automatically. Frozenness is a property of a benchmark version: when corpus
composition changes materially, regenerate, cut a new pool version, and
re-baseline. Comparisons are valid within a version, and a report should refuse
to compare across versions rather than present the difference as a checkpoint
regression.

Pool versions are **generations**, and each is a superset of the last. Split
assignment is stable under corpus growth, so appending games preserves every
game an earlier generation contained and an earlier measurement stays
reproducible on the subset. Removing games, rejecting previously accepted games
through a filter change, or changing the split seed destroys that and ends the
affected series permanently.

Once a generation is designated as the **core**, benchmarks report against both
it and the current full pool. Core gives one continuous line for the rest of the
project; current gives more statistical power on a line that restarts at each
generation. Current is the number that answers how good a checkpoint is, core is
the number that answers whether it improved over the long run, and sustained
divergence between them for one checkpoint is the visible symptom of core
overfitting.

A small set of retained **anchor checkpoints** is re-scored whenever a generation
is cut, so the new generation overlaps the previous one and a shift at the seam
is attributable to the pool rather than mistaken for a model regression.

Comparing checkpoints on the pool applies selection pressure to it over time.
That is accepted rather than designed away, and is why the pool is drawn from a
partition the training loop never consumes. Over a long project the pressure on
a fixed core is more than mild, which is the second reason to keep the growing
current view alongside it.

See `docs/decisions/0011-held-out-test-partition.md`,
`docs/decisions/0012-derived-evaluation-views.md`, and
`docs/decisions/0013-benchmark-result-comparability.md`.

## Evaluation Layers

Different checks belong at different points in development and training.

### Cadence And Cost

Getting feedback early matters. A training run that is going badly should say so
within the hour rather than at the end, which means moving measurements earlier
wherever they are affordable.

The tempting way to organize that is a tier taxonomy, with named bundles of
benchmarks at each level. It collapses into something simpler once two
independent axes are separated: **which metric** is computed, and **how much
data** it is computed over. Most of what differs between an early reading and a
late one is the second, and the same metric at two data sizes is one question
asked at two precisions rather than two different questions.

So cadence is a schedule, not a class of benchmark. Each entry says when it
runs, which metrics it computes, and what view it computes them over. Each
metric declares a cost so an unaffordable pairing fails loudly. Series
separation is automatic, because a smaller view produces a different fingerprint
and can never be plotted on the same line as a full one.

Some measurements are cheap at full precision rather than cheap because they
were shrunk. Optimizer and parameter statistics have no data dependency at all.
A model's exact policy at a fixed position is one forward pass, so distribution
comparisons over early-game positions are exactly computable rather than
estimated from rollouts. These belong at frequent cadences without any loss of
precision. Rollout metrics are the opposite: irreducibly sampled, with the
number of games generated as the only dial.

View size should be declared explicitly rather than resolved from a compute or
time budget. An adaptive budget would make the same cadence resolve differently
on different machines, producing inconsistent fingerprints and noise floors for
what is supposed to be one series.

The **end-of-run suite over the test pool is canonical**, and every metric
appears there in full form. Earlier readings are previews of it. A preview view
may subsample and may not filter, which is what keeps it an unbiased estimate of
the canonical quantity with wider error bars instead of a different measurement
needing a documented conversion.

The schedule is declared in a training run's own configuration and resolved
before the first optimizer step, so an unknown metric, a missing view, or an
unaffordable pairing fails before a run spends time rather than after. Cost is
priced in scored positions per optimizer step, amortized over a cadence's
interval and compared against a declared budget. Counting in positions rather
than seconds is what keeps one schedule resolving identically on two machines.

A metric a training run cannot compute is rejected by name rather than silently
skipped. Generated-play metrics are the clearest case: they need whole games
rather than a pass over stored positions, so they belong to a post-training
benchmark and not to a cadence. Efficiency metrics are rejected for a different
reason: they pass over no view at all, so they look free, but taken beside a
training step they would report contention with that step rather than a property
of the checkpoint.

In-training readings are written to the results store like any other benchmark
result, with a preview and its canonical counterpart carrying the same
checkpoint label and different fingerprints. Held-out previews and training
health are recorded as separate results, because a training-batch statistic
carries no evaluation inputs at all. `anthro_chess.training.cadence` owns the
schema, the cost model, and the budget default.

### Unit And Integration Tests

These should run when code changes.

They should cover:

- chess-rule behavior;
- board reconstruction;
- incremental position synchronization, replacement, and takeback behavior;
- legal move generation;
- model-facing encodings;
- data parsing and preprocessing;
- sequence construction and causal-mask behavior;
- clock-state simulation;
- runtime legal masking;
- random-stream lifecycle and explicit-seed reproducibility;
- preservation and correct invalidation of process, game, and history caches.

Sampling tests should not assert that two uncontrolled random runs happen to
differ. Inject the entropy source or use explicit seeds. Required relationships
include temperature-zero seed independence, identical fixed-seed reproduction,
different controlled seeds exercising distinct samples on a fixture
distribution, `position` updates preserving the active stream, and
`ucinewgame` following the configured fresh-or-fixed game-seed policy.

### Training-Time Metrics

These should run during normal training and validation.

Default metrics should include:

- validation move loss;
- validation timing loss when timing is enabled;
- illegal-move mask penalty;
- rating-sliced move loss;
- timing-sliced loss when timing is enabled.

The initial move-validation implementation lives in
`anthro_chess.evaluation`. It consumes ordinary loader batches, evaluates raw
action logits with the explicit action-loss and rating-presence masks, and uses
the aligned legal actions from the exact chess layer. It reports both raw move
loss and move loss after exact legal masking, allowing a direct comparison with
uniform-over-legal selection while preserving the raw-logit legality
diagnostics. Its versioned structured result is the source of truth for exact
metric fields and rating-band boundaries.

In-training previews run against the `validation` split rather than the test
pool. Running them against the test pool would let those readings influence when
training stops and which checkpoint is kept, which is precisely the selection
pressure the held-out partition exists to prevent. Because both splits are
uniform hash assignments over the same corpus, a validation-split reading is an
unbiased estimate of the same population quantity the test pool measures, so the
early peek costs nothing in leakage.

The one caveat worth stating when reading them together: validation numbers
drift optimistic over a long run precisely because checkpoint selection presses
on them. Previews are most trustworthy early and increasingly flattering late,
which is the honest reason the canonical reading is on the test pool.

### Per-Step Health Metrics

A small set of measurements can run every optimizer step, subject to a hard
rule: they must be derivable from tensors the training step already computed. No
extra forward pass, no extra data loading. They should also accumulate on-device
and synchronize only at the existing logging interval, because the cost of cheap
telemetry is usually device synchronization rather than arithmetic.

These run on training batches, so they are contaminated and are not quality
measurements. They belong to the training-health family and answer whether
something is going wrong right now, not how good the model is. They must never
share a series with held-out metrics.

Optimizer and parameter statistics are the strongest candidates, since gradient
norm and update-to-weight ratio distinguish divergence, exploding gradients, and
a dead learning rate from each other, which loss alone does not. Metrics
computed from training-batch predictions are weaker, because cross-entropy
against a legal target already moves when the model degrades; they add
specificity rather than early warning.

The two implemented statistics sit on opposite sides of that rule, which is
worth knowing when reading them. Gradient norm reads gradients the backward
pass just wrote, so it runs every step and reports both the value at the
reported step and the interval's maximum, which is what catches a spike between
two logging points. The update-to-weight ratio needs the parameters from before
the update, so measuring it every step means a permanent parameter-sized shadow
copy; it is measured on reported steps instead, and its cost is paid only
there. Measured instrumentation time is reported with the run rather than
assumed to be free.

### Periodic Benchmarks

These should run less often than normal validation.

Useful periodic benchmarks include:

- held-out move distribution checks;
- timing distribution checks;
- move-time coherence checks;
- legality diagnostics on rule-sensitive position slices;
- early rating-calibration checks;
- preference-control checks when preference controls exist.

### Post-Training Benchmarks

These can be slower and should run on promising checkpoints.

Useful post-training benchmarks include:

- self-play rating ladders;
- fixed engine-anchor matches;
- rollout distribution tests;
- full simulated clock-survival tests;
- human-likeness benchmarks;
- preference-control benchmark suites;
- regression comparisons against previously accepted checkpoints.

### Correctness Gates And Benchmarks

Training sanity checks and model benchmarks answer different questions and
should remain complementary. Input inspection, dependency tests, and tiny
overfitting establish that the pipeline is capable of learning the intended
targets. Frozen held-out metrics and later rollout benchmarks establish whether
that learning transfers to behavior the project values.

A tiny sample can be memorized even when inputs are corrupted or meaningless.
A benchmark can also report stable numbers for a consistently miswired
pipeline. The first training implementation should therefore pass the staged
correctness protocol in `docs/training-and-runtime.md` before its benchmark
results become the accepted baseline.

After that baseline exists, routine model changes should use focused structural
checks plus comparisons on frozen evaluation inputs. Repeat the full staged
protocol when a change alters a foundational data, encoding, alignment, model,
or loss contract, rather than for unrelated software changes.

### Dependency Tests

A dependency test checks that a conditioning input actually changes model
behavior in the intended direction. Ordinary metrics cannot do this. Loss
sliced by a context value measures how hard those examples are to predict, not
whether the model reads the input, and it looks unremarkable on a model that
ignores the input entirely.

The basic form evaluates frozen held-out examples under the true conditioning
value and again under corrupted conditioning, such as a shuffled value, a fixed
constant, or explicit absence. A conditioning input that the model uses should
show clearly worse held-out prediction when corrupted.

The absent treatment answers a different question from the other two when the
training corpus never contained rating-absent positions. Removing the input
then measures how the model handles an input distribution it never saw, not how
much it relies on the value, and a large absent degradation beside negligible
shuffled and constant degradations says the model reacts to rating *presence*
far more than to rating *value*. Read the three together rather than treating
absence as the strongest form of the same test.

Direction matters as well as magnitude. Evaluating each context slice under
each conditioning value produces a cross-conditioning comparison whose best
result should fall on the matching pair. This distinguishes a model that merely
reacts to the input from one that has learned its intended meaning.

Dependency tests belong to the correctness family rather than the quality
family. They should run on frozen inputs whenever a conditioning contract
changes, and their results should be interpreted against training maturity: an
undertrained checkpoint can show weak dependency because it has not yet learned
the conditioning, not because the input is miswired.

Rating conditioning also supports a within-game form. Held-out prefixes at one
rating can be split by how strong the play so far has been, then compared to see
whether the predicted distribution shifts to compensate. Human games contain
this pattern, so a model that treats rating as a static prior and one that
tracks it across a trajectory should be distinguishable. Both outcomes are
useful to know.

Strength of prior play needs a proxy, since no engine is in this loop. The
implemented one is the model's own conditioning: for each earlier decision by
the player to move, how much better the high-rating conditioning explains the
move actually played than the low-rating conditioning does. Positions at one
stated rating are then split at the median of that prefix signal, and the two
halves are compared on how far the policy at the true rating leans toward the
high-conditioned policy rather than the low-conditioned one. A model treating
rating as a static prior separates the halves by nothing.

The proxy's limitation is worth stating plainly: it measures how the model reads
the prior moves, not their objective quality, and the two halves can differ in
position difficulty as well as in prefix strength. It is a signal about
trajectory tracking rather than a strength measurement, and an engine-derived
label would be the way to strengthen it if the question ever justifies the
dependency.

Dependency results are recorded in the correctness family with the training
maturity they were measured at. Nothing in them returns a pass or a fail: weak
dependency on an undertrained checkpoint means the conditioning has not been
learned yet, which is not the same finding as a miswired input.

Two reported quantities are not means over positions and can carry no sampling
floor: the cross-conditioning match rate counts rating slices, and the
within-game response splits each slice at that slice's own median. A report
says `unqualifiable` for those rather than `unknown`, since only the latter is
waiting on work somebody could do.
`docs/decisions/0028-qualifying-the-rating-dependency-family.md` owns the
choice, and its estimator for the remaining quantities is superseded by
`docs/decisions/0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`.
Until this family resamples its own scored games for a dispersion, those
quantities report `unknown` rather than a floor.

## Held-Out Prediction

Held-out human games should be the core offline evaluation source.

The model should be evaluated by rating band, game phase, color, clock setting
when timing is enabled, and other relevant context. Useful prediction metrics
include:

- move cross-entropy;
- top-k human move accuracy;
- timing likelihood when timing is enabled;
- calibration by rating and time context.

These metrics do not prove that generated games are good, but they are the
fastest way to tell whether a training run is learning the intended human move
and timing distributions.

Top-k accuracy is reported over the legal-masked policy, which is the
distribution the runtime samples from, while legality is measured separately on
the raw logits. Mixing the two would let a legality problem hide inside an
accuracy number. One scoring pass computes both, along with the per-position
quantities later benchmarks need, so decision decomposition and rollout
comparisons share this code path instead of recomputing a policy of their own.

### Opening Family And The Rare-Opening Tail

`docs/decisions/0016-sampling-axes-versus-measured-distributions.md` declines to
resample or reweight training by opening family, and accepts whatever
sample-efficiency cost the long tail of rare openings carries on the belief that
the cost is small. This slice is what makes that belief falsifiable. It is a
benchmark and never a loss term; weighting the loss by family is the operation
that record closes.

Per-family loss alone settles nothing. Rare openings are genuinely harder to
predict, so higher loss on the rare ones is the expected result whether or not
they are undertrained, and only the relationship between loss and *training*
frequency separates the two. So the whole reading hangs off one opt-in: counting
how often the training split saw each family, which costs a replay per game. An
ordinary reading does not pay it and does not slice by opening at all.

When it is asked for, every scored position carries its game's classified
family, and the per-family table reports move loss, mask penalty, and top-k
accuracy through the same slice machinery every other dimension uses. The label
is a game-level one — a Sicilian's endgame counts toward the Sicilian — which
dilutes the reading with positions the opening stopped constraining plies ago.
That is why the table is read for its shape across families rather than for any
one family's level.

Families are then grouped into a small set of tiers by their share of the
training selection, plus one tier for families that selection never held and one
for games the book never named. The tiers are what reaches the committed store,
because the per-family table is unbounded, and they are ordinary slices of the
same scoring pass, so the bootstrap qualifies each of them the way it qualifies
a phase or a rating band. Read across the tiers: loss
that is still falling as frequency rises, all the way into the rarest, is the
shape that says more data on rare families would help. Beside them the detail
tier carries the same statement as one number — the fitted slope of loss against
log training frequency over the tail families — and, for each scored family, the
tier and training share to join against its row in the slice table.

A tier is a share of the *training* corpus, and a series fingerprint's data
component covers only the games scored. Those two pin each other only when one
corpus supplies both, so the reading is refused outright on a checkpoint that
trained somewhere other than where the pool came from, rather than committing a
series whose slice membership nothing recorded. For the same reason the tier
boundaries are code-owned rather than configurable: a configured boundary would
move families between series without changing any metric identity.

Reopening decision 0016 needs a second signal beside this one — a generated
opening distribution underrepresenting the tail relative to humans beyond the
noise floor, which the repertoire comparison below supplies. Either alone is
weak, and even both together argue for more data before they argue for
reweighting.

## Legality Metrics

Runtime legal masking guarantees that Anthro Chess does not submit illegal
moves. Evaluation should still measure how much probability the raw model gives
to illegal moves before masking.

The primary legality metric should be mask penalty:

```text
p = softmax(raw_move_logits)
legal_mass = sum(p[legal_moves])
mask_penalty = -log(legal_mass)
```

Perfect mask penalty is `0`. Larger values mean the runtime legal mask is
changing the model's distribution more heavily.

Companion diagnostics:

```text
illegal_mass = 1 - legal_mass
top1_illegal_rate = percent of positions where the raw argmax is illegal
top5_illegal_fraction = average fraction of raw top-5 moves that are illegal
legal_margin = max_legal_logit - max_illegal_logit
```

Because positions can have very different numbers of legal moves, legality
metrics should be averaged per position and reported by legal-move-count slices,
such as `1-10`, `11-25`, and `26+` legal moves.

They should also be reported by phase, and phase is the slice a comparison has
to hold fixed. Measured mask penalty on held-out human positions rises by
roughly sevenfold from opening to endgame, so a pool-wide average describes none
of the three populations and moves whenever game length or phase composition
does.

A normalized diagnostic can compare the model against uniform probability over
the move portion of the action vocabulary:

```text
uniform_legal_mass = num_legal_moves / move_vocab_size
legality_lift = logit(legal_mass) - logit(uniform_legal_mass)
```

Validation should use held-out human positions throughout, including for
rule-sensitive cases. Averaging legality over a whole pool hides them: a model
systematically confused about en passant would barely move an aggregate mask
penalty, because so few positions offer the capture.

The fix is to slice rather than to hand-author. The slice layer derives
rule-sensitive characteristics from exact chess logic, covering check, pins,
castling rights and castling availability, en passant, promotions, only-move
situations, and terminal states, so legality metrics can be reported per case
against real games at their true distribution.

A hand-authored fixed suite was considered and rejected. It would supply one
picked example per rule where a slice supplies thousands of real ones, it adds a
labeling contract that can itself be wrong, and the correctness question it
appeared to answer, whether legal move generation handles these rules, is
already covered by the chess-logic tests. Rule-case slices carry no such
duplication.

Deriving these characteristics is more expensive than phase or color, so it
belongs to the positions a benchmark actually scores rather than to pool-wide
coverage statistics.

Legal-move lists or masks may be computed during evaluation. If that becomes
too slow, preprocessing can store them for validation examples.

Mask penalty can be considered as an auxiliary training loss only if evaluation
shows that the model continues to place too much probability on illegal moves:

```text
loss = move_ce + lambda * mask_penalty
```

This should not be part of the initial core loss. Human move cross-entropy
already pushes probability toward legal moves because the target move is legal.

## Adjudicated Decisions

Some decisions have an answer the deterministic chess layer supplies outright:
mate available to the side to move, mate threatened against it on the reply,
stalemate available to either side, and positions with one legal move. These are
a tiny fraction of any pool, so a model that never converts a forced mate looks
unremarkable in an aggregate move loss. Measuring them separately is the only
way the failure becomes visible.

The reference is **what humans at that rating actually do**, not perfect play.
A model far below the human rate is the finding; matching a human rate that is
itself well below one hundred percent is correct behavior. Treating these as a
correctness gate would push toward special-case handling, which the project
rejects.

Positions where the right answer needs an evaluation function are a second,
weaker class. Material-gain probes belong here: simple material counting will
admit positions where the capture is unsound, so the criterion cannot claim the
certainty the cases above have. That does not disqualify it, because **the human
reference absorbs the criterion's noise**. If an admitted position's "winning"
capture is actually a blunder, humans avoid it too, so the human rate for that
position class is correspondingly low and the model-versus-human comparison
stays valid even though the absolute rate is polluted. The requirement is that
the criterion be deterministic and applied identically to both sides, not that
it be sound.

The consequence is a reporting rule rather than a better criterion: heuristic
predicates are reported **only relative to a reference**, never as an absolute
rate. Predicates should record which class they belong to, since the two carry
different weight in a report.

Deciding whether mate is available means pushing every legal move and testing
for checkmate, which is the most expensive characteristic derived so far. It
belongs to the positions a benchmark actually scores, never to pool-wide
coverage statistics, and a subsampled view is appropriate if it proves slow.

The implemented predicate registry lives in
`anthro_chess.evaluation.slices`. It records whether a predicate is decidable or
heuristic and owns the `only move` derivation that legality slicing also reads.
Immediate threats use the conventional null-move question: if the side to move
passed, could the opponent mate or create stalemate on the reply? The label is
derived only for evaluation; null is never exposed as a model action. The
checkpoint runner scores each realized predicate during its existing policy
pass and writes human rate, legal-greedy model rate, raw policy mass, their
signed gap, the best successful action's rank, rating-band drill-down, and
clustered sample counts through the shared result envelope.

The rank is reported because mass alone cannot separate a near miss from an
absence: a predicate whose best action sits second and one whose best action
sits twentieth can carry the same small probability, and they are different
findings. It is absent rather than zero where no legal action handles the
predicate, which is a real state — a threatened mate nothing prevents offers
nothing to rank.

Material gain is the one implemented heuristic predicate, and it resolves the
exchange on the target square rather than counting the captured piece. Plain
counting would admit every capture of a defended piece, which is not a gain at
all. The resolution uses exact legal move generation rather than a bitboard
approximation, so pins, discovered attacks, and a king that cannot recapture
into check are handled by the chess layer instead of by a second implementation
of the rules. It prices pieces from the same shared table the resignation
guardrail reads, so a pawn is worth the same on both sides of either
comparison; the king's ordering price is applied on top and belongs to choosing
an attacker rather than to any valuation. That is a deterministic, identically
applied criterion rather than a sound one, which is what the reporting rule
above requires. It is also far more common than the decidable predicates, so it
is the one that carries useful statistics onto the perturbed arms of the
novelty sweep.

## Decision Decomposition

A decision can go wrong in two ways that need opposite fixes. The model can
prefer a bad action, or sampling can draw an action the model did not prefer.
Two consecutive decisions in one played game failed in exactly those two ways:
in the first the model ranked the three winning captures first, second, and
third and the draw took its seventh choice; in the next it ranked a free queen
capture sixth and played its own top choice. Lowering the temperature would have
prevented the first and guaranteed the second. Both classes appear in an
aggregate only as a worse result, so no metric that reports outcomes can say
which fix a checkpoint needs.

The decomposition reports both classes over the decisions a run actually made:
how often the selection was the model's own preference, and how much probability
the draws that departed from it gave up. Regret over departures is reported apart
from regret over all decisions, because a small pooled figure means either that
the draw rarely overrode the model or that it overrode it only on near ties, and
those are different findings. Rank and probability are both reported, since rank
two can carry nearly the preferred action's probability or almost none of it.

Every quantity comes from the model's own untempered distribution over enabled
actions, which is what the runtime records at selection time. The temperature is
carried beside them rather than applied to them, so a reading describes the model
and not the dial. Readings are cell-scoped by target rating and temperature: the
balance between the two classes depends on both, and a pooled figure over a grid
would move with grid composition rather than with the model. A pass that varied
either dial therefore has no single committed reading; the caller names the cell
it means.

Nothing in the family declares a direction of improvement. Each metric moves
with temperature by construction — a greedy run follows its policy every time
and gives up nothing — which is a different tradeoff between the two classes
rather than a better model. Choosing a default temperature is out of scope; this
is the measurement such a choice needs.

The decomposition is derivable for games the runtime did not originate. Given a
move sequence and the settings it was played under, each decision is re-scored
through the same session path that would have produced it, so a game played
through a chess GUI is analyzed by the same code as a benchmark rollout. That is
what the UCI adapter's reconstructable debug events are for; `docs/interfaces.md`
owns the event format and `anthro eval decisions` is the reading surface. A
reconstructed session is deliberately not turned into a game record: a log can
end mid-game, and an invented termination would corrupt the one format the
termination benchmarks read.

Per-decision records are retained rather than summarized away, in the
machine-local detail tier. A checkpoint's interesting decisions are individual
ones, and no mean recovers them. A decomposition over one manually played game is
a diagnostic rather than a series, so it is not appended to the results store;
committed measurements come from suites that declare their inputs.

## Novelty

The model degrades on positions unlike those it trained on. The observed form is
specific: the same material win was found when it arose from an ordinary
recapture and missed in a position humans never reach, while raw-logit legality
at the second position was unremarkable. The distinguishing property was neither
difficulty nor legality but distance from training data.

Measuring this by slicing the pool with a familiarity proxy does not work, and
the reason is worth recording so it is not attempted again. The pool is human
games, so its positions are in distribution nearly by construction; the far tail
is thin exactly where the question lives. Every candidate proxy also fails on
its own terms. Exact position frequency has no resolution past the opening,
since almost every later position is unique whether it is ordinary or bizarre.
Distance from the opening book saturates rather than merely trending with
ply: by the middlegame every game is out of book, so there is no variance left
to correct for, and adjusting for the trend recovers nothing.
Model-derived signals are circular, because conditioning a benchmark that
measures model failure on the model's own confidence is blind to the
confidently-wrong case being hunted, and they would move the slice boundary with
every checkpoint. Hand-specified position features reintroduce the labeling
contract the project rejected for rule cases.

**Perturbation replaces detection.** Deriving positions by perturbing pool games
supplies novelty at a known dose by construction, so there is nothing to detect,
validate, or keep stable across checkpoints. The dose is the axis, an
unperturbed control arm drawn from the same games gives it a baseline, and phase
is held fixed throughout, since phase dominates the absolute level and an
unsliced comparison mistakes one for the other.

### What Remains Measurable

Perturbation breaks human-referenced metrics rather than degrading them. Once a
prefix is perturbed the game diverges from what the humans played, so there is
no human move at the resulting position and the human's actual continuation may
not even be legal. Move cross-entropy, top-k accuracy, and distribution
comparison are undefined there, not merely noisier.

**Only measurements whose ground truth comes from the chess layer survive out of
distribution.** Legality qualifies, needing no target at all. So do the
adjudicated decisions above. This is what makes the two sections one benchmark:
perturbation supplies the novelty axis, and chess-derived predicates supply the
ground truth that still exists on it.

Human rates exist only on the control arm. On perturbed arms the reference
becomes the model's own unperturbed rate, so results there are reported as
**retention** rather than as an absolute conversion rate. Without that rule the
perturbed arm quietly becomes the correctness gate this project rejects.

### Expected Shape

Predictions are worth stating in advance, because several of them run opposite
to the naive expectation and would otherwise be misread.

Against an opponent playing randomly, a competent player does better, not worse:
free material, easy tactics, quick mates. So conversion should be near ceiling
at most configured ratings, and it should **rise** with configured rating rather
than staying flat. That makes this a rating-behavior instrument as well as a
robustness one, and it gives a clean failure signature: conversion flat across
configured ratings while matched-opponent play orders correctly says the rating
dial controls style matching but not robustness. Monotonicity may break at the
very bottom of the rating range, where the model cannot reliably convert even a
won position.

The dose-response curve may be non-monotonic at low dose. A small perturbation
takes the model off book without yet giving away material, so it loses learned
guidance and gains nothing, while a large one hands over enough material that
conversion becomes easy. A dip at intermediate dose is therefore expected rather
than anomalous, and it is the region where playing something odd on purpose
would be a real tactic against the bot.

This benchmark has no human reference, and that is structural rather than an
omission: humans do not play random opponents, so no pool curve exists. Its
validity rests on the control arm and on ordering across configured ratings.

Perturbation is one-sided, applied to the opponent's moves. That matches the
situation being measured — someone playing garbage to knock the engine out of
distribution — where the model still chooses its own moves. Perturbing both
sides measures a situation nobody will create. The recipe is versioned because
results are not comparable across recipes, and random legal moves are the
headline recipe: sampling from the model's own low-probability tail would make
the derived positions model-dependent, which defeats the multi-checkpoint trend
the benchmark exists to report.

The offline form scores the policy at derived positions and gives the
dose-response curve exactly. The rollout form plays whole games against a random
opponent and is the direct product test. Both are wanted, and they should not be
conflated: one-sided perturbation leaves the model with a large material edge,
so conversion there partly measures whether it can finish a won position.

### The Implemented Offline Sweep

`anthro_chess.evaluation.novelty` implements the offline form and
`anthro eval novelty` is its reading surface. Each arm derives its own
continuations from the same view of the frozen pool, and each dose writes its
own result, because a dose is a declared workload rather than a sample size: two
doses measure different quantities and cannot share a line. The source games
stay the data component, which is what lets a perturbed arm and the control it
is read against share evaluation inputs while sitting on different series.

The derivation opens a **window** at a configured onset ply and runs for a
configured number of the opponent's moves, each followed by the player reply the
benchmark scores. Holding the window fixed across arms is what makes the control
and the perturbed arms paired position by position rather than merely drawn from
the same games.

**Divergence is absorbing**, and that is what the dose actually controls. Once
one opponent move has been replaced, the human's later opponent moves belong to
a game that no longer exists, so every later opponent move in the window is
drawn too. The configured dose is therefore the per-move rate at which
divergence *starts*, and the realized share of replaced moves is reported beside
every reading rather than assumed equal to it.

The player's side is the human's own continuation, replayed while it stays
legal. That keeps the arm model-independent, and it ends the derived game the
moment the human's move is illegal in the diverged position. The resulting
truncation is a real selection effect — the surviving continuations are the ones
whose human replies happened to stay legal — so the share of the control arm's
positions that survived is reported as its own metric rather than left for a
reader to infer from differing sample sizes.

**Retention is paired on position**, and this is not a refinement. A perturbed
arm scores a subset of the control's positions, so reading its mean against the
control's mean over everything reports the composition difference as a novelty
effect. On a shakedown reading the artifact inverted the answer at every
checkpoint measured: legality appeared to *improve* under perturbation, by three
to ten percent depending on how far the checkpoint had trained. Restricting the
control to the plies the arm actually reached moved the same readings to at or
just below one, which is the honest result. Every retention here reads the
control over the arm's own positions.

The material-gain probe is not a private criterion here. It is a heuristic entry
in the shared predicate registry, so the same pass scores it on every arm and it
carries a human reference at dose zero, which is the reporting rule heuristic
predicates require.

`docs/decisions/0024-one-sided-perturbation-derived-novelty.md` owns the
derivation contract and why the alternatives were rejected.

## Rating Calibration

The selected target rating should correspond to the strength and style level
the bot actually plays.

Offline metrics can compare move prediction quality and move quality by rating
band. Rollout benchmarks should also test whether configured ratings behave in
the expected order.

A useful rollout benchmark is a self-play rating ladder. Run games across a grid
of configured ratings, then fit empirical ratings from the results using a
standard logistic rating model or Bradley-Terry model. This needs only the
runtime, so it becomes available as soon as there is a checkpoint that plays
coherently, well before the late post-training benchmarks.

A self-play ladder establishes ordering but is internally consistent by
construction: every configured rating can be uniformly too weak while the
ordering stays perfect. Anchoring that scale requires an external reference,
which is what engine-anchor matches provide. The two are complementary and
neither substitutes for the other.

Expected score between two ratings can be compared with:

```text
expected_score = 1 / (1 + 10 ^ ((opponent_rating - player_rating) / 400))
```

Useful rating metrics:

- rating order accuracy;
- rating ladder error;
- fitted-rating slope versus configured rating;
- score consistency across the rating spectrum;
- rating preservation when temperature or preference controls change.

Rating calibration work should be measurement before correction. The primary
deliverable is the transfer function from configured rating to fitted empirical
rating, reported with enough shape to be actionable: ordering, slope, and where
the relationship degrades. A slope below one usually points at uneven rating
coverage in the training data rather than at anything to hand-tune, and the
expected response is better data, weighting, or capacity. Applying an explicit
mapping at the configuration boundary should require evidence that training
improvements have stopped moving the measurement.

Calibration should also be reported against a declared reference temperature so
the number means one thing. Rating and temperature are deliberately independent
controls: temperature must never be used to reach a rating target, and rating
must never adjust temperature internally. That is a design constraint, not a
claim that temperature leaves strength untouched.

Whether it does is a measured quantity. Raising temperature is expected to cost
strength, because sampling a weak move loses more than sampling a strong one
gains, and because a lost position cannot be recovered by later play at the same
rating ceiling. A rating-conditioned model may resist part of that drift if it
has learned to compensate within a game. The size of that effect should be
measured rather than assumed, by comparing the temperature response of the
conditioned model against the same measurement with rating conditioning
ablated. Reporting the difference as an attenuation keeps this an observation
about the model instead of a property the project has to promise.

Temperature also changes the shape of mistakes, not only their frequency.
Errors introduced by sampling are spread across the policy rather than
concentrated where a human of that rating would err, so matching average
strength at an unusual temperature is not the same as playing at that rating.
Strength and error-profile metrics should be read together.

### The Implemented Ladder

`anthro_chess.evaluation.ladder` implements this as one round robin and one
fit, and `anthro eval ladder` is its reading surface. The unit that competes is
a **seat**: a conditioning and a temperature. Every seat plays every other, and
a single Bradley-Terry fit places them all on one internal scale.

One joint fit rather than one per temperature is the load-bearing choice. A fit
of this kind is invariant to shifting every rating in it by a constant, so two
independently fitted rows share no origin and their difference is arithmetic
rather than a measurement. Fitting the whole surface at once is therefore what
makes the temperature response — and the ablated comparison it is read
against — a quantity at all. The rating response is read along one axis of that
surface and the temperature response along the other.

The ablated control arm is the same model with **no target rating supplied**,
which is the `absent` treatment the dependency tests already define, applied to
whole games. It joins the same round robin for the same reason the temperatures
do. Read it with that treatment's caveat: a corpus that never contained
rating-absent positions makes this partly a reading about input presence rather
than input value, so a large ablated effect is not on its own evidence about
how much the model uses the rating *value*.

The absolute level enters in exactly one place, the anchor, which pins the mean
fitted rating of the conditioned seats at the reference temperature to the mean
of their configured ratings. Everything else the benchmark reports — ordering,
slope, span, ladder error at the reference row, and both temperature
responses — is invariant to it. Rows away from the reference temperature carry
the offset temperature imposed on them, which is the reading rather than a
distortion of it.

Ordering is reported twice, over all pairs and over adjacent configured ratings
alone, because a ladder can order distant pairs perfectly while every
neighbouring pair is indistinguishable. The adjacent pairs the fit did not order
are retained, which is what localizes where the relationship degrades rather
than only reporting that it does.

Two states are results rather than errors. A fit that does not converge, and a
seat that won or lost every game and therefore has no finite maximum-likelihood
rating, are both reported with the state named and the affected seats listed. A
flat ladder on an early checkpoint is information about the model, not a fault
in the instrument. What the benchmark cannot do is fit a ladder with no scored
game at all: a game that reaches the ply limit has no result and informs no
comparison, so it is counted rather than adjudicated into a draw, and a suite
where nothing finished fails loudly as a generation problem.

**Every number carries what the reading can resolve.** A flat transfer and a
sample too thin to see one in are the same output without that, which is the
distinction the first full-size reading could not draw. Nothing here is a
per-game additive contribution the suite's data-sampling bootstrap could
resample, and two checkpoints generate their own games so there is no shared
sample to pair on either, so the ladder estimates its own floor a third way: it
redraws the games each pairing played, refits, and reads the spread of
everything one fit yields. Ordering, slope, span, ladder error and both
temperature responses are functions of the fitted ratings, so they are reached
by the reduction rather than by propagating a standard error through it — which
would not work for an ordering, since a step function has no derivative to
propagate through.

The floor is evaluation noise for the reason generated play always is, and it
travels on the measurement rather than being characterized against the series,
because a ladder's sample size is deliberately outside its identity and a floor
looked up beside a reading of a different size would be wrong by the difference.
Pairings whose seats are all greedy replay rather than redraw and are held
fixed, so a ladder is qualified against the games that would actually have
differed, and one whose every seat is greedy states a floor of zero rather than
estimating one.

Two situations are treated apart from the rest. A number the redraw could not
move carries no floor and the reading names it: a seat that scored nothing or
scored everything has no finite fitted rating and reports the declared spread
instead, and a step function that saturates cannot be resampled either, so a
floor of zero from a bootstrap would read as perfect resolution rather than as
the exact statement a replayed ladder makes. The error profile beside each seat
is a mean over decisions rather than an output of the fit, so the refit does not
reach it and its noise reports as unknown — a floor somebody could still
produce, rather than one that cannot exist.
`docs/decisions/0034-qualifying-a-rating-ladder-reading.md` owns all of this.

The draw is over a pairing's games without regard to which of the frozen
openings each came from. Two checkpoints are read on the same openings, so a
spread between them would be common-mode and the shared-component rule above
would bar it from the floor. Measured, there is none to bar, and stratifying the
draw by opening and colour would leave every floor narrower than the true
run-to-run spread rather than wider, because a stratum holds few enough games
that drawing inside one understates its own spread by more than the openings
contribute.
`docs/decisions/0039-stratifying-the-ladder-redraw-costs-more-than-it-removes.md`
holds that measurement, what it says for the curve family, and what would
reopen it.

**Read the unfinished count as a reading, not as overhead.** About half a
full-size ladder's games reach the limit, and they are its most expensive games,
but the limit sits past the longest game in the corpus the model trained on — a
model that finished games the way its corpus does would reach it essentially
never. The count is therefore a statement about the seats, and it is the ladder
quantity that has discriminated most sharply between checkpoints so far. It is
concentrated in the sampling seats, which is where the temperature response is
read. It is reported per seat as the share of that seat's games that reached a
result, which is one of the three ladder quantities carrying a direction the
project is willing to name.

Each seat's own error profile is recorded beside its strength, computed through
the shared decision decomposition rather than a private one, since that layer
already groups decisions by the dials they were made under and a ladder's seats
are exactly those groups.

**A full ladder is a scheduled reading rather than a routine one.** The declared
grid plays thousands of games per checkpoint and has run in a couple of hours per
checkpoint at that size, so it is affordable on purpose rather than by habit: it
is taken when a checkpoint is worth that much time, not on every checkpoint, and
it is the one benchmark the reduced sweep leaves out entirely. Seats and their
sample are the two things not to confuse when that cost is under discussion.
Cutting seats cuts cost quadratically and cuts every surviving seat's own sample
linearly, because a round robin gives each seat one pairing per opponent — so a
cheaper ladder is also a noisier one, on the axis the benchmark exists to
measure. Raising seeds, openings, or games per position is the lever that buys
precision instead, at linear cost and without ending a series.

That lever is inert on the pairings whose seats are both greedy, and a ladder
does not pull it there: those pairings play one replicate and record the seeds
they played, so a smaller game count beside one of them is the sample it
realized rather than an omission. Decision 0027 carries what that leaves the
declared grid playing.

`docs/decisions/0022-one-joint-rating-ladder-fit.md` owns the joint-fit rule,
why the ablated arm sits inside it, and what the round robin costs.
`docs/decisions/0027-settled-rating-ladder-grid.md` settles the grid at its
declared size and records what that is worth against what it costs.
`docs/decisions/0030-ladder-ply-limit-at-the-trained-bound.md` settles the ply
limit against the corpus's longest game and owns why the unfinished half is a
reading rather than a dial to move.
`docs/decisions/0034-qualifying-a-rating-ladder-reading.md` owns how a reading
is qualified and what the two degenerate fits are qualified as.

Fixed engine-anchor matches are useful secondary rating diagnostics. Run a grid
of Anthro target ratings against one or more fixed external engine
configurations, such as Stockfish `Skill Level` or `UCI_Elo` settings, and
check whether higher Anthro ratings score better in a stable, mostly monotonic
way.

Engine-anchor ratings should be treated as relative measuring sticks, not as
ground truth for the Anthro rating scale. Stockfish `UCI_Elo`, CCRL ratings,
and other engine-list ratings are calibrated in engine testing pools and
conditions. They are useful for consistency, stress testing, and checkpoint
comparison, but they should not define whether a target rating is calibrated to
Lichess-like human play.

Engine-analysis metrics such as centipawn loss or engine-best agreement may be
useful supporting diagnostics. They should not be treated as the definition of
human rating, especially at lower ratings.

### Puzzle Rating Response

A published puzzle set whose puzzles carry difficulty ratings supports a third
rating diagnostic, and it is the cheapest of the three: solve rate as a
continuous function of puzzle rating across a configured-rating grid, from
forward passes alone, with no matches, no external engine process, and no
sampling noise at temperature zero. The human reference curve needs no
empirical solve-rate bins, because a puzzle rating is itself a difficulty
calibrated from human attempts, so expected human solve rate follows from the
same expected-score formula used above.

What this measures is calibration, not tactical strength. The quantity of
interest is whether solve rate tracks configured rating the way human solve rate
tracks player rating; a model at a low configured rating should fail the puzzles
players at that rating fail. Reading it as a strength target would import a
goal this project does not have.

**Puzzle ratings do not share an origin with game ratings.** They are computed
in a separate pool against each player's own puzzle rating, which is a different
number from their game rating, measuring a different latent ability in a
different population. The relationship between the two is not a constant offset
and is not derivable from the published data, since nothing pairs a player's
game rating with their puzzle rating. So this anchors ordering, slope, and the
region where the relationship degrades, in puzzle-rating units. It does not
establish that a configured rating is calibrated in absolute terms, and it
should never be reported as though it does.

Its distinctive value is that the yardstick does not move. A self-play ladder
measures ordering in internal units that every checkpoint redefines, while a
published puzzle set fixes the scale externally, so checkpoints separated by a
year were measured against the same thing. It is also the one benchmark whose
inputs are immune to pool generation cuts, needing no re-baselining at a seam.

The fixed yardstick is the selected rows, not the source they are cut from.
Upstream publishes puzzles at a single rolling URL with no dated snapshot
beside it and no history, so a pinned source digest stops being fetchable the
moment upstream regenerates — which it did three days after the first pin was
taken. The selection is therefore vendored in the repository and the build reads
it rather than the archive, so the pinned identity stays reachable on a machine
that has never downloaded anything.
`docs/decisions/0044-the-puzzle-selection-is-vendored-not-refetched.md` says why
that boundary moved.

Re-pinning to whatever upstream now serves selects different puzzles and so
changes both digests and the set identity. That is free only while no puzzle
reading has been committed to the results store, because there is then nothing
to be incomparable with; afterwards it is a new set version rather than a
repair, and the readings on either side of it are separate series. So what
decides a re-pin is the store, not the source.

Greedy and sampled solve rates should both be reported against a declared
reference temperature, since the gap between them is the same quantity the
decision decomposition measures. Multi-move puzzles distinguish first-move
accuracy from completing the line, and those are separate metrics.

The puzzle set is an external dependency with its own identity and license
record because a set version change alters what a number means. The selection
recipe, expected identity, and selected rows are all committed; the generated
artifact and the raw archive stay under the data root. Puzzle positions derive
from real games on the same platform the corpus is drawn from, so a
source-game-key join against the training selection reports the overlap rate as
provenance. The measured risk is small, since one exposure among millions does
not produce recall and worst-case inflation is bounded by the overlap fraction.
It is worth reporting anyway because it grows silently as the corpus expands,
and the join is cheap enough that there is no reason to carry the uncertainty.

`anthro eval prepare-puzzles` builds the artifact from the vendored selection
and the pin in `configs/evaluation/lichess-puzzles-v1.toml`, refusing when the
two disagree; `anthro eval puzzles`, selected by
`configs/evaluation/puzzle-rating-response.toml`, reads it.
`scripts/vendor-puzzle-selection.py` is the only path that reads the archive,
and it runs when the set is deliberately re-pinned. The canonical set is
sized from a conservative two-independent-proportions calculation at declared
confidence and power. That is a planning bound on the selection rather than a
floor on any delta. The command prints that bound at the size actually scored,
because a reading beside no
resolution at all was read as a finding about the model once already; it is
labelled as the independent-sample bound it is, and is not the family's floor.

The solve rates are not the whole reading, and what remains needs a different
qualifier again. The fitted puzzle rating at each configured rating, the slope
through them and the pairwise ordering are nonlinear functions of the whole
draw, so no per-unit retention reproduces them. They are qualified inside the
reading instead: the scored puzzles are redrawn within exact-rating strata, by
the rescaled draw a small stratum needs, and *every* configured rating is refit
from that one draw, because the configured-rating grid is one draw asked the
same question several times rather than several independent readings. Each
refitted quantity is printed with its own spread. A quantity no redraw moved
says so rather than reporting a spread of zero, which would license every delta;
that is not a corner case here, since an ordering saturates as soon as the fit
separates two configured ratings and a fit pinned at the bottom of its search
range cannot move at all. Those spreads are what this family contributes to a
delta floor under
`docs/decisions/0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`;
until they are attached to the stored measurements they stay in the output and
the detail payload, and a delta between two puzzle readings reports its noise as
unknown.

Selection is uniform over every exact integer puzzle rating in the declared
range, with deterministic hash ranking only among eligible puzzles at that
rating. This removes the source population's rating-density bias without
creating arbitrary selection discontinuities at a handful of wide band
boundaries.

A reading may score fewer puzzles than the artifact holds, which is how a
reduced sweep affords this benchmark. The dial counts puzzles per exact rating
rather than puzzles outright, and keeps the lowest-ranked of them under the
same hash the build ranks by, so a subsample is precisely the artifact a build
at that setting would have written: uniform over exact ratings, nested inside
every larger reading, and identical on any machine. A flat count would sample
the design away, leaving some exact ratings unscored and others overweighted.
Its floor is two per rating rather than one, because the response redraw
stratifies by exact rating and a stratum holding one puzzle can only redraw
that puzzle, so the redraw would have nothing to take. The realized selection
is recorded in the artifact and its resolution is printed beside the reading,
and because the puzzles scored are the data component, a subsampled run is its
own series rather than a partial full one.

The primary drill-down uses the shared nearest-neighbour curve machinery with a
frozen bandwidth and grid. The analytic human reference and model response are
smoothed at the same local bandwidth, preserving the bias-cancellation rule
used by other human-reference comparisons. The reference is the whole set even
when fewer puzzles are scored, which is the same rule the generated-play and
termination curves follow — at a neighbour-count bandwidth the reference's size
is a smoothing radius rather than a sample size. It costs nothing to hold here,
because this reference is analytic rather than played, so a subsampled reading
is estimated at exactly the radii a full one uses rather than on a differently
smoothed curve. Wide rating bands remain as a readable secondary table, not as
the estimator. The generated manifest records
the power assumptions, source candidate coverage, quality filters, exact source
and selected-content digests, license, and rating-design identity. Tests use
small generated fixtures rather than the canonical records.

Each solution is scored on the canonical verified line. First-move accuracy and
full-line completion stay separate; later player moves are conditioned on the
published preceding solution, so a miss does not invent an opponent reply. The
Lichess mate-in-one exception is preserved by accepting every legal checkmate
rather than only the move written in the export. The
sampled reading is the exact probability of drawing the verified move, or the
product of those probabilities for a line, under the declared temperature.
That is the infinite-sample solve rate without Monte Carlo noise and remains
directly comparable with the greedy reading at temperature zero.

The detail artifact carries the configured-rating grid, continuous human and
model curves with effective local sample sizes, the rating-band drill-down, and
the resampled response resolution. The summary tier carries overall solve
rates, continuous curve distance, fitted-rating slope and pairwise ordering,
plus the source-game overlap rate.
The overlap join reads only Lichess train and validation keys; test-only games
remain excluded because training never consumes that partition.

## Timing Evaluation

Timing evaluation should include both offline likelihood and simulated
rollouts.

Benchmarks should simulate clocks. They should not wait in wall-clock time.

During rollout timing benchmarks:

```text
sample move_time_ms
clock_ms -= move_time_ms
if clock_ms < 0: timeout
else clock_ms += increment_ms
continue immediately
```

Important timing metrics:

- timing validation loss;
- move-time distribution match;
- timeout-rate error;
- remaining-clock-at-end distribution;
- low-clock behavior.

The main timing validation metric should measure whether the model assigns good
likelihood to human move times in held-out games.

Move-time distribution benchmarks should compare sampled model move times
against human move times under similar rating, starting clock, increment,
remaining clock, and game phase.

Clock-survival rollouts should simulate full games and report timeout rate plus
remaining clock at game end. This catches accumulated time-management errors
that do not appear from individual move-time predictions.

Low-clock stress tests should start from realistic positions with little time
remaining and check whether the model speeds up without collapsing into a fixed
move time.

Target timing values should come from human data where possible. The goal is not
to minimize timeouts, but to match human timing behavior for the configured
rating and clock context.

## Move-Time Coherence

Timing metrics should also check whether sampled actions and sampled move times
make sense together. Independent timing metrics can miss cases where the move
head chooses one kind of action while the time head emits a delay that fits a
different kind of action.

The primary guardrail should be the conditional timing likelihood on held-out
human triples:

```text
-log p(human_time | context, human_action)
```

This should be reported by rating band, clock context, phase, and simple move
categories. It tests the supervised target the data actually provides.

Generated-game audits should log action-time pairs without waiting in wall
clock time:

```text
context
sampled_action
action_probability_or_rank
sampled_time
clock_state
```

Useful coherence diagnostics include:

- generated move-time distribution by rating, clock context, and phase;
- very low-probability actions paired with extremely short sampled times;
- obvious simple actions paired with extremely long sampled times;
- repeated instant actions in positions that the model otherwise treats as
  uncertain;
- generated timeout rate and remaining-clock distribution.

"Obvious" and "difficult" should not become hand-authored product concepts.
The model is responsible for learning timing from human data. Evaluation should
use cheap, explicit proxies to catch glaring incoherence rather than invent a
manual move-difficulty model.

Useful first proxies include:

- model action probability or rank;
- timing percentile for the generated action compared with similar held-out
  human examples;
- simple deterministic move categories such as only legal move, recapture,
  capture, check, promotion, castling, and opening-book move;
- rating band, clock pressure, phase, and legal-move-count slices.

Engine-derived labels can be useful later as diagnostic slices, not as a core
coherence score. Examples include positions with one acceptable move, positions
with many acceptable moves, large evaluation swings, tactical positions, or
candidate moves with high engine rank. These labels should not be collapsed
into a weighted "move difficulty" score unless a later implementation proves
that such a score is stable and useful.

The evaluation should avoid assuming that longer time always means a better
engine move. Strong moves can be automatic, weaker moves can be slow, and the
relationship between move difficulty and time depends heavily on rating and
clock context.

## Human-Likeness

Training on human games makes human-like play likely, but it does not guarantee
that generated games will feel human. A model can average patterns into bland,
repetitive, overly safe, or otherwise bot-like behavior.

Human-likeness should be measured with several repeatable signals:

- human-vs-engine classifier score;
- engine agreement and centipawn-loss diagnostics;
- feature distribution matching;
- move entropy and diversity;
- rollout realism.

The preferred late-stage classifier is a frozen evaluator trained to distinguish
human games or segments from conventional engine games or segments. Anthro
Chess outputs can then be scored for engine-likeness.

The classifier should be scoped as an evaluation tool, not a full anti-cheat
system. It should not make claims about real player cheating, require private
platform data, or become a separate research project.

For a first version:

- train on held-out human games and generated engine games or continuations;
- use short game segments or position-plus-move examples;
- match by rating band, phase, and timing context where practical;
- freeze the evaluator dataset and model version before comparing Anthro Chess
  checkpoints;
- report an engine-likeness score as one human-likeness metric, not the only
  truth.

Feature distribution matching should compare generated games against human games
using measurable chess events and patterns:

- opening frequencies;
- move diversity;
- capture, check, and trade rates;
- castling rates;
- game length;
- material imbalance frequency;
- sacrifice-like material changes;
- repetition and draw patterns;
- timing patterns when timing is enabled.

These metrics help catch blandness and repetition even when the model is not
simply engine-like.

## Rollout Distribution Tests

Offline prediction metrics should be paired with generated-game tests.

Two useful rollout forms:

- generate full games from the starting position;
- continue games from fixed human prefixes or benchmark positions.

Generated games should be classified afterward for distribution features,
timing behavior when timing is enabled, game length, result patterns, repeated
lines, and drift away from human-like play.

### Shared Generation Machinery

Every benchmark that needs whole games shares one record format, one harness,
and one analysis layer, in `anthro_chess.evaluation.games`. A shared *corpus*
of generated games is deliberately not the goal and would not work: a self-play
game at one rating yields no pairwise comparison for a ladder, an engine anchor
needs an external opponent, and timing games need clocks the model has no head
for. The saving is in the machinery, not in the games.

The abstraction the harness exposes is **two player configurations plus a
position source**. Self-play puts one configuration in both seats, a rating
ladder puts two ratings in them, an engine anchor puts an external process in
one, and a robustness arm puts a uniform-random opponent there. None of those
is a mode the harness knows about.

Model seats decide through the ordinary game session rather than through a
private sampling path, so a benchmark measures the policy the engine actually
plays and inherits its legal masking, position synchronization, and seeding
rules. The session reports the policy behind each selected action, which is
what lets generated games and games reconstructed from live runtime logs carry
the same per-decision quantities and be read by one analysis pass. Those
probabilities are the model's own distribution over enabled actions rather than
the tempered one the draw used, so they describe the model rather than the dial
and stay meaningful at temperature zero.

Reproduction is from explicit seeds throughout. A suite is identified by one
base seed, and each game's seed and each seat's stream are derived from it, so
one game reproduces on its own from the seed its record carries. Nothing waits
in wall-clock time; an external engine is bounded by depth or nodes, because a
time limit would make results depend on how loaded the machine was.

Games are played in lock-step waves so one player configuration's pending
decisions can be resolved in a single forward pass instead of one position at
a time. Concurrency is a declared setting, and the sequential path is the same
code with a wave of one. Games never observe each other: each keeps its own
board, each seat its own random stream, and a game that ends simply stops
contributing decisions while the rest of its wave continues. A player whose
seats cannot overlap, such as one external engine process serving every game,
holds its suites at one game at a time.

What batching does change is which floating-point kernels run, and a padded
batched pass is not bit-for-bit identical to a single-history pass. Measured on
a trained checkpoint, the *games* were unaffected — the same moves and the same
endings at every concurrency, on both CPU and MPS — while the recorded policy
probabilities moved in their last bits, by under `3e-6`. That is immaterial to
every metric computed from them, but it means concurrency belongs in a run's
recorded configuration, and that a game regenerated on its own reproduces its
moves rather than its stored floats to the last digit. A decision sitting on an
exact tie could in principle fall the other way; none did in the measured
suites.

Endings are recorded precisely enough to tell them apart. Rule endings come
from the chess layer, while the learned terminal actions and the ply limit that
stops an unfinished game are decided elsewhere. A claimed draw keeps the rule
that made it claimable rather than becoming a category of its own; whether a
seat claimed it or the harness settled it is what the record's adjudication
flag says. The harness does not claim draws by default: claiming on the model's
behalf would report the harness's policy as the model's behavior, the seats
have a draw-claim action of their own to use or decline, and games still end on
their own through the fivefold and seventy-five-move rules. The adjudication
path remains as the fallback for seats that cannot claim.

Analysis functions consume retained records rather than re-running games, so a
new distribution feature is recomputed over games already on disk instead of by
regenerating them. Records are bulk diagnostics and stay in the machine-local
detail tier; only the metrics computed from them reach the committed tier.

Opening distributions should group games by a classified family computed from
our own versioned book rather than by literal move prefix. Prefix grouping
fragments broad openings across many buckets while narrow ones keep their mass
in one, and it splits transpositions that reach the same position by different
move orders, so it cannot support a statement like "this checkpoint plays too
few Sicilians." Source ECO and opening headers are deliberately not used: their
granularity is fixed, their assignment is not standardized across databases, and
their name strings differ per source.

The book is vendored into the evaluation package with its own identity and
license record, and classification is derived in the view layer rather than
stored in normalized artifacts, so a book or granularity change never
regenerates the corpus. A game is labeled by the deepest book position it
reaches, so transpositions land together; one pass emits several granularity
levels so a benchmark picks the level it needs; and a game the book does not
name carries an explicit unclassified label rather than a nearest family. Any
artifact carrying opening labels should record the book identity, because
updating the book changes what a label means. See
`docs/decisions/0015-owned-opening-book.md`. Per-ply multi-label opening
metadata for preference conditioning is separate later work described in
`docs/preference-controls.md`.

### Opening Repertoire And Book Depth

Two measurements come out of one classification pass, and they answer different
questions. **Repertoire** is which openings the model chooses, which is a
statement about preference. **Book depth** is how far into catalogued theory it
stays in the opening it chose, which is a statement about knowledge. A model can
plausibly get the first right and the second wrong; the reverse would be
surprising.

Both are human-reference curve comparisons against rating, so they use the shape
described earlier rather than a private implementation each.

Repertoire is measured at family level over full game depth. A fixed ply cutoff
cannot serve here: families become nameable at different plies, so no single
cutoff catches them all at the moment they are determined. Cutting at four
plies would erase the Ruy Lopez, Italian, and Scotch distinction, which is a
repertoire choice real players make and discuss, while `1.e4 c5` is already
decided at two. The naming hierarchy is depth-adaptive by construction, so the
level dial does this job and a ply cutoff does not.

Family granularity is uneven, since it follows naming convention rather than a
uniform level of abstraction. The largest family holds a few hundred lines and
the median holds a handful. Distributional distance is mass-weighted, so tiny
families contribute proportionally little, but a per-family drill-down should
show category mass beside the delta so a share change on a broad family is not
read like one on a narrow line.

Some book names are **waypoints** rather than destinations: positions many
named openings pass through on the way to being chosen. A game keeps such a
label only when it left the book before committing to anything more specific,
which makes the label a statement about depth rather than about preference, and
lets a depth effect leak into a distribution that is supposed to measure
choice. Waypoints are identifiable structurally, without a curated list, by how
many other families remain reachable from the position. Report the repertoire
distribution over destinations, and report the rate of ending on a waypoint as
its own scalar. That rate is a real and strongly rating-sensitive behavior; it
is simply not a repertoire choice.

The rule that follows from that is threshold-free: a position is a waypoint at
the level being reported when more than one label at that level is still
reachable from it, and reachability is positional exactly as matching is. A
position reaches itself, so a destination is a position whose only reachable
label is its own. The same structure supplies the depth reading, since the
deepest reachable entry is the theory still available onward; one index over the
book answers both. Its consequences are worth naming. The rule is level-relative,
so the deepest name of a line is a waypoint at family level and a destination at
line level, or the reverse. And a broad name a transposition can leave — a bare
`Sicilian Defense`, which a handful of unrelated families pass back through —
counts as a waypoint, so games that stopped exactly there sit in the waypoint
rate rather than in the Sicilian's repertoire share. That is the intended
reading: those games left the book at the second ply and chose no Sicilian line,
while every game that stayed in book still lands in the family.

Book depth is reported as three quantities, because the raw depth conflates
choosing a well-analyzed line with knowing it. Record the deepest matched ply,
the deepest theory available onward from that position, and the fraction of it
consumed. A model with a human-like repertoire that abandons theory early and
one that plays offbeat lines both show shallow raw depth, and only the
decomposition tells them apart.

All three are in **book** coordinates rather than game plies. The same position
is the same distance into theory however unusual the move order that reached it,
and a continuation from a human prefix shares no origin with the book at all, so
counting game plies would make depth partly a statement about move order and
partly about where the benchmark started. Games the book never names have no
depth at all and are left out of the depth quantities rather than contributing a
zero, which would otherwise read as an opening played badly rather than an
opening not played.

Depth is a property of the pair, not of one player, since either side leaving
book ends it. Matched-rating games control this on the human side and a
self-play grid controls it on the model side.

Divergence as a function of book depth — the same distance recomputed with the
classification truncated at each ply — says where the model departs from human
play rather than whether it does. It is a diagnostic and belongs in the detail
tier: category count grows with depth, so its noise floor grows too, and it must
never be shown without that floor. If it ever becomes the number people quote,
that is the signal it was a mistake.

That sweep reads raw labels rather than destinations. The waypoint distinction
is about where a game *stopped*, and at a truncation the reading imposed every
line is still on its way somewhere, so excluding waypoints there would drop
nearly everything at the shallow end and would be measuring the truncation.

The shallow end of that curve is exactly computable rather than sampled. A
model's policy at a fixed position is one forward pass, so an opening-tree walk
that keeps lines above a cumulative-probability threshold produces the
repertoire distribution with no sampling noise at all and a reported bound on
the pruned mass. Deep readings must still be sampled, which is a second reason
repertoire and depth belong apart: they differ in computational character as
well as in meaning.

The walk's depth and threshold decide what its numbers mean, so it is scoped as
its own series rather than sharing the sampled reading's. Deepening the walk or
loosening its threshold should end the walk's own history and leave the curves
recorded beside it untouched. Prefixes are never merged across transpositions,
because the policy conditions on the trajectory rather than on the position
alone.

The bound travels with the reading for the same reason a distance travels with
its floor. But the obvious bound is the wrong one to report. Probability
disperses across dozens of legal moves per ply, so on a real policy nearly every
individual line falls below any affordable threshold even while the dominant
ones reach full depth, and a bound that assumes pruned mass could go anywhere
saturates near one — declaring an informative reading unusable. Measured on a
proof-scale checkpoint it read 0.96.

What rescues it is the same structure that separates a waypoint from a
destination. A destination has one reachable label at the reported level, so a
line pruned there keeps the label it already has however it continues. Only mass
pruned while still uncommitted — sitting on a waypoint, or off book — can move
the distribution. Report that as the bound and the assumption-free number beside
it; on the same reading they were 0.38 against 0.96. The tighter bound inherits
the book's canonical-path notion of reachability, so a game transposing out of a
destination into another family by a route no book entry takes would escape it.
That is narrow, and the alternative is two definitions of reachability in one
reading.

Report the depth the walk actually reached, too. It normally equals the declared
depth, since the leading line usually stays above the threshold the whole way,
but a walk that stopped early is reporting something shallower than its
configuration claims and nothing else in the record would say so.

The threshold is a measured choice rather than a felt one. Across an order of
magnitude either side of the declared value, the leading family share moved by
0.02 tightening once and by 0.002 tightening again: the distribution converges
well before the cost does, and the point to stop is where further work stops
changing the answer.

The walk answers a question only the standard-start arm asks. On a prefix arm
the opening was decided by the view before the model moved, so there is nothing
of the model's to enumerate.

A quantity some games do not have can leave a side with no observations at all —
a checkpoint whose every game stops on a waypoint has made no repertoire choice.
That is a reading about the checkpoint rather than a failed run, so the affected
quantity is reported as explicitly unavailable with its reason and the rest of
the comparison stands.

Human prefixes are especially useful early, when the model may not yet be good
at creating coherent full games from the start position.

Rollout suites should vary more than one axis at a time only when that
interaction is the measurement target. The reusable core should cover:

- multiple explicit seeds with exact reproducibility;
- both color assignments;
- the standard starting position and frozen human opening prefixes;
- rating and temperature grids as independent controls;
- enough games per configuration to distinguish one deterministic trajectory
  from a stable behavioral pattern.

`anthro_chess.evaluation.rollout` implements that core and owns the matrix, the
metric set it reports, and the artifact it writes. It produces three units, and
the distinctions matter. A **cell** is one arm at one rating and one temperature,
and carries the raw rollout scalars; the seeds inside it are replicates whose
spread is that cell's evaluation noise, so they are kept apart in the artifact
rather than only pooled. A **reading** spans one arm's whole rating grid at one
temperature and carries the distances against matched human play. An **exact
walk** enumerates the shallow repertoire instead of playing it, and is scoped by
its own depth and threshold so changing either cannot end the sampled series
recorded beside it. Both arms run in one pass over one checkpoint so they share a
grid and a seed derivation.

A cell at temperature zero plays one replicate rather than the configured grid
of them, and records the seeds it played. Greedy seats replay one game per
position, so its remaining seeds and games per position would each be that game
recorded again, and the spread across them it appears to sample is zero by
construction rather than by measurement. The declared colour swap is untouched,
so a cell that swaps colours — as the shipped selection does — still plays that
pair of games and its distinct-game fraction still reads as collapsed; one
configured with neither a colour swap nor a second position plays a single game
and reports what any single game reports.

The reading is the unit because the rating grid is the curve's axis: a single
cell has one rating and therefore no curve at all. Temperature stays fixed
across a reading rather than being a second axis, since mixing two temperatures
into one curve would report a sampling setting as a rating effect.

The curve is evaluated at exactly the conditioning ratings the suite played,
rather than at a separately declared grid. Generated games are produced on
demand and none is committed, so there is no reason to estimate the comparison
where the model was never asked to play: such a point has a human curve and
nothing to compare it against. A declared grid would also be a second list that
has to agree with the configured ratings with nothing forcing it to. The points
are still part of series identity, since the distance is a mean over them, and
they reach the fingerprint through the rating grid the workload already carries.
Want a finer curve, play more ratings. The **bandwidth** stays declared and
frozen, because it is the smoothing rather than the points.

`anthro_chess.evaluation.reference` owns the human side and the quantity
definitions both sides are read through. One definition per quantity is the
point: a game length measured one way on generated games and another way on
human games yields a distance that is partly an artifact of the two
implementations. Human games are reduced straight from the frozen pool rather
than reconstructed into game records, because a human game has no seat
configuration, no seed, and no ending the harness vocabulary can express —
"lost on time" is not a rule outcome — so building a record would mean inventing
all three. The shared trajectory analysis is what makes that unnecessary.

Reference games are placed at the mean of the two players' ratings, and a game
whose players are far apart is excluded rather than averaged into the middle: it
is a mismatch rather than a game at the average of its two ratings, and its
length and result belong to neither player's level.

The floor is the comparison's own bootstrap over the games it generated. The
per-seed distances are recorded beside it as a diagnostic and are deliberately
*not* used as a floor: each seed plays only its share of the suite's games, so a
per-seed reading is a smaller-sample one — noisier, and biased away from the
reference, since a distributional distance estimated from few games per rating
point reads high. Comparing that spread against the pooled reading's floor
compares two sample sizes and makes the bootstrap look roughly the square root
of the seed count too narrow. Checked against forty independent draws at a fixed
size, the bootstrap reproduces the true spread to within a few percent, which is
what a floor has to do.

The temperature-zero row is the exception, and it states its floor rather than
bootstrapping one. Its seats are greedy, so another run of it replays the same
games and the distances do not move at all; resampling games that cannot be
redrawn reports the sample size the suite happened to play instead of anything
the reading has. Every one of that row's floors is therefore exactly zero, and
the artifact records which of the two ways it arrived there, since a bootstrap
over plentiful games also lands near zero and the values alone cannot tell them
apart.

The declared bandwidth is one value for every quantity rather than one each.
Selected over thousands of matched-rating games of the frozen blitz pool, only
game length, opening, and move diversity have an interior cross-validation
optimum; result and cycle improve monotonically to the largest candidate,
because human result and cycle behavior barely varies with rating. Freezing
those at the boundary would make the neighbourhood most of the reference at
every point, which is a global average wearing the shape of a curve, and would
collapse the conditional reading into the pooled one it is meant to be read
against. The shared value is game length's own optimum and costs every other
quantity under a quarter of a percent against its own best error, which is
inside the noise of those optima.

A sparse rating grid leaves evaluation points with no generated game nearby.
Those drop out of the conditional reading rather than being interpolated, and
the count of unsupported points is reported, because a distance averaged over a
third of the grid should not read like one averaged over all of it.

**The unfinished rate gates two of the quantities.** Game length and result are
only about the model when the model's games actually end. Measured on the proof
run, 72% of generated games hit the ply limit while 0.075% of human games in the
same pool exceed it, which makes the result distance mostly a statement that the
model does not finish, and makes the mean length a censored lower bound rather
than an estimate. Neither is a defect in the comparison — the ply limit is in
the declared workload and the unfinished rate is reported beside the distances —
but the two should be read together, and a high unfinished rate means those two
distances are measuring the ply limit rather than the checkpoint. Repetition,
cycle, opening, and move diversity are computed from the play itself and stay
interpretable regardless.

The two arms are not interchangeable for every reading. On the prefix arm the
opening distribution belongs to the view rather than to the model, because the
prefix decided the opening before the model moved; measured on a real
checkpoint, the prefix arm reports the identical opening counts at every rating.
Repertoire is a statement about the model only on the standard-start arm, and
the prefix arm's opening labels exist to slice its other readings rather than to
be read as choices.

Diversity is measured over *trajectories* rather than over record identities. A
record's identity is derived from the whole record, seed included, so two
replicates that played the identical game carry different ids; counting ids
would report a suite collapsed onto one trajectory as fully diverse, which is
the failure the measurement exists to catch. Temperature zero collapses it by
construction, which is why the reading is scoped to the temperature it was
played at rather than treated as something to maximize.

Repetition needs both correctness tests and quality benchmarks. Correctness
tests should use constructed move sequences with known non-repeating,
claimable-threefold, and automatic-draw outcomes so result detection and
aggregation are exact. Generated-game artifacts should then report repetition
and cycle diagnostics, including termination frequency, when recurrence first
appears, and the extent to which later play remains in a repeated cycle.

For model quality, reconstruct the same diagnostics on frozen held-out human
games and compare matched distributions by rating, game phase, time control
when available, and opening-prefix family where practical. Human data is the
reference for whether repetition and draw patterns look human, not a hardcoded
zero-repetition target. Uniform-over-legal and other simple rollout baselines
remain useful sanity checks for learned structure, but they are not substitutes
for the human comparison: a random policy can avoid cycles while still playing
nothing like a person.

Rating-response rollout diagnostics should include action agreement across
configured ratings on the same frozen prefixes, distributional divergence,
result and termination shifts, and color-swapped matchup summaries. These
diagnostics complement held-out dependency tests. They show whether a rating
change survives discrete action selection and compounds into different games;
they do not by themselves prove calibrated playing strength.

## Game Termination

How a game ends is a behavior the model chooses, not only a property of the
final position, and aggregate move prediction cannot surface it. A checkpoint
that never resigns and one that resigns while winning can post the same move
cross-entropy.

The headline reading is a human-reference curve comparison over derived
termination categories, sliced by rating and by time control. It shares the
shape described under human-reference curve comparisons rather than defining a
new one. Categories the model cannot produce, such as abandonment, stay visible
as their own bucket instead of being folded into a neighbor, because hiding them
inside a comparable category creates a permanent gap no checkpoint can close.

Resignation carries a cheap held-out reading and an expensive generated one. On
frozen human games, measure whether the policy assigns resignation mass at the
ply where the player actually resigned, and how much mass it assigns at plies
where the player moved instead. On generated games, measure the distribution of
how far behind the model was when it resigned, against the human distribution
for the same rating band.

Two guardrails matter more than closeness of fit, because their failure modes
are not symmetric. **Premature resignation**, meaning resigning from positions
that are not lost, is the product-critical failure: it is worse than never
resigning and it is invisible to every other benchmark. **Silent non-use** is
the opposite failure, where an enabled terminal action is never selected. Both
should be reported explicitly rather than inferred from a distribution distance.

Judging whether a resignation was premature needs a position-quality signal.
Material balance is the dependency-free proxy and is enough to catch the
egregious cases; an engine-derived signal would be sharper and is subject to the
engine-dependency decision recorded elsewhere in this document. Because the
proxy is heuristic rather than decidable, the model's premature rate is reported
beside the same rate computed over the human reference, following the rule the
adjudicated decisions above already state: a heuristic predicate is read against
a reference, never as an absolute.

Draw claims are rare enough in human data that a distribution comparison carries
little information. The reading that matters is the untimed non-termination
rate: generated untimed games that reach a claimable dead position and never
end. That is the failure the claim action exists to prevent. Correctness gates
should also cover constructed claimable-threefold and automatic-draw sequences,
so claim availability and claim handling are exact rather than sampled.

### The Implemented Family

`anthro_chess.evaluation.termination` implements this and `anthro eval
termination` is its reading surface. It produces three kinds of record, because
it measures three kinds of thing. A **generated reading** spans one
temperature's whole rating grid and carries the deficit and the guardrails. A
**mix** additionally names the human time-control class it was compared
against, since two classes are two questions rather than two samples of one. The
**held-out resignation** reading generated nothing, so it is scoped by the human
content it scored rather than by a generation recipe, which is also what makes
it the one reading here cheap enough to take often.

The two sides are counted over one vocabulary formed as the union of the derived
human categories and the harness's own. The ply limit is the model-only bucket
that mirrors abandonment, kept visible for the same reason: a generated game the
harness stopped has no human counterpart, and folding it into a comparable
category would move mass a checkpoint cannot move.

The generated side is untimed, because the harness plays no clock. A
time-control class therefore slices the *reference*, which is the useful
direction anyway: the question is which human population a checkpoint's endings
resemble. A class no reference game belongs to reports as unavailable rather
than as a distance over nothing.

The mix distance **saturates while the model produces none of the human
categories**, and a reader tracking it early will otherwise mistake that for a
broken instrument. A total-variation distance is the mass that has to move for
the two distributions to agree, so a model producing zero of some set of human
endings cannot score below the human mass on that set, whatever it does with
its own. The first shakedown reading sat exactly there: across five checkpoints
of one run the model ended games only by resigning or by hitting the ply limit,
producing none of the eight other human categories, which carry 0.626 of human
mass — and the distance read 0.626 at every checkpoint, to sixteen significant
figures.

That is the correct answer rather than a defect, and the number is not stuck:
moving one percent of the model's mass onto checkmate moves the distance by
exactly one percent. It unpins the moment a checkpoint checkmates anyone. Until
then the composition change that *is* happening shows up in the category
drill-down and in the rating-variation metric rather than in the headline, which
is the reason the distance is read with its drill-down rather than on its own.

Every reading with no population behind it reports an explicit unavailable with
its reason rather than a zero. This matters most where a zero is a plausible
measurement: a model that never resigned has no median deficit, and writing zero
there would read as resigning while exactly level. The same shape covers a pool
holding no game that carries a terminal action, which is what a corpus prepared
before the terminal actions existed looks like from here once it is compatible
enough to load at all — an incompatible one is refused by the pool loader and
the model runner on vocabulary identity, well before any metric is computed.

## Training Efficiency

What a training configuration costs to run. Unlike every other family here,
this one cannot be measured after the fact: a finished run cannot be re-timed,
so the instrumentation lives inside the training loop and the result is scoped
to a **run** rather than to a checkpoint. That is also why it is not part of
the end-of-run checkpoint suite.

The headline is **active non-padding positions per second**. Padding is kept
out of the numerator because a configuration that pads more is not learning
faster, and the realized padding fraction is reported beside the headline
rather than folded into it, so a throughput change caused by batch composition
is distinguishable from one caused by execution speed. Wall-clock training
time, total processed positions, and peak device memory are reported alongside.

Two exclusions decide whether the number means anything.

**Warmup is excluded**, because the first steps pay for lazy kernel compilation
and allocator growth. Those are real costs and they are reported, but they are
startup costs; leaving them in would make a short run look slower than a long
one for reasons unrelated to the configuration. A run that never leaves warmup
reports no throughput rather than a figure taken from it.

**Overhead is identified rather than averaged away.** Startup, checkpoint
writes, cadence evaluation, final validation, and the run's own instrumentation
are each timed and each subtracted, and the share of the run spent outside
training is reported as its own metric. A run that evaluates itself every ten
steps is not slower at training, and a number saying otherwise would make the
cadence look like a regression.

**Almost nothing breaks the series**, and that is the point. The model
architecture, the dataset, the loader configuration, the effective batch and
accumulation, the determinism setting, and the machine are all recorded as
coordinates rather than as identity, so a report subtracts across them and
names whichever moved. Only the benchmark version identifies the series,
because only a changed definition makes the delta mean nothing.

That follows from the rule in `docs/design-principles.md`: a coordinate you
might want to measure the difference across cannot be part of identity.
Freezing the model into identity would refuse the family's headline question —
what did this change cost us — which is what
`docs/decisions/0021-efficiency-identity-excludes-compared-conditions.md`
records, refining 0018's inference-shaped wording.

A condition change is therefore reported as `confounded` with the moved
coordinate named, not as a refusal. It is the confounder most likely to pass
unnoticed, because a regenerated corpus changes neither the machine nor the
checkpoint label while changing positions per second.

The realized sequence-length distribution is not a coordinate either — that is
an outcome of those choices rather than an input — and is reported as
measurements instead.

The environment pivot works here by pinning the declared conditions rather than
the parameter digest. Training the same configuration on two machines produces
two different sets of weights, so pinning parameters would make "did the new
machine help" unaskable rather than rigorous; the architecture and corpus are
what has to hold still.

`anthro_chess.training.efficiency` owns the exact metrics, defaults, and
workload fields. `docs/training-and-runtime.md` owns the deferred read-back the
measurement depends on, including why the measurement's unit is a logging
interval rather than a step.

### Quality Against Budget

A throughput ranking is not an efficiency verdict. A step that runs twice as
fast while learning less per position is a regression that every efficiency
metric in isolation reports as a win. The question worth asking is what
held-out quality a run reached for a given number of processed positions or a
given amount of wall clock.

That spans two families, so it is a **report joining them** rather than a third
family duplicating both: `training-efficiency` supplies the budget axes and
`held-out-prediction` supplies the quality. It needs no benchmark of its own —
the points already exist, written by the run as it trained and by the cadence
readings taken beside them — and the join is by checkpoint label, which is why
every reading of the same parameters has to agree on that name.

A budget answer reports the best **recorded** point within the budget rather
than an interpolation, because the curve between two cadence readings was not
measured. Two refusals keep the curve honest: every point's quality must sit on
one series, so a view or pool change cannot masquerade as learning, and every
point's efficiency must sit on one declared workload, so a batch-size change
cannot masquerade as a faster machine. An environment change is not a refusal —
it is recorded per point and surfaced, matching decision 0018's posture
everywhere else.

`anthro eval budget` reads it; `anthro_chess.evaluation.results.budget` owns
the join and its comparability rules.

## Inference Efficiency

What a checkpoint costs to play with. An opponent too slow to play against is a
product failure regardless of how it scores on move loss, so this is part of the
checkpoint suite rather than an operational aside.

Three quantities are kept apart, because folding them together lets a win in one
hide a regression in another:

**Batch-one move latency**, reported as percentiles rather than a mean. This is
what a person waiting for a move experiences. It is measured end to end through
the decision runtime, spanning encoding, batch construction, model execution,
legal masking, and sampling, because that is what a move actually costs; timing
the forward pass alone would report a number no player ever sees and would stay
healthy while the encoder regressed. A mean is reported alongside for capacity
arithmetic, but the median says what play usually feels like and the tail says
whether it ever stalls.

A stage attribution accompanies that mean, cut from the measured decisions
rather than taken beside them: the benchmark times the context assembly, then
the prediction call, then everything that follows, inside the single window it
reports. Timing the stages separately is what let the parts sum past the whole,
and what let a from-scratch re-encode — work the engine never does, and two
orders of magnitude above what it does do — render as a leading term.

There is deliberately no encode stage, and that is the substantive finding
rather than an omission. A session encodes one ply as it advances rather than
encoding a history per decision, so the only encode a decision pays for is that
single ply; it is flat in history length and falls inside the remainder
alongside masking and sampling. Prediction — batch construction, the forward
pass, and the host copy — is the overwhelming majority of a decision, and
assembling the context is around one percent of it.

**Declared-batch throughput**, in decisions per second at one declared batch
size. Batching trades latency for throughput, so quoting a serving figure as an
interactive one is the usual way that trade gets hidden.

Two figures are reported here and they are not interchangeable. The headline
resolves **whole batched decisions** through the same loop the generated
benchmarks run — collect every pending context, resolve them in one padded
forward pass, mask and sample each result — so it carries the batch construction
and the masking and sampling a generated decision pays for. The **forward pass
alone** is measured on a batch built once and re-run, which isolates launch cost
for a kernel or precision change; that is the right number for that question and
the wrong one for sizing a run, since it exceeds the whole-decision figure by
over an order of magnitude at larger batches. It also scores the whole padded
sequence rather than the one row per game a decision reads, which makes it a
companion to the batched figure rather than a component of it. Each declares
which it is, in the rendered output and in the metric registry, because the
failure worth splitting them over was the isolated figure being read —
including by the author of the reading — as the cost of playing a move.

Both are taken from the median batch rather than the mean, so one descheduled
batch cannot carry the reported rate. That does not narrow run-to-run spread,
which is a property of the machine between invocations; qualifying a delta
against that is what a characterized execution floor is for, and this benchmark
reports readings rather than floors.

The depth sweep reaches the 300-ply cap the generated benchmarks play to, since
around half a full-size ladder's games reach that cap and they are its most
expensive ones, so a sweep stopping at 80 leaves a reader extrapolating across
exactly the band whose cost it was asked about.
Reaching that depth requires the synthetic history to arrive somewhere a session
can still decide from: random play thins the board down, so a walk that never
ran out of legal moves still lands on a position that is over by rule often
enough to abort a run.

**Cold start**, split into model-load time and the first decision after loading.
Lazy kernel compilation and allocator warmup land in the first decision rather
than inflating the steady-state percentiles, which is why warmup is excluded
there and measured here.

The workload is synthetic and self-contained rather than drawn from the
evaluation pool. Latency depends on history length and legal-move count, not on
which human played the game, so binding this benchmark to the pool would break
its series at every pool generation without changing what it measures. Positions
come from a seeded legal-move walk, so the same declared workload replays the
same positions on every machine.

Headline metrics are taken at one declared reference point: one ply depth for
latency and one batch size for throughput. Sweeps over depth and batch size are
retained as drill-down and are deliberately outside series identity, so
extending a sweep does not end the headline series.

Accelerator work is asynchronous, so every measured window synchronizes queued
device work before stopping its timer. Without that, a benchmark would time the
enqueue and attribute the real work to whichever window happened to block next.

### Comparing Efficiency Readings

Three questions are worth asking of these numbers, and they differ in what is
held fixed. Did a model change cost us speed? Did an environment change buy us
any? And what is the net effect on the thing we actually ship? A report
therefore declares a **pivot** rather than assuming one.

The default pivot varies the checkpoint. When the environment moved as well,
the delta is still shown — it is a real, interpretable number — but the verdict
is reported as `confounded` rather than better or worse, with an attribution
naming which of model, environment, and workload changed. The honesty lives in
the verdict rather than in a withheld delta, because any reader holding both
values can subtract them, and automation reads the verdict.

The environment pivot is the mirror image: the model is pinned by parameter
digest and the machine, precision, or software version varies. That is the
question an optimization asks, and pinning by digest rather than by label is
what stops a model change being sold as a hardware win.

Metric history is one continuous line per workload, annotated where the
environment changed, which is what makes long-run drift answerable at all. A
workload change does break the line, because that genuinely is a different
measurement.

Whether a delta between two of these readings means anything is a separate
question from whether it is comparable, and it is answered by the execution
noise floor described above. Without one characterized on the machine that took
both readings, a report says the noise is unknown rather than calling ordinary
run-to-run jitter an improvement.

`anthro eval inference` records; `anthro eval report --pivot` reads.
`anthro_chess.evaluation.inference` owns the exact metrics, defaults, and
workload fields, and
`docs/decisions/0018-workload-scoped-efficiency-series.md` owns the
comparability contract.

## Preference-Control Evaluation

Preference controls should be evaluated as soft tendencies.

Required checks:

- legality is unchanged;
- target rating remains approximately calibrated;
- the intended preference increases when its slider increases;
- unrelated preferences do not change too much;
- the bot can pivot when the position no longer supports a preference;
- multiple sliders combine without obvious instability;
- timing behavior remains plausible when timing is enabled;
- games still look human rather than mechanically forced.

Opening sliders can be evaluated with opening-family classifiers and structure
frequency. Style sliders may need structural metrics, event metrics, and
human-likeness checks.

## Regression Comparisons

Promising checkpoints should be compared against previous accepted checkpoints
on frozen benchmark sets. Regressions should be visible even when a headline
training loss improves.

Useful regression dimensions:

- move loss;
- timing loss when timing is enabled;
- legality mask penalty;
- rating calibration;
- rollout distribution;
- timing rollout behavior;
- human-likeness;
- preference-control behavior when applicable.

### Reading A Model Change Against A Control Arm

The comparison above watches the project's own lineage. This one attributes a
change: a change that decides what a training run learns is read as a delta
between two **arms**, a control trained without the change and a treatment
trained with it, identical in everything else. The control is what makes the
delta attributable. The most recent recorded reading is not one, because it came
from a run whose corpus, step budget, machine, and code have all moved since,
and a delta against it confounds the change with everything else in between.
`docs/decisions/0029-model-change-control-arm.md` owns why the control is
required rather than recommended, what it costs, and what it still does not buy.

Both arms are read the same way — the default reduced sweep at the same
checkpoint step, on one machine — and the claim is written down before either
arm runs: which metric moves, in which direction. Stating it first is what makes
the reading falsifiable, for the same reason a shakedown states its expectation
first.

The reduced sweep is the default because a reduction is confined to sample
counts, so it estimates the same quantities with wider floors: reading two arms
at less precision cannot let a weak claim through, it raises the bar the delta
has to clear. What it cannot do is read a benchmark that has no reduced form.
A claim naming such a family — strength, whose only reading is the ladder,
whose cost is a grid rather than a sample — is not testable by a reduced sweep
at all, so that benchmark is read at its declared size on both arms and the
comparison says which scale each family was read at.

The scale is therefore part of the claim rather than a response to it. It is
chosen before the arms run, and a delta inside its floor at the chosen scale is
a null result, not a reason to re-read at a larger view: a reading widened
because its answer was unwelcome is the same failure as an arm retrained for a
better number, and reduced and full are separate series in any case rather than
two precisions of one. Where a small effect is expected, `uv run anthro eval
noise plan` reports how many games an axis needs to resolve an effect of a given
size, which is a question for before the reading.

The comparison itself needs nothing new. A training run is a coordinate rather
than a component of series identity, so two arms of one configuration land in
the same series, and `uv run anthro eval report` reads the delta between them
from their checkpoint labels. Arms are recorded into a machine-local store
rather than the committed one: a candidate arm is not project history, and an
arm nobody adopted would otherwise become some later report's baseline.

**What makes a delta admissible is narrower than the machinery suggests**,
because of which floors exist. Almost every floor that qualifies a checkpoint
delta today is combined from what the two readings' own games could have moved,
and such a floor says the delta survives a different draw of evaluation games
rather than that the change produced it. The report says so beside the verdict:
`cleared` means larger than benchmark noise, and never that the change caused
it. The exception claims less rather than more: a replayed reading states a
floor of zero, which says its games cannot be redrawn at all and therefore says
nothing about a draw that could be. Two arms differ by their initialization
seeds as well as by the change, so clearing either kind establishes that two
models differ, not that the change is why. That is not a theoretical gap.
Measured at proof scale, two arms differing only by their initialization seed
cleared 14 of 54 floored metrics and read better on every held-out and legality
metric; decision 0029 holds the reading.

A claim therefore rests on one of two things: a delta far enough outside seed
variance that nothing else explains it, or a training floor characterized from
arms trained at several seeds, which `uv run anthro eval noise characterize`
already produces. Such a floor describes the training configuration its arms
shared, records it, and is resolved only within it, so characterizing one once
qualifies every later comparison against that same base.
Anything narrower is reported as what it is, a delta not distinguished from seed
variance, rather than as an improvement. A family with no floor at all can show
that nothing else moved; it cannot carry the claim.

A null reading is a reading. Arms are not re-run until a number improves, and a
delta inside its floor is a null result rather than a small win.
`docs/issue-workflow.md` owns what that means for the pull request and the
issue.
