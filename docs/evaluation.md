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
committed to the repository and bulk diagnostics stay machine-local.
`docs/decisions/0014-evaluation-result-storage.md` owns that split, what the
committed tier buys, and the reasoning behind not adopting an
experiment-tracking platform.

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

One benchmark declares the device it ran on as workload rather than leaving it
a coordinate, and it is the exception that shows the rule: the inference
benchmark measures the host and the accelerator in one invocation, so those are
two declared conditions rather than one measurement taken on two machines. Which
accelerator remains a coordinate.

`docs/decisions/0018-workload-scoped-efficiency-series.md` owns the rule,
`0020-declared-settings-scope-generated-series.md` extends it to generated
play, including why a rollout's human prefixes are provenance rather than a
data component,
`0021-efficiency-identity-excludes-compared-conditions.md` draws the line
between identity and coordinates, and
`0082-inference-cost-is-counted-where-a-clock-cannot-see-it.md` records the
exception.

### Where The Store Lives

`anthro_chess.evaluation.results` implements this layer and owns the exact
record schema, metric registry, fingerprint algorithm, and size budget.

The summary tier is one small JSON file per result under the store root, beside
the bridges that rejoin a broken series. Each measurement carries
the spread its own reading measured, so nothing separate has to be stored to
qualify a delta. One file per record is what keeps concurrent
appends and Git merges additive; a concurrent write into the same store fails
on an exclusive lock rather than producing a partial record.

**A benchmark writes machine-local, and a record reaches the committed store by
being promoted into it.** The store root resolves from
`ANTHRO_CHESS_RESULTS_ROOT`, or beneath `ANTHRO_CHESS_RUN_ROOT` the way the
detail tier does, or in the working directory the way every other unset root
does — so a reading lands where candidate work belongs, and nothing resolves
into `results/`, which is reached by naming it. `anthro eval promote` is what
names it, copying one checkpoint's records — every benchmark's and every cost
record's — into the committed store, where committing them in a pull request is
the promotion. It copies rather than moves, so a machine keeps every reading it
has taken and a comparison finds both of its arms in one store.
`docs/decisions/0063-the-full-sweep-decides-a-change-and-the-canonical-line-is-its-byproduct.md`
owns what the committed line is; `docs/issue-workflow.md` owns when a session
does the copying.

A bridge reaches the committed store the same way, by being recorded there:
`anthro eval bridge add --store results` asserts one about the committed
history, and one recorded into a machine-local store applies to that store's
reports alone.

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
`anthro eval noise plan` answers how many games an axis needs to resolve an
effect of a given size, read off the newest reading that measured its own
spread. `anthro eval inference` counts what a
decision costs the model and times what the checkpoint costs to play with; see
inference efficiency below. `anthro eval decisions` separates
model error from sampling error over a payload of generated games or a played
session's log; see decision decomposition below. `anthro eval puzzles` measures
the external puzzle-rating response described in the rating section, and
`anthro eval ladder` measures the self-play rating ladder and its temperature
response described beside it. `anthro eval termination` measures what the policy
says about resigning at positions humans reached; see game termination below,
where how a checkpoint's own games end is read by the rollout instead.
`anthro eval budget` reports held-out quality against the training budget that
bought it, joining two families rather than defining a third; see training
efficiency below. Training efficiency itself has no command, because it is
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
results store it projects. Decision 0023 owns this constrained projection and
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
dependency on a benchmark the sweep does not include, or a step configured to
discard output another step reads all fail in the first second. This is the same
rule training cadences follow: a run that will fail should fail before it spends
time, not after.

Resolution reads the selections, not the artifacts they name: a pool that is not
on this host fails the step that reads it rather than the plan.
`docs/decisions/0055-a-missing-artifact-fails-its-step-not-the-plan.md` owns why
that is preferred to refusing the sweep.

**Ordering is enforced rather than documented.** Decision decomposition reads
the games the rollout generated, so the suite orders it after the rollout and
refuses a sweep where it could find nothing to read. The games are handed over
as a payload in the sweep directory rather than in memory, which is what lets a
resumed sweep retry the decomposition without replaying the rollout.

**Recording is decided per benchmark within one sweep.** A sweep that records
one baseline reading and leaves the rest as evidence about the instrument is
the normal case, not an exception, so the decision belongs to each step rather
than to the sweep.

**A sweep that dies late keeps what it already read.** Each step's outcome is
written to a machine-local ledger as it finishes, and a failed step does not
end the sweep: the independent benchmarks after it still run, and only the
steps that read its output are skipped. A resumed sweep refuses a ledger
belonging to a different plan, so it can never continue another sweep's or
another checkpoint's.

**The sweep has one size, and each step's own selection declares it.**
`docs/decisions/0079-one-declared-size-per-benchmark.md` withdraws the reduced
scale the suite used to carry. The scored units are the data component, so a
smaller reading carries its own fingerprint and answers only against other
smaller readings, a history nothing consumes. Where the dial is a view size the
smaller reading is a strict prefix of the larger as well, since one hash-rank
ordering is sliced at both. A cheaper reading is `--set` on the invocation that
wants it, against the benchmark's own schema directly or through
`benchmarks.<step>.overrides` on a sweep.

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
because scope decisions are made on these numbers, such as what a step's
declared size costs and whether the sweep as a whole stays affordable, and
before it
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
counts out of a latency series. Three things are taken out of it: the model
selection and its label, since the checkpoint is the coordinate a cost line
varies along; the machine prefix of every artifact path, since the artifact is
the same work wherever it is rooted; and the pool generation a selection pins,
since that names which data was realized rather than how much work there is,
and a generation cut would otherwise fracture every cost line.

**A cost reading carries no spread**, so a cost delta is reported as unknown
noise and nothing in a record says the machine was busy. It is the one recorded
family with no dispersion producer, and no open issue owns closing that;
decision 0031 carries the measurements and what a cost reading is worth without
one.

`anthro_chess.evaluation.cost` owns the record and the workload normalization;
`docs/decisions/0031-committed-benchmark-cost.md` owns the reasoning.

### The Checkpoint Evaluation Runner

`anthro eval run` scores one compatible checkpoint over a deterministic view of
the frozen pool and appends the held-out prediction, legality and adjudicated
decision readings. It is the canonical end-of-run reading,
and it is a library before it is a command, so in-training evaluation at
declared cadences calls the same entry point over a smaller view instead of
growing a second implementation that has to be kept consistent with this one.

A **leakage check runs before any scoring**, so a checkpoint that trained on
these games fails loudly rather than producing a plausible number nobody
re-examines. It reads no games at all. A corpus gives each game exactly one
split, so a pool cut from one split and a run that read another are disjoint by
construction, and where both sides declare the same split recipe the pool's own
game ids are put back through it, which checks the pool's claim about which
split it holds rather than trusting it. A game keeps its id as a corpus grows,
so that argument still settles a checkpoint trained on one generation against a
pool cut from the next.

Where neither holds, because the corpora are unrelated or their recipes differ,
nothing establishes disjointness. The reading is still taken and the result
records that it is **unverified**, with the reason on it and a warning in the
log, rather than being refused or carrying an assurance nobody earned.

Which sliced series are **committed** is a deliberate, bounded choice, because
only a committed series can be compared over the life of the project. Overall
prediction and legality headlines are committed; so are move loss and mask
penalty per phase, move loss per default rating band, and mask penalty per rule
case. Phase is committed on the evidence that held-out mask penalty varies
severalfold between opening, middlegame, and endgame positions: a pool-wide
average sits between those populations, and a comparison that does not hold
phase fixed reads a shift in game-length or phase composition as a legality
change. Everything else — color, speed, legal-move-count buckets,
cross-conditioning tables, per-position records — stays in the machine-local
detail tier. Speed stays there while one corpus is one class: the slice is what
makes a mixed pool readable, and committing a series over it is a question for
the generation that spans speeds.

Per-position records are the one part of that tier a run has to ask for. Two
kinds qualify: every scored decision, and every decision a predicate
adjudicated. Each is one record per position over a pool that holds millions,
and every reported quantity is computed from the summaries beside them rather
than from the records. So they are written when the reading's detail
configuration turns them on, for a session that means to look at the decisions
themselves, and left out otherwise.

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
it**: shrinking the reference smooths the curve more heavily rather than
sampling it more coarsely. The reference is therefore declared at a size the
grid can resolve rather than left uncapped, and joins the declared workload so
two readings smoothed differently cannot share a series.
`docs/decisions/0037-the-human-reference-is-bandwidth-not-sample-size.md` owns
that rule and the measurements behind it.

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

Both resample the **stream** a game came out of rather than the game, because a
stream is what re-running the reading redraws and one of them plays a game at
every rating of the grid. A comparison whose model side varies and holds fewer
than three of them estimates neither number: three resamples are not a spread,
and the level's model half is read off the same three. A replayed side is the
exception and states its zero floor and its levels at any stream count, since
neither asks how far its games would move.

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

A delta is not a finding until it is larger than the noise in the measurement.
Reports should annotate every change with the noise floor it did or did not
clear, and a delta inside the floor should be visible but marked rather than
hidden, so a consistent small regression is not lost.

**Every reading measures the spread of its own units and stores it.** That
number is the **dispersion**, and it is the only thing a benchmark reports about
its own noise. Comparing two readings combines them, because the variance of a
difference is the sum of the two variances. Nothing is characterized ahead of a
comparison, stored between runs, or looked up.
`docs/decisions/0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`
owns the design, and replaced a taxonomy of four noise sources with scope rules
and a stored characterization per series.

The dispersions are not interchangeable even though the reported quantity is,
and what separates them is whether a benchmark can produce its own. Held-out
prediction scores four hundred games, so it holds four hundred numbers to
resample and one run yields the value and its spread together. A latency
percentile is a single number with nothing inside it to resample, and the spread
that matters is the one *between processes*, so the inference benchmark measures
its own by running itself again. **Execution Noise** below.

A floor is that dispersion expressed as a delta, because a delta is what a
report shows and a standard deviation is not directly comparable to one.
Coverage is applied at comparison time rather than stored on a reading, since a
floor is a claim the comparison makes; the readings only say how far their own
units move. Where the two readings happen to agree, the arithmetic reduces to
the familiar `sqrt(2)`; they routinely do not, and the two readings committed to
this repository differ by up to two orders of magnitude on the same metric.
`anthro_chess.evaluation.results` owns the arithmetic.

Which reading qualifies which claim is easy to conflate. Whether one run
improved between two of its own steps is a checkpoint delta. Whether a change to
the model, the data, or the training setup improved anything is a configuration
change, and clearing a floor does not establish it: the two arms differ by their
initialization seeds as well as by the change, and no floor built from a
reading's own units can see that. One configuration is the exception, and it is
the exception by construction: the ablation vehicle carries a seed dispersion
stored against its identity digest, so a comparison whose **control** carries
that digest can be qualified on seed as well. The treatment carries a different
one, necessarily: a candidate that left the digest where it was would not be a
candidate. The floor is found by the base it was measured on and describes that
base, which is what **Regression Comparisons** below means by its describing
baseline arms rather than the arm.
`docs/decisions/0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
owns why that is affordable there and nowhere else. **Regression Comparisons**
below holds what a claim rests on, and decision 0029 holds the measurement that
settled it.

A floor that qualifies a delta must exclude anything the two sides of that delta
share. Two checkpoints are compared against the *same fixed* human reference, so
the reference's own sampling error is common-mode and cancels; including it can
only inflate the floor and hide real movement. Only the side being compared is
resampled. How much this matters depends on how thin the reference is relative
to the bandwidth: on a reference of a few hundred games it widened floors
noticeably, while at the declared bandwidth over the frozen blitz pool the
difference was under a percent. It is excluded because it is not part of the
question, rather than because it is always large.

Re-measuring does not always redraw. Greedy seats replay their games, so a
temperature-zero reading is deterministic: another seed reproduces it exactly,
and its dispersion is zero rather than small. Such a reading states a spread of
zero rather than estimating one, and records that it was stated.
`docs/decisions/0032-a-replayed-reading-has-no-evaluation-noise.md` owns the
rule, what a bootstrap over those games reports instead, and why no spread at
all would understate what is known.

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
number of values in hand. Bootstrap resamples are drawn from one sample, so the
**games** are the replicates and the resample count is not; readings repeated
inside one process share a warm allocator and a compiled kernel, so the
**processes** are the replicates and re-reading inside one buys nothing.
`docs/decisions/0026-conservative-dispersion-bounds.md` owns this rule and what
counting either of the cheap numbers would buy.

Which games are independent of each other is its own question, and a generated
rating grid is where it bites. A game's seed is derived without the rating it was
conditioned on, so one **stream** plays a game at every point of the grid and a
grid multiplies games without multiplying the draws behind them. The curve
comparisons resample the stream for that reason, and count streams rather than
games as the replicates behind their bound;
`docs/decisions/0060-a-curve-resamples-the-stream-not-the-game.md` owns the rule
and the measurement that fixed it, including what drawing games instead reported
instead. Raising `games_per_position` or the seed count buys streams; widening
the rating grid does not.

Which games those are is a per-metric question. A sliced metric — a rule case,
an opening tier, a rating band — is realized in a fraction of the games a pass
scored, and only those carry evidence about how far it moves. Each metric's
spread is bounded for the games that realized it and records that count, so a
rare slice reads as the thin estimate it is rather than borrowing the whole
pass's confidence. A metric only one game realized reports no spread at all,
since one replicate observes none.

The ladder's refit answers the same question about its own quantities. A seat's
score rate counts the games that seat played; its fitted rating counts every
redrawn game in the grid, because the fit that produces a rating is joint and
every game anywhere in it moves every rating.

The bound is severe at small replicate counts, which is what the replicate
defaults are chosen against. Resting a floor on two bounds rather than one does
not weaken the confidence either carries: the floor needs only their combination
to hold, and a bound that overshoots covers for one that undershoots.

The estimators differ even though the reported quantity does not. A sampling
spread is bootstrapped by resampling the **games** a run scored, since positions
within one game are far from independent and resampling them would report a
floor several times too narrow. An execution spread is measured by running the
benchmark again in fresh processes, because nothing inside a timing reading can
be resampled into one.

### Execution Noise

Timing is where a floor matters most and where the usual estimators do not
apply. Two readings of one checkpoint taken minutes apart differ, and nothing
about the model, the data, or the seed moved; a report with no floor can only
say the number changed, so sub-percent jitter renders as a regression.

The noise source here is the machine: scheduler contention, thermal state, other
processes, allocator and kernel warmth. It cannot be bootstrapped out of an
already-measured latency, and it is measured by **measuring again**, in a fresh
process each time. The process is the unit because it is where nearly all of the
noise lives: repeating a measurement inside one process reproduces it several
times more closely than a fresh process does, so measuring more decisions buys
no resolution and only the process count does. One reading per process.

**The inference benchmark does this during its own run**, and those processes pay
for themselves twice. The committed value is their mean, and the dispersion the
reading carries is that mean's own spread rather than one process's, since a
floor built from the per-process spread would price the reading as though the
extra processes had not been paid for. The reading being
qualified is itself one of the processes. The rest run one after another rather
than together, since they would otherwise contend for the device they are
timing, and each is a complete run of the benchmark; they record nothing,
because they are evidence about the machine rather than about the model.
`anthro eval noise sample` is the single-process entry point the benchmark
spawns, and `anthro_chess.evaluation.execution_noise` owns the procedure.

Measuring inside the run is also what keeps the answer honest over time. A
stored floor is scoped to the machine that produced it but says nothing about
*when*: a characterization taken on a quiet machine licensed four times as many
false findings once the machine was hot, which is further than the dispersion
bound moves anything. A dispersion measured beside the reading it qualifies
cannot be applied across that drift, because it is never applied to a second
reading at all.

The process count is the one setting that trades measurement time for resolving
power, and it moves both halves: cutting it widens the floor and leaves the
committed value nearer to a single process's. The two do not move at the same
rate. The mean's spread falls as the square root of the count, but the bound on
an estimated spread is a chi-square limit at one fewer degree of freedom than
there are processes, and that limit is punishing when there are few. Below four
processes it costs more than the pooling wins, and the reading resolves less than
one taken without pooling at all. The count is paid on every inference reading
rather than once per machine, so it is a live cost; the code owns the default and
the configuration overrides it.

### What A Missing Floor Means

**No floor at all is two situations, not one.** A floor may be missing because a
reading did not measure its spread, which is work somebody could do, or because
no floor can exist for the metric, which is not. The second covers two unlike
cases that share a consequence: a metric whose units resampling would estimate a
different quantity from, and a metric counted rather than sampled, which has no
spread at all and whose delta is therefore exact rather than unqualified. Both
declare why in the registry, and a report renders them `unqualifiable` rather
than `unknown`. Reporting both as unknown sets a reader to
work that cannot be done, and it is the same ambiguity as a floor rendering as
exactly zero.

That second ambiguity is why a bootstrap that could not move a quantity reports
no spread rather than a spread of zero. What the resample observed is that *this*
draw could not move the number, which is not the observation that nothing could:
a quantity identical in every unit scored reads that way at any sample size, and
the wider draw that would move it is exactly the work `unknown` points at. A zero
would instead clear every later delta, which is the failure a floor exists to
prevent. The genuine zero is the *stated* one, where re-measuring replays the
same games, and a reading records that it stated rather than estimated. The
arithmetic that bounds a dispersion refuses a zero outright, so an estimator
that reaches it has already decided which of the two it holds. Generated-play
curves are the one family that keeps a zero its own resample produced, for the
reason decision 0042 gives — and they read it off whether the draw moved the
curve rather than off what the reduction summed to, because a draw that moved
nothing leaves the last bits of the arithmetic behind rather than a zero, and
those bound into a floor that clears every delta.

**A floor needs both operands.** A dispersion one reading measured and the other
did not describes that operand rather than the difference, and nothing licenses
assuming the side carrying none is the quieter, so the delta is reported as
unknown rather than floored by half of itself.

Sampling spreads are also what size the evaluation inputs. A conservative
independent-input estimate is what a benchmark is sized from before any
checkpoint has been read, and a reading's own measured spread replaces it as
soon as one exists. Either shrinks with the square root of the units behind it,
so how many games an axis needs in order to resolve an effect of a given size is
computable rather than guessed, and `anthro eval noise plan` computes it from the
newest reading that measured its own spread over a counted sample. No
benchmark-level resolution constant is declared or kept current for it.

The answer is in the same units the spread was read over, so for a sliced metric
it counts games that realize the slice rather than games in the pool. The
command reports both, converting through the rate the reading itself observed —
an identity where every game realizes the metric, and an order of magnitude for
a rare rule case. A reading recorded before that count became per-metric is
refused rather than converted, because its count answers the other question and
nothing on the record distinguishes the two.

## Benchmark Data Layers

Benchmark inputs are layered as partition, pool, and views. Keeping them
separate is what lets many benchmarks with different needs share one set of
evaluation inputs instead of accumulating a tailored dataset each.

The **partition** decides what a game may be used for. `test` is held back from
training entirely; `docs/data.md` owns the split contract.

The **pool** is a bounded uniform sample of the `test` partition, materialized
as one versioned, checksummed artifact with its own manifest and coverage
statistics. It carries no
per-benchmark tailoring, and it is a regenerable pipeline output rather than
committed data. Its manifest records source, split recipe, schema,
preprocessing, action, encoding, and benchmark versions, the selected game ids
and their content hashes, a build-time overlap check against the train split,
and the generation it was verified to contain. Coverage
statistics report ply counts, results, games by speed class and by clock
presence, the span of source dates, and position counts by phase, color,
legal-move-count bucket, and rating band, so a thin slice is visible before a
benchmark reports a number computed from it.

The bound is an admission fraction, applied by ranking a game id under a fixed
seed, so a game is admitted on its id alone and corpus growth only ever adds.
A game count could not: a later generation would rank a larger split, and the
games it gains would push some of the previous generation's past the count.
The fraction is sized from the games each metric needs to resolve an effect,
which `uv run anthro eval noise plan` reports from measured dispersion. It can
be raised by a later cut and can never be lowered, so it is chosen as the
smallest size that resolves what the project intends to read. Without it the
pool is the whole split.

The cut is also where the marked-account rejection is applied when preparation
did not apply it. A pool selection may name the snapshot `docs/data.md`
describes, and every admitted game either player's digest appears in is left out
of the generation. The manifest records the recall that snapshot claimed and how
many games it took: containment makes the cut permanent, so what a generation
claims has to be readable off the generation itself, and a rejection nobody can
audit is indistinguishable from one that never ran. A snapshot that never
counted an archive the corpus holds is refused rather than applied to it. Which
recall a pool carries is therefore settled when it is cut, by whoever cuts it.

A pool also carries the **position labels** its readings share: what exact
chess logic says about each of its positions, which is a pure function of a
generation that is frozen and refused when superseded. The first reading of a
pool with no matching artifact derives one and saves it beside the pool, so
nothing has to remember to build it, and the artifact carries a key over the
pool identity and every scheme the labels were derived under, so a stale one is
a miss rather than a wrong answer.
`docs/decisions/0069-a-frozen-pool-carries-its-position-labels.md` records why.

**Views** are per-benchmark deterministic selections over the pool: filtering by
ply count, rating presence, or the day the source dated a game; projecting to
prefixes; subsampling by hash rank.
Each benchmark records its resolved view spec,
including the digest of the selected game ids, in its own artifact. Views are
derivations, never new stored data. A benchmark needing something the view layer
cannot derive is a signal that the field belongs in the normalized schema.

Benchmarks that must run quickly subsample in their own view rather than forcing
a smaller pool, so what one benchmark costs is its own to choose. The pool's
bound answers what a view cannot: what every benchmark process materializes
before any view is applied, and what the `canonical` view scores, since that one
declares no bound and is the whole pool.

A recorded view name describes the selection that ran rather than the one its
config asked for: a view a cap actually shortened records that cap, so a sweep
override cannot leave a stored name claiming more than the reading behind it.

**A view size is chosen once rather than tuned.** The scored games are a data
component, so raising a cap starts a new series instead of continuing the old
one more precisely, and decision 0013 forbids bridging that seam as explicitly
as the fingerprint breaks it. A series meant to last therefore takes the
unbounded view, which has no cap to regret; the only seam it then meets is a
generation cut. This is what separates a pool-reading count, chosen per generation,
from a generating count such as seeds or games per position, which stays
provenance under
`docs/decisions/0020-declared-settings-scope-generated-series.md` and can be
raised whenever cost or precision argues for it.

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
through a filter change, changing the split seed, and lowering the pool's
admission fraction or changing the seed it ranks under all destroy that and end
the affected series permanently.

A benchmark selection names the generation of the pool it loads, and a pool that
is not that one is refused rather than scored. Everything else the loader checks
asks only whether the pool is intact and readable by this code, which a
superseded pool left where it was materialized is: it would keep scoring after
the selection moved on, labelled as itself and comparable to nothing.

A benchmark reports one number, over the pool it is pinned to. Cutting a new
generation therefore breaks the series it affects, and the break is taken rather
than bridged: fingerprints detect it, a report renders the seam as a seam, and
readings after the cut re-baseline. What that gives up is comparing a reading
from before a re-cut against one from after.

Comparing checkpoints on the pool applies selection pressure to it over time.
That is accepted rather than designed away, and a re-cut is what relieves it.
`docs/decisions/0068-a-pool-re-cut-breaks-benchmark-history-and-that-is-accepted.md`
records what that costs and why one number per reading was preferred to two.

See `docs/decisions/0011-held-out-test-partition.md`,
`docs/decisions/0012-derived-evaluation-views.md`,
`docs/decisions/0013-benchmark-result-comparability.md`, and
`docs/decisions/0052-a-bounded-pool-is-a-fixed-admission-fraction.md`.

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
- sequence construction and stacked-history behavior;
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

That rests on split assignment being the only thing separating the two, so a
marked-account snapshot has to reach both or neither. A preview applies the one
its validation selection names — the single filter it applies at all, since this
rejection defines which games count as human play rather than narrowing what a
run trains on. Naming a snapshot on the pool selection and not on the validation
selection is therefore a configuration mistake rather than a limitation: the two
would estimate populations differing by roughly a tenth of their games, most
visibly on the human-likeness family, since what the pool drops is
engine-assisted play.

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

The implemented statistics sit on both sides of that rule, which is worth
knowing when reading them. Gradient norm reads gradients the backward pass just
wrote, so it runs every step and reports both the value at the reported step and
the interval's maximum, which is what catches a spike between two logging
points. The clip rate is that same norm compared against the ceiling and
counted, so it needs no second reduction, and it is reported as a share of the
interval's steps because the question it answers is whether clipping is
insurance against a spike or a learning-rate cap acting on every step. The
update-to-weight ratio needs the parameters from before the update, so measuring
it every step means a permanent parameter-sized shadow copy; it is measured on
reported steps instead, and its cost is paid only there. Measured
instrumentation time is reported with the run rather than assumed to be free.

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
behavior in the intended direction.

It is not the only reading that would catch a rating input the model ignores,
and it is not written as though it were. What it adds is that it changes the
input and reads the same move at the same position with everything else held,
cheaply enough to run on any checkpoint, and so separates a model that cannot
read the input from one that reads it and whose behavior does not follow.

The basic form evaluates frozen held-out examples under the true conditioning
value and again under corrupted conditioning: a shuffled value, or explicit
absence. A conditioning input that the model uses should show clearly worse
held-out prediction when corrupted. Pinning every position at one fixed rating
is a third corruption, and the cross-conditioning grid below already scores it
at every grid rating, so it is read off that table rather than paid for again.

The absent treatment answers a different question from the others. The corpus
this project trains on carries a rating on every game, so the model's
rating-absent embedding is never trained and the treatment substitutes a vector
the model has not seen. That measures how it handles an unseen input rather
than how much it relies on the value, and it is not comparable to the shuffled
degradation beside it. Read absence as an out-of-distribution probe until
training masks the rating on some fraction of examples; it would become a
dependency treatment in the ordinary sense if it did.

Direction matters as well as magnitude. Evaluating each context slice under
each conditioning value produces a cross-conditioning comparison whose best
result should fall on the matching pair. This distinguishes a model that merely
reacts to the input from one that has learned its intended meaning.

That comparison is reported twice, because the obvious summary of it saturates.
Counting the slices whose best value is the matching one is a fraction of four,
and a checkpoint that has learned the ordering at all scores one on it, so the
rate reports a regression and never progress. The graded form is what a
position pays for being scored outside its own band, averaged: the same
comparison, left as a distance instead of a count, and unlike the rate it is a
mean over positions and so carries a sampling floor.

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

`anthro eval dependency` is the reading surface, and it declares its own view
rather than sharing the checkpoint reading's. Every batch is scored once under
the true conditioning and again under each corrupted and each fixed one, so it
pays a forward pass per distinct conditioning where a held-out reading pays one
in total. Sharing one view would have made the family that tolerates a different
sample set the cost of the family that does not.

Its size is set by the narrowest quantity rather than by the cheapest reading
that would detect a dead input. Detection needs almost nothing: a degradation
collapsing to zero is visible on a handful of games. What the view is sized for
is the finer question of whether conditioning is still being learned between
two checkpoints, and there the shuffled degradation is the binding one. The
configuration file carries the measurement.

Two reported quantities are not means over positions and can carry no sampling
floor: the cross-conditioning match rate counts rating slices, and the
within-game response splits each slice at that slice's own median. A report
says `unqualifiable` for those rather than `unknown`, since only the latter is
waiting on work somebody could do. Both are rendered beside the per-band table
they summarize, which is where the evidence for either actually is: four bands
moving together is a different reading from four cancelling out, and neither
scalar can tell them apart.
`docs/decisions/0028-qualifying-the-rating-dependency-family.md` owns the
choice, and its estimator for the remaining quantities is superseded by
`docs/decisions/0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`.
The rest resample the games the reading scored, on the pass that produced it.
Each is weighted by the positions that carry it, which for the two anchor
quantities are the ones a trajectory signal was computed for rather than every
rated position.

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
stalemate available to the side to move, and positions with one legal move. These are
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
for checkmate, which is the most expensive characteristic derived so far. Over a
frozen pool it is derived once per generation into the artifact beside it and
read back; anything scoring positions no pool holds, such as a perturbed
continuation, resolves it live.

The implemented predicate registry lives in
`anthro_chess.evaluation.slices`. It records whether a predicate is decidable or
heuristic and owns the `only move` derivation that legality slicing also reads.
Immediate threats use the conventional null-move question: if the side to move
passed, could the opponent mate on the reply? The label is derived only for
evaluation; null is never exposed as a model action. The
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
either dial therefore commits one reading per cell rather than a blend of them.
That way round on purpose: a pooled series cannot be split later, while a
per-cell one can always be averaged, and the rating axis is the one a model that
starts answering its conditioning would move first. A greedy cell reports
nothing, since its seats follow their own policy every time and give up nothing
by construction, so both quantities there are pinned by the temperature rather
than measured on the model.

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
generated-play readings count endings over.

A generating benchmark decomposes the games it just played rather than writing
them out and reading them back. The games are already in hand, and a payload
written for this reason alone tracks the games per cell, which buys curve
precision and buys a decomposition nothing: its spread is an order of magnitude
below the rating effect it has to resolve at a small fraction of that count. It
therefore reads its own declared sample of each cell, drawn evenly across the
seeds and across the plan order so it spans the positions and both colours,
rather than whatever games happened to be kept.

Per-decision records can be retained beside it in the machine-local detail tier,
which a session that means to look at individual decisions turns on. They are off
otherwise, like every other per-position record here, because every reported
quantity comes from the summary rather than from them. A decomposition over one
manually played game is a diagnostic rather than a series, so it is not appended
to the results store; committed measurements come from suites that declare their
inputs.

## Novelty

Whether the model degrades on positions unlike those it trained on is the
question this benchmark exists to answer, and distance from training data is
what it has to isolate. Difficulty and legality vary across a human pool on
their own, so a reading that does not hold them apart from novelty reports
whichever of the three moved.

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
validate, or keep stable across checkpoints. The dose is the axis, and an
unperturbed control arm drawn from the same games gives it a baseline. What has
to be held fixed beside the dose is whatever else the perturbation moves, which
is decided per quantity and named where each is defined.

### What Remains Measurable

Perturbation breaks human-referenced metrics rather than degrading them. Once a
prefix is perturbed the game diverges from what the humans played, so there is
no human move at the resulting position and the human's actual continuation may
not even be legal. Move cross-entropy, top-k accuracy, and distribution
comparison are undefined there, not merely noisier.

**Only measurements whose ground truth comes from the chess layer survive out of
distribution.** Legality qualifies, needing no target at all. So does material
gain. This is what makes the two sections one benchmark: perturbation supplies
the novelty axis, and chess-derived predicates supply the ground truth that
still exists on it.

Perturbation also decides which predicates fire, and most of them stop. The
forcing predicates depend on a coherent opponent: check falls from 2.3% of
scored positions on the control arm to 0.2% at full dose, and the positions with
a single legal reply are overwhelmingly replies to a check. Over sixteen times
the pool, full dose leaves 136 mate-available opportunities, 45
mate-threatened, 1 only-move and no stalemate-available. No sample size reaches
them, so this benchmark reports material gain alone and the others keep their
sample over human positions in the adjudicated-decisions family.

Human rates exist only on the control arm, and the reference on a perturbed arm
is the model's own control reading of the same quantity. The reading is a curve
across doses rather than a level to pass, so no absolute target is implied and
the perturbed arm does not become the correctness gate this project rejects.

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

A dip at intermediate dose was predicted here and the offline form does not
show one. The reasoning was that a small perturbation takes the model off book
without yet giving away material, while a large one hands over enough that the
remaining decisions are easy. Read at a fixed size of win, policy mass falls at
every dose step on every checkpoint measured, so the second half of that
reasoning was the mix moving rather than the decisions getting easier. The
prediction stands for the rollout form, where the model plays its own
continuation and a material edge has to be converted.

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

**Legality is paired on position**, and this is not a refinement. A perturbed
arm scores a subset of the control's positions, so reading its mean against the
control's mean over everything reports the composition difference as a novelty
effect. The legality delta reads the control over the arm's own plies.

**Material gain is read at a fixed size of win**, for the same reason one level
up. Pairing on the ply does not reach this one: whether a position offers a
material win, and how large, is a property of the board, and the board is what
the dose changes. A random opponent hangs a queen where a human hangs a pawn, so
pawn-only wins fall from half the control's opportunities to a fifth of the full
dose's while free queens triple. Averaged across sizes that mix shift is large
enough to invert the reading, which is why the bands are the unit and the
opportunity share of each is reported beside it.

Phase is not sliced beneath the bands, though truncation moves that mix too. The
phase gap inside a band is an order smaller than the win-size gap, so holding
phase fixed reproduces the unsliced curve; the per-arm phase counts are kept in
the detail tier so a surprising reading can be checked against what it was taken
over.

The material-gain probe is not a private criterion here. It is a heuristic entry
in the shared predicate registry, and it carries a human reference at dose zero,
which is the reporting rule heuristic predicates require.

`docs/decisions/0024-one-sided-perturbation-derived-novelty.md` owns the
derivation contract, the shakedown reading where unpaired retention inverted the
answer, and why the alternatives were rejected.

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

**Every number carries what the reading can resolve.** The ladder estimates its
own floor a third way: it redraws the games each pairing played, refits, and
reads the spread of everything one fit yields, so ordering, slope, span, ladder
error and both temperature responses are reached by the reduction rather than by
propagating a standard error through it.

The floor is evaluation noise for the reason generated play always is, and it
travels on the measurement rather than being characterized against the series.
Pairings whose seats are all greedy replay rather than redraw and are held
fixed, and one whose every seat is greedy states a floor of zero rather than
estimating one.

Two situations are treated apart from the rest. A number the redraw could not
move carries no floor and the reading names it: a seat that scored nothing or
scored everything has no finite fitted rating and reports the declared spread
instead, and a saturated step function cannot be resampled either. The error
profile beside each seat is a mean over decisions rather than an output of the
fit, so the refit does not reach it and its noise reports as unknown — a floor
somebody could still produce, rather than one that cannot exist.
`docs/decisions/0034-qualifying-a-rating-ladder-reading.md` owns all of this,
including why refitting beats propagating and what the degenerate readings cost.

The draw is over a pairing's games without regard to which of the frozen
openings each came from. Stratifying by opening and colour would leave every
floor narrower than the true run-to-run spread rather than wider.
`docs/decisions/0039-stratifying-the-ladder-redraw-costs-more-than-it-removes.md`
holds that measurement, why the openings' own contribution is not there to
remove, what it says for the curve family, and what would reopen it.

**Read the unfinished count as a reading, not as overhead.** An unfinished game
is the most expensive kind the benchmark plays, since it runs the full limit by
definition, but the limit sits past the longest game in the corpus the model
trained on, so a model that finished games the way its corpus does would reach
it essentially never. The count is therefore a statement about the seats rather
than a cost to tune away, and it is reported per seat as the share of that
seat's games that reached a result, which is one of the three ladder quantities
carrying a direction the project is willing to name. What that share currently
is belongs to a reading rather than to this document.

Each seat's own error profile is recorded beside its strength, computed through
the shared decision decomposition rather than a private one, since that layer
already groups decisions by the dials they were made under and a ladder's seats
are exactly those groups.

**The openings are drawn at one speed class and one rating pool.** The class
comes from each game's own time control and the pool from the source's own
label, which are different derivations that can disagree, so a reading names
both rather than inferring one from the other. Every pairing plays the same
roots, so how that draw is composed reaches every fitted rating: which openings
people play is a strong function of the clock, and a rating is a number in a
pool rather than a point on one scale.

**What naming them does not settle is the scale the fitted ratings sit on**, and
the reading says so beside them. The openings decide what the seats play; what a
*configured* rating meant was decided by the corpus the model trained on, and
nothing in this benchmark can check that the two name one pool. A ladder read
over one population is therefore still a ladder about the dial rather than about
that population's rating scale.

**The ladder is the routine cost of deciding a change.** The declared grid
plays thousands of games per checkpoint, and a sweep is what an adopt-or-drop
comparison is read at, so that cost is paid per accepted change rather than per
milestone. Pairings share nothing until the fit, so they are played across
worker processes rather than one after another; a decision costs more Python
outside the model call than inside it, and spreading them is what reaches that
half. `workers` sets how many play at once, and it decides scheduling rather than
workload, so it stays out of series identity beside the seeds. Seats and their sample are the two
things not to confuse when that cost is under discussion.
Cutting seats cuts cost quadratically and cuts every surviving seat's own sample
linearly, because a round robin gives each seat one pairing per opponent, so a
cheaper ladder is also a noisier one on the axis the benchmark exists to
measure. **Openings are the lever that buys precision instead.** Two things move
a reading: the sampling inside the games that were played, and which games those
were. A seed redraws the first and leaves the second alone; openings narrow
both, because more openings are also more games. Decision 0080 measures how far
apart the second one puts four disjoint draws, and 0020 records that a wider set
of openings continues no series a narrower one started.
Cutting *pairings* while keeping the seats is not a third option at all: at a
fixed game budget the complete round robin is the design that resolves the fit
best, so a subset spends the same games for a worse reading.

Openings are also the only lever that reaches a pairing of two greedy seats,
which replays whatever it is given: a second replicate there would enter one
result into the fit twice, so such a pairing plays one and records the seed it
played. Decision 0080 carries what that leaves the declared grid playing.

`docs/decisions/0022-one-joint-rating-ladder-fit.md` owns the joint-fit rule,
why the ablated arm sits inside it, and what the round robin costs.
`docs/decisions/0080-the-ladder-widens-and-openings-replace-seeds.md` settles
the grid at its declared size, and owns why openings rather than seeds are what
make a reading of it finer.
`docs/decisions/0030-ladder-ply-limit-at-the-trained-bound.md` settles the ply
limit against the corpus's longest game and owns why an unfinished game is a
reading rather than a dial to move.
`docs/decisions/0034-qualifying-a-rating-ladder-reading.md` owns how a reading
is qualified and what the two degenerate fits are qualified as.
`docs/decisions/0064-the-complete-round-robin-is-the-optimal-ladder-design.md`
owns why the pairing structure is complete and what an incomplete design was
measured to cost.

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
sampling noise at temperature zero.

**The reading is ordering and level on the puzzle scale, never agreement with
the configured one.** A Lichess puzzle rating and a Lichess game rating are
separate Glicko pools, so an expected-score formula fed one of each states a
difference between scales rather than between a player and a puzzle. The
benchmark reported such a reference and two distances from it until
`docs/decisions/0077-the-puzzle-scale-is-not-the-game-scale.md` withdrew them.
What survives is scale-free or stated in puzzle points: solve rates, the fitted
puzzle rating each configured rating produces, the slope through those fits,
and their pairwise ordering.

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

The fixed yardstick is the selected rows, not the source they are cut from. The
selection is vendored in the repository and the build reads it rather than the
archive, so the pinned identity stays reachable on a machine that has never
downloaded anything.
`docs/decisions/0044-the-puzzle-selection-is-vendored-not-refetched.md` says why
that boundary moved and how long the first pin survived upstream.

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
from real games on the same platform the corpus is drawn from, so a fifth of
the set is cut from games the corpus holds outside its test partition. That
overlap was measured against the solve rates and does not move them:
`docs/decisions/0078-puzzle-training-overlap-is-measured-and-not-corrected.md`
carries the comparison and why no filter is applied.

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
range cannot move at all. The solve rates are read off that same draw rather
than a separate one, since a redraw of the scored puzzles moves every quantity
the reading reports together. Every stored puzzle measurement carries its own
spread, so a delta between two puzzle readings is floored under
`docs/decisions/0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md`
like any other.

Selection is uniform over every exact integer puzzle rating in the declared
range, with deterministic hash ranking only among eligible puzzles at that
rating. This removes the source population's rating-density bias without
creating arbitrary selection discontinuities at a handful of wide band
boundaries.

A reading may score fewer puzzles than the artifact holds. The dial counts
puzzles per exact rating rather than puzzles outright, and keeps the
lowest-ranked of them under the
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

Wide rating bands are the drill-down. The generated manifest records
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

The detail artifact carries the configured-rating grid, the rating-band
drill-down, and the resampled response resolution. The summary tier carries
overall solve rates, fitted-rating slope, and pairwise ordering.

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

Timing readings keep the corpus's full clock precision. Coarsening one to match
an external model trained on second-resolution clocks would put timing metrics
on two scales at once, and would give up the precision the corpus was chosen
for. An external comparison instead gets a slice holding more than thirty
seconds remaining, which is inside the operating range every published time
model reports on, and where second-resolution labels cost that model least.

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

**The human reference is one population rather than a mixture of them.** Game
length and how a game ends are strong functions of the clock, so a reference
drawn from however the pool happens to be composed reports that composition as a
distance. The harness plays untimed, so a speed class slices the reference
rather than the model, and it is the class `anthro_chess.data.speed` derives
from the game's own time control. The reference names a **rating pool** as well,
because rating is this comparison's own axis: every reference game is placed at
the mean of its two players' ratings, and numbers drawn from two pools are two
scales plotted as one. **One class per reading**, so a population the pool holds
no game of is not a distance over nothing but a suite with nothing to compare:
it fails in the pool pass, before the games it would have measured are played.
The same class selects the human-prefix arm's roots, since a mixed set of
openings compared against one class's reference would put that difference into
every distance.

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

Divergence as a function of book depth, the same distance recomputed with the
classification truncated at each ply, says *where* the model departs from human
play rather than whether it does. The size is the exact walk's to report, and
two checkpoints unlike humans by the same amount can arrive there from the first
move or from the middle of theory.

The curve itself stays a diagnostic in the detail tier. Category count grows
with depth, so a per-ply distance read against another checkpoint's compares two
different numbers of categories, and no point of it may be shown without the
null and floor it carries. What is committed is one location taken from it: the
book ply by which half the distance past its null has accumulated, interpolated
so it is continuous rather than quantized to the truncations it was read at, and
reported only where that excess clears the floor. A half of an excess that is
itself noise is a depth drawn from noise.

The sweep runs to the deepest ply at which a label on either side still changes
rather than to the deepest book match, since past that point every truncation
classifies both sides identically. It carries its own workload for the reason
the walk does: the depth it ran to and the resamples behind its null both decide
what a half of it means.

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
saturates near one, which declares an informative reading unusable.

What rescues it is the same structure that separates a waypoint from a
destination. A destination has one reachable label at the reported level, so a
line pruned there keeps the label it already has however it continues. Only mass
pruned while still uncommitted — sitting on a waypoint, or off book — can move
the distribution. Report that as the bound and the assumption-free number beside
it. The tighter bound inherits
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

The floor is the comparison's own bootstrap over the **streams** it generated
rather than over the games, because a game's seed is derived without the rating
it was conditioned on and one stream therefore plays a game at every point of
the grid. The per-seed distances are recorded beside it as a diagnostic and are
deliberately *not* used as a floor: each seed plays only its share of the suite's
games, so a per-seed reading is a smaller-sample one — noisier, and biased away
from the reference, since a distributional distance estimated from few games per
rating point reads high. Comparing that spread against the pooled reading's floor
compares two sample sizes and makes the bootstrap look roughly the square root
of the seed count too narrow.

Read like for like — each seed's own reading against the spread of that reading
across thirty-two seeds — the draw over streams reproduces the true spread from
four streams upward, while a draw over games reports about two thirds of it at
any size, and about half of it at four streams.
`docs/decisions/0060-a-curve-resamples-the-stream-not-the-game.md` owns that
measurement, and the floor a reading below three streams states instead: none,
because two streams leave three resamples and their agreeing is not a zero.

The temperature-zero row is not compared against human play at all. Its seats
are greedy, so it plays one game per position and the model side is a point mass
where the human side is a distribution. A distributional distance against a point
mass reads one minus the human mass of whichever single category the model landed
on, whatever the model plays, which is a statement about how popular an opening
is rather than about how well it was chosen. No sample size repairs that, since
the row plays one game per position by construction. Its cells are still
measured, because a collapsed distinct-game fraction is what the sampled rows'
fraction is read against, and the question the row wants to ask belongs to the
prefix arm, where many distinct starting positions give greedy play a
distribution of its own.

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
only about the model when the model's games actually end. A game the ply limit
stopped contributes a result humans essentially never produce, so a high
unfinished rate makes the result distance partly a statement that the model does
not finish, and makes the mean length a censored lower bound rather than an
estimate. Neither is a defect in the comparison, since the ply limit is in the
declared workload and the unfinished rate is reported beside the distances, but
the two are read together. Repetition, cycle, opening, and move diversity are
computed from the play itself and stay interpretable regardless.

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

The readings split by where they are measured rather than by subject. What a
checkpoint's own games end as is a property of generated play and is read by
the rollout, over games it plays with both terminal actions enabled. Whether
the policy knows when to resign is answerable on games humans already played,
costs one pass, and is `anthro eval termination`.

### Endings On Generated Play

The mix is a human-reference curve comparison over derived termination
categories, and shares the shape described under human-reference curve
comparisons rather than defining a new one. It is one of generated play's
compared quantities, beside the result distribution it refines.

**The human side counts only endings the model could have produced.** A game
that expired on a clock the harness does not run, that a player walked away
from, or that the two players agreed, is left out of this quantity and stays in
every other one. A total variation distance is the mass that has to move, so
counting a category the model is barred from producing puts a floor under the
distance, and the floor is not the damage. The mass has to land somewhere,
which pushes the model above the human rate on every category it does produce,
and a distance whose two sides sit on one side of each other is flat in exactly
the redistribution it exists to report.
`docs/decisions/0083-the-termination-mix-compares-reachable-endings.md` records
what that cost and what it took to see it.

The ply limit is the mirror image and stays where it is: a game the harness
stopped has no human counterpart, so it sits in the model's own vocabulary with
nothing opposite it. That is a gap a checkpoint can close, unlike the four the
human side drops, and it is charged here as well as in the unfinished rate.

Two guardrails matter more than closeness of fit, because their failure modes
are not symmetric. **Premature resignation**, meaning resigning from positions
that are not lost, is the product-critical failure: it is worse than never
resigning and it is invisible to every other benchmark. **Silent non-use** is
the opposite failure, where an enabled terminal action is never selected. Both
are reported explicitly rather than inferred from a distribution distance, and
per cell, so the rating axis stays readable.

Judging whether a resignation was premature needs a position-quality signal.
Material balance is the dependency-free proxy and is enough to catch the
egregious cases; an engine-derived signal would be sharper and is subject to the
engine-dependency decision recorded elsewhere in this document. Because the
proxy is heuristic rather than decidable, the model's premature rate is reported
beside the same rate computed over the human reference, following the rule the
adjudicated decisions above already state: a heuristic predicate is read against
a reference, never as an absolute. The two are attributed differently and have
to be: a generated resignation belongs to the seat holding the move, while a
human platform accepts one on either turn, so the human side reads the seat the
result went against.

Draw claims are rare enough in human data that a distribution comparison carries
little information. The reading that matters is the untimed non-termination
rate: generated games that reached a claimable dead position and never ended.
That is the failure the claim action exists to prevent. Correctness gates should
also cover constructed claimable-threefold and automatic-draw sequences, so
claim availability and claim handling are exact rather than sampled.

**The terminal actions are enabled for generated play.** A seat that cannot
resign plays every lost position out to mate or the ply limit, and humans end
better than a third of their games by resigning, so its length, result, and
termination distributions are held away from any human population by a gap no
checkpoint can close. Both switches join the declared workload, so a run with
them off cannot share a series with one that has them on.

### Resignation Prediction On Human Games

`anthro_chess.evaluation.termination` implements this and `anthro eval
termination` is its reading surface. One deterministic pass over a fixed view
of frozen human games, scoped by the content it scored rather than by a
generation recipe, which is what makes it cheap enough to take at a training
cadence.

The **mass separation** is how much more probability the policy puts on
resigning at the plies where a human resigned than at the plies where one moved.
Neither half means much alone, since both rise together on a model that has
merely learned the action exists.

The **deficit calibration** is the same mass read against material rather than
pooled over every ply: the policy's mean resignation mass in each band of
material the player to move was behind, against the share of plies in that band
where the human resigned rather than moved. Both sides come from the same plies
of the same games, so the position distribution is shared rather than each side
bringing its own, and a model that spends as much resignation mass as humans
while spending it in the wrong positions is separated from one that does not.
The headline weights each band by how often a position in it comes up, so it
reads as the gap at a ply drawn at random; the bands themselves travel with the
reading, because the tail is where the interesting disagreement is and the
weighting is not where it shows.

Every reading with no population behind it reports an explicit unavailable with
its reason rather than a zero. This matters most where a zero is a plausible
measurement: a view holding no game that carries a terminal action has no
resignation to score the policy against, and writing zero mass there would read
as a policy that never wants to resign. That is also what a corpus prepared
before the terminal actions existed looks like from here once it is compatible
enough to load at all; an incompatible one is refused by the pool loader and the
model runner on vocabulary identity, well before any metric is computed.

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

**None of these readings carries a floor, and every metric says why.** The only
replicate of a training reading is a second training run, so each metric
declares its reason in the registry and a report reads `unqualifiable` rather
than `unknown`, which stops a reader waiting on a spread nothing can produce.
Attributing a cost change to a training change is therefore a control-arm
reading, as it is for any other causal claim.
`docs/decisions/0061-a-training-cost-reading-has-no-replicate-to-resample.md`
owns why, including what a within-run interval spread was measured to cost.

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

Two questions, and they want different instruments. What does this checkpoint
cost to play with? And what did a model change cost it?

The first is a wall clock. An opponent too slow to play against is a product
failure regardless of how it scores on move loss, so this is part of the
checkpoint suite rather than an operational aside.

The second is a count. One decision reads a fixed number of square tokens, which
is far too little work to occupy an accelerator: nearly all of a single forward
pass there is kernel-launch overhead, so batch-one latency is close to
independent of model size, and collapsing the launches into a graph replay
leaves it that way. A benchmark that reported only wall clocks would therefore
say a substantially larger model costs nothing, which is the blindsiding this
exists to prevent.

### What A Decision Costs The Model

**Parameters and floating-point operations per decision**, counted from the
loaded module rather than timed. They carry no noise, need no floor, and no
process count buys resolution on them.

They are not a proxy for the timings and the timings are not a proxy for them.
Wall clock understates an arithmetic increase whenever the device is launch or
bandwidth bound, which is where this model sits at every batch size it serves,
so a larger model reads as cheaper than its operation count says. Both are
reported because that gap is the answer to what a size tradeoff actually costs:
the count says how much more work, and the clock says how much of it is paid for
on this hardware.

**Peak device memory** at the declared compute batch. An absolute figure that
includes the resident weights and the batch itself rather than the forward
pass's increment over them, because what it answers is whether a device can hold
this batch at all.

### What It Costs To Play With

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
and what let a from-scratch re-encode, work the engine never does, render as a
leading term.

There is deliberately no encode stage, and that is the substantive finding
rather than an omission. A session encodes one ply as it advances rather than
encoding a history per decision, so the only encode a decision pays for is that
single ply; it is flat in history length and falls inside the remainder
alongside masking and sampling.

Latency is measured at one depth rather than swept across several. The model
folds history into the channels of a fixed set of square tokens rather than into
a sequence, so a decision costs one position whatever its ply count, and one
depth says what every depth says.

**Serving throughput**, in whole batched decisions per second at the declared
serving batch size, through the same loop the generated benchmarks run: collect
every pending context, resolve them in one forward pass, then mask and sample
each result. Batching trades latency for throughput, so quoting a serving figure
as an interactive one is the usual way that trade gets hidden.

**The non-model share of a batched decision**, taken as the difference between
that loop and the forward pass alone: context assembly, batch construction,
legal masking and sampling. The device-to-host copy is inside both timed windows
and cancels out of the difference rather than appearing here. It does not
amortize with batch size, because it is one decision's own host work, and it is
most of a batched decision at the sizes that matter. It is also where a change
to the encoding or the action vocabulary lands, rather than in the forward pass,
which is why it is reported rather than left implicit in the throughput figure.

**The model call alone**, at a compute batch size wide enough that the device is
no longer launch bound. Nothing serves that wide; it is an instrument, and it is
the only wall clock here that separates two model sizes at all. It reads one row
per game rather than scoring every historical ply, which is what serving does,
and it stops at the module: the host copy and the finite check a served decision
pays are fixed costs that do not grow with the model, and leaving them in would
dilute this figure by more the wider the instrument got. It is declared on its
own workload, so moving the instrument does not end the product timings' history
and moving the serving batch does not end its own.

The serving figure's forward half is the other way round, and deliberately: it
times the seam a decision actually goes through, because it is subtracted from a
whole decision and both sides have to pay the same host costs for the difference
to be that decision's own work.

**Cold start**, split into model-load time and the first decision after loading.
Lazy kernel compilation and allocator warmup land in the first decision rather
than inflating the steady-state percentiles, which is why warmup is excluded
there and measured here.

### The Host Reading

A run on an accelerator measures the host as well, and the two do not pool: the
device is part of each reading's declared workload rather than an environment
coordinate, so a report cannot read one line moving when it is looking at two
devices.

It answers two things nothing else here does. Whether the engine is playable
without an accelerator, which is a product question the accelerator reading
cannot address. And what a model change costs in a wall clock at a single
decision, since the host has no launch floor for the arithmetic to hide under
and its batch-one latency tracks the operation count where the accelerator's
does not. It is also the quieter of the two readings from process to process.

### Reading The Numbers

The workload is synthetic and self-contained rather than drawn from the
evaluation pool. Latency depends on history length and legal-move count, not on
which human played the game, so binding this benchmark to the pool would break
its series at every pool generation without changing what it measures. Positions
come from a seeded legal-move walk, so the same declared workload replays the
same positions on every machine. Reaching a deep position requires that walk to
arrive somewhere a session can still decide from: random play thins the board
down, so a walk that never ran out of legal moves still lands on a position that
is over by rule often enough to abort a run.

Every batch is timed on its own so the reported rates come from the median
rather than the mean, which stops one descheduled batch carrying either. That
does not narrow run-to-run spread, which is a property of the process a reading
was taken in; the benchmark handles that by taking the reading in several
processes, as **Execution Noise** above describes.

Accelerator work is asynchronous, so every measured window synchronizes queued
device work before stopping its timer. Without that, a benchmark would time the
enqueue and attribute the real work to whichever window happened to block next.

### Comparing Efficiency Readings

Three questions are worth asking of these numbers, and they differ in what is
held fixed. Did a model change cost us speed? Did an environment change buy us
any? And what is the net effect on the thing we actually ship? A report
therefore declares a **pivot** rather than assuming one.

The default pivot varies the checkpoint. When the environment moved as well,
the delta is still shown, since it is a real and interpretable number, but the
verdict is reported as `confounded` rather than better or worse, with an
attribution naming which of model, environment, and workload changed. The
honesty lives in the verdict rather than in a withheld delta, because any reader
holding both values can subtract them, and automation reads the verdict.

The environment pivot is the mirror image: the model is pinned by parameter
digest and the machine, precision, or software version varies. That is the
question an optimization asks, and pinning by digest rather than by label is
what stops a model change being sold as a hardware win.

Metric history is one continuous line per workload, annotated where the
environment changed, which is what makes long-run drift answerable at all. A
workload change does break the line, because that genuinely is a different
measurement.

Whether a delta between two of these readings means anything is a separate
question from whether it is comparable, and it is answered by the spread each
reading measured across its own processes. A reading taken in one process
carries none, and a report then says the noise is unknown rather than calling
ordinary run-to-run jitter an improvement. The counted quantities need none of
this: they read identically in every process, so a difference in one is a
difference in the model.

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
delta attributable.

A control is a **training identity**, not a simultaneous run. A prior checkpoint
is a control when its compatibility identity matches the candidate's in
everything but the change under test, and an arbitrary recorded reading is not
one, because a baseline that drifted in corpus, step budget, or machine carries
those differences into the delta. Which of the two a comparison holds is decided
by the recorded identity rather than by recollection.

Two identities serve as controls, for different questions. The **canonical
line's** prior checkpoint is the control for a change to that line, and is what
the paragraph above describes. The **ablation vehicle** is the control for a
candidate being evaluated for adoption, and it differs in one respect that
matters here: it is frozen, so adopting a candidate does not advance it, and the
seed dispersion characterized against its digest stays current however many
candidates are accepted. That is the whole reason it exists, and
`docs/scaling.md` owns the program it serves.

A vehicle comparison is qualified by both floors — the combined evaluation floor
every reading carries, and the vehicle's seed floor. The seed floor describes
baseline arms, so it does not describe a treatment whose training-health readings
depart from the vehicle's: instability widens an arm's spread past what the floor
allows, which makes the floor read too narrow rather than too wide. It describes
the vehicle at the horizon it was characterized at as well, and the digest does
not hold that, so a reading from a cooldown branched at another horizon matches
the floor without having been shown to share its spread. A comparison in either
state reports the mismatch instead of quoting the floor.
`docs/decisions/0029-model-change-control-arm.md` owns why the control is
required rather than recommended, what it costs, and what it still does not buy;
`docs/decisions/0063-the-full-sweep-decides-a-change-and-the-canonical-line-is-its-byproduct.md`
owns the qualification and why a second training run buys nothing the identity
check does not.

`anthro eval report` states that identity once, above its rows, and the rows go
on saying which way they moved. The caveat is a header rather than a verdict
because the identity moves on every comparison that tests a change — an
architecture, a learning rate, a corpus filter all reach the digest — so a
per-row label keyed on it would read the same on every row of every report and
discriminate nothing. An identity missing from either reading, recorded before
the field existed or taken through a runner supplied rather than loaded, is
reported as unverified rather than as a match; most of the committed store
predates it.

What the header buys is therefore the negative check. A match says nothing on
the training side moved, which is what a comparison between two checkpoints of
one run, or between seed replicates, is claiming. A mismatch says only that
something moved, never that it moved by the change under test, so it confirms
what the reader already knew in the ordinary case and catches the one that
matters: a baseline believed to differ by one change that also drifted.

A match is one training configuration rather than one run, and the difference is
what the digest leaves out: the seed by construction, and the step budget and
the device with it. So it settles the corpus and the hyperparameters, and two
arms still have to be read at the same checkpoint step for the same reason they
are read on one machine. The environment pivot asks the opposite question and is
told nothing here: the arithmetic a machine does is inside the identity, so an
upgrade moves it, and naming that as a caveat would refuse the comparison the
pivot exists for.

Both arms are read the same way: the sweep at its declared sizes, at the same
checkpoint step, on one machine — and the claim is written down before either arm runs: which metric
moves, in which direction. Stating it first is what makes the reading
falsifiable, for the same reason a shakedown states its expectation first.

The declared size is what a claim is read at. A smaller reading only widens
floors, which protects against admitting a weak claim and does nothing about
discarding a real improvement the reading was too coarse to resolve; under a
comparison that decides adopt-or-drop, those cost the same.

A size chosen with `--set` is therefore part of the claim rather than a response
to it. It is chosen before the arms run, and a delta inside its floor is a null
result rather than a reason to re-read larger: a reading widened because its
answer was unwelcome is the same failure as an arm retrained for a better
number, and two sizes are separate series in any case rather than two precisions
of one. Where a small effect is expected, `uv run anthro eval noise plan`
reports how many games an axis needs to resolve an effect of a given size, which
is a question for before the reading.

The comparison itself needs nothing new. A training run is a coordinate rather
than a component of series identity, so two arms of one configuration land in
the same series, and `uv run anthro eval report` reads the delta between them
from their checkpoint labels. Arms are recorded into a machine-local store
rather than the committed one, which needs no arranging because it is where a
benchmark writes by default: a candidate arm is not project history, and an arm
nobody adopted would otherwise become some later report's baseline.

So the reading resolves once, runs per arm, and compares once:

```console
uv run anthro eval suite --config <suite> --plan
uv run anthro eval suite --config <suite>
uv run anthro eval report --current <treatment> --baseline <control>
```

`--plan` first for the reason "The Benchmark Suite" gives: a sweep that will
fail should fail in its first second rather than after it has spent. Neither
sweep names a store, because the default is already the machine-local one.

More than one treatment arm reads against one control where what is being
chosen is a dial — a loss weight, a capacity, a budget — rather than a set of
independent candidates. State the response expected across the dial before the
arms run: which metric moves with it, and where a guard metric is expected to
turn. Asking a separate adopt-or-drop question of each arm asks one question
several times and adopts whichever arm seed luck favoured, while a predicted
response across the dial is a single claim that an uncorrelated per-arm
perturbation cannot fake. The novelty benchmark reads a dose the same way.

A reading reaches the committed store when its change is accepted — by
`anthro eval promote` copying it there in that change's pull request — so the
committed line is the sequence of accepted checkpoints rather than a log of
everything attempted. **Nothing is run for that line.** It accumulates from the
comparisons already being made, which is what keeps a durable history from
carrying a cost of its own. A series that has to last is read at the unbounded
view, for the reason "Benchmark Data Layers" gives above, so its precision never
needs raising later.

**What makes a delta admissible is narrower than the machinery suggests**,
because of what a floor is built from. **A benchmark floor does not see
training-seed noise.** It is combined from what the two readings' own units
could have moved, and seed variance is a property of the training run rather
than of the benchmark, so nothing a reading measures can reach it. Such a floor
says the delta survives a different draw rather than that the change produced
it. The report says so beside the verdict: `cleared` means larger than benchmark
noise, and never that the change caused it. The exception claims less rather
than more: a replayed reading states a spread of zero, which says its games
cannot be redrawn at all and therefore says nothing about a draw that could be.
Two arms differ by their initialization seeds as well as by the change, so
clearing that floor alone establishes that two models differ, not that the
change is why.

**The seed floor is what does see it, and only where one has been
characterized.** Arms of one training configuration differing only in their
initialization seed give the spread directly, and the spread is stored against
that configuration's `training_sha256` where a comparison finds it by exact
digest or reports that it has none. That is affordable for a configuration that
stands still and nowhere else, so the ablation vehicle is the one base carrying
one; `anthro eval seed-dispersion` is what characterizes a base, and
`anthro eval report` prints the second verdict beside the first. Two scope
limits withhold it rather than widening it: an arm whose training-health
readings depart from the base's arms, and a reading taken at a horizon the
characterization was not, since the horizon sits outside the digest.

A claim therefore rests on a delta clearing both floors, or on one far enough
outside seed variance that nothing else explains it, or on arms read at several
seeds — a deliberate, occasional act for a result worth its cost, rather than
machinery riding on every comparison. Anything narrower is reported as what it
is, a delta not distinguished from seed variance, rather than as an improvement.
A family with no floor at all can show that nothing else moved; it cannot carry
the claim.

Two numbers make that a bar stated in advance rather than a judgement made after
the reading: a delta carries a claim at **twice the printed floor**, and
**several seeds is three**. Both are starting points rather than a ratified
standard, set where no measurement yet says otherwise so that a bar exists at
all. A comparison says which it used, and a reading that argues for a better
number says so.

Seed replicates belong to the control rather than to the change under test. A
base that a run of changes will be tested against is trained at several seeds
once, and each comparison against that base reads against the spread those arms
showed; only a new base pays again. That storage is narrower than the four
kinds, six producers, and fingerprint-keyed index decision 0043 collapsed: one
characterization per training identity, findable by exact digest and by nothing
else, which decision 0065 reopened for a base that cannot move and for no other.

A null reading is a reading. Arms are not re-run until a number improves, and a
delta inside its floor is a null result rather than a small win.
`docs/issue-workflow.md` owns what that means for the pull request and the
issue.
