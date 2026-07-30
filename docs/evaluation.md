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
rating behavior, generated play, decision decomposition, timing, training
efficiency, inference efficiency, and later additions such as move-time
coherence, human-likeness, and preference controls. The metric registry in
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

Efficiency metrics carry one further component: the **declared workload**, the
settings that decide what was timed. Change the ply depth a latency figure is
taken at and the number measures a different quantity, so that belongs in
identity just as scored content does. The **machine** deliberately does not. A
cross-machine latency delta is interpretable rather than meaningless — it is
just attributable to the environment rather than to the model — so it is
attributed by a report instead of ending a series.
`docs/decisions/0018-workload-scoped-efficiency-series.md` owns the rule.

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
explicit options. `anthro eval bridge` records, lists, and revokes bridges.
`anthro eval noise` characterizes floors, lists them, and answers how many
games an axis needs. `anthro eval inference` measures what a checkpoint costs
to play with; see inference efficiency below. `anthro eval decisions` separates
model error from sampling error over a payload of generated games or a played
session's log; see decision decomposition below. `anthro eval puzzles` measures
the external puzzle-rating response described in the rating section.

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

Report the effective local sample size alongside the curve, so a difference
where the human reference is thin is not read as the same strength of claim as
one where it is dense.

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

Three sources of noise are distinct, and conflating them is the usual mistake:

- **evaluation noise**: the same checkpoint re-measured on the same data.
  Deterministic offline metrics over a frozen pool have none; rollout metrics
  have a lot, driven by seeds.
- **data-sampling noise**: how much the metric would move on a different draw of
  the same size from the same population. Estimable by bootstrapping from a
  single run, so it costs nothing, and it is what says whether a pool or a view
  is large enough.
- **training noise**: the same configuration trained from a different seed. This
  is the floor that decides whether a *model change* is real, and it is the
  expensive one, since it needs several training runs.

All three reduce to one reportable quantity: the spread of the metric across
replicates of that noise source. A **floor** is that spread expressed as a
delta, because a delta is what a report shows and a standard deviation is not
directly comparable to one. When two measurements use independent inputs, the
floor covers the difference between two independent replicates at a declared
confidence. When comparable checkpoints score the same frozen units, the
data-sampling floor instead comes from resampling their paired per-unit
differences. One coverage factor is declared per characterization, and
`anthro_chess.evaluation.results` owns the arithmetic, stored inputs, and
lookup.

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

Because a data-sampling floor costs only a resampling of numbers a run already
computed, the checkpoint evaluation runner produces its own and records it
alongside the reading where its inputs are independent. A deterministic
fixed-input benchmark retains aligned per-unit contributions in the detail tier
instead; reporting joins those contributions and bootstraps the checkpoint
delta. Such a floor belongs to the comparison and cannot correctly be attached
to either checkpoint alone. If either detail payload is unavailable, its paired
floor is unknown rather than replaced with an independent-input estimate. A
benchmark whose floor is a function of its own configuration rather than of a
series — a distributional distance, whose floor grows with the category count
and shrinks with the sample — attaches the floor to its measurement instead,
because that is the only place it can be correct.

A delta is judged against the widest floor that applies to it, since a finding
has to clear every noise source, and the report names which one that was. A
delta inside its floor is still shown with its value, so a small regression that
repeats across checkpoints stays visible instead of being filtered away.

Training noise should be characterized early, while runs are short. It is the
most valuable of the three and the only one that becomes harder to obtain over
time: once runs are long and expensive, several repeat runs stop being
affordable, and the project loses the ability to distinguish a small improvement
from seed luck for the rest of its life.

Sampling-noise estimates are also what size the evaluation inputs. A
conservative independent-input estimate is suitable before representative
checkpoint pairs exist. Once they do, paired pilot deltas give the more relevant
power calculation for a frozen benchmark. Either floor shrinks with the square
root of the units behind it, so how many games an axis needs in order to resolve
an effect of a given size is computable rather than guessed.

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
ply count, clock presence, or rating presence; projecting to prefixes;
subsampling by hash rank. Each benchmark records its resolved view spec,
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
precision. Rollout metrics are the opposite: irreducibly sampled, with view size
as the only dial.

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
signed gap, rating-band drill-down, and clustered sample counts through the
shared result envelope.

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

Greedy and sampled solve rates should both be reported against a declared
reference temperature, since the gap between them is the same quantity the
decision decomposition measures. Multi-move puzzles distinguish first-move
accuracy from completing the line, and those are separate metrics.

The puzzle set is an external dependency with its own identity and license
record because a set version change alters what a number means. It follows the
same boundary as the frozen evaluation pool: the acquisition and selection
recipe plus expected identity are committed, while the generated records and
raw source stay under the data root. Puzzle positions derive from real games on
the same platform the corpus is drawn from, so a source-game-key join against
the training selection reports the overlap rate as provenance. The measured
risk is small, since one exposure among millions does not produce recall and
worst-case inflation is bounded by the overlap fraction. It is worth reporting
anyway because it grows silently as the corpus expands, and the join is cheap
enough that there is no reason to carry the uncertainty.

`anthro eval prepare-puzzles` builds the artifact selected by
`configs/evaluation/lichess-puzzles-v1.toml`; `anthro eval puzzles`, selected by
`configs/evaluation/puzzle-rating-response.toml`, reads it. The canonical set is
sized from a conservative two-independent-proportions calculation at declared
confidence and power. That is a planning bound made before representative
checkpoint pairs exist. Actual checkpoint reports resample the
source-game-aligned differences retained in their machine-local detail
payloads within exact-rating strata, preserving the selection design; they
never use the independent-input bound as the comparison floor. Selection is
uniform over every exact integer puzzle rating in the declared range, with
deterministic hash ranking only among eligible puzzles at that rating. This
removes the source population's rating-density bias without creating arbitrary
selection discontinuities at a handful of wide band boundaries.

The primary drill-down uses the shared nearest-neighbour curve machinery with a
frozen bandwidth and grid. The analytic human reference and model response are
smoothed at the same local bandwidth, preserving the bias-cancellation rule
used by other human-reference comparisons. Wide rating bands remain as a
readable secondary table, not as the estimator. The generated manifest records
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
the aligned per-source-game values needed for later paired checkpoint floors.
The summary tier carries overall solve rates, continuous curve distance,
fitted-rating slope and pairwise ordering, plus the source-game overlap rate.
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
from the chess layer, a learned resignation and the ply limit that stops an
unfinished game are the harness's own, and a claimed draw is marked as
adjudicated while the seats have no draw-claim action. The harness does not
claim draws by default: claiming on the model's behalf would report the
harness's policy as the model's behavior, and games still end on their own
through the fivefold and seventy-five-move rules. Once the action vocabulary
carries a draw claim, claimed draws become a model ending rather than an
adjudicated one, and the adjudication path stays only as the fallback for seats
that cannot claim.

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

Book depth is reported as three quantities, because the raw depth conflates
choosing a well-analyzed line with knowing it. Record the deepest matched ply,
the deepest theory available onward from that position, and the fraction of it
consumed. A model with a human-like repertoire that abandons theory early and
one that plays offbeat lines both show shallow raw depth, and only the
decomposition tells them apart.

Depth is a property of the pair, not of one player, since either side leaving
book ends it. Matched-rating games control this on the human side and a
self-play grid controls it on the model side.

Divergence as a function of book depth — the same distance recomputed with the
classification truncated at each ply — says where the model departs from human
play rather than whether it does. It is a diagnostic and belongs in the detail
tier: category count grows with depth, so its noise floor grows too, and it must
never be shown without that floor. If it ever becomes the number people quote,
that is the signal it was a mistake.

The shallow end of that curve is exactly computable rather than sampled. A
model's policy at a fixed position is one forward pass, so an opening-tree walk
that keeps lines above a cumulative-probability threshold produces the
repertoire distribution with no sampling noise at all and a reported bound on
the pruned mass. Deep readings must still be sampled, which is a second reason
repertoire and depth belong apart: they differ in computational character as
well as in meaning.

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
engine-dependency decision recorded elsewhere in this document.

Draw claims are rare enough in human data that a distribution comparison carries
little information. The reading that matters is the untimed non-termination
rate: generated untimed games that reach a claimable dead position and never
end. That is the failure the claim action exists to prevent. Correctness gates
should also cover constructed claimable-threefold and automatic-draw sequences,
so claim availability and claim handling are exact rather than sampled.

## Inference Efficiency

What a checkpoint costs to play with. An opponent too slow to play against is a
product failure regardless of how it scores on move loss, so this is part of the
checkpoint suite rather than an operational aside.

Three quantities are kept apart, because folding them together lets a win in one
hide a regression in another:

**Batch-one move latency**, reported as percentiles rather than a mean. This is
what a person waiting for a move experiences. It is measured end to end through
the decision runtime, spanning encoding, model execution, legal masking, and
sampling, because that is what a move actually costs; timing the forward pass
alone would report a number no player ever sees and would stay healthy while the
encoder regressed. A mean is reported alongside for capacity arithmetic, but the
median says what play usually feels like and the tail says whether it ever
stalls.

**Declared-batch throughput**, in decisions per second at one declared batch
size. Batching trades latency for throughput, so quoting a serving figure as an
interactive one is the usual way that trade gets hidden.

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
