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

Families cover training health, held-out prediction, legality, rating behavior,
timing, generated play, and later additions such as move-time coherence,
human-likeness, and preference controls. The metric registry in
`anthro_chess.evaluation.results` owns the exact family and metric identifiers,
their declared directions, and their definition versions; `anthro eval metrics`
prints them.

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

### Where The Store Lives

`anthro_chess.evaluation.results` implements this layer and owns the exact
record schema, metric registry, fingerprint algorithm, and size budget.

The committed summary tier is one small JSON file per result under the store
root, beside the bridges that rejoin a broken series. One file per result is
what keeps concurrent appends and Git merges additive; a concurrent write into
the same store fails on an exclusive lock rather than producing a partial
record. The store root defaults to `results/` in the repository and can be
pointed elsewhere with `ANTHRO_CHESS_RESULTS_ROOT`.

The detail tier is machine-local and holds per-position diagnostics, slice
tables, and generated game records. A summary record references a detail
payload by path and digest rather than embedding it, and the store refuses a
record whose payload belongs on the other side of that boundary. The detail
root resolves from `ANTHRO_CHESS_RESULT_DETAIL_ROOT`, or beneath
`ANTHRO_CHESS_RUN_ROOT` when that is where runs already live.

`anthro eval report` is the reading surface: a compact delta view by default,
with slices, provenance, per-series history, and machine-readable output behind
explicit options. `anthro eval bridge` records, lists, and revokes bridges.

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

Noise characterizations are stored in the results store under the same
fingerprint rules as any other measurement, so they invalidate on the same
terms rather than lingering as stale constants.

Training noise should be characterized early, while runs are short. It is the
most valuable of the three and the only one that becomes harder to obtain over
time: once runs are long and expensive, several repeat runs stop being
affordable, and the project loses the ability to distinguish a small improvement
from seed luck for the rest of its life.

Sampling-noise estimates are also what size the evaluation inputs. How many
games an axis needs in order to resolve an effect of a given size is a
computable quantity, not a guess, and it should be computed rather than assumed
when a pool generation is planned.

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

Opening distributions should group games by a classified family computed from
our own versioned book rather than by literal move prefix. Prefix grouping
fragments broad openings across many buckets while narrow ones keep their mass
in one, and it splits transpositions that reach the same position by different
move orders, so it cannot support a statement like "this checkpoint plays too
few Sicilians." Source ECO and opening headers are deliberately not used: their
granularity is fixed, their assignment is not standardized across databases, and
their name strings differ per source.

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
