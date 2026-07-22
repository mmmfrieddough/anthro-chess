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

Useful metric families:

- training health;
- legality;
- rating calibration;
- timing;
- move-time coherence;
- human-likeness;
- preference controls.

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

## Evaluation Layers

Different checks belong at different points in development and training.

### Unit And Integration Tests

These should run when code changes.

They should cover:

- chess-rule behavior;
- board reconstruction;
- legal move generation;
- model-facing encodings;
- data parsing and preprocessing;
- sequence construction and causal-mask behavior;
- clock-state simulation;
- runtime legal masking.

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

### Periodic Benchmarks

These should run less often than normal validation.

Useful periodic benchmarks include:

- held-out move distribution checks;
- timing distribution checks;
- move-time coherence checks;
- legality diagnostics on tricky rule positions;
- fixed position suites;
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

A normalized diagnostic can compare the model against uniform probability over
the move portion of the action vocabulary:

```text
uniform_legal_mass = num_legal_moves / move_vocab_size
legality_lift = logit(legal_mass) - logit(uniform_legal_mass)
```

Normal validation should use held-out human positions. A separate tricky-rule
suite should stress positions involving check, pins, castling rights, en
passant, promotions, stalemate-adjacent states, only-move situations, crowded
tactical positions, and positions with very few legal moves.

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

Human prefixes are especially useful early, when the model may not yet be good
at creating coherent full games from the start position.

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
