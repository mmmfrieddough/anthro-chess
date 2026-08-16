# Related Research

This document tracks outside work that may inform Anthro Chess. It is a curated
reference list, not a roadmap and not a list of product requirements.

Each entry notes what part of Anthro Chess it applies to and how it differs
from this project.

## Machine-Learning Development Practice

### A Recipe For Training Neural Networks

Link: <https://karpathy.github.io/2019/04/25/recipe/>

Key information:

- Explains why neural-network failures often reduce quality without causing an
  obvious runtime error.
- Recommends inspecting the exact data presented to the model, simplifying the
  initial setup, overfitting a small sample, and adding complexity one step at
  a time.
- Uses dependency and gradient checks to detect accidental information flow in
  vectorized and autoregressive models.

Applies to Anthro Chess:

- Fixed-batch inspection across board, action, sequence, legal-action, and mask
  boundaries.
- Tiny-overfit and causal-dependency checks before baseline training.
- Incremental introduction of rating, timing, preference, and optimization
  features.

Different from Anthro Chess:

- The source is general practical guidance rather than a chess-specific
  specification.
- Anthro Chess keeps exact chess reconstruction and legal-action handling
  enabled because they are deterministic correctness boundaries, not model
  complexity to remove.

### CS231n Neural Network Training Guidance

Link: <https://cs231n.github.io/neural-networks-3/>

Key information:

- Recommends checking whether initial loss has the expected scale before
  expensive optimization.
- Recommends disabling regularization for a tiny-data overfit check.
- Warns that successful memorization does not prove that inputs are meaningful
  or that the model will generalize.

Applies to Anthro Chess:

- Initial action-loss, masking, and finite-value sanity checks.
- A deterministic tiny-sample capacity test followed by separate held-out
  validation.
- Clear separation between optimization correctness and learned chess signal.

Different from Anthro Chess:

- The examples are oriented toward image classification and generic neural
  networks.
- Anthro Chess also needs causal-sequence, legal-action, nullable-context, and
  chess-state alignment checks.

### Deep Learning Tuning Playbook

Link: <https://github.com/google-research/tuning_playbook>

Key information:

- Recommends beginning with a simple, fast, low-resource configuration.
- Builds performance incrementally from a trusted baseline and adopts added
  complexity only when evidence supports it.
- Separates the work of understanding a change from later broad optimization.

Applies to Anthro Chess:

- A small CPU-friendly initial configuration and frozen comparison inputs.
- One coherent model, context, data, or optimization change per comparison
  where practical.
- Removing changes that do not justify their implementation and tuning cost.

Different from Anthro Chess:

- The playbook focuses mainly on tuning an already running supervised-learning
  pipeline.
- Anthro Chess first needs correctness gates for its data, chess-state,
  action-alignment, and causal-training boundaries before systematic tuning.

## Scaling And Capacity

These entries back the rules in `docs/scaling.md`. All of them measure language
models on text, so their constants do not transfer; what transfers is the shape
of the argument and the failure modes each one names.

### Training Compute-Optimal Large Language Models

Link: <https://arxiv.org/abs/2203.15556>

Key information:

- Fits loss as a floor plus a term decaying in parameter count and a term
  decaying in training tokens, and derives how a fixed compute budget should be
  split between the two.
- Reports that the split found by three independent estimation methods is close
  to even, so parameters and data scale together rather than one outpacing the
  other.
- The loss surface is flat near the optimum, so a size moderately away from it
  wastes only a few percent of compute. Later replication found the published
  parametric fit was produced by a prematurely terminated optimizer, and its
  reported confidence interval is far narrower than its sample size supports.

Applies to Anthro Chess:

- The ladder in `docs/scaling.md` fits the same shape to decide the target's
  split between capacity and training positions.
- The flatness result is why a size within roughly 1.5x of the fitted optimum is
  not re-litigated.
- Checking a fitted interval's width against the number of runs behind it is the
  cheapest available detector of a broken fit.

Different from Anthro Chess:

- Assumes effectively unlimited training data, which holds here only because the
  corpus exceeds every horizon in the plausible range.
- The compute-optimal point is chosen for a model that is trained once and never
  served; this project serves far more than it trains.

### Resolving Discrepancies In Compute-Optimal Scaling Of Language Models

Link: <https://arxiv.org/abs/2406.19146>

Key information:

- Reproduces two published scaling studies that disagreed by more than tenfold
  on prescribed model size, and walks one result into the other by removing four
  experimental confounds one at a time.
- The confounds are: omitting the output head from the compute count, holding
  warmup at a fixed step count instead of scaling it, using a decay schedule not
  matched to the actual horizon, and not re-tuning batch size, learning rate and
  the optimizer's second-moment decay at each size.
- Every intermediate fit was statistically well-behaved. No goodness-of-fit
  statistic distinguished the wrong answer from the right one; only ablating the
  protocol did.

Applies to Anthro Chess:

- A ladder here fixes all four before it is fitted, or it measures the protocol
  rather than the model.
- The output-head omission is the largest of the four and is worst at the
  smallest sizes, which is exactly where a ladder's lower rungs sit.
- It is the direct source of the rule in `docs/scaling.md` that a scale-dependent
  setting is recorded as a rule rather than as a number.

Different from Anthro Chess:

- Its smallest model is far larger than this project's current configurations,
  so the size-dependent share of each confound is not directly comparable.

### Beyond Chinchilla-Optimal: Accounting For Inference In Language Model Scaling Laws

Link: <https://arxiv.org/abs/2401.00448>

Key information:

- Adds lifetime inference cost to the objective, which moves the optimum toward
  smaller models trained on more data than the training-only optimum.
- Refitting the standard form on subsets restricted to short training ratios
  gives noticeably different exponents than refitting with heavily over-trained
  runs included, so a law fitted in one regime misprices the other.
- The authors state their own fitted curves describe their longest-ratio runs
  poorly, so the effect is clearer than its size.

Applies to Anthro Chess:

- The serving-to-training ratio here is high and the serving constraint is loose,
  which argues for the smaller-and-longer end of the size band.
- If the project intends to over-train, over-trained points belong in the ladder
  rather than being extrapolated to.

Different from Anthro Chess:

- Its cost model prices inference in tokens generated by a served language model,
  not in positions evaluated by a chess engine.

### Do Transformer Modifications Transfer Across Implementations And Applications?

Link: <https://arxiv.org/abs/2102.11972>

Key information:

- Re-implements roughly forty proposed Transformer modifications in one codebase
  and evaluates them under matched conditions.
- Most modifications did not reproduce their reported gains. Gated-activation
  feedforward variants were the clearest survivors; several arms that beat the
  baseline on every reported metric were nonetheless not adopted by the field.
- Hyperparameters were deliberately held constant across arms, which
  systematically penalizes any modification that needs retuning.
- Reports per-arm run-to-run standard deviations, and unstable arms show
  several times the baseline's spread.

Applies to Anthro Chess:

- Sets the prior for the candidate-change phase: most ideas will measure as
  nothing, and the reading that says so is the useful outcome.
- The constant-hyperparameter caveat is why a candidate discarded on a negative
  reading is only discarded where it had its own learning rate.
- Instability inflating an arm's spread is the case where the vehicle's seed
  floor does not describe the treatment arm.

Different from Anthro Chess:

- Evaluates on text transfer tasks with a text encoder-decoder, so neither its
  rankings nor its effect sizes carry over.

### Tensor Programs V: Tuning Large Neural Networks Via Zero-Shot Hyperparameter Transfer

Link: <https://arxiv.org/abs/2203.03466>

Key information:

- Defines a parametrization under which the optimal learning rate and several
  related settings stay stable as width grows, so they can be tuned on a small
  proxy and reused at scale.
- Transfer across width is well supported; transfer across depth is materially
  weaker and later work finds limits for the residual-block depths modern
  Transformers use.
- Independent replication finds transfer holds in most settings but is broken by
  specific architectural choices, including certain attention-logit scalings and
  trainable normalization gains.

Applies to Anthro Chess:

- One of two viable routes for the hyperparameter-rule step, the other being an
  empirically fitted rule per setting.
- The architectural breakers are worth avoiding regardless of the route, since a
  broken transfer rule silently confounds every comparison across a size change.

Different from Anthro Chess:

- Depth transfer is the weaker half, and a small model reaches a useful size by
  adding layers as readily as width.

### Scaling Laws And Compute-Optimal Training Beyond Fixed Training Durations

Link: <https://arxiv.org/abs/2405.18392>

Key information:

- Evaluates a constant-learning-rate trunk followed by a short cooldown against a
  decay schedule shaped to a fixed horizon, and finds the two reach comparable
  loss.
- Because the trunk is horizon-independent, several horizons can be read by
  branching cooldowns from one run, which the authors estimate roughly halves the
  cost of fitting a scaling ladder.
- Cooldown benefit plateaus at a modest fraction of total steps, and a
  square-root-shaped decay beat a linear one.
- Its own models are small and it reports validation loss only, running no
  downstream benchmarks.

Applies to Anthro Chess:

- The direct input to the schedule-family decision, which is made before the
  ablation vehicle is frozen because it is part of the vehicle's configuration.
- What makes a data-scaling curve one run with several cooldowns rather than
  several runs.

Different from Anthro Chess:

- Loss-only evidence, so it says nothing about whether benchmark rankings agree
  between the two schedule families.
- Loss from a branched cooldown is not identical to a from-scratch run at that
  horizon, so the two cannot be mixed in one fit.

### Grandmaster-Level Chess Without Search

Link: <https://arxiv.org/abs/2402.04494>

Key information:

- Trains transformers of roughly 9M, 136M and 270M parameters on Lichess games
  annotated by an engine, predicting a move or a value without any search.
- Reports playing strength rising with both model and data scale, with the
  largest model reaching grandmaster-level Lichess blitz and the smallest still
  playing at a strong club level.
- Includes a scale ablation over model and data size rather than reporting a
  single configuration.

Applies to Anthro Chess:

- The closest available anchor for what model sizes reach what strength on
  chess, and the basis for the target-scale band in `docs/scaling.md`.
- Its smallest model already exceeds this project's competence target, which is
  why strength is not what argues for a larger model here.

Different from Anthro Chess:

- Trains toward engine-annotated best moves, so it targets strength where this
  project targets the move distribution of a human at a stated rating. Its
  strength-versus-size curve therefore bounds this project's sizing question
  without answering it.
- Its models take a single position rather than a game history, and have no
  rating or clock conditioning.

## Interfaces And Engine Protocols

### Universal Chess Interface

Link: <https://backscattering.de/chess/uci/>

Key information:

- Defines the common text protocol used by chess GUIs to launch and talk to
  chess engines over standard input and standard output.
- Covers engine identification, option advertisement, option setting, position
  setup, search/start commands, stop handling, and `bestmove` responses.
- UCI options support defaults, bounded integer `spin` values, booleans,
  enumerated strings, buttons, and free strings.

Applies to Anthro Chess:

- Default compatibility interface for local chess GUIs and engine tools.
- A way to expose target rating, temperature, optional timing settings, and
  optional preference controls to existing GUI configuration dialogs.
- A source of clock context through `go` fields such as `wtime`, `btime`,
  `winc`, and `binc`.

Different from Anthro Chess:

- UCI is an outside interface, not a model input format.
- Standard UCI expects `bestmove` and does not provide a portable engine-to-GUI
  resignation action.
- UCI settings are not a bidirectional synchronized config system; native
  Anthro interfaces may expose richer controls.

## Human-Like Chess Modeling

### Maia: Aligning Superhuman AI With Human Behavior

Link: <https://arxiv.org/abs/2006.01855>

Key information:

- Trains chess models on human games to predict human moves rather than engine
  best moves.
- Shows that human move prediction improves when modeling specific player skill
  levels.
- Introduces Maia as a human-aligned chess model based on human games.

Applies to Anthro Chess:

- Core move-modeling goal.
- Rating-conditioned human move prediction.
- Evaluation against held-out human moves.

Different from Anthro Chess:

- Maia-1 uses separate models for different rating levels, but Maia-2 and
  Maia-3 close that gap. A single rating-conditioned model is the field's
  default rather than something this project does differently.
- Maia-1 models move choice only: no clock input or output, and no terminal
  actions.
- Anthro Chess also treats deterministic board reconstruction and runtime legal
  masking as core engineering boundaries.

### Maia-2: A Unified Model For Human-AI Alignment In Chess

Link: <https://arxiv.org/abs/2409.20553>

Key information:

- Extends Maia-style human move modeling into a unified model across skill
  levels.
- Conditions **inside the trunk, in every block**. Its skill-aware attention
  projects the rating embedding to the attention inner width and adds it to the
  **query** vectors before the dot product, so the rating changes what each
  layer attends to rather than only how a finished representation is read. Both
  players' ratings are embedded and concatenated into that one signal.
- Runs a residual CNN over the board first and then treats its **channels** as
  the transformer's tokens — "channel-wise patching" — so a token is one feature
  map over all 64 squares rather than a square.
- Carries a policy head, an auxiliary head over move components, and a value
  head. None of them predicts time.
- Trains on rapid games only, and uses the clock solely to drop positions where
  either player has under thirty seconds remaining.

Applies to Anthro Chess:

- Target rating as a model control.
- Avoiding a separate model per rating band.
- Rating-sliced evaluation and calibration.

Different from Anthro Chess:

- Maia-2 focuses on skill-conditioned human move prediction.
- Anthro Chess also needs optional time-to-move output, runtime play, legal
  masking, and later preference controls.
- Maia-2 treats time pressure as noise to filter out. Anthro Chess keeps those
  plies, because they are part of what it intends to model.

### Chessformer / Maia-3

Link: <https://arxiv.org/abs/2605.19091>

Key information:

- Introduces a chess-specific transformer architecture using board-square
  tokens, each carrying a one-hot piece indicator, with the board flipped to the
  side to move.
- Adds geometric attention bias and an attention-based source-destination move
  head. The bias is generated per position from a compressed view of the board
  and added to the attention logits, one 64-by-64 map per head, so it is dynamic
  rather than a fixed positional encoding. The head scores a move as a
  source-square query against a destination-square key, and handles promotion as
  an additive bias on last-rank destinations.
- Conditions **in the input representation**: two skill embeddings, one per
  player, are prepended to each of the 64 square tokens, so the rating is
  present before the first layer. Each is an interpolation between a learned
  weak anchor and a learned strong anchor rather than a free map from rating to
  vector, which makes the embedding monotone in the rating by construction.
- Presents history by concatenating the previous seven positions into each
  token's input depth rather than along a sequence axis.
- Reaches 57.1% move-matching accuracy at 79M parameters, above Allie at 355M.
- Runs 8 layers for every human-emulation size, widening rather than deepening
  from 5M to 79M.
- Trains on Lichess blitz from 2023-01 to 2025-07, on eight A100s for about a
  week at the largest size.
- Models no timing at all, and drops every position after a player first falls
  under thirty seconds.
- Ships as an AGPL-3.0 Python package with PyTorch checkpoints and a UCI
  wrapper. Released checkpoints are 5M, 23M, 79M, and a 3M ablation; none
  predicts time.

Applies to Anthro Chess:

- Board representation.
- Move-head design.
- Human move prediction benchmarks.
- Interpretability of board-square activations.

Different from Anthro Chess:

- Chessformer is primarily a board-position encoder architecture.
- Anthro Chess currently prefers a causal trajectory model with one timestep per
  ply, exact board embeddings at each timestep, and optional timing output.
- Chessformer's board encoder and move head may still be useful even if the
  overall sequence model differs.
- Being a plain PyTorch package makes it the most practical external baseline to
  score against this project's benchmarks. Its AGPL-3.0 license bears on
  distribution, not on benchmarking.

### Allie: Human-Aligned Chess With A Bit Of Search

Link: <https://arxiv.org/abs/2410.03893>

Key information:

- Models human move choice and non-move behavior from real game logs.
- Includes pondering time and resignation behavior.
- Uses a time-adaptive search procedure at inference.
- Trains on 2022 Lichess blitz, 91M games. The centisecond export ends at
  2021-06, so its clock labels are one-second ones.
- Predicts pondering time as a squared-error regression on a scalar, optimized
  alongside move and value from the same argument:
  `-log p(m_i | m_<i) + (t(m_<i) - t_i)^2 + (v(m_<i) - v)^2`.
- Omits moves made with under thirty seconds remaining from evaluation.

Applies to Anthro Chess:

- Human move prediction.
- Timing behavior.
- Rating calibration.
- Evaluation of human-like play beyond raw move accuracy.
- A reference point for what other human-aligned chess systems include.

Different from Anthro Chess:

- Allie uses search as part of its final play procedure; Anthro Chess should
  treat that as out of scope unless the core design changes later.
- Anthro Chess should start as a direct learned policy with runtime legal
  masking and sampled timing, not a search-assisted engine.
- Anthro Chess treats resignation as another learned game action, not as a
  search-driven or engine-evaluation rule.
- Allie's move, pondering-time, and value heads all read the same shared state
  and take the same argument, so its move and time are conditionally
  independent given history. Anthro Chess instead makes timing conditional on
  the chosen action so the sampled action and sampled delay remain coherent.
- Allie's time output is a point estimate. Anthro Chess predicts a sampleable
  distribution, which is what multimodal human move times need.

### ChessMimic

Link: <https://arxiv.org/abs/2606.04473>

Key information:

- Trains small transformer models for human move, clock, and outcome prediction
  in online blitz chess.
- Conditions on position, recent move history, player rating, and clock state.
- Uses separate per-rating-band models, around 9M parameters each.
- Releases code and per-band weights.
- Reports a think-time correlation of r = 0.41 against Allie's 0.70, under
  Allie-style filters that exclude time pressure.

Applies to Anthro Chess:

- Move prediction with rating and clock context.
- Thinking-time evaluation.
- Clock-aware modeling and benchmarks.
- Released weights make it the cheapest external baseline that predicts think
  time at all, and it reports its own figures against Allie's.

Different from Anthro Chess:

- ChessMimic uses separate model instances by rating band.
- Anthro Chess aims for a unified configurable model.
- Anthro Chess does not currently need a separate outcome model as a core
  product feature.
- ChessMimic uses separate models for move, clock, and outcome prediction.
  Anthro Chess should use a shared trajectory model and condition optional
  timing on the selected action.

### UniMaia: Learning Unified Human-Aligned Chess With Textual Descriptions

Link: <https://arxiv.org/abs/2605.27767>

Key information:

- Builds a controllable human-aligned chess model from a frozen Lc0-style
  policy network.
- Uses textual descriptions and a conditioning mechanism to steer behavior.
- Reports controls for strength and opening preference, and no non-opening style
  concepts.
- Derives opening labels from game-level Lichess opening annotations rather than
  per-position matching, and notes this may partly reward memorization.
- Does not measure whether playing strength holds still while opening control
  varies.
- Reports sensitivity to prompt phrasing, where small wording changes noticeably
  alter the policy.

Applies to Anthro Chess:

- Controllable human-like chess policy.
- Strength conditioning.
- Opening and style preference controls.
- A close comparison point for steering behavior without hand-authored move
  rules.

Different from Anthro Chess:

- UniMaia uses text-conditioned control over a frozen policy network.
- Anthro Chess currently prefers a causal trajectory model trained on per-ply
  human-game sequences, with optional timing output and runtime action sampling.
- Anthro Chess preference controls are planned around activation-space steering
  from project-owned labels, with direct input conditioning only as a fallback.
- Anthro Chess labels openings per ply from position matching, and treats
  rating preservation under a preference setting as something to measure before
  a control is exposed. UniMaia does neither.

### Learning To Imitate With Less

Link: <https://arxiv.org/abs/2507.21488>

Key information:

- Introduces Maia4All for adapting population-level chess behavior models to
  individual players.
- Focuses on learning individual move tendencies from limited personal data.
- Reports that individual behavior can be modeled with far fewer games than
  earlier approaches required.

Applies to Anthro Chess:

- Possible later player-style controls.
- Efficient adaptation to specific players or player clusters.
- Style presets derived from real games rather than hand-authored profiles.

Different from Anthro Chess:

- Maia4All focuses on personalized move prediction.
- Anthro Chess would need to preserve target rating, optional timing behavior,
  legal runtime behavior, and the broader application controls while using any
  player-style signal.

### Elo-Disentangled Player-Style Embeddings

Link: <https://arxiv.org/abs/2606.25176>

Key information:

- Learns compact per-player style embeddings while trying to separate style
  from playing strength.
- Uses a rating-conditioned base move model and represents individual style as
  deviations from rating-typical play.
- Evaluates whether learned embeddings capture player identity without simply
  encoding rating.

Applies to Anthro Chess:

- Player-style imitation.
- Rating-preserving preference controls.
- Style embeddings or steering directions that should not collapse into target
  rating.

Different from Anthro Chess:

- The paper is about representation learning for player style, not a full
  playable bot.
- Anthro Chess may use activation steering, sliders, timing controls, and
  runtime action sampling rather than exposing raw player embeddings directly.

### Skill-Group N-Gram Move Models

Link: <https://arxiv.org/abs/2512.01880>

Key information:

- Treats human move prediction as skill-group-specific language modeling over
  move sequences.
- Uses lightweight n-gram models rather than neural board-state models.
- Demonstrates a simple baseline for skill-level move-pattern prediction.

Applies to Anthro Chess:

- Simple baseline for move-sequence prediction.
- Sanity check for rating-conditioned behavior.
- Possible lightweight comparison for early experiments.

Different from Anthro Chess:

- N-gram models are much less expressive than the intended neural architecture.
- They do not use exact board embeddings, rich clock context, legal masks, or
  learned board representations.

## Structured Output And Multi-Target Modeling

### Multi-Target Prediction: A Unifying View

Link: <https://arxiv.org/abs/1809.02352>

Key information:

- Surveys problems where a model predicts multiple target variables at once.
- Emphasizes that target variables may have dependencies, constraints, or
  relations.
- Frames independent per-target prediction as the simplest baseline, with
  dependency modeling as a central reason to use multi-target methods.

Applies to Anthro Chess:

- The model output can be viewed as a structured target containing action and,
  when timing is enabled, move time.
- Supports treating action and timing as dependent outputs rather than
  independent marginal samples.

Different from Anthro Chess:

- This is a broad survey, not chess-specific.
- Anthro Chess only needs a small practical structured-output design, not a
  general multi-target learning framework.

### Multi-Target Regression Via Input Space Expansion

Link: <https://arxiv.org/abs/1211.6581>

Key information:

- Studies multi-target regression where continuous target variables can have
  statistical dependencies.
- Adapts stacked single-target and regressor-chain ideas, where predictions or
  target values become inputs for related targets.
- Notes the train/prediction discrepancy that can appear when training uses
  true target values but inference uses predicted target values.

Applies to Anthro Chess:

- Move-conditioned timing is analogous to a small regressor chain:
  `action -> move_time`.
- Names the input mismatch a chained regressor carries: the second target is
  conditioned on a true value in training and a predicted one at inference.
- The known train/inference discrepancy is a reason to evaluate generated
  action-time coherence.

Different from Anthro Chess:

- The paper focuses on conventional continuous multi-target regression, not a
  mixed discrete action plus continuous time output.
- Anthro Chess should use the idea as a simple factorization, not as a full
  classical regressor-chain implementation.

### Classifier Chains: A Review And Perspectives

Link: <https://arxiv.org/abs/1912.13405>

Key information:

- Reviews classifier-chain methods for multi-label prediction.
- Chained methods preserve label dependencies by feeding earlier predicted
  labels into later classifiers.

Applies to Anthro Chess:

- Provides the discrete-output analogue for conditioning one output on another.
- Supports an action-first decoding order over a discrete first output, which
  `docs/decisions/0003-action-conditioned-timing.md` settles for this project.

Different from Anthro Chess:

- Classifier chains are primarily for multi-label classification.
- Anthro Chess uses the same dependency idea for one categorical action and one
  optional sampled timing distribution.

### Scheduled Sampling For Sequence Prediction

Link: <https://arxiv.org/abs/1506.03099>

Key information:

- Describes the train/inference mismatch in sequence prediction when training
  uses ground-truth previous tokens but inference uses model-generated tokens.
- Proposes gradually exposing the model to its own predictions during training.

Applies to Anthro Chess:

- Names the ordinary exposure-bias concern created when the time head trains on
  human actions but runs on sampled actions.
- Suggests a possible later mitigation if move-time coherence fails in
  generated games.

Different from Anthro Chess:

- The initial Anthro Chess design should use ordinary teacher forcing. Scheduled
  sampling is a possible later tool, not a planned core requirement.

### Better Conditional Density Estimation For Neural Networks

Link: <https://arxiv.org/abs/1606.02321>

Key information:

- Argues that many neural prediction tasks need a full conditional distribution
  over outputs rather than a point estimate.
- Compares conditional density estimation approaches against mixture density
  networks and categorical discretization baselines.

Applies to Anthro Chess:

- Supports predicting a sampleable move-time distribution instead of a single
  average move time.
- Reinforces evaluating timing by likelihood/calibration, not just mean error.

Different from Anthro Chess:

- This is not chess-specific and does not define the final time-output
  parameterization.

## Chess As Sequence Prediction And State Tracking

### Chess As A Testbed For Language Model State Tracking

Link: <https://arxiv.org/abs/2102.13249>

Key information:

- Studies transformer language models trained on chess move sequences.
- Finds that transformers can learn legal move prediction and state tracking
  from notation with enough data.
- Also finds that access to full game history matters for good state tracking.

Applies to Anthro Chess:

- Causal sequence modeling over chess games.
- Full-game or chunked training with causal attention.
- Legality evaluation.

Different from Anthro Chess:

- The paper investigates whether models can infer board state from move
  notation.
- Anthro Chess should compute board state exactly outside the model and provide
  an encoded state at each ply, while still using causal history for behavior.

### Tracking World States With Language Models

Link: <https://arxiv.org/abs/2508.19851>

Key information:

- Proposes model-agnostic state-based evaluation using chess.
- Evaluates whether a language model preserves structured game state by
  analyzing downstream legal move distributions.
- Argues that state-aware chess metrics can reveal failures hidden by ordinary
  string metrics.

Applies to Anthro Chess:

- Legality and mask-penalty evaluation.
- State-coherence diagnostics.
- The idea that legal move distributions are useful evaluation signals.

Different from Anthro Chess:

- The paper evaluates LLM state tracking without relying on internal model
  activations.
- Anthro Chess does not need the neural model to internally reconstruct the
  board from text; deterministic chess logic provides exact state and legal
  moves.

## Data And Labels

### Lichess Open Database

Link: <https://database.lichess.org/>

Key information:

- Lichess publishes open database exports under CC0.
- The database includes games with ratings, moves, time controls, and clock data
  when available.
- The site also includes puzzle and engine-evaluation data, but those should not
  be treated as normal human move-choice data.

Applies to Anthro Chess:

- Main candidate source for supervised human-game training.
- Rating labels.
- Clock and move-time data.
- Held-out evaluation data.

Different from Anthro Chess:

- Lichess is raw data, not a modeling approach.
- Anthro Chess still needs filtering, preprocessing, exact board reconstruction,
  training examples, and benchmark splits.

### lichess-org/chess-openings

Link: <https://github.com/lichess-org/chess-openings>

Key information:

- Provides opening metadata with ECO code, name, PGN, UCI, and EPD.
- Recommends classifying games by walking backward to a known opening position.
- Released under CC0.

Applies to Anthro Chess:

- Opening-family labels.
- Preference-control data.
- Evaluation of opening sliders and opening distribution.

Different from Anthro Chess:

- The raw dataset contains opening names and positions, not the final categories
  Anthro Chess should expose.
- Anthro Chess should map raw openings into broader project-owned categories
  and avoid labeling entire games as one opening after the opening is no longer
  relevant.

### Lichess Puzzle Database

Link: <https://database.lichess.org/#puzzles>

Key information:

- Published under CC0, derived from real games on the platform.
- Each puzzle carries a difficulty rating computed from human solve attempts,
  along with rating deviation, popularity, play count, theme tags, and the
  source game reference.
- Puzzle ratings live in their own rating pool, scored against each player's
  separate puzzle rating rather than their game rating.
- Solutions are verified sequences rather than single moves.

Applies to Anthro Chess:

- Solve rate against puzzle rating gives a response curve whose human reference
  is derivable from the puzzle ratings themselves, so no separate human baseline
  has to be collected.
- The cheapest external rating instrument available, needing only forward
  passes.
- A benchmark input immune to evaluation pool regeneration, so its scale stays
  fixed across the life of the project.

Different from Anthro Chess:

- Puzzle rating and game rating do not share an origin, and nothing in the
  published data pairs them per player, so this anchors ordering and slope
  rather than absolute calibration.
- Puzzles are selected for tactical interest, so solve rate describes puzzles
  rather than play, and it is not a strength target for a project aiming at
  human-like rather than tactically sound play.
- Puzzle positions come from platform games, so an overlap check against the
  training selection belongs with any reported result.

## Preference Steering Background

### Policy Gradient Steering

Link: <https://arxiv.org/abs/2607.27574>

Key information:

- Reports that existing contrastive activation-steering methods, CAA among
  them, fail to steer even a simple two-route gridworld policy. The
  decision-local contrast is zero there, so what those methods recover describes
  the consequences of a choice rather than the choice itself.
- Proposes fitting a removable task vector from accumulated policy gradients of
  a temporary behavioral objective, assigning credit to the actions that
  produced an outcome instead of contrasting representations reached after it.
- Fits vectors on frozen Maia-1100, Maia-1500 and Maia-1900 from Lichess puzzle
  motifs, and finds that compatible objectives compose constructively.

Applies to Anthro Chess:

- Preference-control method selection. It bears directly on the
  activation-difference method `docs/preference-controls.md` lists first, and on
  the choice of primary integration path.
- Steering a frozen human-move policy without retraining it.
- Composing several sliders at once.

Different from Anthro Chess:

- The chess objectives are tactical motifs, which push the policy toward
  stronger play. Anthro Chess wants taste at a fixed strength.
- The paper measures puzzle likelihood and states that this is tactical
  preference rather than playing strength, so it does not show whether steering
  preserves a configured rating.
- Objectives come from puzzle motif tags rather than per-ply position labels.

### Scaling Monosemanticity

Link: <https://arxiv.org/abs/2605.29358>

Key information:

- Uses sparse autoencoders to find interpretable features in model activations.
- Shows that manipulating some features can steer model behavior.
- Also emphasizes that feature coverage and faithfulness are limited.

Applies to Anthro Chess:

- Late-stage preference controls.
- Activation-space steering for openings, structures, aggression, solidity, or
  other human-play concepts.

Different from Anthro Chess:

- The work studies language-model internals, not chess models.
- Anthro Chess needs chess-specific labels, chess-specific evaluation, and
  strong safeguards that steering remains soft and rating-preserving.

### AxBench: Steering LLMs?

Link: <https://arxiv.org/abs/2501.17148>

Key information:

- Benchmarks steering and concept-detection methods.
- Finds that simple baselines can outperform sparse autoencoders in some
  steering setups.

Applies to Anthro Chess:

- Preference-control method selection.
- Reason to compare activation-difference vectors, supervised vectors, and any
  sparse-autoencoder approach.

Different from Anthro Chess:

- AxBench is not chess-specific.
- Anthro Chess should judge steering methods with chess metrics: legality,
  rating preservation, position coherence, and whether the intended preference
  actually changes.

## Evaluation And Human-Likeness

### Lichess Kaladin

Link: <https://github.com/lichess-org/kaladin>

Key information:

- Open-source Lichess project described as a machine-learning tool for
  automating cheat detection using insights data.
- Aimed at platform cheating detection rather than bot evaluation.

Applies to Anthro Chess:

- Background for classifying chess behavior from game-derived signals.
- Inspiration for a late-stage human-vs-engine classifier.

Different from Anthro Chess:

- Anthro Chess should not build a real anti-cheat system.
- Any classifier should be an internal benchmark for engine-likeness, not a
  claim about real players.

### Kenneth Regan-Style Engine Agreement

Link: <https://time.com/6227677/magnus-carlsen-hans-niemann-kenneth-regan-chess-scandal/>

Key information:

- Public reporting describes engine-agreement analysis that compares human
  moves with engine recommendations while accounting for player strength.
- The reporting also emphasizes uncertainty and false-positive risk.

Applies to Anthro Chess:

- Engine agreement.
- Centipawn-loss diagnostics.
- Rating-aware move-quality evaluation.

Different from Anthro Chess:

- Anthro Chess is not trying to detect cheating.
- Engine agreement should be a supporting metric, not the definition of
  human-likeness or rating.

### Large-Scale Analysis Of Chess Games With Chess Engines

Link: <https://arxiv.org/abs/1607.04186>

Key information:

- Uses Stockfish evaluations over large chess datasets.
- Discusses applications such as skill assessment, cheating detection, and
  studying human decision-making.
- Notes the cost of large-scale engine analysis.

Applies to Anthro Chess:

- Engine-derived evaluation diagnostics.
- Centipawn loss.
- Move-quality and rating-calibration support metrics.

Different from Anthro Chess:

- Anthro Chess is not an engine-analysis project.
- Engine evaluation should be used selectively because it can be expensive and
  because engine-best moves are not the target behavior.
