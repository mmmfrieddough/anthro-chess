# Related Research

This document tracks outside work that may inform Anthro Chess. It is a curated
reference list, not a roadmap and not a list of product requirements.

Each entry notes what part of Anthro Chess it applies to and how it differs
from this project.

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

- Maia uses separate models for different rating levels.
- Anthro Chess aims for one configurable bot with runtime target rating,
  optional timing, temperature, and optional preference controls.
- Anthro Chess also treats deterministic board reconstruction and runtime legal
  masking as core engineering boundaries.

### Maia-2: A Unified Model For Human-AI Alignment In Chess

Link: <https://arxiv.org/abs/2409.20553>

Key information:

- Extends Maia-style human move modeling into a unified model across skill
  levels.
- Uses skill-aware conditioning to capture how human move choice changes with
  player strength.

Applies to Anthro Chess:

- Target rating as a model control.
- Avoiding a separate model per rating band.
- Rating-sliced evaluation and calibration.

Different from Anthro Chess:

- Maia-2 focuses on skill-conditioned human move prediction.
- Anthro Chess also needs optional time-to-move output, runtime play, legal
  masking, and later preference controls.

### Chessformer / Maia-3

Link: <https://arxiv.org/abs/2605.19091>

Key information:

- Introduces a chess-specific transformer architecture using board-square
  tokens.
- Adds geometric attention bias and an attention-based source-destination move
  head.
- Reports strong human move prediction results with Maia-3.

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

### Allie: Human-Aligned Chess With A Bit Of Search

Link: <https://arxiv.org/abs/2410.03893>

Key information:

- Models human move choice and non-move behavior from real game logs.
- Includes pondering time and resignation behavior.
- Uses a time-adaptive search procedure at inference.

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

### ChessMimic

Link: <https://arxiv.org/abs/2606.04473>

Key information:

- Trains small transformer models for human move, clock, and outcome prediction
  in online blitz chess.
- Conditions on position, recent move history, player rating, and clock state.
- Uses separate per-rating-band models.

Applies to Anthro Chess:

- Move prediction with rating and clock context.
- Thinking-time evaluation.
- Clock-aware modeling and benchmarks.

Different from Anthro Chess:

- ChessMimic uses separate model instances by rating band.
- Anthro Chess aims for a unified configurable model.
- Anthro Chess does not currently need a separate outcome model as a core
  product feature.

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

## Preference Steering Background

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
