# Related Research

This document tracks outside work that may inform Anthro Chess. It is background
context, not a roadmap and not a list of product requirements.

## Activation Features And Steering

### Scaling Monosemanticity: Extracting Interpretable Features From Claude 3 Sonnet

Link: <https://arxiv.org/abs/2605.29358>

Key information:

- Anthropic trained sparse autoencoders on internal activations from Claude 3
  Sonnet.
- The work reports interpretable features for concrete entities, abstract
  concepts, and behavioral tendencies.
- Manipulating some features can steer model behavior in ways consistent with
  their interpretations.
- The paper also emphasizes limitations: the feature set is incomplete, and
  faithfulness is hard to evaluate rigorously.

Relevance:

- Most relevant to optional soft preference controls and late-stage activation
  steering.
- Suggests a possible path for sliders that bias model behavior through learned
  internal directions rather than hardcoded move rules.
- For Anthro Chess, the analogous targets would be opening families, attacking
  posture, fianchetto structures, sacrifice tendency, solidity, or other
  human-play concepts.

Project areas:

- `docs/engine-behavior.md`: soft preference controls.
- `docs/training-and-runtime.md`: derived labels and preference-control
  metadata.
- `docs/planning/roadmap.md`: late-stage activation steering exploration.

### AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders

Link: <https://arxiv.org/abs/2501.17148>

Key information:

- Introduces a benchmark for steering and concept detection methods.
- Compares prompting, finetuning, sparse autoencoders, supervised steering
  vectors, linear probes, and representation finetuning.
- Finds that simple baselines can outperform sparse autoencoders for steering
  under the benchmark setup, while representation-based methods can still be
  useful for concept detection.

Relevance:

- Useful caution against assuming sparse autoencoders are automatically the best
  way to build preference sliders.
- Supports trying simpler activation-difference or supervised steering-vector
  methods before investing in a full sparse-autoencoder pipeline.

Project areas:

- `docs/planning/roadmap.md`: choosing late-stage steering experiments.
- Future data/evaluation docs: benchmark preference sliders against simpler
  alternatives.

### Steering LLMs? Actually, Sparse Autoencoders Can Outperform Simple Baselines

Link: <https://arxiv.org/abs/2605.31183>

Key information:

- Re-examines the AxBench result and argues sparse autoencoders can perform much
  better with a different supervised feature-selection pipeline.
- Suggests that SAE steering quality depends strongly on feature selection and
  evaluation details.

Relevance:

- Supports keeping sparse autoencoders as a plausible later option rather than
  dismissing them after baseline steering tests.
- Reinforces that preference steering should be evaluated empirically instead of
  chosen for aesthetic reasons alone.

Project areas:

- `docs/planning/roadmap.md`: late-stage activation steering.
- `docs/evaluation.md`: compare steering approaches under chess-specific
  metrics.

### When The Coffee Feature Activates On Coffins

Link: <https://arxiv.org/abs/2601.03047>

Key information:

- Stress-tests feature extraction and steering with open-source sparse
  autoencoders.
- Finds that steering can be sensitive to layer choice, steering strength, and
  context.
- Warns that thematically similar features can be difficult to distinguish.

Relevance:

- Directly relevant to concerns about opening and style sliders behaving
  strangely in some positions.
- Suggests Anthro Chess should treat any activation steering as soft,
  calibrated, and reversible.
- Points toward evaluation requirements: legality, rating preservation,
  position coherence, and whether the intended preference actually increases.

Project areas:

- `docs/engine-behavior.md`: preferences should remain soft and coherent.
- `docs/planning/roadmap.md`: calibration and evaluation of preference sliders.

## Chess Data And Opening Labels

### Lichess Open Database

Link: <https://database.lichess.org/>

Key information:

- Lichess publishes open database exports under CC0.
- The database can be downloaded, modified, and redistributed.
- The site includes game exports, puzzle data, broadcasts, variants, and engine
  evaluation data.
- Puzzle records include themes and opening tags, but puzzles are biased toward
  tactical positions and should not be treated as normal human move-choice data.

Relevance:

- Primary candidate source for human games, ratings, moves, time controls, and
  clock data.
- Useful as raw material for base supervised training.
- Also useful for deriving optional preference labels, especially when combined
  with independent opening-position classification.

Project areas:

- `docs/training-and-runtime.md`: game records, ratings, clocks, and derived
  preference labels.
- Future data docs: data ingestion, filtering, licensing, and preprocessing.

### lichess-org/chess-openings

Link: <https://github.com/lichess-org/chess-openings>

Key information:

- Provides an aggregated dataset of opening names.
- Fields include ECO code, opening name, PGN line, UCI line, and EPD for the
  opening position.
- Names are structured by opening family and variations.
- The repo recommends classifying games by walking moves backward until a named
  position is found, with extra entries for common transpositions.
- The dataset is released under CC0.

Relevance:

- Strong starting point for opening-family and structure labels.
- The backward-to-known-position method fits Anthro Chess better than relying on
  a single PGN `Opening` tag for an entire game.
- The project can map these names into its own higher-level categories, such as
  Sicilian family, French structures, fianchetto systems, open games, or gambit
  play.

Project areas:

- `docs/training-and-runtime.md`: optional multi-label preference metadata.
- `docs/planning/roadmap.md`: late-stage opening-family labels and activation
  steering.
- Future data docs: opening taxonomy and derived label generation.

## Evaluation And Human-Likeness

### Lichess Kaladin

Link: <https://github.com/lichess-org/kaladin>

Key information:

- Open-source Lichess project described as a machine-learning tool for
  automating cheat detection using insights data.
- The README says it uses CNNs with Keras/TensorFlow.
- It is aimed at platform cheating detection, not at evaluating a human-like
  chess bot.

Relevance:

- Useful background for the idea that chess behavior can be classified from
  game-derived signals.
- Not something Anthro Chess should copy wholesale as a product requirement.
- Reinforces the need to scope any human-likeness classifier as an internal
  evaluation tool, not a real anti-cheat system.

Project areas:

- `docs/evaluation.md`: human-vs-engine classifier as a late-stage evaluation
  metric.

### Kenneth Regan-Style Engine Agreement

Link: <https://time.com/6227677/magnus-carlsen-hans-niemann-kenneth-regan-chess-scandal/>

Key information:

- Public reporting describes Kenneth Regan's chess cheating-detection work as
  comparing player moves to engine recommendations and estimating how likely
  that agreement is for a player of a given strength.
- The reporting emphasizes uncertainty, thresholds, and the risk of false
  accusations.

Relevance:

- Supports using engine agreement, centipawn loss, and strength-adjusted move
  quality as supporting evaluation metrics.
- Also cautions against presenting Anthro Chess evaluation as a cheating
  detector or as proof about real players.

Project areas:

- `docs/evaluation.md`: engine agreement and human-likeness supporting metrics.

### Large-Scale Analysis Of Chess Games With Chess Engines

Link: <https://arxiv.org/abs/1607.04186>

Key information:

- Describes large-scale analysis of chess games with Stockfish.
- Discusses engine evaluations as useful for applications such as cheating
  detection, intrinsic ratings, skill assessment, and studying human
  decision-making.
- Emphasizes the cost of analyzing large numbers of positions with engines.

Relevance:

- Supports using engine-derived metrics such as centipawn loss, move quality,
  and engine agreement as evaluation diagnostics.
- Also warns that large engine-evaluation datasets can be expensive to produce,
  so Anthro Chess should use them selectively.

Project areas:

- `docs/evaluation.md`: rating calibration, engine agreement, and
  human-likeness diagnostics.
