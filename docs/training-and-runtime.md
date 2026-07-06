# Training And Runtime

The training approach is supervised learning from human games.

The model should learn move choice from data. When timing is enabled, it should
also learn move timing from clock data. Exact chess logic handles state
reconstruction and legal-output filtering.

## Training Data

Useful game records should include:

- moves;
- player ratings;
- starting clock time and increment when timing is available;
- clock states or move times when timing is available;
- enough metadata to derive bot color and game phase.

Clock precision below one second is preferred for realistic fast-game behavior.
Centisecond or millisecond precision would be better if available.

Opening and preference labels are not required for the base move model, but
derived labels may be useful for optional preference controls. These labels
should be treated as training metadata rather than fixed product requirements.

## Preprocessing

For each game:

1. Parse the move list.
2. Reconstruct exact board state before each ply.
3. Generate legal moves for each position.
4. Encode the previous move.
5. Encode static game settings.
6. Encode dynamic clock features when available, plus phase features.
7. Build one training timestep per ply.

Board generation should use exact chess rules. The neural model should not need
to infer the current board from raw notation.

Optional preference labels should be allowed to be multi-label. A single ply may
belong to several useful concepts, such as an opening family, a pawn structure,
an attacking pattern, or a material-sacrifice pattern. These labels should be
attached at the ply or position-window level when possible rather than copied
blindly across an entire game.

Opening-family labels should usually apply only while the opening or related
structure is active.

Other preference labels should be designed case by case. Some concepts may be
mostly structural, such as fianchetto setups or closed centers. Others may need
event-style definitions, such as sacrifices or early piece development. The goal
is to create useful contrast sets for learning or discovering behavior controls,
not to build a perfect chess taxonomy.

## Sequence Training

Training should use full game sequences, or fixed-length chunks of game
sequences, whenever practical. The transformer should receive one timestep per
ply and use a causal attention mask so every ply prediction can be trained in a
single parallel forward pass while still preventing access to future moves.

This is compatible with exact board reconstruction because the board state for
each ply can be computed before training and included in that ply's input
embedding. It is also compatible with teacher forcing: previous moves are known
from the human game record during training, while the model predicts the current
move and optional move time at each timestep.

## Losses

The core losses are:

- move cross-entropy over the fixed move vocabulary;
- sampling cross-entropy for the sampleable move-time distribution when timing
  is enabled.

Illegal move masking is required at inference. Training may also use legal-move
awareness depending on the final implementation.

Other useful losses may include:

- auxiliary value or phase prediction;
- calibration losses for timing, when timing is enabled, or skill control.

Optional preference-control mechanisms should be learned or derived from data,
not implemented as hardcoded post-processing rules over move logits.

## Inference Loop

When it is the bot's turn:

1. Update exact game state from all moves so far.
2. Build current timestep features:
   - board before the bot move;
   - previous move;
   - current clocks and previous move times when timing is enabled;
   - static game settings.
3. Run the model with the causal KV cache.
4. Mask illegal moves.
5. Sample a legal move using temperature.
6. If timing is enabled, sample move time from the time distribution.
7. If timing is enabled, set `submit_at = received_at + sampled_time_ms`.
8. If timing is enabled, wait until the current time is at or after `submit_at`.
9. Submit the move and update local game state.

## Runtime Requirements

The runtime must:

- never submit illegal moves;
- support untimed play;
- respect clocks when timing is enabled;
- sample timing plausibly rather than always moving at fixed intervals when
  timing is enabled;
- expose configuration for target rating, temperature, optional preferences, and
  optional clock settings.

## Hardware Assumptions

The project should favor architectures that can be trained on high-end consumer
GPUs. Main experiments should ideally be practical in days, with a larger final
run possibly taking weeks.
