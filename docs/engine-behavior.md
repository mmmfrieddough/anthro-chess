# Engine Behavior

Anthro Chess should always produce a chess move. When timing is enabled, it
should also produce a human-like time-to-move.

The model should imitate human play patterns under configurable conditions
rather than behave like a top engine with random weakening.

## Controls And Context

The bot's behavior is shaped by both user-chosen controls and game-derived
context.

User-chosen controls include target rating, temperature, and optional style
settings. Game-derived context includes the current position, game phase, move
history, and clock information when timing is enabled.

The full model input shape belongs in `docs/architecture.md` and
`docs/training-and-runtime.md`. This document describes the behavior those
inputs are intended to produce.

## Move Selection

The model outputs a policy over a fixed move vocabulary. Runtime then:

1. computes legal moves from exact chess state;
2. masks illegal move logits;
3. samples a legal move using the configured temperature.

The sampled policy should reflect the human-game distribution represented by
the configured rating, optional time context, and style settings.

## Optional Timing Behavior

Timing should be optional. The bot should be able to play a totally untimed
game, in which case it only needs to choose a move.

When timing is enabled, human move times are multimodal. In the same position,
one player may move instantly while another spends significant time.

The time head should therefore predict a sampleable distribution, not just an
average and not a fixed bucket classification. Runtime should sample from that
distribution and submit the move after a concrete millisecond delay.

```text
Model output:    distribution over log(move_time_ms + 1)
Runtime sample:  sampled_time_ms
Move received:   received_at, the timestamp when the opponent move was received
Submit at:       received_at + sampled_time_ms
Runtime:         wait until the current time is at or after submit_at
```

Sub-second or centisecond clock labels are preferred for training realistic
fast-game behavior.

## Rating And Style

Rating should affect the kind of human play being imitated:

- move preferences;
- timing behavior when timing is enabled;
- consistency;
- risk appetite;
- familiarity with common patterns.

Temperature should remain a separate knob. A low-rating model setting with low
temperature and a high-rating model setting with high temperature should be
possible because they represent different user intents.
