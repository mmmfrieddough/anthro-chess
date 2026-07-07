# Engine Behavior

Anthro Chess should always produce a game action: either a chess move or, when
enabled, resignation. When timing is enabled, it should also produce a
human-like time-to-move.

The model should imitate human play patterns under configurable conditions
rather than behave like a top engine with random weakening.

## Controls And Context

The bot's behavior is shaped by both user-chosen controls and game-derived
context.

User-chosen controls include target rating, temperature, and optional soft
preference settings. Game-derived context includes the current position, game
phase, move history, and clock information when timing is enabled.

The full model input shape belongs in `docs/architecture.md` and
`docs/training-and-runtime.md`. This document describes the behavior those
inputs are intended to produce.

## Action Selection

The model outputs a policy over a fixed action vocabulary. Runtime then:

1. computes legal moves from exact chess state;
2. masks illegal move logits;
3. keeps any enabled non-move game actions, such as resignation, available;
4. samples a valid action using the configured temperature.

The sampled policy should reflect the human-game distribution represented by
the configured rating, optional time context, and preference settings.

Resignation should be learned from human game records when resignation labels
are available. It should not be implemented as a hardcoded engine-evaluation
rule.

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

## Rating And Preferences

Rating should affect the kind of human play being imitated:

- move preferences;
- timing behavior when timing is enabled;
- consistency;
- risk appetite;
- familiarity with common patterns.

Optional preference settings should express taste rather than skill. Examples
may include broad opening-family preferences, aggression, solidity, or other
human-play concepts. These controls should be soft: they can bias the bot toward
certain kinds of play when the position supports them, but they should not force
incoherent moves or override legal move generation.

The same target rating should still apply when preference settings are changed.
A lower-rated bot with an opening preference should still behave like a
lower-rated human who favors that kind of position, while a higher-rated bot with
the same preference should remain higher-rated.

Temperature should remain a separate knob. A low-rating model setting with low
temperature and a high-rating model setting with high temperature should be
possible because they represent different user intents.
