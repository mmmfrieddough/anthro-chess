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
3. keeps the enabled terminal actions the position allows available;
4. samples a valid action using the configured temperature.

The sampled policy should reflect the human-game distribution represented by
the configured rating, optional time context, and preference settings.

Resignation should be learned from human game records. Source data rarely labels
it directly, but it is derivable: a decisive game that ended in ordinary play
without checkmate ended because someone resigned. It should not be implemented
as a hardcoded engine-evaluation rule, and endings that were not a decision,
such as clock expiry or abandonment, should not be relabelled as resignation to
enlarge the training signal.

Claiming a draw by repetition or the fifty-move rule is a separate learned
action, not a variant of resignation and not the same thing as offering a draw.
Claim availability is an exact function of board and history, so it masks like
any other action and adds no game state outside the board. It is the condition
the current position already satisfies rather than one an announced move would
create: the rules allow claiming alongside such a move, but the action carries
no move, so offering it there would offer something the model cannot mean. A
claimable position stays playable until somebody claims it. Human players in
fast games frequently decline available claims, so a model that rarely claims
under a clock is imitating the corpus correctly. The behavior matters most in untimed
play, which has no other terminator: a model that reaches a claimable dead
position needs a way to end the game that is not a hardcoded move limit.

Offering and accepting draws is deliberately out of scope. No source in scope
records offers, declined offers leave no trace at all, and a pending offer would
introduce game state that exact chess logic does not own. See
[`0017-derived-termination-and-terminal-actions.md`](decisions/0017-derived-termination-and-terminal-actions.md).

## Optional Timing Behavior

Timing should be optional. The bot should be able to play a totally untimed
game, in which case it only needs to choose a move.

When timing is enabled, human move times are multimodal. In the same position,
one player may move instantly while another spends significant time.

The time head should therefore predict a sampleable distribution, not just an
average and not a fixed bucket classification. The timing distribution should
be conditioned on the action that will be submitted, so a simple move and a
more unusual move from the same position can have different plausible delays.
Runtime should sample from that distribution and submit the move after a
concrete millisecond delay.

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

Rating and temperature stay independent as controls. Temperature is never a
mechanism for reaching a target rating, and rating never adjusts temperature
internally. Sampling at the reference temperature draws from the learned human
distribution for the configured rating; other values deliberately distort that
distribution, and the strength they produce is a measured property rather than a
guarantee. See `docs/decisions/0008-rating-temperature-independence.md`.

Optional preference settings should express taste rather than skill. Examples
may include broad opening-family preferences, aggression, solidity, or other
human-play concepts. When timing is enabled, preferences may also include
timing style, such as whether the bot tends to spend time readily or conserve
clock. These controls should be soft: they can bias the bot toward certain
kinds of play when the position supports them, but they should not force
incoherent moves or override legal move generation.

The same target rating should still apply when preference settings are changed.
A lower-rated bot with an opening preference should still behave like a
lower-rated human who favors that kind of position, while a higher-rated bot with
the same preference should remain higher-rated.

Player-style controls, if added, should also preserve rating. A famous-player
style should mean a learned tendency profile inspired by that player's games,
not a hidden way to set the bot to that player's strength.

Temperature should remain a separate knob. A low-rating model setting with low
temperature and a high-rating model setting with high temperature should be
possible because they represent different user intents.

## Randomness And Reproducibility

Temperature zero selects the highest-logit legal action and is deterministic
for a fixed checkpoint, position history, and controls. Replaying an identical
temperature-zero matchup may therefore reproduce an identical game; this is
useful for debugging but does not provide behavioral diversity.

At nonzero temperature, ordinary interactive play should use a persistent
random stream for the game and fresh randomness for each new game by default.
Synchronizing, replacing, or replaying exact board state must not accidentally
reseed that stream on every move. Users, tests, and benchmarks should also be
able to select an explicit seed when exact reproduction is wanted.

Seed, temperature, and rating are independent controls:

- seed determines which reproducible draws are made from a policy;
- temperature determines the shape of that sampling distribution;
- rating changes the learned policy being sampled.

Different seeds should create rollout diversity without being treated as
different model configurations. Temperature-zero behavior must not depend on
the seed. The runtime-state rationale is recorded in
[`0010-separate-position-sync-from-randomness.md`](decisions/0010-separate-position-sync-from-randomness.md).
