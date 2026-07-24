# 0010: Separate Position Synchronization From Sampling Randomness

Date: 2026-07-24

## Status

Accepted as initial design direction.

## Context

An interface such as UCI may send the complete current position and move
history before every request for a move. Anthro must accept those updates
because callers can also load arbitrary positions, take moves back, or replace
one game with another.

The first UCI implementation handles every `position` command as a complete
game-session reset. `GameSession.reset` also returns its random generator to the
configured seed. As a result, a nonzero-temperature UCI game repeatedly uses
the first draw from the same random stream instead of maintaining a stochastic
game stream. Replaying the same settings is deterministic even when the user
expected sampling variation, and a repeated position is especially likely to
select the same continuation again.

Position synchronization, expensive model lifetime, reusable inference state,
and sampling randomness have different invalidation rules. Treating them as one
reset operation is correct enough for a deterministic playable proof but is not
the intended runtime design.

## Decision

Keep checkpoint loading, device placement, and other expensive model-runner
initialization alive for the engine-process lifetime. Position updates must not
recreate those resources.

Synchronize exact chess state with the least reconstruction that remains
correct:

- append moves incrementally when the supplied position extends the canonical
  history;
- retain reusable encoded or model history for the unchanged prefix when the
  model runner supports it;
- invalidate only the affected suffix;
- fall back to atomic full validation and replacement for a new FEN, takeback,
  divergence, or otherwise unrelated position.

Incremental reuse is an optimization, not a weakening of the interface
contract. Arbitrary valid UCI positions must continue to work. A transformer
key-value cache should be added only through this boundary and only when its
correctness and measured value justify the complexity.

Sampling randomness belongs to the game or sampling lifecycle, not to position
synchronization:

- temperature zero is greedy and deterministic regardless of seed;
- at nonzero temperature, ordinary interactive play starts each game with
  fresh randomness by default;
- an explicit seed provides reproducible games and benchmark runs;
- a routine `position` update never reseeds or rewinds the active random
  stream;
- `ucinewgame` establishes the next game stream, using either fresh randomness
  or the explicit reproducible seed policy;
- external interfaces, including UCI, expose seed control without coupling it
  to temperature or target rating.

Exact option names, integer sentinels, seed ranges, and entropy sources belong
in the runtime and interface configuration schemas.

## Consequences

Identical temperature-zero games remain expected. At nonzero temperature,
normal GUI games can vary, while users and CI can deliberately select a fixed
seed for exact reproduction.

Repeated `position` commands no longer collapse stochastic play into a
position-local deterministic policy. Incremental synchronization can also
avoid rebuilding unchanged exact state and creates a correct home for future
inference caching.

Tests need injected or explicit seeds rather than probabilistic assertions.
They must cover append-only updates, replacements and takebacks, cache
invalidation, temperature-zero seed independence, fixed-seed reproduction,
fresh-game variation through an injected entropy source, and preservation of
expensive process-lifetime resources.
