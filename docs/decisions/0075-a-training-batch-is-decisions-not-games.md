# 0075: A Training Batch Is Decisions, Not Games

## Status

Accepted.

Refines `0070-one-decision-per-pass-and-history-in-the-token-depth.md`, which
kept sequence batching "as a loader convenience, because every ply of a game is
a supervised decision and batching them costs one pass rather than one each."
That reasoning holds and is why decisions are still batched at all. What it does
not establish is that a batch has to be game-shaped, and this record separates
the two.

Rests on that record for the property that makes packing possible at all:
decisions are independent, so which game a decision came from stops deciding
which batch it can sit in.

Lands before
`0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md` freezes
a training identity, for the reason `0070` and
`0074-the-vehicle-freezes-on-bfloat16-with-tf32-without-a-quality-reading.md`
landed before it: packing changes which decisions share a batch, so it is a
change to what a model learns rather than an execution setting, and after the
freeze it would have to be an arm rather than an adoption.

## Context

A game-shaped batch pads its rows to the longest game in them. That padding is
not skipped: a padded timestep is encoded as 64 square tokens, run through every
layer and the move head, and then dropped by the loss. Length bucketing holds it
down without removing it, and `training.active_position_fraction` read 0.933 on
the configuration measured here, so about one timestep in fifteen was compute
spent on nothing.

Padding is not the only cost a game-shaped batch carries. Its width is the
longest game in the batch, so the width moves from batch to batch, and a
compiled step guards on it.

## Decision

**A training batch is a fixed number of decisions, and games are laid end to end
to fill it.** The loader cuts that stream at the batch boundary, so a game
longer than what is left of a batch is split and its remainder opens the next
one. `positions_per_batch` selects that shape and is the batch's only size dial;
`batch_size` and `length_bucket_width` belong to the game-shaped shape and
cannot be set beside it.

**A decision's stacked history stops at the first column of its own game.** That
column is the difference of the two indices a batch already carries, so a row
holding several games records no game boundary separately, and a decision never
reads the game laid down in front of it.

**Evaluation keeps game-shaped rows.** A cut costs the few decisions after it
their full history, which is a rounding error against a training run and a
reading that moves with the batch size against a benchmark. Density is worth
having where the compute is spent and not worth having where a number is
recorded.

## What Was Measured

One idle RTX 4090, the widened corpus, the 256-wide eight-layer model, 1500
steps at four accumulation steps, compilation and `bfloat16` at their shipped
defaults, and strict determinism. The padded arm's mean micro-batch was 284.2
timesteps wide, so the packed arm was set to 287: both give the device about the
same work per micro-batch, with the treatment holding the 1% that would round in
its favour.

| Arm | Active positions/s | Step seconds | Active fraction | Graphs | Peak allocated |
| --- | --- | --- | --- | --- | --- |
| game-shaped, batch 4, bucket width 16 | 8595.70 | 0.123427 | 0.933346 | 6 | 4.03 GiB |
| decision-shaped, 287 positions | 11333.95 | 0.101266 | 0.999781 | 1 | 1.35 GiB |

**Packing is worth 1.32x the training throughput**, and less than a third of
that is the padding it removes. Turning padded timesteps into decisions accounts
for 1.07x, which is what the active fraction alone predicts. The other 1.23x is
that the batch stopped changing width: the padded arm's rate over its *padded*
extent is 9210 positions per second against the packed arm's 11336, for the same
nominal work per micro-batch.

The width was carrying more than a recompile count. A padded batch is as wide as
the longest game in it, so its peak memory and its slowest step belong to its
tail rather than to its mean: the padded arm reserved 11.53 GiB against 1.47 GiB,
and its slowest measured interval was 4.5x its fastest where the packed arm's was
1.02x. A run on that shape has to be sized for a batch it takes rarely.

The packed arm's active fraction is 0.9998 rather than 1.0 because a planning
window ends with one short batch, which is padded out to the same width so that
a compiled step does not recompile for it.

## What This Does Not Claim

**Nothing about quality.** A packed batch and a padded one draw the same number
of decisions from about the same number of games, because a game contributes the
same plies either way, so the decorrelation a packed batch was expected to buy
is not there to buy. What does change is that length bucketing put games of
similar length in a batch together and packing does not, so a packed batch's mix
of game lengths is representative where a bucketed one was homogeneous.

That effect was measured rather than argued. At a fixed model and about 285
decisions a batch, the dispersion of the gradient across 96 independently drawn
batches was lower packed at both points it was taken: 2.4x lower at
initialization, and 1.27x lower at a 1500-step checkpoint. The direction is
packing's and the magnitude shrinks as the model trains, which is what a reading
at two points can say and no more. It is not a quality claim, and neither point
is floored.

A quality claim is not available here at all. This project has no seed floor for
any configuration; `#488` builds one and is blocked behind the freeze this
record precedes. A packed-versus-padded pair at one seed would be unqualified on
seed whichever way it fell, so the throughput and active-fraction readings decide
this on their own terms and any quality question is left to be reopened as an arm
against the vehicle once there is a denominator for it.

## Consequences

`length_bucket_width` still exists and is what evaluation batches by. Training
does not use it. `chunk_length` existed to bound the positions one batch holds,
which is what `positions_per_batch` now does directly, so no packed configuration
sets either.

A run's efficiency record carries the unit its batch is counted in, because a
batch of four games and a batch of four decisions are not the same coordinate
and nothing else in that record tells them apart. Warmup is declared in the same
unit for the same reason: a game-shaped batch fixes only how many games it holds,
and a packed one only how many decisions.

A packed batch's width is constant, so a compiled step guards on one shape
instead of recompiling as the longest game in a batch moves.

A cut game costs the shard-backed loader a second decode. Its two halves land in
consecutive batches, each batch is one job, and a job decodes the games it needs
from scratch, so the straddling game is replayed twice. That is
`mean_game_length / positions_per_batch` extra decoding, measured at 22% of the
loader's decode work at the width above and rising as the width falls. The
worker pool absorbed it at the configuration measured here, which is why the
throughput reading is what it is, but the term belongs to whoever picks the
vehicle's width: it is the one cost of this shape that a narrower batch makes
worse rather than better.
