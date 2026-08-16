# 0009: Condition Decisions Without Rating The History

Date: 2026-07-22

## Status

Accepted as initial design direction. Superseded on placement by
`0066-the-trunk-sees-the-rating-and-the-board-keeps-its-shape.md`: the rating is
now embedded into the input representation, so the transformer is no longer
rating-neutral. The experiment this record's Consequences section named as the
one to run if evaluation showed late conditioning limiting is what `#177`
triggered and `0066` settles. Every other constraint below still holds — one
encoding per game, the mover's rating only, no rating on past moves, and no
opponent-rating input.

## Context

Anthro needs one runtime target rating for its own move choice. An opponent
rating is not reliably available at runtime, and attaching Anthro's rating to
every historical move gives observed opponent moves the wrong meaning.

Splitting each training game into white-controlled and black-controlled views
avoids exposing the opponent rating, but duplicates the transformer work and
uses only half of the moves for loss in each view. Broadcasting one player's
rating across either view also labels that player's earlier actions with the
same rating redundantly.

## Decision

Encode each game once and train on every valid ply. For each supervised move,
select only the side-to-move player's normalized rating when available. Keep
ratings outside the board, move-history, and causal-transformer inputs.

After the causal transformer produces a rating-neutral position and history
feature, combine that feature with the current decision-maker's optional rating
through a small nonlinear feature-modulation layer before the action head. At
runtime, apply Anthro's single configured target rating only to Anthro's current
decision. Do not require a controlled-color model input or an opponent rating;
exact board state already identifies the side to move.

## Consequences

Training keeps one sequence representation per game and receives supervision
from both sides' moves in one pass. Historical states can be cached and reused
across rating choices, and an unknown opponent rating cannot leak into move
prediction.

The causal transformer analyzes history before it knows the requested rating.
The decision conditioner still performs learned nonlinear computation, but it
cannot change which historical details the transformer itself emphasizes. If
rating-control evaluation shows this is limiting, test a small
rating-conditioned query or cross-attention reader over the rating-neutral
causal states. That experiment must preserve the single backbone pass and must
not reintroduce ratings on past moves or require an opponent rating.

Model, encoding, and loader-state identities must change because checkpoints
and saved cursors from the earlier paired-view contract are incompatible.
