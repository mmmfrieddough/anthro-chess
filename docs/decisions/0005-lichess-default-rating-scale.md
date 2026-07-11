# 0005: Lichess As Default Rating Scale

Date: 2026-07-11

## Status

Accepted as initial design direction.

## Context

Anthro Chess needs a target rating control. Most early training data is expected
to come from Lichess, whose ratings are plentiful and directly attached to the
games. Other sources may use different systems, such as chess.com ratings, FIDE
ratings, engine rating lists, or source-local ratings.

Rating systems are not cleanly interchangeable. Simple offsets or linear
conversions can be misleading because pools, time controls, rating algorithms,
and population strengths differ.

## Decision

Use a Lichess-like rating scale as the default project rating scale for initial
rating-conditioned training and inference.

Store source rating metadata when available, including the source rating system.
For initial training, use Lichess ratings directly as normalized ratings.

Do not force all ratings into FIDE or another global scale. Data with ratings
from other systems can either omit normalized rating for rating-conditioned
training, be used for non-rating-conditioned purposes, or later pass through an
explicit conversion process if there is enough evidence to justify it.

## Consequences

The target rating dial is practical from the start because it matches the main
training source.

Cross-source data can still be useful, but it should not silently contaminate
the rating scale. Any future conversion method should be documented and
evaluated instead of treated as a trivial offset.

External engine ratings, CCRL ratings, Stockfish `UCI_Elo`, and human rating
systems should be treated as useful benchmarks or controls, not as direct
ground truth for the Anthro rating scale.
