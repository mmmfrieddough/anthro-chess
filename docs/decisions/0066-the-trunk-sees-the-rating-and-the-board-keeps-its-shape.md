# 0066: The Trunk Sees The Rating And The Board Keeps Its Shape

Date: 2026-08-16

## Status

Accepted. Supersedes
`0009-decision-only-rating-conditioning.md` on where rating enters the network,
and keeps every other constraint that record placed on the contract.

Lands before `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
freezes a training identity, because 0065 names a fundamental architecture
rethink after the freeze as the vehicle's main risk.

## Context

The architecture that reached a working training loop was assembled from
defaults rather than designed. That is a reasonable way to get a loop running
and a bad thing to carry into a program that intends to scale two orders of
magnitude, because the vehicle's digest covers the model and every comparison
read against it inherits whatever the model happens to be.

Three defaults were never chosen by anyone:

- **Rating entered after the trunk.** `encode_history` was *rating-neutral* by
  name and by construction, and rating arrived as a scale-and-shift on the
  finished hidden state.
- **The board was flattened.** Sixty-four squares times a piece embedding, run
  through one linear projection into a single per-ply vector. Every spatial
  relation in the position had to be relearned from a representation that had
  already destroyed it.
- **A move was an index.** The head was a projection onto a flat vocabulary of
  1,968 moves, in which `e2e4` and `e2e5` are unrelated coordinates.

`#177` measured what the first of those costs. Across a 900-Elo configured
range the fitted strength span was **12 Elo**, slope −0.003, with pairwise
ordering at 0.333 — no better than chance. Removing conditioning entirely cost
about 38 Elo, so the model was using the rating; it simply was not using it for
strength. `#496` established the leading hypothesis as architectural rather than
one of capacity, on the ground that a larger rating-neutral trunk is still
rating-neutral.

## What The Field Does, Read For Mechanism

`docs/research.md` already carried these systems. This is the second pass `#500`
asked for, which is a different question of the same papers: not what they
achieved but where the conditioning enters and what shape the network is.

**Maia-2 conditions inside every block.** Its `EloAwareAttention` projects the
rating embedding to the attention inner width and *adds it to the query vectors*
before the dot product, in each transformer layer. The rating therefore changes
what every layer attends to. Its trunk is a residual CNN whose channels become
tokens, and it carries a policy head, an auxiliary head over move components,
and a value head.

**Chessformer / Maia-3 conditions in the input representation.** It prepends two
128-wide *soft embeddings* — one per player — to each of the 64 square tokens,
so the rating is part of the representation before layer zero. The embedding for
rating `k` is an interpolation between two learned anchors,
`e_k = γ·e_weak + (1−γ)·e_strong`, rather than a free map from rating to vector.
Its body is 64 board-square tokens with a Geometric Attention Bias — a per-head
64-by-64 bias generated from a compressed view of the position and added to the
attention logits — and its policy is an attention between source-square queries
and destination-square keys, with promotion handled as an additive bias on
last-rank destinations. It reaches 57.1% move-matching at 79M parameters against
Allie's 355M.

**ChessMimic ships separate per-rating-band models.** Avoiding exactly that was
Maia-2's stated contribution, so the retreat is informative: a unified
conditioned model is not free, and a conditioning path that does not work is
worse than no conditioning at all, because it claims a dial that does not turn.

The reading that matters here is that **both unified models put the conditioning
where the representation can use it, by two different mechanisms**, and this
project was alone in putting it after. That is not a subtle difference of
opinion in the literature to be resolved by experiment. It is a place where this
project had an idiosyncratic design and a measurement saying it did not work.

## Decision

Three changes, each of which redefines what a checkpoint means, so none of them
can be read as an arm against the current baseline and all of them land before
the vehicle is designated.

### The rating is part of the representation

A learned rating embedding is added to every square token **before the first
layer runs**, so every stage of the network computes with it. It is placed by
interpolating between a learned weak anchor and a learned strong anchor, in
Chessformer's form.

The interpolation is the part worth defending, because a free map would be the
obvious choice and is what version 5 had. The dial's whole job is to be
**ordered**: a user asking for 1200 expects something weaker than 1800, and
`#177` measured an ordering no better than chance. Interpolating between two
anchors makes the representation monotone along a single learned direction by
construction. That is a property of the parameterization rather than something
the loss has to discover and can lose, and it is the one property the product
needs.

It does not follow that *strength* is monotone — a monotone representation is
necessary for an ordered dial, not sufficient. What it removes is the failure
mode where the representation itself is non-monotone and no amount of downstream
capacity can recover an ordering that was never encoded.

**Everything else 0009 decided is kept.** One encoding per game and supervision
from every valid ply; the mover's rating only; no rating on past moves; no
opponent-rating input; no controlled-color input. Only the placement is
reversed, and 0009's own Consequences section anticipated this exact revision,
naming a rating-aware reader over the causal states as the experiment to run "if
rating-control evaluation shows this is limiting." The evaluation showed it.

This is a deliberate divergence from both Maia-2 and Maia-3, which condition on
**both** players' ratings. 0009's reason for the mover's rating alone is a
runtime one rather than a modeling one: an opponent rating is not reliably
available when the engine is asked to move, and a model that requires it cannot
answer. That reasoning is untouched by anything measured here.

### The board keeps its shape

A position is read as **64 square tokens** rather than as one pooled vector, with
a **geometric attention bias** generated per position and added to the attention
logits per head. The bias generator's output layer is initialized to zero, so a
fresh model is ordinary dot-product attention and training adds the geometry
rather than first having to undo a random one.

**History stays on the ply axis.** Chessformer presents history by concatenating
the previous seven positions into each square token's input depth; this project
keeps a causal transformer over plies. That is not an oversight and it is where
the product lives: an unbounded history rather than a seven-position window, a
per-ply place for clock features and timing output, terminal actions that belong
to a trajectory rather than to a position, and the common-prefix reuse the
runtime already depends on. Taking the mechanics without taking the shape is the
whole of what `#500` asked for.

Reconciling the two gives the model three stages:

1. a **spatial encoder** over each position's 64 square tokens, reading no
   position but its own;
2. a **causal trunk** over the ply axis, on the pooled per-ply feature;
3. a **spatial decoder** that adds the trunk's history feature back onto that
   ply's square tokens.

The third stage exists because of an ordering problem rather than for capacity.
The move head has to score squares, and the squares leaving stage 1 have not
seen the history, while the vector leaving stage 2 has seen it but is no longer
addressed by square. Stage 3 is what makes a square representation that knows
the game so far, and without it the head would be choosing moves from a position
read in isolation.

### A move is a source and a destination

Move logits are an attention from source-square queries against
destination-square keys, with a per-destination promotion bias, in Chessformer's
form. Castling is already the king moving two squares in this project's
vocabulary and en passant is already a diagonal, so neither needs special
handling.

**The flat action vocabulary is unchanged as the external contract.** Legal
masking, UCI, the benchmarks, and the stored action-vocabulary identity all
speak flat action ids, and none of them changes. A precomputed gather joins the
square-by-square board to that flat list, built from the vocabulary itself so
the two cannot drift, and asserted against it by a test — a one-entry
disagreement would train the model toward the wrong move while every shape and
vocabulary check still passed.

### What scales

The shape the model grows along is `model_dim` with depth split across the three
stages. `geometric_bias_dim` is called out separately because it is the one
width whose cost does not fall with `model_dim`: each template is 4096 values
regardless, so a setting taken from a large model dominates a small one's
parameter count outright. `docs/scaling.md` owns the sizes.

## What Was Taken On Argument, And What Was Measured

The split follows `#500`: where a published comparison already answers the
question under conditions close enough to transfer, take the answer; where the
answer would be this project's own, measure it.

**Taken on argument.** Square tokens, the geometric bias, and the
source-destination head. Chessformer's 57.1% at 79M against Allie's 355M was run
with more compute than this project has, and re-deriving it costs runs and
returns the same answer.

**Measured here.** The conditioning path, because no published result covers it:
Maia-2's contribution was avoiding per-band models, ChessMimic retreated to
shipping them, and neither targets a dial that spans strength the way
`docs/vision.md` asks for. The arms and what they read are recorded in the pull
request that landed this record; `docs/evaluation.md` owns what such a reading is
and is not.

A pre-vehicle bake-off is **unqualified on seed**, because the vehicle whose
dispersion would qualify it does not exist yet — 0029 measured up to 14 of 54
floored metrics clearing on initialization alone. It is therefore an instrument
for large effects only, which is the right instrument for this question: a
conditioning path that fixes a 12-Elo span across a 900-Elo range should not need
a floor to be visible. Where the reading is close, it did not decide anything,
and the choice rests on the mechanism argument above rather than on a margin.

## What This Gives Up, Deliberately

**A position costs 64 tokens instead of one.** The spatial stages run 64 tokens
per ply where the old encoder ran a single projection, which is roughly the cost
Chessformer pays and is most of why it wins at a quarter of Allie's parameters.
The causal trunk is unchanged and still costs one token per ply, so the growth is
in the stages that were previously doing almost nothing.

**Three changes land together, so their individual contributions are not
separated.** A reading against this architecture says what the three are worth
jointly and cannot apportion it. That is accepted because the alternative is
three sequential re-freezes of the vehicle, and because two of the three are
taken on published evidence rather than on this project's own reading — the one
that is genuinely open is the one the arms address.

**No auxiliary heads.** Maia-2 and Maia-3 both carry a value head, and Maia-2
also an auxiliary head over move components. Neither is adopted here. They are
ordinary candidates: the current baseline survives adding one, so a comparison
against the vehicle means something, which is exactly the test `#500` sets for
what waits.

**A monotone representation is not a monotone dial.** The parameterization
guarantees ordering in the embedding and nothing about the strength that comes
out the far end. What it buys is the removal of one specific failure mode, not
the product guarantee.

## Consequences

**Model identity moves to version 6, and no version 5 checkpoint can be read.**
All three changes alter what a checkpoint means, so the refusal is by name rather
than by a state-dict mismatch. Every retained run from the version 5 line is
readable only by the code that wrote it.

**0009 is superseded on placement and kept on everything else.** A reader
arriving at 0009 is sent here rather than left with a reversed rule.

**`docs/architecture.md` states the three-stage shape**, and its
"Possible Rating-Aware History Reader" section retires: it described an
experiment to run if late conditioning proved too shallow, and that is now
settled rather than pending.

**The vehicle can be designated on a model somebody chose.** That was the point
of landing before `#487` rather than after it.

## References

- `0009-decision-only-rating-conditioning.md` — the placement this reverses, and
  the constraints it keeps
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md` — the
  freeze this lands before, and why re-freezing is the expensive error
- `0029-model-change-control-arm.md` — why a comparison without a seed floor is
  an instrument for large effects only
- `0038-the-encoding-owns-token-vocabularies-the-model-owns-transforms.md` — the
  boundary the square tokens respect
- `#177` — the 12-Elo span, and the ablation arm isolating conditioning at 38 Elo
- `#496` — the diagnosis, and the argument against capacity as the explanation
- `docs/research.md` (Human-Like Chess Modeling) — the mechanisms read for here
