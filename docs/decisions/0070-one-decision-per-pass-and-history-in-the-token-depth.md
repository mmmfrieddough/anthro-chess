# 0070: One Decision Per Pass, And History In The Token Depth

Date: 2026-08-19

## Status

Accepted. Supersedes
`0066-the-trunk-sees-the-rating-and-the-board-keeps-its-shape.md` on the ply
axis, which 0066 kept as scope control while recording that the argument for it
had not survived. Everything else 0066 decided stands: the rating is in the
input representation, the board is 64 square tokens, the move head is a
source-destination attention, and the flat action vocabulary is unchanged.

Lands before
`0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md` freezes
a training identity, for the reason 0066 landed before it.

Removes one of the two mappings
`0038-the-encoding-owns-token-vocabularies-the-model-owns-transforms.md` placed,
the previous-action token, without reopening the rule that placed it.

`0071-the-target-is-the-size-the-published-ladder-flattens-at.md` rests on this
record: adopting Chessformer's shape is what makes that project's published size
ladder a reading of this architecture rather than an analogy, and the target is
taken from where that ladder flattens.

`0073-compilation-is-on-by-default-and-plain-fusion-beats-graph-capture.md`
rests on this record too: removing the ply axis removed the ragged per-ply
iteration that had made the step impossible to compile.

`0075-a-training-batch-is-decisions-not-games.md` refines the sequence batching
this record kept as a loader convenience, and rests on the independence it
established: a batch is now decisions rather than games.

## Context

0066 left the model in three stages: a spatial encoder over each position's 64
square tokens, a causal trunk over the ply axis reading one pooled vector per
ply, and a spatial decoder folding the trunk's output back onto the squares. It
recorded that the trunk was retained for scope control rather than on its
merits, and left whether it survives open.

Three things closed it.

**The efficiency argument was already void, and 0066 is what voided it.** When a
ply was one cheap projection, a causal mask bought many supervised decisions for
roughly one forward pass. Square tokens moved the dominant cost into
per-position encoding, which every architecture pays per position. Counted over
a game of `T` plies both shapes cost `O(T x 64 x layers)` and both yield `T`
supervised decisions, so the trunk is a further cost on top of a wash.

**Trunk depth was measured and buys almost nothing.** The sweep `#500` ran
before handing this over held everything fixed but the layer counts, at two
seeds per arm: across a 24% parameter change the whole range of
`held_out.move_loss` was 0.028 and the ordering was not monotone, with one fewer
trunk layer than the shipped configuration beating it on both seeds. Spending
the same parameters on a spatial layer bought 0.068.

**The reach the axis offered was measured by somebody else, and is worth
nothing.** Chessformer ablates history depth directly: 54.0% move matching with
no history, 55.4% with seven prior positions, 55.4% with thirty-one. That is the
question 0066 could not answer and this issue would otherwise have taken on
faith. Unbounded history is not being given up against an unknown; it is being
given up against a published reading that says reach past seven does not pay.

## Decision

**One forward pass is one decision, and history is the depth of each square
token.** The causal attention across plies goes, and the spatial decoder with
it, since it existed only to fold trunk output back onto squares. The last `n`
boards are stacked into each token's input depth. Sequence batching stays as a
loader convenience, because every ply of a game is a supervised decision and
batching them costs one pass rather than one each.

`n = 8`, meaning the decision's own board and seven behind it, which is
Chessformer's setting and the one its ablation reads as sufficient. Where fewer
boards exist the earliest is repeated, and training truncates a decision's
history to a uniform random length 5% of the time, so that the short histories
every game opens with are not out of distribution. Both are theirs.

**Depth is fixed and width is the dial.** Eight layers at every size, a
feed-forward twice the model width, and an attention head dimension of 32, which
is what every published Chessformer size runs. `docs/scaling.md` owns the rule.
This is the layer budget the trunk's share was freed for, and it is taken rather
than swept: the question of how a chess board encoder should spend depth is one
a laboratory with more compute has already answered under conditions close
enough to transfer, which is `#500`'s stated test.

**The geometric bias template bank is one bank for the whole model.** Each layer
still generates its own mixture, but the 64-by-64 templates that mixture weights
are shared, which is what Chessformer does and what this project had wrong: a
bank per layer multiplies by depth, and a template is 4096 values whatever the
model width is. Fixing it is what makes a fixed depth of eight affordable at all.

**Every board is presented from the side to move**, mirroring ranks and swapping
piece colours, with the move head undoing it by reordering the 64 tokens before
they reach the vocabulary. The objection under 0066 was that the ply axis gave
alternating frames across consecutive plies; with one orientation per decision
it does not arise. Each stacked board keeps the frame of whoever was to move
when it was current, so orientation alternates down the stack rather than being
rotated into the decision's, which is again what Chessformer trains.

**The colour the deciding player is playing is put back as its own input.** The
flip is exactly the map that erases it, and human play is not symmetric under
it: a repertoire as white is not the mirror of a repertoire as black, and the
propensity to press or to settle for a draw differs by colour. Chessformer's
human model drops the bit and can, having no draw claim and no terminal actions;
its engine variant carries it, and this project already takes that variant's
rule state for the same reason. It costs one embedding row.

This does not weaken `0009`'s runtime constraint. That rule is about what a
caller must supply, and the colour is read off the board rather than asked for.

**Repetition reaches the model as a count per stacked board**, capped so that the
top state means a threefold claim is available now. A model that cannot see a
repetition cannot decide when to claim a draw, and the previous model was blind
to it through the pooling bottleneck, so this closes a standing gap rather than
one this change opened. Castling rights, the en-passant square, and the halfmove
clock stay for the same reason: the fifty-move clock is a draw-claim
precondition here, where Chessformer's human model can accept a non-Markov input
because it never claims anything.

**The explicit previous-move token goes.** Differencing consecutive stacked
boards recovers the move, including castling, en passant, and promotion.

## What This Restores

0066 had to reason about a wrinkle that no longer exists. Its rating embedding
sat on every ply's square tokens, so the trunk at ply `t` attended over earlier
plies carrying the other player's rating, and a served history had to broadcast
one rating across the whole trajectory to present a shape training contained.
With one decision per pass the model reads exactly one rating, the mover's own,
in training and at runtime alike. 0009's mover-only conditioning is exact again
rather than approximated, and the approximation 0066 named as unmeasured is
gone rather than still owed a reading.

Serving also stops paying for the game so far. A decision reads a fixed window
of boards, so per-decision cost is flat in game length where it used to grow
with it, and the key-value caching `docs/architecture.md` left open as the next
thing worth measuring has nothing left to cache.

## What This Gives Up

**Unbounded history.** The grounds are the ablation above rather than an
argument, which is a stronger footing than this change was expected to have.
What is genuinely untested is whether a task this project cares about and
Chessformer does not, such as recognizing a repetition being steered toward
across more than seven plies, wants more reach. The repetition count is what
carries that signal instead, and it is exact.

**The two changes that are this project's own are unmeasured against
alternatives.** The repetition count and the colour bit are additions no
published human-emulation model carries, because no published human-emulation
model claims draws. Neither is read against an arm without it, and neither will
be: they are inputs a rule requires rather than candidates.

**A reading against the `#500` architecture is joint.** Six changes land
together and the layer budget moves with them, so the comparison says what the
architecture is worth as a whole and cannot apportion it. That is accepted for
0066's reason, and with the same qualification: the parts taken on published
evidence were never the parts in doubt.

## Consequences

**Model identity moves to version 7, and no version 6 checkpoint can be read.**
Every change here alters what a checkpoint means. The encoding moves to version
5 for the repetition count, which no stored corpus carries: encodings are built
at load time from normalized games, so nothing is rebuilt and the bump only
refuses checkpoints.

**The model's module and class are renamed.** `anthro_chess.models.causal` and
`CausalMoveModel` described a property the model no longer has.

**The declared context length goes, along with everything that guarded it.** A
model with no ply axis has no context to exceed, so the configuration field, the
position table, and the corpus check that refused a run whose longest game
reached past it are all removed.

**`docs/architecture.md` states the one-stage shape**, `docs/scaling.md` owns
the fixed depth and the shared template bank, and `docs/design-principles.md`
asks for bounded per-decision context rather than a compact per-ply one.

**A pool stops being gated on the encoding.** The evaluation pool's manifest
records which encoding it was cut under and `load_pool` refused any pool whose
record disagreed, so this change made the frozen canonical pool unreadable and
every benchmark with it. A pool holds normalized games, which the current code
encodes as it reads them, exactly as it does a corpus; a corpus manifest carries
no encoding identity at all. The gate goes and the record stays, which is what
keeps `0068`'s re-cut the only thing that re-baselines benchmark history.

**Chunked selections now lose seven plies of history at a boundary rather than
everything before it.** A chunk does not overlap its predecessor, so its first
decisions read a repeated board where the game has a real one. That is strictly
better than the trunk, which saw nothing before the chunk at all, and it is
still a cost: `chunk_length` says so where a caller turning it on will read it.
No checked-in configuration chunks.

## References

- `0066-the-trunk-sees-the-rating-and-the-board-keeps-its-shape.md` for what
  `#500` decided, and where it records the ply axis as unjustified
- `0009-decision-only-rating-conditioning.md` for the runtime constraint this
  restores exactly
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md` for
  the freeze this lands before
- `0038-the-encoding-owns-token-vocabularies-the-model-owns-transforms.md` for
  why the repetition count is the encoding's and the flip is the model's
- `docs/research.md` (Human-Like Chess Modeling) for the Chessformer readings
  this takes, including the history-depth ablation
