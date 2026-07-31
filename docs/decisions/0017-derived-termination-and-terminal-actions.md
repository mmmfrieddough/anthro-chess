# 0017: Derived Termination And Terminal Actions

Date: 2026-07-27

## Status

Accepted.

## Context

The action vocabulary has carried a resignation slot since the standard action
vocabulary was defined, and the runtime samples it, applies it, and ends the
game. Nothing upstream ever produces it. The data pipeline stores the source
`Termination` header as opaque text and writes only moves into the action
sequence, so no training example has ever contained the resignation action. The
runtime disables resignation by default, which has hidden the gap: the head has
a slot it was never taught to use.

Reviewing that gap raised the adjacent question of whether the project should
also model draws, which turned out to be three separate questions with three
different answers: offering a draw, accepting one, and claiming one under the
repetition or fifty-move rules. Only the last is a unilateral action against a
condition exact chess logic can compute.

### What the sources actually record

Lichess distinguishes game endings internally, but its PGN export collapses
them. Its dump generator maps resignation, agreed draw, stalemate, checkmate,
and variant end to a single `Normal` termination, and maps both clock expiry and
player abandonment to a single time-forfeit value. No source in scope records
draw offers at all, in the bulk dumps or in the per-game API.

Resignation survives that collapse anyway, because for standard chess only two
statuses can produce a decisive result with a `Normal` termination: resignation
and checkmate. Replaying the game and testing the final position for checkmate
separates them exactly. This is a derivation, not a heuristic, and it holds for
any PGN source that reports a result and distinguishes clock expiry, not only
for Lichess.

Draw claims survive for a similar reason. Automatic draws leave a final position
that exact chess logic recognizes as terminal on its own, while a claimed draw
leaves a final position where a claim was merely available. The two are
distinguishable by replay.

Draw offers do not survive, and cannot be recovered. An offer is a mid-game
event rather than a property of the final position, and a declined offer leaves
no trace whatsoever. The supervised pair that would be needed to learn offering
or accepting behavior does not exist in the corpus at any level of effort.

### What the corpus looks like

Measured on real Lichess games in July 2026, before any of this work landed.
Two samples: 600 recent games from two titled players across mixed speeds, and
819 arena games spanning under-1400 through 2200+ in bullet and blitz.

Resignation is the largest non-mate termination category. It was 55.7% of the
titled-player sample and 20-24% of arena games, and the arena figure was flat
across all four rating bands rather than concentrated at the top. There is no
shortage of signal.

Draw claims are real but rare: 2.7% of the titled-player sample and at most 1.2%
of any arena band. Draws by agreement were rarer still, between 0.3% and 0.7%,
which is the empirical reason offers are not worth pursuing even setting aside
the missing labels.

Between 7.8% and 10.6% of resignations happened when the resigning player was
not the side to move. Lichess allows resigning on the opponent's clock, so those
players made a move and then quit before the reply.

Time forfeits do not resemble resignations. Measuring material balance from the
losing player's point of view at the final position, resignations sat at a
median of six pawns down, with 73.0% down at least three and only 5.8% ahead.
Time forfeits sat at a median of one pawn down, with 38.7% of the losing players
actually ahead on material. Clock traces confirm these are genuine flag falls
rather than abandonment: 79.8% of forfeiting players had under 2% of their
initial time left at their last move, and only 2.4% had more than 30% left.

## Decision

**Derive a termination category during preprocessing rather than storing source
text alone.** Replay each game to its final position and classify it using the
result, the source termination field, and exact chess logic. Keep the raw source
value alongside the derived category for provenance. The derivation is defined
over what a PGN reports rather than over Lichess status values, so it stays
source-agnostic in the sense `0004-source-agnostic-normalized-data.md` requires.

**Treat resignation as a learned terminal action with a derived label.** Append
the resignation action to the action sequence when the derived category is
resignation and the resigning player was the side to move. Record why the
terminal action was omitted when it was, so the drop is auditable rather than
silent.

**Do not fold abandonment or time forfeit into resignation.** They are excluded
even where clock traces make abandonment identifiable, for three reasons. The
error directions are not symmetric: under-labelling makes the bot resign
slightly less than humans do, which is harmless, while over-labelling teaches it
to resign in positions it is winning, which is the worst failure mode a
human-like opponent has. The bucket does not carry the intended meaning, since
nearly two in five forfeiting players were ahead on material. And resignation is
already abundant, so there is no case for accepting contamination to grow it.

Gating abandonment on the player being materially behind is specifically
rejected. Using "was losing" to decide that an ending counts as a resignation
builds the rule "resign when losing" into the labels by construction, instead of
letting the human threshold be learned, and it would make the premature-
resignation benchmark report the health of an assumption rather than of the
model. That is the same objection `docs/engine-behavior.md` raises against
hardcoded evaluation rules.

**Add a draw-claim action to the vocabulary.** Claim availability is an exact
function of board and history, so it masks like any other action and introduces
no game state outside the board. Claims are rare enough that imitation fidelity
is a weak justification on its own; the load-bearing reason is that untimed play
has no other terminator. In a timed game, declining to claim is human-like and
the clock resolves the position. In an untimed game, which the project supports,
a model that shuffles into a claimable dead position has no way to end the game.

**Keep draw offers and acceptances out of scope.** This is a deliberate
omission, not an oversight. They are absent from the data, absent from UCI, and
would require a pending-offer state that is not derivable from the board,
weakening the property that exact chess logic owns game state. Revisit only if a
protocol surface that carries offers becomes a target, and then prefer a
documented host-layer policy over a model action.

**Land terminal-action vocabulary changes in one version bump.** The action
vocabulary identity is stamped into both data and model artifacts, so each
change to it invalidates existing artifacts and forces a corpus regeneration.
Resignation labelling and the claim action are therefore one vocabulary change
rather than two, so that regeneration is paid once. This is a compute argument;
the comparability break it also causes is free until the core is designated,
per `0013-benchmark-result-comparability.md`.

## Consequences

The preprocessing version advances and the baseline corpus is regenerated. The
action vocabulary identity changes, which invalidates existing checkpoints and
starts a new comparability series under
`0013-benchmark-result-comparability.md`. The regeneration cost is the reason
the two vocabulary changes are batched. The series break is not a reason to
defer either of them: no protected history exists before the core is
designated, and the proof-scale checkpoints this invalidates are not worth
keeping.

Resignation becomes a measurable behavior rather than an unreachable slot, and
it needs benchmarks before it is enabled by default. Premature resignation is a
product-critical failure that aggregate move prediction cannot surface, so
generated play is compared against the human termination mix rather than against
a target rate.

Abandonment keeps its own derived category rather than being merged into time
forfeit. The model cannot produce it, so a reference distribution that hides it
inside another bucket would show a permanent gap no checkpoint could close.

Draw claims will be heavily outnumbered in training. A model that rarely claims
in timed play is behaving like the humans in the corpus; the untimed
non-termination rate is the reading that matters.

## Reopening

Reopen the abandonment decision if a source that distinguishes abandonment from
clock expiry becomes a bulk training source, and if the abandonment population
measured on that source resembles resignations rather than flag falls.

Reopen the draw-offer decision if a target protocol surface carries offers, or
if a source that records offer events, including declined ones, becomes
available.

The corpus measurements behind this record were taken on bullet and blitz, where
flagging is common and abandonment is masked by the clock expiring first. Rerun
the termination mix and the forfeit clock profile on rapid and classical games
before extending these conclusions to slower time controls.
