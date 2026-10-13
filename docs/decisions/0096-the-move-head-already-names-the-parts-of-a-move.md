# 0096: The Move Head Already Names The Parts Of A Move

Date: 2026-10-13

## Status

Accepted. Answers `#503`.

## Context

Maia-2 carries an auxiliary head beside its policy, trained to name the parts
of the played move: the piece type moved and captured, a gives-check flag, the
source and destination squares, and a multi-hot over every legal move. `0066`
left it out as an ordinary candidate, to be read against the vehicle like any
other addition that leaves the baseline intact.

`#503` expected little from it here. The source-destination head already scores
a move as a pair of squares, so most of what the auxiliary objective supervises
is structural in this architecture rather than something the trunk has to be
pushed into learning.

## Decision

**No move-component head is adopted, and no setting for one is kept.** The
auxiliary objective made held-out move loss, top-1 accuracy, and legality worse,
each far outside seed variance, and the model's own policy already carried
everything the objective asked for.

## What Was Read

One arm: the vehicle at seed 17 on the vehicle's loader seed, trained across
both cards, with one change. A linear layer reads the mean-pooled final square
tokens and names four parts of the played move, each scored by its own
cross-entropy:

- the piece type moved;
- the piece type captured, or none;
- the source square, in the mover's frame;
- the destination square, in the mover's frame.

The mean of the four is added to the move loss at weight 1.0, Maia-2's
coefficient. The coefficient matches but the strength does not. Maia-2 averages
one binary cross-entropy over a vector of roughly two thousand entries, almost
all of them the legal-move multi-hot and almost all zero, so each part it
supervises carries a small share of the gradient. Four categorical
cross-entropies averaged together are about as large as the move loss itself,
so this arm applied a much stronger dose than Maia-2 does. Terminal actions name
no parts. The head held 18,060 parameters,
and nothing served them. Both arms trained at 0.003, which is what `anthro
scale` gives at the vehicle's width and horizon, since the head moves neither.

Maia-2's gives-check flag and legal-move multi-hot were left out. The multi-hot
is a property of the position rather than of the move, and a training batch
carries no legal actions. The check flag needs the move applied to the board,
which nothing at the batch boundary does.

The claim, posted on `#503` before the arm ran, was a null on
`held_out.move_loss`, with top-1 accuracy and legality holding.

| metric | replicates | `arm-53-two-card-s17` | arm | sd |
| --- | ---: | ---: | ---: | ---: |
| `held_out.move_loss` | 1.42915 | 1.42839 | 1.44112 | +13.7 |
| `held_out.top1_accuracy` | | 0.53782 | 0.53491 | -15 |
| `legality.legal_mass` | | 0.99948 | 0.99914 | -11 |

The replicates column is the mean of the six `#488` arms. Deviations are in
the seed standard deviation stored for the vehicle, which its five distinct
seeds carry, read against that mean for move loss and against
`arm-53-two-card-s17` otherwise. The arm reads worse than the worst replicate
by 0.0107. Its training move loss sat 0.009 to 0.024 above the control's at
every logged step of the run.

**The seed floor is withheld, and the reading does not need it.** `anthro eval
report` withholds the floor because `training_health.gradient_norm` reads 0.220
against the replicates' 0.136. That is mechanical rather than unstable: the norm
is taken over the whole objective, so an auxiliary term raises it by itself, and
the clip rate stayed at zero. The move-loss delta is six times the stored seed
bound, far enough outside seed variance that nothing else explains it, which
`docs/evaluation.md` (Regression Comparisons) accepts in place of the floor.

**The source-destination head had already made the signal redundant.** On
133,623 held-out move decisions, each checkpoint's legal policy was summed into
the four parts and scored against the played move:

| part | vehicle, implied by its policy | arm, implied by its policy | arm, its own head |
| --- | ---: | ---: | ---: |
| piece moved | 0.6980 | 0.7020 | 0.7050 |
| piece captured | 0.2032 | 0.2049 | 0.2068 |
| source square | 0.9603 | 0.9661 | 0.9831 |
| destination square | 1.2951 | 1.3065 | 1.3245 |

The vehicle's move distribution already names each part better than a head
trained on nothing else. The piece-type groups, the ones `#503` named as
genuinely additional, are no exception. The objective found no gap to fill.
The likeliest reading of the loss it cost is that its gradient competed with the
move loss for the trunk, though nothing here measures that directly.

Training throughput fell from 103.8k to 100.9k positions a second.

## What This Does Not Claim

**A smaller weight is not read, including one at Maia-2's effective
strength.** Nothing above suggests one would help. At weight 1.0 the head found
nothing the policy did not already carry, and a smaller weight only shrinks a
term with no gap to fill.

**The check flag and the legal-move multi-hot are not read.** The first is a
function of the move and the board that the policy is already scored on, so the
same argument applies. The second is legality supervision, which
`docs/training-and-runtime.md` treats as its own candidate.

**This is a vehicle reading.** Whether a wider trunk has spare capacity that
an auxiliary objective would organize differently is not measured.

## Consequences

The arm `arm-503-move-components` is retained and loads only with the head
restored.

The checkpoint suite was stopped after inference efficiency and held-out
prediction. The adjudicated rows that step also records are mixed, and every
remaining benchmark reads generated play or conditioning built on the policy
this reading shows got worse.

## References

- `#503`, `#488`, `#53`, `#502`
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0066-the-trunk-sees-the-rating-and-the-board-keeps-its-shape.md`
- `docs/research.md` (Human-Like Chess Modeling)
- `docs/evaluation.md` (Regression Comparisons)
