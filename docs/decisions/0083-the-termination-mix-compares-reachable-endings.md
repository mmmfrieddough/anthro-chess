# 0083: The Termination Mix Compares Reachable Endings

Date: 2026-08-29

## Status

Accepted. Refines `0017-derived-termination-and-terminal-actions.md`, which
settled the shared ending vocabulary and the rule that a category neither side
can produce stays visible rather than being folded into a neighbour. That rule
holds. What this record changes is which side counts those categories.

## Context

The termination mix compared the generated and human ending distributions over
the union of both vocabularies, as a total variation distance. Four human
categories have no model counterpart: the harness runs no clock, so no game can
expire; a seat cannot walk away; there is no channel to agree a draw; and no
generated ending is unknown. On the frozen blitz pool they carry 0.333 of human
mass, clock expiry alone 0.296.

Total variation is the mass that has to move for two distributions to agree, so
that 0.333 is an exact lower bound the distance cannot go below. Measured on a
555,727-step checkpoint over 480 games, the pooled distance read 0.33407 against
a bound of 0.33289. The movable part was 0.00118, and the reading's own
dispersion bound was 0.00141: the entire content of the metric sat inside its
own noise.

The offset was not the problem, and this is the part that took a measurement to
see. A constant cancels in a delta between two checkpoints. What the offset did
was put the model in the flat region of the distance. Moving mass from one
category to another changes a total variation distance by
`0.5 * mass * (sign(model - human at the destination) - sign(model - human at
the source))`, which is exactly zero when the model is above the human rate on
both. The 0.333 it cannot place on the unreachable categories lands on the ones
it does produce and pushes it above human on every one of them, so redistribution
among them is free. That checkpoint had 0.185 of headroom on checkmate and 0.084
on resignation, and inside that box the number does not move at all.

Four independent 480-game draws of one checkpoint read 0.33407, 0.33371,
0.33371, 0.33371 while their checkmate share ranged from 0.433 to 0.492. The
distance was constant to four decimals across a behaviour change it exists to
report. Across two checkpoints 4x apart in training it moved 0.0009 against a
claim bar of 0.0051, and moved the wrong way.

Two fixes were measured and rejected. A divergence with no flat region removes
the exact zero and not the problem: Jensen-Shannon over the same union moved
0.003 on a base of 0.203 for a 0.05 mass shift, and still ordered the two
checkpoints backwards, because the clock term dominates it and the reachable
information is a ripple on top. The same divergence over the renormalized side
ordered them backwards too, weighting a 5% stalemate share by its ratio to a
1.4% human one.

## Decision

The human side leaves the four unreachable categories out of the termination
quantity and is renormalized over what remains by doing so. A human game that
ended on the clock still contributes its length, its result, its repertoire, and
every other compared quantity; it contributes no observation of this one. The
distance stays total variation, which is the functional that answers "how much
mass is in the wrong place".

The model-only ply limit stays in the model's own vocabulary with no human
counterpart, so a suite that stops its games is charged for it here as well as
in the unfinished rate. That is 0017's rule applied in the direction it still
holds: the gap is real, a checkpoint can close it, and hiding it would move mass
onto a category the checkpoint did not produce.

The excluded set is declared rather than derived from what a checkpoint happened
to produce. Deriving it would drop stalemate from the vocabulary for a model
that never stalemates, which is the reading rather than the vocabulary.

The mix is a generated-play compared quantity rather than a family of its own.
It calls the same comparison, over the same games, against the same reference,
and the only reason it was kept apart was that a human game ends in a richer
vocabulary than the model can produce. That reason is what this record removes.

## Consequences

On the same data the reading goes from 0.334 to 0.102, orders two checkpoints
correctly rather than backwards, and responds to a mass shift the union distance
was flat in: moving 0.05 from checkmate to resignation takes it to 0.052, with a
minimum near 0.10, because humans resign 55% and checkmate 40% of the endings a
clockless model could produce while that checkpoint did 45/45.

Every termination-mix series ends, which any fix to the distance would have done.

The reading is noisier than the union distance was, and that is the fix working
rather than a cost of it: sd 0.028 across 480-game draws, where the union
distance's 0.00018 was blindness rather than precision. It needs the game counts
the rollout already plays, which is what moving it there buys.

When the project gains a clock, clock expiry becomes reachable and moves out of
the excluded set, which shrinks to 0.037. The series ends then too, because a
clock joins the declared workload. What survives the transition is the question:
"given endings you could produce, how far off is your mix" means the same thing
on both sides of it, where the union distance would go from floored at 0.333 to
floored at 0.037 under one metric name.
