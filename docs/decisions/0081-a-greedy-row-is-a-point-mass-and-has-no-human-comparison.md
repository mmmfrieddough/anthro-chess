# 0081: A Greedy Row Is A Point Mass And Has No Human Comparison

Date: 2026-08-27

## Status

Accepted. Withdraws the temperature-zero curve reading whose floor
`0032-a-replayed-reading-has-no-evaluation-noise.md` states and
`0060-a-curve-resamples-the-stream-not-the-game.md` leaves standing. Neither is
wrong about that floor. This record says the reading it qualifies should not be
taken.

## Context

The rollout's human-reference comparison estimates a curve on each side and
reports the distance between them. Both earlier records ask what floor the
temperature-zero row of that comparison carries, and answer that it states a
zero rather than bootstrapping one, because greedy seats replay their games and
a fresh seed draws nothing.

That answer is correct and this record does not disturb it. What neither asked is
whether the distance it qualifies means anything.

It does not, for a reason that no sample size reaches. Greedy selection is an
argmax, so the row plays one game per position and `collapse_replicates` pins it
there: the shipped selection's colour swap gives two games per cell and twelve
across the grid, and raising the seeds or the games per position produces the
same game again. The model side is therefore a point mass at each rating where
the human side is a distribution over openings, game lengths, and results.

The total variation distance from a point mass to a distribution is one minus
the mass the distribution puts on that single category. It cannot go lower
however well the model plays, and it moves when the *human* distribution moves.
On `n527-f32` step 20000 the repertoire reading was 0.9407 and the model played
the Italian Game in every game; the human mass on the Italian Game is 0.059, and
0.9407 is 1 minus 0.059 to four decimals. The number reports how popular an
opening is among humans, not how the checkpoint chose it.

The same shape is already documented for the termination mix, which saturates
while the model produces none of the human categories. There the saturation
unpins as soon as a checkpoint reaches one of them. Here it does not, because
the model side has one category by construction rather than by weakness.

## Decision

The rollout compares only temperatures whose replicates vary. The greedy row is
still played and still recorded: its cells carry the rollout scalars, and its
collapsed distinct-game fraction is what the sampled rows' fraction is read
against. It is the comparison against human play, and the exact walk beside it,
that are not taken.

The predicate is `replicates_vary`, which already decides how many replicates a
cell is worth playing and what re-measuring a result would mean. This is the
same question a third time.

## Consequences

The question the row wanted to ask is real and belongs to the human-prefix arm.
Greedy play from many distinct starting positions is a distribution again, so
"does the model's best move look human" is answerable there and is not
answerable from the standard start.

`compare_curves` keeps `model_varies`, and the stated zero of
`0032-a-replayed-reading-has-no-evaluation-noise.md` keeps its tests.
A caller outside the rollout with a deterministic model side and a real
distribution behind it still gets the floor those records specify. What changed
is that the rollout is not such a caller.

Two readings the rollout does keep gained the null they had been missing, which
is a separate matter recorded in the pull request rather than here: an exactly
enumerated model side still meets a finite human sample, and a distance against
it is not a distance from zero.
