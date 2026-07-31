# 0024: One-Sided Perturbation-Derived Novelty

Date: 2026-07-31

## Status

Accepted.

## Context

The model degrades on positions unlike those it trained on. Measuring that by
slicing the frozen pool with a familiarity proxy does not work, and
`docs/evaluation.md` records why: the pool is human games, so it is in
distribution nearly by construction, and every candidate proxy fails on its own
terms. Perturbation replaces detection because it supplies novelty at a known
dose, with nothing to detect, validate, or hold stable across checkpoints.

Deciding to perturb does not decide what a perturbed continuation *is*. An
offline benchmark needs a whole action sequence to score a policy against, and
the human game stops being available the moment the line diverges. Something has
to supply the rest of it, and the obvious candidates each break a property the
benchmark depends on.

## Decision

A derived arm replaces **only the opponent's moves**, inside a fixed window, and
replays the human's own moves for the player being measured.

Three rules make that concrete.

**The window is fixed across arms.** It opens at a configured onset ply and runs
for a configured number of the opponent's moves, each followed by the player
reply the benchmark scores. Every arm of a suite uses the same window over the
same games with the same measured color, so the control and the perturbed arms
are paired position by position rather than merely drawn from the same games.

**Divergence is absorbing.** Once one opponent move has been replaced, every
later opponent move in the window is drawn too. The configured dose is the
per-move rate at which divergence *starts*; the realized share of replaced moves
is reported beside every reading rather than assumed equal to the configured
one.

**The human side is replayed while it stays legal.** When the human's own move
is no longer legal in the diverged position, the derived game ends there. The
share of the control arm's positions that survived is reported as its own
metric.

Replacements are uniform over the legal moves, drawn from a stream keyed by the
seed, the recipe, the recipe version, the game, and the position in the window,
so an arm reproduces from its recorded workload alone and no draw depends on the
order games were processed in.

The whole recipe — name, version, seed, onset, window, and dose — is the
declared workload, so two doses are two series and a recipe change ends both.
The source games remain the data component, which is what lets a perturbed arm
and its control share evaluation inputs while sitting on different series.

## Alternatives Considered

**Perturbing both sides.** This removes the truncation problem entirely, since
neither side needs a human continuation. It also measures a situation nobody
will create: the thing being modelled is an opponent playing garbage while the
model still chooses its own moves. A both-sided arm reports how the model scores
positions from a game it was not playing.

**Letting the model supply its own replies.** This keeps the continuation
coherent and on-policy. It also makes the derived positions model-dependent,
which defeats the multi-checkpoint trend the benchmark exists to report: two
checkpoints would be measured on two different sets of positions, and the
difference between them would be partly a difference in inputs.

**Sampling perturbations from the model's low-probability tail.** Rejected for
the same reason, and more sharply: the positions would move with every
checkpoint, so no series could survive training.

**A per-ply independent Bernoulli draw over the whole window.** This is what
"dose as a rate" first suggests, and it is not implementable. After the first
replacement the human's later opponent moves are moves in a game that no longer
exists, so a draw that says "leave this one alone" has nothing to leave alone.
Making divergence absorbing is that rule's only coherent form, and stating it
explicitly is better than a per-ply draw that silently degenerates into one.

**Continuing past an illegal human reply by substituting something.** Any
substitute is either a model choice, which reintroduces model dependence, or a
random one, which makes the arm two-sided. Ending the derivation is the only
option that preserves both properties, and its cost is a selection effect that
can at least be measured and reported.

## Consequences

Derived games are short past the onset, because a human reply stops being legal
fairly quickly once the position diverges. The benchmark measures the model
shortly after it is knocked off distribution, which is the region the motivating
evidence came from, but it cannot say much about deep off-distribution play.

The surviving continuations are selected: they are the ones whose human replies
happened to stay legal. That biases derived positions toward those where the
perturbation disturbed less, which if anything makes the measured dose response
conservative. The reported survival share is what keeps this visible rather than
hidden inside differing sample sizes.

That selection also forces **retention to be paired on position**: the control
is read over the plies the perturbed arm actually reached, never over every ply
it had. The first shakedown reading of this benchmark was taken with the
unpaired ratio and the difference was not cosmetic — legality appeared to
*improve* under perturbation, by three to ten percent depending on how far the
checkpoint had trained, because the surviving positions were systematically
easier than the ones the control also held. The artifact inverted the sign at
every checkpoint measured. Pairing moved the same readings to at or just below
one.

Legality turns out to be nearly flat in the dose on real checkpoints, which is
consistent with the motivating evidence rather than a disappointment: the
observed failure was a missed material win at a position whose raw-logit
legality was unremarkable. Legality is what still *has* a ground truth out of
distribution, not what the damage shows up in. The predicate readings are where
the dose response actually lives, and they move a great deal: across one
training run the material-gain retention at full dose went from nothing to most
of the unperturbed rate.

The dose response is **not monotonic**, which the expected shape predicted in
advance and the reading confirms. On a trained checkpoint the intermediate dose
retains less than the full dose. A small perturbation takes the model off book
without yet giving away material, so it loses learned guidance and gains
nothing; a large one hands over enough material that the remaining decisions are
easy. A dip in the middle is the reading, not an anomaly.

A rollout companion is still wanted and is not covered here. It plays whole
games against a random opponent, where the model genuinely does choose its own
moves, and it answers the conversion question this offline form cannot.
