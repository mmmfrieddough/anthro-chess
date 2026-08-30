# 0084: A Perturbed Opportunity Is An Easier One

Date: 2026-08-30

## Status

Accepted. Refines `0024-one-sided-perturbation-derived-novelty.md`, which
established that retention is paired on position after an unpaired reading
inverted the sign for legality. That pairing holds and is unchanged. What this
record adds is that pairing on the ply does not reach a confound that lives in
the board, and that the material-gain reading had the same inversion for that
second reason.

## Context

The novelty sweep reported `novelty.material_gain_retention`: the rate at which
the model takes an available material win, over the same checkpoint's rate at
the plies the perturbed arm reached. On the 555k-step checkpoint it read 1.30 at
full dose, which says capability improves under perturbation.

The populations behind that ratio are not the same. Over the same 1367 plies at
full dose the perturbed arm holds 869 material-gain opportunities and the
control 191, overlapping in 153. Pairing on the ply is what `0024` added, and it
works for legality because legality is defined at every position. Whether a
position offers a material win, and how large, is a property of the board, and
the board is exactly what the dose replaces.

Measured over 1600 games, model-free, the mix moves hard. The share of
material-gain opportunities whose best capture nets a pawn, a minor piece, or a
rook or more:

| dose | pawn | minor | rook or better |
| --- | ---: | ---: | ---: |
| 0.000 | 57.3% | 34.4% | 8.4% |
| 1.000 | 33.3% | 36.4% | 30.2% |

A random opponent hangs a queen where a human hangs a pawn.

## Decision

**The material-gain reading is banded by how much the best win nets**, in the
shared material scale, and every band reports its own policy mass, selected
rate, and share of the arm's positions. The retention ratio is withdrawn.

**A band scores the captures that win the most, not every capture that wins
something.** The mix effect returns inside a band otherwise: mass over a set
counts its members, and the number of winning captures per position rises with
the dose in every band, 1.70 to 2.29 in the minor one. Restricted to the
captures tied at the best gain that count is flat, 1.52 to 1.63, and the
measured decline is larger for it: minor-band mass falls 0.268 rather than
0.203, and the rook-or-better band 0.287 rather than 0.151. Reading the whole
set was hiding a quarter to a half of the effect.

Held at a fixed win size the reading inverts. The minor band on the 555k
checkpoint, policy mass by dose: 0.8550, 0.7811, 0.7471, 0.6746, 0.6250, 0.5869.
Monotone at every step, on all three checkpoints measured and on every draw. The
rate rose because the mix got easier, not because the model held up.

**Policy mass leads and the selected rate follows it.** The failure this
benchmark was opened for is recorded in the motivating evidence of its issue: a
winning capture ranked sixth at 3.7% in a position humans never reach, beside
the same win taken first one ply earlier. That is a mass collapse, and the rate
sees it only once the mass has fallen far enough to change the argmax.

**The opportunity share of each band is reported beside it**, so a reading that
moved because the mix moved is visible rather than inferred from the counts.

**Every band reading carries a sampling floor**, because the view is sized on
one of them and a value with no floor cannot support a claim that a difference
survived the draw.

**Phase is not sliced beneath the bands.** Truncation moves that mix as well,
from 43% opening on the control to 68% at full dose. Held fixed, the minor band
falls 0.195 in the opening and 0.231 in the middlegame against 0.203 unsliced,
because the phase gap inside a band is around 0.03 where the win-size gap is
0.34. The per-arm phase counts stay in the detail tier so the check can be
repeated rather than trusted.

## Alternatives Considered

**Pairing on the opportunity rather than the ply.** Score only the positions
where both arms realize the predicate. At full dose that is 153 of the arm's
869, and the two boards still differ at those plies, so it buys a fifth of the
sample and fixes nothing.

**Reweighting the control to the arm's mix.** Needs the same stratification
variable the bands are, and produces one number where three carry more: the
bands show that the pawn band is nearly flat while the larger wins fall.

**Generating positions with the rare predicates.** This was considered for the
predicates the perturbation destroys rather than for material gain, and it is
rejected on the same ground `docs/evaluation.md` rejects hand-specified position
features. A synthesized position has no human game behind it, so there is no
control arm and no dose axis, and what is left is an absolute rate on positions
nobody faced.

## Consequences

The novelty family reports material gain alone. Over sixteen times the pool,
full dose leaves 136 mate-available opportunities, 45 mate-threatened, 1
only-move and no stalemate-available. Only-move is unreachable at any pool size,
because check falls from 2.3% of scored positions to 0.2% and those positions
are replies to a check. Those predicates keep their sample over human positions
in the adjudicated-decisions family, which is where they were defined.

The legality half collapses to one number. `novelty.legal_mass_retention` read
0.9941 to 0.9974 across checkpoints whose absolute legality spans 0.8185 to
0.9993, and `novelty.mask_penalty_ratio` rose monotonically with checkpoint
quality against a declared direction of lower is better, because both sides of
it approach zero and the denominator falls faster. What replaces them is a
difference on the same paired plies.

Every stored novelty measurement predates this and is unreadable against what
follows, which `0068-a-pool-re-cut-breaks-benchmark-history-and-that-is-accepted.md`
already accepts for a re-cut pool and applies here for the same reason.

The reading no longer builds a slice table, since it reported one number off it
and the rule-case dimension resolved every predicate for every position to do
so. With the derivation and the labelling moved across processes, a reading over
6400 games falls from 291 to 74 seconds.

Reading the tied-best captures rather than all of them costs discrimination as
well as buying honesty, because less of each position's mass is counted. At the
declared size the full-dose minor band has a spread of 0.0042 over four draws,
against 0.0067 between two checkpoints of one recipe four times apart in
training. That is 1.6 times the floor. The same pair read in distribution moves
0.0008 against a floor of 0.0032, so the perturbed reading separates them where
an unperturbed one does not, and neither separates them cleanly. Reaching twice
the floor on a pair this close would take about ten thousand games, and whether
that is worth its wall clock is a sizing question rather than a design one.
