# 0030: The Ladder's Ply Limit Stays Where The Training Data Ends

Date: 2026-08-03

## Status

Accepted. Extends `0027-settled-rating-ladder-grid.md`.

## Context

Decision 0027 settled the ladder's seat grid and deliberately left the ply limit
open, because the measurement that would decide it had not been taken. The
observation behind the question is large: between 47% and 66% of a full-size
ladder's games reached `generation.maximum_generated_plies`, produced no result
under 0022's exclusion rule, and informed no pairwise comparison. Those are also
the most expensive games in the benchmark, since an unfinished game runs the full
limit by definition, so the majority of the ladder's cost buys nothing and the
effective sample behind every fitted rating is about half the games played.

The two obvious readings of that point in opposite directions. If the games that
stop at the limit are mostly slow finishers, the limit is too low and cutting
them off wastes work already paid for. If they are shuffles that would still be
running at any limit worth paying for, the tail is pure cost and the limit is too
high.

### The Generated Distribution Does Not Choose Between Them

An 840-game reading at the full fifteen-seat grid, taken on `training-blitz-1m`
step 500, measured every game's length and termination. It reproduces the
attrition — 413 of 840 games, 49.2%, ended at the limit, consuming 61.9% of all
plies generated — and it rules out lowering the limit outright. Truncating the
same games at 200 plies rather than 300 yields **1.007 scored games per second of
generation against 1.081**, so a shorter limit makes the benchmark slightly
*less* efficient at producing results, not more. The final band before the cap
costs 0.533 s per scored game against a 0.925 s run average.

It does not settle raising, and it cannot. The hazard — the chance a game still
running at the start of a 20-ply band ends inside it — fluctuates between 3.5%
and 9.7% across the range with no downward trend, and is 7.0% in the final band
before the cap. There is no tail where games stop finishing, and so no ply value
at which the discard would go away. The reading is censored at exactly the point
the question is about, and every extrapolation past it says the same unhelpful
thing: raising buys more scored games at a roughly constant exchange rate,
indefinitely.

### The Human Corpus Does Choose Between Them

The quantity the reading is missing is not on the generated side.
`training-blitz-1m` trained on `lichess-blitz-2017-04`, whose million accepted
games are the only statement the project has about how long a game is supposed
to run.

Game length in plies, over the whole corpus and over the ladder games that
reached a result:

| | p50 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| human corpus, 1,000,000 games | 65 | 113 | 127 | 153 | 306 |
| ladder games that finished | 176 | 280 | 297 | 307 | 308 |

The corpus mean is 69.3 plies against 183.2 for a completed ladder game, and the
corpus reaches 190 plies only at its 99.9th percentile and 234 at its 99.99th.
Games longer than 200 plies number 548 of a million, longer than 260 number 29,
longer than 300 number **two**, and **none is longer than 308**. There is no
truncation upstream producing that: preparation filters on `minimum_plies` and
has no maximum, and the tail decays smoothly to a single longest game rather than
piling up against a wall.

What the model was trained at is narrower still, and the split matters. Training
reads the train split alone, whose longest game is **301** plies; the 306-ply
game is in the test split, which is the ladder's own opening pool and which the
model never saw. A game encodes to one token past its move count, so
`training-blitz-1m` has training signal at ply indices 0 through 301 and no
further.

The ladder plays an 8-ply human prefix and then allows 300 generated plies, so
its last decision is taken at **ply index 307** — six indices past the trained
bound, not at it.

That bound is also softer than a single number suggests. The same corpus split
three ways by a hash of the game id gives longest games of 301, 300, and 306, so
a 6-ply spread falls out of partitioning alone. The end of the trained ply range
is "about 300", and it is known to about the size of the overhang above.

## Decision

**`generation.maximum_generated_plies` stays at 300.** It is a round number,
not a derived one; what this record establishes is that the evidence brackets it
from both sides and leaves nothing inside the bracket worth ending a series for.

### Raising It Buys Results From Where The Model Was Never Trained

Every scored game a higher limit recovers is a game decided past ply 308, and the
model has no training signal beyond ply index 301 — not a degraded signal, none.
The seats' behaviour there is extrapolation, and a result decided by it would
enter the joint fit as a level of play.

That is the objection 0022 already makes to adjudicating an unfinished game into
a draw, applied to the other end of the same problem. Refusing to score the ply
limit as a draw and then extending the limit until the same games score anyway
would report the harness's reach rather than the seats' strength. The marginal
cost argument is real and it loses to this: cheap sample of the wrong quantity is
not precision.

The boundary is not a cliff, and this record does not claim one. A ladder game
that reaches 240 plies is already in a regime 0.007% of human games occupy, and
the median *completed* ladder game at 176 plies is longer than 99% of them. 300
is not where generated play leaves the human distribution; it is roughly the last
point at which it is still inside it at all. What the argument rules out is
buying *more* results from past that point, and zero training signal is the one
threshold nameable without inventing a percentile.

### Nor Is It Worth Trimming The Overhang

Applied strictly the same argument would put the limit at 294, so that the last
decision lands on ply index 301 exactly. This record declines that, for the
reason the split spread gives: the trained bound is known to about six plies, and
294 chases a correction the size of its own uncertainty. Under
`0013-benchmark-result-comparability.md` it would still end every series the
ladder has written, and the measurement above says a shorter limit does not even
pay for itself in throughput. A 2% trim is not worth a comparability break.

### The Attrition Is A Reading About The Model, Not A Mis-Set Dial

Not one of the million corpus games is long enough to have reached this limit, so
a model that finished games the way its corpus does would essentially never hit
it. The ladder's seats hit it in about half. The gap is the finding, and the
ladder already reports it: scored games rose from 3,405 to 5,273 of 10,080
between step 100 and step 8000, which is the clearest discrimination the first
full-size reading produced, on a quantity that is not among the benchmark's
headline outputs.

Attrition is also almost entirely a property of sampling. By pairing temperature
it runs 3.8% for two greedy seats, 48.8% at (0.7, 0.7), and 63.8% at (1.0, 1.0);
two temperature-zero seats repeat into fivefold at around 177 plies instead.

So the discard resolves as the model improves rather than as the limit moves.

### Precision Has A Lever That Does Not Reopen This

The discard costs the ladder sample, and sample is what 0027 found it short of.
That is an argument for buying precision, not for buying it here.
`grid.seeds`, `generation.games_per_position`, and `openings.view.maximum_games`
are sample sizes rather than measurement settings and stay out of series identity
for exactly this reason. They cost compute alone. The ply limit costs compute,
ends every series the ladder has written, and changes what the fit is over.

## Consequences

The ladder's declared workload is now settled end to end. 0027 fixed the grid and
this fixes the limit, both inside the window
`0013-benchmark-result-comparability.md` leaves open before the evaluation core
is designated, so neither costs a protected series.

After that designation the limit is effectively permanent, and the thing that
would reopen it is a corpus rather than a cost. A model trained on materially
longer games — rapid or classical rather than blitz — would have training signal
past ply 308, and matching the limit to it would be a new benchmark generation
taken at a seam, not a tuning step. The number to match is the longest game in
that corpus's **train** split: the manifest's `games.plies.maximum_per_game` is
corpus-wide, which is the right thing for training to refuse an unencodable
corpus on and the wrong thing here, since it includes the held-out splits the
model never reads. It is measured from the normalized data rather than
remembered, and "materially longer" means enough to clear the several plies of
spread that partitioning alone produces.

Nothing in this record changes what the ladder computes. It changes what the
declared value rests on, and it closes the second of the two questions 0027 left
open. The first — that the ladder states ordering, slope, and span with no
resolution beside them — is untouched and remains the more consequential.

## References

- `docs/decisions/0022-one-joint-rating-ladder-fit.md`
- `docs/decisions/0027-settled-rating-ladder-grid.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/planning/first-full-suite-reading.md`
- `configs/evaluation/rating-ladder.toml`
- `configs/data/lichess-blitz-2017-04.toml`
