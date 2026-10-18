# 0100: The Dial Accepts Its Span And Is Measured Against Humans

Date: 2026-10-18

## Status

Accepted. Answers `#577`. Refines what `0080-the-ladder-widens-and-openings-replace-seeds.md`
settled about the ladder's grid and openings, and reframes `#496`, whose
0.30 Elo per Elo asked was read against a target no human pairing reaches.

## Context

The engine advertised `UCI_Elo` from 400 to 2500. No record chose those
numbers: they were copied from an example line in `docs/interfaces.md` into the
UCI configuration by `#68`, and UCI leaves the range to the engine. The rating
embedding itself spans its anchors, 0 to 5000. The corpus's ratings run from
600, Lichess's floor before 2023 and a censoring pile-up about 25 times its
neighbours, to 3958, and thin out past about 3000: 99.9% of player ratings
are below about 2660.

The ladder read 1050 to 2100 and reported a fitted slope, implicitly against a
slope of one.

## Decision

**`UCI_Elo` accepts the embedding's span, read from its anchors.** A request
outside the ratings training covers is conditioned on the nearer edge of them,
and turning the strength limit off selects the top of the span, so it plays the
upper edge. The edges live in the UCI configuration until each checkpoint
records its own trained range, which `#600` owns.

**The dial is measured against human games, not the Elo formula.** For each
pair of ratings the ladder compares how often the higher seat scores with how
often the higher-rated human does in held-out blitz games (`#603`). It reports
two numbers and the curve behind them: the close-end calibration along the
dial, as a curve over rating location with its weighted mean and its worst
stretch, and the far-end match as the gap grows.

**The ladder reads the whole accepted range every time.** A core from 600 to
3000 carries the tuning numbers; an extension to 0 and 5000 reports separately,
so behaviour outside the data never dilutes the core. Most games go to
neighbouring seats, where the calibration lives, and readings start from the
initial position at T=1. `#604` builds it.

## What Was Measured

**Humans are not a slope of one.** From held-out blitz games with both players
within 25 points of each rating, the higher-rated player's score by gap, against
the Elo formula and the width-512 clock checkpoint (`arm-497-clock-w512`, T=1,
96 games per pairing):

| gap | 150 | 300 | 450 | 600 | 750 | 900 | 1050 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Elo formula | 70.3% | 84.9% | 93.0% | 96.9% | 98.7% | 99.4% | 99.8% |
| humans | 66.8% | 78.8% | 87.0% | 90.6% | 93.0% | 95.1% | 96.4% |
| width 512 | 62.2% | 70.6% | 77.7% | 83.3% | 88.6% | 91.3% | 95.0% |
| same run, halfway | 57.1% | 65.9% | 71.3% | 76.3% | 82.0% | 85.6% | 89.2% |

As Elo per rating point, humans realise about 0.8 at close gaps and about 0.55
past 1000; every model is flat across gaps, the width-512 checkpoint at about
0.5. A human rating is a noisy label and mis-rated players produce the
far-gap upsets, which a seat asked for one rating never is. So the close end is
the fair target, and a model matching humans there would be more decisive than
humans far apart. Between the halfway and final checkpoints of one run the
curve moved 5 to 6 points toward the human one at every gap.

80% of these human games are between players within 100 points and 97.5%
within 400: close pairings are what the model trains on.

**Pooling hid the top.** Read by location, every model stops separating players
above about 2200 while humans keep separating at the same rate up to 2850;
`#605` holds those readings.

**Outside the data the dial does not extrapolate.** Read across 0 to 5000, every
checkpoint rises through the corpus, peaks near 3000 and falls, the width-512
one to about the strength it shows asked for 0. `#602` traced that to the
rating vector's growing length and read several conditioning paths; each
collapsed, drifted down or held flat above the data, and none moved the dial
inside it. Below 600 part of the extra weakness was breakage: asked for 0, the
width-512 checkpoint drew a uniformly random player in 53 of 96 games, a copy
with its rating vector's length held at the data's edge 18.

## What This Does Not Claim

**Where the dial sits.** The ladder measures spacing, never location: a dial 300
Elo too strong everywhere reads the same. `#30` anchors it.

**That the human reference is clean at the far end.** Far-gap human pairings are
selected toward new and provisional accounts, which `#599`'s per-game rating
change would identify.

**That the edges are final.** 600 and 3000 are this corpus's; a checkpoint
trained on a corpus with a lower floor carries a wider range under `#600`.

## Consequences

`#496`'s gap is smaller than it read: the width-512 checkpoint realises roughly
70% of the human spread at close gaps rather than 30 to 50% of an unreachable
one, and what remains looks like something scale and top-end data close.

A setting above about 3000 plays the same as 3000, so the advertised span is
wider than the range where the dial moves; the interface can say so once the
checkpoint records its range.

## Alternatives Considered

**Narrowing `UCI_Elo` to the measured 1050 to 2100.** It drops the weak play
`docs/vision.md` asks for and would follow the measurement rather than the data.

**A wider ladder grid as one series.** Pooling the collapse outside the data
into the tuning numbers dilutes them, and a complete round robin over the whole
range costs about ten times today's.

**Conditioning paths that extrapolate.** `#602` read an arc, a fixed-length
line, a capped line, block modulation, post-norm blocks and a linear output
shift; the output shift carried about 50 Elo of further strength past the data
at T=1 and none greedily, which does not pay for its head.

## References

- `#577`, `#600`, `#602`, `#603`, `#604`, `#605`, `#601`, `#599`, `#496`, `#498`, `#30`, `#68`
- `0005-lichess-default-rating-scale.md`
- `0064-the-complete-round-robin-is-the-optimal-ladder-design.md`
- `0080-the-ladder-widens-and-openings-replace-seeds.md`
- `0092-the-rating-axis-is-balanced-by-thinning-at-four.md`
- `0094-the-dial-is-not-widened-by-rating-dropout-or-the-anchor-range.md`
- `0099-the-rating-benchmarks-declare-the-control-of-their-games.md`
- `docs/interfaces.md`, `docs/evaluation.md` (Rating Calibration)
