# 0092: The Rating Axis Is Balanced By Thinning, At Four

Date: 2026-10-03

## Status

Accepted. Answers `#498`. Applies
`0016-sampling-axes-versus-measured-distributions.md`, which opens rating to
reweighting because the model is conditioned on it and closes speed until
`#497` gives the model the clock. Refined by
`0093-the-target-run-draws-each-batch-across-sixteen-days.md`, which finds that
most of the draw noise below came from training on one day at a time.

## Context

The training selection inherits the corpus's rating shape: roughly normal about
1635 with a standard deviation of 351, so the dial's endpoints sit in thin data.
1050 is the 5th percentile and 2100 the 91st. `#496` measures the dial
delivering 0.30 Elo per Elo asked, and thin endpoints were one candidate cause.

The corpus holds about 125.6e9 training decisions. A vehicle arm reads 1.14e9
and the target run's budget is about 2.4e10, so every run discards most of the
corpus whatever it does, and the question is only which part.

## Decision

**A selection balances its rating axis by thinning games, not by weighting the
loss.** A game is kept with probability proportional to its decisions' weights,
drawn against a digest of its id. Nothing repeats, every kept decision counts
once, and a gradient is no noisier than an unbalanced one. Loss weighting would
spend the same compute on examples it then scales down, and buys nothing here,
where data is not scarce.

**One dial, `rating_balance`: the most a rating is drawn above its natural
rate, relative to the commonest.** Every rating at least `1 / balance` as common
as the peak is drawn equally often, and rarer ones at `balance` times their own
rate. One leaves the population as it comes, and the limit draws every rating
alike, which keeps almost nothing. The weight is taken against a smoothed
density of mover ratings, not against bands.

The issue specified a strength exponent on `target / actual` with a separate
clip. Under thinning the two are not independent. At full strength the clip
alone decides how far the flat region reaches, and below it the clip still sets
how much of the middle is discarded while barely shaping anything: strength 0.5
kept less data than strength 1.0. One dial with a direct reading replaces them.

**The data budget is the ceiling, and every reweighted axis spends from it.** A
balanced selection holds the corpus times its retained share, after every other
filter, and a run reading more than that repeats data; the runner warns when it
would. Retained shares multiply across axes, so a speed balance after `#497`
draws on what the rating balance leaves.

| balance | drawn equally | retained | effective sample | pool against the target budget |
| ---: | --- | ---: | ---: | ---: |
| 1 | none | 100% | 100% | 5.2x |
| 2 | 1224 to 2102 | 69% | 93% | 3.6x |
| 4 | 1039 to 2250 | 41% | 78% | 2.1x |
| 8 | 927 to 2365 | 23% | 62% | 1.2x |
| 16 | 843 to 2461 | 12% | 46% | 0.6x |

**The target run takes a balance of four.** It equalizes the whole range the
rating ladder measures and leaves room for a second axis. It is reached through
`#495`, the combined arm that reads every accepted candidate together. The
vehicle does not change.

## What Was Measured

Vehicle arms at seed 17, full horizon, scored on the checkpoint suite. Draw 1 is
the vehicle's loader seed, read against the mean of the six `#488` replicates,
which share it. Draw 2 is a second loader seed, read against the vehicle trained
on that draw. Held-out loss deltas, with negative better:

| | below 1200 | 1200 to 1599 | 1600 to 1999 | 2000 and up |
| --- | ---: | ---: | ---: | ---: |
| balance 2, draw 1 | -0.0018 | -0.0003 | +0.0001 | -0.0039 |
| balance 4, draw 1 | -0.0046 | +0.0005 | +0.0010 | -0.0063 |
| balance 4, draw 2 | -0.0066 | -0.0026 | -0.0028 | -0.0111 |
| balance 4 minus uniform thinning at its retention, draw 1 | -0.0049 | +0.0009 | +0.0014 | -0.0060 |
| balance 8, draw 1 | -0.0036 | +0.0023 | +0.0032 | -0.0063 |
| vehicle, draw 2 against draw 1 | +0.0037 | +0.0036 | +0.0043 | +0.0057 |

`ladder.fitted_rating_slope` at temperatures 0, 0.7 and 1.0, from 0.30:

| | T=0 | T=0.7 | T=1.0 |
| --- | ---: | ---: | ---: |
| balance 2, draw 1 | +0.004 | -0.020 | -0.005 |
| balance 4, draw 1 | +0.030 | +0.038 | +0.027 |
| balance 4, draw 2 | +0.025 | +0.021 | +0.005 |
| balance 4 minus uniform thinning, draw 1 | +0.032 | +0.064 | +0.038 |
| balance 8, draw 1 | +0.033 | +0.005 | -0.001 |

The tails gain on both draws by the same margin over the middle, 0.0062. A draw
change alone moves every band together. Uniform thinning to the same retained
share, which reads as many shards, moves no band and lowers the slope, so the
gain belongs to the balance and not to the shard diversity it brings. Balance 2
buys under half of the tail gain and none of the slope. Balance 8 gains nothing
over 4 in the bands measured, gives back the slope at two of three
temperatures, and costs the middle about three replicate deviations. The curve
turns between 4 and 8.

## What This Does Not Claim

**No seed floor qualifies any of it.** A draw change alone moves held-out loss
by several times the replicates' initialization spread, and the report declines
the floor for every arm here on training health, which a draw change alone also
trips. The claim rests on the effect repeating across two draw-matched pairs and
against a diversity-matched control, not on a floor. `#574` owns the draw noise.

**Nothing outside 1050 to 2100 is measured.** The engine accepts 400 to 2500,
and a balance above 4 only reaches ratings no benchmark reads. `#577` owns
that, and a case for raising the balance waits on it.

**The costs are measured and accepted.** The middle bands pay about one
replicate deviation. Conditioned strength loses 15 to 20 Elo more to
temperature in every reading that measured it.

## Consequences

The ultrabullet slice reads 1.949 on the vehicle's draw and 1.97 to 2.00 on
every perturbation of it: a second draw, uniform thinning, and either balance. It
is a property of the draw on 1.1% of positions, not of the balance.

If a wider balance is ever worth more than the corpus can feed at the target's
horizon, the lever is a larger source, such as the exports after 2021-06, which
trade away the centisecond clocks `0045` chose this source for. The nine months
`0058` dropped add about 7% and move the ceiling by less than that.

## Alternatives Considered

**Weighting the loss by the same ratio.** Equal in expectation, rejected for
spending compute on down-weighted examples and adding gradient variance when the
corpus can simply be thinned.

**Bands.** A banded sampler puts edges into the weight that the data does not
have. Bands remain how the readings are reported.

**The issue's strength and clip.** Rejected above: two dials that do one job
under thinning, with a weaker setting discarding more.

## References

- `#498`, `#496`, `#497`, `#495`, `#574`, `#577`
- `0016-sampling-axes-versus-measured-distributions.md`
- `0062-the-breadth-corpus-filters-for-validity-alone.md`
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `docs/data.md` (Selecting Within A Corpus)
