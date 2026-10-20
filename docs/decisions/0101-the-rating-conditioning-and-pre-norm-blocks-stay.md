# 0101: The Rating Conditioning And Pre-Norm Blocks Stay

Date: 2026-10-19

## Status

Accepted. Answers `#602`. Keeps where and how
`0066-the-trunk-sees-the-rating-and-the-board-keeps-its-shape.md` puts the
rating, against alternative conditioning paths and Chessformer's post-norm
blocks.

## Context

Read across the embedding's whole span
(`0100-the-dial-accepts-its-whole-span-and-is-measured-against-humans.md`),
every checkpoint's dial rises through the corpus, peaks near 3000 and then
falls; the width-512 clock model asked for 5000 plays about as strong as asked
for 0. The rating vector moves along a line between two learned anchors, so its
length is smallest near the corpus median and grows in both directions: on that
model 1.4 at 1600 and 11.6 at 5000, against 4.2 for a square's identity vector.
Capping a trained model's rating vector at its length at 3000 held play within
about 50 Elo of the peak to 5000, so the length is the cause.

The interpolation matches Chessformer / Maia-3's exactly. Normalization does
not: Maia-3 feeds the projected sum straight into post-norm blocks, while this
model normalizes the summed embedding and runs pre-norm blocks. Both arrived in
`#505` without a record choosing them.

## Decision

**The rating conditioning is unchanged.** No path tried here moved the dial
inside the data, and what the top end does reaches a player only through the
advertised range, which `#600` will set from each checkpoint's measurement.

**The blocks stay pre-norm, a deliberate departure from Chessformer.** At an
equal learning rate post-norm was better on loss and level on the puzzle slope
and the full-range ladder, though the suite's ladder slope read about two
replicate deviations lower at T=0 and T=1 on one seed each. It diverged at the
vehicle's rate, and at the rates it trained at it was level on
loss and a few percent weaker on rating response. Every rate rule
(`0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md`, `0090`) is
fitted for pre-norm, so adopting post-norm means finding its rate at the target,
where a divergence costs a target run. Maia-3's code offers no pre-norm option
and trains at 5e-5, about sixty times below the vehicle's rate, so nothing shows
that it chose post-norm over pre-norm.

A post-norm rate sweep that trains at or near pre-norm's rate (0.0025, or 0.003 with a
longer warmup) and matches its dial would reopen this.

## What Was Measured

Each arm is the ablation vehicle with one change, at seed 17 unless noted,
read on the checkpoint suite against the six `#488` replicates and on a
full-range ladder (0 to 4950 every 150) at T=1. `#602` holds every reading.

| arm | inside the data | above the data | held-out loss, replicate deviations |
| --- | --- | --- | ---: |
| arc: one learned length, rotating direction | unchanged | the fall about halved | -0.7 |
| modulation: each block scaled and shifted by the rating | unchanged | the fall about twice as steep | +0.2 |
| fixed-length line, seeds 17 and 29 | unchanged | holds within about 60 Elo of its peak, which is about 32 Elo lower | -0.1 |
| fixed-length line with a linear move-logit shift | unchanged | about 50 Elo further to 4950 at T=1, flat at T=0 | -0.9 |
| rating balance 16, beside `0092`'s 8 | 8 separates the top about 5 points more; 16 adds nothing | | flat |

The arc and fixed-length line also inflate the absent-rating reading, because
the never-trained unrated vector keeps its starting size while every rated
vector takes the learned length.

**Post-norm.** At the vehicle's 0.003 it diverged by step 12000.

| run | learning rate | held-out loss | puzzle sampled rating slope | strength gained 600 to 3000, T=1 |
| --- | ---: | ---: | ---: | ---: |
| width 128, pre-norm | 0.001 | 1.4334 | 0.191 | 559 |
| width 128, post-norm | 0.001 | 1.4294 | 0.191 | 572 |
| width 128, post-norm, seeds 17 and 29 | 0.002 | 1.4273, 1.4289 | 0.196, 0.189 | 548, 553 |
| width 128, pre-norm replicates | 0.003 | 1.4280 to 1.4304 | 0.196 to 0.209 | 570 (seed 17) |
| width 512 with clock, pre-norm | 0.000689 | 1.3492 | 0.285 | 876 |
| width 512 with clock, post-norm | 0.000459 | 1.3503 | 0.277 | 832 |

The width-512 post-norm rate is two-thirds of the control's, the ratio width
128 needed. There the dial is narrower below 600 (126 Elo from 0 to 600 against
229) and by about 20 Elo per 450 points from 1050 to 1950, with no seed floor
at this width to read either against. Above the data post-norm falls far more slowly: asked for 4800 it
plays 402 above its 600 setting, against -165 for the control.

## What This Does Not Claim

**That post-norm weakens the rating response.** The puzzle slope follows the
learning rate at both norms (puzzle slope about 0.191 at 0.001, 0.193 at 0.002,
0.203 at 0.003), and post-norm trained only at the lower rates. Width 512 is one
run per side with no seed floor.

**That the top end is settled.** Nothing here separates players above about
2200 the way humans do (`#605`); balance and conditioning did not, and capacity
and top-end data are what remain.

## Consequences

The arms branch, `602-rating-conditioning-arms`, is not merged; its runs stay
on the machine that trained them. The target run keeps the vehicle's
conditioning and pre-norm blocks.

## Alternatives Considered

**The fixed-length line.** It holds play above the data on both seeds, but its
peak sits about 32 Elo below the vehicle's on both, and its gain is above the
data, which `#600` will keep from players once it lands.

**Post-norm for parity with Chessformer.** Weighed under Decision.

## References

- `#602`, `#577`, `#600`, `#605`, `#505`, `#488`
- `0066-the-trunk-sees-the-rating-and-the-board-keeps-its-shape.md`
- `0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md`
- `0090-the-size-is-not-fitted-here-and-the-rate-is-measured-at-the-target.md`
- `0092-the-rating-axis-is-balanced-by-thinning-at-four.md`
- `0094-the-dial-is-not-widened-by-rating-dropout-or-the-anchor-range.md`
- `0100-the-dial-accepts-its-whole-span-and-is-measured-against-humans.md`
