# 0071: The Target Is The Size The Published Ladder Flattens At

Date: 2026-08-19

## Status

Accepted. Fixes step 1 of the order in `docs/scaling.md`, which every later step
is sized from.

Rests on `0070-one-decision-per-pass-and-history-in-the-token-depth.md`. Adopting
Chessformer's shape is what makes that project's published size ladder a reading
of this architecture rather than an analogy, so a size chosen before 0070 would
have priced in an inefficiency 0070 removed.

`0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md` sizes
the vehicle from this figure.

Refined by
`0073-compilation-is-on-by-default-and-plain-fusion-beats-graph-capture.md`,
which measured graph compilation rather than inferring it from this batch scan
and found it worth a third at the target width.

`0074-the-vehicle-freezes-on-bfloat16-with-tf32-without-a-quality-reading.md`
uses the saturation this scan found: it is why the activation memory reduced
precision returns is worth nothing at width 256, and why at width 512 it is a
fit question rather than only a speed one.

## Context

The size had to be written down before the vehicle could be designated, and
nothing in the repository determined it. `docs/scaling.md` owns the method.

Two things about that method turned out to be wrong here in ways that mattered
more than the arithmetic it prescribes.

**The compute rule undercounts this architecture by about 60x.** Measured with
PyTorch's flop counter against the real forward and backward pass, the true cost
per decision is 57 to 61 times `6 x parameters` across widths 256 to 768, because
a decision is 64 square tokens rather than one. Read literally, the rule stated
over positions would have priced the budget nearly two orders of magnitude low,
and every positions-per-parameter figure compared against a language-model ratio
would have been wrong by the same factor. `docs/scaling.md` now states the rule
over tokens.

**Utilization was measured rather than estimated**, on both sides of the ratio:
throughput was read off runs at each candidate width, and the device ceiling it
is read against was measured too, at 163 TFLOP/s of achievable bf16 matrix
multiply on this card. The model reaches 17% of that at width 256 and 39% at
width 768. Quoting a vendor figure here would have been wrong by 2x, because the
number a consumer card advertises for float32 is not the rate a bf16 autocast
path runs at.

## What Was Measured

One idle RTX 4090, bf16 mixed precision with TF32, 200 steps per point, the
largest micro-batch of 32/16/8/4 that fits, shape held at 0070's rules.

| width | parameters | positions/s |
| --- | --- | --- |
| 128 | 1,422,662 | 31,322 |
| 192 | 3,006,790 | 20,761 |
| 256 | 5,205,830 | 15,132 |
| 384 | 11,546,950 | 9,738 |
| 512 | 20,642,630 | 6,818 |
| 768 | 47,884,102 | 3,914 |
| 1024 | 88,503,110 | 2,551 |

**The per-card memory ceiling does not bind.** Peak reserved memory sits near the
card at most widths, which reads as a constraint and is not one. At the target
width throughput saturates at batch 4, holding 14.3 GiB, and is flat through
batch 10 while reserved memory climbs to 22.9 GiB. Width 384 behaves the same
way. Everything above the saturation point is the allocator caching rather than
work the model needs, so activation checkpointing would buy no throughput at
these sizes.

## The Anchor

Chessformer publishes a size ladder on this architecture, this task, and this
metric. This project's parameter counts land on theirs at matching widths, the
differences being input-side: they carry two rating embeddings, this project
carries one along with repetition state and a colour bit.

| width | this project | Chessformer | their move matching |
| --- | --- | --- | --- |
| 192 | 3.01M | 3M | ablation only |
| 256 | 5.21M | 5M | 55.4% |
| 512 | 20.6M | 23M | 56.6% |
| 1024 | 88.5M | 79M | 57.1% |

**Their curve is already flat by 23M**, which is a measurement of diminishing
returns on the exact architecture this project runs rather than a scaling law
extrapolated across a task boundary. `docs/research.md` (Chessformer / Maia-3)
carries the ladder and the recipe behind it.

## Decision

**The target is `model_dim` 512, about 20.6M parameters, trained on roughly
1.6e10 positions.** The confident band is 10M to 50M, widths 384 to 768, and 512
is the point selected inside it. **A later reading within roughly 1.5x is not
grounds to reopen it.**

Those two are different tests and the band is the wider of them deliberately. The
band says where the evidence puts the answer; the flatness rule says how far a
later reading has to land from the selected point before re-deriving is worth
anything. A `#54` fit at 45M would be unsurprising and would still reopen this.

The horizon follows from the measured throughput and the envelope
`docs/vision.md` allows: 6,818 positions per second, two cards, fourteen days.
That lands within a few percent of the data budget Chessformer trained its ladder
on, which is the second place their work constrains this decision rather than
merely informing it.

**The intended regime is about 800 positions per parameter**, against their 713
at 23M. Same regime, marginally further into over-training, which is the
direction `docs/scaling.md` argues for on the ground that this project serves far
more than it trains.

**Going larger fails twice over.** Their own curve prices 79M at half a point
above 23M. And the budget cannot feed it: width 1024 over the same fourteen days
on two cards reaches 6.2e9 positions at 70 positions per parameter, well under
half their data budget. That is a badly under-trained 88M model, and there is no
reason to expect it to beat a well-trained 20M one.

## What This Rests On, And What Would Move It

**Two cards.** The figure assumes distributed training, which `#53` has not built
yet. One card halves the positions rather than the size: width 512 over the same
fourteen days reaches 8.25e9, which is about 400 positions per parameter. That is
still above where Chessformer trained its own largest model, so the size holds
and the regime moves. The size would only follow the budget down under a
compute-optimal derivation, and this is not one.

**The wall clock.** Fourteen days is the working figure for `docs/vision.md`'s
"weeks". Twenty-eight days moves the target by 1.41x, again inside the band.

**The recipe behind their data budget is inferred rather than stated**, on the
reading `docs/research.md` records. If that reading is wrong the horizon moves,
not the size.

**A throughput change is a hardware change in effect, and one is pending.**
`torch.compile` measures at up to 1.9x on the training step at the target width,
under fixed shapes, and is not wired into the runner. If that survives the
loader's real shapes it does not move where the published curve flattens, but it
does move which points can be fed: width 768 would reach 376 positions per
parameter, better fed than Chessformer's largest model was. The selection is then
made on whether the half point their curve prices is worth 2.3x the parameters
and 2.3x the serving cost, rather than on affordability. Recompute against the
measured figure rather than assuming this record's conclusion survives, even
though it is expected to.

**What would genuinely reopen it** is a ladder fit here disagreeing by more than
1.5x, an architecture change of 0070's magnitude, or a change of that size in
either the hardware or the throughput it delivers. A candidate change adopted
against the vehicle does not.

## What This Gives Up, Deliberately

**A ladder was not fitted, and this is not a compute-optimal derivation.** It
takes a published curve at four sizes on the same architecture and reads where it
flattens. `#54` fits the ladder properly, and a disagreement past the flatness
rule is a finding. Fitting first would have meant running that program before the
vehicle every arm in it is read against existed, which `0065` rejected for the
vehicle and which applies here for the same reason.

**Capacity is what this leans on, and it is not the only candidate.**
`docs/scaling.md` names as open whether a rating-conditioned distribution
saturates at a different size than raw strength, and nothing found answers it.
The size is chosen so the model can represent a second conditional distribution
at all. It explicitly does not lean on selection composition, which is `#498`'s
lever and a different mechanism. Neither substitutes for the other, and a flat
dial at this size would implicate the composition rather than acquit the
capacity.

**The move-matching figures are theirs, not a prediction.** Their ladder was
trained on their corpus with their rating-balanced sampling and both players'
ratings. Nothing here claims this project reaches 56.6% at 20.6M. What transfers
is the shape of the curve and the size at which it flattens.

## Consequences

**The vehicle is sized from this.** At the target's regime, a vehicle at width
128 reaches roughly 790 positions per parameter in a ten-hour arm on one card,
matching the target almost exactly and fitting two arms a day across two cards.
Width 256 in the same slot reaches a tenth of that. `#487` designates the vehicle
and owns the trade; the measurements above are what it needs.

**`docs/scaling.md` states the compute rule over tokens**, because the positions
form was wrong for this architecture by roughly 60x and it was the form the
method section carried.

## References

- `0070-one-decision-per-pass-and-history-in-the-token-depth.md`
- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `docs/research.md` (Chessformer / Maia-3): the ladder, the recipe, and what
  they do and do not establish
- `docs/scaling.md`: the method, the flatness rule, and the order
- `#54`: the ladder fit this decision expects to agree with
- `#53`: the two-card assumption
