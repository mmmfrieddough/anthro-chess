# 0076: The Vehicle Is Width 128, Trained To The Target's Regime

Date: 2026-08-21

## Status

Accepted. Fixes step 3 of the order in `docs/scaling.md`.

Rests on `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`,
which decided that one configuration is frozen and why, and on
`0071-the-target-is-the-size-the-published-ladder-flattens-at.md`, which fixes
the target this size is derived from and priced an arm at each candidate width.

Carries the arithmetic `0074-the-vehicle-freezes-on-bfloat16-with-tf32-without-a-quality-reading.md`
settled and the batch shape `0075-a-training-batch-is-decisions-not-games.md`
settled, by taking both as shipped defaults rather than by restating them.

`0067-a-horizon-is-a-branch-not-a-restart.md` owns the schedule family this
configuration instantiates, and the qualification that the horizon sits outside
the digest.

`#488` characterizes the seed dispersion against the identity this record pins.

## Context

`0071` derived the target and, in its Consequences, priced the vehicle at each
candidate width against the throughput it had measured. Three things have landed
since that scan and all three move the price: compilation is on by default
(`0073`), batches are packed rather than padded (`0075`), and the corpus opens in
seconds rather than hours (`#517`). A size derived from the old numbers would
have been derived from a machine this project no longer runs.

So the scan was retaken before the size was fixed.

## What Was Measured

One idle RTX 4090 per point, the widened corpus through the shard-backed loader,
`bfloat16-mixed` with `high` matmul precision and compilation at their shipped
defaults, relaxed determinism, one accumulation step, roughly 3e6 positions per
point past a warmup exclusion. Shape held at `0071`'s rules, which reproduce its
published parameter counts exactly at every width.

**The packed micro-batch has an optimum rather than a ceiling, and it is 1024
positions.**

| width 128, positions per batch | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| active positions/s | 21,100 | 40,789 | 56,993 | 52,070 | 49,166 | 49,192 |

Below it the step is launch bound. Above it there is nothing to gain, and at
width 192 and above the card runs out of memory before the throughput improves.
This reverses the reading `#524` took at width 256, and the difference is
`0075`: that scan varied a game-shaped micro-batch, whose gains were the padding
packing has since removed. Under packed batches the micro-batch is a throughput
dial with a measured optimum, and the effective batch is set by accumulation.

At that micro-batch, and against the 800 positions per parameter `0071` puts the
target's regime at:

| width | parameters | active positions/s | hours per arm at the target's regime |
| --- | ---: | ---: | ---: |
| 128 | 1,422,662 | 56,993 | 5.55 |
| 160 | 2,139,398 | 42,690 | 11.14 |
| 192 | 3,006,790 | 35,122 | 19.02 |
| 256 | 5,205,830 | 25,784 | 44.85 |

Strict determinism was measured on the same shape: 17,609 positions per second
at width 128 against 56,993, a factor of 3.24, and 2.78 at width 160.

## Decision

**The vehicle is `model_dim` 128, 1,422,662 parameters, trained on 1.138e9
positions, which is 800 positions per parameter and a 5.55 hour arm on one card.
`configs/training/ablation-vehicle.toml` is the designation and
`anthro_chess.training.vehicle` pins its identity.**

### Width Is 128 Because 160 Is The Worse Half Of A Trade Nothing Wins

Only multiples of 32 are candidates, since that is what holds the attention head
dimension `0070` fixed. So the choice was 128 or 160, everything larger being
priced in days per arm.

160 doubles the arm and buys 1.5x the parameters. That is the shape of a
compromise that gets neither thing: if 1.42M parameters cannot represent a
rating-conditioned distribution, 2.14M does not fix it, and the width that would
is 256 at 44.85 hours. There is no capacity threshold between the two, and the
instrument's whole value is that an arm is cheap enough to spend on a question
that may return nothing.

**What this leans on is that the regime transfers and the size does not have
to.** A vehicle at the target's positions-per-parameter is a small model trained
the way the target will be trained. A vehicle at 160 in the same wall clock would
be a slightly larger model trained at 400 positions per parameter, which is a
regime the target never occupies, and a candidate that helps an undertrained
model is not the same finding as one that helps a trained one.

### Determinism Is Relaxed

Strict costs 3.24x on every arm, every replicate `#488` runs, and every ladder
point that reads through this vehicle. What it buys is a reproducible weight
sequence, which no comparison here reads: an arm is compared by its readings,
and `#488` measures the spread of those readings rather than asserting they are
reproducible.

The consequence is that two arms at one seed do not agree, and their
disagreement is the nondeterminism term. That is a reading `#488` can take at
this vehicle's own digest, which a strict pair could not be.

### It Trains On Everything

No speed filter, no rating bound, no game cap, and no marked-account snapshot.

The first three are what let the vehicle be the unconditioned control for clock
conditioning in Milestone 5, which `docs/planning/roadmap.md` names as a
decision the vehicle's selection makes rather than a cost it merely carries.
Spanning speeds costs nothing at a corpus of 2.09e9 games against a horizon of
1.138e9 positions, and it saves training that arm separately.

The snapshot is left out because applying it is a candidate change to read
against this vehicle rather than a property of it. It reaches `dataset_sha256`,
so an arm that applies it carries a different training identity, correctly.

### The Rate Was Swept Rather Than Inherited

Every checked-in configuration before this one sat at 1e-3 or 3e-3, which is a
framework default that had never been compared against anything.
`docs/scaling.md` marks the peak learning rate hard against six of eight columns
of its coupling table, so a vehicle frozen at an untuned rate would make a
candidate that happens to suit that rate look good for the wrong reason, and the
digest would make it permanent.

The sweep and its result are recorded below. It is one sweep at one size, and it
does not pretend to be the cross-scale rule `#489` fits.

## What This Gives Up, Deliberately

**The vehicle has no affordable validation reading.** An end-of-run validation
over the widened corpus streams the whole 104,360,891 game validation split
before its subsample applies; a `fraction` of 1e-5 was still reading after four
minutes of a 40 step run. So an arm is judged by scoring its checkpoint through
`anthro eval` against the frozen pool, which is the path the comparison was
always meant to take, and the sweep below was judged on training loss instead.
That substitution is sound at this horizon rather than a compromise: 1.138e9
positions against a corpus of 138.7e9 plies repeats nothing, so a training
position is a held-out position.

**A width-128 result is a small-model result.** `docs/scaling.md` owns what
transfers and what does not, and nothing here narrows it. What is new is only
that the regime is matched, which removes one of the two ways a vehicle-scale
reading can mislead and leaves the other.

**The horizon is not pinned by machine.** `training_sha256` excludes the step
budget by construction, which is what lets a cooldown branch match its trunk. So
an edit to `steps` alone changes what the vehicle is while leaving its identity
intact. A test asserts the horizon against the regime it was derived from, which
is the only place that check can live.

## Consequences

**`#488` characterizes the dispersion against this identity**, at this horizon,
on the initialization seed with the data draw held. Its notes on the issue carry
why the strict-determinism check and the data-order axis it asked for do not
compose with a digest-keyed floor.

**An arm is a copy of this configuration with one thing changed**, run with its
own name. `docs/issue-workflow.md` carries the command.

**Promotion refuses this identity.** `anthro eval promote` is how a reading
joins the canonical line, and a frozen base is worth having only while no
success advances it.

## References

- `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
- `0071-the-target-is-the-size-the-published-ladder-flattens-at.md`
- `0067-a-horizon-is-a-branch-not-a-restart.md`
- `0075-a-training-batch-is-decisions-not-games.md`
- `docs/scaling.md`: the program, the coupling table, and what transfers
- `#488`: the dispersion stored against this identity
- `#489`: the cross-scale rules this sweep is not
