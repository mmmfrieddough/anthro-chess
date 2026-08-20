# 0074: The Vehicle Freezes On bfloat16 With TF32, Without A Quality Reading

Date: 2026-08-20

## Status

Accepted.

Rests on `0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`,
which designates the base this precision is frozen into and owns why a candidate
change is read against it.

Sits beside `0073-compilation-is-on-by-default-and-plain-fusion-beats-graph-capture.md`,
which turned the third execution setting on and deliberately left this one open.

Uses the batch scan in `0071-the-target-is-the-size-the-published-ladder-flattens-at.md`,
which is what makes the memory this trade returns worth nothing at width 256 and
worth something at width 512.

## Context

`precision` and `matmul_precision` are execution-compatibility keys. A run
declares them, a continuation has to match them, and the ablation vehicle carries
whatever they are set to at the freeze. Every arm ever read against that vehicle
inherits the same pair, because an arm that changed either would be comparing
against a base computed in different arithmetic.

Nothing owned the decision. `docs/training-and-runtime.md` deferred the default
to the batch capacity selection, which sits a milestone after the freeze, so the
vehicle was on track to freeze at `float32` with `highest` because those were the
values a framework default happened to leave there.

## Decision

**The default is `bfloat16-mixed` with `high` matmul precision, and the vehicle
freezes on it. It is chosen on throughput, on published prior art, and on a
numerics check that says the arithmetic is sound. It is adopted without a quality
reading, because no reading that could qualify one can exist before the freeze.**

## Why No Quality Reading

The obvious acceptance test is a control arm at `float32` against a treatment arm
at `bfloat16-mixed`, read on the suite. That reading cannot be interpreted here,
and running it anyway would produce a number that looks like an answer.

A model comparison is qualified by a seed floor, and this project has none for
any configuration. `#488` builds one; it needs the vehicle, and the vehicle needs
this decision. The circle is not a sequencing accident. Precision is a
reference-frame decision in the sense `docs/scaling.md` gives: it invalidates
readings already taken rather than adding to them, so it cannot be deferred and
re-read as an ordinary candidate afterwards.

What an unfloored single-seed pair is worth here is already recorded, in
direction rather than in magnitude. `0029-model-change-control-arm.md` ran that
measurement once, and `#488` deliberately declines to repeat its figures because
they were taken on a model three orders of magnitude smaller and quoting them
invites reading them as a size. What transfers is that seed variance was large
enough to push a substantial share of floored metrics past their floor on its
own, and that one pair read better on every held-out and legality metric at once
from nothing but initialization. A control-versus-treatment pair at one seed is
exactly that shape, so it can produce a clean sweep in either direction and mean
nothing by it.

## The Residual Risk

If bfloat16 autocast degrades this model's quality subtly, freezing on it means
the project may never find out. The vehicle will carry it, every arm read against
the vehicle will carry it, and the comparison that would expose it is the one
that cannot be run.

What makes that acceptable rather than reckless is that the configuration is the
standard one rather than a novel one, and the alternative is paying between
2.0x and 2.4x for every arm and every ladder point the scaling program runs,
against a hypothesis nobody has evidence for. `docs/research.md` carries the prior art that turns that from
an assertion into a citation, and the two points that matter are these:

- **Chessformer, the anchor this project reads its size ladder from, trains in
  float16 with a loss scaler.** Its reproducibility table gives `use_amp` true,
  `amp_init_scale` 256, and `amp_max_scale` 8,192; a scaler exists only because
  float16's exponent range cannot hold a gradient distribution unshifted.
  bfloat16 carries float32's exponent range and needs none. So this project's
  choice is the same class of decision at a wider numeric margin than the
  published model it is shaped after.
- **The DeepMind chess transformer sets no precision at all**, so its float32
  matmuls take JAX's TPU default, which is a bfloat16 multiply with float32
  accumulation. That is TF32's arrangement at a narrower mantissa, 7 bits against
  10, and it is what that model trains under whether or not its authors framed it
  as a choice.

Neither is evidence about this architecture on this data. Both are evidence that
the risk being carried is one the field carries as a matter of course, at margins
narrower than the one adopted here.

## The Two Dials Are Separable

`high` matmul precision rounds a float32 matmul's inputs to TF32 and accumulates
in float32; parameters, activations, and gradients stay float32 throughout.
`bfloat16-mixed` autocasts the forward pass while master weights and optimizer
state stay float32. They were measured separately so that the cheaper one could
be adopted alone if the other read badly.

TF32 is worth +29% at width 256 and costs nothing in memory. There is no argument
against it that does not apply to the bfloat16 dial with more force, so it would
have shipped on either way.

## What Was Measured

One idle RTX 4090 against the widened corpus, micro-batch 4 with four
accumulation steps for an effective batch of 16, 400 steps per point with a
200-step steady-state window. Throughput is `training.active_positions_per_second`
and memory is peak reserved. The micro-batch is held at 4 across every row, so a
row differs from its neighbours in the dials alone.

Width 256:

| precision | matmul | compiled | positions/s | peak reserved |
| --- | --- | --- | ---: | ---: |
| float32 | highest | off | 8,083 | 14.58 GB |
| float32 | high | off | 10,423 | 14.58 GB |
| bfloat16-mixed | high | off | 12,506 | 10.89 GB |
| float32 | highest | on | 9,334 | 13.85 GB |
| bfloat16-mixed | high | on | 18,435 | 9.23 GB |

Width 512:

| precision | matmul | compiled | positions/s | peak reserved |
| --- | --- | --- | ---: | ---: |
| float32 | highest | off | 3,521 | 24.39 GB |
| float32 | high | off | 4,488 | 24.39 GB |
| bfloat16-mixed | high | off | 6,906 | 21.87 GB |
| float32 | highest | on | 3,878 | 24.50 GB |
| float32 | high | on | 5,077 | 24.55 GB |
| bfloat16-mixed | high | on | 9,117 | 20.32 GB |

**TF32 alone is worth +29% at width 256 and +27% at 512, and costs nothing in
memory.** Eager float32 reserves 14.58 GB at width 256 and 24.39 GB at 512
whether the matmul rounds or not, to the byte. Compiled, the dial reads +31% at
width 512.

**The pair is worth 2.35x at width 512 and 1.98x at 256**, compiled on both
sides. Read against the setting the vehicle would otherwise have frozen at,
eager float32 with `highest`, the shipped configuration is 2.28x at width 256
and 2.59x at 512.

**At width 512 every float32 configuration sits against the ceiling.** The
runtime reports 25.24 GB usable on this card. The four float32 rows reserve
between 96.6% and 97.3% of that, leaving 690 MB at worst; bfloat16 with
compilation reserves 80.5% and leaves 4.9 GB. At the target width the dial is
deciding fit before it decides speed, which is not true at 256, where `0071`'s
saturation finding holds and the returned memory converts into nothing.

**The `512 / float32 / highest / on` row is a correction.** `0073` recorded 3,493
for it and read the row as no gain at all. That figure came from a run whose
record on disk carries no efficiency measurement and no checkpoint pointer, and
whose memory figure is the width-256 one. Re-measured cleanly here it is 3,878,
which is +10% rather than nothing. `0073`'s conclusion is unaffected, since
+10% at float32 against +32% at bfloat16 is still the reading that would have
rejected compilation if taken at float32 alone, but the number itself was
wrong.

## Consequences

**Every run in flight stops resuming.** Both keys sit in the runner's execution
compatibility set and inside the training-configuration digest, so a run started
before this record and resumed after it is refused with the two values named. A
run that wants to finish under the arithmetic it started in declares the old pair
explicitly; there is no path that continues one set of weights across the change,
and that is the point of the keys rather than a cost of this record.
`CHECKPOINT_VERSION` does not move, because no key was added and no stored
checkpoint changed shape.

**bfloat16 autocast is refused on a CUDA device that cannot run it**, before the
run directory exists rather than at the first forward pass. Torch raises on the
same condition several hundred lines later, by which point an incomplete run
record is already on disk.

**Full precision stays selectable and stays exercised.** The CPU and MPS smoke
configurations and the runner test helper pin both dials, because the readings
here were taken on CUDA and `high` changes what a float32 matmul does on a CPU
too, on whatever hardware happens to run the suite. The CUDA smoke takes the
defaults, so one checked-in configuration proves the shipped arithmetic on a new
host.

**A mixed-precision checkpoint had to become loadable for any of this to mean
anything.** The inference runner gated on the run's `precision` alongside its
`parameter_dtype`, which reads the training autocast setting as though it were a
property of the stored weights. It is not: master weights stay float32. Every
bfloat16 run this project had taken reported as unloadable, and the vehicle would
have frozen on a precision whose checkpoints nothing here could score or play.

**The next revisit needs a floor, not a rerun.** Once `#488` stores the vehicle's
seed dispersion, a float32 arm against the vehicle is readable in the ordinary
way, and that is the only cheap route back to this question. Nothing schedules
it, and nothing should: the throughput this buys is worth more than the answer
unless something else gives a reason to look.
## The Numerics Check

This asks whether reduced-precision arithmetic tracks float32's on this model. It
is a question about the arithmetic rather than about the weights, and it is not
evidence either way about whether the model is better or worse. Nothing here
should be read as though it were.

Width 256, micro-batch 4 with four accumulation steps, 20,000 optimizer steps
against the widened corpus, strict determinism, compilation on, one data order
throughout. `move_loss` is logged every step and read as a 200-step rolling mean,
because one step's loss is one micro-batch of 16 sequences and is far noisier
than anything being compared.

**Two controls establish that a divergence means something.** A second float32
arm at the same seed, run afterwards on the same card, agrees with the first at
every one of 20,000 steps to the last digit logged. A third, on the other card of
the pair these arms were split across, agrees the same way over 2,000 steps. So
the pipeline is exactly reproducible with compilation on, and the two cards
compute alike.

Neither had been established. The configuration that claims a repeated run
reaches the same weights turns compilation off, and the compiler benchmarks some
choices as it builds, so run-to-run variation was a live possibility rather than
a paranoid one. It is also the check `#488` asks for before any seed is varied.

**A seed change moves the curve further than either dial does.** The fourth arm
changes only the run seed, which is initialization and dropout; the data order is
held fixed by a separate seed, so this is a lower bound on seed dispersion rather
than a measurement of it.

| arm | smoothed loss at 20,000 | largest smoothed gap from the control |
| --- | ---: | ---: |
| float32, seed 17 | 1.605014 | control |
| float32 repeated, seed 17 | 1.605014 | 0, bit-identical |
| float32 with TF32, seed 17 | 1.603368 | 0.034 at step 228 |
| bfloat16 with TF32, seed 17 | 1.602056 | 0.066 at step 296 |
| float32, seed 29 | 1.606803 | 0.095 at step 424 |

Every largest gap falls inside the first five hundred steps, where the loss is
descending steeply and the arms have barely separated. By step 5,000 the two
precision arms sit within 0.0005 of the control.

**Both dials perturb the curve less than the weaker of the two seed draws, and
they are ordered by how much rounding each does.** TF32 rounds matmul inputs to
ten mantissa bits and moves the curve half as far as bfloat16, which autocasts to
seven. At the end of the horizon the bfloat16 arm reads 0.0030 below the control
and the second float32 seed reads 0.0018 above it, so the precision difference and
a single seed draw are the same size to within a factor of two.

**Training health agrees.** Gradient clipping never fires on any arm. The
update-to-weight ratio spans 0.00174 to 0.00176 across all five. Mean gradient
norm over the last thousand steps is 0.546 for both float32 arms at seed 17, 0.547
with TF32, 0.540 at seed 29, and 0.526 with bfloat16, which is the one reading
that falls outside the range the two float32 seeds span.

**What this does not say.** It does not say bfloat16 trains as good a model, and
the arm reading lower at 20,000 steps is not evidence that it trains a better one:
two trajectories that separate will differ, and one of them has to be lower. Two
seeds are not a dispersion. What it does say is that the arithmetic is sound, that
nothing diverges, overflows, or destabilizes, and that the perturbation this
change introduces is smaller than one already present in every comparison this
project takes.
