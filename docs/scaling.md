# Scaling

This document owns how the project grows the model: what is decided before
anything is measured, what order the remaining decisions are taken in, and which
of them a later change forces back open.

`docs/evaluation.md` owns what a reading is. `docs/issue-workflow.md` owns when a
model change owes one and how a session without the hardware routes it.
`docs/training-and-runtime.md` owns training and runtime behavior. This document
owns the program those three serve during scaling, and nothing else.

The rules below come from outside work rather than from measurement here.
`docs/research.md` (Scaling And Capacity) carries the sources and what each one
does and does not establish.

## The Reference Frame Is Fixed Before Readings Accumulate

Two kinds of decision run through this program, and confusing them is what makes
the order feel arbitrary.

**A reference-frame decision is one whose change invalidates readings already
taken.** The corpus, the evaluation core, the ablation vehicle, and the schedule
family are all of this kind. Being wrong about one is expensive not because the
item is expensive to redo but because everything measured against it is.

**A reading answers a question and invalidates nothing.** How the model scales,
whether a candidate change helps, where the loss plateaus: each is a finding, and
findings do not spoil one another.

The frame is fixed first, and the test is that single question — *does changing
this invalidate readings already taken?* It is the same rule
`docs/decisions/0013-benchmark-result-comparability.md` applies to the evaluation
core, generalized to everything else a reading is expressed in terms of.

## Record A Setting As A Rule, Not A Value

**A hyperparameter that depends on scale is recorded as the rule that produces
it, not as the number the rule produced.** A value is correct at one point and
silently wrong everywhere else, and no goodness-of-fit statistic detects that.

This is the difference between a project where a scale change is arithmetic and
one where it is a fresh sweep. A learning rate stored as a number must be
re-tuned when the model widens; stored as a function of width it recomputes.
Weight decay stored as a coefficient must be re-tuned when the horizon moves;
stored as the optimizer timescale it does not.

Three tiers, and they behave differently:

- **Derived rules terminate.** Where the rule follows from the optimizer's
  algebra or from a width limit rather than from a fit, there is nothing above it
  left to tune.
- **Fitted rules move up one level and mostly stop.** A fitted rule is valid over
  the range it was fitted on, so the trigger to revisit it is leaving that range —
  a checkable condition, not a standing worry. Record the range beside the rule
  and refuse to evaluate outside it.
- **Bare values recur.** Every scale change reopens them.

The regress does not terminate because the tiers run out. It terminates because
the loss surface is flat enough near its optimum that the remaining error stops
being measurable, which is what the next section is about.

## Model Size Is Derived, Not Tuned

**The target model size follows from the compute budget and the deployment
envelope. It is not an experimental result and no experiment is run to choose
it.**

Compute is an input: the hardware, multiplied by the wall-clock the project will
spend, multiplied by realized utilization. `docs/vision.md` bounds the second
term — iterations in days, a final run in weeks, on high-end consumer hardware.
Total training compute is approximately six times the parameter count times the
number of training positions processed, which leaves the ratio between those two
as the only free quantity once the budget is fixed. That ratio is what a ladder
measures. The absolute size is arithmetic.

Two consequences worth stating because they are counter-intuitive:

**Positions processed and training steps are one axis, not two.** Steps times
batch size is the number of positions the run sees. Unique corpus size is a
genuinely separate quantity, and it matters only as the ceiling past which data
is being repeated.

**Being wrong about the size is cheap.** The loss surface around the
compute-optimal point is flat: outside work puts a factor of roughly 1.5 within a
few percent of optimal compute, which is below what any reading here can resolve.
A size within that band is not re-litigated. A size wrong by a factor of four is
a real error and is fixed.

This project serves far more inference than it spends training, and its serving
constraint is loose — a model this size answers in milliseconds. That argues for
the smaller-and-longer end of the band rather than the compute-optimal point,
which is chosen for a run that is never served.

## The Ablation Vehicle

**One frozen training configuration is the instrument every candidate change is
read against.** It is designated once, its identity digest is pinned by a test,
and its seed dispersion is characterized against that digest.
`docs/decisions/0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md`
owns why it exists and what it gives up.

Its size is derived from the target rather than chosen: far enough below the
target that an arm is cheap enough to spend on a question that may return
nothing, close enough that its readings sit in the regime the target occupies.
The binding criterion in practice is that an arm finishes fast enough to run
several in a day.

**Adopting a change does not advance the vehicle.** Promotions go to the
canonical line, which is what
`docs/decisions/0063-the-full-sweep-decides-a-change-and-the-canonical-line-is-its-byproduct.md`
describes. A vehicle that no success redefines is one whose floor stays current.

Because comparisons against a frozen base yield main effects and no interaction
terms, a set of individually accepted changes is run together as one further arm
before any of them reaches the canonical line.

## What A Later Change Forces Back Open

The table states which decisions a change invalidates. It is the reason the
program has an order at all: never settle a row before the column that would
undo it.

- **Hard** — the earlier decision is now wrong rather than merely stale, and any
  comparison read across the change is confounded.
- **Recompute** — mechanical if the setting was recorded as a rule; a fresh sweep
  if it was recorded as a value.
- **Soft** — survives within the ranges outside work has tested. Spot-check; do
  not re-tune.
- **Open** — the sources disagree on whether it fires.

| Decision | Width | Depth | Batch | Horizon | Positions per parameter | Corpus or selection | Architecture | Optimizer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Action vocabulary and board encoding | recompute | — | — | soft | soft | **hard** | — | — |
| Parametrization | — | **hard** | soft | open | — | soft | **hard** | **hard** |
| Stability settings | recompute | recompute | soft | soft | — | soft | **hard** | **hard** |
| Optimizer choice | recompute | — | — | recompute | recompute | — | soft | — |
| Model size | — | — | — | recompute | **hard** | open | recompute | — |
| **Peak learning rate** | **hard** | **hard** | **hard** | **hard** | soft | soft | **hard** | **hard** |
| Batch size | soft | — | — | **hard** | soft | soft | — | soft |
| Adam second-moment decay | — | — | **hard** | soft | — | — | — | **hard** |
| Weight decay | recompute | — | recompute | recompute | **hard** | — | — | **hard** |
| Warmup length | recompute | — | recompute | recompute | — | — | — | soft |
| Schedule — matched-horizon decay | — | — | — | **hard** | — | — | — | — |
| Schedule — constant trunk with cooldown | — | — | — | soft | — | — | — | — |
| Selection filters | — | — | — | **hard** | **hard** | **hard** | — | — |
| Allocation rule | — | — | — | — | **hard** | open | **hard** | soft |

Three rows carry most of the weight.

**Peak learning rate is the most coupled quantity in the system**, hard against
six of eight columns. This is why a candidate change compared against a baseline
whose learning rate suits the baseline and not the candidate measures the tuning
rather than the change, and why that is the most common way a comparison here
produces a confident wrong answer.

**The schedule family decides whether a horizon change is a branch or a
restart.** A decay schedule shaped to a step count is invalidated by changing
that step count, so every horizon question costs a fresh run. A constant trunk
cooled down at the end is not, so one trunk yields several horizons and "train
longer" stays cheap for the life of the project. This is the single
highest-leverage entry in the table and it is nearly free to get right before the
vehicle is frozen.

**The repetition row is absent because it does not apply.** At the corpus this
project is building, every model size in the plausible range trains on well under
one pass, so nothing here repeats data and the outside work on repeated-data
scaling does not bear. Revisit only if a selection filter cuts the pool far
enough that the horizon exceeds it.

## What Transfers From The Vehicle To The Target

A vehicle-scale result is evidence about the target, not a measurement of it.
What transfers depends on the kind of effect:

- **A constant offset** — the change lowers the curve by roughly the same amount
  at every scale. Most architecture and loss changes are this, and one scale is
  enough to rank them.
- **A change in slope** — the benefit grows or shrinks with scale. Rare, far more
  valuable, and not distinguishable from an offset at a single scale. Establishing
  which one a candidate is costs arms at three or more sizes, and is worth it only
  for a change the project would restructure around.
- **A stability effect** — visible at small scale and better measured there, by
  widening the learning-rate range rather than the model. Judge these on the
  training-health readings rather than on loss, since they often improve neither
  loss nor any benchmark at the scale they are measured on.

A candidate that reads negative at vehicle scale is discarded only where it was
given its own learning rate. Otherwise the reading may be about the tuning.

Rank a candidate by how much additional compute the baseline would need to match
it, at a stated scale, rather than by the raw delta. A candidate whose answer is
smaller than the compute wasted by an ordinary sizing error does not justify
delaying a run.

## The Order

Each step is entered only when the one above it has an answer, and every exit is
a reading or a recorded decision rather than a judgement that enough was done.

1. **Target scale.** Derived from budget and envelope. A written decision, not an
   experiment.
2. **Schedule family.** Free to decide, expensive to change after the vehicle is
   frozen.
3. **Vehicle designation and digest pin.** Requires a corpus and a pool to read
   against, so it follows the breadth and generation work.
4. **Seed dispersion of the vehicle.** The denominator every later comparison is
   read against.
5. **Hyperparameter rules across scale.** Fitted, and validated at one size not
   used in the fit.
6. **The allocation ladder.** Several small sizes spanning one to two decades,
   yielding the size-versus-data rule and, with it, the target's data budget.
7. **Candidate changes, one arm each against the vehicle**, then the accepted set
   as one further arm.
8. **Confirmation**, at the target size but a fraction of its horizon, which holds
   the size term fixed and removes the hardest extrapolation.
9. **The run.**

Steps 1 through 4 are reference-frame work and belong before any reading
accumulates. Steps 5 onward are readings.

## What Remains Open

**Whether human-likeness saturates at a different size than strength.** Outside
work anchors what model sizes reach what playing strength on human games; nothing
found says where the accuracy of a rating-conditioned move distribution stops
improving. This project's target is the latter, so the anchor bounds the question
without answering it, and the ladder is what answers it.

**Whether a candidate's benefit holds at the target.** Nothing here measures the
rate at which vehicle-scale rankings survive a size gap, and no outside source
found publishes it for a gap this large either. Treat every vehicle-scale
adoption as provisional until the confirmation run.

**Whether state-frequency skew costs capacity.** Human game archives are heavily
concentrated in openings, and outside work reports capacity flowing to frequent
positions at the expense of rarer decisive ones in some board games. Whether that
happens here is unmeasured.
