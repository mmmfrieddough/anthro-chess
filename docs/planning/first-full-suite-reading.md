# First Full Suite Reading

This document records the first time the evaluation harness was pointed at real
checkpoints end to end and its output read. It is evidence about the
*instrument*, not a claim about the model. Every number here is pinned to the
corpus, action vocabulary, and checkpoints that produced it, and to one machine.
They will go stale, which is normal and is what the milestone-1 proof already
documents about its own figures.

The reading was taken on a reduced sweep. What that costs in interpretation is
stated in "What this reading cannot settle".

## What Was Read

Two checkpoints far apart in one training run, `training-blitz-30k-v4`:

| | step 100 | step 8000 |
| --- | --- | --- |
| training | 100 optimizer steps | 8,000 optimizer steps |

Both were read with:

```console
anthro eval suite --config configs/evaluation/checkpoint-suite.toml \
  --set 'model.checkpoint_path="<run>/checkpoints/step-<step>.pt"' \
  --no-record
```

Apple Silicon MPS, float32, seven steps per checkpoint. The held-out records
were taken separately and committed; see "Committed records".

Cost: **244.9 s and 256.2 s** — 8.4 minutes for both checkpoints.

## Directions Stated Before The Run

Recorded before any output was inspected:

- held-out move loss lower at step 8000; accuracy and legal mass higher;
- puzzles show a stronger rating response at step 8000;
- rollout and termination read closer to the human reference at step 8000;
- inference is unchanged — same architecture, so it is the control. If it
  moves, that is a finding about the instrument rather than the model.

## What The Readings Said

Held-out scoring over 400 pool games, 28,668 positions:

| metric | step 100 | step 8000 | expected | agreed |
| --- | --- | --- | --- | --- |
| `move_loss` | 6.240095 | 3.632417 | lower | yes |
| `legal_move_loss` | 3.586189 | 2.890620 | lower | yes |
| `top1_accuracy` | 0.077055 | 0.239431 | higher | yes |
| `top5_accuracy` | 0.322415 | 0.567183 | higher | yes |
| `mask_penalty` | 2.653906 | 0.741797 | lower | yes |
| `top1_illegal_rate` | 0.297649 | 0.230082 | lower | yes |
| `uniform_over_legal` | 3.262018 | 3.262018 | invariant | yes |

`uniform_over_legal` is a property of the pool rather than the model, and it is
bit-identical across both checkpoints. That is the strongest available evidence
that the scoring path is reading the data it claims to read.

At step 100 `move_loss` (6.24) exceeds `uniform_over_legal` (3.26): the model is
measurably worse than sampling uniformly among legal moves. That is the correct
reading for a checkpoint 100 steps into training and is a useful anchor for what
the low end of this metric looks like.

Rating dependency moved the way a model learning to use its conditioning should:

| quantity | step 100 | step 8000 |
| --- | --- | --- |
| `absent` degradation | +0.315239 | +0.824292 |
| `constant` degradation | −0.000316 | +0.001730 |
| `shuffled` degradation | −0.000050 | +0.002459 |
| cross-conditioning match rate | 0.250000 | 1.000000 |
| anchor policy divergence | 0.000658 | 0.019841 |
| anchor top-1 agreement | 0.982594 | 0.863018 |

Novelty dose response only acquires its shape once there is something to lose.
At step 100 retention is flat to slightly rising across the dose grid; at step
8000 it declines monotonically to 0.9727 at full dose, with legal mass at dose
zero rising from 0.1092 to 0.6608.

Inference, the control, did not move: batch-one p50 at 40 plies of history read
5.7 ms and 5.4 ms. Both are within the run-to-run spread of the benchmark.

## What This Found

### The reading surface does not say when a reading is uninterpretable

This is the substantive finding, and it appeared three times through three
different mechanisms. In two of them a careful reader with the source available
drew a wrong conclusion while taking this reading.

**Puzzles report no resolution.** `conservative_detectable_difference` exists in
`anthro_chess.evaluation.puzzles.dataset` and is applied when the artifact is
built, but nothing in the command's output reports it, and the puzzle benchmark
computes no noise floor of any kind. The rating-response spread observed here is
about 0.5 percentage points against a detectable difference of roughly 1.4 at
20,000 puzzles — that is, below resolution. Nothing in the output says so.

**The rollout table shows a floor the verdict is not computed against.** The
`floor` column renders the *conditional* floor, while the mismatch verdict
compares the pooled distance against a *pooled* floor that never appears. At
step 8000 `game-length` reads pooled 53.4575 against a displayed floor of
80.5785 and still reads `mismatch`, which is correct and looks self-contradictory.

**A saturated metric reads like a measurement.** The generated termination arm
returns `conditional 0.5855 ... reads as mismatch` identically at both
checkpoints. That is a real property of both models rather than a fault, but the
output presents it in the same form as a discriminating reading.

Several floors also render as exactly `0.0000`, against which any distance reads
as a mismatch. Whether those are genuine absences of sampling variation or
floors that could not be computed is not distinguishable from the output.

Filed as #172 (the floor column), #173 (the missing puzzle resolution), and #175
(a saturated arm reading like a measurement, and the zero floors).

### Puzzle rating response reads inverted

Greedy slope −0.0019 at step 100 and −0.0135 at step 8000, with ordering 0.417
and 0.000: a higher configured rating produces *worse* puzzle solving, more so
after training. Overall puzzle skill improved as expected, with the fitted
rating rising from 247.6 to 493.2.

The effect is below the benchmark's own resolution, so this is a question rather
than a defect. The most likely explanation is visible elsewhere in the same
sweep: the model's competence is concentrated in the opening, where
`mask_penalty` reads 0.190006 against 0.949785 in the middlegame and 1.255727 in
the endgame, and where the rollout `repertoire` quantity appears at step 8000
having been absent at step 100. Puzzles are middlegame and endgame tactics, so
rating conditioning learned largely on opening choice has little to transfer.

A second mechanism is worth keeping in view: puzzles are selected precisely
because a human blundered and a forcing continuation exists, so a model trained
to predict the human-plausible move is being scored on the positions where
human-plausible and correct diverge most.

Filed as #174, to be re-read on a checkpoint strong enough for the effect to
clear the benchmark's resolution.

### The generated termination arm saturates

Both checkpoints resign in 24 of 24 generated games. This follows from the
models' own probabilities rather than from any judgment about the position:
held-out resignation mass at ordinary move plies is 0.00677 at step 100 and
0.00803 at step 8000, which over a 300-ply limit implies roughly 87% and 91% of
games resigning from that baseline alone.

The arm should begin to discriminate once the model learns to suppress
resignation. The held-out resignation arm already discriminates, with separation
rising from 0.00129 to 0.01029.

Filed as #175. The saturation itself resolves as the model improves; what is
worth fixing is that the output does not say the reading is saturated.

## What This Reading Cannot Settle

**The ladder was not read.** It is excluded from the reduced sweep because its
cost is a seat grid rather than a sample size, and it has never run at its
declared size. A full run on both checkpoints was started when this document was
written; if it completes, this section is amended with the result.

**The generated-play side is thin.** Rollout and termination read 24 to 36 games
at one seed. Both open findings above are on that side, which is where more
games would most change the reading.

**Reduced views widen every floor.** A real regression smaller than a reduced
view can resolve would not appear here. This reading orients; it does not
certify.

## Committed Records

Two `anthro eval run` records are committed, one per checkpoint, at the reduced
400-game view rather than the canonical pool view. They exist to demonstrate
that metric movement shows up as a reviewable Git diff, which had never been
exercised with real data.

They are **pre-core** records. Nothing is protected before the evaluation core is
designated, so they are deleted at #90 at the cost of one clean diff. See
`docs/decisions/0013-benchmark-result-comparability.md`.

## Cost, And Why These Numbers Are Not The Earlier Ones

This reading was taken after the device-synchronization fix in #166. Any earlier
cost figure for this harness is not comparable. On the same checkpoint and
machine, batch-8 inference throughput moved from 93 to 3,018 decisions per
second and batch-one latency at 40 plies from 27.6 ms to 5.5 ms, and the
throughput curve straightened from 58/85/93/98 across batch sizes 1 through 16
to 493/1,258/3,018/5,056.

The practical consequence for planning is that generated-game cost fell from
roughly 15–18 seconds per game to about 1.8, which is what makes a full ladder
an overnight run rather than a multi-day one.
