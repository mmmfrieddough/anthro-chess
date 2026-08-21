# Checked-In Configurations

This directory holds reproducible configuration selections for commands that
have become real. Command-specific Pydantic models own schema fields,
validation, and defaults; the shared loading boundary lives under
`anthro_chess.config`. TOML files here select values but do not define the
schema.

`data/` contains the offline sample selection and the pinned Lichess
universal-export corpus selection. The latter owns the archive identities,
their published checksums, the rating namespace prefix, the split seed, and
the normalized shard size.
It can also name a marked-account snapshot, though it does not yet;
`data/marked-accounts/` explains why that one input is checked in rather than
regenerated on demand, since account status is a live judgement and a corpus
that re-asked for it would shrink on every run.
`training/` contains the strict CPU and explicitly selected MPS and CUDA smoke
paths against the prepared sample artifact, including step-keyed checkpoint and
resume coverage. The accelerator smoke selections enable synchronized phase
profiling for bounded device verification; larger-corpus batch and accumulation
choices belong in resolved run configuration and measured artifacts.

The two accelerator smoke selections differ in their determinism setting, and
the difference belongs to the backend rather than to taste: the CUDA backward
pass has a deterministic implementation for every operation this model needs
and the MPS one does not. A selection asking for strict determinism where the
backend cannot supply it fails before the first optimizer step.

They differ in precision for the same kind of reason. The CUDA selection takes
the shipped defaults, so it proves the arithmetic a real run takes; the CPU and
MPS ones pin both dials at full precision, because the readings that chose those
defaults were taken on CUDA and neither dial has been measured on a backend this
project does not train on.

A training selection may also declare in-training evaluation cadences: when an
entry runs, which registered metrics it computes, and the explicitly sized
validation subsample it computes them over. The schedule is resolved before the
first optimizer step, so an unaffordable or impossible entry fails there rather
than mid-run. The canonical end-of-run reading over the frozen test pool stays a
separate command.

`evaluation/` contains frozen evaluation-pool selections for `anthro eval
freeze`. A pool selection names the normalized corpus it draws from and the
split it freezes, and it should record the identity digest printed by the first
successful build so a rebuild elsewhere is verified rather than assumed. A
mismatch afterwards means the corpus, its filters, or its split seed moved, and
the benchmark needs a new pool version rather than a quietly different pool.

A pool selection may also name one of the `data/marked-accounts/` snapshots,
relative to the selection naming it as a corpus selection does. The cut then
leaves out every game a listed account played and records the recall that
snapshot had reached, which is where a corpus prepared without one applies the
rejection.

A selection that *reads* a pool records that digest too, because the checks a
materialized pool answers on its own ask whether it is intact and readable by
this code, and every generation of it is. Without the digest a superseded pool
left at the configured path keeps scoring, labelled as itself and comparable to
nothing the selection has moved on to. A selection recording none loads whatever
the path holds, which is what a pool with no designated generation needs.

It also contains the checkpoint-evaluation selection for `anthro eval run`,
which names the pool and the view to score without naming a checkpoint. Leaving
the model selection out keeps the canonical reading defined by its inputs
rather than by whichever checkpoint was current when the file was written; the
machine-local default selection or an explicit override supplies the rest.
The puzzle-rating selection for `anthro eval puzzles` likewise leaves out the
checkpoint, declares the rating grid and reference temperature, and names both
the generated puzzle artifact and the normalized training selection used only
for the source-game overlap report. Its companion puzzle-set selection pins the
source archive, exact-rating sampling recipe, statistical size target, quality
filters, and expected generated identity for `anthro eval prepare-puzzles`. The
generated puzzle artifact remains under `ANTHRO_CHESS_DATA_ROOT`; the rows it is
built from are vendored beside the puzzles package, and the build refuses when
they and this pin disagree.

It also contains the generated-play rollout selection for `anthro eval rollout`,
which declares the matrix a suite plays rather than naming a checkpoint. Values
that decide what is measured — the arms, the rating and temperature grids, the
ply limit, the color and draw-claim settings — are part of series identity, so
changing one starts a new series. The seed list, games per position, resample
count, and concurrency are sample and throughput settings and can be raised
freely.

The same selection names the pool the human reference is read from. That
comparison is what turns the rollout scalars into a statement about
human-likeness, so it is on by default and a suite without a pool has to say so
explicitly. The curve bandwidth is deliberately not configurable: it is selected
once from the corpus with `anthro eval curve-bandwidth` and frozen in code,
because re-selecting it per run would measure two checkpoints differently.

Add a focused `runtime/` subdirectory when the corresponding command exists. Do
not add speculative example files for commands that have not been implemented.

Commands should use their code-owned defaults unless an explicit configuration
path is supplied. They must not search the current working directory or infer a
repository checkout. Command-line overrides should use the shared strict loader
and reject unknown fields. Runs and artifacts should store the loader's resolved
configuration record alongside code, model, encoding, and data-provenance
versions that are relevant to that artifact.

When configured, `ANTHRO_CHESS_DATA_ROOT` and `ANTHRO_CHESS_RUN_ROOT` relocate
checked-in relative artifact paths outside the worktree. Explicit absolute
paths and strict command-line path overrides remain authoritative, and run
records retain the resolved paths.
