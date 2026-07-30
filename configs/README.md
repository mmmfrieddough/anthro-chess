# Checked-In Configurations

This directory holds reproducible configuration selections for commands that
have become real. Command-specific Pydantic models own schema fields,
validation, and defaults; the shared loading boundary lives under
`anthro_chess.config`. TOML files here select values but do not define the
schema.

`data/` contains the offline sample selection and the pinned, bounded Lichess
baseline-corpus selection. The latter owns the archive identity, published
checksum, rating namespace, deterministic maximum, and normalized shard size.
`training/` contains the strict CPU and explicitly selected MPS smoke paths
against the prepared sample artifact, including step-keyed checkpoint and
resume coverage. It also contains the first measured many-game MPS proof
selection; `docs/planning/minimal-training-proof.md` owns the reproducible
corpus slice, evidence, and interpretation. The MPS smoke selection enables
synchronized phase profiling for bounded device verification; larger-corpus
batch and accumulation choices belong in resolved run configuration and
measured artifacts.

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
filters, and expected generated identity for `anthro eval prepare-puzzles`.
Puzzle records remain under `ANTHRO_CHESS_DATA_ROOT`; only the recipe and
expected identity are committed.

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
