# Checked-In Configurations

This directory holds reproducible configuration selections for commands that
have become real. Command-specific Pydantic models own schema fields,
validation, and defaults; the shared loading boundary lives under
`anthro_chess.config`. TOML files here select values but do not define the
schema.

`data/` contains the offline sample selection and the pinned, bounded Lichess
baseline-corpus selection. The latter owns the archive identity, published
checksum, rating namespace, deterministic selection bound, and normalized shard
size. `training/` contains strict CPU and explicitly selected MPS smoke paths
against the prepared sample artifact, including step-keyed checkpoint and
resume coverage. The MPS selection enables synchronized phase profiling for
bounded device verification; larger-corpus batch and accumulation choices
belong in resolved run configuration and measured artifacts. Add focused
`evaluation/` or `runtime/` subdirectories when the corresponding command
exists. Do not add speculative example files for commands that have not been
implemented.

Commands should use their code-owned defaults unless an explicit configuration
path is supplied. They must not search the current working directory or infer a
repository checkout. Command-line overrides should use the shared strict loader
and reject unknown fields. Runs and artifacts should store the loader's resolved
configuration record alongside code, model, encoding, and data-provenance
versions that are relevant to that artifact.
