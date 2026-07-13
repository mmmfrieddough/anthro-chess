# Checked-In Configurations

This directory holds reproducible configuration selections for commands that
have become real. Schema fields, validation, and defaults live in Python under
`anthro_chess.config` and the command-specific package; TOML files here select
values but do not define the schema.

Add focused `training/`, `evaluation/`, or `runtime/` subdirectories when the
corresponding command exists. Do not add speculative example files for commands
that have not been implemented.

Commands should use their code-owned defaults unless an explicit configuration
path is supplied. They must not search the current working directory or infer a
repository checkout. Command-line overrides should use the shared strict loader
and reject unknown fields. Runs and artifacts should store the loader's resolved
configuration record alongside code, model, encoding, and data-provenance
versions that are relevant to that artifact.
