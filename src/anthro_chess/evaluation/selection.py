"""Which checkpoint a benchmark measures, and what its readings call it.

Every benchmark selection names one checkpoint and, optionally, the label its
readings are recorded under. That pair was written out in all seven benchmark
schemas and in the suite's, identically, so the convention that a selection
names its checkpoint *here* — at the top level, under ``model`` — was held by
repetition rather than by a type.

That mattered beyond the duplication. The sweep replaces a benchmark's
selection through ``model_copy(update={"model": ...})``, which pydantic does
not validate, so a schema that named the field anything else would have taken
the update as a stray attribute and quietly measured its own checkpoint
instead of the sweep's. Inheriting the field is what makes that a type error.

Inherited rather than nested, because a configuration schema's shape is its
file's shape: a field typed as another model is a table in the TOML, so
composing this in would move ``model`` and ``checkpoint_label`` inside a new
one and rewrite every shipped selection and every documented override.
"""

from __future__ import annotations

from pydantic import Field

from anthro_chess.config import ConfigModel
from anthro_chess.inference.config import ModelRunnerConfig


class CheckpointSelection(ConfigModel):
    """The checkpoint a selection measures, and the label it is recorded as."""

    #: Which checkpoint to load. A sweep replaces this wholesale, so a
    #: benchmark file that pinned its own run measures the sweep's checkpoint
    #: rather than quietly measuring something else than the rest of it.
    model: ModelRunnerConfig = ModelRunnerConfig()
    #: What a reading calls the checkpoint. Left unset, a reading derives the
    #: label from the run and the step it was taken at.
    checkpoint_label: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )


__all__ = ["CheckpointSelection"]
