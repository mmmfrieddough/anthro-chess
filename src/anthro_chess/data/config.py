"""Strict configuration for data preparation and sequence loading."""

from typing import Literal

from pydantic import Field, StrictBool

from anthro_chess.config import ConfigModel


class SourceConfig(ConfigModel):
    """Identity and rating semantics for one source selection."""

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: str = Field(min_length=1)
    url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    rating_namespace: str | None = None
    rating_system: str | None = None
    ratings_are_normalized: StrictBool = False


class SplitConfig(ConfigModel):
    """Deterministic game-level train/validation split selection."""

    seed: str = Field(default="anthro-sample-v1", min_length=1)
    validation_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)


class FilterConfig(ConfigModel):
    """Filters for the initial standard-human-game corpus."""

    minimum_plies: int = Field(default=1, ge=1)
    require_rated: StrictBool = True
    exclude_bots: StrictBool = True


class PrepareConfig(ConfigModel):
    """Code-owned schema for ``anthro data prepare``."""

    source: SourceConfig
    split: SplitConfig = SplitConfig()
    filters: FilterConfig = FilterConfig()


class SequenceLoaderConfig(ConfigModel):
    """Deterministic batching choices for normalized game sequences."""

    split: Literal["train", "validation"] = "train"
    batch_size: int = Field(default=8, ge=1)
    length_bucket_width: int | None = Field(default=32, ge=1)
    chunk_length: int | None = Field(default=None, ge=1)
    shuffle: StrictBool = True
    seed: str = Field(default="anthro-sequence-loader-v1", min_length=1)
    drop_last: StrictBool = False
