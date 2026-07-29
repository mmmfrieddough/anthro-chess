"""Strict configuration for data acquisition, preparation, and sequence loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, StrictBool, model_validator

from anthro_chess.config import ConfigModel
from anthro_chess.data.schema import SplitName


class SourceConfig(ConfigModel):
    """Identity and rating semantics for one source selection."""

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: str = Field(min_length=1)
    url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    rating_namespace: str | None = None
    rating_system: str | None = None
    ratings_are_normalized: StrictBool = False


class ArchiveConfig(ConfigModel):
    """Pinned downloadable archive used by a reproducible source selection."""

    artifact_name: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    url: str = Field(min_length=1, pattern=r"^https?://")
    file_name: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    compression: Literal["zstd"] = "zstd"


class SplitConfig(ConfigModel):
    """Deterministic game-level train/validation/test split selection.

    Assignment is a pure function of ``seed`` and the internal game id, so
    growing or refiltering a corpus never moves an existing game between
    splits. Changing ``seed`` breaks that guarantee and can place a previously
    held-out test game into training; treat it as frozen once a benchmark
    pool has been built from a selection.
    """

    seed: str = Field(default="anthro-sample-v1", min_length=1)
    validation_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    require_nonempty: StrictBool = False

    @model_validator(mode="after")
    def _validate_fractions(self) -> SplitConfig:
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                "validation and test fractions must leave a nonempty train split"
            )
        return self


class FilterConfig(ConfigModel):
    """Filters for the initial standard-human-game corpus."""

    minimum_plies: int = Field(default=1, ge=1)
    require_rated: StrictBool = True
    require_ratings: StrictBool = False
    exclude_bots: StrictBool = True
    event_speed: (
        Literal[
            "bullet",
            "blitz",
            "rapid",
            "classical",
            "correspondence",
        ]
        | None
    ) = None
    maximum_games: int | None = Field(default=None, ge=1)


class TerminationConfig(ConfigModel):
    """Thresholds for deriving how a game ended.

    Sources that collapse clock expiry and player abandonment into one
    termination value leave the clock trace as the only separating evidence.
    A losing player still holding this share of their initial time at their
    last move is treated as having walked away rather than flagged. The
    default comes from the measured arena sample in
    ``0017-derived-termination-and-terminal-actions.md``, but the split
    between the two populations depends on the source and time control, so
    preparation reports how much of the population the threshold judged
    instead of presenting the value as settled.
    """

    abandonment_clock_share: float = Field(default=0.3, gt=0.0, le=1.0)


class OutputConfig(ConfigModel):
    """Normalized shard sizing for bounded-memory preparation."""

    games_per_shard: int | None = Field(default=None, ge=1)


class PrepareConfig(ConfigModel):
    """Code-owned schema for ``anthro data prepare``."""

    artifact_name: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    source: SourceConfig
    archive: ArchiveConfig | None = None
    split: SplitConfig = SplitConfig()
    filters: FilterConfig = FilterConfig()
    termination: TerminationConfig = TerminationConfig()
    output: OutputConfig = OutputConfig()


class SequenceLoaderConfig(ConfigModel):
    """Deterministic batching choices for normalized game sequences."""

    split: SplitName = "train"
    batch_size: int = Field(default=8, ge=1)
    length_bucket_width: int | None = Field(default=32, ge=1)
    chunk_length: int | None = Field(default=None, ge=1)
    shuffle: StrictBool = True
    seed: str = Field(default="anthro-sequence-loader-v1", min_length=1)
    drop_last: StrictBool = False


class SequenceDataConfig(ConfigModel):
    """One explicit normalized-data and loader selection."""

    normalized: Path
    manifest: Path
    loader: SequenceLoaderConfig
