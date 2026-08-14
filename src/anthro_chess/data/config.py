"""Strict configuration for data acquisition, preparation, and sequence loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, StrictBool, model_validator

from anthro_chess.config import ConfigModel
from anthro_chess.data.rating_namespace import names_a_pool
from anthro_chess.data.schema import SplitName
from anthro_chess.data.speed import Speed


class SourceConfig(ConfigModel):
    """Identity and rating semantics for one source selection.

    ``rating_namespace_prefix`` names the family of pools a source rates in
    rather than one of them; which pool rated a game is read per game by
    :func:`~anthro_chess.data.rating_namespace.rating_namespace_from_event`.
    """

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: str = Field(min_length=1)
    url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    rating_namespace_prefix: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    rating_system: str | None = None
    ratings_are_normalized: StrictBool = False

    @model_validator(mode="after")
    def _validate_prefix_leaves_the_pool_to_the_game(self) -> SourceConfig:
        # A selection that moved a whole namespace under this key rather than
        # splitting it would stamp every row ``lichess_blitz_blitz``.
        prefix = self.rating_namespace_prefix
        if prefix is not None and names_a_pool(prefix):
            raise ValueError(
                f"rating namespace prefix {prefix!r} already names a pool, and "
                "the pool is read per game from the source's own label"
            )
        return self


_COMPRESSION_SUFFIXES = {"zstd": ".zst", "bzip2": ".bz2"}


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
    compression: Literal["zstd", "bzip2"] = "zstd"

    @model_validator(mode="after")
    def _validate_compression_matches_name(self) -> ArchiveConfig:
        # ``compression`` is declaration only: ``open_pgn_text`` dispatches on
        # the suffix, so a disagreement describes one format while another is
        # read. A new format is added in both places or in neither.
        expected = _COMPRESSION_SUFFIXES[self.compression]
        if not self.file_name.endswith(expected):
            raise ValueError(
                f"archive declares {self.compression} but {self.file_name} "
                f"does not end in {expected}"
            )
        return self


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
    """Filters for the initial standard-human-game corpus.

    ``marked_accounts`` names a snapshot of the accounts a source has marked
    for breaking its rules, and every game either player appears in is
    rejected. A relative path resolves against the selection file naming it,
    because the snapshot is checked in beside that selection rather than built
    during preparation; ``configs/data/marked-accounts/`` says why.

    ``speed`` is matched against the class
    :func:`~anthro_chess.data.speed.speed_from_time_control` derives, so a game
    whose time control says nothing is rejected along with the other classes.
    """

    minimum_plies: int = Field(default=1, ge=1)
    require_rated: StrictBool = True
    require_ratings: StrictBool = False
    exclude_bots: StrictBool = True
    marked_accounts: Path | None = None
    speed: Speed | None = None
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
    archives: tuple[ArchiveConfig, ...] = ()
    split: SplitConfig = SplitConfig()
    filters: FilterConfig = FilterConfig()
    termination: TerminationConfig = TerminationConfig()
    output: OutputConfig = OutputConfig()

    @model_validator(mode="after")
    def _validate_archives(self) -> PrepareConfig:
        digests = [archive.sha256 for archive in self.archives]
        if len(set(digests)) != len(digests):
            raise ValueError("each archive may appear once in a selection")
        # Acquisition resolves a directory per archive, so a shared file name
        # only collides when the artifact it resolves to is shared too.
        targets = [
            (archive.artifact_name or self.artifact_name, archive.file_name)
            for archive in self.archives
        ]
        if len(set(targets)) != len(targets):
            raise ValueError(
                "two archives in a selection would be acquired to the same path"
            )
        return self


class SelectionConfig(ConfigModel):
    """Load-time selection within one prepared corpus.

    Filtering here is not the same operation as filtering during preparation.
    Preparation runs before split assignment, so a filter there removes games
    from every split at once and shifts the training data and the evaluation
    reference in the same direction, where no benchmark can detect it.
    Selection runs after, touches only the split being loaded, and therefore
    stays visible as a mismatch against a clean reference.

    The axes are the ones worth comparing models across: time control and
    rating. The rating bounds read the normalized rating and require it from
    both players, because a game whose rating is unknown cannot be placed in a
    band. Subsampling ranks by a digest of the game id, so a fraction selects
    the same games on any machine and a smaller fraction is a subset of a
    larger one.

    ``speed`` matches the class
    :func:`~anthro_chess.data.speed.speed_from_clock_ms` derives, the same
    derivation the benchmark slices report, so an arm and its slice name one
    set of games rather than two. Preparation's filter of the same name reads
    the PGN header instead, and the two part over correspondence: a clockless
    game reaches these columns as no class at all.

    ``marked_accounts`` is not one of those comparison axes. It names the
    snapshot
    ``docs/decisions/0041-games-of-marked-accounts-leave-the-corpus.md``
    rejects on, matched against the account digests the row carries, and it
    belongs here because a corpus prepared without one keeps those games in
    every split;
    ``docs/decisions/0062-the-breadth-corpus-filters-for-validity-alone.md``
    says why the breadth corpus is prepared that way. A relative path resolves
    against the selection naming it.

    The exclusion runs before subsampling, so ``maximum_games`` still delivers the
    count it names to a run that sets this and to one that does not — which is
    what lets two such runs differ in which games they read rather than in how
    many. ``fraction`` does not: it takes a share of whatever survived, so the
    filtered run reads fewer games and confounds the two.
    """

    speed: Speed | None = None
    marked_accounts: Path | None = None
    minimum_time_initial_ms: int | None = Field(default=None, ge=0)
    maximum_time_initial_ms: int | None = Field(default=None, ge=0)
    minimum_time_increment_ms: int | None = Field(default=None, ge=0)
    maximum_time_increment_ms: int | None = Field(default=None, ge=0)
    minimum_rating: int | None = Field(default=None, ge=0)
    maximum_rating: int | None = Field(default=None, ge=0)
    require_ratings: StrictBool = False
    fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    maximum_games: int | None = Field(default=None, ge=1)
    seed: str = Field(default="anthro-training-selection-v1", min_length=1)

    @model_validator(mode="after")
    def _validate_bounds(self) -> SelectionConfig:
        bounds = (
            (
                "time_initial_ms",
                self.minimum_time_initial_ms,
                self.maximum_time_initial_ms,
            ),
            (
                "time_increment_ms",
                self.minimum_time_increment_ms,
                self.maximum_time_increment_ms,
            ),
            ("rating", self.minimum_rating, self.maximum_rating),
        )
        for name, minimum, maximum in bounds:
            if minimum is not None and maximum is not None and maximum < minimum:
                raise ValueError(
                    f"selection maximum_{name} must not be below minimum_{name}"
                )
        return self


class SequenceLoaderConfig(ConfigModel):
    """Deterministic batching choices for normalized game sequences."""

    split: SplitName = "train"
    selection: SelectionConfig = SelectionConfig()
    batch_size: int = Field(default=8, ge=1)
    length_bucket_width: int | None = Field(default=32, ge=1)
    chunk_length: int | None = Field(default=None, ge=1)
    shuffle: StrictBool = True
    seed: str = Field(default="anthro-sequence-loader-v1", min_length=1)
    drop_last: StrictBool = False


class StreamingLoaderConfig(ConfigModel):
    """Bounded-memory shard-backed reading of one normalized selection.

    Declaring this section selects the shard-backed loader over the eager one.
    Every field is a memory bound rather than a tuning preference, and they
    divide by whether they change which examples land in a batch:

    ``planning_window_examples`` does, because a window is the span over which
    length buckets are filled and flushed, so it belongs to the loader identity
    a resumed run has to match. ``workers`` and ``prefetch_batches`` do not;
    they decide how far ahead the same batches are built and on how many
    processes, so a run may be resumed on a machine that affords a different
    number of either. ``prefetch_batches`` applies only alongside workers:
    with none, building a batch ahead would cost the memory and save nothing,
    so exactly one is held.

    The two add rather than compete: the loader keeps one outstanding job per
    worker and ``prefetch_batches`` more on top, because a worker whose next
    job is not already queued waits for the consumer to come back for the one
    it just finished. Their sum is what is resident, so raising the pool raises
    the memory held in flight as much as raising the depth does.

    The remaining bound is not configured here at all. Materialization reads
    one row group at a time, so preparation's shard and row-group sizing is
    what caps the columnar read; ``docs/data.md`` owns that end.
    """

    planning_window_examples: int = Field(default=16384, ge=1)
    workers: int = Field(default=0, ge=0)
    prefetch_batches: int = Field(default=4, ge=1)


class SequenceDataConfig(ConfigModel):
    """One explicit normalized-data and loader selection."""

    normalized: Path
    manifest: Path
    loader: SequenceLoaderConfig
    streaming: StreamingLoaderConfig | None = None
