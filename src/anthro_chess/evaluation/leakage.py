"""Verification that a checkpoint never trained on the games it is scored on.

A corpus gives every game exactly one split. So within one corpus, a pool cut
from one split and a run that read another are disjoint by construction, and
establishing that costs a comparison of two names rather than a read of the
games. Where the corpus also declares a split recipe this code can evaluate,
the pool's own game ids are put back through it, which turns the pool's claim
about which split it holds into something checked rather than trusted.

Preparation writes the split column through the same function read back here,
so a corpus whose column disagrees with its own recipe is outside what any of
this can settle.

Disjointness cannot be argued at all when the checkpoint trained on a different
corpus than the pool was drawn from, since splits of unrelated corpora say
nothing about each other. The reading is not refused there: it records that the
check could not be established and says so loudly, so a result carries the fact
rather than an unearned assurance.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthro_chess.data.artifacts import DataLoadingError, normalized_shard_paths
from anthro_chess.data.schema import SPLIT_ALGORITHM, split_name
from anthro_chess.evaluation.pool import FrozenPool

LEAKAGE_CHECK_VERSION = 2

SPLIT_DISJOINT_ALGORITHM = "split-disjoint-v1"
UNVERIFIED_ALGORITHM = "unverified-v1"

logger = logging.getLogger(__name__)


class LeakageError(ValueError):
    """Raised when training and benchmark inputs overlap."""


@dataclass(frozen=True)
class SplitRecipe:
    """How one corpus assigns its games to splits."""

    seed: str
    test_fraction: float
    validation_fraction: float

    def split_of(self, game_id: int) -> str:
        """Return the split this recipe puts one game in."""

        return split_name(
            game_id,
            seed=self.seed,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
        )


@dataclass(frozen=True)
class LeakageCheck:
    """The recorded outcome of one train/pool overlap check.

    ``verified`` is the field to read. The rest says how the answer was
    reached, or why it could not be, so a stored reading stays interpretable
    without the code that produced it.
    """

    algorithm: str
    verified: bool
    unverified_reason: str | None
    #: Whether the pool's game ids were put back through the corpus' declared
    #: split recipe, which is available only where the corpus declares one this
    #: code can evaluate. Disjointness does not rest on it.
    recipe_recomputed: bool
    training_split: str
    pool_split: str
    pool_games: int
    overlapping_games: int
    same_source_corpus: bool
    split_recipe_matches: bool
    training_manifest_sha256: str | None
    pool_source_manifest_sha256: str | None
    training_normalized_paths: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        """Return the stable record stored with an evaluation artifact."""

        return {
            "version": LEAKAGE_CHECK_VERSION,
            "algorithm": self.algorithm,
            "verified": self.verified,
            "unverified_reason": self.unverified_reason,
            "recipe_recomputed": self.recipe_recomputed,
            "training_split": self.training_split,
            "pool_split": self.pool_split,
            "pool_games": self.pool_games,
            "overlapping_games": self.overlapping_games,
            "same_source_corpus": self.same_source_corpus,
            "split_recipe_matches": self.split_recipe_matches,
            "training_manifest_sha256": self.training_manifest_sha256,
            "pool_source_manifest_sha256": self.pool_source_manifest_sha256,
            "training_normalized_paths": list(self.training_normalized_paths),
        }


def check_leakage(
    pool: FrozenPool,
    checkpoint_metadata: Mapping[str, Any],
    *,
    training_normalized: Path | None = None,
) -> LeakageCheck:
    """Establish that a checkpoint's training split excludes this pool's games."""

    training = _training_provenance(checkpoint_metadata)
    split = _training_split(checkpoint_metadata)
    paths = _training_paths(training, training_normalized)

    pool_source = _mapping(pool.manifest.get("source"), "evaluation pool source")
    pool_split = pool.split
    pool_manifest_sha256 = _optional_string(pool_source.get("manifest_sha256"))
    training_manifest_sha256 = _optional_string(training.get("manifest_sha256"))
    training_manifest = _mapping(training.get("manifest"), "training data manifest")
    same_corpus = (
        pool_manifest_sha256 is not None
        and training_manifest_sha256 == pool_manifest_sha256
    )
    split_matches = _same_split_recipe(
        training_manifest.get("split"),
        pool_source.get("split"),
    )

    def build(
        *,
        algorithm: str,
        verified: bool,
        reason: str | None,
        recomputed: bool,
        overlapping: int,
    ) -> LeakageCheck:
        return LeakageCheck(
            algorithm=algorithm,
            verified=verified,
            unverified_reason=reason,
            recipe_recomputed=recomputed,
            training_split=split,
            pool_split=pool_split,
            pool_games=len(pool.games),
            overlapping_games=overlapping,
            same_source_corpus=same_corpus,
            split_recipe_matches=split_matches,
            training_manifest_sha256=training_manifest_sha256,
            pool_source_manifest_sha256=pool_manifest_sha256,
            training_normalized_paths=tuple(str(path) for path in paths),
        )

    if same_corpus and split == pool_split:
        raise LeakageError(
            f"the checkpoint trained on the {split} split, which is the split "
            f"this pool was cut from; every one of its {len(pool.games)} game(s) "
            "was available to that run"
        )

    # Recomputation needs the two sides to agree on the recipe rather than on
    # the whole corpus. A game keeps its id and therefore its split as a corpus
    # grows, so a checkpoint trained on one generation can still be shown
    # disjoint from a pool cut from the next.
    recipe = _recipe_for(pool_source.get("split")) if split_matches else None
    overlapping = 0
    if recipe is not None:
        overlapping = sum(
            1 for game_id in pool.game_ids if recipe.split_of(game_id) == split
        )
        if overlapping:
            raise LeakageError(
                f"{overlapping} pool game(s) belong to the checkpoint's {split} "
                f"split under the corpus' own split recipe; this checkpoint "
                "cannot be scored on this pool"
            )

    if recipe is None and not same_corpus:
        reason = _unverifiable_reason(split_matches)
        logger.warning(
            "Leakage could not be verified: %s. This reading is recorded as "
            "unverified; nothing here establishes that the checkpoint did not "
            "train on these games",
            reason,
        )
        return build(
            algorithm=UNVERIFIED_ALGORITHM,
            verified=False,
            reason=reason,
            recomputed=False,
            overlapping=0,
        )

    check = build(
        algorithm=SPLIT_DISJOINT_ALGORITHM,
        verified=True,
        reason=None,
        recomputed=recipe is not None,
        overlapping=0,
    )
    logger.info(
        "Leakage check passed: this pool holds %s %s game(s) and the checkpoint "
        "trained on %s%s",
        check.pool_games,
        pool_split,
        split,
        ", confirmed against the corpus' split recipe" if recipe else "",
    )
    return check


def _unverifiable_reason(split_matches: bool) -> str:
    """Return why disjointness could not be argued for this pair."""

    if not split_matches:
        return (
            "the checkpoint's training corpus and this pool's declare different "
            "split recipes, so an assignment made under one says nothing about "
            "the other"
        )
    return (
        "the checkpoint trained on a different normalized corpus than this pool "
        "was drawn from, and its split recipe is not one this code can evaluate"
    )


def _recipe_for(split: object) -> SplitRecipe | None:
    """Return the recipe a corpus declared, when this code can evaluate it."""

    if not isinstance(split, Mapping) or split.get("algorithm") != SPLIT_ALGORITHM:
        return None
    seed = split.get("seed")
    test = split.get("test_fraction")
    validation = split.get("validation_fraction")
    if (
        not isinstance(seed, str)
        or not isinstance(test, int | float)
        or not isinstance(validation, int | float)
    ):
        return None
    return SplitRecipe(
        seed=seed,
        test_fraction=float(test),
        validation_fraction=float(validation),
    )


def _training_provenance(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    data = _mapping(metadata.get("data"), "checkpoint data provenance")
    return _mapping(data.get("train"), "checkpoint training data provenance")


def _training_split(metadata: Mapping[str, Any]) -> str:
    resolved = _mapping(metadata.get("resolved_config"), "resolved configuration")
    config = _mapping(resolved.get("config"), "resolved training configuration")
    train = _mapping(config.get("train"), "resolved training data selection")
    loader = _mapping(train.get("loader"), "resolved training loader selection")
    split = loader.get("split")
    if not isinstance(split, str) or not split:
        raise LeakageError("checkpoint does not record which split it trained on")
    return split


def _training_paths(
    training: Mapping[str, Any],
    override: Path | None,
) -> tuple[Path, ...]:
    """Return where the checkpoint's training corpus is, as provenance.

    Resolved but not checked for existence, so a machine without the corpus can
    still take every reading that does not read it.
    """

    if override is not None:
        try:
            return normalized_shard_paths(override)
        except DataLoadingError as error:
            raise LeakageError(
                f"configured training corpus cannot be read: {error}"
            ) from error

    recorded = training.get("normalized_paths")
    if not isinstance(recorded, Sequence) or isinstance(recorded, str | bytes):
        raise LeakageError("checkpoint does not record its normalized training paths")
    paths = tuple(Path(str(item)) for item in recorded)
    if not paths:
        raise LeakageError("checkpoint records no normalized training paths")
    return paths


def _same_split_recipe(training: object, pool: object) -> bool:
    """Compare split recipes, ignoring the per-corpus counts they carry."""

    if not isinstance(training, Mapping) or not isinstance(pool, Mapping):
        return False
    return _recipe_json(training) == _recipe_json(pool)


def _recipe_json(split: Mapping[str, Any]) -> str:
    return json.dumps(
        {key: value for key, value in split.items() if key != "counts"},
        sort_keys=True,
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LeakageError(f"{label} is missing or invalid")
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
