"""Verification that a checkpoint never trained on the games it is scored on.

The pool records which games it holds and what they contain; a checkpoint
records which normalized corpus and split its training loader consumed. This
module compares the two and refuses to report a number when they intersect.

Two comparisons are possible and they are not equally cheap. When the
checkpoint trained on the same normalized corpus the pool was drawn from,
internal game ids mean the same thing on both sides and comparing them reads
two small columns. When the corpora differ, an id says nothing, so the games
are compared by what they contain, which costs a full read of both sides. The
cheap comparison is used only when it is sound.

The content key deliberately excludes the internal game id, unlike the pool's
own per-game digest, which identifies a game *within* one corpus. Two distinct
games that agree on every move, result, rating, and clock would be treated as
one; that direction of error refuses an evaluation rather than permitting a
contaminated one.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from anthro_chess.data.artifacts import (
    DataLoadingError,
    file_sha256,
    normalized_shard_paths,
    read_normalized_rows,
)
from anthro_chess.data.schema import (
    NormalizedColumn,
    row_game_id,
)
from anthro_chess.evaluation.pool import FrozenPool

LEAKAGE_CHECK_VERSION = 1

GAME_ID_ALGORITHM = "game-id-intersection-v1"
CONTENT_HASH_ALGORITHM = "content-hash-intersection-v1"

logger = logging.getLogger(__name__)

_IDENTITY_COLUMNS = (
    NormalizedColumn.SOURCE_ID.value,
    NormalizedColumn.SOURCE_GAME_KEY.value,
    NormalizedColumn.SPLIT.value,
)
_CONTENT_COLUMNS = (
    NormalizedColumn.RULESET.value,
    NormalizedColumn.INITIAL_POSITION.value,
    NormalizedColumn.ACTION_IDS.value,
    NormalizedColumn.RESULT.value,
    NormalizedColumn.WHITE_NORMALIZED_RATING.value,
    NormalizedColumn.BLACK_NORMALIZED_RATING.value,
    NormalizedColumn.CLOCK_REMAINING_DELTA_MS.value,
    NormalizedColumn.SPLIT.value,
)
_CONTENT_KEY_COLUMNS = tuple(
    column for column in _CONTENT_COLUMNS if column != NormalizedColumn.SPLIT.value
)

#: Scans from earlier in this process. A sweep checks one checkpoint against
#: one pool once per benchmark, and each check re-reads every training row to
#: re-derive the same non-overlap. Both dictionaries are keyed on the
#: checksums of the files a scan read and on what it selected from them, so a
#: hit is proof that the same bytes were scanned; checksumming the corpus
#: costs 0.1 s against the 40 s read it saves, and a corpus rewritten in place
#: is scanned again rather than remembered.
_TRAINING_IDS: dict[tuple[str, ...], set[int]] = {}
_CONTENT_KEYS: dict[tuple[str, ...], set[str]] = {}


class LeakageError(ValueError):
    """Raised when training and benchmark inputs overlap or cannot be compared."""


@dataclass(frozen=True)
class LeakageCheck:
    """The recorded outcome of one train/pool overlap comparison."""

    algorithm: str
    training_split: str
    training_games: int
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
            "training_split": self.training_split,
            "training_games": self.training_games,
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
    """Compare a checkpoint's training games against the pool it is scored on."""

    training = _training_provenance(checkpoint_metadata)
    split = _training_split(checkpoint_metadata)
    paths = _training_paths(training, training_normalized)

    pool_source = _mapping(pool.manifest.get("source"), "evaluation pool source")
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

    if not split_matches:
        logger.warning(
            "The checkpoint's training corpus and this pool were split by "
            "different recipes; split assignment guarantees nothing here and "
            "the comparison below is the only thing keeping them apart"
        )

    overlapping = 0
    if same_corpus:
        algorithm = GAME_ID_ALGORITHM
        training_ids, training_games = _training_game_ids(paths, split)
        overlapping = len(training_ids & set(pool.game_ids))
    else:
        algorithm = CONTENT_HASH_ALGORITHM
        logger.info(
            "Checkpoint and pool name different normalized corpora; comparing "
            "recorded game content instead of internal ids"
        )
        keys, training_games = _training_content_keys(paths, split)
        overlapping = len(keys & _pool_content_keys(pool))

    if overlapping:
        raise LeakageError(
            f"{overlapping} pool game(s) appear in the checkpoint's {split} "
            f"split under {algorithm}; this checkpoint cannot be scored on "
            "this pool"
        )

    check = LeakageCheck(
        algorithm=algorithm,
        training_split=split,
        training_games=training_games,
        pool_games=len(pool.games),
        overlapping_games=0,
        same_source_corpus=same_corpus,
        split_recipe_matches=split_matches,
        training_manifest_sha256=training_manifest_sha256,
        pool_source_manifest_sha256=pool_manifest_sha256,
        training_normalized_paths=tuple(str(path) for path in paths),
    )
    logger.info(
        "Leakage check passed: %s training game(s) share no game with %s pool game(s)",
        check.training_games,
        check.pool_games,
    )
    return check


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
    if override is not None:
        try:
            return normalized_shard_paths(override)
        except DataLoadingError as error:
            raise LeakageError(
                f"configured training corpus cannot be read: {error}"
            ) from error

    recorded = training.get("normalized_paths")
    if not isinstance(recorded, Sequence) or isinstance(recorded, (str, bytes)):
        raise LeakageError("checkpoint does not record its normalized training paths")
    paths = tuple(Path(str(item)) for item in recorded)
    if not paths:
        raise LeakageError("checkpoint records no normalized training paths")
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise LeakageError(
            f"the checkpoint's training corpus is not readable here: {missing[0]}. "
            "Point leakage.training_normalized at a copy of the corpus this "
            "checkpoint trained on."
        )
    return paths


def _scan_key(paths: Sequence[Path], selected: str) -> tuple[str, ...]:
    """Identify a scan by the bytes it read and what it kept from them."""

    return (selected, *(file_sha256(path) for path in paths))


def _training_game_ids(paths: Sequence[Path], split: str) -> tuple[set[int], int]:
    key = _scan_key(paths, split)
    identifiers = _TRAINING_IDS.get(key)
    if identifiers is None:
        identifiers = set()
        for path in paths:
            for row in _read(path, _IDENTITY_COLUMNS):
                if row[NormalizedColumn.SPLIT.value] == split:
                    identifiers.add(row_game_id(row))
        if not identifiers:
            raise LeakageError(
                f"the checkpoint's training corpus holds no {split} split games"
            )
        _TRAINING_IDS[key] = identifiers
    return identifiers, len(identifiers)


def _training_content_keys(
    paths: Sequence[Path],
    split: str,
) -> tuple[set[str], int]:
    key = _scan_key(paths, split)
    keys = _CONTENT_KEYS.get(key)
    if keys is None:
        keys = set()
        for path in paths:
            for row in _read(path, _CONTENT_COLUMNS):
                if row[NormalizedColumn.SPLIT.value] == split:
                    keys.add(_content_key(row))
        if not keys:
            raise LeakageError(
                f"the checkpoint's training corpus holds no {split} split games"
            )
        _CONTENT_KEYS[key] = keys
    return keys, len(keys)


def _pool_content_keys(pool: FrozenPool) -> set[str]:
    # An empty selection: a pool holds one split and every row of it counts.
    # Sharing the training side's dictionary is safe because the checksums are
    # part of the key, so a hit across the two can only be the same bytes read
    # the same way.
    key = _scan_key((pool.games_path,), "")
    keys = _CONTENT_KEYS.get(key)
    if keys is None:
        keys = {
            _content_key(row) for row in _read(pool.games_path, _CONTENT_KEY_COLUMNS)
        }
        _CONTENT_KEYS[key] = keys
    return keys


def _content_key(row: Mapping[str, Any]) -> str:
    """Digest what a game contains, independently of where it is stored."""

    content = {column: row[column] for column in _CONTENT_KEY_COLUMNS}
    return sha256(
        json.dumps(
            content, sort_keys=True, separators=(",", ":"), default=list
        ).encode()
    ).hexdigest()


def _read(path: Path, columns: Sequence[str]) -> list[dict[str, Any]]:
    try:
        return read_normalized_rows(path, columns)
    except DataLoadingError as error:
        raise LeakageError(f"cannot read training corpus {path}: {error}") from error


def _same_split_recipe(training: object, pool: object) -> bool:
    """Compare split recipes, ignoring the per-corpus counts they carry."""

    if not isinstance(training, Mapping) or not isinstance(pool, Mapping):
        return False
    return _recipe(training) == _recipe(pool)


def _recipe(split: Mapping[str, Any]) -> str:
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
