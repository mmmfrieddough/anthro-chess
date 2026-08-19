"""Rule-sensitive position labels, derived once per pool generation.

Predicates and characteristics are a pure function of a frozen pool, so every
reading of every checkpoint resolves the same answer. They are derived once
into an artifact beside the pool and read back after that. The artifact is
keyed by the pool's identity and by every scheme the labels depend on, so a
changed predicate or a re-cut pool is a miss rather than a stale hit, and the
miss rebuilds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

from anthro_chess.data import encode_game
from anthro_chess.evaluation.dependency import PositionKey
from anthro_chess.evaluation.pool import FrozenPool, PoolProjection
from anthro_chess.evaluation.scoring import encoding_input, row_game_id
from anthro_chess.evaluation.slices import (
    SLICE_SCHEME_VERSION,
    PositionCharacteristic,
    PositionLabels,
    PositionPredicate,
    PredicateMatch,
    board_from_encoding,
    position_labels,
)

logger = logging.getLogger(__name__)

POSITION_LABEL_ARTIFACT_VERSION = 1

#: Written beside the pool, so it shares the pool's lifetime and goes with it.
POSITION_LABEL_FILE_NAME = "position-labels.parquet"

_ARTIFACT_KEY_FIELD = b"anthro-position-labels-key"

#: Each characteristic's bit in the stored mask, read off the enum rather than
#: restated. The names go into the artifact key, so reordering or adding one
#: rebuilds instead of reinterpreting bits written under the old order.
_CHARACTERISTIC_BITS: Mapping[PositionCharacteristic, int] = {
    characteristic: 1 << index
    for index, characteristic in enumerate(PositionCharacteristic)
}

_PREDICATE_COLUMNS: tuple[PositionPredicate, ...] = tuple(PositionPredicate)

#: Games per unit of work handed to a worker, and per row group written. Large
#: enough that pickling a result is not most of the job, small enough that the
#: last worker to finish does not hold up the build.
_BUILD_CHUNK_GAMES = 256

#: Chunks queued beyond the workers themselves. Deep enough that a worker never
#: waits for the parent to gather rows, shallow enough that the pool's rows are
#: not all materialized as Python dictionaries at once.
_BUILD_PREFETCH = 8


class PositionLabelError(ValueError):
    """Raised when a pool's position labels cannot be derived or read."""


class PositionLabelStore:
    """Every pool position's labels, held columnar and gathered per batch.

    The columns stay in Arrow. A pool holds millions of positions and six
    predicate columns over them, and the same data as Python objects is
    gigabytes; a batch's worth is not.
    """

    def __init__(self, table: Any) -> None:
        import pyarrow.compute as pc  # type: ignore[import-untyped]

        self._characteristics = table.column("characteristics").combine_chunks()
        self._predicates = tuple(
            (predicate, table.column(predicate.value).combine_chunks())
            for predicate in _PREDICATE_COLUMNS
        )
        runs = pc.run_end_encode(table.column("game_id").combine_chunks())
        ends = runs.run_ends.to_pylist()
        starts = [0, *ends[:-1]]
        self._spans = {
            game_id: (start, end - start)
            for game_id, start, end in zip(
                runs.values.to_pylist(), starts, ends, strict=True
            )
        }

    @property
    def position_count(self) -> int:
        """Return how many positions the artifact holds."""

        return len(self._characteristics)

    def labels(self, game_ids: Sequence[int]) -> dict[PositionKey, PositionLabels]:
        """Return the labels of every position in the named games."""

        gathered: dict[PositionKey, PositionLabels] = {}
        for game_id in game_ids:
            span = self._spans.get(game_id)
            if span is None:
                raise PositionLabelError(
                    f"the position-label artifact does not hold game {game_id}"
                )
            offset, count = span
            characteristics = self._characteristics.slice(offset, count).to_pylist()
            realized = [
                (predicate, column.slice(offset, count).to_pylist())
                for predicate, column in self._predicates
            ]
            for ply_index in range(count):
                # A null column is a predicate the position does not realize;
                # an empty one realizes it with no successful action, which is
                # a threat nothing prevents rather than an absent predicate.
                matches = {
                    predicate: PredicateMatch(
                        predicate=predicate,
                        successful_action_ids=frozenset(action_ids),
                    )
                    for predicate, column in realized
                    if (action_ids := column[ply_index]) is not None
                }
                gathered[(game_id, ply_index)] = PositionLabels(
                    predicates=matches,
                    characteristics=_characteristics_from_mask(
                        characteristics[ply_index]
                    ),
                )
        return gathered


def open_position_labels(
    pool: FrozenPool,
    projection: PoolProjection,
) -> PositionLabelStore:
    """Return the pool's labels, deriving and saving them the first time."""

    path = pool.games_path.parent / POSITION_LABEL_FILE_NAME
    key = artifact_key(pool)
    table = _read_matching(path, key)
    if table is None:
        logger.info(
            "Deriving position labels for %s game(s); later readings of this "
            "pool read %s",
            len(pool.games),
            path.name,
        )
        _build(pool, projection, path, key=key)
        table = _read_matching(path, key)
        if table is None:
            raise PositionLabelError(
                f"the position-label artifact just written to {path} does not "
                "carry the key it was built under"
            )
    return PositionLabelStore(table)


def artifact_key(pool: FrozenPool) -> str:
    """Return the digest a stored artifact has to carry to be read.

    Covers the pool generation and every scheme the labels are derived under,
    so a re-cut pool, a changed slice scheme, and an added predicate all miss.
    """

    identity = pool.manifest.get("pool")
    if not isinstance(identity, Mapping):
        raise PositionLabelError("evaluation pool manifest has no pool identity")
    payload = {
        "artifact_version": POSITION_LABEL_ARTIFACT_VERSION,
        "pool_id": str(identity["id"]),
        "pool_version": int(identity["version"]),
        "game_ids_sha256": _game_ids_sha256(pool),
        "slice_scheme_version": SLICE_SCHEME_VERSION,
        "characteristics": [str(item) for item in PositionCharacteristic],
        "predicates": [str(item) for item in _PREDICATE_COLUMNS],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _game_ids_sha256(pool: FrozenPool) -> str:
    identity = pool.manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise PositionLabelError("evaluation pool manifest has no identity record")
    return str(identity["game_ids_sha256"])


def _characteristics_from_mask(mask: int) -> frozenset[PositionCharacteristic]:
    return frozenset(
        characteristic
        for characteristic, bit in _CHARACTERISTIC_BITS.items()
        if mask & bit
    )


def _schema(key: str) -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("game_id", pa.uint64()),
            pa.field("characteristics", pa.uint16()),
            *(
                pa.field(predicate.value, pa.list_(pa.uint16()))
                for predicate in _PREDICATE_COLUMNS
            ),
        ],
        metadata={_ARTIFACT_KEY_FIELD: key.encode("utf-8")},
    )


def _read_matching(path: Path, key: str) -> Any | None:
    """Return the stored artifact when it was built under ``key``."""

    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        table = pq.read_table(path)
    except (OSError, ValueError) as error:
        raise PositionLabelError(
            f"cannot read the position-label artifact {path}: {error}"
        ) from error
    metadata = table.schema.metadata or {}
    if metadata.get(_ARTIFACT_KEY_FIELD, b"").decode("utf-8") != key:
        logger.info(
            "The position-label artifact at %s was built under a different pool "
            "or slice scheme; deriving it again",
            path,
        )
        return None
    return table


def _build(
    pool: FrozenPool,
    projection: PoolProjection,
    path: Path,
    *,
    key: str,
) -> None:
    """Derive every pool position's labels, writing one row group per chunk.

    Written through a temporary file and renamed, so two readings racing each
    other leave one whole artifact rather than a torn one.
    """

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _schema(key)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for derived in _derive_all(projection, pool.game_ids):
                writer.write_table(pa.table(derived, schema=schema))
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise PositionLabelError(
            f"cannot write the position-label artifact {path}: {error}"
        ) from error
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _derive_all(
    projection: PoolProjection,
    game_ids: Sequence[int],
) -> Iterator[dict[str, list[Any]]]:
    """Yield each chunk's derived columns in game-id order.

    Order is what keeps the worker count out of the artifact: a build on one
    process and a build on thirty write the same bytes.
    """

    chunks = [
        tuple(game_ids[start : start + _BUILD_CHUNK_GAMES])
        for start in range(0, len(game_ids), _BUILD_CHUNK_GAMES)
    ]
    if len(chunks) < 2:
        for chunk in chunks:
            yield _derive_rows(projection.rows(chunk))
        return
    pool = ProcessPoolExecutor()
    pending = deque(chunks)
    inflight: deque[Future[dict[str, list[Any]]]] = deque()
    capacity = (os.process_cpu_count() or 1) + _BUILD_PREFETCH
    try:
        while pending or inflight:
            while pending and len(inflight) < capacity:
                inflight.append(
                    pool.submit(_derive_rows, projection.rows(pending.popleft()))
                )
            yield inflight.popleft().result()
    finally:
        pool.shutdown(cancel_futures=True)


def _derive_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Return the label columns of every position in one chunk of games."""

    derived: dict[str, list[Any]] = {
        "game_id": [],
        "characteristics": [],
        **{predicate.value: [] for predicate in _PREDICATE_COLUMNS},
    }
    for row in rows:
        game_id = row_game_id(row)
        for ply in encode_game(encoding_input(row)):
            labels = position_labels(board_from_encoding(ply.board))
            derived["game_id"].append(game_id)
            mask = 0
            for characteristic in labels.characteristics:
                mask |= _CHARACTERISTIC_BITS[characteristic]
            derived["characteristics"].append(mask)
            for predicate in _PREDICATE_COLUMNS:
                match = labels.predicates.get(predicate)
                derived[predicate.value].append(
                    None if match is None else sorted(match.successful_action_ids)
                )
    return derived


__all__ = [
    "POSITION_LABEL_ARTIFACT_VERSION",
    "POSITION_LABEL_FILE_NAME",
    "PositionLabelError",
    "PositionLabelStore",
    "artifact_key",
    "open_position_labels",
]
