"""Shared access to normalized Parquet artifacts and their manifests.

Preparation writes these artifacts, training and evaluation read them, and all
three need the same answers about where shards live, whether a manifest is
compatible, and whether the bytes on disk still match what was recorded. Those
rules belong in one place so they cannot drift between consumers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from anthro_chess.chess import action_vocabulary_identity
from anthro_chess.data.schema import (
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    normalized_parquet_schema,
)

_PARQUET_MISSING = "Parquet support is unavailable; install anthro-chess[data]"


class DataLoadingError(ValueError):
    """Raised when normalized data or saved loader state is incompatible."""


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file, read in bounded chunks."""

    digest = sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_shard_paths(path: Path) -> tuple[Path, ...]:
    """Resolve a normalized selection that is either one shard or a directory."""

    if path.is_file():
        return (path,)
    if path.is_dir():
        paths = tuple(sorted(path.glob("games*.parquet")))
        if paths:
            return paths
        raise DataLoadingError(
            f"normalized data directory has no games*.parquet files: {path}"
        )
    raise DataLoadingError(f"normalized data path does not exist: {path}")


def read_normalized_rows(
    path: Path,
    columns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Read normalized rows, optionally projecting an explicit column subset."""

    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - exercised by wheel smoke only
        raise DataLoadingError(_PARQUET_MISSING) from error
    try:
        table = pq.read_table(
            path,
            columns=list(columns) if columns is not None else None,
        )
    except (OSError, ValueError) as error:
        raise DataLoadingError(
            f"cannot read normalized data {path}: {error}"
        ) from error
    return cast(list[dict[str, Any]], table.to_pylist())


def write_normalized_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write rows as a zstd-compressed shard using the canonical schema."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - exercised by wheel smoke only
        raise DataLoadingError(_PARQUET_MISSING) from error

    table = pa.Table.from_pylist(list(rows), schema=normalized_parquet_schema())
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def validate_manifest_compatibility(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    """Reject a manifest whose data contract does not match this code."""

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DataLoadingError(
            f"{manifest_path} uses normalized schema version "
            f"{manifest.get('schema_version')}; expected {SCHEMA_VERSION}"
        )
    if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise DataLoadingError(
            f"{manifest_path} uses preprocessing version "
            f"{manifest.get('preprocessing_version')}; "
            f"expected {PREPROCESSING_VERSION}"
        )
    if manifest.get("action_vocabulary") != action_vocabulary_identity():
        raise DataLoadingError(
            f"{manifest_path} uses an incompatible action vocabulary"
        )


def validate_manifest_outputs(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    paths: Sequence[Path],
) -> None:
    """Check that the selected shards are exactly the ones the manifest recorded."""

    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise DataLoadingError(f"{manifest_path} has no output record")
    shards = output.get("shards")
    if not isinstance(shards, list):
        raise DataLoadingError(f"{manifest_path} has no output shard records")

    expected: dict[Path, str] = {}
    artifact_root = manifest_path.parent.parent
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise DataLoadingError(f"{manifest_path} has an invalid output shard")
        relative_path = shard.get("path")
        digest = shard.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(digest, str):
            raise DataLoadingError(f"{manifest_path} has an invalid output shard")
        expected[(artifact_root / relative_path).resolve()] = digest

    if {path.resolve() for path in paths} != set(expected):
        raise DataLoadingError(
            "configured normalized paths do not match the data manifest outputs"
        )
    for path in paths:
        if file_sha256(path) != expected[path.resolve()]:
            raise DataLoadingError(f"normalized data checksum mismatch: {path}")
