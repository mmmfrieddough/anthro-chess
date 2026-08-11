"""Shared access to normalized Parquet artifacts and their manifests.

Preparation writes these artifacts, training and evaluation read them, and all
three need the same answers about where shards live, whether a manifest is
compatible, and whether the bytes on disk still match what was recorded. Those
rules belong in one place so they cannot drift between consumers.
"""

from __future__ import annotations

import bz2
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from io import TextIOWrapper
from pathlib import Path
from typing import Any, TextIO, cast

from anthro_chess.chess import action_vocabulary_identity
from anthro_chess.data.schema import (
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    NormalizedColumn,
    normalized_parquet_schema,
)

_PARQUET_MISSING = "Parquet support is unavailable; install anthro-chess[data]"
#: How this project identifies itself to a source. One value, because a source
#: that blocks it should block all of this project's traffic rather than half.
SOURCE_USER_AGENT = "anthro-chess-data-acquisition/1"


class DataLoadingError(ValueError):
    """Raised when normalized data or saved loader state is incompatible."""


def write_text_atomically(path: Path, text: str) -> None:
    """Replace a file's contents in one step, or leave the old ones in place.

    A partial write of an artifact that is the only record of something — a
    corpus manifest, an account snapshot — loses whatever it recorded, and the
    data it describes is not recoverable without it.

    The staging file carries the writer's process id, because two processes
    otherwise share one and the atomicity is lost between them rather than
    within either: each truncates what the other is part-way through writing,
    and both then rename the result into place. Preparation and the account
    census both write an archive's counts, and nothing stops them doing it at
    the same moment.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.{os.getpid()}.writing")
    try:
        partial.write_text(text, encoding="utf-8")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


@contextmanager
def open_pgn_text(source_path: Path) -> Iterator[TextIO]:
    """Yield a PGN archive as text, decompressing in the stream.

    A second, uncompressed copy of the archive is larger than the corpus it
    produces, so one is never written to disk. Bzip2 is here because the
    universal export publishes it and nothing else; it decompresses roughly an
    order of magnitude slower than Zstandard and still outpaces the game decode
    it feeds.
    """

    if source_path.suffix == ".bz2":
        try:
            with bz2.open(source_path, "rt", encoding="utf-8") as pgn_file:
                yield pgn_file
        except (OSError, EOFError) as error:
            # A truncated stream raises EOFError, which is not an OSError and
            # would otherwise escape every handler between here and the CLI.
            raise DataLoadingError(
                f"cannot decompress input PGN {source_path}: {error}"
            ) from error
        return

    if source_path.suffix != ".zst":
        with source_path.open("r", encoding="utf-8") as pgn_file:
            yield pgn_file
        return

    try:
        import zstandard
    except ImportError as error:  # pragma: no cover - exercised by wheel smoke only
        raise DataLoadingError(
            "Zstandard support is unavailable; install anthro-chess[data]"
        ) from error

    with source_path.open("rb") as compressed_file:
        decompressor = zstandard.ZstdDecompressor()
        try:
            with (
                decompressor.stream_reader(compressed_file) as reader,
                TextIOWrapper(reader, encoding="utf-8") as pgn_file,
            ):
                yield pgn_file
        except zstandard.ZstdError as error:
            raise DataLoadingError(
                f"cannot decompress input PGN {source_path}: {error}"
            ) from error


@dataclass(frozen=True)
class ShardIdentity:
    """One normalized shard, named by the digest its manifest recorded.

    A consumer that has verified a shard against its manifest already knows
    what the shard is, so identifying a corpus does not have to read it again.
    """

    path: Path
    sha256: str


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file, read in bounded chunks."""

    digest = sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def game_ids_sha256(game_ids: Sequence[int]) -> str:
    """Return the order-independent identity digest for a set of game ids.

    Both a frozen evaluation pool and a load-time training selection identify
    themselves by which games they hold, so they share one digest rather than
    two that could drift apart.
    """

    return sorted_game_ids_sha256(sorted(game_ids))


#: How many ids are joined before being folded in. Large enough that the joining
#: is what costs rather than the call, small enough that the joined form stays a
#: buffer rather than becoming the structure this exists to avoid building.
_DIGEST_CHUNK_GAMES = 1 << 16


def sorted_game_ids_sha256(game_ids: Iterable[int]) -> str:
    """Return the same digest over ids that are already in ascending order.

    The joined form the digest is defined over is fed in a chunk at a time
    rather than built. A corpus-scale selection's is tens of gigabytes of
    string, read once, to produce sixty-four characters.
    """

    digest = sha256()
    chunk: list[str] = []
    for game_id in game_ids:
        chunk.append(str(game_id))
        if len(chunk) == _DIGEST_CHUNK_GAMES:
            digest.update(",".join(chunk).encode())
            # The empty entry is the comma between this chunk and the next,
            # written by the join that flushes the next one. A tail that never
            # grows past it joins to nothing and updates nothing.
            chunk = [""]
    digest.update(",".join(chunk).encode())
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


@dataclass(frozen=True)
class _OpenShard:
    """An opened shard, carrying the path its read failures have to name.

    A failed page read raises a bare thrift message, so the row-group index
    this module supplies is the whole locator, and an index identifies no file
    on a corpus of tens of thousands of shards.
    """

    path: Path
    reader: Any


def open_normalized_shard(path: Path) -> Any:
    """Open one shard for row-group reads, parsing its footer only.

    The reader, its row groups, and the tables they produce stay opaque to
    callers. Keeping every Parquet call in this module is what lets the storage
    format be one decision rather than one per consumer.
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - exercised by wheel smoke only
        raise DataLoadingError(_PARQUET_MISSING) from error
    try:
        return _OpenShard(path=path, reader=pq.ParquetFile(path))
    except (OSError, ValueError) as error:
        raise DataLoadingError(
            f"cannot read normalized data {path}: {error}"
        ) from error


def normalized_row_group_count(shard: Any) -> int:
    """Return how many row groups one opened shard holds."""

    return int(shard.reader.metadata.num_row_groups)


def read_normalized_row_group(
    shard: Any,
    row_group: int,
    columns: Sequence[str],
) -> Any:
    """Read one row group's projected columns as an opaque columnar table."""

    try:
        return shard.reader.read_row_group(row_group, columns=list(columns))
    except (OSError, ValueError) as error:
        raise DataLoadingError(
            f"cannot read normalized data {shard.path} row group {row_group}: {error}"
        ) from error


def row_group_column(table: Any, column: str) -> list[Any]:
    """Return one column of a row-group table as a Python list.

    Reading a column at a time rather than a row at a time is what keeps an
    index pass over a corpus-scale shard cheap: it builds one list per column
    instead of one dictionary per game.
    """

    return cast(list[Any], table.column(column).to_pylist())


def rows_at_positions(table: Any, positions: Sequence[int]) -> list[dict[str, Any]]:
    """Materialize the named row positions of a row-group table."""

    return cast(list[dict[str, Any]], table.take(list(positions)).to_pylist())


#: Columns whose values never repeat, so a dictionary of them is as large as
#: the data it indexes and costs about a quarter of the column for nothing.
#: Merely high cardinality is not the test: a player digest is distinct in
#: most rows and still measures smaller encoded, because an account plays
#: many of a shard's games.
_UNIQUE_PER_ROW_COLUMNS = frozenset(
    {
        NormalizedColumn.SOURCE_GAME_KEY.value,
    }
)


def _dictionary_columns(schema: Any) -> list[str]:
    """Return the leaf column paths that should be dictionary encoded.

    Parquet names a list column's leaf ``<field>.list.element``, and a path
    matching no leaf is ignored rather than rejected, so a wrong one silently
    drops the encoding.
    """

    import pyarrow as pa

    paths: list[str] = []
    for field in schema:
        if field.name in _UNIQUE_PER_ROW_COLUMNS:
            continue
        leaf = (
            f"{field.name}.list.element" if pa.types.is_list(field.type) else field.name
        )
        paths.append(leaf)
    return paths


def write_normalized_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write rows as a zstd-compressed shard using the canonical schema."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - exercised by wheel smoke only
        raise DataLoadingError(_PARQUET_MISSING) from error

    schema = normalized_parquet_schema()
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=_dictionary_columns(schema),
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
) -> tuple[ShardIdentity, ...]:
    """Check that the selected shards are exactly the ones the manifest recorded.

    The verified digests are returned rather than discarded, because a caller
    that needs to identify this corpus would otherwise read every shard a
    second time to learn what this pass already established.
    """

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
    identities: list[ShardIdentity] = []
    for path in paths:
        digest = file_sha256(path)
        if digest != expected[path.resolve()]:
            raise DataLoadingError(f"normalized data checksum mismatch: {path}")
        identities.append(ShardIdentity(path=path, sha256=digest))
    return tuple(identities)
