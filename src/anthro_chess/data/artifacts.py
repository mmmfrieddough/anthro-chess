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
from concurrent.futures import ThreadPoolExecutor
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
    """One normalized shard, as its manifest describes it and its footer confirms.

    A consumer that has checked a shard against its manifest already knows what
    the shard is, so identifying a corpus does not have to read it again. It
    also had the footer open, so how many row groups the shard holds and how its
    games divide between the splits travel out of that same pass rather than
    costing another one.
    """

    path: Path
    sha256: str
    split_counts: Mapping[str, int]
    row_groups: int


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file, read in bounded chunks."""

    digest = sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def game_ids_sha256(game_ids: Iterable[int]) -> str:
    """Return the order-independent identity digest for a set of game ids.

    A frozen evaluation pool and the views cut from it both identify themselves
    by which games they hold, so they share one digest rather than two that
    could drift apart.
    """

    return sha256(
        ",".join(str(game_id) for game_id in sorted(game_ids)).encode()
    ).hexdigest()


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
    return materialize_rows(table)


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


def normalized_row_count(shard: Any) -> int:
    """Return how many rows one opened shard holds."""

    return int(shard.reader.metadata.num_rows)


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


def take_rows(table: Any, positions: Sequence[int]) -> Any:
    """Return the named row positions of a row-group table, still columnar.

    Gathering the rows is a buffer copy; turning them into dictionaries of
    Python values is an object per field, and this leaves that half to the
    caller through :func:`materialize_rows`.
    """

    return table.take(list(positions))


def materialize_rows(table: Any) -> list[dict[str, Any]]:
    """Return one dictionary of Python values per row of a columnar table."""

    return cast(list[dict[str, Any]], table.to_pylist())


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


def manifest_archive_records(manifest: Mapping[str, Any]) -> Any:
    """Return what a corpus manifest says it was prepared from, verbatim.

    A manifest predating corpora that span archives records a single ``input``
    rather than a list. Reading from one is legitimate where appending to it is
    not, so the older shape is carried rather than dropped.
    """

    inputs = manifest.get("inputs")
    if inputs is None and "input" in manifest:
        return [manifest["input"]]
    return inputs


def manifest_archive_digests(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> tuple[str, ...]:
    """Return the digest of every archive a corpus was prepared from.

    A caller holding a marked-account snapshot needs these to check that the
    snapshot covers this corpus, and a manifest that cannot answer has to fail
    rather than leave that check silently unmade.
    """

    recorded = manifest_archive_records(manifest)
    archives = recorded if isinstance(recorded, list) else []
    digests = tuple(
        archive["sha256"]
        for archive in archives
        if isinstance(archive, Mapping) and isinstance(archive.get("sha256"), str)
    )
    if not digests or len(digests) != len(archives):
        raise DataLoadingError(
            f"{manifest_path} does not record the digest of every archive this "
            "corpus was prepared from"
        )
    return digests


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


#: How many shards are hashed at once. Hashing releases the interpreter lock and
#: saturates the drive well before the processor, so threads reach the ceiling
#: and more of them do not raise it. The footer pass gets none of this: parsing
#: one holds the lock throughout, which measures as no gain at any width.
_SHARD_HASHERS = 8


@dataclass(frozen=True)
class _ExpectedShard:
    """What a manifest recorded about one output shard."""

    sha256: str
    games: int
    split_counts: Mapping[str, int]


def _expected_shards(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> dict[Path, _ExpectedShard]:
    """Return a manifest's output shards, resolved against the artifact root."""

    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise DataLoadingError(f"{manifest_path} has no output record")
    shards = output.get("shards")
    if not isinstance(shards, list):
        raise DataLoadingError(f"{manifest_path} has no output shard records")

    expected: dict[Path, _ExpectedShard] = {}
    artifact_root = manifest_path.parent.parent
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise DataLoadingError(f"{manifest_path} has an invalid output shard")
        relative_path = shard.get("path")
        digest = shard.get("sha256")
        games = shard.get("games")
        split_counts = shard.get("split_counts")
        if (
            not isinstance(relative_path, str)
            or not isinstance(digest, str)
            or type(games) is not int
            or not isinstance(split_counts, Mapping)
            or any(type(count) is not int for count in split_counts.values())
        ):
            raise DataLoadingError(f"{manifest_path} has an invalid output shard")
        expected[(artifact_root / relative_path).resolve()] = _ExpectedShard(
            sha256=digest,
            games=games,
            split_counts=dict(split_counts),
        )
    return expected


def _checked_extent(recorded: _ExpectedShard, path: Path) -> ShardIdentity:
    """Check one shard's footer against its manifest record.

    A truncated or replaced file either fails to parse its footer or reports a
    different row count, which is what an interrupted preparation and a partial
    copy both leave behind. A page rewritten in place survives this, and hashing
    the file is what sees that.
    """

    shard = open_normalized_shard(path)
    rows = normalized_row_count(shard)
    if rows != recorded.games:
        raise DataLoadingError(
            f"normalized shard holds {rows} row(s) where the manifest records "
            f"{recorded.games}: {path}"
        )
    return ShardIdentity(
        path=path,
        sha256=recorded.sha256,
        split_counts=recorded.split_counts,
        row_groups=normalized_row_group_count(shard),
    )


def _checked_contents(shard: ShardIdentity) -> None:
    """Refuse one shard whose bytes are not the ones the manifest recorded."""

    if file_sha256(shard.path) != shard.sha256:
        raise DataLoadingError(f"normalized data checksum mismatch: {shard.path}")


def validate_manifest_outputs(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    paths: Sequence[Path],
    *,
    verify_contents: bool = False,
) -> tuple[ShardIdentity, ...]:
    """Check that the selected shards are the ones the manifest recorded.

    The default check compares each shard's recorded row count against its
    Parquet footer, reading kilobytes of a file rather than all of it.

    ``verify_contents`` hashes every shard end to end as well. That is the only
    check which sees a page rewritten in place, and it costs a full read of the
    corpus at whatever the drive sustains, so it is asked for rather than paid
    by every run.

    What the manifest recorded is returned either way. It identifies this corpus
    to a caller and says how its games divide between the splits, and the check
    has just established that the shards on disk are the ones it describes.
    """

    expected = _expected_shards(manifest, manifest_path)
    # Resolved once and carried, because a corpus is tens of thousands of paths
    # and resolving one is a system call.
    resolved = [path.resolve() for path in paths]
    if set(resolved) != set(expected):
        raise DataLoadingError(
            "configured normalized paths do not match the data manifest outputs"
        )
    shards = tuple(
        _checked_extent(expected[resolved[index]], path)
        for index, path in enumerate(paths)
    )
    if verify_contents:
        hashers = ThreadPoolExecutor(max_workers=_SHARD_HASHERS)
        try:
            for _ in hashers.map(_checked_contents, shards):
                pass
        finally:
            hashers.shutdown(wait=False, cancel_futures=True)
    return shards
