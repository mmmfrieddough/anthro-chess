"""What checking a corpus against its manifest catches, and what it does not."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from anthro_chess.data.artifacts import (
    DataLoadingError,
    normalized_shard_paths,
    validate_manifest_outputs,
)


def _corpus(
    write_corpus: Callable[..., tuple[Path, Path]],
    directory: Path,
    normalized_row: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], Path, tuple[Path, ...]]:
    rows = [normalized_row(game_id, split="train", plies=6) for game_id in range(1, 9)]
    normalized, manifest_path = write_corpus(directory, rows, games_per_shard=4)
    manifest: dict[str, Any] = json.loads(manifest_path.read_bytes())
    return manifest, manifest_path, normalized_shard_paths(normalized)


def test_the_default_check_returns_the_digests_the_manifest_recorded(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    manifest, manifest_path, paths = _corpus(write_corpus, tmp_path, normalized_row)

    shards = validate_manifest_outputs(manifest, manifest_path, paths)

    recorded = {
        shard["path"].rsplit("/", 1)[-1]: shard["sha256"]
        for shard in manifest["output"]["shards"]
    }
    assert {shard.path.name: shard.sha256 for shard in shards} == recorded


@pytest.mark.parametrize("verify_contents", [False, True], ids=["default", "hashing"])
def test_either_check_refuses_a_truncated_shard(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    verify_contents: bool,
) -> None:
    """An interrupted preparation and a partial copy both leave one of these."""

    manifest, manifest_path, paths = _corpus(write_corpus, tmp_path, normalized_row)
    raw = paths[0].read_bytes()
    paths[0].write_bytes(raw[: len(raw) // 2])

    with pytest.raises(DataLoadingError, match=str(paths[0])):
        validate_manifest_outputs(
            manifest, manifest_path, paths, verify_contents=verify_contents
        )


def test_only_the_content_check_sees_a_shard_rewritten_in_place(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """This is the whole of what the default check gives up, stated as a test.

    Rewriting a data page leaves the footer and the row count intact, so the
    cheap check passes it and only hashing the bytes refuses it. A reader
    deciding whether to pay for the full read is deciding about exactly this.
    """

    manifest, manifest_path, paths = _corpus(write_corpus, tmp_path, normalized_row)
    raw = bytearray(paths[0].read_bytes())
    footer = len(raw) - 8 - int.from_bytes(raw[-8:-4], "little")
    raw[4:footer] = bytes(footer - 4)
    paths[0].write_bytes(bytes(raw))

    assert validate_manifest_outputs(manifest, manifest_path, paths)

    with pytest.raises(DataLoadingError, match="checksum mismatch"):
        validate_manifest_outputs(manifest, manifest_path, paths, verify_contents=True)


def test_a_shard_the_manifest_does_not_name_is_refused(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    manifest, manifest_path, paths = _corpus(write_corpus, tmp_path, normalized_row)
    manifest["output"]["shards"] = manifest["output"]["shards"][:1]

    with pytest.raises(DataLoadingError, match="do not match the data manifest"):
        validate_manifest_outputs(manifest, manifest_path, paths)


def test_a_shard_holding_the_wrong_number_of_rows_is_refused(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """A shard swapped for another corpus's keeps a valid footer."""

    manifest, manifest_path, paths = _corpus(write_corpus, tmp_path, normalized_row)
    manifest["output"]["shards"][0]["games"] = 99

    with pytest.raises(DataLoadingError, match="row"):
        validate_manifest_outputs(manifest, manifest_path, paths)
