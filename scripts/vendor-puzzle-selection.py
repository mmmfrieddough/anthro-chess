#!/usr/bin/env python3
"""Regenerate the vendored puzzle selection from the pinned upstream archive.

This is a maintenance script, not part of the runtime package. It runs only
when the puzzle set is deliberately re-pinned or its selection design changes,
which is also when the set version is bumped, so the benchmark identity never
changes underneath a recorded reading.

Usage:

    scripts/vendor-puzzle-selection.py --config <puzzle-set-selection.toml>

Upstream serves one rolling object with no history, so the archive a rebuild
would fetch is not the archive this selection came from. Pass ``--archive``
when a verified copy of the pinned snapshot is already on the machine; without
it the script downloads whatever the pinned URL now serves, refuses it unless
the digest still matches, and keeps it beneath the data root so the next run
reuses it rather than chasing an object that has since moved.

The output is the checked-in selection under the puzzles package: the selected
rows, plus the record of the upstream revision and population they were drawn
from. The command prints the identity to write back into the configuration,
and ``anthro eval prepare-puzzles`` refuses to build until the two agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from anthro_chess.config import ConfigError, load_config  # noqa: E402
from anthro_chess.evaluation.puzzles.dataset import (  # noqa: E402
    PUZZLE_FILE_NAME,
    VENDORED_RECORD_FILE_NAME,
    PuzzleSetBuildConfig,
    PuzzleSetError,
    build_vendored_puzzle_set,
)
from anthro_chess.machine import DATA_ROOT_VARIABLE, required_root  # noqa: E402

DATA_DIRECTORY = (
    REPOSITORY_ROOT / "src" / "anthro_chess" / "evaluation" / "puzzles" / "data"
)


def main(argv: list[str] | None = None) -> int:
    """Select puzzles from the pinned archive and write the vendored pair."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="TOML puzzle-source and selection recipe to select under.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Local copy of the pinned archive; downloaded when omitted.",
    )
    arguments = parser.parse_args(argv)

    try:
        config = load_config(PuzzleSetBuildConfig, path=arguments.config).value
        archive_directory = (
            None
            if arguments.archive is not None
            else required_root(
                DATA_ROOT_VARIABLE,
                alternative="--archive must name a copy of the pinned archive",
            )
            / config.artifact_name
        )
        vendored = build_vendored_puzzle_set(
            config,
            source_path=arguments.archive,
            archive_directory=archive_directory,
        )
    except (ConfigError, PuzzleSetError) as error:
        raise SystemExit(f"vendor-puzzle-selection: {error}") from error

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    puzzle_path = DATA_DIRECTORY / PUZZLE_FILE_NAME
    puzzle_path.write_text(vendored.content, encoding="utf-8", newline="\n")
    record_path = DATA_DIRECTORY / VENDORED_RECORD_FILE_NAME
    record_path.write_text(
        json.dumps(vendored.as_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {vendored.entries} puzzle(s) to {puzzle_path}")
    print(f"Wrote source and population record to {record_path}")
    print(f"Set expected_entries = {vendored.entries} in {arguments.config}")
    print(
        f'Set expected_puzzles_sha256 = "{vendored.puzzles_sha256}" '
        f"in {arguments.config}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
