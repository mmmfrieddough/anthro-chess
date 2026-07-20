"""Command-line interface for Anthro Chess."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from anthro_chess import __version__

CommandHandler = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""
    parser = argparse.ArgumentParser(
        prog="anthro",
        description="Command-line tools for Anthro Chess.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subcommands = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subcommands.add_parser(
        "smoke",
        help="Verify that the package and command-line interface are installed.",
    )
    smoke_parser.set_defaults(handler=_run_smoke)

    data_parser = subcommands.add_parser(
        "data",
        help="Acquire and prepare source chess data for training and evaluation.",
    )
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    acquire_parser = data_commands.add_parser(
        "acquire",
        help="Download and verify a configured source archive.",
    )
    acquire_parser.add_argument(
        "output", type=Path, help="Artifact root where raw/ will be written."
    )
    acquire_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML source and archive selection.",
    )
    acquire_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict dotted TOML override; may be repeated.",
    )
    acquire_parser.set_defaults(handler=_run_data_acquire)

    prepare_parser = data_commands.add_parser(
        "prepare",
        help="Normalize a PGN file into Parquet and manifest artifacts.",
    )
    prepare_parser.add_argument("input", type=Path, help="Raw source PGN file.")
    prepare_parser.add_argument(
        "output", type=Path, help="Artifact root for normalized/ and manifests/."
    )
    prepare_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML source and preprocessing selection.",
    )
    prepare_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict dotted TOML override; may be repeated.",
    )
    prepare_parser.set_defaults(handler=_run_data_prepare)

    train_parser = subcommands.add_parser(
        "train",
        help="Run bounded move-model training from explicit configuration.",
    )
    train_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML training, model, and data selection.",
    )
    train_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict dotted TOML override; may be repeated.",
    )
    train_parser.set_defaults(handler=_run_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Anthro Chess command-line interface."""
    arguments = build_parser().parse_args(argv)
    handler: CommandHandler = arguments.handler
    return handler(arguments)


def _run_smoke(_arguments: argparse.Namespace) -> int:
    print(f"Anthro Chess {__version__} is installed and ready.")
    return 0


def _run_data_acquire(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.data import (
        DataPreparationError,
        PrepareConfig,
        acquire_archive,
    )

    try:
        resolved = load_config(
            PrepareConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        result = acquire_archive(arguments.output, resolved)
    except (ConfigError, DataPreparationError) as error:
        print(f"anthro data acquire: {error}", file=sys.stderr)
        return 2

    disposition = "Reused" if result.reused else "Acquired"
    print(f"{disposition} verified archive: {result.archive_path}")
    print(f"SHA-256: {result.sha256}")
    return 0


def _run_data_prepare(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.data import DataPreparationError, PrepareConfig, prepare_pgn

    try:
        resolved = load_config(
            PrepareConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        result = prepare_pgn(arguments.input, arguments.output, resolved)
    except (ConfigError, DataPreparationError) as error:
        print(f"anthro data prepare: {error}", file=sys.stderr)
        return 2

    print(
        f"Prepared {result.accepted_games} game(s); rejected {result.rejected_games}."
    )
    if len(result.normalized_paths) == 1:
        print(f"Normalized: {result.normalized_path}")
    else:
        print(
            f"Normalized: {len(result.normalized_paths)} shard(s) under "
            f"{result.normalized_paths[0].parent}"
        )
    print(f"Manifest: {result.manifest_path}")
    return 0


def _run_train(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.training import TrainingConfig, TrainingError, run_training

    try:
        resolved = load_config(
            TrainingConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        result = run_training(resolved)
    except (ConfigError, TrainingError) as error:
        print(f"anthro train: {error}", file=sys.stderr)
        return 2

    print(f"Completed {result.steps} optimizer step(s).")
    print(f"Run: {result.run_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Checkpoint: {result.checkpoint_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - console scripts call main directly
    raise SystemExit(main())
