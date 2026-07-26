"""Command-line interface for Anthro Chess."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from anthro_chess import __version__
from anthro_chess.application_logging import (
    DEFAULT_LOG_LEVEL,
    LOG_LEVEL_NAMES,
    configure_application_logging,
)
from anthro_chess.config import ResolvedConfig

if TYPE_CHECKING:
    from anthro_chess.data import SequenceDataConfig
    from anthro_chess.evaluation import PoolConfig
    from anthro_chess.training import TrainingConfig

CommandHandler = Callable[[argparse.Namespace], int]
logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVEL_NAMES,
        default=DEFAULT_LOG_LEVEL,
        help="Operational log verbosity on standard error (default: %(default)s).",
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
        "output",
        type=Path,
        nargs="?",
        help=(
            "Artifact root where raw/ will be written. Defaults beneath "
            "ANTHRO_CHESS_DATA_ROOT."
        ),
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
    prepare_parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Raw source PGN file. Defaults to the configured acquired archive.",
    )
    prepare_parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help=(
            "Artifact root for normalized/ and manifests/. Defaults beneath "
            "ANTHRO_CHESS_DATA_ROOT."
        ),
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

    eval_parser = subcommands.add_parser(
        "eval",
        help="Build and inspect frozen evaluation inputs.",
    )
    eval_commands = eval_parser.add_subparsers(dest="eval_command", required=True)
    freeze_parser = eval_commands.add_parser(
        "freeze",
        help="Freeze the held-out test split into a checksummed evaluation pool.",
    )
    freeze_parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help=(
            "Directory for the pool artifact. Defaults beneath ANTHRO_CHESS_DATA_ROOT."
        ),
    )
    freeze_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML evaluation-pool selection.",
    )
    freeze_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict dotted TOML override; may be repeated.",
    )
    freeze_parser.set_defaults(handler=_run_eval_freeze)

    report_parser = eval_commands.add_parser(
        "report",
        help="Show the compact benchmark delta view over the results store.",
    )
    _add_store_argument(report_parser)
    report_parser.add_argument(
        "--current",
        help="Checkpoint label to report (default: the most recently recorded).",
    )
    report_parser.add_argument(
        "--baseline",
        help="Checkpoint label to compare against (default: the previous one).",
    )
    report_parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Restrict the report to one metric family; may be repeated.",
    )
    report_parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Restrict the report to one metric; may be repeated.",
    )
    report_parser.add_argument(
        "--history",
        metavar="METRIC",
        help="Show one metric's full recorded history instead of a delta.",
    )
    report_parser.add_argument(
        "--provenance",
        action="store_true",
        help="Also show how the two compared checkpoints were produced.",
    )
    report_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s).",
    )
    report_parser.set_defaults(handler=_run_eval_report)

    metrics_parser = eval_commands.add_parser(
        "metrics",
        help="List registered metric families, metrics, and their directions.",
    )
    metrics_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s).",
    )
    metrics_parser.set_defaults(handler=_run_eval_metrics)

    bridge_parser = eval_commands.add_parser(
        "bridge",
        help="Record, list, or revoke an explicit series bridge.",
    )
    bridge_commands = bridge_parser.add_subparsers(
        dest="bridge_command",
        required=True,
    )
    bridge_add_parser = bridge_commands.add_parser(
        "add",
        help="Assert that two fingerprints name the same series.",
    )
    _add_store_argument(bridge_add_parser)
    bridge_add_parser.add_argument("--from", dest="from_fingerprint", required=True)
    bridge_add_parser.add_argument("--to", dest="to_fingerprint", required=True)
    bridge_add_parser.add_argument(
        "--reason",
        required=True,
        help="Why the fingerprint moved for a reason independent of the metric.",
    )
    bridge_add_parser.add_argument(
        "--author",
        required=True,
        help="Who is asserting the equivalence.",
    )
    bridge_add_parser.set_defaults(handler=_run_eval_bridge_add)

    bridge_list_parser = bridge_commands.add_parser(
        "list",
        help="List recorded bridges.",
    )
    _add_store_argument(bridge_list_parser)
    bridge_list_parser.set_defaults(handler=_run_eval_bridge_list)

    bridge_revoke_parser = bridge_commands.add_parser(
        "revoke",
        help="Remove a recorded bridge.",
    )
    _add_store_argument(bridge_revoke_parser)
    bridge_revoke_parser.add_argument("bridge_id")
    bridge_revoke_parser.set_defaults(handler=_run_eval_bridge_revoke)

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
    configure_application_logging(level=arguments.log_level, stream=sys.stderr)
    handler: CommandHandler = arguments.handler
    logger.debug("Starting command %s", arguments.command)
    return_code = handler(arguments)
    logger.debug("Completed command %s with status %s", arguments.command, return_code)
    return return_code


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
        archive = resolved.value.archive
        artifact_name = (
            archive.artifact_name
            if archive is not None and archive.artifact_name is not None
            else resolved.value.artifact_name
        )
        output = _data_output_path(arguments.output, artifact_name)
        result = acquire_archive(output, resolved)
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
        output = _data_output_path(arguments.output, resolved.value.artifact_name)
        input_path = arguments.input
        if input_path is None:
            if resolved.value.archive is None:
                raise ConfigError(
                    "input path is required because the selected data "
                    "configuration has no archive"
                )
            archive_root = _data_output_path(
                arguments.output,
                resolved.value.archive.artifact_name or resolved.value.artifact_name,
            )
            input_path = archive_root / "raw" / resolved.value.archive.file_name
        result = prepare_pgn(input_path, output, resolved)
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


def _run_eval_freeze(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.evaluation import EvaluationPoolError, PoolConfig, freeze_pool

    try:
        resolved = load_config(
            PoolConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        resolved = _resolve_pool_roots(resolved, arguments.set)
        output = _data_output_path(arguments.output, resolved.value.pool_id)
        result = freeze_pool(resolved, output)
    except (ConfigError, EvaluationPoolError) as error:
        print(f"anthro eval freeze: {error}", file=sys.stderr)
        return 2

    print(f"Froze {result.games} game(s) and {result.plies} ply/plies.")
    print(f"Pool: {result.games_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Identity: {result.game_ids_sha256}")
    return 0


def _add_store_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store",
        type=Path,
        help=(
            "Committed results store directory. Defaults to "
            "ANTHRO_CHESS_RESULTS_ROOT or ./results."
        ),
    )


def _run_eval_report(arguments: argparse.Namespace) -> int:
    from anthro_chess.evaluation.results import (
        BridgeIndex,
        ReportError,
        ResultsStore,
        ResultsStoreError,
        build_delta_report,
        build_history,
        render_history,
        render_provenance,
        render_report,
        resolve_store_root,
    )

    try:
        store = ResultsStore(resolve_store_root(arguments.store))
        results = store.results()
        bridges = BridgeIndex(store.bridges())
        if arguments.history is not None:
            history = build_history(results, bridges, arguments.history)
            if arguments.format == "json":
                print(json.dumps(history.as_record(), indent=2, sort_keys=True))
            else:
                print(render_history(history), end="")
            return 0
        report = build_delta_report(
            results,
            bridges,
            current=arguments.current,
            baseline=arguments.baseline,
            families=arguments.family or None,
            metrics=arguments.metric or None,
        )
    except (ReportError, ResultsStoreError) as error:
        print(f"anthro eval report: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(json.dumps(report.as_record(), indent=2, sort_keys=True))
        return 0
    print(render_report(report), end="")
    if arguments.provenance:
        print()
        print(render_provenance(report), end="")
    return 0


def _run_eval_metrics(arguments: argparse.Namespace) -> int:
    from anthro_chess.evaluation.results import iter_registry, registry_record

    if arguments.format == "json":
        print(json.dumps(registry_record(), indent=2, sort_keys=True))
        return 0
    for family, metrics in iter_registry():
        print(f"{family.identifier}  {family.title}")
        if not metrics:
            print("  no metric is registered for this family yet")
        for metric in metrics:
            projection = metric.projection or "no data dependency"
            print(
                f"  {metric.identifier:<44} {metric.direction.value:<18} "
                f"v{metric.definition_version}  {projection}"
            )
    return 0


def _run_eval_bridge_add(arguments: argparse.Namespace) -> int:
    from anthro_chess.evaluation.results import (
        ResultsStore,
        ResultsStoreError,
        build_bridge,
        resolve_store_root,
    )

    try:
        bridge = build_bridge(
            from_fingerprint=arguments.from_fingerprint,
            to_fingerprint=arguments.to_fingerprint,
            reason=arguments.reason,
            author=arguments.author,
        )
        path = ResultsStore(resolve_store_root(arguments.store)).append_bridge(bridge)
    except (ResultsStoreError, ValueError) as error:
        print(f"anthro eval bridge add: {error}", file=sys.stderr)
        return 2

    print(f"Recorded bridge {bridge.bridge_id}")
    print(f"Path: {path}")
    return 0


def _run_eval_bridge_list(arguments: argparse.Namespace) -> int:
    from anthro_chess.evaluation.results import (
        ResultsStore,
        ResultsStoreError,
        resolve_store_root,
    )

    try:
        bridges = ResultsStore(resolve_store_root(arguments.store)).bridges()
    except ResultsStoreError as error:
        print(f"anthro eval bridge list: {error}", file=sys.stderr)
        return 2

    if not bridges:
        print("No bridges are recorded.")
        return 0
    for bridge in bridges:
        print(
            f"{bridge.bridge_id}  {bridge.from_fingerprint[:12]} -> "
            f"{bridge.to_fingerprint[:12]}  {bridge.author}: {bridge.reason}"
        )
    return 0


def _run_eval_bridge_revoke(arguments: argparse.Namespace) -> int:
    from anthro_chess.evaluation.results import (
        ResultsStore,
        ResultsStoreError,
        resolve_store_root,
    )

    try:
        path = ResultsStore(resolve_store_root(arguments.store)).revoke_bridge(
            arguments.bridge_id
        )
    except ResultsStoreError as error:
        print(f"anthro eval bridge revoke: {error}", file=sys.stderr)
        return 2

    print(f"Revoked bridge {arguments.bridge_id}")
    print(f"Path: {path}")
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
        resolved = _resolve_training_roots(resolved, arguments.set)
        result = run_training(resolved)
    except (ConfigError, TrainingError) as error:
        print(f"anthro train: {error}", file=sys.stderr)
        return 2

    print(f"Completed {result.steps} optimizer step(s).")
    print(f"Run: {result.run_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Checkpoint: {result.checkpoint_path}")
    if result.validation is not None:
        print(
            "Validation: "
            f"raw_move_loss={result.validation.move_loss:.6f} "
            f"legal_move_loss={result.validation.legal_move_loss:.6f} "
            "uniform_over_legal="
            f"{result.validation.uniform_over_legal_move_loss:.6f}"
        )
    return 0


def _resolve_pool_roots(
    resolved: ResolvedConfig[PoolConfig],
    overrides: Sequence[str],
) -> ResolvedConfig[PoolConfig]:
    """Resolve checked-in relative source paths beneath the shared data root."""
    if not os.environ.get("ANTHRO_CHESS_DATA_ROOT", "").strip():
        return resolved

    config = resolved.value
    override_keys = {item.partition("=")[0] for item in overrides}
    update: dict[str, object] = {}
    for field_name in ("normalized", "manifest"):
        path = getattr(config, field_name)
        if not path.is_absolute() and field_name not in override_keys:
            update[field_name] = _rooted_artifact_path(
                _environment_root("ANTHRO_CHESS_DATA_ROOT"),
                path,
            )
    if not update:
        return resolved
    return ResolvedConfig(
        value=config.model_copy(update=update),
        provenance=resolved.provenance,
    )


def _data_output_path(output: Path | None, artifact_name: str) -> Path:
    if output is not None:
        return output
    root = _environment_root("ANTHRO_CHESS_DATA_ROOT")
    return root / artifact_name


def _environment_root(name: str) -> Path:
    value = os.environ.get(name)
    if value is None or not value.strip():
        from anthro_chess.config import ConfigError

        raise ConfigError(
            f"a directory must be provided explicitly or {name} must be set"
        )
    return Path(value).expanduser().resolve()


def _resolve_training_roots(
    resolved: ResolvedConfig[TrainingConfig],
    overrides: Sequence[str],
) -> ResolvedConfig[TrainingConfig]:
    """Resolve checked-in artifact paths beneath configured machine roots."""
    config = resolved.value
    update: dict[str, object] = {}
    override_keys = {item.partition("=")[0] for item in overrides}

    output_directory = config.output_directory
    if (
        not output_directory.is_absolute()
        and "output_directory" not in override_keys
        and os.environ.get("ANTHRO_CHESS_RUN_ROOT", "").strip()
    ):
        update["output_directory"] = _rooted_artifact_path(
            _environment_root("ANTHRO_CHESS_RUN_ROOT"),
            output_directory,
        )

    data_root_available = bool(os.environ.get("ANTHRO_CHESS_DATA_ROOT", "").strip())
    selections: tuple[tuple[str, SequenceDataConfig | None], ...] = (
        ("train", config.train),
        ("validation", config.validation),
    )
    for selection_name, selection in selections:
        if selection is None or not data_root_available:
            continue
        selection_update: dict[str, Path] = {}
        for field_name in ("normalized", "manifest"):
            path = getattr(selection, field_name)
            dotted_key = f"{selection_name}.{field_name}"
            if not path.is_absolute() and dotted_key not in override_keys:
                selection_update[field_name] = _rooted_artifact_path(
                    _environment_root("ANTHRO_CHESS_DATA_ROOT"),
                    path,
                )
        if selection_update:
            update[selection_name] = selection.model_copy(update=selection_update)

    if not update:
        return resolved
    return ResolvedConfig(
        value=config.model_copy(update=update),
        provenance=resolved.provenance,
    )


def _rooted_artifact_path(root: Path, configured_path: Path) -> Path:
    parts = configured_path.parts
    if parts and parts[0] == "artifacts":
        parts = parts[1:]
    return root.joinpath(*parts)


if __name__ == "__main__":  # pragma: no cover - console scripts call main directly
    raise SystemExit(main())
