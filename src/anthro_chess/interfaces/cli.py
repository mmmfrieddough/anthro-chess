"""Command-line interface for Anthro Chess."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
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
    from anthro_chess.evaluation import (
        CheckpointEvaluationConfig,
        CheckpointEvaluationResult,
        DecisionDecomposition,
        InferenceBenchmarkResult,
        LadderBenchmarkConfig,
        LadderBenchmarkResult,
        PoolConfig,
        PuzzleBenchmarkConfig,
        PuzzleBenchmarkResult,
        RolloutBenchmarkResult,
    )
    from anthro_chess.evaluation.results import BridgeIndex, ResultEnvelope
    from anthro_chess.evaluation.rollout import RolloutReading
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

    prepare_puzzles_parser = eval_commands.add_parser(
        "prepare-puzzles",
        help="Acquire and build the pinned external puzzle benchmark artifact.",
    )
    prepare_puzzles_parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Pinned Lichess puzzle archive; downloaded when omitted.",
    )
    prepare_puzzles_parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help=(
            "Directory for the puzzle artifact. Defaults beneath "
            "ANTHRO_CHESS_DATA_ROOT."
        ),
    )
    prepare_puzzles_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML puzzle-source and selection recipe.",
    )
    prepare_puzzles_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict dotted TOML override; may be repeated.",
    )
    prepare_puzzles_parser.set_defaults(handler=_run_eval_prepare_puzzles)

    run_parser = eval_commands.add_parser(
        "run",
        help="Evaluate a checkpoint over the frozen pool and record the result.",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML checkpoint-evaluation selection.",
    )
    run_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict dotted TOML override; may be repeated.",
    )
    _add_store_argument(run_parser)
    run_parser.add_argument(
        "--detail-root",
        type=Path,
        help=(
            "Machine-local detail-tier directory. Defaults to "
            "ANTHRO_CHESS_RESULT_DETAIL_ROOT or a directory beneath "
            "ANTHRO_CHESS_RUN_ROOT."
        ),
    )
    run_parser.add_argument(
        "--no-record",
        action="store_true",
        help="Compute and print results without writing them to the store.",
    )
    run_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s).",
    )
    run_parser.set_defaults(handler=_run_eval_run)

    puzzles_parser = eval_commands.add_parser(
        "puzzles",
        help="Measure rating response against the owned calibrated puzzle set.",
    )
    puzzles_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML puzzle-rating benchmark selection.",
    )
    puzzles_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict dotted TOML override; may be repeated.",
    )
    _add_store_argument(puzzles_parser)
    puzzles_parser.add_argument(
        "--detail-root",
        type=Path,
        help=(
            "Machine-local detail-tier directory. Defaults to "
            "ANTHRO_CHESS_RESULT_DETAIL_ROOT or a directory beneath "
            "ANTHRO_CHESS_RUN_ROOT."
        ),
    )
    puzzles_parser.add_argument(
        "--no-record",
        action="store_true",
        help="Compute and print results without writing them to the store.",
    )
    puzzles_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s).",
    )
    puzzles_parser.set_defaults(handler=_run_eval_puzzles)

    inference_parser = eval_commands.add_parser(
        "inference",
        help="Measure a checkpoint's move latency, throughput, and cold start.",
    )
    inference_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML inference-benchmark selection.",
    )
    inference_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict dotted TOML override; may be repeated.",
    )
    _add_store_argument(inference_parser)
    inference_parser.add_argument(
        "--detail-root",
        type=Path,
        help=(
            "Machine-local detail-tier directory. Defaults to "
            "ANTHRO_CHESS_RESULT_DETAIL_ROOT or a directory beneath "
            "ANTHRO_CHESS_RUN_ROOT."
        ),
    )
    inference_parser.add_argument(
        "--no-record",
        action="store_true",
        help=(
            "Measure and print without writing to the store. Use this on a "
            "machine that is doing other work, where the figures are real but "
            "do not belong in the committed history."
        ),
    )
    inference_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s).",
    )
    inference_parser.set_defaults(handler=_run_eval_inference)

    rollout_parser = eval_commands.add_parser(
        "rollout",
        help=(
            "Play a declared matrix of generated games and report what whole "
            "games look like."
        ),
    )
    rollout_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML rollout-benchmark selection.",
    )
    rollout_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict dotted TOML override; may be repeated.",
    )
    _add_store_argument(rollout_parser)
    rollout_parser.add_argument(
        "--detail-root",
        type=Path,
        help=(
            "Machine-local detail-tier directory. Defaults to "
            "ANTHRO_CHESS_RESULT_DETAIL_ROOT or a directory beneath "
            "ANTHRO_CHESS_RUN_ROOT."
        ),
    )
    rollout_parser.add_argument(
        "--no-record",
        action="store_true",
        help=(
            "Play and print without writing to the store. Use this for an "
            "exploratory reading at one temperature, which is real but does "
            "not belong in the committed history."
        ),
    )
    rollout_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s).",
    )
    rollout_parser.set_defaults(handler=_run_eval_rollout)

    ladder_parser = eval_commands.add_parser(
        "ladder",
        help=(
            "Play a self-play rating ladder and report the transfer function "
            "from configured to fitted rating, plus its temperature response."
        ),
    )
    ladder_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML rating-ladder selection.",
    )
    ladder_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict dotted TOML override; may be repeated.",
    )
    _add_store_argument(ladder_parser)
    ladder_parser.add_argument(
        "--detail-root",
        type=Path,
        help=(
            "Machine-local detail-tier directory. Defaults to "
            "ANTHRO_CHESS_RESULT_DETAIL_ROOT or a directory beneath "
            "ANTHRO_CHESS_RUN_ROOT."
        ),
    )
    ladder_parser.add_argument(
        "--no-record",
        action="store_true",
        help=(
            "Play and print without writing to the store. Use this for an "
            "exploratory ladder, which is real but does not belong in the "
            "committed history."
        ),
    )
    ladder_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s).",
    )
    ladder_parser.set_defaults(handler=_run_eval_ladder)

    bandwidth_parser = eval_commands.add_parser(
        "curve-bandwidth",
        help=(
            "Select each generated-play curve's bandwidth from the human "
            "reference. An offline step whose output is declared in code."
        ),
    )
    bandwidth_parser.add_argument(
        "pool",
        type=Path,
        help="Frozen evaluation pool the human reference is read from.",
    )
    bandwidth_parser.add_argument(
        "--maximum-games",
        type=int,
        help=(
            "Subsample the reference to this many games. Selection is over the "
            "whole reference by default, which is what a declared bandwidth "
            "should be chosen from."
        ),
    )
    bandwidth_parser.add_argument(
        "--maximum-rating-gap",
        type=int,
        default=200,
        help="Widest rating gap a reference game may have (default: %(default)s).",
    )
    bandwidth_parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=(64, 128, 256, 512, 1024, 2048, 4000, 6000),
        help=(
            "Candidate neighbour counts to score. The default reaches well past "
            "the declared bandwidth on purpose: a candidate set that stops at "
            "the optimum cannot show whether the optimum is interior."
        ),
    )
    bandwidth_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s).",
    )
    bandwidth_parser.set_defaults(handler=_run_eval_curve_bandwidth)

    decisions_parser = eval_commands.add_parser(
        "decisions",
        help=(
            "Separate decisions the model preferred badly from decisions "
            "sampling drew against the model."
        ),
    )
    decisions_source = decisions_parser.add_mutually_exclusive_group(required=True)
    decisions_source.add_argument(
        "--games",
        type=Path,
        help=(
            "Stored game-record payload. Its decisions already carry their "
            "policy, so no checkpoint is loaded."
        ),
    )
    decisions_source.add_argument(
        "--log",
        type=Path,
        help=(
            "UCI debug log of a played session. Its decisions are re-scored "
            "against the checkpoint named by --config."
        ),
    )
    decisions_parser.add_argument(
        "--config",
        type=Path,
        help="Explicit TOML model-runner selection; required with --log.",
    )
    decisions_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict dotted TOML override; may be repeated.",
    )
    decisions_parser.add_argument(
        "--output",
        type=Path,
        help="Write the full decomposition, per-decision records included, here.",
    )
    decisions_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s).",
    )
    decisions_parser.set_defaults(handler=_run_eval_decisions)

    report_parser = eval_commands.add_parser(
        "report",
        help="Show the compact benchmark delta view over the results store.",
    )
    _add_store_argument(report_parser)
    report_parser.add_argument(
        "--detail-root",
        type=Path,
        help=(
            "Machine-local detail-tier directory used for paired checkpoint "
            "floors. Defaults to ANTHRO_CHESS_RESULT_DETAIL_ROOT or a "
            "directory beneath ANTHRO_CHESS_RUN_ROOT."
        ),
    )
    report_parser.add_argument(
        "--pivot",
        choices=("checkpoint", "environment"),
        default="checkpoint",
        help=(
            "What varies. 'checkpoint' asks whether the model changed; "
            "'environment' pins the model and asks whether the machine, "
            "precision, or software did (default: %(default)s)."
        ),
    )
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

    budget_parser = eval_commands.add_parser(
        "budget",
        help=(
            "Report held-out quality against the training budget that bought "
            "it, joining the training-efficiency and held-out families."
        ),
    )
    _add_store_argument(budget_parser)
    budget_parser.add_argument(
        "--metric",
        default=None,
        help=(
            "Held-out quality metric to plot against budget "
            "(default: held_out.move_loss)."
        ),
    )
    budget_parser.add_argument(
        "--positions",
        type=int,
        action="append",
        default=[],
        metavar="COUNT",
        help=(
            "Processed-position budget to answer for; may be repeated. Reports "
            "the best recorded quality reached without exceeding it."
        ),
    )
    budget_parser.add_argument(
        "--seconds",
        type=float,
        action="append",
        default=[],
        metavar="SECONDS",
        help="Training wall-clock budget to answer for; may be repeated.",
    )
    budget_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s).",
    )
    budget_parser.set_defaults(handler=_run_eval_budget)

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

    noise_parser = eval_commands.add_parser(
        "noise",
        help="Characterize, list, and apply benchmark noise floors.",
    )
    noise_commands = noise_parser.add_subparsers(
        dest="noise_command",
        required=True,
    )
    noise_characterize_parser = noise_commands.add_parser(
        "characterize",
        help="Estimate a noise floor from recorded replicate measurements.",
    )
    _add_store_argument(noise_characterize_parser)
    noise_characterize_parser.add_argument(
        "--kind",
        choices=("evaluation", "training"),
        required=True,
        help=(
            "Which noise source the replicates vary. Data-sampling noise is "
            "bootstrapped by the evaluation run itself and is not estimated here."
        ),
    )
    noise_characterize_parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL",
        help="A recorded checkpoint label to use as one replicate; repeat it.",
    )
    noise_characterize_parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Restrict the characterization to one metric; may be repeated.",
    )
    noise_characterize_parser.add_argument(
        "--source",
        required=True,
        help="What the replicates are, such as which seeds produced them.",
    )
    noise_characterize_parser.add_argument(
        "--coverage",
        type=float,
        help=(
            "Normal coverage factor for the floor. Defaults to the two-sided "
            "95 percent factor."
        ),
    )
    noise_characterize_parser.set_defaults(handler=_run_eval_noise_characterize)

    noise_list_parser = noise_commands.add_parser(
        "list",
        help="List recorded noise characterizations and their floors.",
    )
    _add_store_argument(noise_list_parser)
    noise_list_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s).",
    )
    noise_list_parser.set_defaults(handler=_run_eval_noise_list)

    noise_plan_parser = noise_commands.add_parser(
        "plan",
        help="Report how many games an axis needs to resolve a given effect.",
    )
    _add_store_argument(noise_plan_parser)
    noise_plan_parser.add_argument("--metric", required=True)
    noise_plan_parser.add_argument(
        "--effect",
        type=float,
        required=True,
        help="The smallest metric difference the axis has to resolve.",
    )
    noise_plan_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s).",
    )
    noise_plan_parser.set_defaults(handler=_run_eval_noise_plan)

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
    _add_store_argument(train_parser)
    train_parser.add_argument(
        "--detail-root",
        type=Path,
        help=(
            "Machine-local detail-tier directory for the efficiency "
            "breakdown. Defaults to ANTHRO_CHESS_RESULT_DETAIL_ROOT or a "
            "directory beneath ANTHRO_CHESS_RUN_ROOT."
        ),
    )
    train_parser.add_argument(
        "--no-record",
        action="store_true",
        help=(
            "Run declared evaluation cadences and measure efficiency without "
            "writing either to the store."
        ),
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


def _run_eval_prepare_puzzles(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.evaluation import (
        PuzzleSetBuildConfig,
        PuzzleSetError,
        prepare_puzzle_set,
    )

    try:
        resolved = load_config(
            PuzzleSetBuildConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        output = _data_output_path(
            arguments.output,
            resolved.value.artifact_name,
        )
        result = prepare_puzzle_set(
            resolved,
            output,
            source_path=arguments.input,
        )
    except (ConfigError, PuzzleSetError) as error:
        print(f"anthro eval prepare-puzzles: {error}", file=sys.stderr)
        return 2

    disposition = "Reused" if result.source_reused else "Acquired"
    print(f"{disposition} verified source: {result.source_path}")
    print(f"Prepared {result.entries} puzzle(s): {result.puzzle_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Identity: {result.puzzles_sha256}")
    return 0


def _run_eval_run(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.evaluation import (
        CheckpointEvaluationConfig,
        CheckpointEvaluationError,
        LeakageError,
        evaluate_checkpoint,
    )
    from anthro_chess.evaluation.results import (
        DetailStore,
        ResultsStore,
        ResultsStoreError,
        resolve_detail_root,
        resolve_store_root,
    )

    try:
        resolved = load_config(
            CheckpointEvaluationConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        resolved = _resolve_evaluation_roots(resolved, arguments.set)
        store = (
            None
            if arguments.no_record
            else ResultsStore(resolve_store_root(arguments.store))
        )
        detail = (
            None
            if arguments.no_record
            else DetailStore(resolve_detail_root(arguments.detail_root))
        )
        result = evaluate_checkpoint(
            resolved,
            run_root=_optional_environment_root("ANTHRO_CHESS_RUN_ROOT"),
            store=store,
            detail=detail,
        )
    except (
        CheckpointEvaluationError,
        ConfigError,
        LeakageError,
        ResultsStoreError,
    ) as error:
        print(f"anthro eval run: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(json.dumps(result.as_record(), indent=2, sort_keys=True))
        return 0
    print(_render_evaluation(result), end="")
    return 0


def _render_evaluation(result: CheckpointEvaluationResult) -> str:
    from anthro_chess.evaluation.aggregation import PHASE_DIMENSION

    overall = result.slices.overall
    lines = [
        f"Checkpoint: {result.checkpoint.label} (step {result.checkpoint.step})",
        (
            f"Pool: {result.dataset.pool_id} v{result.dataset.pool_version} "
            f"view {result.dataset.view} "
            f"({result.view.selected_games} game(s), "
            f"{overall.position_count} position(s))"
        ),
        (
            f"Leakage: no overlap with {result.leakage.training_games} "
            f"{result.leakage.training_split} game(s) "
            f"[{result.leakage.algorithm}]"
        ),
        "",
        f"move_loss                 {overall.move_loss:.6f}",
        f"legal_move_loss           {overall.legal_move_loss:.6f}",
        f"uniform_over_legal        {overall.uniform_over_legal_move_loss:.6f}",
        f"top1_accuracy             {overall.accuracy(1):.6f}",
        f"top5_accuracy             {overall.accuracy(5):.6f}",
        f"mask_penalty              {overall.mask_penalty:.6f}",
        f"top1_illegal_rate         {overall.top1_illegal_rate:.6f}",
        "",
        "Legality and move loss by phase:",
    ]
    for name, summary in sorted(
        result.slices.dimensions.get(PHASE_DIMENSION, {}).items()
    ):
        lines.append(
            f"  {name:<12} mask_penalty={summary.mask_penalty:.6f} "
            f"move_loss={summary.move_loss:.6f} n={summary.position_count}"
        )
    dependency = result.dependency
    if dependency is not None:
        match_rate = dependency.cross_conditioning.match_rate
        response = dependency.within_game.response
        lines.extend(
            [
                "",
                (
                    "Rating dependency (a degradation to interpret against "
                    f"training maturity, at step {dependency.maturity.step}):"
                ),
                *(
                    f"  {item.conditioning.name:<10} "
                    f"degradation={item.degradation:+.6f}"
                    for item in dependency.corruptions
                ),
                f"  cross-conditioning match rate: {_optional(match_rate)}",
                f"  within-game response:          {_optional(response)}",
                f"  anchor policy divergence:      {dependency.anchor_divergence:.6f}",
                f"  anchor top-1 agreement:        "
                f"{dependency.anchor_agreement_rate:.6f}",
            ]
        )
    noise = result.noise
    if noise is not None:
        lines.extend(
            [
                "",
                (
                    f"Noise: data-sampling floors for {len(noise.floors)} metric(s) "
                    f"from {noise.replicates} resamples of "
                    f"{result.view.selected_games} game(s). "
                    "See `anthro eval noise list`."
                ),
            ]
        )
    if result.recorded_paths:
        lines.extend(["", *(f"Recorded: {path}" for path in result.recorded_paths)])
    else:
        lines.extend(["", "Recorded: nothing; this run did not write to the store"])
    return "\n".join(lines) + "\n"


def _run_eval_puzzles(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.evaluation import (
        PuzzleBenchmarkConfig,
        PuzzleBenchmarkError,
        benchmark_puzzles,
    )
    from anthro_chess.evaluation.results import (
        DetailStore,
        ResultsStore,
        ResultsStoreError,
        resolve_detail_root,
        resolve_store_root,
    )

    try:
        resolved = load_config(
            PuzzleBenchmarkConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        resolved = _resolve_puzzle_roots(resolved, arguments.set)
        store = (
            None
            if arguments.no_record
            else ResultsStore(resolve_store_root(arguments.store))
        )
        detail = (
            None
            if arguments.no_record
            else DetailStore(resolve_detail_root(arguments.detail_root))
        )
        result = benchmark_puzzles(
            resolved,
            run_root=_optional_environment_root("ANTHRO_CHESS_RUN_ROOT"),
            store=store,
            detail=detail,
        )
    except (ConfigError, PuzzleBenchmarkError, ResultsStoreError) as error:
        print(f"anthro eval puzzles: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(json.dumps(result.as_record(), indent=2, sort_keys=True))
        return 0
    print(_render_puzzles(result), end="")
    return 0


def _render_puzzles(result: PuzzleBenchmarkResult) -> str:
    lines = [
        f"Checkpoint: {result.checkpoint.label} (step {result.checkpoint.step})",
        (
            f"Puzzle set: {result.dataset.pool_id} v{result.dataset.pool_version} "
            f"({result.dataset.selected_games} puzzle(s))"
        ),
        f"Reference temperature: {result.reference_temperature:.3f}",
        "",
        "Configured rating response:",
    ]
    for rating in result.ratings:
        lines.append(
            f"  {rating.target_rating:>4}  "
            f"greedy first={rating.greedy_first_move_accuracy:.3f} "
            f"line={rating.greedy_line_completion:.3f} "
            f"fit={rating.greedy_fitted_puzzle_rating:7.1f}  "
            f"sampled first={rating.sampled_first_move_solve_rate:.3f} "
            f"line={rating.sampled_line_completion:.3f} "
            f"fit={rating.sampled_fitted_puzzle_rating:7.1f}  "
            f"curve gap={rating.greedy_curve_distance:.3f}/"
            f"{rating.sampled_curve_distance:.3f}"
        )
    lines.extend(
        [
            "",
            (
                f"Greedy slope={result.greedy_rating_slope:.4f}, "
                f"order={result.greedy_order_accuracy:.3f}"
            ),
            (
                f"Sampled slope={result.sampled_rating_slope:.4f}, "
                f"order={result.sampled_order_accuracy:.3f}"
            ),
            (
                f"Training overlap: {result.overlapping_puzzles}/"
                f"{result.dataset.selected_games} puzzle source games "
                f"({result.overlap_rate:.3%}) across "
                f"{result.training_games} training/validation games"
            ),
        ]
    )
    if result.recorded_path is None:
        lines.extend(["", "Recorded: nothing; this run did not write to the store"])
    else:
        lines.extend(["", f"Recorded: {result.recorded_path}"])
    return "\n".join(lines) + "\n"


def _run_eval_inference(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.evaluation import (
        InferenceBenchmarkConfig,
        InferenceBenchmarkError,
        benchmark_inference,
    )
    from anthro_chess.evaluation.results import (
        DetailStore,
        ResultsStore,
        ResultsStoreError,
        resolve_detail_root,
        resolve_store_root,
    )

    try:
        resolved = load_config(
            InferenceBenchmarkConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        store = (
            None
            if arguments.no_record
            else ResultsStore(resolve_store_root(arguments.store))
        )
        detail = (
            None
            if arguments.no_record
            else DetailStore(resolve_detail_root(arguments.detail_root))
        )
        result = benchmark_inference(
            resolved,
            run_root=_optional_environment_root("ANTHRO_CHESS_RUN_ROOT"),
            store=store,
            detail=detail,
        )
    except (ConfigError, InferenceBenchmarkError, ResultsStoreError) as error:
        print(f"anthro eval inference: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(json.dumps(result.as_record(), indent=2, sort_keys=True))
        return 0
    print(_render_inference(result), end="")
    return 0


def _render_inference(result: InferenceBenchmarkResult) -> str:
    execution = result.execution
    latency = result.reference_latency
    throughput = result.reference_throughput
    threads = (
        "" if execution.cpu_threads is None else f", {execution.cpu_threads} thread(s)"
    )
    lines = [
        f"Checkpoint: {result.checkpoint.label} (step {result.checkpoint.step})",
        (
            f"Execution: {execution.device} ({execution.device_name}) "
            f"{execution.precision}{threads}"
        ),
        f"Series: workload {execution.workload_sha256[:12]}",
        "",
        (
            f"Batch-one move latency at {latency.history_plies} plies "
            f"({latency.decisions} decision(s), warmup excluded):"
        ),
    ]
    lines.extend(
        f"  p{percentile:<3} {value:8.1f} ms"
        for percentile, value in sorted(latency.percentiles.items())
    )
    lines.extend(
        [
            f"  mean {latency.mean_ms:8.1f} ms "
            f"(min {latency.minimum_ms:.1f}, max {latency.maximum_ms:.1f})",
            "",
            "Where a decision spends its mean latency:",
            f"  encode    {latency.encode_mean_ms:8.1f} ms",
            f"  model     {latency.model_mean_ms:8.1f} ms",
            f"  remainder {latency.remainder_mean_ms:8.1f} ms (masking and sampling)",
            "",
            (
                f"Throughput at batch {throughput.batch_size}: "
                f"{throughput.decisions_per_second:.1f} decisions/s "
                f"({throughput.batch_mean_ms:.1f} ms per batch)"
            ),
            "",
            "Cold start, reported apart from steady state:",
            f"  model load     {result.cold_start.model_load_seconds:8.3f} s",
            f"  first decision {result.cold_start.first_decision_seconds:8.3f} s",
        ]
    )
    if len(result.latency_sweep) > 1:
        lines.extend(["", "Latency by history depth:"])
        lines.extend(
            f"  {sample.history_plies:>4} plies  p50 {sample.percentiles[50]:8.1f} ms  "
            f"mean {sample.mean_ms:8.1f} ms"
            for sample in result.latency_sweep
        )
    if len(result.throughput_sweep) > 1:
        lines.extend(["", "Throughput by batch size:"])
        lines.extend(
            f"  batch {sample.batch_size:>4}  "
            f"{sample.decisions_per_second:8.1f} decisions/s"
            for sample in result.throughput_sweep
        )
    if result.recorded_paths:
        lines.extend(["", *(f"Recorded: {path}" for path in result.recorded_paths)])
    else:
        lines.extend(["", "Recorded: nothing; this run did not write to the store"])
    return "\n".join(lines) + "\n"


def _run_eval_rollout(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.evaluation import (
        RolloutBenchmarkConfig,
        RolloutBenchmarkError,
        benchmark_rollout,
    )
    from anthro_chess.evaluation.results import (
        DetailStore,
        ResultsStore,
        ResultsStoreError,
        resolve_detail_root,
        resolve_store_root,
    )

    try:
        resolved = load_config(
            RolloutBenchmarkConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        store = (
            None
            if arguments.no_record
            else ResultsStore(resolve_store_root(arguments.store))
        )
        detail = (
            None
            if arguments.no_record
            else DetailStore(resolve_detail_root(arguments.detail_root))
        )
        result = benchmark_rollout(
            resolved,
            run_root=_optional_environment_root("ANTHRO_CHESS_RUN_ROOT"),
            store=store,
            detail=detail,
        )
    except (ConfigError, RolloutBenchmarkError, ResultsStoreError) as error:
        print(f"anthro eval rollout: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(json.dumps(result.as_record(), indent=2, sort_keys=True))
        return 0
    print(_render_rollout(result), end="")
    return 0


def _render_rollout(result: RolloutBenchmarkResult) -> str:
    from anthro_chess.evaluation.games import GameTermination

    lines = [
        f"Checkpoint: {result.checkpoint.label} (step {result.checkpoint.step})",
        f"Games: {result.games} across {len(result.cells)} matrix cell(s)",
    ]
    if result.view is not None:
        record = result.view.as_record()
        lines.append(
            f"Prefix view: {result.view.name} "
            f"({result.view.selected_games} of {result.view.eligible_games} "
            f"eligible game(s), prefix {record['prefix_plies']} plies)"
        )
    for cell in result.cells:
        distribution = cell.distribution
        unfinished = distribution.termination_counts.get(
            GameTermination.PLY_LIMIT.value, 0
        )
        lines.extend(
            [
                "",
                f"{cell.label}  "
                f"(series workload {cell.execution.workload_sha256[:12]})",
                (
                    f"  games          {distribution.games} from "
                    f"{cell.positions} position(s) over {len(cell.seeds)} seed(s)"
                ),
                (
                    f"  length         mean {distribution.mean_ply_count:.1f} plies "
                    f"({distribution.mean_generated_plies:.1f} generated)"
                ),
                f"  results        {_counts(distribution.result_counts)}",
                f"  terminations   {_counts(distribution.termination_counts)}",
                f"  unfinished     {unfinished} at the ply limit",
                (
                    f"  repetition     {distribution.repeated_games} repeated, "
                    f"{distribution.threefold_claimable_games} threefold-claimable, "
                    f"cycle fraction {distribution.mean_cycle_ply_fraction:.3f}"
                ),
                (
                    f"  diversity      {distribution.distinct_game_fraction:.3f} "
                    f"distinct games, {distribution.mean_distinct_move_fraction:.3f} "
                    "distinct moves"
                ),
                f"  repertoire     {_counts(distribution.repertoire_counts, limit=5)}",
                (
                    f"  waypoints      "
                    f"{distribution.waypoint_game_rate:.3f} of games stopped "
                    "before choosing"
                ),
                (
                    f"  book depth     {distribution.mean_book_ply:.1f} of "
                    f"{distribution.mean_available_ply:.1f} available plies "
                    f"({distribution.mean_consumed_fraction:.3f} consumed) over "
                    f"{distribution.classified_games} named game(s)"
                ),
            ]
        )
    for reading in result.readings:
        lines.extend(
            [
                "",
                f"Against matched human play — {reading.label}  "
                f"(series {reading.execution.workload_sha256[:12]})",
                (
                    f"  {reading.model_games} generated game(s) across ratings "
                    f"{', '.join(str(rating) for rating in reading.ratings)} "
                    f"vs {reading.human_games} human game(s)"
                ),
                (
                    f"  {'quantity':<16}{'conditional':>13}{'floor':>10}"
                    f"{'seed range':>22}{'pooled':>10}{'reads as':>18}"
                ),
            ]
        )
        for quantity, comparison in reading.comparisons.items():
            floor = (
                "-"
                if comparison.floors is None
                else f"{comparison.floors.conditional.value:.4f}"
            )
            spread = reading.seed_spread.get(quantity)
            seeded = (
                "-"
                if spread is None or spread.floor is None
                else (f"{spread.floor:.4f}")
            )
            lines.append(
                f"  {quantity.value:<16}"
                f"{comparison.conditional_distance:>13.4f}"
                f"{floor:>10}"
                f"{seeded:>10}"
                f"{comparison.pooled_distance:>10.4f}"
                f"{comparison.response.value:>22}"
            )
        replicates = max(
            (len(spread.distances) for spread in reading.seed_spread.values()),
            default=0,
        )
        if replicates > 1:
            # The seed range is a diagnostic, not a second floor: each seed
            # played a fraction of the games, so its reading is noisier and
            # biased high, and comparing its spread to the floor would compare
            # two different sample sizes.
            lines.append(
                f"  floor qualifies a delta; seed range is each of {replicates} "
                f"seeds read alone on {1 / replicates:.0%} of the games"
            )
        lines.extend(_render_unavailable(reading))
        lines.extend(_render_repertoire_drilldown(reading))
        lines.extend(_render_exact_repertoire(reading))
    if result.recorded_paths:
        lines.extend(
            ["", f"Recorded: {len(result.recorded_paths)} result(s)"],
        )
        lines.extend(f"  {path}" for path in result.recorded_paths)
    else:
        lines.extend(["", "Recorded: nothing; this run did not write to the store"])
    return "\n".join(lines) + "\n"


#: Categories the repertoire drill-down shows. Enough to see where the mass
#: went without turning a summary into the whole distribution, which lives in
#: the detail tier.
_DRILLDOWN_CATEGORIES = 8


def _render_unavailable(reading: RolloutReading) -> list[str]:
    """Name the quantities nothing could be compared on, and why."""

    if not reading.unavailable:
        return []
    return [
        f"  unavailable    {quantity.value}: {reason}"
        for quantity, reason in sorted(reading.unavailable.items())
    ]


def _render_repertoire_drilldown(reading: RolloutReading) -> list[str]:
    """Show the largest repertoire categories with their mass beside the delta.

    Family granularity is uneven — the broadest family holds a few hundred lines
    and the median holds a handful — so a delta read without the category's mass
    invites treating a swing on a narrow line as the same finding as one on a
    family half the corpus plays.
    """

    from anthro_chess.evaluation.reference import ComparedQuantity

    comparison = reading.comparisons.get(ComparedQuantity.REPERTOIRE)
    if comparison is None:
        return []
    shares = comparison.category_shares()[:_DRILLDOWN_CATEGORIES]
    if not shares:
        return []
    lines = [
        "  repertoire by family",
        f"    {'family':<40}{'mass':>8}{'model':>8}{'delta':>9}",
    ]
    lines.extend(
        f"    {share.category[:40]:<40}{share.mass:>8.3f}"
        f"{share.model:>8.3f}{share.delta:>+9.3f}"
        for share in shares
    )
    return lines


def _render_exact_repertoire(reading: RolloutReading) -> list[str]:
    """Report the exactly enumerated repertoire beside its pruning bound.

    The bound is not decoration. The walk is exact only above its threshold, so
    a distance quoted without the mass that stopped being expanded is a
    precision claim the reading does not support.
    """

    exact = reading.exact
    if exact is None:
        return []
    return [
        (
            f"  exact repertoire to {exact.plies} plies "
            f"(threshold {exact.threshold:g}, series "
            f"{exact.execution.workload_sha256[:12]})"
        ),
        (
            f"    conditional {exact.conditional_distance:.4f}  "
            f"pooled {exact.pooled_distance:.4f}  "
            f"waypoints {exact.waypoint_mass:.3f}"
        ),
        (
            f"    reached ply {exact.deepest_expanded_ply} of {exact.plies}; "
            f"uncommitted mass at most {exact.unsettled_mass:.3f} "
            f"({exact.pruned_mass:.3f} pruned in all)"
        ),
    ]


def _run_eval_ladder(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.evaluation import (
        LadderBenchmarkConfig,
        LadderBenchmarkError,
        benchmark_ladder,
    )
    from anthro_chess.evaluation.results import (
        DetailStore,
        ResultsStore,
        ResultsStoreError,
        resolve_detail_root,
        resolve_store_root,
    )

    try:
        resolved = _resolve_ladder_roots(
            load_config(
                LadderBenchmarkConfig,
                path=arguments.config,
                overrides=arguments.set,
            ),
            arguments.set,
        )
        store = (
            None
            if arguments.no_record
            else ResultsStore(resolve_store_root(arguments.store))
        )
        detail = (
            None
            if arguments.no_record
            else DetailStore(resolve_detail_root(arguments.detail_root))
        )
        result = benchmark_ladder(
            resolved,
            run_root=_optional_environment_root("ANTHRO_CHESS_RUN_ROOT"),
            store=store,
            detail=detail,
        )
    except (ConfigError, LadderBenchmarkError, ResultsStoreError) as error:
        print(f"anthro eval ladder: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(json.dumps(result.as_record(), indent=2, sort_keys=True))
        return 0
    print(_render_ladder(result), end="")
    return 0


def _render_ladder(result: LadderBenchmarkResult) -> str:
    fit = result.fit
    lines = [
        f"Checkpoint: {result.checkpoint.label} (step {result.checkpoint.step})",
        (
            f"Games: {result.games} scored across {len(result.pairings)} "
            f"pairing(s), {result.unfinished} unfinished at the ply limit"
        ),
        (
            f"Fit: {'converged' if fit.converged else 'did not converge'} after "
            f"{fit.iterations} iteration(s), anchored at {fit.anchor_rating:.0f} "
            f"on the {fit.anchor_basis} basis"
        ),
    ]
    if fit.clamped:
        lines.append(
            "  clamped        "
            + ", ".join(seat.label for seat in fit.clamped)
            + " (won or lost every game, so the fit has no finite estimate)"
        )
    if fit.unscored:
        lines.append(
            "  unplaced       "
            + ", ".join(seat.label for seat in fit.unscored)
            + " (no scored game)"
        )
    for reading in result.readings:
        lines.extend(
            [
                "",
                f"Ladder at {reading.label}  "
                f"(series {reading.execution.workload_sha256[:12]})",
                f"    {'configured':>10}{'fitted':>10}{'error':>9}",
            ]
        )
        lines.extend(
            f"    {rating:>10}{fitted:>10.0f}{fitted - rating:>+9.0f}"
            for rating, fitted in zip(reading.ratings, reading.fitted, strict=True)
        )
        lines.extend(
            [
                (
                    f"  ordering       {reading.order_accuracy:.3f} pairwise, "
                    f"{reading.adjacent_order_accuracy:.3f} adjacent"
                ),
                (
                    f"  transfer       slope {reading.slope:.3f}, span "
                    f"{reading.span:.0f}, ladder error {reading.ladder_error:.1f}"
                ),
            ]
        )
        if reading.inversions:
            lines.append(
                "  degrades at    "
                + ", ".join(f"{lower}-{upper}" for lower, upper in reading.inversions)
            )
    lines.extend(_render_ladder_seats(result))
    lines.extend(_render_temperature_response(result))
    if result.unavailable:
        lines.append("")
        lines.extend(
            f"Unavailable: {name}: {reason}"
            for name, reason in sorted(result.unavailable.items())
        )
    if result.recorded_paths:
        lines.extend(
            ["", f"Recorded: {len(result.recorded_paths)} result(s) to the store"]
        )
    else:
        lines.extend(["", "Recorded: nothing; this run did not write to the store"])
    return "\n".join(lines) + "\n"


def _render_ladder_seats(result: LadderBenchmarkResult) -> list[str]:
    """Show each seat's score beside its error profile.

    Strength and error profile are printed together on purpose: a temperature
    that preserves the score rate while moving the preferred-selection rate has
    changed the shape of the mistakes rather than their number, and that is
    invisible in either column alone.
    """

    lines = [
        "",
        "Seats",
        f"    {'seat':<16}{'games':>7}{'score':>8}{'fitted':>9}"
        f"{'preferred':>11}{'regret':>9}{'rank':>7}",
    ]
    for seat in result.seats:
        profile = seat.decisions
        lines.append(
            f"    {seat.label:<16}{seat.games:>7}{seat.score_rate:>8.3f}"
            + (
                f"{seat.fitted_rating:>9.0f}"
                if seat.fitted_rating is not None
                else f"{'-':>9}"
            )
            + (
                f"{profile.preferred_selection_rate:>11.3f}"
                f"{profile.policy_regret:>9.3f}{profile.selected_rank:>7.2f}"
                if profile is not None
                else f"{'-':>11}{'-':>9}{'-':>7}"
            )
        )
    return lines


def _render_temperature_response(result: LadderBenchmarkResult) -> list[str]:
    """Report what temperature cost, and how much conditioning resisted it."""

    response = result.response
    if response is None:
        return []
    lines = [
        "",
        f"Temperature response  (series {response.execution.workload_sha256[:12]})",
        (
            f"  conditioned    {response.conditioned_response:+.1f} rating points "
            "per unit temperature"
        ),
    ]
    if response.ablated_response is not None:
        lines.append(f"  ablated        {response.ablated_response:+.1f}")
    if response.attenuation is not None:
        lines.append(
            f"  attenuation    {response.attenuation:+.3f} of the ablated drift avoided"
        )
    elif response.attenuation_unavailable is not None:
        lines.append(
            f"  attenuation    unavailable: {response.attenuation_unavailable}"
        )
    lines.extend(
        f"    rating {rating:<6}{value:+.1f}" for rating, value in response.per_rating
    )
    return lines


def _run_eval_curve_bandwidth(arguments: argparse.Namespace) -> int:
    from anthro_chess.data.artifacts import read_normalized_rows
    from anthro_chess.evaluation import EvaluationPoolError, ViewConfig, load_pool
    from anthro_chess.evaluation.reference import (
        ReferenceConfig,
        ReferenceError,
        human_reference,
        select_bandwidths,
    )
    from anthro_chess.evaluation.views import apply_view

    try:
        pool = load_pool(arguments.pool)
        selection = apply_view(
            pool.games,
            ViewConfig(
                name="curve-bandwidth",
                maximum_games=arguments.maximum_games,
                require_ratings=True,
            ),
        )
        wanted = set(selection.game_ids)
        rows = [
            row
            for row in read_normalized_rows(pool.games_path)
            if int(row["game_id"]) in wanted
        ]
        reference = human_reference(
            rows,
            ReferenceConfig(maximum_rating_gap=arguments.maximum_rating_gap),
        )
        selections = select_bandwidths(
            reference,
            candidates=tuple(arguments.candidates),
        )
    except (EvaluationPoolError, ReferenceError, ValueError) as error:
        print(f"anthro eval curve-bandwidth: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(
            json.dumps(
                {
                    "reference": reference.as_record(),
                    "selections": {
                        quantity.value: chosen.as_record()
                        for quantity, chosen in selections.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    lines = [
        f"Human reference: {len(reference.games)} game(s)",
        f"Excluded: {_counts(reference.excluded)}",
        "",
        "Declare these in DECLARED_NEIGHBOURS and freeze them:",
    ]
    for quantity, chosen in selections.items():
        lines.append(
            f"  {quantity.value:<16} neighbours={chosen.neighbours:<6} "
            f"error={chosen.error:.6g}  (from {chosen.observations} observations)"
        )
        # The whole scored curve, because an optimum sitting on the largest
        # candidate is a boundary artifact rather than a selection, and only
        # the neighbouring errors show which it is.
        lines.append(
            "      "
            + "  ".join(
                f"{neighbours}:{error:.6g}" for neighbours, error in chosen.candidates
            )
        )
    print("\n".join(lines))
    return 0


def _counts(counts: Mapping[str, int], *, limit: int | None = None) -> str:
    """Render a count mapping most-frequent first, truncated when asked."""

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    shown = ordered if limit is None else ordered[:limit]
    rendered = ", ".join(f"{name}={count}" for name, count in shown)
    remaining = len(ordered) - len(shown)
    return f"{rendered} (+{remaining} more)" if remaining else rendered or "none"


def _run_eval_decisions(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.evaluation.decisions import (
        DecisionAnalysisConfig,
        DecisionDecompositionError,
        decompose_game_records,
    )
    from anthro_chess.evaluation.reconstruction import (
        ReconstructionError,
        decompose_played_log,
    )

    try:
        if arguments.games is not None:
            decomposition = decompose_game_records(arguments.games)
        else:
            resolved = load_config(
                DecisionAnalysisConfig,
                path=arguments.config,
                overrides=arguments.set,
            )
            decomposition = decompose_played_log(
                arguments.log,
                resolved.value.model,
                run_root=_optional_environment_root("ANTHRO_CHESS_RUN_ROOT"),
            )
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(
                json.dumps(decomposition.as_record(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
    except (
        ConfigError,
        DecisionDecompositionError,
        ReconstructionError,
        OSError,
    ) as error:
        print(f"anthro eval decisions: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(json.dumps(decomposition.as_record(), indent=2, sort_keys=True))
        return 0
    print(_render_decisions(decomposition), end="")
    return 0


def _render_decisions(decomposition: DecisionDecomposition) -> str:
    overall = decomposition.overall
    lines = [
        f"Decisions classified: {overall.decisions}",
    ]
    if decomposition.unscored_decisions:
        lines.append(
            f"Decisions without a policy to classify: "
            f"{decomposition.unscored_decisions} "
            "(a random or external-engine seat reports none)"
        )
    lines.extend(
        [
            "",
            "Where a decision came from:",
            (
                f"  model preference  {overall.followed_policy:6d}  "
                f"({overall.preferred_selection_rate:.3f})"
            ),
            (
                f"  sampling          {overall.departures:6d}  "
                f"({1.0 - overall.preferred_selection_rate:.3f})"
            ),
            "",
            "What the draws gave up, in untempered policy probability:",
            f"  mean over all decisions   {overall.policy_regret:.4f}",
            f"  mean over departures      {overall.departure_policy_regret:.4f}",
            f"  worst single decision     {overall.maximum_policy_regret:.4f}",
            "",
            "What the policy itself looked like:",
            f"  preferred action probability  {overall.preferred_probability:.4f}",
            f"  selected action probability    {overall.selected_probability:.4f}",
            f"  selected action rank          {overall.selected_rank:.2f}",
            f"  enabled actions               {overall.enabled_action_count:.1f}",
        ]
    )
    if len(decomposition.cells) > 1:
        lines.extend(
            [
                "",
                "By setting, because the balance between the two depends on both:",
            ]
        )
        lines.extend(
            f"  {cell.setting.label if cell.setting else 'pooled':<34} "
            f"{cell.decisions:6d} decisions  "
            f"preferred {cell.preferred_selection_rate:.3f}  "
            f"regret {cell.policy_regret:.4f}"
            for cell in decomposition.cells
        )
    return "\n".join(lines) + "\n"


def _optional(value: float | None) -> str:
    return "not computed" if value is None else f"{value:.6f}"


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
        DetailStore,
        NoiseFloorIndex,
        PairedFloorIndex,
        ReportError,
        ResultsStore,
        ResultsStoreError,
        build_delta_report,
        build_environment_report,
        build_history,
        render_history,
        render_provenance,
        render_report,
        resolve_optional_detail_root,
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
        floors = NoiseFloorIndex(store.characterizations(), bridges)
        detail_root = resolve_optional_detail_root(arguments.detail_root)
        comparison_floors = (
            None if detail_root is None else PairedFloorIndex(DetailStore(detail_root))
        )
        if arguments.pivot == "environment":
            report = build_environment_report(
                results,
                bridges,
                floors=floors,
                checkpoint=arguments.current,
                families=arguments.family or None,
                metrics=arguments.metric or None,
            )
        else:
            report = build_delta_report(
                results,
                bridges,
                floors=floors,
                comparison_floors=comparison_floors,
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


def _run_eval_noise_characterize(arguments: argparse.Namespace) -> int:
    from anthro_chess.evaluation.results import (
        DEFAULT_COVERAGE,
        REPLICATE_METHOD,
        BridgeIndex,
        MetricRegistryError,
        NoiseCharacterizationError,
        ResultsStore,
        ResultsStoreError,
        build_characterization,
        metric_definition,
        replicate_floors,
        resolve_store_root,
    )

    labels: list[str] = arguments.checkpoint
    if len(labels) < 2:
        print(
            "anthro eval noise characterize: name at least two checkpoints; a "
            "floor is the spread across replicates and one reading has none",
            file=sys.stderr,
        )
        return 2

    try:
        store = ResultsStore(resolve_store_root(arguments.store))
        results = store.results()
        bridges = BridgeIndex(store.bridges())
        wanted = [metric_definition(name).identifier for name in arguments.metric]
        replicates, skipped = _collect_replicates(results, bridges, labels, wanted)
        if not replicates:
            print(
                "anthro eval noise characterize: no metric is measured on the "
                "same series for every named checkpoint",
                file=sys.stderr,
            )
            return 2
        characterization = build_characterization(
            kind=arguments.kind,
            method=REPLICATE_METHOD,
            replicates=len(labels),
            coverage=(
                DEFAULT_COVERAGE if arguments.coverage is None else arguments.coverage
            ),
            source=arguments.source,
            floors=replicate_floors(
                replicates,
                coverage=(
                    DEFAULT_COVERAGE
                    if arguments.coverage is None
                    else arguments.coverage
                ),
            ),
        )
        path = store.append_characterization(characterization)
    except (
        MetricRegistryError,
        NoiseCharacterizationError,
        ResultsStoreError,
    ) as error:
        print(f"anthro eval noise characterize: {error}", file=sys.stderr)
        return 2

    print(
        f"Characterized {arguments.kind} noise over {len(labels)} replicate(s) "
        f"for {len(characterization.floors)} metric(s)."
    )
    for entry in characterization.floors:
        print(f"  {entry.metric:<44} floor {entry.floor:.6g}")
    for metric, reason in skipped:
        print(f"  {metric:<44} skipped: {reason}")
    print(f"Recorded: {path}")
    return 0


def _collect_replicates(
    results: Sequence[ResultEnvelope],
    bridges: BridgeIndex,
    labels: Sequence[str],
    metrics: Sequence[str],
) -> tuple[dict[str, list[tuple[str, float]]], list[tuple[str, str]]]:
    """Gather one measurement per named checkpoint for every eligible metric.

    A metric is eligible only when every named checkpoint measured it on the
    same series. Replicates drawn from different series would describe the
    spread of two different measurements rather than the noise in one.
    """

    from anthro_chess.evaluation.results import (
        ResultsStoreError,
        latest_measurement,
        registered_metrics,
        results_for_checkpoint,
    )

    by_label = {label: results_for_checkpoint(results, label) for label in labels}
    missing = [label for label, found in by_label.items() if not found]
    if missing:
        raise ResultsStoreError(
            f"no result is recorded for checkpoint(s): {', '.join(sorted(missing))}"
        )

    candidates = (
        list(metrics)
        if metrics
        else [definition.identifier for definition in registered_metrics()]
    )
    replicates: dict[str, list[tuple[str, float]]] = {}
    skipped: list[tuple[str, str]] = []
    for metric in candidates:
        found = [latest_measurement(by_label[label], metric) for label in labels]
        if any(item is None for item in found):
            if metrics:
                skipped.append((metric, "not measured for every named checkpoint"))
            continue
        values = [item[1] for item in found if item is not None]
        series = {bridges.series(value.fingerprint) for value in values}
        if len(series) > 1:
            skipped.append((metric, "the named checkpoints are not on one series"))
            continue
        replicates[metric] = [(value.fingerprint, value.value) for value in values]
    return replicates, skipped


def _run_eval_noise_list(arguments: argparse.Namespace) -> int:
    from anthro_chess.evaluation.results import (
        ResultsStore,
        ResultsStoreError,
        resolve_store_root,
    )

    try:
        store = ResultsStore(resolve_store_root(arguments.store))
        characterizations = store.characterizations()
    except ResultsStoreError as error:
        print(f"anthro eval noise list: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(
            json.dumps(
                [record.as_record() for record in characterizations],
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not characterizations:
        print("No noise characterization is recorded.")
        return 0
    for record in characterizations:
        print(
            f"{record.recorded_at.date().isoformat()}  {record.kind}  "
            f"{record.method}  {record.replicates} replicate(s)  {record.source}"
        )
        for entry in record.floors:
            units = (
                "" if entry.sampling_units is None else f"  n={entry.sampling_units}"
            )
            print(
                f"  {entry.metric:<44} floor {entry.floor:.6g}  "
                f"dispersion {entry.dispersion:.6g}{units}"
            )
    return 0


def _run_eval_noise_plan(arguments: argparse.Namespace) -> int:
    from anthro_chess.evaluation.results import (
        MetricRegistryError,
        NoiseCharacterizationError,
        ResultsStore,
        ResultsStoreError,
        games_to_resolve,
        metric_definition,
        resolve_store_root,
    )

    try:
        metric = metric_definition(arguments.metric).identifier
        store = ResultsStore(resolve_store_root(arguments.store))
        # Characterizations arrive in recording order, so the last data-sampling
        # record covering this metric is the one that still describes it.
        newest = None
        entry = None
        for record in store.characterizations():
            candidate = record.entry(metric) if record.kind == "data-sampling" else None
            if candidate is not None:
                newest, entry = record, candidate
        if newest is None or entry is None:
            print(
                f"anthro eval noise plan: no data-sampling floor is recorded for "
                f"{metric}",
                file=sys.stderr,
            )
            return 2
        required = games_to_resolve(entry, arguments.effect)
    except (
        MetricRegistryError,
        NoiseCharacterizationError,
        ResultsStoreError,
    ) as error:
        print(f"anthro eval noise plan: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(
            json.dumps(
                {
                    "metric": metric,
                    "effect": arguments.effect,
                    "required_games": required,
                    "measured_games": entry.sampling_units,
                    "measured_floor": entry.floor,
                    "source": newest.source,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(
        f"{metric}: resolving an effect of {arguments.effect:.6g} needs about "
        f"{required} game(s)."
    )
    print(
        f"Measured floor {entry.floor:.6g} over {entry.sampling_units} game(s) "
        f"({newest.source})."
    )
    return 0


def _run_eval_budget(arguments: argparse.Namespace) -> int:
    from anthro_chess.evaluation.results import (
        ReportError,
        ResultsStore,
        ResultsStoreError,
        resolve_store_root,
    )
    from anthro_chess.evaluation.results.budget import (
        DEFAULT_QUALITY_METRIC,
        build_budget_report,
        render_budget_report,
    )

    try:
        results = ResultsStore(resolve_store_root(arguments.store)).results()
        report = build_budget_report(
            results,
            metric=arguments.metric or DEFAULT_QUALITY_METRIC,
            position_budgets=arguments.positions,
            time_budgets=arguments.seconds,
        )
    except (ReportError, ResultsStoreError) as error:
        print(f"anthro eval budget: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(json.dumps(report.as_record(), indent=2, sort_keys=True))
        return 0
    print(render_budget_report(report), end="")
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
                f"v{metric.definition_version}  {metric.cost.value:<14} "
                f"{projection}"
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
    from anthro_chess.evaluation.results import (
        DetailStore,
        ResultsStore,
        ResultsStoreError,
        resolve_optional_detail_root,
        resolve_store_root,
    )
    from anthro_chess.training import TrainingConfig, TrainingError, run_training

    try:
        resolved = load_config(
            TrainingConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        resolved = _resolve_training_roots(resolved, arguments.set)
        store = (
            None
            if arguments.no_record
            else ResultsStore(resolve_store_root(arguments.store))
        )
        # The efficiency breakdown is a nice-to-have, not a reason to refuse a
        # run: an unconfigured machine records the summary tier and skips it.
        detail_root = (
            None
            if arguments.no_record
            else resolve_optional_detail_root(arguments.detail_root)
        )
        detail = None if detail_root is None else DetailStore(detail_root)
        result = run_training(resolved, store=store, detail=detail)
    except (ConfigError, ResultsStoreError, TrainingError) as error:
        print(f"anthro train: {error}", file=sys.stderr)
        return 2

    print(f"Completed {result.steps} optimizer step(s).")
    print(f"Run: {result.run_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Checkpoint: {result.checkpoint_path}")
    if result.readings:
        recorded = sum(len(reading.recorded_paths) for reading in result.readings)
        print(
            f"Cadence readings: {len(result.readings)} "
            f"({recorded} recorded in the results store)"
        )
    if result.validation is not None:
        print(
            "Validation: "
            f"raw_move_loss={result.validation.move_loss:.6f} "
            f"legal_move_loss={result.validation.legal_move_loss:.6f} "
            "uniform_over_legal="
            f"{result.validation.uniform_over_legal_move_loss:.6f}"
        )
    if result.efficiency is not None:
        from anthro_chess.training.efficiency import render_efficiency

        print()
        print(render_efficiency(result.efficiency), end="")
        if result.efficiency_paths:
            print(f"Recorded {len(result.efficiency_paths)} efficiency result(s).")
    return 0


def _resolve_evaluation_roots(
    resolved: ResolvedConfig[CheckpointEvaluationConfig],
    overrides: Sequence[str],
) -> ResolvedConfig[CheckpointEvaluationConfig]:
    """Resolve checked-in relative pool paths beneath the shared data root."""
    if not os.environ.get("ANTHRO_CHESS_DATA_ROOT", "").strip():
        return resolved

    config = resolved.value
    root = _environment_root("ANTHRO_CHESS_DATA_ROOT")
    override_keys = {item.partition("=")[0] for item in overrides}
    update: dict[str, object] = {}
    if not config.pool.is_absolute() and "pool" not in override_keys:
        update["pool"] = _rooted_artifact_path(root, config.pool)
    training = config.leakage.training_normalized
    if (
        training is not None
        and not training.is_absolute()
        and "leakage.training_normalized" not in override_keys
    ):
        update["leakage"] = config.leakage.model_copy(
            update={"training_normalized": _rooted_artifact_path(root, training)}
        )
    if not update:
        return resolved
    return ResolvedConfig(
        value=config.model_copy(update=update),
        provenance=resolved.provenance,
    )


def _resolve_puzzle_roots(
    resolved: ResolvedConfig[PuzzleBenchmarkConfig],
    overrides: Sequence[str],
) -> ResolvedConfig[PuzzleBenchmarkConfig]:
    """Resolve puzzle and training-overlap artifacts beneath the data root."""

    if not os.environ.get("ANTHRO_CHESS_DATA_ROOT", "").strip():
        return resolved
    config = resolved.value
    override_keys = {item.partition("=")[0] for item in overrides}
    root = _environment_root("ANTHRO_CHESS_DATA_ROOT")
    update: dict[str, Path] = {}
    if not config.puzzle_set.is_absolute() and "puzzle_set" not in override_keys:
        update["puzzle_set"] = _rooted_artifact_path(root, config.puzzle_set)
    if (
        not config.training_normalized.is_absolute()
        and "training_normalized" not in override_keys
    ):
        update["training_normalized"] = _rooted_artifact_path(
            root,
            config.training_normalized,
        )
    if not update:
        return resolved
    return ResolvedConfig(
        value=config.model_copy(update=update),
        provenance=resolved.provenance,
    )


def _optional_environment_root(name: str) -> Path | None:
    """Return a configured machine root, or ``None`` when it is unset."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return Path(value).expanduser().resolve()


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


def _resolve_ladder_roots(
    resolved: ResolvedConfig[LadderBenchmarkConfig],
    overrides: Sequence[str],
) -> ResolvedConfig[LadderBenchmarkConfig]:
    """Resolve the checked-in relative opening pool beneath the shared data root.

    The shipped selection names its pool the way every other checked-in artifact
    path is named, so it has to be rooted the same way `anthro eval run` roots
    its own pool. Without this the shipped configuration only works from a
    directory that happens to hold an `artifacts/` tree.
    """

    if not os.environ.get("ANTHRO_CHESS_DATA_ROOT", "").strip():
        return resolved

    config = resolved.value
    pool = config.openings.pool
    if (
        pool is None
        or pool.is_absolute()
        or "openings.pool" in {item.partition("=")[0] for item in overrides}
    ):
        return resolved
    rooted = _rooted_artifact_path(_environment_root("ANTHRO_CHESS_DATA_ROOT"), pool)
    return ResolvedConfig(
        value=config.model_copy(
            update={"openings": config.openings.model_copy(update={"pool": rooted})}
        ),
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
