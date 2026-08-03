"""Command-line interface for Anthro Chess."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anthro_chess import __version__
from anthro_chess.application_logging import (
    DEFAULT_LOG_LEVEL,
    LOG_LEVEL_NAMES,
    configure_application_logging,
)
from anthro_chess.config import ResolvedConfig
from anthro_chess.machine import (
    DATA_ROOT_VARIABLE,
    RUN_ROOT_VARIABLE,
    MachineReport,
    inspect_machine,
    optional_root,
    required_root,
)

if TYPE_CHECKING:
    from anthro_chess.data import SequenceDataConfig
    from anthro_chess.evaluation import (
        CheckpointEvaluationResult,
        DecisionDecomposition,
        InferenceBenchmarkResult,
        LadderBenchmarkResult,
        NoveltyBenchmarkResult,
        PoolConfig,
        PuzzleBenchmarkResult,
        RolloutBenchmarkResult,
        TerminationBenchmarkResult,
    )
    from anthro_chess.evaluation.results import (
        BridgeIndex,
        DetailStore,
        NoiseCharacterization,
        ResultEnvelope,
        ResultsStore,
    )
    from anthro_chess.evaluation.rollout import RolloutReading
    from anthro_chess.evaluation.suite import StepOutcome, SuitePlan, SuiteRun
    from anthro_chess.evaluation.termination import Guardrails
    from anthro_chess.training import TrainingConfig

CommandHandler = Callable[[argparse.Namespace], int]
logger = logging.getLogger(__name__)


# Flags repeated verbatim across subcommands, declared once and inherited
# through `parents=`. A subcommand that words a flag differently, or that
# shares it too narrowly to earn a parent, keeps its own definition.
_SET_FLAG = argparse.ArgumentParser(add_help=False)
_SET_FLAG.add_argument(
    "--set",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="Strict dotted TOML override; may be repeated.",
)

_STORE_FLAG = argparse.ArgumentParser(add_help=False)
_STORE_FLAG.add_argument(
    "--store",
    type=Path,
    help=(
        "Committed results store directory. Defaults to "
        "ANTHRO_CHESS_RESULTS_ROOT or ./results."
    ),
)

_DETAIL_ROOT_FLAG = argparse.ArgumentParser(add_help=False)
_DETAIL_ROOT_FLAG.add_argument(
    "--detail-root",
    type=Path,
    help=(
        "Machine-local detail-tier directory. Defaults to "
        "ANTHRO_CHESS_RESULT_DETAIL_ROOT or a directory beneath "
        "ANTHRO_CHESS_RUN_ROOT."
    ),
)

_FORMAT_FLAG = argparse.ArgumentParser(add_help=False)
_FORMAT_FLAG.add_argument(
    "--format",
    choices=("text", "json"),
    default="text",
    help="Output format (default: %(default)s).",
)


def _named_directory(value: str) -> Path:
    """Parse a directory that has to end in a name.

    A run is identified by its directory's name, so `.` or a trailing separator
    would leave it nameless and fail at the first recorded reading — after the
    training it was measuring has already run.
    """

    path = Path(value)
    if not path.name:
        raise argparse.ArgumentTypeError(f"{value!r} does not name a directory")
    return path


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
        parents=[_SET_FLAG],
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
    acquire_parser.set_defaults(handler=_run_data_acquire)

    prepare_parser = data_commands.add_parser(
        "prepare",
        help="Normalize a PGN file into Parquet and manifest artifacts.",
        parents=[_SET_FLAG],
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
    prepare_parser.set_defaults(handler=_run_data_prepare)

    eval_parser = subcommands.add_parser(
        "eval",
        help="Build and inspect frozen evaluation inputs.",
    )
    eval_commands = eval_parser.add_subparsers(dest="eval_command", required=True)
    freeze_parser = eval_commands.add_parser(
        "freeze",
        help="Freeze the held-out test split into a checksummed evaluation pool.",
        parents=[_SET_FLAG],
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
    freeze_parser.set_defaults(handler=_run_eval_freeze)

    prepare_puzzles_parser = eval_commands.add_parser(
        "prepare-puzzles",
        help="Acquire and build the pinned external puzzle benchmark artifact.",
        parents=[_SET_FLAG],
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
    prepare_puzzles_parser.set_defaults(handler=_run_eval_prepare_puzzles)

    suite_parser = eval_commands.add_parser(
        "suite",
        help=(
            "Run every benchmark against one checkpoint, composing their "
            "existing selections."
        ),
        parents=[_SET_FLAG, _STORE_FLAG, _DETAIL_ROOT_FLAG, _FORMAT_FLAG],
    )
    suite_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML suite selection.",
    )
    suite_parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run every benchmark at its own declared size. The default is the "
            "reduced sweep, because a sweep measured in hours is not a default "
            "anyone will run on a new checkpoint. A reduced view is its own "
            "series, so the two do not accumulate into one history."
        ),
    )
    suite_parser.add_argument(
        "--sweep-root",
        type=Path,
        help=(
            "Where the sweep keeps its ledger and the payloads it hands "
            "between steps. Defaults beneath ANTHRO_CHESS_RUN_ROOT."
        ),
    )
    recording = suite_parser.add_mutually_exclusive_group()
    recording.add_argument(
        "--no-record",
        action="store_true",
        help=(
            "Run the whole sweep without writing to the store. This is what a "
            "shakedown reading uses: evidence about the instruments rather "
            "than about the model."
        ),
    )
    recording.add_argument(
        "--record",
        action="append",
        default=[],
        metavar="BENCHMARK",
        help=(
            "Commit only the named benchmark's reading, overriding what the "
            "suite selection decided; may be repeated."
        ),
    )
    suite_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue the sweep this selection and checkpoint already started, "
            "skipping the benchmarks it finished."
        ),
    )
    suite_parser.add_argument(
        "--plan",
        action="store_true",
        help="Resolve and print what would run, without running any of it.",
    )
    suite_parser.set_defaults(handler=_run_eval_suite)

    run_parser = eval_commands.add_parser(
        "run",
        help="Evaluate a checkpoint over the frozen pool and record the result.",
        parents=[_SET_FLAG, _STORE_FLAG, _DETAIL_ROOT_FLAG, _FORMAT_FLAG],
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML checkpoint-evaluation selection.",
    )
    run_parser.add_argument(
        "--no-record",
        action="store_true",
        help="Compute and print results without writing them to the store.",
    )
    run_parser.set_defaults(handler=partial(_run_eval_benchmark, name="run"))

    puzzles_parser = eval_commands.add_parser(
        "puzzles",
        help="Measure rating response against the owned calibrated puzzle set.",
        parents=[_SET_FLAG, _STORE_FLAG, _DETAIL_ROOT_FLAG, _FORMAT_FLAG],
    )
    puzzles_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML puzzle-rating benchmark selection.",
    )
    puzzles_parser.add_argument(
        "--no-record",
        action="store_true",
        help="Compute and print results without writing them to the store.",
    )
    puzzles_parser.set_defaults(handler=partial(_run_eval_benchmark, name="puzzles"))

    novelty_parser = eval_commands.add_parser(
        "novelty",
        help=(
            "Measure what a checkpoint retains under a controlled dose of "
            "perturbation-derived novelty."
        ),
        parents=[_SET_FLAG, _STORE_FLAG, _DETAIL_ROOT_FLAG, _FORMAT_FLAG],
    )
    novelty_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML novelty dose-response selection.",
    )
    novelty_parser.add_argument(
        "--no-record",
        action="store_true",
        help=(
            "Compute and print without writing to the store. This is what a "
            "shakedown reading uses: evidence about the instrument rather than "
            "about the model."
        ),
    )
    novelty_parser.set_defaults(handler=partial(_run_eval_benchmark, name="novelty"))

    inference_parser = eval_commands.add_parser(
        "inference",
        help="Measure a checkpoint's move latency, throughput, and cold start.",
        parents=[_SET_FLAG, _STORE_FLAG, _DETAIL_ROOT_FLAG, _FORMAT_FLAG],
    )
    inference_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML inference-benchmark selection.",
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
    inference_parser.set_defaults(
        handler=partial(_run_eval_benchmark, name="inference")
    )

    rollout_parser = eval_commands.add_parser(
        "rollout",
        help=(
            "Play a declared matrix of generated games and report what whole "
            "games look like."
        ),
        parents=[_SET_FLAG, _STORE_FLAG, _DETAIL_ROOT_FLAG, _FORMAT_FLAG],
    )
    rollout_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML rollout-benchmark selection.",
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
    rollout_parser.set_defaults(handler=partial(_run_eval_benchmark, name="rollout"))

    termination_parser = eval_commands.add_parser(
        "termination",
        help=(
            "Measure how a checkpoint ends games against the human termination "
            "mix, with the premature-resignation guardrails."
        ),
        parents=[_SET_FLAG, _STORE_FLAG, _DETAIL_ROOT_FLAG, _FORMAT_FLAG],
    )
    termination_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML game-termination selection.",
    )
    termination_parser.add_argument(
        "--no-record",
        action="store_true",
        help=(
            "Measure and print without writing to the store. This is what a "
            "shakedown reading uses: real evidence about the benchmark that "
            "does not belong in the committed history."
        ),
    )
    termination_parser.set_defaults(
        handler=partial(_run_eval_benchmark, name="termination")
    )

    ladder_parser = eval_commands.add_parser(
        "ladder",
        help=(
            "Play a self-play rating ladder and report the transfer function "
            "from configured to fitted rating, plus its temperature response."
        ),
        parents=[_SET_FLAG, _STORE_FLAG, _DETAIL_ROOT_FLAG, _FORMAT_FLAG],
    )
    ladder_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML rating-ladder selection.",
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
    ladder_parser.set_defaults(handler=partial(_run_eval_benchmark, name="ladder"))

    bandwidth_parser = eval_commands.add_parser(
        "curve-bandwidth",
        help=(
            "Select each generated-play curve's bandwidth from the human "
            "reference. An offline step whose output is declared in code."
        ),
        parents=[_FORMAT_FLAG],
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
    bandwidth_parser.set_defaults(handler=_run_eval_curve_bandwidth)

    decisions_parser = eval_commands.add_parser(
        "decisions",
        help=(
            "Separate decisions the model preferred badly from decisions "
            "sampling drew against the model."
        ),
        parents=[_SET_FLAG, _FORMAT_FLAG],
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
        "--output",
        type=Path,
        help="Write the full decomposition, per-decision records included, here.",
    )
    decisions_parser.set_defaults(handler=_run_eval_decisions)

    report_parser = eval_commands.add_parser(
        "report",
        help="Show the compact benchmark delta view over the results store.",
        parents=[_STORE_FLAG, _DETAIL_ROOT_FLAG, _FORMAT_FLAG],
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
    report_parser.set_defaults(handler=_run_eval_report)

    tensorboard_parser = eval_commands.add_parser(
        "tensorboard",
        help="Project checkpoint history from the results store into TensorBoard.",
        parents=[_STORE_FLAG],
    )
    tensorboard_parser.add_argument(
        "output",
        type=Path,
        help=(
            "Disposable TensorBoard log directory. Must be outside the committed "
            "results store."
        ),
    )
    tensorboard_parser.set_defaults(handler=_run_eval_tensorboard)

    budget_parser = eval_commands.add_parser(
        "budget",
        help=(
            "Report held-out quality against the training budget that bought "
            "it, joining the training-efficiency and held-out families."
        ),
        parents=[_STORE_FLAG, _FORMAT_FLAG],
    )
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
    budget_parser.set_defaults(handler=_run_eval_budget)

    metrics_parser = eval_commands.add_parser(
        "metrics",
        help="List registered metric families, metrics, and their directions.",
        parents=[_FORMAT_FLAG],
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
        parents=[_SET_FLAG, _STORE_FLAG],
    )
    noise_characterize_parser.add_argument(
        "--kind",
        choices=("evaluation", "training", "execution"),
        required=True,
        help=(
            "Which noise source the replicates vary. Data-sampling noise is "
            "bootstrapped by the evaluation run itself and is not estimated "
            "here. Execution noise is measured rather than read from the "
            "store, so it takes --config instead of --checkpoint."
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
        "--config",
        type=Path,
        help=(
            "Execution noise only: the inference-benchmark selection to repeat. "
            "The floor it produces describes this machine under that workload."
        ),
    )
    noise_characterize_parser.add_argument(
        "--processes",
        type=int,
        help=(
            "Execution noise only: how many separate processes to measure in. "
            "A reading a report compares is one process's, so this is the "
            "count that decides the floor."
        ),
    )
    noise_characterize_parser.add_argument(
        "--repeats",
        type=int,
        help=(
            "Execution noise only: readings per process. These say how much of "
            "the spread a repeat inside one process reproduces."
        ),
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
    noise_characterize_parser.add_argument(
        "--confidence",
        type=float,
        help=(
            "How sure the floor is that the dispersion is no larger than it "
            "assumes. A measured dispersion is a point estimate that lands "
            "below the truth about half the time, so the floor is built from a "
            "chi-squared upper limit at this confidence instead. Defaults to "
            "95 percent."
        ),
    )
    noise_characterize_parser.set_defaults(handler=_run_eval_noise_characterize)

    noise_sample_parser = noise_commands.add_parser(
        "sample",
        help="Measure one process's repeated efficiency readings, recording nothing.",
        parents=[_SET_FLAG, _FORMAT_FLAG],
    )
    noise_sample_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML inference-benchmark selection to repeat.",
    )
    noise_sample_parser.add_argument(
        "--repeats",
        type=int,
        help="Readings to take in this process.",
    )
    noise_sample_parser.set_defaults(handler=_run_eval_noise_sample)

    noise_list_parser = noise_commands.add_parser(
        "list",
        help="List recorded noise characterizations and their floors.",
        parents=[_STORE_FLAG, _FORMAT_FLAG],
    )
    noise_list_parser.set_defaults(handler=_run_eval_noise_list)

    noise_plan_parser = noise_commands.add_parser(
        "plan",
        help="Report how many games an axis needs to resolve a given effect.",
        parents=[_STORE_FLAG, _FORMAT_FLAG],
    )
    noise_plan_parser.add_argument("--metric", required=True)
    noise_plan_parser.add_argument(
        "--effect",
        type=float,
        required=True,
        help="The smallest metric difference the axis has to resolve.",
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
        parents=[_STORE_FLAG],
    )
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
        parents=[_STORE_FLAG],
    )
    bridge_list_parser.set_defaults(handler=_run_eval_bridge_list)

    bridge_revoke_parser = bridge_commands.add_parser(
        "revoke",
        help="Remove a recorded bridge.",
        parents=[_STORE_FLAG],
    )
    bridge_revoke_parser.add_argument("bridge_id")
    bridge_revoke_parser.set_defaults(handler=_run_eval_bridge_revoke)

    train_parser = subcommands.add_parser(
        "train",
        help="Run bounded move-model training from explicit configuration.",
        parents=[_SET_FLAG, _STORE_FLAG, _DETAIL_ROOT_FLAG],
    )
    train_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML training, model, and data selection.",
    )
    train_parser.add_argument(
        "--no-record",
        action="store_true",
        help=(
            "Run declared evaluation cadences and measure efficiency without "
            "writing either to the store."
        ),
    )
    train_parser.add_argument(
        "--output-directory",
        type=_named_directory,
        help=(
            "Write the run here instead of beneath ANTHRO_CHESS_RUN_ROOT. For "
            "the occasional run that belongs somewhere specific; ordinary runs "
            "are placed by the run root."
        ),
    )
    train_parser.set_defaults(handler=_run_train)

    machine_parser = subcommands.add_parser(
        "machine",
        help="Report the configured artifact roots and what they hold.",
        description=(
            "Report what this machine is configured to hold: its artifact "
            "roots, the retained training runs and prepared data artifacts "
            "beneath them, and how the default model selection resolves. "
            "Exits nonzero when a root is configured in a way that would read "
            "as an empty machine."
        ),
        parents=[_FORMAT_FLAG],
    )
    machine_parser.set_defaults(handler=_run_machine)

    model_parser = subcommands.add_parser(
        "model",
        help="Maintain the machine-local default model selection.",
    )
    model_commands = model_parser.add_subparsers(dest="model_command", required=True)
    select_parser = model_commands.add_parser(
        "select",
        help="Record which retained run and checkpoint commands default to.",
        description=(
            "Write the machine-local default model selection. Commands and "
            "the UCI process resolve this record when no explicit checkpoint "
            "or run is configured."
        ),
    )
    select_parser.add_argument(
        "run",
        help="Retained run directory name, relative to the run root.",
    )
    select_parser.add_argument(
        "--checkpoint",
        help=(
            "Pin one checkpoint file name within the run. Omit to follow the "
            "run's latest pointer, so the selection tracks a continuing run."
        ),
    )
    select_parser.add_argument(
        "--run-root",
        type=Path,
        help=(
            "Directory holding retained training runs. Defaults to "
            "ANTHRO_CHESS_RUN_ROOT."
        ),
    )
    select_parser.set_defaults(handler=_run_model_select)
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


def _run_machine(arguments: argparse.Namespace) -> int:
    report = inspect_machine()
    if arguments.format == "json":
        print(json.dumps(report.as_record(), indent=2, sort_keys=True))
    else:
        print(_render_machine(report), end="")
    # A half-configured machine is the failure this command exists to catch, so
    # it has to be visible to whatever ran the command and not only to a reader.
    return 1 if report.problems else 0


def _render_machine(report: MachineReport) -> str:
    lines: list[str] = ["roots"]
    width = max(len(root.variable) for root in report.roots)
    for root in report.roots:
        if root.path is None:
            state = f"not set; {root.fallback}"
        elif root.exists:
            state = str(root.path)
        else:
            state = f"{root.path} (missing)"
        lines.append(f"  {root.variable:<{width}}  {state}")

    lines.append("")
    lines.append("retained runs")
    if not report.runs:
        lines.append(f"  none beneath {_root_note(report, RUN_ROOT_VARIABLE)}")
    for run in report.runs:
        latest = run.latest_checkpoint or "no latest pointer"
        record = "" if run.has_run_record else ", no run record"
        lines.append(f"  {run.name}  {run.checkpoints} checkpoint(s), {latest}{record}")

    lines.append("")
    lines.append("data artifacts")
    if not report.artifacts:
        lines.append(f"  none beneath {_root_note(report, DATA_ROOT_VARIABLE)}")
    for artifact in report.artifacts:
        lines.append(f"  {artifact.name}  {artifact.kind}")

    lines.append("")
    lines.append("default model selection")
    selection = report.selection
    if selection.resolved is not None:
        lines.append(f"  {selection.resolved['checkpoint_path']}")
        lines.append(f"  recorded in {selection.record_path}")
    elif selection.error is not None:
        lines.append(f"  unresolved: {selection.error}")
    else:
        lines.append("  not determined")

    for heading, entries in (
        ("problems", report.problems),
        ("not reported here", report.unavailable),
    ):
        if not entries:
            continue
        lines.append("")
        lines.append(heading)
        lines.extend(f"  {entry}" for entry in entries)
    return "\n".join(lines) + "\n"


def _root_note(report: MachineReport, variable: str) -> str:
    """Describe a root the way an empty listing beneath it needs to be read."""

    status = next(root for root in report.roots if root.variable == variable)
    if status.path is None:
        return f"{variable} (not set)"
    return str(status.path)


def _run_model_select(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError
    from anthro_chess.inference.config import LATEST_CHECKPOINT
    from anthro_chess.inference.selection import (
        ModelSelectionError,
        write_model_selection,
    )

    try:
        run_root = arguments.run_root or required_root(
            RUN_ROOT_VARIABLE,
            alternative="a run root must be provided with --run-root",
        )
        record_path = write_model_selection(
            run_root,
            run=arguments.run,
            checkpoint=arguments.checkpoint or LATEST_CHECKPOINT,
        )
    except (ConfigError, ModelSelectionError) as error:
        print(f"anthro model select: {error}", file=sys.stderr)
        return 2

    print(f"Default model selection: {record_path}")
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


def _run_eval_suite(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.evaluation.results import (
        DetailStore,
        ResultsStore,
        ResultsStoreError,
        resolve_detail_root,
        resolve_store_root,
    )
    from anthro_chess.evaluation.suite import (
        SuiteConfig,
        SuiteError,
        SuiteScale,
        resolve_suite,
        run_suite,
        sweep_directory,
    )

    scale = SuiteScale.FULL if arguments.full else SuiteScale.REDUCED
    try:
        plan = resolve_suite(
            load_config(SuiteConfig, path=arguments.config, overrides=arguments.set),
            scale=scale,
            no_record=arguments.no_record,
            record_only=arguments.record or None,
        )
    except (ConfigError, SuiteError) as error:
        print(f"anthro eval suite: {error}", file=sys.stderr)
        return 2

    if arguments.plan:
        if arguments.format == "json":
            print(json.dumps(plan.as_record(), indent=2, sort_keys=True))
            return 0
        print(_render_plan(plan), end="")
        return 0

    try:
        store = (
            ResultsStore(resolve_store_root(arguments.store))
            if plan.recording
            else None
        )
        detail = (
            DetailStore(resolve_detail_root(arguments.detail_root))
            if plan.recording
            else None
        )
        run = run_suite(
            plan,
            sweep_root=sweep_directory(_sweep_root(arguments.sweep_root), plan),
            run_root=_run_root(),
            store=store,
            detail=detail,
            resume=arguments.resume,
            observer=None if arguments.format == "json" else _report_step,
        )
    except (ConfigError, ResultsStoreError, SuiteError) as error:
        print(f"anthro eval suite: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(json.dumps(run.as_record(), indent=2, sort_keys=True))
    else:
        print(_render_sweep(run), end="")
    # A failed benchmark is a real outcome the sweep reports rather than an
    # error it raises, so the readings beside it survive; the exit status is
    # what makes it visible to whatever ran the sweep.
    return 1 if run.failed else 0


def _sweep_root(explicit: Path | None) -> Path:
    """Resolve where sweeps keep their ledgers on this machine."""

    if explicit is not None:
        return explicit
    return (
        required_root(
            RUN_ROOT_VARIABLE,
            alternative="a sweep directory must be provided with --sweep-root",
        )
        / "benchmark-sweeps"
    )


def _report_step(outcome: StepOutcome) -> None:
    """Print one finished benchmark's own reading, as its command would.

    Printed as the sweep goes rather than collected for the end: a sweep runs
    for long enough that a reading held back until the last benchmark finishes
    is a reading nobody sees until then.
    """

    from anthro_chess.evaluation.suite import StepStatus

    print(f"=== {outcome.name} [{outcome.status.value}, {outcome.seconds:.1f}s] ===")
    if outcome.status is not StepStatus.COMPLETED:
        print(f"  {outcome.note}")
        print()
        return
    renderer = _STEP_RENDERERS.get(outcome.name)
    if renderer is None or outcome.result is None:
        print(f"  {outcome.note or 'no reading to print'}")
    else:
        print(renderer(outcome.result), end="")
    print()


def _render_plan(plan: SuitePlan) -> str:
    """Render what a sweep would run, in the order it would run it."""

    committing = ", ".join(plan.recording) or "nothing"
    lines = [
        f"Suite: {plan.suite} ({plan.scale.value})",
        f"Checkpoint: {plan.checkpoint_label or 'the machine-local default'}",
        f"Plan: {plan.sha256[:12]}, {len(plan.steps)} step(s), committing {committing}",
        "",
    ]
    for position, step in enumerate(plan.steps, start=1):
        source = "reads another step's output" if step.source is None else step.source
        lines.append(
            f"  {position}. {step.name:<12} "
            f"{'record ' if step.record else '       '} {source}"
        )
        if step.benchmark.games_from is not None:
            lines.append(
                f"     after {step.benchmark.games_from}, whose games it reads"
            )
        for override in step.overrides:
            lines.append(f"     {override}")
    return "\n".join(lines) + "\n"


def _render_sweep(run: SuiteRun) -> str:
    """Render what a finished sweep ran, read, and cost."""

    plan = run.plan
    lines = [
        f"Suite: {plan.suite} ({plan.scale.value}), {len(run.outcomes)} step(s) "
        f"in {run.seconds:.1f}s",
        f"Sweep: {run.sweep_root}",
        "",
        f"  {'benchmark':<12} {'status':<10} {'seconds':>9} {'results':>8} "
        f"{'metrics':>8}",
    ]
    for outcome in run.outcomes:
        lines.append(
            f"  {outcome.name:<12} {outcome.status.value:<10} "
            f"{outcome.seconds:>9.1f} {outcome.results:>8} "
            f"{outcome.measurements:>8}"
        )
    recorded = sum(len(outcome.recorded_paths) for outcome in run.outcomes)
    lines.extend(
        [
            "",
            (
                f"Recorded: {recorded} result file(s)"
                if recorded
                else "Recorded: nothing; this sweep did not write to the store"
            ),
        ]
    )
    unfinished = [
        outcome for outcome in run.outcomes if outcome.status.value != "completed"
    ]
    if unfinished:
        lines.append("")
        lines.extend(
            f"{outcome.status.value}: {outcome.name} — {outcome.note}"
            for outcome in unfinished
        )
    return "\n".join(lines) + "\n"


def _result_stores(
    arguments: argparse.Namespace,
) -> tuple[ResultsStore | None, DetailStore | None]:
    """Return the committed and detail stores a benchmark records through.

    A committed summary references its detail payloads by path and digest, so a
    benchmark that records needs both tiers. `anthro train`, whose breakdown is
    best-effort, resolves an optional detail root instead and is not a caller.
    """

    from anthro_chess.evaluation.results import (
        DetailStore,
        ResultsStore,
        resolve_detail_root,
        resolve_store_root,
    )

    if arguments.no_record:
        return None, None
    return (
        ResultsStore(resolve_store_root(arguments.store)),
        DetailStore(resolve_detail_root(arguments.detail_root)),
    )


def _run_eval_benchmark(arguments: argparse.Namespace, *, name: str) -> int:
    """Run one benchmark from its selection file and render what it read.

    Every benchmark's subcommand is this, and the name is all that separates
    them: the registry entry holds the schema, the artifact paths to root and
    the errors a bad reading raises, and the same name selects the text view
    the sweep already prints that benchmark's reading through. A benchmark that
    drifts off what it declared now breaks its command and the suite together,
    rather than leaving two hand-written copies of one convention to disagree.
    """

    from anthro_chess.config import ConfigError
    from anthro_chess.evaluation.benchmarks import (
        benchmark_registry,
        resolve_benchmark,
        run_benchmark,
    )

    benchmark = benchmark_registry()[name]
    reading_failed: tuple[type[Exception], ...] = (ConfigError, *benchmark.errors)
    try:
        resolved = resolve_benchmark(
            benchmark,
            path=arguments.config,
            overrides=arguments.set,
        )
        store, detail = _result_stores(arguments)
        result = run_benchmark(
            benchmark,
            resolved,
            run_root=_run_root(),
            store=store,
            detail=detail,
        )
    except reading_failed as error:
        print(f"anthro eval {name}: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(json.dumps(result.as_record(), indent=2, sort_keys=True))
        return 0
    print(_STEP_RENDERERS[name](result), end="")
    return 0


def _recorded_lines(recorded_paths: Sequence[Path]) -> list[str]:
    """Return how one benchmark's render reports where its reading was written."""

    if not recorded_paths:
        return ["", "Recorded: nothing; this run did not write to the store"]
    return ["", *(f"Recorded: {path}" for path in recorded_paths)]


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
    lines.extend(_recorded_lines(result.recorded_paths))
    return "\n".join(lines) + "\n"


def _render_novelty(result: NoveltyBenchmarkResult) -> str:
    control = result.control
    lines = [
        f"Checkpoint: {result.checkpoint.label} (step {result.checkpoint.step})",
        (
            f"Pool: {result.dataset.pool_id} v{result.dataset.pool_version} "
            f"view {result.view.name!r} ({result.view.selected_games} game(s))"
        ),
        "",
        "Dose response (retention is against this checkpoint's own control):",
    ]
    for arm in result.arms:
        overall = arm.slices.overall
        # Paired on position: the control is read over the plies this arm
        # reached, so a truncated arm is not compared against positions it
        # never saw.
        keys = arm.measured_keys
        reference = control.paired_legality(keys)
        observed = arm.paired_legality(keys)
        lines.append(
            f"  dose={arm.dose:<6.3f} realized={arm.realized_dose:.3f}  "
            f"positions={arm.scored_positions:<6} "
            f"truncated={arm.truncated_games:<5} "
            f"legal_mass={overall.legal_mass:.4f} "
            f"retention={_render_ratio(observed.legal_mass, reference.legal_mass)} "
            f"mask_penalty={overall.mask_penalty:.4f}"
        )
    lines.append("")
    lines.append("Predicate retention by dose:")
    for arm in result.arms:
        keys = arm.measured_keys
        for predicate, reading in sorted(
            arm.predicates.items(), key=lambda item: item[0].value
        ):
            paired = control.paired_predicate(predicate, keys)
            rank = (
                "-"
                if reading.mean_best_rank is None
                else f"{reading.mean_best_rank:.2f}"
            )
            retention = (
                _MISSING_RATIO
                if paired is None
                else _render_ratio(reading.selected_rate, paired.selected_rate)
            )
            lines.append(
                f"  dose={arm.dose:<6.3f} {predicate.value:<20} "
                f"n={reading.opportunities:<6} "
                f"rate={reading.selected_rate:.4f} "
                f"retention={retention} "
                f"mass={reading.policy_mass:.4f} rank={rank}"
            )
    if result.recorded_paths:
        lines.append("")
        lines.append(f"Recorded {len(result.recorded_paths)} result file(s).")
    return "\n".join(lines) + "\n"


#: Placeholder for a retention with no reference to divide by, kept the same
#: width as a rendered ratio so the columns stay readable.
_MISSING_RATIO = "     -"


def _render_ratio(value: float, reference: float) -> str:
    """Render a retention ratio, naming an absent reference rather than faking one."""

    return _MISSING_RATIO if reference == 0.0 else f"{value / reference:.4f}"


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
    lines.extend(_recorded_lines(result.recorded_paths))
    return "\n".join(lines) + "\n"


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
    lines.extend(_recorded_lines(result.recorded_paths))
    return "\n".join(lines) + "\n"


#: Column widths for the rollout comparison table. The conditional and pooled
#: arms are rendered identically so a reader can tell at a glance that a column
#: under one arm's rule belongs to that arm and to nothing else.
_ROLLOUT_MINIMUM_QUANTITY_WIDTH = 16
_ROLLOUT_CELL_WIDTHS = (10, 9, 9, 9)
_ROLLOUT_ARM_WIDTH = sum(_ROLLOUT_CELL_WIDTHS)
_ROLLOUT_VERDICT_WIDTH = 19
_ROLLOUT_ARM_HEADER = "".join(
    f"{name:>{width}}"
    for name, width in zip(
        ("distance", "null", "floor", "seed"), _ROLLOUT_CELL_WIDTHS, strict=True
    )
)


def _rollout_arm_rule(name: str) -> str:
    """Return the rule that spans one arm's columns, and only that arm's.

    The leading space keeps the two arms' rules from joining into one, which
    would undo the grouping the rule exists to show.
    """

    return " " + f" {name} ".center(_ROLLOUT_ARM_WIDTH - 1, "-")


def _rollout_quantity_width(readings: Iterable[RolloutReading]) -> int:
    """Return the quantity column shared by every table in one rollout.

    Sized across the whole result rather than per reading, so two readings of
    the same suite print the same table rather than two that merely resemble
    each other. The names run to twenty-odd characters and a column narrower
    than the longest pushes that whole row out from under its own headings,
    which is the defect this table is being fixed for.
    """

    longest = max(
        (
            len(quantity.value)
            for reading in readings
            for quantity in reading.comparisons
        ),
        default=0,
    )
    return max(longest + 1, _ROLLOUT_MINIMUM_QUANTITY_WIDTH)


def _rollout_arm(
    distance: float,
    *,
    null: float | None,
    floor: float | None,
    seed: float | None,
) -> str:
    """Render one arm of a curve comparison beside its own three qualifiers.

    An arm is self-contained on purpose. The distance, the null level its
    verdict is judged against, its delta floor, and its seed diagnostic are all
    properties of the same reading, and every one of them exists separately for
    the conditional and the pooled arm.
    """

    cells = (
        format(distance, ".4f"),
        _optional_value(null, ".4f"),
        _optional_value(floor, ".4f"),
        _optional_value(seed, ".4f"),
    )
    return "".join(
        f"{cell:>{width}}"
        for cell, width in zip(cells, _ROLLOUT_CELL_WIDTHS, strict=True)
    )


def _render_comparison_table(reading: RolloutReading, width: int) -> list[str]:
    """Return one reading's comparison table, each arm beside its own numbers.

    Every quantity is read twice, rating-conditionally and pooled, and each of
    those readings has its own null level, its own delta floor, and its own seed
    spread. Rendering one arm's qualifier next to the other arm's distance is
    how the first full suite reading came to record a correct verdict as a
    contradiction, so the two arms are laid out as separate blocks under their
    own rules and share nothing but the quantity that names the row.
    """

    if not reading.comparisons:
        # Headings over nothing read as a table that found nothing rather than
        # one that was never filled. The unavailable lines say which quantities.
        return []
    lines = [
        (
            f"  {'':<{width}}"
            f"{_rollout_arm_rule('conditional')}"
            f"{_rollout_arm_rule('pooled')}"
        ),
        (
            f"  {'quantity':<{width}}"
            f"{_ROLLOUT_ARM_HEADER}{_ROLLOUT_ARM_HEADER}"
            f"{'reads as':>{_ROLLOUT_VERDICT_WIDTH}}"
        ),
    ]
    for quantity, comparison in reading.comparisons.items():
        spread = reading.seed_spread.get(quantity)
        references = comparison.references
        floors = comparison.floors
        lines.append(
            f"  {quantity.value:<{width}}"
            + _rollout_arm(
                comparison.conditional_distance,
                null=None if references is None else references.conditional,
                floor=None if floors is None else floors.conditional.value,
                seed=None if spread is None else spread.floor,
            )
            + _rollout_arm(
                comparison.pooled_distance,
                null=None if references is None else references.pooled,
                floor=None if floors is None else floors.pooled.value,
                seed=None if spread is None else spread.pooled_floor,
            )
            + f"{comparison.response.value:>{_ROLLOUT_VERDICT_WIDTH}}"
        )
    # Which column answers which question. A distance read on its own is judged
    # against its null; a distance read against another checkpoint is judged
    # against its floor.
    lines.extend(
        [
            "  null:  what a model that already matched would still read at,"
            " and what 'reads as' judges against",
            "  floor: how far this distance moves between runs, for reading"
            " it against another checkpoint",
        ]
    )
    replicates = max(
        (len(spread.distances) for spread in reading.seed_spread.values()),
        default=0,
    )
    if replicates > 1:
        # The seed spread is a diagnostic, not a third floor: each seed played a
        # fraction of the games, so its reading is noisier and biased high, and
        # comparing its spread to the floor would compare two different sample
        # sizes.
        lines.append(
            f"  seed:  each of {replicates} seeds read alone on "
            f"{1 / replicates:.0%} of the games, as a diagnostic rather "
            "than a floor"
        )
    return lines


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
    width = _rollout_quantity_width(result.readings)
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
            ]
        )
        lines.extend(_render_bandwidth(reading))
        lines.extend(_render_comparison_table(reading, width))
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


def _render_bandwidth(reading: RolloutReading) -> list[str]:
    """Show how far the smoother actually reached around each grid point.

    The declared bandwidth is a neighbour count, so quoting it back says nothing
    about this reading: it is the same number whatever the reference holds. What
    varies is the rating span those neighbours occupy, and that is what decides
    whether the grid resolves the points it plots. A span approaching the grid
    spacing means adjacent points were estimated from largely the same games, so
    the conditional distance is closer to the pooled one than the two column
    headings suggest.

    Quantities disagree on the span, because one that some games lack reaches
    further for its neighbours than the rest. The worst of them is what decides
    how much of the curve is really there, so that is the one reported.
    """

    comparisons = list(reading.comparisons.values())
    if not comparisons:
        return []
    spans = (
        max(comparison.points[index].bandwidth for comparison in comparisons)
        for index in range(len(reading.ratings))
    )
    return [
        "  bandwidth      reaches "
        + " ".join(f"±{span:.0f}" for span in spans)
        + " rating points at those grid points"
    ]


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


def _termination_arm(
    name: str,
    distance: float,
    *,
    null: float | None,
    floor: float | None,
) -> str:
    """Render one arm of the termination mix beside its own qualifiers."""

    return (
        f"  {name:<11} {distance:.4f}  "
        f"null {_optional_value(null, '.4f')}  "
        f"floor {_optional_value(floor, '.4f')}"
    )


def _render_termination(result: TerminationBenchmarkResult) -> str:
    lines = [
        f"Checkpoint: {result.checkpoint.label} (step {result.checkpoint.step})",
        f"Games: {result.games} generated against {result.reference_games} human "
        f"reference game(s)",
    ]
    for reading in result.generated:
        guardrails = reading.guardrails
        deficit = reading.deficit
        lines.extend(
            [
                "",
                f"{reading.label}  "
                f"(series workload {reading.execution.workload_sha256[:12]})",
                f"  games          {reading.games} across ratings "
                f"{', '.join(str(rating) for rating in reading.ratings)}",
                f"  terminations   {_counts(reading.category_counts)}",
                f"  resignations   {guardrails.resignations} "
                f"({_optional_rate(guardrails.premature_rate)} premature, "
                f"human {_optional_rate(guardrails.human_premature_rate)})",
                f"  deficit        median "
                f"{_optional_value(deficit.model_median)} pawns vs human "
                f"{_optional_value(deficit.human_median)} "
                f"(distance {_optional_value(deficit.distance, '.4f')})",
                f"  silent actions {_silent_actions(guardrails)}",
                f"  never ended    {guardrails.claimable_unfinished_games} claimable "
                f"({_optional_rate(guardrails.untimed_non_termination_rate)})",
            ]
        )
        for name, reason in sorted(reading.unavailable.items()):
            lines.append(f"  unavailable    {name}: {reason}")
    for mix in result.mixes:
        comparison = mix.comparison
        references = comparison.references
        floors = comparison.floors
        # One line per arm rather than one line for both. The verdict is
        # computed from each distance against its own null, so a layout that
        # puts one arm's qualifier beside the other arm's distance invites the
        # misreading #172 records.
        lines.extend(
            [
                "",
                f"Termination mix — {mix.label}  "
                f"(series {mix.execution.workload_sha256[:12]})",
                f"  {mix.model_games} generated vs {mix.human_games} human game(s)",
                # The realized span rather than the declared neighbour count,
                # for the reason the rollout reading gives: the count is the
                # same whatever the reference holds, and it is the span that
                # says whether the grid resolves the points it plots.
                "  bandwidth   reaches "
                + " ".join(f"±{point.bandwidth:.0f}" for point in comparison.points)
                + " rating points",
                _termination_arm(
                    "conditional",
                    comparison.conditional_distance,
                    null=None if references is None else references.conditional,
                    floor=None if floors is None else floors.conditional.value,
                ),
                _termination_arm(
                    "pooled",
                    comparison.pooled_distance,
                    null=None if references is None else references.pooled,
                    floor=None if floors is None else floors.pooled.value,
                ),
                f"  reads as    {comparison.response.value}",
            ]
        )
        for share in comparison.category_shares()[:6]:
            lines.append(
                f"    {share.category:<24}human {share.human:.3f}  "
                f"model {share.model:.3f}  delta {share.delta:+.3f}"
            )
    held_out = result.held_out
    if held_out is not None:
        lines.extend(
            [
                "",
                "Held-out resignation prediction",
                f"  {held_out.games} game(s), {held_out.resignation_plies} "
                f"resignation ply/plies, {held_out.move_plies} move ply/plies",
                f"  mass at resignation "
                f"{_optional_value(held_out.mass_at_resignation, '.5f')}  "
                f"at moves {_optional_value(held_out.mass_at_moves, '.5f')}  "
                f"separation {_optional_value(held_out.separation, '.5f')}",
            ]
        )
        for name, reason in sorted(held_out.unavailable.items()):
            lines.append(f"  unavailable    {name}: {reason}")
    for name, reason in sorted(result.unavailable.items()):
        lines.append(f"Unavailable: {name}: {reason}")
    if result.recorded_paths:
        lines.append("")
        lines.append(f"Recorded {len(result.recorded_paths)} result(s)")
    return "\n".join(lines) + "\n"


def _silent_actions(guardrails: Guardrails) -> str:
    """Return a readable summary of which enabled actions went unused."""

    from anthro_chess.chess import DRAW_CLAIM_ACTION_ID, RESIGNATION_ACTION_ID

    if not guardrails.enabled_terminal_actions:
        return "no terminal action was enabled"
    if not guardrails.silent_terminal_actions:
        return f"none of {len(guardrails.enabled_terminal_actions)} enabled"
    names = {RESIGNATION_ACTION_ID: "resignation", DRAW_CLAIM_ACTION_ID: "draw claim"}
    unused = ", ".join(
        names.get(action_id, str(action_id))
        for action_id in guardrails.silent_terminal_actions
    )
    return f"{unused} never selected"


def _optional_rate(value: float | None) -> str:
    """Return a percentage, or a dash when there was nothing to divide."""

    return "-" if value is None else f"{value:.1%}"


def _optional_value(value: float | None, spec: str = ".2f") -> str:
    """Return a formatted number, or a dash when the reading is unavailable."""

    return "-" if value is None else format(value, spec)


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
                run_root=_run_root(),
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


def _run_eval_tensorboard(arguments: argparse.Namespace) -> int:
    from anthro_chess.evaluation.results import (
        ResultsStore,
        ResultsStoreError,
        resolve_store_root,
    )

    try:
        from anthro_chess.evaluation.tensorboard import (
            TensorBoardProjectionError,
            project_results,
        )
    except ImportError as error:
        print(
            "anthro eval tensorboard: TensorBoard projection requires the "
            f"model dependencies: {error}",
            file=sys.stderr,
        )
        return 2

    try:
        store = ResultsStore(resolve_store_root(arguments.store))
        projection = project_results(
            store.results(),
            arguments.output,
            store_root=store.root,
        )
    except (ResultsStoreError, TensorBoardProjectionError) as error:
        print(f"anthro eval tensorboard: {error}", file=sys.stderr)
        return 2

    print(
        f"Projected {projection.points} points across {projection.runs} runs "
        f"and {projection.checkpoints} checkpoints into {projection.output}"
    )
    return 0


def _run_eval_noise_characterize(arguments: argparse.Namespace) -> int:
    from anthro_chess.evaluation.results import (
        DEFAULT_CONFIDENCE,
        DEFAULT_COVERAGE,
        REPLICATE_METHOD,
        BridgeIndex,
        MetricRegistryError,
        NoiseCharacterizationError,
        ResultsStore,
        ResultsStoreError,
        build_characterization,
        metric_column_width,
        metric_definition,
        replicate_floors,
        resolve_store_root,
    )

    if arguments.kind == "execution":
        return _run_eval_noise_characterize_execution(arguments)
    if arguments.config is not None or arguments.set:
        print(
            f"anthro eval noise characterize: --config and --set describe a "
            f"measurement to repeat, which a {arguments.kind} floor is not; it "
            "reads replicates the store already holds",
            file=sys.stderr,
        )
        return 2
    if arguments.processes is not None or arguments.repeats is not None:
        print(
            "anthro eval noise characterize: --processes and --repeats apply "
            "only to --kind execution",
            file=sys.stderr,
        )
        return 2

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
        coverage = (
            DEFAULT_COVERAGE if arguments.coverage is None else arguments.coverage
        )
        confidence = (
            DEFAULT_CONFIDENCE if arguments.confidence is None else arguments.confidence
        )
        characterization = build_characterization(
            kind=arguments.kind,
            method=REPLICATE_METHOD,
            replicates=len(labels),
            coverage=coverage,
            confidence=confidence,
            source=arguments.source,
            floors=replicate_floors(
                replicates,
                coverage=coverage,
                confidence=confidence,
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
    print(_confidence_note(characterization))
    width = metric_column_width(
        [entry.metric for entry in characterization.floors]
        + [metric for metric, _ in skipped]
    )
    for entry in characterization.floors:
        print(
            f"  {entry.metric:<{width}} floor {entry.floor:.6g}  "
            f"dispersion {entry.dispersion:.6g} "
            f"bounded at {entry.dispersion_bound:.6g}"
        )
    for metric, reason in skipped:
        print(f"  {metric:<{width}} skipped: {reason}")
    print(f"Recorded: {path}")
    return 0


def _confidence_note(characterization: NoiseCharacterization) -> str:
    """Say what the recorded floors claim, and how much ignorance widened them.

    Printed because the gap between the measured dispersion and the bound the
    floor was built from is the actionable part of a characterization: a wide
    gap says the floor is wide for lack of replicates, which more of them fix,
    and a narrow one says the machine really is that noisy.
    """

    freedoms = sorted({entry.degrees_of_freedom for entry in characterization.floors})
    span = f"{freedoms[0]}" if len(freedoms) == 1 else f"{freedoms[0]}-{freedoms[-1]}"
    inflation = sorted(
        entry.dispersion_bound / entry.dispersion
        for entry in characterization.floors
        if entry.dispersion > 0.0
    )
    widened = ""
    if inflation:
        low, high = inflation[0], inflation[-1]
        factor = f"{low:.2f}x" if low == high else f"{low:.2f}x-{high:.2f}x"
        widened = f", which widens the measured dispersion by {factor}"
    return (
        f"Floors cover {characterization.coverage:.6g} sigma of a same-weights "
        f"delta, with {characterization.confidence:.0%} confidence in the "
        f"dispersion from {span} degree(s) of freedom{widened}."
    )


def _run_eval_noise_characterize_execution(arguments: argparse.Namespace) -> int:
    """Measure this machine's own timing noise and record the floor.

    Unlike every other kind, the replicates do not exist yet: the noise is the
    machine's, so it is observed by measuring again rather than by reading
    numbers the store already holds.
    """

    from anthro_chess.evaluation.execution_noise import (
        DEFAULT_PROCESSES,
        DEFAULT_REPEATS,
        ExecutionNoiseError,
        characterize_execution_noise,
        subprocess_sampler,
    )
    from anthro_chess.evaluation.results import (
        DEFAULT_CONFIDENCE,
        DEFAULT_COVERAGE,
        NoiseCharacterizationError,
        ResultsStore,
        ResultsStoreError,
        metric_column_width,
        resolve_store_root,
    )

    if arguments.config is None:
        print(
            "anthro eval noise characterize: --kind execution measures a "
            "workload rather than reading one, so it needs --config",
            file=sys.stderr,
        )
        return 2
    if arguments.checkpoint:
        print(
            "anthro eval noise characterize: an execution floor describes a "
            "machine rather than a set of checkpoints, so it takes no "
            "--checkpoint; the configuration names the model it measures with",
            file=sys.stderr,
        )
        return 2
    if arguments.metric:
        print(
            "anthro eval noise characterize: an execution floor covers every "
            "metric the measured benchmark reports, so it takes no --metric",
            file=sys.stderr,
        )
        return 2

    processes = (
        DEFAULT_PROCESSES if arguments.processes is None else arguments.processes
    )
    repeats = DEFAULT_REPEATS if arguments.repeats is None else arguments.repeats
    coverage = DEFAULT_COVERAGE if arguments.coverage is None else arguments.coverage
    confidence = (
        DEFAULT_CONFIDENCE if arguments.confidence is None else arguments.confidence
    )
    try:
        store = ResultsStore(resolve_store_root(arguments.store))
        characterization = characterize_execution_noise(
            subprocess_sampler(
                config_path=arguments.config,
                overrides=arguments.set,
                repeats=repeats,
            ),
            processes=processes,
            source=arguments.source,
            coverage=coverage,
            confidence=confidence,
        )
        path = store.append_characterization(characterization)
    except (
        ExecutionNoiseError,
        NoiseCharacterizationError,
        ResultsStoreError,
    ) as error:
        print(f"anthro eval noise characterize: {error}", file=sys.stderr)
        return 2

    execution = characterization.execution
    assert execution is not None  # the record refuses an execution floor without one
    print(
        f"Characterized execution noise over {characterization.replicates} "
        f"reading(s) in {processes} process(es) for "
        f"{len(characterization.floors)} metric(s)."
    )
    print(f"Valid on: {execution.environment_label()}")
    print(_confidence_note(characterization))
    width = metric_column_width(entry.metric for entry in characterization.floors)
    for entry in characterization.floors:
        within = (
            ""
            if entry.within_process_dispersion is None
            else f"  in-process {entry.within_process_dispersion:.6g}"
        )
        print(
            f"  {entry.metric:<{width}} floor {entry.floor:.6g}  "
            f"dispersion {entry.dispersion:.6g} "
            f"bounded at {entry.dispersion_bound:.6g}{within}"
        )
    print(f"Recorded: {path}")
    return 0


def _run_eval_noise_sample(arguments: argparse.Namespace) -> int:
    """Take one process's readings, so a caller can measure across processes."""

    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.evaluation import InferenceBenchmarkConfig
    from anthro_chess.evaluation.execution_noise import (
        DEFAULT_REPEATS,
        ExecutionNoiseError,
        sample_execution_noise,
    )
    from anthro_chess.evaluation.results import metric_column_width

    repeats = DEFAULT_REPEATS if arguments.repeats is None else arguments.repeats
    try:
        resolved = load_config(
            InferenceBenchmarkConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        sample = sample_execution_noise(
            resolved,
            repeats=repeats,
            run_root=_run_root(),
        )
    except (ConfigError, ExecutionNoiseError) as error:
        print(f"anthro eval noise sample: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(json.dumps(sample.as_record(), indent=2, sort_keys=True))
        return 0
    print(
        f"Checkpoint: {sample.checkpoint.label} on "
        f"{sample.execution.environment_label()}"
    )
    print(f"{len(sample.readings)} reading(s) in this process, recorded nowhere:")
    width = metric_column_width(sample.readings[0])
    for metric in sorted(sample.readings[0]):
        values = "  ".join(f"{reading[metric]:.6g}" for reading in sample.readings)
        print(f"  {metric:<{width}} {values}")
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
        metric_column_width,
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
    width = metric_column_width(
        entry.metric for record in characterizations for entry in record.floors
    )
    for record in characterizations:
        processes = (
            "" if record.processes is None else f" in {record.processes} process(es)"
        )
        print(
            f"{record.recorded_at.date().isoformat()}  {record.kind}  "
            f"{record.method}  {record.replicates} replicate(s){processes}  "
            f"coverage {record.coverage:.6g}  "
            f"confidence {record.confidence:.0%}  "
            f"{record.source}"
        )
        if record.execution is not None:
            # An execution floor is only valid where it was measured, so where
            # that was belongs beside it rather than in the record alone.
            print(f"  valid on {record.execution.environment_label()}")
        for entry in record.floors:
            units = (
                "" if entry.sampling_units is None else f"  n={entry.sampling_units}"
            )
            within = (
                ""
                if entry.within_process_dispersion is None
                else f"  in-process {entry.within_process_dispersion:.6g}"
            )
            print(
                f"  {entry.metric:<{width}} floor {entry.floor:.6g}  "
                f"dispersion {entry.dispersion:.6g} "
                f"bounded at {entry.dispersion_bound:.6g} "
                f"(df {entry.degrees_of_freedom}){units}{within}"
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
    from anthro_chess.evaluation.results import (
        iter_registry,
        metric_column_width,
        registry_record,
    )
    from anthro_chess.evaluation.results.reporting import MAXIMUM_LINE_WIDTH

    if arguments.format == "json":
        print(json.dumps(registry_record(), indent=2, sort_keys=True))
        return 0
    registry = list(iter_registry())
    # One width across every family, because the listing is read top to bottom
    # as a single table rather than as one table per family.
    width = metric_column_width(
        metric.identifier for _, metrics in registry for metric in metrics
    )
    for family, metrics in registry:
        print(f"{family.identifier}  {family.title}")
        if not metrics:
            print("  no metric is registered for this family yet")
        for metric in metrics:
            projection = metric.projection or "no data dependency"
            print(
                f"  {metric.identifier:<{width}} {metric.direction.value:<18} "
                f"v{metric.definition_version}  {metric.cost.value:<14} "
                f"{projection}"
            )
            if metric.no_sampling_floor_reason is not None:
                # Where a report reads "unqualifiable", this is what it points
                # at, so the reason belongs in the listing rather than only in
                # the source.
                for line in textwrap.wrap(
                    f"no sampling floor can exist: {metric.no_sampling_floor_reason}",
                    width=MAXIMUM_LINE_WIDTH,
                    initial_indent="    ",
                    subsequent_indent="      ",
                ):
                    print(line)
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
        # The run root places a named run; without one a fresh clone resolves
        # inside the working directory, which several commands depend on.
        placement = _run_root() or Path("artifacts")
        result = run_training(
            resolved,
            output_directory=(
                arguments.output_directory or placement / resolved.value.run_name
            ),
            store=store,
            detail=detail,
        )
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


def _run_root() -> Path | None:
    """Return the configured run root, or ``None`` when it is unset.

    Optional on purpose: a fresh clone resolves checked-in relative paths
    inside the working directory, and several commands depend on that. The
    commands that cannot proceed without a root fail where they need it, and
    name it when they do.
    """

    return optional_root(RUN_ROOT_VARIABLE)


def _resolve_pool_roots(
    resolved: ResolvedConfig[PoolConfig],
    overrides: Sequence[str],
) -> ResolvedConfig[PoolConfig]:
    """Resolve checked-in relative source paths beneath the shared data root."""

    from anthro_chess.evaluation.roots import (
        POOL_ARTIFACT_FIELDS,
        resolve_artifact_roots,
    )

    return resolve_artifact_roots(
        resolved,
        fields=POOL_ARTIFACT_FIELDS,
        overrides=overrides,
    )


def _data_output_path(output: Path | None, artifact_name: str) -> Path:
    if output is not None:
        return output
    return _environment_root(DATA_ROOT_VARIABLE) / artifact_name


def _environment_root(name: str) -> Path:
    return required_root(name, alternative="a directory must be provided explicitly")


def _resolve_training_roots(
    resolved: ResolvedConfig[TrainingConfig],
    overrides: Sequence[str],
) -> ResolvedConfig[TrainingConfig]:
    """Resolve checked-in artifact paths beneath configured machine roots."""
    config = resolved.value
    update: dict[str, object] = {}
    override_keys = {item.partition("=")[0] for item in overrides}

    data_root = optional_root(DATA_ROOT_VARIABLE)

    selections: tuple[tuple[str, SequenceDataConfig | None], ...] = (
        ("train", config.train),
        ("validation", config.validation),
    )
    for selection_name, selection in selections:
        if selection is None or data_root is None:
            continue
        selection_update: dict[str, Path] = {}
        for field_name in ("normalized", "manifest"):
            path = getattr(selection, field_name)
            dotted_key = f"{selection_name}.{field_name}"
            if not path.is_absolute() and dotted_key not in override_keys:
                selection_update[field_name] = _rooted_artifact_path(data_root, path)
        if selection_update:
            update[selection_name] = selection.model_copy(update=selection_update)

    if not update:
        return resolved
    return ResolvedConfig(
        value=config.model_copy(update=update),
        provenance=resolved.provenance,
    )


def _rooted_artifact_path(root: Path, configured_path: Path) -> Path:
    from anthro_chess.evaluation.roots import rooted_artifact_path

    return rooted_artifact_path(root, configured_path)


#: How a benchmark's reading is printed, keyed the way the registry is keyed.
#: A sweep and the benchmark's own subcommand both read it, so one text view
#: serves both rather than a second reporting surface drifting from the first.
_STEP_RENDERERS: Mapping[str, Callable[[Any], str]] = {
    "inference": _render_inference,
    "run": _render_evaluation,
    "novelty": _render_novelty,
    "puzzles": _render_puzzles,
    "rollout": _render_rollout,
    "decisions": _render_decisions,
    "termination": _render_termination,
    "ladder": _render_ladder,
}


if __name__ == "__main__":  # pragma: no cover - console scripts call main directly
    raise SystemExit(main())
