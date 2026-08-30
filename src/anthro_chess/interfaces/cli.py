"""Command-line interface for Anthro Chess."""

from __future__ import annotations

import argparse
import json
import logging
import os
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
from anthro_chess.data.speed import Speed
from anthro_chess.machine import (
    DATA_ROOT_VARIABLE,
    RUN_ROOT_VARIABLE,
    WORKING_ARTIFACTS_DIRECTORY,
    MachineReport,
    RetainedRun,
    inspect_machine,
    optional_root,
    required_root,
)

if TYPE_CHECKING:
    from anthro_chess.data import ArchiveConfig, PrepareConfig, SequenceDataConfig
    from anthro_chess.data.census import PinnedArchive
    from anthro_chess.evaluation import (
        CheckpointEvaluationResult,
        DecisionDecomposition,
        DependencyBenchmarkResult,
        InferenceBenchmarkResult,
        InferenceDeviceReading,
        LadderBenchmarkResult,
        NoveltyBenchmarkResult,
        PoolConfig,
        PoolResult,
        PuzzleBenchmarkResult,
        PuzzleSpread,
        RolloutBenchmarkResult,
        TerminationBenchmarkResult,
    )
    from anthro_chess.evaluation.dependency import (
        CrossConditioningCell,
        DependencyTestResult,
        WithinGameGroup,
    )
    from anthro_chess.evaluation.results import (
        DetailStore,
        ResultsStore,
    )
    from anthro_chess.evaluation.rollout import (
        RolloutCell,
        RolloutReading,
        TerminationGuardrails,
    )
    from anthro_chess.evaluation.suite import StepOutcome, SuitePlan, SuiteRun
    from anthro_chess.evaluation.views import ViewSelection
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
        "Results store directory. Defaults to ANTHRO_CHESS_RESULTS_ROOT, or a "
        "machine-local directory beneath ANTHRO_CHESS_RUN_ROOT. Name the "
        "committed store to read or write it."
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

#: Counting an archive's accounts decompresses tens of gigabytes and is pure
#: work per archive, so several run at once. Kept well below the core count
#: because this machine's cores belong to training runs.
_DEFAULT_CENSUS_WORKERS = 8

#: Decoders one reader can keep fed. Preparation's reader frames every game in
#: its own process and so cannot exceed one core, which makes the pool stop
#: paying once it consumes games as fast as that core frames them. Beyond this
#: a decoder adds the cost of dispatching to it and nothing else, so the
#: default is a property of the pipeline rather than of the machine.
#: `docs/decisions/0053-the-pool-is-sized-to-the-reader-it-waits-on.md`.
_MAXIMUM_PREPARE_WORKERS = 12

#: Shards a freeze scans at once. Scanning divides perfectly, but one parent
#: unpickles every shard's result and merges it into a single list and a single
#: set, and that half does not divide at all — so the same reasoning 0053 gives
#: for preparation applies here, against a different consumer. Measured over
#: disjoint cold shards of the widened corpus and projected to all 41,763:
#: 139 min serially, 24.0 at eight, 19.7 at sixteen, 21.7 at twenty, 26.4 at
#: thirty-two.
_MAXIMUM_FREEZE_WORKERS = 16

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


def _worker_count(value: str) -> int:
    """Read a pool size, refusing a negative that would read as serial."""

    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError(f"{value!r} is not a worker count")
    return count


def _available_cores() -> int:
    # What this process may run on rather than what the machine holds: under a
    # cpuset or a taskset the two disagree, and the machine's count would fork
    # a decoder per core onto a handful of them.
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)


def _prepare_concurrency(requested: int | None) -> int:
    """Return how many archives to decode at once: the fewest that fill the machine.

    One archive cannot fill it, because the reader framing its games holds a
    single core and caps the pool that waits on it. Several can, and throughput
    peaks where their processes come to the machine's own count. The fewest is
    preferred among the arrangements that reach it, since each archive in
    flight is one more that has to be on disk and one more marked-account
    snapshot held.
    """

    if requested is not None:
        return requested
    cores = _available_cores()
    best, most = 1, 0
    for count in range(1, cores + 1):
        processes = count * (_prepare_workers(None, count) + 1)
        if processes <= cores and processes > most:
            best, most = count, processes
    return best


def _freeze_concurrency(requested: int | None) -> int:
    """Return how many shards to scan at once, stopping where the merge does.

    An explicit count is obeyed exactly, past the cap included, for the reason
    0053 gives: a caller running this beside other work is sizing it against
    that rather than against the machine.
    """

    if requested is not None:
        return requested
    return min(_available_cores(), _MAXIMUM_FREEZE_WORKERS)


def _archive_count(value: str) -> int:
    """Read how many archives to prepare at once, which is at least one."""

    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError(f"{value!r} is not an archive count")
    return count


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
    prepare_parser.add_argument(
        "--workers",
        type=_worker_count,
        help=(
            "Processes decoding games per archive, 0 to decode in the "
            "reader's own. Defaults to as many as one reader can keep fed, "
            "divided by the archives sharing the machine. Nothing about it "
            "reaches the artifact."
        ),
    )
    prepare_parser.add_argument(
        "--concurrency",
        type=_archive_count,
        help=(
            "Archives to prepare at once, writing one manifest for all of "
            "them. Defaults to the fewest that fill this machine. Above one it "
            "prepares every acquired archive this selection pins and takes no "
            "input path."
        ),
    )
    prepare_parser.set_defaults(handler=_run_data_prepare)

    census_parser = data_commands.add_parser(
        "census",
        help=(
            "Spend one day's allowance asking the source which of the "
            "selection's accounts it has marked, busiest accounts first."
        ),
        parents=[_SET_FLAG],
    )
    census_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML source and archive selection.",
    )
    census_parser.add_argument(
        "--accounts",
        type=int,
        help=(
            "How many accounts to ask about, overriding the day's allowance "
            "derived from the source's rate limiter."
        ),
    )
    census_parser.add_argument(
        "--pause-seconds",
        type=float,
        help=(
            "Seconds between requests, overriding the pace derived from the "
            "source's rate limiter and whether LICHESS_TOKEN is set."
        ),
    )
    census_parser.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_CENSUS_WORKERS,
        help=(
            "How many archives to count at once when an archive has no counts "
            f"yet. Defaults to {_DEFAULT_CENSUS_WORKERS}."
        ),
    )
    census_parser.add_argument(
        "--priority",
        type=Path,
        help=(
            "Accounts to ask about before any other, one name per line. "
            "Coverage is then reported over this set as well as the corpus."
        ),
    )
    census_parser.set_defaults(handler=_run_data_census)

    mark_accounts_parser = data_commands.add_parser(
        "mark-accounts",
        help=(
            "Cut a marked-account snapshot from the census as it stands, so "
            "preparation can reject their games without querying the source."
        ),
        parents=[_SET_FLAG],
    )
    mark_accounts_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML source and archive selection.",
    )
    mark_accounts_parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Snapshot to write. Defaults to the configured "
            "filters.marked_accounts path."
        ),
    )
    mark_accounts_parser.set_defaults(handler=_run_data_mark_accounts)

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
    freeze_parser.add_argument(
        "--workers",
        type=_worker_count,
        help=(
            "How many shards to scan at once, or 0 to scan them in this "
            "process. Defaults to the cores this process may run on, at most "
            f"{_MAXIMUM_FREEZE_WORKERS}; an explicit count is obeyed past that."
        ),
    )
    freeze_parser.set_defaults(handler=_run_eval_freeze)

    prepare_puzzles_parser = eval_commands.add_parser(
        "prepare-puzzles",
        help="Install the committed external puzzle benchmark artifact.",
        parents=[_SET_FLAG],
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

    dependency_parser = eval_commands.add_parser(
        "dependency",
        help="Measure whether a checkpoint reads its rating conditioning.",
        parents=[_SET_FLAG, _STORE_FLAG, _DETAIL_ROOT_FLAG, _FORMAT_FLAG],
    )
    dependency_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit TOML rating-dependency selection.",
    )
    dependency_parser.add_argument(
        "--no-record",
        action="store_true",
        help="Compute and print results without writing them to the store.",
    )
    dependency_parser.set_defaults(
        handler=partial(_run_eval_benchmark, name="dependency")
    )

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
        "--speed",
        choices=tuple(speed.value for speed in Speed),
        help=(
            "Select one speed class, as the benchmark's own reference does. A "
            "bandwidth chosen over a mixture is not the smoothing a sliced "
            "reference is read at."
        ),
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
        parents=[_STORE_FLAG, _FORMAT_FLAG],
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

    promote_parser = eval_commands.add_parser(
        "promote",
        help="Copy one checkpoint's records into the committed results store.",
    )
    promote_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint label whose records are promoted, as a record names it.",
    )
    # Its own rather than the shared flag: here --store is the source end of a
    # copy, and the shared wording points at the committed store, which is the
    # other end.
    promote_parser.add_argument(
        "--store",
        type=Path,
        help="Store the records are copied from, defaulting as every --store does.",
    )
    promote_parser.add_argument(
        "--into",
        type=Path,
        help=(
            "Where the records are copied to, defaulting to the committed "
            "store in the repository. Committing the copy in a pull request is "
            "what makes the promotion, so merging is the acceptance and an "
            "unmerged one costs nothing to unwind."
        ),
    )
    promote_parser.set_defaults(handler=_run_eval_promote)

    tensorboard_parser = eval_commands.add_parser(
        "tensorboard",
        help="Project checkpoint history from the results store into TensorBoard.",
        parents=[_STORE_FLAG],
    )
    tensorboard_parser.add_argument(
        "output",
        type=Path,
        help=(
            "Disposable TensorBoard log directory. Must be outside the results "
            "store it projects."
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
        help="Sample a process's efficiency readings, and size a pool from one.",
    )
    noise_commands = noise_parser.add_subparsers(
        dest="noise_command",
        required=True,
    )
    noise_sample_parser = noise_commands.add_parser(
        "sample",
        help="Measure one process's efficiency reading, recording nothing.",
        parents=[_SET_FLAG, _FORMAT_FLAG],
    )
    noise_selection = noise_sample_parser.add_mutually_exclusive_group(required=True)
    noise_selection.add_argument(
        "--config",
        type=Path,
        help="Explicit TOML inference-benchmark selection to repeat.",
    )
    noise_selection.add_argument(
        "--selection",
        help=(
            "An already-resolved selection as JSON, or '-' to read it from "
            "standard input. This is how the inference benchmark hands its own "
            "resolved selection to a replicate process, so that the replicate "
            "measures what the parent measured rather than re-resolving it."
        ),
    )
    noise_sample_parser.set_defaults(handler=_run_eval_noise_sample)

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
        "--verify-data",
        action="store_true",
        help=(
            "Hash every normalized shard against the data manifest before "
            "training. The default check compares each shard's recorded row "
            "count against its Parquet footer, which catches a missing, "
            "truncated or replaced shard; this also catches a page rewritten "
            "in place, and costs a full read of the corpus."
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


#: Minute resolution in UTC. The column exists to order runs and to expose a
#: recent stamp on stale contents, and neither needs seconds or a local zone.
_STAMP_FORMAT = "%Y-%m-%dT%H:%MZ"


def _load_verdict(run: RetainedRun) -> str:
    """State whether a run can be loaded from, and why not when it cannot.

    Silent when the report could not determine it, because the one reason
    covers every run and is already reported beside the roots.
    """

    if run.loadable is None:
        return ""
    if run.loadable:
        return "  loadable"
    return f"  not loadable: {run.blocker}"


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
    lines.append("retained runs, newest first")
    if not report.runs:
        lines.append(f"  none beneath {_root_note(report, RUN_ROOT_VARIABLE)}")
    stamps = [
        ""
        if run.latest_modified is None
        else run.latest_modified.strftime(_STAMP_FORMAT)
        for run in report.runs
    ]
    run_width = max((len(run.name) for run in report.runs), default=0)
    stamp_width = max((len(stamp) for stamp in stamps), default=0)
    for run, stamp in zip(report.runs, stamps, strict=True):
        latest = run.latest_checkpoint or "no latest pointer"
        lines.append(
            f"  {run.name:<{run_width}}  {stamp:<{stamp_width}}  "
            f"{run.checkpoints} checkpoint(s), {latest}{_load_verdict(run)}"
        )

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
        acquire_configured_archive,
    )

    try:
        resolved = load_config(
            PrepareConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        archives = resolved.value.archives
        if not archives:
            raise ConfigError(
                "configuration has no archive selection for data acquisition"
            )
        reused = 0
        for archive in archives:
            # An archive naming its own artifact lets two selections that pin
            # the same month share one download.
            output = _data_output_path(
                arguments.output,
                archive.artifact_name or resolved.value.artifact_name,
            )
            result = acquire_configured_archive(output, archive)
            disposition = "Reused" if result.reused else "Acquired"
            print(f"{disposition} verified archive: {result.archive_path}")
            print(f"SHA-256: {result.sha256}")
            reused += int(result.reused)
    except (ConfigError, DataPreparationError) as error:
        print(f"anthro data acquire: {error}", file=sys.stderr)
        return 2

    print(f"{len(archives)} archive(s) verified, {reused} already present.")
    return 0


def _prepare_workers(requested: int | None, concurrency: int = 1) -> int:
    """Size the decoding pool one archive gets, of however many share the machine.

    Two bounds, whichever is smaller. A reader cannot feed more than
    ``_MAXIMUM_PREPARE_WORKERS`` whatever else is running, and archives prepared
    at once divide the machine between them rather than each taking all of it.
    """

    if requested is not None:
        return requested
    share = max(_available_cores() // concurrency - 1, 0)
    return min(share, _MAXIMUM_PREPARE_WORKERS)


def _run_data_prepare(arguments: argparse.Namespace) -> int:
    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.data import (
        DataPreparationError,
        PrepareConfig,
        prepare_archives,
        prepare_pgn,
    )

    try:
        resolved = load_config(
            PrepareConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        output = _data_output_path(arguments.output, resolved.value.artifact_name)
        # A selection pinning many archives has no single default input, so
        # naming none of them asks for all of them. Naming one still prepares
        # exactly that one, whatever the machine could have run beside it.
        if arguments.input is None and len(resolved.value.archives) > 1:
            concurrency = _prepare_concurrency(arguments.concurrency)
            acquired = _acquired_archive_inputs(resolved, arguments.output)
            if not acquired:
                raise ConfigError(
                    "no archive this selection pins has been acquired yet"
                )
            result = prepare_archives(
                [path for path, _ in acquired],
                output,
                resolved,
                workers=_prepare_workers(arguments.workers, concurrency),
                concurrency=concurrency,
                counts_paths=[counts_path for _, counts_path in acquired],
            )
        else:
            if arguments.concurrency is not None and arguments.concurrency > 1:
                raise ConfigError(
                    "--concurrency prepares the archives a selection pins, so "
                    "it cannot be given an input path"
                )
            input_path = _configured_archive_path(
                resolved, arguments.input, arguments.output
            )
            result = prepare_pgn(
                input_path,
                output,
                resolved,
                workers=_prepare_workers(arguments.workers),
                counts_path=_archive_counts_path(
                    resolved, input_path, arguments.output
                ),
            )
    except (ConfigError, DataPreparationError) as error:
        print(f"anthro data prepare: {error}", file=sys.stderr)
        return 2

    if result.disposition == "corpus_complete":
        print("Corpus already holds its configured maximum; nothing prepared.")
    else:
        if result.disposition == "already_prepared":
            print(
                f"Archive already in this corpus, contributing "
                f"{result.accepted_games} game(s)."
            )
        else:
            print(
                f"Prepared {result.accepted_games} game(s); "
                f"rejected {result.rejected_games}."
            )
        if not result.normalized_paths:
            print("Normalized: no shards, because this archive accepted no games.")
        elif len(result.normalized_paths) == 1:
            print(f"Normalized: {result.normalized_path}")
        else:
            print(
                f"Normalized: {len(result.normalized_paths)} shard(s) under "
                f"{result.normalized_paths[0].parent}"
            )
    print(
        f"Corpus: {sum(result.split_counts.values())} game(s) from "
        f"{result.corpus_archives} archive(s)."
    )
    print(f"Manifest: {result.manifest_path}")
    return 0


def _run_data_census(arguments: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.data import DataLoadingError, PrepareConfig
    from anthro_chess.data.census import (
        LICHESS_USERS_BATCH,
        CensusError,
        counted_archives,
        daily_account_allowance,
        read_census,
        read_prioritized_accounts,
        refresh_archive_counts,
        run_census,
        source_token,
    )

    try:
        resolved = load_config(
            PrepareConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        if arguments.accounts is not None and arguments.accounts < 1:
            raise ConfigError("--accounts asks about at least one account")
        pinned = _pinned_archives(resolved)
        # An archive's counts outlive the archive, so reclaiming a raw file once
        # it is prepared costs the census nothing. Selecting on the raw file
        # instead drops its accounts out of the queue without a word, and the
        # snapshot would claim archives the census had stopped asking about.
        counted = counted_archives(pinned)
        countable = [archive for archive in pinned if archive.path.is_file()]
        if not counted and not countable:
            raise ConfigError(
                "none of this selection's archives are counted or on disk to "
                "count; acquire one first"
            )
        if not counted:
            # Asking comes before counting below, which it cannot do on the one
            # run that has nothing counted to ask about.
            refresh_archive_counts(countable, workers=arguments.workers)
            counted = counted_archives(pinned)
        for archive in pinned:
            if archive not in counted:
                logger.warning(
                    "Censusing without %s, which is neither counted nor on disk; "
                    "its accounts are absent from the queue and from the coverage "
                    "below until it is acquired",
                    archive.path.name,
                )
        answers_path = _census_answers_path(resolved)
        prioritized = (
            read_prioritized_accounts(arguments.priority)
            if arguments.priority is not None
            else ()
        )
        census = read_census(counted, answers_path, prioritized)
        if arguments.priority is not None and not census.prioritized_total:
            raise ConfigError(
                f"{arguments.priority} names no account these archives hold, so "
                "prioritizing it would silently ask about the corpus at large"
            )
        budget = arguments.accounts
        if budget is None:
            budget = daily_account_allowance(
                LICHESS_USERS_BATCH, authenticated=source_token() is not None
            )
        queue = census.queue(budget)
        run = run_census(
            queue,
            answers_path,
            queried_at=datetime.now(tz=UTC).date().isoformat(),
            pause_seconds=arguments.pause_seconds,
        )
    except (ConfigError, DataLoadingError, CensusError) as error:
        print(f"anthro data census: {error}", file=sys.stderr)
        return 2

    accounts_queried = census.accounts_queried + run.accounts_asked
    slots_queried = census.slots_queried + sum(
        census.games_by_account[name] for name in run.asked
    )
    unanswered = census.accounts_total - accounts_queried
    if run.refused:
        outcome = "The source's allowance is spent."
    elif not census.accounts_total:
        outcome = "No archive is counted yet, so this run counts rather than asks."
    elif not unanswered:
        outcome = "Every account in these archives has been asked about."
    elif arguments.accounts is None:
        outcome = "The day's allowance is spent."
    else:
        outcome = "The requested accounts are asked about."
    print(
        f"Asked about {run.accounts_asked} account(s); "
        f"{run.accounts_marked} marked. {outcome}"
    )
    print(
        f"Coverage: {accounts_queried} of {census.accounts_total} account(s) "
        f"({_share(accounts_queried, census.accounts_total)}), "
        f"{_share(slots_queried, census.slots_total)} of player-slots, over "
        f"{len(census.archives)} of {len(pinned)} pinned archive(s)."
    )
    if census.prioritized_total:
        asked = census.prioritized & set(run.asked)
        prioritized_queried = census.prioritized_queried + len(asked)
        prioritized_slots = census.prioritized_slots_queried + sum(
            census.games_by_account[name] for name in asked
        )
        print(
            f"Prioritized: {prioritized_queried} of {census.prioritized_total} "
            f"account(s) "
            f"({_share(prioritized_queried, census.prioritized_total)}), "
            f"{_share(prioritized_slots, census.prioritized_slots_total)} of "
            "their player-slots."
        )

    # Counting last. The day's allowance does not roll over and a count of a
    # newly acquired archive is hours, so a run that counted first would spend
    # the scarce thing only if it survived the cheap one.
    try:
        refresh_archive_counts(countable, workers=arguments.workers)
    except (DataLoadingError, CensusError) as error:
        print(f"anthro data census: {error}", file=sys.stderr)
        return 2
    return 0


def _run_data_mark_accounts(arguments: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from anthro_chess.config import ConfigError, load_config
    from anthro_chess.data import DataLoadingError, PrepareConfig
    from anthro_chess.data.accounts import MarkedAccountError, resolve_snapshot_path
    from anthro_chess.data.census import CensusError, read_census, snapshot_from_census

    try:
        resolved = load_config(
            PrepareConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        output_path = arguments.output
        if output_path is None:
            configured_output = resolved.value.filters.marked_accounts
            if configured_output is None:
                raise ConfigError(
                    "--output is required because the selection sets no "
                    "filters.marked_accounts path"
                )
            output_path = resolve_snapshot_path(
                configured_output, resolved.provenance.source
            )
        if output_path.exists():
            raise ConfigError(
                f"{output_path} already exists. A snapshot cut from a later census "
                "answers for more accounts, so it rejects games this one keeps and "
                "starts a new corpus rather than amending it; write it to its own "
                "--output path"
            )

        pinned = _pinned_archives(resolved)
        uncounted = [archive for archive in pinned if not archive.counts_path.is_file()]
        if uncounted:
            raise ConfigError(
                f"the census has counted {len(pinned) - len(uncounted)} of this "
                f"selection's {len(pinned)} archive(s). Preparation appends one "
                "archive at a time and refuses any the snapshot does not cover, so "
                "a snapshot cut now stops a corpus partway through and it cannot be "
                "repaired incrementally; acquire and census the rest first"
            )
        census = read_census(pinned, _census_answers_path(resolved))
        if not census.answers:
            raise ConfigError(
                "the census has answered for none of these archives' accounts, so "
                "a snapshot would claim nobody is marked; run `anthro data census "
                "--config <selection>` first"
            )
        snapshot = snapshot_from_census(
            census, queried_at=datetime.now(tz=UTC).date().isoformat()
        )
        snapshot.write(output_path)
    except (ConfigError, DataLoadingError, MarkedAccountError, CensusError) as error:
        print(f"anthro data mark-accounts: {error}", file=sys.stderr)
        return 2

    print(
        f"Marked {snapshot.accounts_marked} of {snapshot.accounts_queried} account(s) "
        f"asked about ({_share(snapshot.accounts_marked, snapshot.accounts_queried)})."
    )
    print(
        f"Coverage: {_share(snapshot.accounts_queried, snapshot.accounts_total)} of "
        f"accounts and {_share(snapshot.slots_queried, snapshot.slots_total)} of "
        f"player-slots, over {len(snapshot.covers_archives)} archive(s)."
    )
    print(f"Snapshot: {output_path}")
    return 0


def _share(part: int, whole: int) -> str:
    return f"{100 * part / whole:.2f}%" if whole else "0.00%"


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
        result = freeze_pool(
            resolved, output, workers=_freeze_concurrency(arguments.workers)
        )
    except (ConfigError, EvaluationPoolError) as error:
        print(f"anthro eval freeze: {error}", file=sys.stderr)
        return 2

    print(f"Froze {result.games} game(s) and {result.plies} ply/plies.")
    print(f"Pool: {result.games_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Identity: {result.game_ids_sha256}")
    _print_coverage(result)
    return 0


def _print_coverage(result: PoolResult) -> None:
    """Print the per-axis composition of the pool just cut.

    Every one of these numbers is in the manifest already. They are printed
    because what a reading can resolve on an axis follows from how many games
    the pool holds on it, and a number nobody read when the pool was cut is one
    nobody acts on afterwards.
    """

    print("\nPer-axis coverage:")
    for axis, counts in sorted(result.coverage_axes.items()):
        readings = ", ".join(
            f"{name} {count}" for name, count in sorted(counts.items())
        )
        print(f"  {axis}: {readings}")


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
        result = prepare_puzzle_set(resolved, output)
    except (ConfigError, PuzzleSetError) as error:
        print(f"anthro eval prepare-puzzles: {error}", file=sys.stderr)
        return 2

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
        resolve_suite,
        run_suite,
        sweep_directory,
    )

    try:
        plan = resolve_suite(
            load_config(SuiteConfig, path=arguments.config, overrides=arguments.set),
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
        f"Suite: {plan.suite}",
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
        f"Suite: {plan.suite}, {len(run.outcomes)} step(s) in {run.seconds:.1f}s",
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
    """Return the summary and detail stores a benchmark records through.

    A summary record references its detail payloads by path and digest, so a
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
            f"Leakage: none of {result.leakage.pool_games} pool game(s) fall "
            f"in the {result.leakage.training_split} split "
            f"[{result.leakage.algorithm}]"
            if result.leakage.verified
            else (
                f"Leakage: NOT VERIFIED - {result.leakage.unverified_reason} "
                f"[{result.leakage.algorithm}]"
            )
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
    if result.dispersions:
        lines.extend(
            [
                "",
                (
                    f"Noise: data-sampling dispersions for "
                    f"{len(result.dispersions)} metric(s), bootstrapped over "
                    f"{result.view.selected_games} game(s). A delta against "
                    "another reading is floored by combining the two."
                ),
            ]
        )
    lines.extend(_recorded_lines(result.recorded_paths))
    return "\n".join(lines) + "\n"


def _band_conditioning_ratings(
    by_band: Mapping[str, Mapping[int, CrossConditioningCell]],
    ratings: Sequence[int],
) -> dict[str, int]:
    """Return the grid rating that falls inside each band, where one does."""

    from anthro_chess.evaluation.slices import rating_band_name

    own: dict[str, int] = {}
    for rating in ratings:
        band = rating_band_name(rating)
        if band is not None and band in by_band:
            own.setdefault(band, rating)
    return own


def _render_cross_conditioning(dependency: DependencyTestResult) -> list[str]:
    """Show the band-by-conditioning table beside the scalars that summarize it.

    The match rate is a fraction of four, and a checkpoint that has learned the
    ordering at all posts a one; the table it stands for keeps moving long
    after. Reading the diagonal against its own row is what says whether a
    conditioning value means anything, and the row ends are what say how much.
    """

    cross = dependency.cross_conditioning
    if not cross.cells:
        return []
    ratings = sorted({cell.conditioning_rating for cell in cross.cells})
    by_band: dict[str, dict[int, CrossConditioningCell]] = {}
    for cell in cross.cells:
        by_band.setdefault(cell.rating_band, {})[cell.conditioning_rating] = cell

    # The star is the band's own rating rather than the row's smallest, so a
    # band that fails to prefer its own shows a star off the minimum. Marking
    # the minimum would render every band as though it had matched.
    own = _band_conditioning_ratings(by_band, ratings)
    header = "".join(f"{rating:>10}" for rating in ratings)
    lines = [
        "Cross-conditioning move loss (* marks each band's own rating):",
        f"  {'band':<16}{header}{'positions':>11}",
    ]
    for band, cells in by_band.items():
        marked = own.get(band)
        columns: list[str] = []
        for rating in ratings:
            scored = cells.get(rating)
            if scored is None:
                columns.append(f"{'-':>10}")
                continue
            star = "*" if rating == marked else ""
            columns.append(f"{star + format(scored.move_loss, '.4f'):>10}")
        row = "".join(columns)
        positions = max(cell.position_count for cell in cells.values())
        lines.append(f"  {band:<16}{row}{positions:>11}")

    pinned = dict(cross.pinned_degradations)
    lines.append(
        f"  {'pinned there':<16}"
        + "".join(
            f"{pinned[rating]:>+10.4f}" if rating in pinned else f"{'-':>10}"
            for rating in ratings
        )
    )
    lines.append(
        f"  match rate {_optional(cross.match_rate)} over "
        f"{len(cross.compared_bands)} band(s), "
        f"away-band penalty {_optional(cross.penalty)} "
        f"over {cross.penalty_positions} position(s)"
    )
    if cross.excluded_bands:
        lines.append(f"  too thin to compare: {', '.join(cross.excluded_bands)}")
    return lines


def _render_within_game(dependency: DependencyTestResult) -> list[str]:
    """Show each band's own prefix split beside the one number pooled from them.

    The response has no sampling floor, so whether the bands agree is the only
    thing standing in for one. Four bands moving together is a different reading
    from four cancelling out, and the pooled number cannot tell them apart.
    """

    from anthro_chess.evaluation.dependency import (
        STRONGER_PREFIX_GROUP,
        WEAKER_PREFIX_GROUP,
    )

    within = dependency.within_game
    if not within.groups:
        return ["Within-game response: no band held enough held-out prefixes"]
    halves: dict[str, dict[str, WithinGameGroup]] = {}
    for group in within.groups:
        halves.setdefault(group.rating_band, {})[group.group] = group

    lines = [
        (
            f"Within-game response at a fixed stated rating "
            f"(prefix judged at {within.anchor_low_rating} against "
            f"{within.anchor_high_rating}):"
        ),
        (
            f"  {'band':<16}{'n-':>8}{'n+':>8}{'prefix-':>10}{'prefix+':>10}"
            f"{'align-':>10}{'align+':>10}{'shift':>9}"
        ),
    ]
    for band, groups in halves.items():
        weak = groups.get(WEAKER_PREFIX_GROUP)
        strong = groups.get(STRONGER_PREFIX_GROUP)
        if weak is None or strong is None:
            continue
        lines.append(
            f"  {band:<16}{weak.position_count:>8}{strong.position_count:>8}"
            f"{weak.mean_prefix_strength:>+10.3f}{strong.mean_prefix_strength:>+10.3f}"
            f"{weak.mean_alignment:>+10.3f}{strong.mean_alignment:>+10.3f}"
            f"{strong.mean_alignment - weak.mean_alignment:>+9.3f}"
        )
    lines.append(f"  pooled response {_optional(within.response)}")
    return lines


def _render_dependency(result: DependencyBenchmarkResult) -> str:
    dependency = result.dependency
    lines = [
        f"Checkpoint: {result.checkpoint.label} (step {result.checkpoint.step})",
        (
            f"Pool: {result.dataset.pool_id} v{result.dataset.pool_version} "
            f"view {result.dataset.view} "
            f"({result.view.selected_games} game(s))"
        ),
        (
            f"Leakage: none of {result.leakage.pool_games} pool game(s) fall "
            f"in the {result.leakage.training_split} split "
            f"[{result.leakage.algorithm}]"
            if result.leakage.verified
            else (
                f"Leakage: NOT VERIFIED - {result.leakage.unverified_reason} "
                f"[{result.leakage.algorithm}]"
            )
        ),
        "",
        (
            "Rating dependency (a degradation to interpret against training "
            f"maturity, at step {dependency.maturity.step}):"
        ),
        *(
            f"  {item.conditioning.name:<10} degradation={item.degradation:+.6f}"
            for item in dependency.corruptions
        ),
        f"  anchor policy divergence:      {dependency.anchor_divergence:.6f}",
        f"  anchor top-1 agreement:        {dependency.anchor_agreement_rate:.6f}",
        "",
        *_render_cross_conditioning(dependency),
        "",
        *_render_within_game(dependency),
    ]
    if result.dispersions:
        lines.extend(
            [
                "",
                (
                    f"Noise: data-sampling dispersions for "
                    f"{len(result.dispersions)} metric(s), bootstrapped over "
                    f"{result.view.selected_games} game(s). A delta against "
                    "another reading is floored by combining the two."
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
    from anthro_chess.evaluation.puzzles.dataset import (
        PUZZLE_DETECTION_CONFIDENCE,
        PUZZLE_DETECTION_POWER,
    )

    selection = result.selection
    scope = (
        f"{selection.selected_puzzles} of {selection.eligible_puzzles} "
        f"puzzle(s), {selection.puzzles_per_rating} per rating"
        if selection.subsampled
        else f"{selection.selected_puzzles} puzzle(s)"
    )
    lines = [
        f"Checkpoint: {result.checkpoint.label} (step {result.checkpoint.step})",
        (
            f"Puzzle set: {result.dataset.pool_id} v{result.dataset.pool_version} "
            f"({scope})"
        ),
        (
            f"Resolution: {selection.minimum_detectable_difference * 100:.2f} pp "
            f"for independent readings at {PUZZLE_DETECTION_CONFIDENCE:.0%} "
            f"confidence, {PUZZLE_DETECTION_POWER:.0%} power"
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
            f"fit={rating.sampled_fitted_puzzle_rating:7.1f}"
        )
    resolution = result.resolution

    if resolution is None:
        spreads = ("", "", "", "")
    else:
        spreads = (
            f" ({_spread(resolution.greedy_rating_slope, 4)})",
            f" ({_spread(resolution.greedy_order_accuracy, 3)})",
            f" ({_spread(resolution.sampled_rating_slope, 4)})",
            f" ({_spread(resolution.sampled_order_accuracy, 3)})",
        )
    lines.extend(
        [
            "",
            (
                f"Greedy slope={result.greedy_rating_slope:.4f}{spreads[0]}, "
                f"order={result.greedy_order_accuracy:.3f}{spreads[1]}"
            ),
            (
                f"Sampled slope={result.sampled_rating_slope:.4f}{spreads[2]}, "
                f"order={result.sampled_order_accuracy:.3f}{spreads[3]}"
            ),
        ]
    )
    if resolution is None:
        lines.append(
            "Response resolution: not estimated; this run switched noise "
            "estimation off, or the scored puzzles are too thin to redraw"
        )
    else:
        # The widest of the configured grid rather than all of them, because
        # the line has to hold at any grid size and the detail payload keeps
        # each one. They differ by a few percent on a real reading.
        lines.extend(
            [
                (
                    "Solve-rate spread: greedy "
                    f"first={_spread(resolution.greedy_first_move_accuracy, 4)} "
                    f"line={_spread(resolution.greedy_line_completion, 4)}"
                ),
                (
                    "                   sampled "
                    f"first={_spread(resolution.sampled_first_move_solve_rate, 4)} "
                    f"line={_spread(resolution.sampled_line_completion, 4)}"
                ),
                (
                    f"Response resolution: {resolution.resamples} stratified "
                    f"refits of {resolution.puzzles} redrawn puzzle(s), "
                    f"{resolution.coverage:.3g}x coverage at "
                    f"{resolution.confidence:.0%} confidence"
                ),
                (
                    "  fitted puzzle rating: "
                    f"{_spread(resolution.widest_greedy_fit, 1)} greedy, "
                    f"{_spread(resolution.widest_sampled_fit, 1)} sampled, "
                    "widest over the configured grid"
                ),
            ]
        )
    lines.extend(_recorded_lines(result.recorded_paths))
    return "\n".join(lines) + "\n"


def _spread(spread: PuzzleSpread | None, digits: int) -> str:
    """Render a resampled spread, naming an unmoved quantity rather than zero."""

    return "spread unknown" if spread is None else f"±{spread.bound:.{digits}f}"


def _render_inference(result: InferenceBenchmarkResult) -> str:
    lines = [
        f"Checkpoint: {result.checkpoint.label} (step {result.checkpoint.step})",
        "",
        "What one decision costs the model, counted rather than timed:",
        f"  parameters      {result.cost.parameters / 1e6:8.3f} M",
        f"  per decision    {result.cost.decision_gflops:8.4f} GFLOP",
    ]
    for reading in result.readings:
        lines.extend(_device_lines(reading))
    lines.extend(_process_lines(result))
    lines.extend(_recorded_lines(result.recorded_paths))
    return "\n".join(lines) + "\n"


def _device_lines(reading: InferenceDeviceReading) -> list[str]:
    execution = reading.execution
    latency = reading.latency
    serving = reading.serving
    threads = (
        "" if execution.cpu_threads is None else f", {execution.cpu_threads} thread(s)"
    )
    lines = [
        "",
        (
            f"On {execution.device} ({execution.device_name}) "
            f"{execution.precision}{threads}, as this process measured it "
            f"(workload {execution.workload_sha256[:12]}):"
        ),
        (
            f"  Batch-one move latency at {latency.history_plies} plies "
            f"({latency.decisions} decision(s), warmup excluded):"
        ),
    ]
    lines.extend(
        f"    p{percentile:<3} {value:8.1f} ms"
        for percentile, value in sorted(latency.percentiles.items())
    )
    lines.extend(
        [
            f"    mean {latency.mean_ms:8.1f} ms "
            f"(min {latency.minimum_ms:.1f}, max {latency.maximum_ms:.1f})",
            "  Where a decision spends its mean latency:",
            f"    context   {latency.context_mean_ms:8.3f} ms "
            "(assemble the encoded trajectory)",
            f"    predict   {latency.predict_mean_ms:8.3f} ms "
            "(batch build, forward, host copy)",
            f"    remainder {latency.remainder_mean_ms:8.3f} ms "
            "(masking, sampling, encoding the new ply)",
            (
                f"  Serving {serving.batch_size} concurrent decisions: "
                f"{serving.decisions_per_second:.1f} decisions/s "
                f"(median batch {serving.batch_median_ms:.1f} ms)"
            ),
            (
                f"    {serving.decision_overhead_ms:.3f} ms of each is host work "
                "around the model, which does not amortize."
            ),
        ]
    )
    if reading.compute is not None:
        compute = reading.compute
        lines.append(
            f"  Forward pass alone at batch {compute.batch_size}: "
            f"{compute.forward_decisions_per_second:.1f} decisions/s "
            f"({compute.forward_median_ms:.1f} ms per batch), where the device "
            "is no longer launch bound."
        )
        if compute.peak_memory_mb is not None:
            lines.append(f"    peak device memory {compute.peak_memory_mb:.1f} MB")
    if reading.cold_start is not None:
        lines.extend(
            [
                "  Cold start, reported apart from steady state:",
                f"    model load     {reading.cold_start.model_load_seconds:8.3f} s",
                f"    first decision "
                f"{reading.cold_start.first_decision_seconds:8.3f} s",
            ]
        )
    return lines


def _process_lines(result: InferenceBenchmarkResult) -> list[str]:
    """Say what the extra processes bought, which is most of the run time.

    Printed because a reader who cannot see the spread has no way to judge
    whether the process count is worth its wall clock.
    """

    from anthro_chess.evaluation.results import metric_column_width

    if not result.pooled:
        return [
            "",
            "Taken in one process: the value is that process's reading and a delta "
            "against it reports unknown noise.",
        ]
    lines = [
        "",
        f"Committed, pooled over {result.processes} processes. Each value is their "
        "mean, beside the spread of that mean, which floors a delta against it:",
    ]
    width = metric_column_width(result.pooled)
    for label, (value, spread) in sorted(result.pooled.items()):
        qualified = (
            f"±{spread:.6g}"
            if spread > 0.0
            else "(no floor; every process read it identically)"
        )
        lines.append(f"  {label:<{width}} {value:.6g}  {qualified}")
    return lines


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

    from anthro_chess.evaluation.curves import CURVE_DETERMINISTIC_METHOD

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
        spreads = comparison.dispersions
        lines.append(
            f"  {quantity.value:<{width}}"
            + _rollout_arm(
                comparison.conditional_distance,
                null=None if references is None else references.conditional,
                floor=None if spreads is None else spreads.conditional_floor,
                seed=None if spread is None else spread.floor,
            )
            + _rollout_arm(
                comparison.pooled_distance,
                null=None if references is None else references.pooled,
                floor=None if spreads is None else spreads.pooled_floor,
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
    if any(
        comparison.dispersions is not None
        and comparison.dispersions.method == CURVE_DETERMINISTIC_METHOD
        for comparison in reading.comparisons.values()
    ):
        # A floor of zero here is the reading's own answer rather than a
        # bootstrap that happened to land small, and the two look identical in
        # the column.
        lines.append(
            "  floors are exactly zero: greedy seats replay these games, so "
            "re-running this reading moves nothing"
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


def _render_guardrails(cell: RolloutCell) -> list[str]:
    """Show what one cell did with the terminal actions it was offered."""

    guardrails = cell.guardrails
    return [
        (
            f"  resignations   {guardrails.resignations} "
            f"({_optional_rate(guardrails.premature_rate)} premature), "
            f"{_silent_actions(guardrails)}"
        ),
        (
            f"  never ended    {guardrails.claimable_unfinished_games} game(s) "
            "reached a claimable position and ran out of plies"
        ),
    ]


def _render_rollout(result: RolloutBenchmarkResult) -> str:
    from anthro_chess.evaluation.games import GameTermination

    lines = [
        f"Checkpoint: {result.checkpoint.label} (step {result.checkpoint.step})",
        f"Games: {result.games} across {len(result.cells)} matrix cell(s)",
    ]
    if result.view is not None:
        record = result.view.as_record()
        lines.append(
            f"Prefix view: {result.view.name} ({_view_population(result.view)}, "
            f"prefix {record['prefix_plies']} plies)"
        )
    if result.reference_view is not None:
        lines.append(
            f"Reference view: {result.reference_view.name} "
            f"({_view_population(result.reference_view)})"
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
                *_render_guardrails(cell),
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
        if reading.human_premature_rate is not None:
            lines.append(
                "  human resignations were "
                f"{reading.human_premature_rate:.1%} premature by the same proxy"
            )
        lines.extend(_render_bandwidth(reading))
        lines.extend(_render_comparison_table(reading, width))
        lines.extend(_render_unavailable(reading))
        lines.extend(_render_category_drilldowns(reading))
        lines.extend(_render_divergence_depth(reading))
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


def _render_category_drilldowns(reading: RolloutReading) -> list[str]:
    """Show each categorical distance beside the categories behind it.

    A distance over categories says how much mass is in the wrong place and
    never which place. Opening families are uneven, so a delta read without the
    category's own mass invites treating a swing on a narrow line as the same
    finding as one on a family half the corpus plays. Results and endings are
    few enough that the table is the reading and the scalar summarizes it.
    """

    from anthro_chess.evaluation.curves import CurveQuantity

    lines: list[str] = []
    for quantity, comparison in reading.comparisons.items():
        if comparison.spec.quantity is not CurveQuantity.CATEGORICAL:
            continue
        shares = comparison.category_shares()[:_DRILLDOWN_CATEGORIES]
        if not shares:
            continue
        lines.append(f"  {quantity.value} by category")
        lines.append(f"    {'category':<40}{'mass':>8}{'model':>8}{'delta':>9}")
        lines.extend(
            f"    {share.category[:40]:<40}{share.mass:>8.3f}"
            f"{share.model:>8.3f}{share.delta:>+9.3f}"
            for share in shares
        )
    return lines


def _render_divergence_depth(reading: RolloutReading) -> list[str]:
    """Show where the opening divergence accumulates, beside how large it is."""

    from anthro_chess.evaluation.rollout import divergence_half_depth

    half = divergence_half_depth(reading.divergence)
    if half is None:
        return []
    deepest = reading.divergence[-1]
    return [
        (
            f"  divergence     half by book ply {half:.2f}, "
            f"{deepest.conditional_distance:.4f} by ply {deepest.ply}"
            f"{_against_null(deepest.conditional_null)}"
        )
    ]


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
            f"    conditional {exact.conditional_distance:.4f}"
            f"{_against_null(exact.conditional_null)}  "
            f"pooled {exact.pooled_distance:.4f}"
            f"{_against_null(exact.pooled_null)}  "
            f"waypoints {exact.waypoint_mass:.3f}"
        ),
        (
            f"    reached ply {exact.deepest_expanded_ply} of {exact.plies}; "
            f"uncommitted mass at most {exact.unsettled_mass:.3f} "
            f"({exact.pruned_mass:.3f} pruned in all)"
        ),
    ]


def _against_null(null: float | None) -> str:
    """Return the level a matching policy would read at, where one was estimated.

    Beside the distance rather than on a line of its own: an exact reading with
    no null next to it is read as a distance from zero, which it is not.
    """

    return "" if null is None else f" (null {null:.4f})"


def _render_termination(result: TerminationBenchmarkResult) -> str:
    held_out = result.held_out
    calibration = held_out.calibration
    lines = [
        f"Checkpoint: {result.checkpoint.label} (step {result.checkpoint.step})",
        f"Scored {held_out.games} human game(s): {held_out.resignation_plies} "
        f"resignation ply/plies, {held_out.move_plies} move ply/plies",
        "",
        "Resignation mass",
        f"  at resignation {_optional_value(held_out.mass_at_resignation, '.5f')}  "
        f"at moves {_optional_value(held_out.mass_at_moves, '.5f')}  "
        f"separation {_optional_value(held_out.separation, '.5f')}",
    ]
    if calibration.buckets:
        lines.extend(
            [
                "",
                "By material the player to move was behind",
                f"  error {_optional_value(calibration.error, '.5f')}  "
                f"gap {_optional_value(calibration.gap, '+.5f')} over "
                f"{calibration.plies} scored ply/plies",
                f"    {'pawns behind':<16}{'plies':>8}{'human':>10}{'model':>10}"
                f"{'gap':>10}",
            ]
        )
        for bucket in calibration.buckets:
            lines.append(
                f"    {bucket.bucket:<16}{bucket.plies:>8}"
                f"{bucket.human_rate:>10.5f}{bucket.model_mass:>10.5f}"
                f"{bucket.gap:>+10.5f}"
            )
    for name, reason in sorted(held_out.unavailable.items()):
        lines.append(f"Unavailable: {name}: {reason}")
    if result.recorded_paths:
        lines.append("")
        lines.append(f"Recorded {len(result.recorded_paths)} result(s)")
    return "\n".join(lines) + "\n"


def _silent_actions(guardrails: TerminationGuardrails) -> str:
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
    from anthro_chess.evaluation.results.metrics import (
        LADDER_ADJACENT_RATING_ORDER_ACCURACY,
        LADDER_FITTED_RATING_SLOPE,
        LADDER_FITTED_RATING_SPAN,
        LADDER_RATING_ERROR,
        LADDER_RATING_ORDER_ACCURACY,
    )

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
            + " (pinned at the declared spread rather than fitted within it)"
        )
    if fit.unscored:
        lines.append(
            "  unplaced       "
            + ", ".join(seat.label for seat in fit.unscored)
            + " (no scored game)"
        )
    lines.extend(_render_ladder_openings(result.view))
    lines.extend(_render_ladder_resolution(result))
    for reading in result.readings:
        # Each row's per-seat floor is printed once, in the Seats table below,
        # rather than again beside the same fitted rating here.
        resolves = partial(_resolves, result, reading.label)
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
                    f"  ordering       {reading.order_accuracy:.3f}"
                    + resolves(LADDER_RATING_ORDER_ACCURACY.identifier, 3)
                    + f" pairwise, {reading.adjacent_order_accuracy:.3f}"
                    + resolves(LADDER_ADJACENT_RATING_ORDER_ACCURACY.identifier, 3)
                    + " adjacent"
                ),
                (
                    f"  transfer       slope {reading.slope:.3f}"
                    + resolves(LADDER_FITTED_RATING_SLOPE.identifier, 3)
                    + f", span {reading.span:.0f}"
                    + resolves(LADDER_FITTED_RATING_SPAN.identifier, 0)
                    + f", ladder error {reading.ladder_error:.1f}"
                    + resolves(LADDER_RATING_ERROR.identifier, 1)
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
    lines.extend(_render_ladder_unqualifiable(result))
    if result.recorded_paths:
        lines.extend(
            ["", f"Recorded: {len(result.recorded_paths)} result(s) to the store"]
        )
    else:
        lines.extend(["", "Recorded: nothing; this run did not write to the store"])
    return "\n".join(lines) + "\n"


def _view_population(view: ViewSelection) -> str:
    """Return how much of a pool a view took, and which population it took.

    Both axes are named where both are declared: they read different fields and
    can disagree, so a reader of a rating reading needs the pool rather than
    having to infer it from the class.
    """

    named = f"{view.selected_games} of {view.eligible_games} eligible game(s)"
    if view.speed is not None:
        named += f", {view.speed} alone"
    if view.rating_namespace is not None:
        named += f", {view.rating_namespace} ratings"
    return named


def _render_ladder_openings(view: ViewSelection | None) -> list[str]:
    """Name the human population the seats played from, and what it settles.

    A reader who sees a class and a pool named here could take the fitted
    ratings to be on that pool's scale. They are not: the openings decide what
    the seats play, and the corpus behind the dial decides what their configured
    rating meant. The caveat travels beside the slice rather than in a document.
    """

    if view is None:
        return []
    lines = [f"Openings: {view.name} ({_view_population(view)})"]
    if view.speed is not None or view.rating_namespace is not None:
        lines.extend(
            _wrapped_reason(
                "scale: this names the games the seats start from, not what a "
                "configured rating means — that came from the corpus the model "
                "trained on, and nothing here checks the two name one pool"
            )
        )
    return lines


def _render_ladder_resolution(result: LadderBenchmarkResult) -> list[str]:
    """State how the ± beside every number below was arrived at.

    Printed once, above the readings, because a stated floor and an estimated
    one look identical at the point of use and mean different things.
    """

    from anthro_chess.evaluation.ladder import (
        LADDER_DETERMINISTIC_METHOD,
        LADDER_UNRESOLVED_METHOD,
    )

    resolution = result.resolution
    if resolution is None:
        return ["Resolution: not estimated; this run switched it off"]
    lines = [f"Resolution: {resolution.method}"]
    if resolution.method == LADDER_DETERMINISTIC_METHOD:
        lines.append(
            "  stated         every pairing replays, so the floor beside each "
            "number is exactly zero"
        )
        return lines
    if resolution.method == LADDER_UNRESOLVED_METHOD:
        lines.append(
            f"  too thin       {resolution.redrawn_games} redrawn game(s) show "
            "no spread to bound, so nothing here is qualified"
        )
        return lines
    lines.append(
        f"  resamples      {resolution.fitted_resamples} of "
        f"{resolution.resamples} fitted, over "
        f"{resolution.redrawn_games} redrawn game(s)"
    )
    if resolution.replayed_pairings:
        lines.append(
            f"  held fixed     {resolution.replayed_pairings} pairing(s) that "
            "replay rather than redraw"
        )
    if resolution.non_convergent_resamples:
        lines.append(
            f"  unconverged    {resolution.non_convergent_resamples} resample(s) "
            "ran out of iterations"
        )
    return lines


def _render_ladder_unqualifiable(result: LadderBenchmarkResult) -> list[str]:
    """Name every reported number that has no floor, and why it cannot."""

    resolution = result.resolution
    if resolution is None or not resolution.unqualifiable:
        return []
    lines = ["", "Unqualifiable"]
    for (scope, metric), reason in sorted(resolution.unqualifiable.items()):
        lines.extend(_wrapped_reason(f"{scope} {metric}: {reason}"))
    return lines


def _wrapped_reason(text: str) -> list[str]:
    """Wrap one indented explanation printed beneath the line it qualifies."""

    from anthro_chess.evaluation.results.reporting import MAXIMUM_LINE_WIDTH

    return textwrap.wrap(
        text,
        width=MAXIMUM_LINE_WIDTH,
        initial_indent="    ",
        subsequent_indent="      ",
    )


def _resolves(
    result: LadderBenchmarkResult,
    scope: str,
    metric: str,
    precision: int,
) -> str:
    """Return what one reported number can resolve, as a suffix to print."""

    resolution = result.resolution
    if resolution is None:
        return ""
    floor = resolution.floor(scope, metric)
    if floor is not None:
        return f" ±{floor:.{precision}f}"
    return " (unqualifiable)" if (scope, metric) in resolution.unqualifiable else ""


def _resolves_column(
    result: LadderBenchmarkResult,
    scope: str,
    metric: str,
    precision: int,
    width: int,
) -> str:
    """Return the same, as a fixed-width column.

    Dashed rather than annotated where nothing qualifies the number, because a
    reason long enough to be useful is long enough to break the table. Where
    there is a number to qualify, the ``Unqualifiable`` block below names the
    dash and says why; a seat the fit could not place at all is dashed in both
    columns and explained by the ``unplaced`` line above.
    """

    resolution = result.resolution
    floor = None if resolution is None else resolution.floor(scope, metric)
    value = "-" if floor is None else f"±{floor:.{precision}f}"
    return value.rjust(width)


def _render_ladder_seats(result: LadderBenchmarkResult) -> list[str]:
    """Show each seat's score beside its error profile.

    Strength and error profile are printed together on purpose: a temperature
    that preserves the score rate while moving the preferred-selection rate has
    changed the shape of the mistakes rather than their number, and that is
    invisible in either column alone.

    The scored share sits between them because it is neither: it is what the
    seat's own play did to the ply limit. Both it and the fitted rating carry
    their own resolution, since each is estimated from a different amount of
    this seat's play.
    """

    from anthro_chess.evaluation.results.metrics import (
        LADDER_FITTED_RATING,
        LADDER_SCORED_GAME_RATE,
    )

    lines = [
        "",
        "Seats",
        f"    {'seat':<16}{'games':>7}{'scored':>8}{'±':>8}{'score':>8}"
        f"{'fitted':>9}{'±':>8}{'preferred':>11}{'regret':>9}{'rank':>7}",
    ]
    for seat in result.seats:
        profile = seat.decisions
        lines.append(
            f"    {seat.label:<16}{seat.games:>7}{seat.scored_game_rate:>8.3f}"
            + _resolves_column(
                result,
                seat.label,
                LADDER_SCORED_GAME_RATE.identifier,
                3,
                width=8,
            )
            + f"{seat.score_rate:>8.3f}"
            + (
                f"{seat.fitted_rating:>9.0f}"
                if seat.fitted_rating is not None
                else f"{'-':>9}"
            )
            + _resolves_column(
                result,
                seat.label,
                LADDER_FITTED_RATING.identifier,
                0,
                width=8,
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

    from anthro_chess.evaluation.ladder import RESPONSE_SCOPE
    from anthro_chess.evaluation.results.metrics import (
        LADDER_ABLATED_TEMPERATURE_RESPONSE,
        LADDER_TEMPERATURE_RESPONSE,
        LADDER_TEMPERATURE_RESPONSE_ATTENUATION,
    )

    response = result.response
    if response is None:
        return []
    lines = [
        "",
        f"Temperature response  (series {response.execution.workload_sha256[:12]})",
        (
            f"  conditioned    {response.conditioned_response:+.1f}"
            + _resolves(
                result,
                RESPONSE_SCOPE,
                LADDER_TEMPERATURE_RESPONSE.identifier,
                1,
            )
            + " rating points per unit temperature"
        ),
    ]
    if response.ablated_response is not None:
        lines.append(
            f"  ablated        {response.ablated_response:+.1f}"
            + _resolves(
                result,
                RESPONSE_SCOPE,
                LADDER_ABLATED_TEMPERATURE_RESPONSE.identifier,
                1,
            )
        )
    if response.attenuation is not None:
        lines.append(
            f"  attenuation    {response.attenuation:+.3f}"
            + _resolves(
                result,
                RESPONSE_SCOPE,
                LADDER_TEMPERATURE_RESPONSE_ATTENUATION.identifier,
                3,
            )
            + " of the ablated drift avoided"
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
    from anthro_chess.evaluation import EvaluationPoolError, ViewConfig, load_pool
    from anthro_chess.evaluation.pool import pool_rows
    from anthro_chess.evaluation.reference import (
        REFERENCE_COLUMNS,
        ReferenceConfig,
        ReferenceError,
        human_reference,
        select_bandwidths,
    )
    from anthro_chess.evaluation.views import apply_view, excluded_summary

    try:
        pool = load_pool(arguments.pool)
        selection = apply_view(
            pool.games,
            ViewConfig(
                name="curve-bandwidth",
                maximum_games=arguments.maximum_games,
                require_ratings=True,
                speed=arguments.speed,
            ),
        )
        if not selection.game_ids:
            raise ValueError(
                "the pool holds no game to select a bandwidth from "
                f"({excluded_summary(selection.excluded_games)})"
            )
        rows = pool_rows(
            pool,
            selection.game_ids,
            REFERENCE_COLUMNS,
            error=ValueError,
        )
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
        ReportError,
        ResultsStore,
        ResultsStoreError,
        build_delta_report,
        build_environment_report,
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
        if arguments.pivot == "environment":
            report = build_environment_report(
                results,
                bridges,
                checkpoint=arguments.current,
                families=arguments.family or None,
                metrics=arguments.metric or None,
            )
        else:
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


def _run_eval_promote(arguments: argparse.Namespace) -> int:
    from anthro_chess.evaluation.results import (
        COMMITTED_STORE_DIRECTORY,
        ResultsStore,
        ResultsStoreError,
        resolve_store_root,
    )
    from anthro_chess.training.vehicle import VEHICLE_TRAINING_SHA256

    try:
        source = ResultsStore(resolve_store_root(arguments.store))
        destination = ResultsStore(arguments.into or Path(COMMITTED_STORE_DIRECTORY))
        promoted = source.promote(
            arguments.checkpoint,
            into=destination,
            refusing=(VEHICLE_TRAINING_SHA256,),
        )
    except ResultsStoreError as error:
        print(f"anthro eval promote: {error}", file=sys.stderr)
        return 2

    print(
        f"Promoted {len(promoted)} record(s) for {arguments.checkpoint} from "
        f"{source.root} into {destination.root}"
    )
    for path in promoted:
        print(f"  {path}")
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


def _run_eval_noise_sample(arguments: argparse.Namespace) -> int:
    """Take one process's readings, so a caller can measure across processes."""

    from anthro_chess.config import ConfigError, load_config, load_config_json
    from anthro_chess.evaluation import InferenceBenchmarkConfig
    from anthro_chess.evaluation.execution_noise import (
        ExecutionNoiseError,
        sample_execution_noise,
    )
    from anthro_chess.evaluation.results import metric_column_width

    try:
        if arguments.selection is not None:
            selected = arguments.selection
            raw = sys.stdin.read() if selected == "-" else selected
            resolved = load_config_json(InferenceBenchmarkConfig, raw)
        else:
            resolved = load_config(
                InferenceBenchmarkConfig,
                path=arguments.config,
                overrides=arguments.set,
            )
        samples = sample_execution_noise(resolved, run_root=_run_root())
    except (ConfigError, ExecutionNoiseError) as error:
        print(f"anthro eval noise sample: {error}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        print(
            json.dumps(
                [sample.as_record() for sample in samples],
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    for sample in samples:
        print(
            f"Checkpoint: {sample.checkpoint.label} on "
            f"{sample.execution.environment_label()}"
        )
        print("One reading in this process, recorded nowhere:")
        labels = {value.metric: value.value for value in sample.values}
        width = metric_column_width(labels)
        for metric, value in sorted(labels.items()):
            print(f"  {metric:<{width}} {value:.6g}")
    return 0


def _run_eval_noise_plan(arguments: argparse.Namespace) -> int:
    from anthro_chess.evaluation.results import (
        REALIZING_UNITS_VERSION,
        MetricRegistryError,
        NoiseCharacterizationError,
        ResultsStore,
        ResultsStoreError,
        games_to_resolve,
        metric_definition,
        resolve_store_root,
        self_combined_floor,
    )

    try:
        metric = metric_definition(arguments.metric).identifier
        store = ResultsStore(resolve_store_root(arguments.store))
        # Results arrive in recording order, so the last reading that read its
        # own spread over a counted draw of games is the one that still
        # describes the metric. An older envelope counted that draw as the whole
        # pass, and extrapolating from it would answer the sizing question in a
        # unit the record cannot be read as carrying.
        reading = next(
            (
                (measured.dispersion, envelope.data)
                for envelope in reversed(store.results())
                if envelope.envelope_version >= REALIZING_UNITS_VERSION
                and (measured := envelope.measurement(metric)) is not None
                and measured.dispersion is not None
            ),
            None,
        )
        if reading is None:
            print(
                f"anthro eval noise plan: no reading at envelope version "
                f"{REALIZING_UNITS_VERSION} or above records a sampled "
                f"dispersion for {metric}",
                file=sys.stderr,
            )
            return 2
        spread, dataset = reading
        required = games_to_resolve(spread, effect=arguments.effect)
        floor = self_combined_floor(spread)
        # `required` counts games realizing the metric, so a pool has to be
        # larger by whatever rate it realizes them at. Rounded up in integers:
        # a tiny effect drives `required` past what a float can divide.
        pool = (
            None
            if dataset is None or spread.units is None
            else (required * dataset.selected_games + spread.units - 1) // spread.units
        )
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
                    "required_realizing_games": required,
                    "required_pool_games": pool,
                    "measured_realizing_games": spread.units,
                    "measured_floor": floor,
                    "source": spread.source,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(
        f"{metric}: resolving an effect of {arguments.effect:.6g} needs about "
        f"{required} game(s) realizing it"
        + ("." if pool is None else f", or about {pool} pool game(s).")
    )
    print(
        f"Measured floor {floor:.6g} over {spread.units} realizing game(s) "
        f"({spread.source})."
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
                for line in _wrapped_reason(
                    f"no sampling floor can exist: {metric.no_sampling_floor_reason}"
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
        placement = _run_root() or Path(WORKING_ARTIFACTS_DIRECTORY)
        result = run_training(
            resolved,
            output_directory=(
                arguments.output_directory or placement / resolved.value.run_name
            ),
            store=store,
            detail=detail,
            verify_data=arguments.verify_data,
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


def _configured_archive_path(
    resolved: ResolvedConfig[PrepareConfig],
    explicit_input: Path | None,
    artifact_root: Path | None,
) -> Path:
    """Return where the archive a data selection names was acquired to."""

    from anthro_chess.config import ConfigError

    if explicit_input is not None:
        return explicit_input
    archives = resolved.value.archives
    if not archives:
        raise ConfigError(
            "input path is required because the selected data "
            "configuration has no archive"
        )
    if len(archives) > 1:
        # A selection spanning many archives has no single default input, and
        # guessing one would silently prepare a different month than intended.
        raise ConfigError(
            f"input path is required because the selected data configuration "
            f"pins {len(archives)} archives"
        )
    return (
        _archive_artifact_root(resolved, archives[0], artifact_root)
        / "raw"
        / (archives[0].file_name)
    )


def _archive_artifact_root(
    resolved: ResolvedConfig[PrepareConfig],
    archive: ArchiveConfig,
    artifact_root: Path | None,
) -> Path:
    """Return the artifact directory one of a selection's archives lives in.

    An archive names its own directory when it has one so that selections can
    share an acquired file, and falls back to the selection's.
    """

    return _data_output_path(
        artifact_root, archive.artifact_name or resolved.value.artifact_name
    )


def _pinned_archives(
    resolved: ResolvedConfig[PrepareConfig],
) -> list[PinnedArchive]:
    """Return where each pinned archive was acquired to and its account counts."""

    from anthro_chess.data.census import PinnedArchive

    archives = []
    for archive in resolved.value.archives:
        root = _archive_artifact_root(resolved, archive, None)
        archives.append(
            PinnedArchive(
                path=root / "raw" / archive.file_name,
                counts_path=_counts_path(root, archive.file_name),
                sha256=archive.sha256,
            )
        )
    return archives


def _counts_path(artifact_root: Path, file_name: str) -> Path:
    from anthro_chess.data.census import ACCOUNT_GAMES_SUFFIX, CENSUS_DIRECTORY

    return artifact_root / CENSUS_DIRECTORY / f"{file_name}{ACCOUNT_GAMES_SUFFIX}"


def _acquired_archive_inputs(
    resolved: ResolvedConfig[PrepareConfig],
    artifact_root: Path | None,
) -> list[tuple[Path, Path]]:
    """Return each pinned archive that is on disk, and where its counts go.

    A selection pins more archives than the machine holds at once, so the ones
    absent are the ones already prepared and deleted rather than an error.
    """

    acquired = []
    for archive in resolved.value.archives:
        root = _archive_artifact_root(resolved, archive, artifact_root)
        path = root / "raw" / archive.file_name
        if path.is_file():
            acquired.append((path, _counts_path(root, archive.file_name)))
    return acquired


def _archive_counts_path(
    resolved: ResolvedConfig[PrepareConfig],
    input_path: Path,
    artifact_root: Path | None,
) -> Path | None:
    """Return where preparing this input leaves the census its account counts.

    An input that is not one of the selection's acquired archives leaves none.
    The census asks about the accounts of archives a corpus is built from, and
    a PGN handed to `--input` from anywhere on the machine is not one of those.
    """

    for archive in resolved.value.archives:
        root = _archive_artifact_root(resolved, archive, artifact_root)
        if root / "raw" / archive.file_name == input_path:
            return _counts_path(root, archive.file_name)
    return None


def _census_answers_path(resolved: ResolvedConfig[PrepareConfig]) -> Path:
    """Return where the census records what the source said about an account.

    Keyed by the source rather than by the selection, because account status is
    the source's judgement about an account and not a property of any corpus.
    A second selection over the same source inherits every answer instead of
    spending weeks of the same rate limit asking again.
    """

    from anthro_chess.data.census import ANSWERS_FILE

    root = _data_output_path(None, f"{resolved.value.source.id}-account-census")
    return root / ANSWERS_FILE


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
    "dependency": _render_dependency,
    "novelty": _render_novelty,
    "puzzles": _render_puzzles,
    "rollout": _render_rollout,
    "decisions": _render_decisions,
    "termination": _render_termination,
    "ladder": _render_ladder,
}


if __name__ == "__main__":  # pragma: no cover - console scripts call main directly
    raise SystemExit(main())
