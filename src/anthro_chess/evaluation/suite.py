"""Running every benchmark against one checkpoint from one entry point.

Seventeen ``anthro eval`` subcommands and none of them ran more than one
benchmark, so a full reading was assembled by hand: an invocation per
benchmark, each with its own selection file, the checkpoint threaded through as
an override, and an order the caller had to remember. At the durations involved
that stops being an ergonomic complaint. A sweep measured in hours fails
partway, and without resume a failure in the last benchmark discards every
reading before it.

So this is a driver rather than a shell script, and it owns four things a
script cannot:

- **Composition.** Each benchmark keeps its own selection file. The suite
  names the file and the checkpoint; it never restates a benchmark's settings,
  because a second copy of a declared workload is a second thing to keep
  correct.
- **Ordering.** Decision decomposition reads the games the rollout generated,
  so it runs after it. That constraint is declared and enforced here rather
  than documented, and the whole plan resolves before the first benchmark
  runs: an unreadable selection, a missing pool, or a rollout that was
  configured to discard its games fails in the first second rather than the
  first hour.
- **Recording.** Whether a reading is committed is decided per benchmark
  within one sweep, so a sweep can commit the readings it meant to keep and
  leave the rest as evidence about the instrument.
- **Resume.** Each step's outcome is written to a machine-local ledger as it
  finishes, so a sweep that dies late keeps what it already read.

The reduced sweep is the default and the full sweep is opt-in, because a sweep
measured in hours is not a default anyone will run on a new checkpoint. Both
shrink only sample counts — view sizes, seeds, games per position, resamples —
never the grids that decide *what* is measured. Even so a reduced view is a
different data component and therefore a different series, so a reduced sweep
is not a partial down payment on a full one, and the scale is named in the
output and in the ledger for exactly that reason.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field, StrictBool, model_validator

from anthro_chess.config import ConfigError, ConfigModel, ResolvedConfig, load_config
from anthro_chess.evaluation import roots
from anthro_chess.inference.config import LATEST_CHECKPOINT, ModelRunnerConfig

if TYPE_CHECKING:
    from anthro_chess.evaluation.games import GameRecord
    from anthro_chess.evaluation.results import DetailStore, ResultsStore

SUITE_VERSION = 1
#: Bumped when a ledger written by an older build can no longer be resumed.
SWEEP_LEDGER_VERSION = 1
LEDGER_FILE_NAME = "sweep.json"
#: Where a step's generated games are handed to the step that reads them. It is
#: sweep state rather than a benchmark artifact: the committed tier never holds
#: games, and a sweep that records nothing still has to feed its own dependent
#: steps.
GAMES_FILE_SUFFIX = "-games.json"

logger = logging.getLogger(__name__)


class SuiteError(ValueError):
    """Raised when a sweep cannot be planned or carried out as configured."""


class SuiteScale(StrEnum):
    """How much data one sweep reads.

    A scale is a sample size rather than a measurement setting, but a smaller
    view still produces its own fingerprint, so the two scales are named and
    never silently mixed within one ledger.
    """

    #: Sample counts shrunk so a new checkpoint can be swept in minutes.
    REDUCED = "reduced"
    #: Every benchmark at its own declared size.
    FULL = "full"


class StepStatus(StrEnum):
    """How one benchmark of a sweep ended."""

    COMPLETED = "completed"
    FAILED = "failed"
    #: Not run: its dependency failed, or a resumed sweep already has it.
    SKIPPED = "skipped"


class SuiteBenchmarkConfig(ConfigModel):
    """One benchmark's place in a sweep.

    Everything about *how* the benchmark measures stays in its own selection
    file. What lives here is only what the file cannot know: where it sits in
    the order, whether this sweep commits its reading, and how it shrinks.
    """

    #: The benchmark's existing selection file. Absent only for a benchmark
    #: that reads another step's output rather than a configuration of its own.
    config: Path | None = None
    #: Steps that must finish first. Ordering constraints a benchmark's inputs
    #: impose are added automatically and need not be repeated here.
    after: tuple[str, ...] = ()
    enabled: StrictBool = True
    #: Whether this sweep commits the benchmark's reading. A sweep can commit
    #: some readings and leave the rest unrecorded, which is what a shakedown
    #: over most of the suite plus one kept baseline looks like.
    record: StrictBool = True
    #: Dotted TOML overrides applied to the selection at every scale.
    overrides: tuple[str, ...] = ()
    #: Further overrides applied only to the reduced sweep. Sample counts, not
    #: grids: a reduced sweep measures the same quantities less precisely
    #: rather than measuring fewer of them.
    reduced: tuple[str, ...] = ()
    #: Which sweeps include this step. A benchmark whose cost is a grid rather
    #: than a sample size has no affordable reduction at all — shrinking the
    #: grid would measure something else — so it declares the full sweep alone
    #: instead of being nominally reduced and still unaffordable.
    scales: tuple[SuiteScale, ...] = (SuiteScale.REDUCED, SuiteScale.FULL)


class SuiteConfig(ConfigModel):
    """Code-owned schema for ``anthro eval suite``."""

    name: str = Field(
        default="checkpoint", min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$"
    )
    #: The one checkpoint every benchmark in the sweep measures. It replaces
    #: whatever model selection a benchmark's own file carries, which is what
    #: makes this an entry point taking a checkpoint rather than a batch of
    #: unrelated runs.
    model: ModelRunnerConfig = ModelRunnerConfig()
    checkpoint_label: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    #: Keyed by benchmark name so one step can be adjusted from the command
    #: line without restating the sweep.
    benchmarks: dict[str, SuiteBenchmarkConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_benchmarks(self) -> SuiteConfig:
        if not any(entry.enabled for entry in self.benchmarks.values()):
            raise ValueError("a suite needs at least one enabled benchmark")
        return self


@dataclass(frozen=True)
class Benchmark:
    """How the driver runs one benchmark that it did not define.

    The benchmarks are independent of the suite and stay that way: each owns
    its configuration schema, its entry point, and its errors, and this record
    is only the adapter that lets one driver call all of them.
    """

    name: str
    #: ``None`` for a step whose input is another step's output.
    schema: type[ConfigModel] | None
    #: Which configured paths are rooted beneath the shared data root, exactly
    #: as the benchmark's own command roots them.
    artifact_fields: tuple[str, ...]
    #: Errors this benchmark raises for a bad reading, which the sweep reports
    #: as a failed step rather than letting them end the whole sweep.
    errors: tuple[type[Exception], ...]
    #: ``(resolved, *, run_root, store, detail) -> result``.
    invoke: Callable[..., Any]
    #: Whether the benchmark writes results to the store at all. Decision
    #: decomposition does not: it has no result kind, and a decomposition over
    #: one payload is a diagnostic rather than a series.
    records_results: bool = True
    #: Returns the games this benchmark generated, when a later step reads them.
    games: Callable[[Any], tuple[GameRecord, ...]] | None = None
    #: Whether a configuration retained its games at all, checked while the
    #: plan resolves so a dependent step cannot be starved an hour in.
    retains_games: Callable[[Any], bool] | None = None
    #: The step whose game payload this benchmark reads.
    games_from: str | None = None


@dataclass(frozen=True)
class PlannedBenchmark:
    """One resolved step of a sweep, ready to run."""

    name: str
    benchmark: Benchmark = field(repr=False)
    #: ``None`` for a step configured entirely by its input.
    resolved: ResolvedConfig[Any] | None = field(default=None, repr=False)
    record: bool = False
    source: Path | None = None
    overrides: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        """Return the readable plan entry, without the resolved settings."""

        return {
            "benchmark": self.name,
            "config": None if self.source is None else str(self.source),
            "overrides": list(self.overrides),
            "record": self.record,
            "reads_games_from": self.benchmark.games_from,
        }


@dataclass(frozen=True)
class SuitePlan:
    """Everything one sweep will do, resolved before any of it runs."""

    suite: str
    scale: SuiteScale
    steps: tuple[PlannedBenchmark, ...]
    checkpoint_label: str | None
    #: Digest over the plan, so a resumed sweep cannot continue a different one.
    sha256: str

    @property
    def recording(self) -> tuple[str, ...]:
        """Return the names of the steps whose readings this sweep commits."""

        return tuple(step.name for step in self.steps if step.record)

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable plan."""

        return {
            "version": SUITE_VERSION,
            "suite": self.suite,
            "scale": self.scale.value,
            "checkpoint_label": self.checkpoint_label,
            "plan_sha256": self.sha256,
            "steps": [step.describe() for step in self.steps],
        }


@dataclass(frozen=True)
class StepOutcome:
    """What one benchmark of a sweep did, read, and cost."""

    name: str
    status: StepStatus
    seconds: float
    results: int = 0
    measurements: int = 0
    recorded_paths: tuple[Path, ...] = ()
    detail_paths: tuple[Path, ...] = ()
    #: Why a step was skipped, or what a failed one raised.
    note: str | None = None
    #: The benchmark's own result, kept for the caller to render. Never
    #: serialized: this is the whole reading, not a summary of it.
    result: Any | None = field(default=None, repr=False)
    #: True when a resumed sweep took this outcome from the ledger.
    from_ledger: bool = False

    def as_record(self) -> dict[str, Any]:
        """Return the summary entry the ledger and the JSON view both use."""

        return {
            "benchmark": self.name,
            "status": self.status.value,
            "seconds": round(self.seconds, 3),
            "results": self.results,
            "measurements": self.measurements,
            "recorded_paths": [str(path) for path in self.recorded_paths],
            "detail_paths": [str(path) for path in self.detail_paths],
            "note": self.note,
        }


@dataclass(frozen=True)
class SuiteRun:
    """One sweep's outcomes, in the order they were attempted."""

    plan: SuitePlan
    sweep_root: Path
    outcomes: tuple[StepOutcome, ...]

    @property
    def failed(self) -> tuple[StepOutcome, ...]:
        """Return the steps that raised."""

        return tuple(item for item in self.outcomes if item.status is StepStatus.FAILED)

    @property
    def seconds(self) -> float:
        """Return what the sweep cost, excluding steps taken from the ledger."""

        return sum(item.seconds for item in self.outcomes if not item.from_ledger)

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable sweep summary."""

        return {
            "version": SUITE_VERSION,
            "suite": self.plan.suite,
            "scale": self.plan.scale.value,
            "checkpoint_label": self.plan.checkpoint_label,
            "plan_sha256": self.plan.sha256,
            "sweep_root": str(self.sweep_root),
            "seconds": round(self.seconds, 3),
            "steps": [item.as_record() for item in self.outcomes],
        }


def benchmark_registry() -> dict[str, Benchmark]:
    """Return every benchmark a sweep can run, keyed by its command name.

    Built on demand rather than at import, because the benchmarks pull in the
    model stack and a suite selection can be planned and printed without it.
    """

    from anthro_chess.evaluation import (
        CheckpointEvaluationConfig,
        CheckpointEvaluationError,
        InferenceBenchmarkConfig,
        InferenceBenchmarkError,
        LadderBenchmarkConfig,
        LadderBenchmarkError,
        LeakageError,
        NoveltyBenchmarkConfig,
        NoveltyBenchmarkError,
        PuzzleBenchmarkConfig,
        PuzzleBenchmarkError,
        RolloutBenchmarkConfig,
        RolloutBenchmarkError,
        TerminationBenchmarkConfig,
        TerminationBenchmarkError,
        benchmark_inference,
        benchmark_ladder,
        benchmark_novelty,
        benchmark_puzzles,
        benchmark_rollout,
        benchmark_termination,
        evaluate_checkpoint,
    )
    from anthro_chess.evaluation.results import ResultsStoreError

    store_errors = (ResultsStoreError,)
    return {
        benchmark.name: benchmark
        for benchmark in (
            Benchmark(
                name="inference",
                schema=InferenceBenchmarkConfig,
                artifact_fields=(),
                errors=(InferenceBenchmarkError, *store_errors),
                invoke=benchmark_inference,
            ),
            Benchmark(
                name="run",
                schema=CheckpointEvaluationConfig,
                artifact_fields=roots.CHECKPOINT_ARTIFACT_FIELDS,
                errors=(CheckpointEvaluationError, LeakageError, *store_errors),
                invoke=evaluate_checkpoint,
            ),
            Benchmark(
                name="novelty",
                schema=NoveltyBenchmarkConfig,
                artifact_fields=roots.NOVELTY_ARTIFACT_FIELDS,
                errors=(NoveltyBenchmarkError, *store_errors),
                invoke=benchmark_novelty,
            ),
            Benchmark(
                name="puzzles",
                schema=PuzzleBenchmarkConfig,
                artifact_fields=roots.PUZZLE_ARTIFACT_FIELDS,
                errors=(PuzzleBenchmarkError, *store_errors),
                invoke=benchmark_puzzles,
            ),
            Benchmark(
                name="rollout",
                schema=RolloutBenchmarkConfig,
                artifact_fields=roots.ROLLOUT_ARTIFACT_FIELDS,
                errors=(RolloutBenchmarkError, *store_errors),
                invoke=benchmark_rollout,
                games=_rollout_games,
                retains_games=lambda config: bool(config.detail.retain_games),
            ),
            Benchmark(
                name="decisions",
                schema=None,
                artifact_fields=(),
                errors=_decision_errors(),
                invoke=_invoke_decisions,
                records_results=False,
                games_from="rollout",
            ),
            Benchmark(
                name="termination",
                schema=TerminationBenchmarkConfig,
                artifact_fields=roots.TERMINATION_ARTIFACT_FIELDS,
                errors=(TerminationBenchmarkError, *store_errors),
                invoke=benchmark_termination,
            ),
            Benchmark(
                name="ladder",
                schema=LadderBenchmarkConfig,
                artifact_fields=roots.LADDER_ARTIFACT_FIELDS,
                errors=(LadderBenchmarkError, *store_errors),
                invoke=benchmark_ladder,
            ),
        )
    }


def resolve_suite(
    resolved_config: ResolvedConfig[SuiteConfig],
    *,
    scale: SuiteScale = SuiteScale.REDUCED,
    no_record: bool = False,
    record_only: Sequence[str] | None = None,
    registry: Mapping[str, Benchmark] | None = None,
) -> SuitePlan:
    """Resolve every benchmark's configuration and the order they run in.

    Everything that can fail cheaply fails here: an unknown benchmark, a
    missing or invalid selection file, a cycle, a dependency on a step that is
    not in the sweep, and a step configured to discard the games another step
    needs. A sweep measured in hours should not discover any of those at the
    end of it.
    """

    known = dict(registry) if registry is not None else benchmark_registry()
    config = resolved_config.value
    entries = {
        name: entry
        for name, entry in config.benchmarks.items()
        if entry.enabled and scale in entry.scales
    }
    if not entries:
        raise SuiteError(f"no benchmark in this suite runs at the {scale} scale")
    unknown = sorted(set(entries) - set(known))
    if unknown:
        raise SuiteError(
            f"unknown benchmark(s) {', '.join(unknown)}; this build runs "
            f"{', '.join(sorted(known))}"
        )
    if record_only is not None:
        absent = sorted(set(record_only) - set(entries))
        if absent:
            raise SuiteError(
                f"cannot record {', '.join(absent)}; this sweep runs "
                f"{', '.join(sorted(entries))}"
            )

    ordered = _order(entries, known)
    steps: list[PlannedBenchmark] = []
    for name in ordered:
        entry = entries[name]
        benchmark = known[name]
        record = _step_records(
            name,
            entry,
            benchmark,
            no_record=no_record,
            record_only=record_only,
        )
        resolved = _resolve_benchmark(name, entry, benchmark, config, scale=scale)
        steps.append(
            PlannedBenchmark(
                name=name,
                benchmark=benchmark,
                resolved=resolved,
                record=record,
                source=entry.config,
                overrides=_applied_overrides(entry, config, scale=scale),
            )
        )
    _validate_game_dependencies(steps, entries)

    return SuitePlan(
        suite=config.name,
        scale=scale,
        steps=tuple(steps),
        checkpoint_label=config.checkpoint_label,
        sha256=_plan_digest(config, scale, steps),
    )


def run_suite(
    plan: SuitePlan,
    *,
    sweep_root: Path,
    run_root: Path | None = None,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
    resume: bool = False,
    observer: Callable[[StepOutcome], None] | None = None,
) -> SuiteRun:
    """Run a resolved plan, writing each step's outcome as it finishes.

    A failed step does not end the sweep. The remaining independent benchmarks
    still run and still record, because one broken instrument is not a reason
    to discard every reading beside it. Only a step that reads a failed step's
    *output* is skipped; a step that merely declared it runs later has no such
    dependency and still runs.
    """

    if plan.recording and store is None:
        raise SuiteError(
            "this sweep commits "
            f"{', '.join(plan.recording)} and needs a results store; pass a "
            "store or run the whole sweep without recording"
        )

    sweep_root.mkdir(parents=True, exist_ok=True)
    ledger = _load_ledger(sweep_root, plan) if resume else {}
    outcomes: list[StepOutcome] = []
    completed: set[str] = {
        name
        for name, entry in ledger.items()
        if entry.get("status") == StepStatus.COMPLETED.value
    }
    starved: dict[str, str] = {}

    for step in plan.steps:
        if step.name in completed:
            outcome = _ledger_outcome(step.name, ledger[step.name])
        elif step.name in starved:
            outcome = StepOutcome(
                name=step.name,
                status=StepStatus.SKIPPED,
                seconds=0.0,
                note=starved[step.name],
            )
        else:
            outcome = _run_step(
                step,
                sweep_root=sweep_root,
                run_root=run_root,
                store=store if step.record else None,
                detail=detail if step.record else None,
            )
        if outcome.status is StepStatus.COMPLETED:
            _hand_off_games(step, outcome, plan, sweep_root)
        else:
            for dependent in _dependents(plan, step.name):
                starved.setdefault(
                    dependent,
                    f"{step.name} did not complete, so there is nothing to read",
                )
        outcomes.append(outcome)
        if observer is not None:
            observer(outcome)
        _write_ledger(sweep_root, plan, outcomes)

    return SuiteRun(plan=plan, sweep_root=sweep_root, outcomes=tuple(outcomes))


def sweep_directory(root: Path, plan: SuitePlan) -> Path:
    """Return where one sweep keeps its ledger and its handed-off payloads.

    Derived from the plan rather than from a timestamp, so re-running the same
    command with ``--resume`` finds the sweep it is continuing, and two
    checkpoints swept side by side never share a directory.
    """

    return root / f"{plan.suite}-{plan.scale.value}-{plan.sha256[:12]}"


def _run_step(
    step: PlannedBenchmark,
    *,
    sweep_root: Path,
    run_root: Path | None,
    store: ResultsStore | None,
    detail: DetailStore | None,
) -> StepOutcome:
    """Run one benchmark, turning its own errors into a failed step."""

    logger.info("Suite step %s: starting", step.name)
    started = time.perf_counter()
    try:
        if step.benchmark.games_from is not None:
            result = step.benchmark.invoke(
                _games_payload(sweep_root, step.benchmark.games_from)
            )
        else:
            result = step.benchmark.invoke(
                step.resolved,
                run_root=run_root,
                store=store,
                detail=detail,
            )
    except step.benchmark.errors as error:
        seconds = time.perf_counter() - started
        logger.warning("Suite step %s: failed after %.1fs", step.name, seconds)
        return StepOutcome(
            name=step.name,
            status=StepStatus.FAILED,
            seconds=seconds,
            note=str(error),
        )

    seconds = time.perf_counter() - started
    envelopes = _reported(result, "envelopes")
    logger.info("Suite step %s: completed in %.1fs", step.name, seconds)
    return StepOutcome(
        name=step.name,
        status=StepStatus.COMPLETED,
        seconds=seconds,
        results=len(envelopes),
        measurements=sum(len(envelope.measurements) for envelope in envelopes),
        recorded_paths=_reported(result, "recorded_paths"),
        detail_paths=_reported(result, "detail_paths"),
        result=result,
    )


def _hand_off_games(
    step: PlannedBenchmark,
    outcome: StepOutcome,
    plan: SuitePlan,
    sweep_root: Path,
) -> None:
    """Write a completed step's games where the step that reads them looks.

    A payload rather than an in-memory hand-off, because resume has to work
    across processes: a sweep whose rollout succeeded and whose decomposition
    failed should not replay the rollout to try the decomposition again.
    """

    extract = step.benchmark.games
    if extract is None or outcome.result is None or not _dependents(plan, step.name):
        return

    from anthro_chess.evaluation.games import write_game_records
    from anthro_chess.evaluation.results import DetailStore

    records = extract(outcome.result)
    if not records:
        return
    path = _games_payload(sweep_root, step.name)
    write_game_records(
        DetailStore(sweep_root),
        path.name,
        records,
        description=f"Games generated by the {step.name} step of one sweep",
    )
    logger.info("Suite step %s: handed off %d game(s)", step.name, len(records))


def _games_payload(sweep_root: Path, producer: str) -> Path:
    return sweep_root / f"{producer}{GAMES_FILE_SUFFIX}"


def _dependents(plan: SuitePlan, producer: str) -> tuple[str, ...]:
    """Return the steps that read one step's games."""

    return tuple(
        step.name for step in plan.steps if step.benchmark.games_from == producer
    )


def _rollout_games(result: Any) -> tuple[GameRecord, ...]:
    """Return every game a rollout retained, across its whole matrix."""

    return tuple(record for cell in result.cells for record in cell.records)


def _decision_errors() -> tuple[type[Exception], ...]:
    from anthro_chess.evaluation.decisions import DecisionDecompositionError

    return (DecisionDecompositionError, OSError)


def _invoke_decisions(path: Path) -> Any:
    from anthro_chess.evaluation.decisions import decompose_game_records

    return decompose_game_records(path)


def _reported(result: Any, name: str) -> tuple[Any, ...]:
    """Return one of a result's reported tuples, or none where it has that one.

    Read by attribute rather than through a shared base class, because the
    benchmarks are independent of the suite and each owns its own result type.
    Decision decomposition reports none of these.
    """

    return tuple(getattr(result, name, ()))


def _order(
    entries: Mapping[str, SuiteBenchmarkConfig],
    known: Mapping[str, Benchmark],
) -> tuple[str, ...]:
    """Return the steps in dependency order, keeping declared order for ties.

    Stable rather than merely valid: a sweep that reordered itself between two
    runs would make two ledgers hard to compare for no benefit.
    """

    blockers = {
        name: _blockers(name, entries[name], known[name], entries) for name in entries
    }
    remaining = list(entries)
    ordered: list[str] = []
    done: set[str] = set()
    while remaining:
        ready = next(
            (name for name in remaining if blockers[name] <= done),
            None,
        )
        if ready is None:
            raise SuiteError(
                f"benchmark ordering has a cycle among {', '.join(sorted(remaining))}"
            )
        ordered.append(ready)
        done.add(ready)
        remaining.remove(ready)
    return tuple(ordered)


def _blockers(
    name: str,
    entry: SuiteBenchmarkConfig,
    benchmark: Benchmark,
    entries: Mapping[str, SuiteBenchmarkConfig],
) -> set[str]:
    """Return what must finish before one step, declared and implied."""

    required = set(entry.after)
    if benchmark.games_from is not None:
        required.add(benchmark.games_from)
    missing = sorted(required - set(entries))
    if missing:
        raise SuiteError(
            f"the {name} step runs after {', '.join(missing)}, which "
            "this sweep does not include"
        )
    return required


def _step_records(
    name: str,
    entry: SuiteBenchmarkConfig,
    benchmark: Benchmark,
    *,
    no_record: bool,
    record_only: Sequence[str] | None,
) -> bool:
    """Decide whether one step commits its reading."""

    asked = entry.record or (record_only is not None and name in set(record_only))
    if asked and not benchmark.records_results:
        raise SuiteError(
            f"the {name} step has no result kind to commit and cannot be "
            "recorded; set record = false for it"
        )
    if no_record or not benchmark.records_results:
        return False
    if record_only is not None:
        return name in set(record_only)
    return bool(entry.record)


def _resolve_benchmark(
    name: str,
    entry: SuiteBenchmarkConfig,
    benchmark: Benchmark,
    suite: SuiteConfig,
    *,
    scale: SuiteScale,
) -> ResolvedConfig[Any] | None:
    """Load one benchmark's own selection and point it at this checkpoint."""

    if benchmark.schema is None:
        if entry.config is not None:
            raise SuiteError(
                f"the {name} step reads another step's output and takes no "
                "configuration file of its own"
            )
        return None
    if entry.config is None:
        raise SuiteError(f"the {name} step needs config to name its selection")

    overrides = _scaled_overrides(entry, scale=scale)
    try:
        resolved = load_config(
            benchmark.schema,
            path=entry.config,
            overrides=overrides,
        )
    except ConfigError as error:
        raise SuiteError(f"the {name} step: {error}") from error
    resolved = roots.resolve_artifact_roots(
        resolved,
        fields=benchmark.artifact_fields,
        overrides=overrides,
    )
    return _with_checkpoint(resolved, suite)


def _with_checkpoint(
    resolved: ResolvedConfig[Any],
    suite: SuiteConfig,
) -> ResolvedConfig[Any]:
    """Replace a benchmark's model selection with the sweep's checkpoint.

    Replaced rather than merged: a suite takes one checkpoint, and a merge
    would let a benchmark file that pinned its own run quietly measure
    something else than the rest of the sweep did.
    """

    from anthro_chess.config import ConfigProvenance

    update: dict[str, Any] = {"model": suite.model}
    if suite.checkpoint_label is not None:
        update["checkpoint_label"] = suite.checkpoint_label
    applied = _checkpoint_overrides(suite)
    return ResolvedConfig(
        value=resolved.value.model_copy(update=update),
        provenance=ConfigProvenance(
            source=resolved.provenance.source,
            overrides=(*resolved.provenance.overrides, *applied),
        ),
    )


def _checkpoint_overrides(suite: SuiteConfig) -> tuple[str, ...]:
    """Render the sweep's checkpoint selection as the overrides it stands for.

    Provenance only: the value itself is replaced wholesale above. Recording
    them keeps a committed result able to say which checkpoint selection the
    sweep applied on top of the benchmark's own file.
    """

    model = suite.model
    applied: list[str] = []
    if model.checkpoint_path is not None:
        applied.append(f"model.checkpoint_path={_toml(str(model.checkpoint_path))}")
    else:
        if model.run_path is not None:
            applied.append(f"model.run_path={_toml(str(model.run_path))}")
        if model.checkpoint != LATEST_CHECKPOINT:
            applied.append(f"model.checkpoint={_toml(model.checkpoint)}")
    if model.device != "auto":
        applied.append(f"model.device={_toml(model.device)}")
    if suite.checkpoint_label is not None:
        applied.append(f"checkpoint_label={_toml(suite.checkpoint_label)}")
    return tuple(applied)


def _scaled_overrides(
    entry: SuiteBenchmarkConfig,
    *,
    scale: SuiteScale,
) -> tuple[str, ...]:
    if scale is SuiteScale.REDUCED:
        return (*entry.overrides, *entry.reduced)
    return tuple(entry.overrides)


def _applied_overrides(
    entry: SuiteBenchmarkConfig,
    suite: SuiteConfig,
    *,
    scale: SuiteScale,
) -> tuple[str, ...]:
    return (*_scaled_overrides(entry, scale=scale), *_checkpoint_overrides(suite))


def _validate_game_dependencies(
    steps: Sequence[PlannedBenchmark],
    entries: Mapping[str, SuiteBenchmarkConfig],
) -> None:
    """Reject a sweep whose dependent step would find nothing to read."""

    planned = {step.name: step for step in steps}
    for step in steps:
        producer_name = step.benchmark.games_from
        if producer_name is None:
            continue
        producer = planned[producer_name]
        if producer.benchmark.games is None:
            raise SuiteError(
                f"the {step.name} step reads generated games, which the "
                f"{producer_name} step does not produce"
            )
        retains = producer.benchmark.retains_games
        if (
            retains is not None
            and producer.resolved is not None
            and not retains(producer.resolved.value)
        ):
            raise SuiteError(
                f"the {step.name} step reads the games the {producer_name} "
                f"step generates, but {entries[producer_name].config} discards "
                "them; retain them or drop the dependent step"
            )


def _plan_digest(
    suite: SuiteConfig,
    scale: SuiteScale,
    steps: Sequence[PlannedBenchmark],
) -> str:
    """Digest what a resumed sweep must still be doing to be the same sweep."""

    payload = {
        "version": SUITE_VERSION,
        "suite": suite.name,
        "scale": scale.value,
        "checkpoint_label": suite.checkpoint_label,
        "model": suite.model.model_dump(mode="json"),
        "steps": [step.describe() for step in steps],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _toml(value: str) -> str:
    """Render one string as the TOML basic string an override carries."""

    return json.dumps(value)


def _load_ledger(sweep_root: Path, plan: SuitePlan) -> dict[str, dict[str, Any]]:
    """Read a previous sweep's ledger, refusing one for a different sweep."""

    path = sweep_root / LEDGER_FILE_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SuiteError(f"cannot resume from {path}: {error}") from error
    if payload.get("ledger_version") != SWEEP_LEDGER_VERSION:
        raise SuiteError(f"{path} was written by another build and cannot be resumed")
    if payload.get("plan_sha256") != plan.sha256:
        raise SuiteError(
            f"{path} records a different sweep; resume the sweep it belongs "
            "to or run this one without --resume"
        )
    steps = payload.get("steps", [])
    return {entry["benchmark"]: entry for entry in steps}


def _write_ledger(
    sweep_root: Path,
    plan: SuitePlan,
    outcomes: Iterable[StepOutcome],
) -> None:
    """Write the ledger after every step, so an interrupted sweep keeps it."""

    run = SuiteRun(plan=plan, sweep_root=sweep_root, outcomes=tuple(outcomes))
    payload = {
        "ledger_version": SWEEP_LEDGER_VERSION,
        "updated_at": datetime.now(tz=UTC).isoformat(),
        **run.as_record(),
    }
    path = sweep_root / LEDGER_FILE_NAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _ledger_outcome(name: str, entry: Mapping[str, Any]) -> StepOutcome:
    """Return a completed step read back from a resumed sweep's ledger."""

    return StepOutcome(
        name=name,
        status=StepStatus.COMPLETED,
        seconds=float(entry.get("seconds", 0.0)),
        results=int(entry.get("results", 0)),
        measurements=int(entry.get("measurements", 0)),
        recorded_paths=tuple(Path(item) for item in entry.get("recorded_paths", ())),
        detail_paths=tuple(Path(item) for item in entry.get("detail_paths", ())),
        note="already read by an earlier run of this sweep",
        from_ledger=True,
    )
