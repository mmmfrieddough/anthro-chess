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
  runs: an unreadable selection or a rollout that was configured to discard
  its games fails in the first second rather than the first hour.
- **Recording.** Whether a reading is committed is decided per benchmark
  within one sweep, so a sweep can commit the readings it meant to keep and
  leave the rest as evidence about the instrument.
- **Resume.** Each step's outcome is written to a machine-local ledger as it
  finishes, so a sweep that dies late keeps what it already read.

Every step runs at the one size its own selection declares. A step is read
smaller through ``benchmarks.<step>.overrides``, which reaches that step's own
schema; the sweep holds no size of its own to shrink.
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

from anthro_chess.config import ConfigError, ConfigModel, ResolvedConfig
from anthro_chess.evaluation.benchmarks import (
    Benchmark,
    benchmark_registry,
    resolve_benchmark,
    run_benchmark,
)
from anthro_chess.evaluation.selection import CheckpointSelection
from anthro_chess.inference.config import LATEST_CHECKPOINT

if TYPE_CHECKING:
    from anthro_chess.evaluation.results import DetailStore, ResultsStore

#: Bumped when a record written by an older build no longer describes the same
#: shape.
SUITE_VERSION = 2
#: Bumped when a ledger written by an older build can no longer be resumed.
SWEEP_LEDGER_VERSION = 2
LEDGER_FILE_NAME = "sweep.json"
#: Where a step's generated games are handed to the step that reads them. It is
#: sweep state rather than a benchmark artifact: the committed tier never holds
#: games, and a sweep that records nothing still has to feed its own dependent
#: steps.
GAMES_FILE_SUFFIX = "-games.json"

logger = logging.getLogger(__name__)


class SuiteError(ValueError):
    """Raised when a sweep cannot be planned or carried out as configured."""


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
    the order and whether this sweep commits its reading.
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
    #: Dotted TOML overrides applied to the selection.
    overrides: tuple[str, ...] = ()


class SuiteConfig(CheckpointSelection):
    """Code-owned schema for ``anthro eval suite``."""

    name: str = Field(
        default="checkpoint", min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$"
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
        """Return the steps that raised, and those a refused hand-off starved."""

        return tuple(item for item in self.outcomes if item.status is StepStatus.FAILED)

    @property
    def seconds(self) -> float:
        """Return what the sweep cost, excluding steps taken from the ledger."""

        return sum(item.seconds for item in self.outcomes if not item.from_ledger)

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable sweep summary.

        Each step carries the overrides it was resolved under, because the sizes
        live in the selections rather than in this file and a capped reading
        would otherwise be indistinguishable here from a declared one.
        """

        sized_by = {step.name: list(step.overrides) for step in self.plan.steps}
        return {
            "version": SUITE_VERSION,
            "suite": self.plan.suite,
            "checkpoint_label": self.plan.checkpoint_label,
            "plan_sha256": self.plan.sha256,
            "sweep_root": str(self.sweep_root),
            "seconds": round(self.seconds, 3),
            "steps": [
                {**item.as_record(), "overrides": sized_by.get(item.name, [])}
                for item in self.outcomes
            ],
        }


def resolve_suite(
    resolved_config: ResolvedConfig[SuiteConfig],
    *,
    no_record: bool = False,
    record_only: Sequence[str] | None = None,
    registry: Mapping[str, Benchmark] | None = None,
) -> SuitePlan:
    """Resolve every benchmark's configuration and the order they run in.

    Everything the configuration alone decides fails here: an unknown
    benchmark, a missing or invalid selection file, a cycle, a dependency on a
    step that is not in the sweep, and a step configured to discard the games
    another step needs. A sweep measured in hours should not discover any of
    those at the end of it.

    What a step will read is not among them, and deliberately: an artifact that
    is not on this host is found by the step that reads it rather than by the
    plan, per
    ``docs/decisions/0055-a-missing-artifact-fails-its-step-not-the-plan.md``.
    """

    known = dict(registry) if registry is not None else benchmark_registry()
    config = resolved_config.value
    entries = {
        name: entry for name, entry in config.benchmarks.items() if entry.enabled
    }
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
        resolved = _resolve_benchmark(name, entry, benchmark, config)
        steps.append(
            PlannedBenchmark(
                name=name,
                benchmark=benchmark,
                resolved=resolved,
                record=record,
                source=entry.config,
                overrides=(
                    *entry.overrides,
                    *_checkpoint_overrides(config),
                ),
            )
        )
    _validate_game_dependencies(steps, entries)

    return SuitePlan(
        suite=config.name,
        steps=tuple(steps),
        checkpoint_label=config.checkpoint_label,
        sha256=_plan_digest(config, steps),
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

    A sweep root that refuses a payload fails the dependent step rather than
    the producer, whose reading stands and stays in the ledger. The steps that
    generate games are the expensive ones, and a resume should not have to run
    one again to retry the cheap step that reads its output.
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
    starved: dict[str, StepOutcome] = {}

    for step in plan.steps:
        if step.name in completed:
            outcome = _ledger_outcome(step.name, ledger[step.name])
        elif step.name in starved:
            outcome = starved[step.name]
        else:
            outcome = _run_step(
                step,
                sweep_root=sweep_root,
                run_root=run_root,
                store=store if step.record else None,
                detail=detail if step.record else None,
            )
        outcomes.append(outcome)
        if observer is not None:
            observer(outcome)
        # Before the hand-off, so a refused payload cannot cost the producer
        # its ledger entry.
        _write_ledger(sweep_root, plan, outcomes)
        if outcome.status is StepStatus.COMPLETED:
            starving = _hand_off_games(step, outcome, plan, sweep_root)
            # A refused write is atomic, so a payload from an earlier run of
            # this same sweep root survives it; the dependent has to be stopped
            # here rather than left to read those games as this run's. Failed
            # rather than skipped, because the producer's reading stands, so
            # nothing else carries the failure and the sweep would exit
            # reporting success.
            blocked = StepStatus.FAILED
        else:
            starving = f"{step.name} did not complete, so there is nothing to read"
            blocked = StepStatus.SKIPPED
        if starving is not None:
            for dependent in _dependents(plan, step.name):
                starved.setdefault(
                    dependent,
                    StepOutcome(
                        name=dependent,
                        status=blocked,
                        seconds=0.0,
                        note=starving,
                    ),
                )

    return SuiteRun(plan=plan, sweep_root=sweep_root, outcomes=tuple(outcomes))


def sweep_directory(root: Path, plan: SuitePlan) -> Path:
    """Return where one sweep keeps its ledger and its handed-off payloads.

    Derived from the plan rather than from a timestamp, so re-running the same
    command with ``--resume`` finds the sweep it is continuing, and two
    checkpoints swept side by side never share a directory.

    A ledger this name cannot reach is invisible rather than refused, so
    changing what goes into the name strands earlier sweeps and a resume starts
    over. ``SWEEP_LEDGER_VERSION`` records that the older ones are gone;
    nothing reports it at the moment of resuming.
    """

    return root / f"{plan.suite}-{plan.sha256[:12]}"


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
            assert step.resolved is not None  # a step with a schema resolved one
            result = run_benchmark(
                step.benchmark,
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
    envelopes, recorded, detail_paths = _step_reading(step, result)
    logger.info("Suite step %s: completed in %.1fs", step.name, seconds)
    return StepOutcome(
        name=step.name,
        status=StepStatus.COMPLETED,
        seconds=seconds,
        results=len(envelopes),
        measurements=sum(len(envelope.measurements) for envelope in envelopes),
        recorded_paths=recorded,
        detail_paths=detail_paths,
        result=result,
    )


def _hand_off_games(
    step: PlannedBenchmark,
    outcome: StepOutcome,
    plan: SuitePlan,
    sweep_root: Path,
) -> str | None:
    """Write a completed step's games where the step that reads them looks.

    A payload rather than an in-memory hand-off, because resume has to work
    across processes: a sweep whose rollout succeeded and whose decomposition
    failed should not replay the rollout to try the decomposition again.

    Returns why the dependent steps have nothing to read, or ``None`` when
    they do.
    """

    extract = step.benchmark.games
    if extract is None or outcome.result is None or not _dependents(plan, step.name):
        return None

    from anthro_chess.evaluation.games import write_game_records
    from anthro_chess.evaluation.results import DetailStore, ResultsStoreError

    records = extract(outcome.result)
    if not records:
        return None
    path = _games_payload(sweep_root, step.name)
    try:
        write_game_records(
            DetailStore(sweep_root),
            path.name,
            records,
            description=f"Games generated by the {step.name} step of one sweep",
        )
    except ResultsStoreError as error:
        logger.warning("Suite step %s: could not hand off its games", step.name)
        return f"{step.name} could not hand off its games: {error}"
    logger.info("Suite step %s: handed off %d game(s)", step.name, len(records))
    return None


def _games_payload(sweep_root: Path, producer: str) -> Path:
    return sweep_root / f"{producer}{GAMES_FILE_SUFFIX}"


def _dependents(plan: SuitePlan, producer: str) -> tuple[str, ...]:
    """Return the steps that read one step's games."""

    return tuple(
        step.name for step in plan.steps if step.benchmark.games_from == producer
    )


def _step_reading(
    step: PlannedBenchmark,
    result: Any,
) -> tuple[tuple[Any, ...], tuple[Path, ...], tuple[Path, ...]]:
    """Return one result's envelopes and the paths it wrote.

    Driven by the registry's ``records_results`` declaration rather than by
    inspecting the result: a benchmark that records carries all three fields,
    and one that does not — decision decomposition, which has no result kind —
    carries none of them. Reading them defensively is what let the puzzle
    result drift to a different shape without a single test failing.
    """

    if not step.benchmark.records_results:
        return (), (), ()
    return (
        tuple(result.envelopes),
        tuple(result.recorded_paths),
        tuple(result.detail_paths),
    )


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
) -> ResolvedConfig[Any] | None:
    """Load one benchmark's own selection and point it at this checkpoint."""

    if benchmark.schema is None:
        if entry.config is not None or entry.overrides:
            raise SuiteError(
                f"the {name} step reads another step's output, so it takes no "
                "configuration file and no overrides of its own"
            )
        return None
    if entry.config is None:
        raise SuiteError(f"the {name} step needs config to name its selection")

    try:
        resolved = resolve_benchmark(
            benchmark,
            path=entry.config,
            overrides=entry.overrides,
        )
    except ConfigError as error:
        raise SuiteError(f"the {name} step: {error}") from error
    return _with_checkpoint(resolved, suite)


def _with_checkpoint(
    resolved: ResolvedConfig[CheckpointSelection],
    suite: SuiteConfig,
) -> ResolvedConfig[CheckpointSelection]:
    """Replace a benchmark's model selection with the sweep's checkpoint.

    Replaced rather than merged: a suite takes one checkpoint, and a merge
    would let a benchmark file that pinned its own run quietly measure
    something else than the rest of the sweep did.

    Both sides are a :class:`CheckpointSelection`, which is what makes the
    update checkable: ``model_copy`` does not validate, so a schema naming its
    selection anything else would take this as a stray attribute and go on
    measuring its own checkpoint.
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
    steps: Sequence[PlannedBenchmark],
) -> str:
    """Digest what a resumed sweep must still be doing to be the same sweep."""

    payload = {
        "version": SUITE_VERSION,
        "suite": suite.name,
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
