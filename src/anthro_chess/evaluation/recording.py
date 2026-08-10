"""What every benchmark does once it has finished measuring.

The benchmarks measure heterogeneously and should: the ladder plays games, the
inference benchmark reads no pool at all, decision decomposition consumes
another step's output. What they share is everything *after* the measuring.
Each names its checkpoint the same way, describes its pool the same way, stamps
one recording time, writes bulk payloads to the detail tier under one layout,
assembles envelopes around one configuration reference, appends them to the
store, and converts the store's errors into its own so a sweep reports one
failed step rather than ending.

That tail was written out once per benchmark, and the copies had already
drifted without anything noticing: the puzzle result carried a single envelope
where the others carried tuples and a relative detail path where the others
carried absolute ones, and the suite grew a dual-shape reader rather than the
drift being caught. So the tail lives here instead, written once, and a
cross-cutting addition to what a reading records is one edit rather than seven.

The split within it is between what belongs to the invocation and what belongs
to the reading. :class:`ResultRecording` is the first: the driver opens one per
invocation and holds it across the whole call, so the store, the detail tier,
the configuration, the stamp and the error conversion are decided once.
:class:`ResultRecorder` is the second, and a benchmark opens one as soon as it
knows what it measured — the checkpoint and the series its readings join, which
are the only parts a driver cannot know for it.

What the invocation cost is the first cross-cutting addition this split was
built for, and it lands entirely on the invocation half: the driver times the
call and asks the recording for a cost record, and no benchmark says anything.
See ``docs/decisions/0031-committed-benchmark-cost.md``.

Only the tail. A rigid interface over the measuring itself would be worse than
the duplication it removed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import torch

from anthro_chess.config import ResolvedConfig
from anthro_chess.data.artifacts import game_ids_sha256
from anthro_chess.evaluation.cost import benchmark_cost_result
from anthro_chess.evaluation.pool import FrozenPool
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
    DatasetReference,
    DetailReference,
    DetailStore,
    ExecutionRecord,
    Measurement,
    MetricDispersion,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    ResultsStoreError,
    build_result,
    configuration_reference,
    dataset_reference,
    default_checkpoint_label,
)
from anthro_chess.evaluation.views import ViewSelection
from anthro_chess.inference import CheckpointModelRunner, ModelRunnerConfig
from anthro_chess.inference.runner import ModelRunnerError

if TYPE_CHECKING:
    from anthro_chess.runtime import ActionModelRunner


def checkpoint_reference(
    runner: CheckpointModelRunner,
    *,
    label: str | None,
) -> CheckpointReference:
    """Return the checkpoint identity a reading is recorded against."""

    run_id = runner.selection.run_path.name
    return CheckpointReference(
        label=label or default_checkpoint_label(run_id, runner.global_step),
        step=runner.global_step,
        run_id=run_id,
        parameter_sha256=runner.parameter_sha256(),
        training_sha256=runner.training_sha256,
    )


def resolve_model(
    model: ModelRunnerConfig,
    runner: ActionModelRunner | None,
    checkpoint: CheckpointReference | None,
    *,
    label: str | None,
    run_root: Path | None,
    error: type[Exception],
) -> tuple[ActionModelRunner, CheckpointReference]:
    """Return the runner to measure and the checkpoint identity to record.

    A benchmark handed a runner is being driven by something that loaded one
    already, and then the identity comes with it: nothing about a bare runner
    says which checkpoint it holds.
    """

    if runner is not None:
        if checkpoint is None:
            raise error(
                "an explicitly supplied runner needs a checkpoint reference to "
                "record its results against"
            )
        return runner, checkpoint
    try:
        loaded = CheckpointModelRunner.load(model, run_root=run_root)
    except ModelRunnerError as failure:
        raise error(str(failure)) from failure
    return loaded, checkpoint_reference(loaded, label=label)


def pool_dataset_reference(
    pool: FrozenPool,
    selection: ViewSelection,
    component: DataComponent,
    *,
    error: type[Exception],
) -> DatasetReference:
    """Describe the human games behind one reading from the pool's identity.

    Whether that description is series identity or provenance is the caller's
    to know and say; a frozen pool is named the same way either way.
    """

    identity = pool.manifest.get("pool")
    if not isinstance(identity, Mapping):
        raise error("evaluation pool manifest has no pool identity")
    return dataset_reference(
        pool_id=str(identity["id"]),
        pool_version=int(identity["version"]),
        view=selection.name,
        selected_games=selection.selected_games,
        game_ids_sha256=game_ids_sha256(selection.game_ids),
        components=[component],
    )


class ResultRecording:
    """Everything a benchmark records through that is not the benchmark's.

    Held open across the measuring rather than around the tail, because the
    error conversion has to cover more than the calls made into it: a benchmark
    builds its measurements outside them, and
    :func:`~anthro_chess.evaluation.results.measurement` raises the same errors
    the store does. A sweep converts only the error types a benchmark's registry
    entry declares, so one that escapes ends the sweep instead of failing one
    step.
    """

    #: When this reading was taken. Stamped in :meth:`measuring` rather than at
    #: construction, because the driver opens the recording before the
    #: measuring, and a ladder stamped when its invocation began would claim to
    #: have been recorded hours before it was.
    recorded_at: datetime

    def __init__(
        self,
        resolved_config: ResolvedConfig[Any],
        *,
        store: ResultsStore | None,
        detail: DetailStore | None,
        error: type[Exception],
    ) -> None:
        self.configuration = configuration_reference(
            resolved_config.as_record(),
            source=resolved_config.provenance.source,
            overrides=resolved_config.provenance.overrides,
        )
        self.store = store
        self.detail = detail
        self.envelopes: list[ResultEnvelope] = []
        self.dispersions: dict[str, MetricDispersion] = {}
        self.detail_paths: list[Path] = []
        self._settings = resolved_config.value
        self._error = error
        self._recorded_paths: tuple[Path, ...] = ()

    def measuring(
        self,
        checkpoint: CheckpointReference,
        *,
        kind: str,
        benchmark: BenchmarkReference,
    ) -> ResultRecorder:
        """Return what a reading is added through, once it has an identity.

        Both arguments arrive here rather than at construction: no checkpoint
        is known until the benchmark has loaded one, and the series a reading
        joins is the benchmark's declaration rather than the driver's.

        One reading, one stamp: this is the end of the measuring, which is
        where every benchmark used to take it.
        """

        self.recorded_at = datetime.now(tz=UTC)
        return ResultRecorder(self, checkpoint, kind, benchmark)

    def cost(
        self,
        benchmark: BenchmarkReference,
        *,
        seconds: float,
        device: torch.device,
    ) -> None:
        """Record what the invocation cost, unless it measured nothing at all.

        Called by the driver, which is the only thing that knows when the
        invocation began. A benchmark that produced no envelope produced no
        reading, and the seconds it spent finding that out belong to no series.

        The checkpoint and the environment are read off what was recorded
        rather than kept beside it: every envelope carries both, they are the
        same for all of them, and a second copy is one more thing that can
        disagree with the list it shadows.
        """

        if not self.envelopes:
            return
        reading = self.envelopes[-1]
        self.envelopes.append(
            benchmark_cost_result(
                benchmark=benchmark,
                checkpoint=reading.checkpoint,
                configuration=self.configuration,
                config=self._settings,
                device=device,
                seconds=seconds,
                environment=reading.environment,
                recorded_at=self.recorded_at,
            )
        )

    def __enter__(self) -> ResultRecording:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Append what was recorded, or convert whatever stopped it.

        The append happens here rather than in a method the caller has to
        remember, because forgetting it would write nothing and report
        success. A reading that failed appends nothing at all: a partial
        record of a broken measurement is worse than none.
        """

        try:
            if error is None:
                self._append()
        except (ResultRecordError, ResultsStoreError) as failure:
            raise self._error(str(failure)) from failure
        if isinstance(error, ResultRecordError | ResultsStoreError):
            raise self._error(str(error)) from error

    def _append(self) -> None:
        if self.store is None:
            return
        self._recorded_paths = tuple(self.store.append(item) for item in self.envelopes)

    @property
    def fields(self) -> dict[str, Any]:
        """Return what was written, to splat into :func:`dataclasses.replace`.

        Read after the block, since the appending happens on the way out. A
        record type here would exist only to be unpacked at the one call site
        that assembles a result; the cost of the dict is that the three keys are
        unchecked against the seven result classes, so a renamed field there
        surfaces as a ``TypeError`` from ``replace`` rather than as a type
        error.
        """

        return {
            "envelopes": tuple(self.envelopes),
            "recorded_paths": self._recorded_paths,
            "detail_paths": tuple(self.detail_paths),
        }


@dataclass(frozen=True)
class ResultRecorder:
    """One benchmark's readings, added against the checkpoint it measured.

    One unit of a reading is one :meth:`add`. The bulk payload and the
    committed envelope are written together there because both are scoped by
    ``kind``, which decides the series the envelope joins *and* the directory
    the payload lands in; naming it twice would let the two drift apart, which
    is the failure this module exists to end rather than to relocate.

    The payload is a callable rather than a value because the store is often
    absent — ``--no-record`` is how a shakedown reading is taken — and some of
    these payloads cost real work to assemble.
    """

    _recording: ResultRecording
    _checkpoint: CheckpointReference
    _kind: str
    _benchmark: BenchmarkReference

    def add(
        self,
        measurements: Sequence[Measurement],
        *,
        payload: Callable[[], Mapping[str, Any]],
        description: str,
        slug: str | None = None,
        kind: str | None = None,
        benchmark: BenchmarkReference | None = None,
        data: DatasetReference | None = None,
        execution: ExecutionRecord | None = None,
    ) -> None:
        """Write one unit's bulk payload and record what it measured.

        Measuring nothing writes the payload and no envelope, because the
        committed tier cannot hold a record with no measurement in it. That is
        the store's constraint rather than a policy, and a diagnostic pass that
        came back empty is exactly when the evidence for why is worth keeping.
        A unit with nothing to say at all is skipped by its caller instead.
        """

        series = kind or self._kind
        reference = self._write(series, payload, description=description, slug=slug)
        if not measurements:
            return
        self._recording.envelopes.append(
            build_result(
                kind=series,
                benchmark=benchmark or self._benchmark,
                checkpoint=self._checkpoint,
                configuration=self._recording.configuration,
                data=data,
                execution=execution,
                measurements=[self._dispersed(item) for item in measurements],
                detail=reference,
                recorded_at=self._recording.recorded_at,
            )
        )

    def _dispersed(self, item: Measurement) -> Measurement:
        """Attach the spread estimated for this measurement's own series.

        Matched on the fingerprint rather than the identifier, because that is
        what says the estimate was taken over the inputs this measurement was
        computed from.

        A benchmark that built its measurement with a spread of its own keeps
        it. Only a benchmark whose estimating pass spans several units registers
        one here, so the two never describe the same series, and letting an
        ambient mapping overwrite what a measurement was constructed with would
        be the wrong way round if they ever did.
        """

        dispersion = self._recording.dispersions.get(item.fingerprint)
        if dispersion is None or item.dispersion is not None:
            return item
        return item.model_copy(update={"dispersion": dispersion})

    def _write(
        self,
        kind: str,
        payload: Callable[[], Mapping[str, Any]],
        *,
        description: str,
        slug: str | None,
    ) -> DetailReference | None:
        detail = self._recording.detail
        if detail is None:
            return None
        stamp = self._recording.recorded_at.strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}.json" if slug is None else f"{stamp}-{slug}.json"
        relative = Path(kind) / self._checkpoint.label / name
        reference = detail.write(relative, payload(), description=description)
        self._recording.detail_paths.append(detail.root / relative)
        return reference

    def disperse(self, dispersions: Mapping[str, MetricDispersion]) -> None:
        """Carry the spread of each series onto the measurements added after it.

        Registered on the recorder rather than passed to :meth:`add` because one
        estimating pass covers several units of a reading, and a benchmark that
        had to route the mapping through each of them would be free to route it
        to the wrong one.
        """

        self._recording.dispersions.update(dispersions)


__all__ = [
    "ResultRecorder",
    "ResultRecording",
    "checkpoint_reference",
    "pool_dataset_reference",
    "resolve_model",
]
