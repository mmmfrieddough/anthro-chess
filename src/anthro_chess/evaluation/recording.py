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

Only the tail. A rigid interface over the measuring itself would be worse than
the duplication it removed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import torch

from anthro_chess.config import ResolvedConfig
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
    NoiseCharacterization,
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
    )


def runner_device(runner: object) -> torch.device:
    """Return the device a runner executes on, defaulting to the CPU.

    A stand-in runner need not carry a device. The environment half of a record
    is attribution rather than identity, so a missing device is recorded as the
    CPU rather than failing the measurement.
    """

    device = getattr(runner, "device", None)
    return device if isinstance(device, torch.device) else torch.device("cpu")


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
    record = selection.as_record()
    return dataset_reference(
        pool_id=str(identity["id"]),
        pool_version=int(identity["version"]),
        view=selection.name,
        selected_games=selection.selected_games,
        game_ids_sha256=str(record["game_ids_sha256"]),
        components=[component],
    )


class ResultRecorder:
    """One benchmark's recording tail: detail payloads, envelopes, and appends.

    Used as a context manager, because the error conversion it owns has to
    cover the whole block rather than only the calls made into it: a benchmark
    builds its measurements there too, and
    :func:`~anthro_chess.evaluation.results.measurement` raises the same errors
    the store does. A sweep converts only the error types a benchmark's registry
    entry declares, so one that escapes ends the sweep instead of failing one
    step.

    One unit of a reading is one :meth:`add`. The bulk payload and the
    committed envelope are written together there because both are scoped by
    ``kind``, which decides the series the envelope joins *and* the directory
    the payload lands in; naming it twice would let the two drift apart, which
    is the failure this module exists to end rather than to relocate.

    The payload is a callable rather than a value because the store is often
    absent — ``--no-record`` is how a shakedown reading is taken — and some of
    these payloads cost real work to assemble.
    """

    def __init__(
        self,
        resolved_config: ResolvedConfig[Any],
        *,
        kind: str,
        benchmark: BenchmarkReference,
        checkpoint: CheckpointReference,
        store: ResultsStore | None,
        detail: DetailStore | None,
        error: type[Exception],
    ) -> None:
        self.recorded_at = datetime.now(tz=UTC)
        self._stamp = self.recorded_at.strftime("%Y%m%dT%H%M%SZ")
        self._kind = kind
        self._benchmark = benchmark
        self._checkpoint = checkpoint
        self._store = store
        self._detail = detail
        self._error = error
        self._configuration = configuration_reference(
            resolved_config.as_record(),
            source=resolved_config.provenance.source,
            overrides=resolved_config.provenance.overrides,
        )
        self._envelopes: list[ResultEnvelope] = []
        self._characterizations: list[NoiseCharacterization] = []
        self._detail_paths: list[Path] = []

    def __enter__(self) -> ResultRecorder:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if isinstance(error, ResultRecordError | ResultsStoreError):
            raise self._error(str(error)) from error

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
        self._envelopes.append(
            build_result(
                kind=series,
                benchmark=benchmark or self._benchmark,
                checkpoint=self._checkpoint,
                configuration=self._configuration,
                data=data,
                execution=execution,
                measurements=measurements,
                detail=reference,
                recorded_at=self.recorded_at,
            )
        )

    def _write(
        self,
        kind: str,
        payload: Callable[[], Mapping[str, Any]],
        *,
        description: str,
        slug: str | None,
    ) -> DetailReference | None:
        if self._detail is None:
            return None
        name = f"{self._stamp}.json" if slug is None else f"{self._stamp}-{slug}.json"
        relative = Path(kind) / self._checkpoint.label / name
        reference = self._detail.write(relative, payload(), description=description)
        self._detail_paths.append(self._detail.root / relative)
        return reference

    def characterize(self, characterization: NoiseCharacterization | None) -> None:
        """Keep one noise floor to append beside the envelopes, if there is one."""

        if characterization is not None:
            self._characterizations.append(characterization)

    def commit(self) -> dict[str, Any]:
        """Append everything recorded, and return the fields it wrote.

        Fields to splat into :func:`dataclasses.replace` rather than a record of
        its own, which would exist only to be unpacked at each call site. The
        cost is that the three keys are unchecked against the seven result
        classes, so a renamed field there surfaces as a ``TypeError`` from
        ``replace`` rather than as a type error.
        """

        recorded: list[Path] = []
        if self._store is not None:
            recorded = [self._store.append(item) for item in self._envelopes]
            recorded.extend(
                self._store.append_characterization(item)
                for item in self._characterizations
            )
        return {
            "envelopes": tuple(self._envelopes),
            "recorded_paths": tuple(recorded),
            "detail_paths": tuple(self._detail_paths),
        }


__all__ = [
    "ResultRecorder",
    "checkpoint_reference",
    "pool_dataset_reference",
    "resolve_model",
    "runner_device",
]
