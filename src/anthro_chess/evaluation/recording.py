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

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
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

#: How a detail payload's file name carries the moment the reading was taken.
DETAIL_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


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

    Writing a payload and assembling an envelope are separate calls because a
    benchmark can legitimately do one without the other. A committed record
    needs at least one measurement, so a diagnostic that measured nothing is
    kept in the detail tier alone; and a unit with nothing to say writes no
    payload either. Which of those a benchmark means is not something this can
    infer, so it is not asked to.

    Obtained from :func:`recording`, which owns the error conversion.
    """

    def __init__(
        self,
        resolved_config: ResolvedConfig[Any],
        *,
        kind: str,
        benchmark: BenchmarkReference,
        checkpoint: CheckpointReference,
        detail: DetailStore | None,
    ) -> None:
        self.recorded_at = datetime.now(tz=UTC)
        self._kind = kind
        self._benchmark = benchmark
        self._checkpoint = checkpoint
        self._detail = detail
        self._configuration = configuration_reference(
            resolved_config.as_record(),
            source=resolved_config.provenance.source,
            overrides=resolved_config.provenance.overrides,
        )
        self._envelopes: list[ResultEnvelope] = []
        self._characterizations: list[NoiseCharacterization] = []
        self._detail_paths: list[Path] = []

    def detail(
        self,
        payload: Mapping[str, Any],
        *,
        description: str,
        slug: str | None = None,
        kind: str | None = None,
    ) -> DetailReference | None:
        """Write one bulk payload to the detail tier and return its reference."""

        if self._detail is None:
            return None
        stamp = self.recorded_at.astimezone(UTC).strftime(DETAIL_STAMP_FORMAT)
        name = f"{stamp}.json" if slug is None else f"{stamp}-{slug}.json"
        relative = Path(kind or self._kind) / self._checkpoint.label / name
        reference = self._detail.write(relative, dict(payload), description=description)
        self._detail_paths.append(self._detail.root / relative)
        return reference

    def add(
        self,
        measurements: Sequence[Measurement],
        *,
        detail: DetailReference | None = None,
        kind: str | None = None,
        benchmark: BenchmarkReference | None = None,
        data: DatasetReference | None = None,
        execution: ExecutionRecord | None = None,
    ) -> None:
        """Assemble one result envelope around measurements already taken."""

        self._envelopes.append(
            build_result(
                kind=kind or self._kind,
                benchmark=benchmark or self._benchmark,
                checkpoint=self._checkpoint,
                configuration=self._configuration,
                data=data,
                execution=execution,
                measurements=measurements,
                detail=detail,
                recorded_at=self.recorded_at,
            )
        )

    def characterize(self, characterization: NoiseCharacterization | None) -> None:
        """Keep one noise floor to append beside the envelopes, if there is one."""

        if characterization is not None:
            self._characterizations.append(characterization)

    def commit(self, store: ResultsStore | None) -> dict[str, Any]:
        """Append everything recorded, and return the fields it wrote.

        Returned as fields to splat into :func:`dataclasses.replace` rather than
        as a record of its own: every benchmark result carries the same three,
        and a type here would exist only to be unpacked at each call site.
        """

        recorded: list[Path] = []
        if store is not None:
            recorded = [store.append(envelope) for envelope in self._envelopes]
            recorded.extend(
                store.append_characterization(item) for item in self._characterizations
            )
        return {
            "envelopes": tuple(self._envelopes),
            "recorded_paths": tuple(recorded),
            "detail_paths": tuple(self._detail_paths),
        }


@contextmanager
def recording(
    resolved_config: ResolvedConfig[Any],
    *,
    kind: str,
    benchmark: BenchmarkReference,
    checkpoint: CheckpointReference,
    detail: DetailStore | None,
    error: type[Exception],
) -> Iterator[ResultRecorder]:
    """Record one benchmark's reading, converting store errors into its own.

    The conversion covers the whole block rather than only the calls made into
    the recorder, because a benchmark builds its measurements there too and
    :func:`~anthro_chess.evaluation.results.measurement` raises the same errors.
    A sweep converts only the error types a benchmark's registry entry declares,
    so one that escapes ends the sweep instead of failing one step.
    """

    try:
        yield ResultRecorder(
            resolved_config,
            kind=kind,
            benchmark=benchmark,
            checkpoint=checkpoint,
            detail=detail,
        )
    except (ResultRecordError, ResultsStoreError) as failure:
        raise error(str(failure)) from failure


__all__ = [
    "DETAIL_STAMP_FORMAT",
    "ResultRecorder",
    "checkpoint_reference",
    "pool_dataset_reference",
    "recording",
    "resolve_model",
    "runner_device",
]
