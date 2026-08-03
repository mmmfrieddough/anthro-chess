"""What one benchmark invocation cost, recorded where its reading is.

Every cost figure this project acted on lived in a comment. The suite resolved
a per-step duration, printed it, wrote it to a machine-local ledger for resume,
and discarded it, so nothing in the repository ever contradicted a comment that
had gone stale — and by the time this module was written every figure in
``configs/evaluation/`` had drifted by an order of magnitude or more. Those
figures decided which benchmarks a reduced sweep could include.

So cost is recorded the way a metric is: one committed envelope per invocation,
assembled by the driver that ran it rather than by the suite. That is what
makes a single ``anthro eval puzzles`` reading carry its cost too, which
matters because the sweep is not how most single-benchmark readings are taken.

The series identity is the whole point and follows
``docs/decisions/0018-workload-scoped-efficiency-series.md``: the declared
workload is identity and the machine is coordinates, so one line runs across
hardware changes and a report attributes a movement rather than crediting it to
the model. What differs from every other workload-scoped metric is *what* the
workload holds. A latency figure keeps sample counts out of its digest, because
measuring more estimates the same quantity more precisely. For cost the
reasoning inverts: measuring more costs more, and the cost is the quantity, so
the workload is a digest of the benchmark's whole configuration.

Two normalizations, and only two: the model selection goes, and every path
drops its machine prefix while keeping the artifact it names.

``docs/decisions/0031-committed-benchmark-cost.md`` owns the reasoning, the
measurements behind it, and why a cost reading needs an execution floor before
a delta in it means much.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel

from anthro_chess.evaluation.execution import execution_record, runner_device
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    ConfigurationReference,
    ResultEnvelope,
    build_result,
    measurement,
)
from anthro_chess.evaluation.results.metrics import BENCHMARK_WALL_CLOCK_SECONDS
from anthro_chess.evaluation.results.records import canonical_json
from anthro_chess.evaluation.roots import artifact_name
from anthro_chess.inference.config import ModelRunnerConfig
from anthro_chess.inference.runner import resolve_inference_device
from anthro_chess.machine import DATA_ROOT_VARIABLE, optional_root

#: Result kind for a committed cost record. Distinct from every benchmark's own
#: kind, because a cost record shares none of their inputs: it reads no
#: projection, has no dataset reference, and would be filtered out by anything
#: reading a benchmark's readings.
BENCHMARK_COST_KIND = "benchmark-cost"


def benchmark_cost_result(
    *,
    benchmark: BenchmarkReference,
    checkpoint: CheckpointReference,
    configuration: ConfigurationReference,
    config: BaseModel,
    device: torch.device,
    seconds: float,
    recorded_at: datetime | None = None,
) -> ResultEnvelope:
    """Return the committed record of what one benchmark invocation cost."""

    execution = execution_record(device, cost_workload(benchmark, config))
    return build_result(
        kind=BENCHMARK_COST_KIND,
        benchmark=benchmark,
        checkpoint=checkpoint,
        configuration=configuration,
        execution=execution,
        measurements=[
            measurement(
                BENCHMARK_WALL_CLOCK_SECONDS.identifier,
                seconds,
                workload=execution.workload_component(),
            )
        ],
        recorded_at=recorded_at,
    )


def cost_device(config: BaseModel, runner: object | None = None) -> torch.device:
    """Return the device a cost reading is attributed to.

    Resolved from the declared selection rather than observed, because the
    driver deliberately does not load the runner: the inference benchmark times
    its own model load, so a pre-loaded one would change what it reports. The
    same public resolution the runner itself uses is called here, so the two
    cannot disagree about a selection they were both given.

    A runner handed to the driver wins, since it is the one that actually ran
    and nothing about the configuration says where a caller loaded it.
    """

    if runner is not None:
        return runner_device(runner)
    selection = _model_selection(config)
    if selection is None:
        return torch.device("cpu")
    return resolve_inference_device(selection.device)


def cost_workload(
    benchmark: BenchmarkReference,
    config: BaseModel,
) -> dict[str, Any]:
    """Return the declared workload one cost reading was taken under.

    The configuration is digested rather than carried in full. Carrying it was
    tried first, because a report labels a series by the workload fields that
    differ between two groups and the full form names the dial that moved. It
    does not survive this family: every benchmark's cost lands in it, their
    schemas share almost no fields, and a label listing every field that
    differs then runs to dozens of lines of mostly absent settings.

    What a reader loses is which setting changed, and the envelope answers that
    another way: ``configuration.source`` and ``configuration.overrides`` name
    the file and the overrides that produced this reading.
    """

    return {
        "benchmark": benchmark.name,
        "benchmark_version": benchmark.version,
        "configuration_sha256": sha256(
            canonical_json(normalized_configuration(config))
        ).hexdigest(),
    }


def normalized_configuration(config: BaseModel) -> dict[str, Any]:
    """Return the configured settings a cost series is scoped by.

    The model selection and the label chosen for it are dropped: the checkpoint
    is the coordinate a cost line varies along, so leaving it in would start a
    new line at every checkpoint. The selection is found by type rather than by
    field name, because a renamed or second one left in the digest would
    fragment every cost series with nothing to notice it.

    Dumped in Python mode rather than the JSON mode every other view of a
    configuration here uses, because JSON mode has already turned each path
    into a rooted string and the machine prefix has to come off while the
    artifact it names stays.
    """

    root = optional_root(DATA_ROOT_VARIABLE)
    dropped = {"checkpoint_label"} | {
        field
        for field in type(config).model_fields
        if isinstance(getattr(config, field), ModelRunnerConfig)
    }
    return {
        field: _workload_value(value, root)
        for field, value in config.model_dump().items()
        if field not in dropped
    }


def _model_selection(config: BaseModel) -> ModelRunnerConfig | None:
    """Return the one checkpoint selection a benchmark declares, if it has one.

    Found by type for the reason :func:`normalized_configuration` drops it by
    type: the field name is a convention, and a benchmark that renamed it would
    otherwise be attributed to the wrong device with nothing to notice.
    """

    for field in type(config).model_fields:
        value = getattr(config, field)
        if isinstance(value, ModelRunnerConfig):
            return value
    return None


def _workload_value(value: Any, root: Path | None) -> Any:
    """Render one configured value as a comparable, machine-independent scalar.

    Key order is left to ``canonical_json``, which sorts at every depth.
    """

    if isinstance(value, Path):
        return artifact_name(value, root)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _workload_value(item, root) for key, item in value.items()}
    if isinstance(value, str | bool | int | float) or value is None:
        return value
    if isinstance(value, Sequence):
        return [_workload_value(item, root) for item in value]
    return str(value)


__all__ = [
    "BENCHMARK_COST_KIND",
    "benchmark_cost_result",
    "cost_device",
    "cost_workload",
    "normalized_configuration",
]
