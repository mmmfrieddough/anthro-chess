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

Three normalizations, and only three: the model selection goes, every path
drops its machine prefix while keeping the artifact it names, and the pool
generation a selection pins goes because it is realized data identity rather
than work.

``docs/decisions/0031-committed-benchmark-cost.md`` owns the reasoning, the
measurements behind it, and why a cost reading needs an execution floor before
a delta in it means much;
``docs/decisions/0046-a-pinned-pool-generation-is-not-a-cost-workload.md`` owns
the third.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import torch

from anthro_chess.evaluation.execution import execution_record, runner_device
from anthro_chess.evaluation.pool import PoolGenerationPin
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    ConfigurationReference,
    EnvironmentRecord,
    ResultEnvelope,
    build_result,
    measurement,
)
from anthro_chess.evaluation.results.metrics import BENCHMARK_WALL_CLOCK_SECONDS
from anthro_chess.evaluation.results.records import canonical_json
from anthro_chess.evaluation.roots import artifact_name
from anthro_chess.evaluation.selection import CheckpointSelection
from anthro_chess.inference.runner import ModelRunnerError, resolve_inference_device
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
    config: CheckpointSelection,
    device: torch.device,
    seconds: float,
    environment: EnvironmentRecord | None = None,
    recorded_at: datetime | None = None,
) -> ResultEnvelope:
    """Return the committed record of what one benchmark invocation cost.

    The reading's own environment is passed in rather than captured again:
    capturing forks ``git rev-parse`` and re-scans the installed distributions
    for a record that is byte-identical to the one taken microseconds earlier
    in the same process.
    """

    execution = execution_record(device, cost_workload(benchmark, config))
    return build_result(
        kind=BENCHMARK_COST_KIND,
        benchmark=benchmark,
        checkpoint=checkpoint,
        configuration=configuration,
        environment=environment,
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


def cost_device(
    config: CheckpointSelection,
    runner: object | None = None,
) -> torch.device | None:
    """Return the device a cost reading is attributed to, or ``None``.

    Resolved from the declared selection rather than observed, because the
    driver deliberately does not load the runner: the inference benchmark times
    its own model load, so a pre-loaded one would change what it reports. The
    same public resolution the runner itself uses is called here, so the two
    cannot disagree about a selection they were both given.

    A runner handed to the driver wins, since it is the one that actually ran
    and nothing about the configuration says where a caller loaded it.

    ``None`` when the selection names an accelerator this machine cannot offer,
    which the benchmark's own load would already have refused — so reaching it
    means the machine changed under a reading that had by then succeeded. The
    cost record is dropped rather than the reading: this half of a record is
    attribution, and no cost line is better than a wrong one or than a
    completed measurement thrown away over a device label.
    """

    if runner is not None:
        return runner_device(runner)
    try:
        return resolve_inference_device(config.model.device)
    except ModelRunnerError:
        return None


def cost_workload(
    benchmark: BenchmarkReference,
    config: CheckpointSelection,
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
            canonical_json(_normalized_configuration(config))
        ).hexdigest(),
    }


def _normalized_configuration(config: CheckpointSelection) -> dict[str, Any]:
    """Return the configured settings a cost series is scoped by.

    Everything :class:`~anthro_chess.evaluation.selection.CheckpointSelection`
    declares is dropped, which is exactly the checkpoint being measured and the
    label it is recorded under: the checkpoint is the coordinate a cost line
    varies along, so leaving it in would start a new line at every one.

    Dumped in Python mode rather than the JSON mode every other view of a
    configuration here uses, because JSON mode has already turned each path
    into a rooted string and the machine prefix has to come off while the
    artifact it names stays. That is also why the walk below is hand-written:
    it exists to reach every path at any depth, which is exactly what a JSON
    dump destroys.
    """

    root = optional_root(DATA_ROOT_VARIABLE)
    declared = {
        field: value
        for field, value in config.model_dump().items()
        if field not in CheckpointSelection.model_fields
    }
    return cast(dict[str, Any], _workload_value(declared, root))


def _workload_value(value: Any, root: Path | None) -> Any:
    """Render one configured value as a comparable, machine-independent scalar.

    Key order is left to ``canonical_json``, which sorts at every depth.

    A pool generation is dropped wherever it sits: decision 0046 keeps realized
    data identity out of a cost workload.
    """

    if isinstance(value, Path):
        return artifact_name(value, root)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            key: _workload_value(item, root)
            for key, item in value.items()
            if key not in PoolGenerationPin.model_fields
        }
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
]
