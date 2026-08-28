"""Where a workload-scoped measurement ran, captured one way.

Two benchmark kinds record execution coordinates: efficiency, whose declared
workload says what was timed, and generated play, whose declared workload says
what was played. Both need the same environment half — device, precision, Torch
version, coarse platform key — and only the workload differs.

Capturing that half here rather than per benchmark keeps two results from
describing the same machine differently, which would read in a report as an
environment change that never happened. The workload stays the caller's, since
it is the one part that is genuinely per benchmark.

None of the environment is a fingerprint input. See
``docs/decisions/0018-workload-scoped-efficiency-series.md`` for why, and
``docs/decisions/0020-declared-settings-scope-generated-series.md`` for how the
same split applies to generated play.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping
from typing import Any

import torch

from anthro_chess.evaluation.results import ExecutionRecord, execution_reference

#: Parameter precision the runtime loads. Recorded rather than configured,
#: because the runner rejects a checkpoint that is not float32 today; when a
#: precision dial arrives it becomes an input to this value rather than a new
#: execution coordinate.
MEASURED_PRECISION = "float32"


def execution_record(
    device: torch.device,
    workload: Mapping[str, Any],
    cpu_threads: int | None = None,
) -> ExecutionRecord:
    """Return one measurement's declared workload and where it ran.

    ``cpu_threads`` names the host's intra-op threads for a caller that pinned
    its own. A benchmark spreading work over processes gives each one thread so
    the pool does not oversubscribe the machine, and recording that would make a
    parallel reading incomparable with a serial one over a scheduling choice.
    """

    return execution_reference(
        device=device.type,
        device_name=device_name(device),
        precision=MEASURED_PRECISION,
        torch_version=torch.__version__,
        platform_key=platform_key(),
        platform=platform.platform(),
        cpu_threads=(
            (torch.get_num_threads() if cpu_threads is None else cpu_threads)
            if device.type == "cpu"
            else None
        ),
        workload=dict(workload),
    )


def platform_key() -> str:
    """Return the coarse machine identity an environment comparison keys on.

    Deliberately not the full platform string. That carries the OS patch
    level, so keying on it would mark every delta as confounded after a point
    release that changed no hardware.
    """

    return f"{platform.system() or 'unknown'}-{platform.machine() or 'unknown'}"


def runner_device(runner: object) -> torch.device:
    """Return the device a runner executes on, defaulting to the CPU.

    A stand-in runner need not carry a device. This half of a record is
    attribution rather than identity, so a missing device is recorded as the
    CPU rather than failing the measurement.
    """

    device = getattr(runner, "device", None)
    return device if isinstance(device, torch.device) else torch.device("cpu")


def device_name(device: torch.device) -> str:
    """Return a stable name for the device a measurement ran on."""

    if device.type == "cuda":  # pragma: no cover - exercised on a CUDA host
        return torch.cuda.get_device_name(device)
    machine = platform.machine() or "unknown"
    processor = platform.processor() or machine
    return processor if device.type == "cpu" else f"{machine}-{device.type}"


def synchronize(device: torch.device) -> None:
    """Wait for queued device work, so a timer measures it rather than a queue.

    Accelerator calls are asynchronous. Without this, every measured window
    would time the enqueue and attribute the real work to whichever window
    happened to block next.
    """

    if device.type == "cuda":  # pragma: no cover - exercised on a CUDA host
        torch.cuda.synchronize(device)
    elif device.type == "mps":  # pragma: no cover - no MPS in the CPU suite
        torch.mps.synchronize()
