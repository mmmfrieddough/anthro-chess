"""Where a measurement says it ran, checked against the device it ran on.

The accelerator branches here are the ones a CPU-only run cannot reach, and
they are exactly the ones a benchmark result's provenance depends on: a
reading that named the host CPU while running on a GPU would compare cleanly
against readings from a machine it has nothing in common with.
"""

from __future__ import annotations

import platform
import time

import pytest
import torch

from anthro_chess.evaluation.execution import (
    MEASURED_PRECISION,
    device_name,
    execution_record,
    platform_key,
    synchronize,
)

from accelerators import accelerator_parameters

WORKLOAD = {"kind": "fixture", "size": 1}


def test_a_cpu_measurement_names_the_processor_and_its_thread_count() -> None:
    record = execution_record(torch.device("cpu"), WORKLOAD)

    assert record.device == "cpu"
    assert record.device_name == (platform.processor() or platform.machine())
    assert record.cpu_threads == torch.get_num_threads()
    assert record.precision == MEASURED_PRECISION
    assert record.platform_key == platform_key()


def test_the_platform_key_stays_coarser_than_the_full_platform_string() -> None:
    """A point release must not read as an environment change."""

    assert platform_key() in (f"{platform.system()}-{platform.machine()}",)
    assert platform_key() != platform.platform()


@pytest.mark.gpu
@pytest.mark.parametrize("accelerator", accelerator_parameters())
def test_a_measurement_names_the_accelerator_rather_than_its_host(
    accelerator: str,
) -> None:
    device = torch.device(accelerator)

    record = execution_record(device, WORKLOAD)

    assert record.device == accelerator
    assert record.device_name
    assert record.device_name != (platform.processor() or platform.machine())
    assert record.cpu_threads is None
    if accelerator == "cuda":
        assert record.device_name == torch.cuda.get_device_name(device)
    assert device_name(device) == record.device_name


@pytest.mark.gpu
@pytest.mark.parametrize("accelerator", accelerator_parameters())
def test_synchronizing_waits_for_queued_work_a_bare_enqueue_leaves_running(
    accelerator: str,
) -> None:
    """Time the same device work with and without waiting for it.

    Accelerator calls return once the work is queued, so a window that does
    not synchronize times the enqueue and leaves the real cost to whichever
    window blocks next. The gap between these two measurements is the whole
    reason this boundary exists.
    """

    device = torch.device(accelerator)
    left = torch.randn(1024, 1024, device=device)
    right = torch.randn(1024, 1024, device=device)

    def queue_work() -> torch.Tensor:
        product = left
        for _ in range(50):
            product = product @ right
        return product

    synchronize(device)
    queue_work()
    synchronize(device)

    enqueue_started = time.perf_counter()
    queue_work()
    enqueue_seconds = time.perf_counter() - enqueue_started

    waited_started = time.perf_counter()
    queue_work()
    synchronize(device)
    waited_seconds = time.perf_counter() - waited_started

    assert waited_seconds > enqueue_seconds
