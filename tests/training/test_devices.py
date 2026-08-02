from __future__ import annotations

import pytest

from anthro_chess.training.devices import (
    DeviceCapabilities,
    DeviceError,
    resolve_training_device,
)

NOTHING = DeviceCapabilities(
    mps_built=True,
    mps_available=False,
    cuda_built=True,
    cuda_available=False,
)
MPS_AVAILABLE = DeviceCapabilities(
    mps_built=True,
    mps_available=True,
    cuda_built=False,
    cuda_available=False,
)
CUDA_AVAILABLE = DeviceCapabilities(
    mps_built=False,
    mps_available=False,
    cuda_built=True,
    cuda_available=True,
    cuda_device_count=2,
)
NEITHER_BUILT = DeviceCapabilities(
    mps_built=False,
    mps_available=False,
    cuda_built=False,
    cuda_available=False,
)


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [
        (MPS_AVAILABLE, "mps"),
        (CUDA_AVAILABLE, "cuda"),
        (NOTHING, "cpu"),
        (NEITHER_BUILT, "cpu"),
    ],
)
def test_auto_prefers_a_present_accelerator_and_otherwise_falls_back(
    capabilities: DeviceCapabilities,
    expected: str,
) -> None:
    assert resolve_training_device("auto", capabilities=capabilities).type == expected


@pytest.mark.parametrize("requested", ["mps", "cuda"])
def test_an_explicit_accelerator_resolves_when_the_host_has_it(
    requested: str,
) -> None:
    capabilities = MPS_AVAILABLE if requested == "mps" else CUDA_AVAILABLE
    resolved = resolve_training_device(requested, capabilities=capabilities)  # type: ignore[arg-type]
    assert resolved.type == requested


@pytest.mark.parametrize(
    ("requested", "capabilities", "message"),
    [
        ("mps", NOTHING, "MPS is not available on this host"),
        ("mps", NEITHER_BUILT, "Torch build has no MPS support"),
        ("cuda", NOTHING, "CUDA is not available on this host"),
        ("cuda", NEITHER_BUILT, "Torch build has no CUDA support"),
    ],
)
def test_an_explicit_accelerator_never_silently_falls_back(
    requested: str,
    capabilities: DeviceCapabilities,
    message: str,
) -> None:
    with pytest.raises(DeviceError, match=message):
        resolve_training_device(requested, capabilities=capabilities)  # type: ignore[arg-type]


def test_cpu_stays_selectable_beside_a_present_accelerator() -> None:
    assert resolve_training_device("cpu", capabilities=CUDA_AVAILABLE).type == "cpu"


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"mps_built": False, "mps_available": True}, "MPS-enabled"),
        (
            {"cuda_built": False, "cuda_available": True, "cuda_device_count": 1},
            "CUDA-enabled",
        ),
        (
            {"cuda_built": True, "cuda_available": True, "cuda_device_count": 0},
            "at least one device",
        ),
        ({"cuda_device_count": 1}, "unavailable CUDA cannot report devices"),
    ],
)
def test_capabilities_reject_an_impossible_host(
    fields: dict[str, object],
    message: str,
) -> None:
    base: dict[str, object] = {
        "mps_built": True,
        "mps_available": False,
        "cuda_built": True,
        "cuda_available": False,
    }
    with pytest.raises(ValueError, match=message):
        DeviceCapabilities(**{**base, **fields})  # type: ignore[arg-type]
