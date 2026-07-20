from __future__ import annotations

import pytest

from anthro_chess.training.devices import (
    DeviceCapabilities,
    DeviceError,
    resolve_training_device,
)

MPS_AVAILABLE = DeviceCapabilities(mps_built=True, mps_available=True)
MPS_UNAVAILABLE = DeviceCapabilities(mps_built=True, mps_available=False)
MPS_NOT_BUILT = DeviceCapabilities(mps_built=False, mps_available=False)


def test_auto_prefers_mps_when_available() -> None:
    assert resolve_training_device("auto", capabilities=MPS_AVAILABLE).type == "mps"


def test_auto_falls_back_to_cpu_when_mps_is_unavailable() -> None:
    assert resolve_training_device("auto", capabilities=MPS_UNAVAILABLE).type == "cpu"


@pytest.mark.parametrize(
    ("capabilities", "message"),
    [
        (MPS_UNAVAILABLE, "MPS is not available"),
        (MPS_NOT_BUILT, "Torch build has no MPS support"),
    ],
)
def test_explicit_mps_never_silently_falls_back(
    capabilities: DeviceCapabilities,
    message: str,
) -> None:
    with pytest.raises(DeviceError, match=message):
        resolve_training_device("mps", capabilities=capabilities)


def test_capabilities_reject_impossible_mps_state() -> None:
    with pytest.raises(ValueError, match="MPS-enabled"):
        DeviceCapabilities(mps_built=False, mps_available=True)
