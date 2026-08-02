"""What the inference selection resolves to, asked of described hosts.

The interesting cases are hosts this suite is not running on. A CUDA machine
cannot be asked what it would have picked with only MPS present, and a
continuous-integration runner has neither, so the capabilities are supplied
rather than detected.
"""

from __future__ import annotations

import pytest

from anthro_chess.inference import (
    InferenceDeviceCapabilities,
    ModelRunnerError,
    detect_inference_device_capabilities,
    resolve_inference_device,
)

CUDA_HOST = InferenceDeviceCapabilities(
    mps_built=False,
    mps_available=False,
    cuda_built=True,
    cuda_available=True,
)
CUDA_WITHOUT_DEVICE = InferenceDeviceCapabilities(
    mps_built=False,
    mps_available=False,
    cuda_built=True,
    cuda_available=False,
)
CUDA_NOT_BUILT = InferenceDeviceCapabilities(mps_built=False, mps_available=False)
MPS_HOST = InferenceDeviceCapabilities(mps_built=True, mps_available=True)
MPS_WITHOUT_DEVICE = InferenceDeviceCapabilities(mps_built=True, mps_available=False)


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [
        (CUDA_HOST, "cuda"),
        (MPS_HOST, "mps"),
        (CUDA_WITHOUT_DEVICE, "cpu"),
        (MPS_WITHOUT_DEVICE, "cpu"),
    ],
)
def test_auto_takes_the_accelerator_the_host_actually_has(
    capabilities: InferenceDeviceCapabilities,
    expected: str,
) -> None:
    assert resolve_inference_device("auto", capabilities=capabilities).type == expected


def test_auto_on_a_cpu_only_host_is_unchanged_by_the_cuda_selection() -> None:
    """A host with neither accelerator resolves exactly as it did before.

    Pinned because adding a backend to an automatic selection is the change
    most likely to move a machine that was never meant to move.
    """

    cpu_only = InferenceDeviceCapabilities(mps_built=True, mps_available=False)

    assert resolve_inference_device("auto", capabilities=cpu_only).type == "cpu"
    assert resolve_inference_device("cpu", capabilities=cpu_only).type == "cpu"


@pytest.mark.parametrize(
    ("requested", "capabilities", "message"),
    [
        ("cuda", CUDA_WITHOUT_DEVICE, "CUDA is not available on this host"),
        ("cuda", CUDA_NOT_BUILT, "Torch build has no CUDA support"),
        ("mps", MPS_WITHOUT_DEVICE, "MPS is not available on this host"),
        (
            "mps",
            InferenceDeviceCapabilities(mps_built=False, mps_available=False),
            "Torch build has no MPS support",
        ),
    ],
)
def test_an_explicit_accelerator_never_silently_falls_back(
    requested: str,
    capabilities: InferenceDeviceCapabilities,
    message: str,
) -> None:
    with pytest.raises(ModelRunnerError, match=message):
        resolve_inference_device(requested, capabilities=capabilities)  # type: ignore[arg-type]


def test_an_unbuilt_backend_is_distinguished_from_a_missing_device() -> None:
    """The two failures want different fixes, so they read differently.

    A CPU-only wheel is reinstalled; a machine with no GPU is not.
    """

    unbuilt = CUDA_NOT_BUILT.unavailability("cuda")
    absent = CUDA_WITHOUT_DEVICE.unavailability("cuda")

    assert unbuilt != absent
    assert "Torch build" in unbuilt
    assert "this host" in absent


def test_capabilities_reject_an_impossible_accelerator_state() -> None:
    with pytest.raises(ValueError, match="CUDA-enabled"):
        InferenceDeviceCapabilities(
            mps_built=False,
            mps_available=False,
            cuda_built=False,
            cuda_available=True,
        )
    with pytest.raises(ValueError, match="MPS-enabled"):
        InferenceDeviceCapabilities(mps_built=False, mps_available=True)


def test_an_unknown_selection_is_rejected_rather_than_resolved() -> None:
    with pytest.raises(ModelRunnerError, match="unknown model runner device"):
        resolve_inference_device("gpu", capabilities=CUDA_HOST)  # type: ignore[arg-type]


def test_an_unknown_backend_is_not_answered_with_another_backends_state() -> None:
    """Asking about a backend nothing supports fails rather than guessing."""

    with pytest.raises(ValueError, match="unknown accelerator backend"):
        CUDA_HOST.available("rocm")


def test_the_detected_capabilities_describe_this_process() -> None:
    detected = detect_inference_device_capabilities()

    assert not (detected.cuda_available and not detected.cuda_built)
    assert not (detected.mps_available and not detected.mps_built)
    assert resolve_inference_device("cpu", capabilities=detected).type == "cpu"
