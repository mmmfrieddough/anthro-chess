"""Shared training-device resolution."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any, Literal

import torch

DeviceSelection = Literal["auto", "cpu", "mps", "cuda"]

#: Accelerators ``auto`` will choose, in the order it prefers them. No host
#: offers two of these at once, so the order is a tie-break that never fires
#: rather than a ranking of the backends against each other.
_AUTOMATIC_PREFERENCE: tuple[str, ...] = ("cuda", "mps")

#: Backends that can run the strict correctness path, meaning the locked Torch
#: build has a deterministic implementation for every operation this model's
#: backward pass needs. MPS is absent because one required gradient operation
#: has none, so selecting strict there fails before optimization rather than
#: quietly training nondeterministically. This is a property of the build and
#: the model rather than of the hardware, so it is checked by asking Torch to
#: refuse — the membership here only decides how early and how clearly.
STRICT_DETERMINISM_BACKENDS = frozenset({"cpu", "cuda"})


class DeviceError(ValueError):
    """Raised when a requested training device cannot be resolved."""


@dataclass(frozen=True)
class DeviceCapabilities:
    """Hardware capabilities relevant to training-device selection.

    Built and available are kept apart for both backends because they fail for
    different reasons and a caller deserves to be told which. A Torch wheel
    compiled without the backend can never reach the hardware, while a build
    that has it can still land on a host with no device or no driver.
    """

    mps_built: bool
    mps_available: bool
    cuda_built: bool
    cuda_available: bool
    #: Devices the CUDA runtime reports. Recorded because a single-device run
    #: on a multi-device host is a choice rather than the only possibility,
    #: and provenance should be able to say which it was.
    cuda_device_count: int = 0

    def __post_init__(self) -> None:
        if self.mps_available and not self.mps_built:
            raise ValueError("available MPS requires an MPS-enabled Torch build")
        if self.cuda_available and not self.cuda_built:
            raise ValueError("available CUDA requires a CUDA-enabled Torch build")
        if self.cuda_available and self.cuda_device_count < 1:
            raise ValueError("available CUDA requires at least one device")
        if not self.cuda_available and self.cuda_device_count:
            raise ValueError("unavailable CUDA cannot report devices")

    def available(self, backend: str) -> bool:
        """Return whether one accelerator backend is usable on this host."""

        if backend == "mps":
            return self.mps_available
        if backend == "cuda":
            return self.cuda_available
        raise ValueError(f"unknown accelerator backend: {backend}")

    def unavailable_reason(self, backend: str) -> str:
        """Return why one accelerator backend cannot be selected here."""

        if backend == "mps":
            built = self.mps_built
        elif backend == "cuda":
            built = self.cuda_built
        else:
            raise ValueError(f"unknown accelerator backend: {backend}")
        name = backend.upper()
        if not built:
            return f"the installed Torch build has no {name} support"
        return f"{name} is not available on this host"


def detect_device_capabilities() -> DeviceCapabilities:
    """Read the current process's training-device capabilities from Torch."""

    cuda_available = torch.cuda.is_available()
    return DeviceCapabilities(
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
        cuda_built=torch.backends.cuda.is_built(),
        cuda_available=cuda_available,
        cuda_device_count=torch.cuda.device_count() if cuda_available else 0,
    )


def resolve_training_device(
    requested: DeviceSelection,
    *,
    capabilities: DeviceCapabilities | None = None,
) -> torch.device:
    """Resolve automatic or explicit selection without silent fallback.

    Only ``auto`` falls back. An explicit accelerator that is not there is an
    error rather than a slow CPU run, because the whole reason to name one is
    that the run is not worth doing without it.
    """

    observed = (
        capabilities if capabilities is not None else detect_device_capabilities()
    )
    if requested == "auto":
        backend = next(
            (
                candidate
                for candidate in _AUTOMATIC_PREFERENCE
                if observed.available(candidate)
            ),
            "cpu",
        )
    elif requested in _AUTOMATIC_PREFERENCE:
        if not observed.available(requested):
            raise DeviceError(
                f"training device {requested} was requested but "
                f"{observed.unavailable_reason(requested)}"
            )
        backend = requested
    elif requested == "cpu":
        backend = "cpu"
    else:  # pragma: no cover - strict configuration and typing prevent this
        raise DeviceError(f"unknown training device selection: {requested}")
    return torch.device(backend)


def hardware_record(device: torch.device) -> dict[str, Any]:
    """Return what a run should retain about the machine it executed on.

    Kept apart from the execution *selection* on purpose. The selection is a
    compatibility identity that a resumed run has to match, so it can only hold
    values another machine could reproduce. This holds the values that make a
    throughput or memory figure interpretable a year later — which accelerator,
    how many of them, which runtime — and precisely because they differ between
    machines, nothing compares them.
    """

    record: dict[str, Any] = {
        "backend": device.type,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
    }
    if device.type == "cuda":  # pragma: no cover - no CUDA in the CPU suite
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        record["cuda"] = {
            "device_index": index,
            "device_name": properties.name,
            "device_count": torch.cuda.device_count(),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    elif device.type == "cpu":
        record["cpu"] = {
            "processor": platform.processor() or platform.machine(),
            "torch_threads": torch.get_num_threads(),
        }
    return record
