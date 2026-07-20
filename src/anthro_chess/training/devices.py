"""Shared training-device resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DeviceSelection = Literal["auto", "cpu", "mps"]


class DeviceError(ValueError):
    """Raised when a requested training device cannot be resolved."""


@dataclass(frozen=True)
class DeviceCapabilities:
    """Hardware capabilities relevant to training-device selection."""

    mps_built: bool
    mps_available: bool

    def __post_init__(self) -> None:
        if self.mps_available and not self.mps_built:
            raise ValueError("available MPS requires an MPS-enabled Torch build")


def detect_device_capabilities() -> DeviceCapabilities:
    """Read the current process's training-device capabilities from Torch."""
    return DeviceCapabilities(
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
    )


def resolve_training_device(
    requested: DeviceSelection,
    *,
    capabilities: DeviceCapabilities | None = None,
) -> torch.device:
    """Resolve automatic or explicit CPU/MPS selection without silent fallback."""
    observed = (
        capabilities if capabilities is not None else detect_device_capabilities()
    )
    if requested == "auto":
        backend = "mps" if observed.mps_available else "cpu"
    elif requested == "mps":
        if not observed.mps_available:
            availability = (
                "the installed Torch build has no MPS support"
                if not observed.mps_built
                else "MPS is not available on this host"
            )
            raise DeviceError(f"training device mps was requested but {availability}")
        backend = "mps"
    elif requested == "cpu":
        backend = "cpu"
    else:  # pragma: no cover - strict configuration and typing prevent this
        raise DeviceError(f"unknown training device selection: {requested}")
    return torch.device(backend)
