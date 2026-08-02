"""What accelerators this host has, and which of them the project accepts.

A module rather than fixtures in ``conftest`` because a ``skipif`` condition
and a ``parametrize`` argument are both evaluated during collection, before any
fixture exists.

The gap between the two lists is the reason this exists. A host can hold a
working accelerator that no device selection accepts, and there a suite that
merely skips is indistinguishable from one running on a machine with no
accelerator at all. Both report the same count of skipped tests for opposite
reasons, and only one of them makes a green run evidence about the accelerator.

The surface is a value rather than a set of lookups so the sentences derived
from it are pure functions of it, and can be checked on a host that is not the
one they describe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, get_args

import pytest
import torch

from anthro_chess.inference.config import InferenceDevice
from anthro_chess.training.devices import DeviceSelection

#: Selections naming no accelerator: the host fallback, and the request to
#: choose one automatically.
_NON_ACCELERATOR_SELECTIONS = frozenset({"auto", "cpu"})

#: The accelerator backends this project targets, in the order a report lists
#: them. A backend absent from here is invisible to this module, which stays
#: correct only while nothing in the project can select it either.
CANDIDATE_BACKENDS = ("mps", "cuda")


@dataclass(frozen=True)
class AcceleratorSurface:
    """What one host holds, beside what the project can ask that host for."""

    #: Present backends with their device counts, in ``CANDIDATE_BACKENDS``
    #: order. A backend with no usable device is absent rather than zero.
    present: tuple[tuple[str, int], ...]
    selectable_training: tuple[str, ...]
    selectable_inference: tuple[str, ...]

    @property
    def present_backends(self) -> tuple[str, ...]:
        """Return the present backend names without their device counts."""

        return tuple(backend for backend, _ in self.present)

    @property
    def usable_training(self) -> tuple[str, ...]:
        """Return present backends a training device selection also accepts."""

        return tuple(
            backend
            for backend in self.present_backends
            if backend in self.selectable_training
        )

    def describe_present(self) -> str:
        """Return a report-ready phrase for what is present, with counts."""

        if not self.present:
            return "none"
        return ", ".join(
            f"{backend} ({count} device(s))" for backend, count in self.present
        )

    def report_lines(self) -> list[str]:
        """Return the header lines that say what this run could exercise."""

        selectable = sorted(set(self.selectable_training + self.selectable_inference))
        lines = [
            f"accelerators present: {self.describe_present()}",
            "accelerators the device selection accepts: "
            + (", ".join(selectable) or "none"),
        ]
        if self.present and not self.usable_training:
            lines.append(
                "no present accelerator is selectable, so nothing here exercised "
                "a training or inference path on one"
            )
        return lines

    def training_skip_reason(self, backend: str) -> str:
        """Return why a training accelerator test cannot run on this host.

        Naming the host is the point. "Requires Apple silicon" reads the same
        on a machine with no accelerator and on one holding two working GPUs,
        and only the second is a coverage gap somebody should be told about.
        """

        wanted = f"requires an available PyTorch {backend.upper()} backend"
        if not self.present:
            return f"{wanted}; this host has no accelerator the project targets"
        if len(self.usable_training) < len(self.present):
            return (
                f"{wanted}; this host has {self.describe_present()}, which the "
                "training device selection does not accept"
            )
        return f"{wanted}; this host has {self.describe_present()}"


def detect_surface() -> AcceleratorSurface:
    """Read this host's accelerator surface from Torch and the selections."""

    present = tuple(
        (backend, count)
        for backend, count in (
            (name, _device_count(name)) for name in CANDIDATE_BACKENDS
        )
        if count > 0
    )
    return AcceleratorSurface(
        present=present,
        selectable_training=_accelerators_in(get_args(DeviceSelection)),
        selectable_inference=_accelerators_in(get_args(InferenceDevice)),
    )


def accelerator_parameters() -> list[Any]:
    """Return one parameter per present accelerator, or one explained skip.

    An empty parameter list would skip with pytest's generic "got empty
    parameter set", which says nothing about the host. This names what was
    looked for instead.
    """

    if HOST.present:
        return [pytest.param(backend, id=backend) for backend in HOST.present_backends]
    return [
        pytest.param(
            "cpu",
            id="none",
            marks=pytest.mark.skip(
                reason=(
                    "this host has none of the accelerator backends the project "
                    f"targets ({', '.join(CANDIDATE_BACKENDS)})"
                )
            ),
        )
    ]


def requires_training_accelerator(backend: str) -> pytest.MarkDecorator:
    """Return the skip marker for a test driving training on ``backend``.

    The condition and the explanation come from one place deliberately. They
    drifted apart before: a test gated on MPS availability announced that it
    needed Apple silicon, which stayed true and stopped being the reason.
    """

    return pytest.mark.skipif(
        backend not in HOST.usable_training,
        reason=HOST.training_skip_reason(backend),
    )


def _accelerators_in(selections: tuple[Any, ...]) -> tuple[str, ...]:
    named = {str(selection) for selection in selections}
    return tuple(sorted(named - _NON_ACCELERATOR_SELECTIONS))


def _device_count(backend: str) -> int:
    if backend == "mps":
        return 1 if torch.backends.mps.is_available() else 0
    if backend == "cuda":
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    raise ValueError(f"unknown accelerator backend: {backend}")


#: Detected once, because every caller is asking about the same process.
HOST = detect_surface()
