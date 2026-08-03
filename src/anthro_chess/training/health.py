"""Per-step training-health statistics, accumulated on device.

These answer whether a run is going wrong right now. They are computed on
training batches from tensors the optimizer step already produced, so they are
contaminated by construction and are never a quality measurement: they carry
no data component at all and can never share a series with a held-out metric.

The cost of cheap telemetry is device synchronization rather than arithmetic,
so every value stays a device tensor until the existing logging interval drains
it. That is also why the update-to-weight ratio is measured only on steps that
will be reported: a per-step measurement needs the parameters from before the
update, and retaining them every step means a permanent parameter-sized shadow
copy for a statistic that moves slowly. Gradient norm is different — it reads
gradients the backward pass just wrote, so it runs every step and reports both
the reported step's value and the interval's maximum, which is what catches a
spike between two logging points.
"""

from __future__ import annotations

import time
from collections.abc import Collection, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.utils import get_total_norm

from anthro_chess.evaluation.results import Measurement, measurement
from anthro_chess.evaluation.results.metrics import (
    TRAINING_HEALTH_GRADIENT_NORM,
    TRAINING_HEALTH_UPDATE_TO_WEIGHT_RATIO,
)

STEP_HEALTH_VERSION = 1

#: The metrics a health monitor can report. A cadence naming anything else has
#: to compute it from data, which is a different affordability question.
STEP_HEALTH_METRICS: tuple[str, ...] = (
    TRAINING_HEALTH_GRADIENT_NORM.identifier,
    TRAINING_HEALTH_UPDATE_TO_WEIGHT_RATIO.identifier,
)


@dataclass(frozen=True)
class StepHealth:
    """What the optimizer looked like over one logging interval."""

    steps: int
    global_step: int
    gradient_norm: float
    gradient_norm_interval_maximum: float
    update_to_weight_ratio: float | None

    def as_record(self) -> dict[str, object]:
        """Return the metrics-stream record for one drained interval."""

        return {
            "version": STEP_HEALTH_VERSION,
            "steps": self.steps,
            "gradient_norm": self.gradient_norm,
            "gradient_norm_interval_maximum": (self.gradient_norm_interval_maximum),
            "update_to_weight_ratio": self.update_to_weight_ratio,
        }

    def measurements(
        self,
        metrics: Collection[str] | None = None,
    ) -> tuple[Measurement, ...]:
        """Return registered measurements for the reported step.

        These carry a null data component, which is what keeps them
        structurally immune to every change in evaluation inputs.
        """

        wanted = None if metrics is None else frozenset(metrics)
        values: list[Measurement] = []
        pairs: tuple[tuple[str, float | None], ...] = (
            (TRAINING_HEALTH_GRADIENT_NORM.identifier, self.gradient_norm),
            (
                TRAINING_HEALTH_UPDATE_TO_WEIGHT_RATIO.identifier,
                self.update_to_weight_ratio,
            ),
        )
        for identifier, value in pairs:
            if value is None or (wanted is not None and identifier not in wanted):
                continue
            values.append(measurement(identifier, value))
        return tuple(values)


class StepHealthMonitor:
    """Accumulate optimizer statistics on device between logging intervals."""

    def __init__(self, parameters: Iterable[Parameter]) -> None:
        self._parameters = tuple(
            parameter for parameter in parameters if parameter.requires_grad
        )
        if not self._parameters:
            raise ValueError("training health needs at least one trainable parameter")
        self._latest_gradient_norm: Tensor | None = None
        self._maximum_gradient_norm: Tensor | None = None
        self._update_ratio: Tensor | None = None
        self._snapshot: tuple[Tensor, ...] | None = None
        self._steps = 0
        self._instrumentation_seconds = 0.0

    @property
    def instrumentation_seconds(self) -> float:
        """Return the wall-clock time spent measuring, for overhead reporting.

        Instrumentation work is queued on the device like any other operation,
        so this is the host-side cost. It is reported as measured rather than
        as an estimate of total device overhead.
        """

        return self._instrumentation_seconds

    def snapshot_parameters(self) -> None:
        """Retain pre-update parameters so an update norm can be measured."""

        started = time.perf_counter()
        self._snapshot = tuple(
            parameter.detach().clone() for parameter in self._parameters
        )
        self._instrumentation_seconds += time.perf_counter() - started

    def observe_gradients(self) -> None:
        """Record the global gradient norm from the gradients just written."""

        started = time.perf_counter()
        gradients = [
            parameter.grad.detach()
            for parameter in self._parameters
            if parameter.grad is not None
        ]
        if gradients:
            gradient_norm = get_total_norm(gradients)
            self._latest_gradient_norm = gradient_norm
            self._maximum_gradient_norm = (
                gradient_norm
                if self._maximum_gradient_norm is None
                else torch.maximum(self._maximum_gradient_norm, gradient_norm)
            )
        self._steps += 1
        self._instrumentation_seconds += time.perf_counter() - started

    def observe_update(self) -> None:
        """Record the update-to-weight ratio against the retained snapshot."""

        if self._snapshot is None:
            return
        started = time.perf_counter()
        updates = [
            parameter.detach() - previous
            for parameter, previous in zip(
                self._parameters, self._snapshot, strict=True
            )
        ]
        weight_norm = get_total_norm(
            [parameter.detach() for parameter in self._parameters]
        )
        self._update_ratio = get_total_norm(updates) / torch.clamp(
            weight_norm, min=torch.finfo(weight_norm.dtype).tiny
        )
        self._snapshot = None
        self._instrumentation_seconds += time.perf_counter() - started

    def drain(self, global_step: int) -> StepHealth | None:
        """Synchronize once, return the interval's health, and reset it."""

        if self._steps == 0 or self._latest_gradient_norm is None:
            return None
        started = time.perf_counter()
        assert self._maximum_gradient_norm is not None
        health = StepHealth(
            steps=self._steps,
            global_step=global_step,
            gradient_norm=float(self._latest_gradient_norm.item()),
            gradient_norm_interval_maximum=float(self._maximum_gradient_norm.item()),
            update_to_weight_ratio=(
                None if self._update_ratio is None else float(self._update_ratio.item())
            ),
        )
        self._latest_gradient_norm = None
        self._maximum_gradient_norm = None
        self._update_ratio = None
        self._steps = 0
        self._instrumentation_seconds += time.perf_counter() - started
        return health


__all__ = [
    "STEP_HEALTH_METRICS",
    "STEP_HEALTH_VERSION",
    "StepHealth",
    "StepHealthMonitor",
]
