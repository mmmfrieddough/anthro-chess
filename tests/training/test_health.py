from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.nn.utils import clip_grads_with_norm_

from anthro_chess.training.health import StepHealth, StepHealthMonitor

#: Far above anything these fixtures produce, so a test that is not about
#: clipping measures the gradients it would have measured without a ceiling.
_LOOSE_CEILING = 1e6


def _module() -> nn.Module:
    torch.manual_seed(11)
    return nn.Linear(3, 2)


def test_gradient_norm_matches_the_gradients_the_step_already_computed() -> None:
    module = _module()
    monitor = StepHealthMonitor(module.parameters(), clip_norm=_LOOSE_CEILING)
    module(torch.ones(1, 3)).sum().backward()

    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    expected = math.sqrt(
        sum(float(gradient.pow(2).sum().item()) for gradient in gradients)
    )
    monitor.observe_gradients()
    health = monitor.drain(global_step=1)

    assert health is not None
    assert health.gradient_norm == pytest.approx(expected)
    assert health.gradient_norm_interval_maximum == pytest.approx(expected)
    assert health.steps == 1
    assert health.update_to_weight_ratio is None


def test_the_interval_maximum_survives_a_spike_between_logging_points() -> None:
    module = _module()
    monitor = StepHealthMonitor(module.parameters(), clip_norm=_LOOSE_CEILING)

    for scale in (1.0, 50.0, 2.0):
        module.zero_grad(set_to_none=True)
        (module(torch.full((1, 3), scale)).sum() * scale).backward()
        monitor.observe_gradients()
    health = monitor.drain(global_step=3)

    assert health is not None
    assert health.steps == 3
    assert health.gradient_norm_interval_maximum > health.gradient_norm


def test_update_to_weight_ratio_measures_the_realized_optimizer_update() -> None:
    module = _module()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.5)
    monitor = StepHealthMonitor(module.parameters(), clip_norm=_LOOSE_CEILING)

    module(torch.ones(1, 3)).sum().backward()
    monitor.observe_gradients()
    monitor.snapshot_parameters()
    before = [parameter.detach().clone() for parameter in module.parameters()]
    optimizer.step()
    monitor.observe_update()

    update = math.sqrt(
        sum(
            float((parameter.detach() - previous).pow(2).sum().item())
            for parameter, previous in zip(module.parameters(), before, strict=True)
        )
    )
    weights = math.sqrt(
        sum(
            float(parameter.detach().pow(2).sum().item())
            for parameter in module.parameters()
        )
    )
    health = monitor.drain(global_step=1)

    assert health is not None
    assert health.update_to_weight_ratio == pytest.approx(update / weights)


def test_observation_defers_every_device_synchronization_to_the_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cheap telemetry costs synchronization, so none happens per step."""

    module = _module()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.5)
    monitor = StepHealthMonitor(module.parameters(), clip_norm=_LOOSE_CEILING)
    synchronizations = 0
    original = torch.Tensor.item

    def counted(self: torch.Tensor) -> float:
        nonlocal synchronizations
        synchronizations += 1
        return original(self)

    monkeypatch.setattr(torch.Tensor, "item", counted)

    for _ in range(3):
        module.zero_grad(set_to_none=True)
        module(torch.ones(1, 3)).sum().backward()
        monitor.observe_gradients()
        monitor.snapshot_parameters()
        optimizer.step()
        monitor.observe_update()
    assert synchronizations == 0

    assert monitor.drain(global_step=3) is not None
    assert synchronizations == 4


def test_a_drain_without_an_observed_step_reports_nothing() -> None:
    monitor = StepHealthMonitor(_module().parameters(), clip_norm=_LOOSE_CEILING)

    assert monitor.drain(global_step=1) is None


def test_draining_resets_the_interval_rather_than_accumulating_forever() -> None:
    module = _module()
    monitor = StepHealthMonitor(module.parameters(), clip_norm=_LOOSE_CEILING)
    module(torch.ones(1, 3)).sum().backward()
    monitor.observe_gradients()
    monitor.drain(global_step=1)

    module.zero_grad(set_to_none=True)
    module(torch.full((1, 3), 3.0)).sum().backward()
    monitor.observe_gradients()
    second = monitor.drain(global_step=2)

    assert second is not None
    assert second.steps == 1


def test_health_reports_only_the_registered_metrics_a_cadence_declares() -> None:
    health = StepHealth(
        steps=4,
        global_step=8,
        gradient_norm=1.5,
        gradient_norm_interval_maximum=2.5,
        update_to_weight_ratio=0.25,
        clip_rate=0.0,
    )

    every = health.measurements()
    declared = health.measurements(["training_health.gradient_norm"])

    assert {item.metric for item in every} == {
        "training_health.clip_rate",
        "training_health.gradient_norm",
        "training_health.update_to_weight_ratio",
    }
    assert [item.metric for item in declared] == ["training_health.gradient_norm"]
    # Training-batch statistics carry no data component, which is what keeps
    # them out of any series a held-out metric belongs to.
    assert all(item.sample_size is None for item in every)
    assert {item.fingerprint for item in every} & {
        item.fingerprint for item in declared
    }


def test_an_unmeasured_ratio_is_absent_rather_than_reported_as_zero() -> None:
    health = StepHealth(
        steps=1,
        global_step=1,
        gradient_norm=1.5,
        gradient_norm_interval_maximum=1.5,
        update_to_weight_ratio=None,
        clip_rate=0.0,
    )

    assert [item.metric for item in health.measurements()] == [
        "training_health.clip_rate",
        "training_health.gradient_norm",
    ]
    assert health.as_record()["update_to_weight_ratio"] is None


def test_the_returned_norm_is_the_one_a_step_clips_against() -> None:
    """The composition the step loop performs: measure once, scale by that."""

    module = _module()
    monitor = StepHealthMonitor(module.parameters(), clip_norm=0.5)
    (module(torch.full((1, 3), 4.0)).sum() * 10.0).backward()

    gradient_norm = monitor.observe_gradients()
    assert gradient_norm is not None
    clip_grads_with_norm_(module.parameters(), 0.5, gradient_norm)
    health = monitor.drain(global_step=1)
    clipped = math.sqrt(
        sum(
            float(parameter.grad.pow(2).sum().item())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
    )

    assert health is not None
    # Reported as measured, before the scaling it caused, or a clipped run
    # could not say how far over the ceiling it had been.
    assert health.gradient_norm > 0.5
    assert clipped == pytest.approx(0.5, rel=1e-4)
    assert health.clip_rate == pytest.approx(1.0)


def test_a_norm_under_the_ceiling_counts_as_unclipped() -> None:
    module = _module()
    monitor = StepHealthMonitor(module.parameters(), clip_norm=_LOOSE_CEILING)
    module(torch.ones(1, 3)).sum().backward()

    monitor.observe_gradients()
    health = monitor.drain(global_step=1)

    assert health is not None
    assert health.clip_rate == 0.0


def test_the_clip_rate_is_the_interval_share_rather_than_the_last_step() -> None:
    module = _module()
    monitor = StepHealthMonitor(module.parameters(), clip_norm=1.0)

    for scale in (0.01, 50.0, 0.01, 0.01):
        module.zero_grad(set_to_none=True)
        (module(torch.full((1, 3), scale)).sum() * scale).backward()
        monitor.observe_gradients()
    health = monitor.drain(global_step=4)

    assert health is not None
    assert health.steps == 4
    assert health.clip_rate == pytest.approx(0.25)


def test_a_monitor_needs_a_trainable_parameter() -> None:
    module = _module()
    for parameter in module.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(ValueError, match="trainable parameter"):
        StepHealthMonitor(module.parameters(), clip_norm=_LOOSE_CEILING)
