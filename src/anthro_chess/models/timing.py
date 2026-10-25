"""The move-time distribution the time head writes, and how it is scored.

A mixture of logistics over ``log(1 + milliseconds)``. Human move times are
multimodal, so one decision can put mass on an instant premove and on a long
think at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

MOVE_TIME_COMPONENTS = 8
#: The corpus clocks' resolution. A recorded time stands for every time that
#: rounds to it, so it is scored as the mass of that interval rather than as a
#: density, which a run of exact zeros would otherwise drive without bound.
MOVE_TIME_RESOLUTION_MS = 10
#: Where the components' locations start, in ``log(1 + ms)``: spread from an
#: instant move to ten minutes, the range human move times occupy.
INITIAL_LOCATION_RANGE = (0.0, 13.3)
#: Bounds on each component's log-scale. The floor stops a component
#: collapsing onto one recorded time. Above the ceiling a component is wider
#: than the whole range of move times, and its inverse underflows to zero.
_LOG_SCALE_RANGE = (-7.0, 7.0)
#: A day, so a sample from a component's far tail stays finite in float32. No
#: timed game's clock holds that much.
_LONGEST_LOGGED_MS = 18.27


@dataclass(frozen=True, eq=False)
class MoveTimeDistribution:
    """One mixture per decision, each field shaped like the decisions by mixture."""

    logits: Tensor
    locations: Tensor
    log_scales: Tensor

    @classmethod
    def from_parameters(cls, parameters: Tensor) -> MoveTimeDistribution:
        """Read the head's raw output, in float32 whatever it was computed in."""

        logits, locations, log_scales = (
            parameters.float().unflatten(-1, (3, MOVE_TIME_COMPONENTS)).unbind(-2)
        )
        return cls(logits, locations, log_scales.clamp(*_LOG_SCALE_RANGE))

    def log_prob(self, time_ms: Tensor) -> Tensor:
        """Return the log-mass of the clock interval each recorded time stands for.

        The interval nearest zero reaches down to minus infinity, which also
        holds the mass of every time below its own.
        """

        half = MOVE_TIME_RESOLUTION_MS / 2
        time_ms = time_ms.float().clamp(min=0.0)
        bottom = (time_ms < half).unsqueeze(-1)
        # Carried as the interval's width rather than as two endpoints, because
        # a long think's interval is narrower than float32 resolves at its
        # endpoints, and the difference of two rounded endpoints can be zero.
        width = torch.log1p(
            MOVE_TIME_RESOLUTION_MS / (time_ms - half + 1.0).clamp(min=1.0)
        )
        inverse_scale = torch.exp(-self.log_scales)
        upper = (
            torch.log1p(time_ms + half).unsqueeze(-1) - self.locations
        ) * inverse_scale
        spread = width.unsqueeze(-1) * inverse_scale
        lower = upper - spread
        # log(sigmoid(upper) - sigmoid(lower)), which loses nothing however
        # close the two are.
        interval = (
            upper
            + torch.log(-torch.expm1(-spread))
            - F.softplus(lower)
            - F.softplus(upper)
        )
        below = -F.softplus(-upper)
        component = torch.where(bottom, below, interval)
        return torch.logsumexp(F.log_softmax(self.logits, dim=-1) + component, dim=-1)

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        """Return one move time per decision, in whole milliseconds."""

        component = torch.multinomial(
            F.softmax(self.logits.flatten(0, -2), dim=-1),
            1,
            generator=generator,
        ).view(self.logits.shape[:-1])
        location = self.locations.gather(-1, component.unsqueeze(-1)).squeeze(-1)
        scale = self.log_scales.gather(-1, component.unsqueeze(-1)).squeeze(-1).exp()
        uniform = torch.rand(
            location.shape,
            generator=generator,
            device=location.device,
        ).clamp(1e-6, 1.0 - 1e-6)
        logged = (location + scale * torch.logit(uniform)).clamp(max=_LONGEST_LOGGED_MS)
        return torch.expm1(logged).clamp(min=0.0).round().to(torch.long)
