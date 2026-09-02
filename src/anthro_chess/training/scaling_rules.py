"""The rules a run's scale-dependent settings are produced by, and their range.

``docs/scaling.md`` says a hyperparameter that depends on scale is recorded as
the rule that produces it rather than as the number the rule produced. This
module is that record for the five settings a size or horizon change moves:
the peak learning rate, the batch, the warmup, the weight-decay timescale, and
the optimizer's second-moment decay.

Two conventions hold everywhere below.

**A parameter count is every trainable tensor of the assembled model, the
action head included.** ``anthro_chess.models.parameter_count`` is the one
implementation, and omitting the head is the confound
``docs/research.md`` (Resolving Discrepancies In Compute-Optimal Scaling) puts
first. A horizon is counted in positions, which is decisions rather than square
tokens; compute is 64 times that, and ``docs/scaling.md`` owns why the two come
apart here.

**A fitted rule is evaluated only inside the range it was fitted over.** Each
range below is the span the arms actually covered, and asking for a scale
outside one raises rather than extrapolating. That refusal is the whole
difference between a rule and a guess: nothing in a fit's residuals says where
it stops holding, so the boundary has to be carried separately.

``docs/decisions/0087-hyperparameter-rules-are-fitted-along-the-regime-ray.md``
records what was run, what each exponent came out at, and what the fit does not
establish.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from anthro_chess.models import MoveModelConfig, parameter_count

#: The shape rules ``docs/scaling.md`` fixes: depth does not scale, a
#: feed-forward is twice the width, an attention head is 32 wide, and the
#: geometric bias bank is a quarter of the width. Width is the single dial, so
#: these are what turn it into a model.
LAYERS = 8
FEEDFORWARD_MULTIPLE = 2
ATTENTION_HEAD_DIM = 32
GEOMETRIC_BIAS_DIVISOR = 4

#: The micro-batch is a throughput dial rather than a learning one, and 1024 is
#: the optimum ``0076`` measured. The effective batch below is reached by
#: accumulation, which is what keeps the two independent.
MICRO_BATCH_POSITIONS = 1024

#: The share of the horizon the cooldown occupies, carried from the schedule
#: family ``0067`` fixed rather than fitted here. It is already horizon
#: invariant, being declared as a fraction, so no rule has to produce it.
COOLDOWN_FRACTION = 0.2


class OutsideFittedRange(ValueError):
    """Raised when a rule is asked for a scale it was never fitted over."""


@dataclass(frozen=True)
class FittedRange:
    """The span of one scale axis the arms behind a rule actually covered."""

    quantity: str
    low: float
    high: float

    def check(self, value: float) -> None:
        """Refuse a value the arms did not reach."""

        if not self.low <= value <= self.high:
            raise OutsideFittedRange(
                f"{self.quantity} of {value:g} is outside the fitted range "
                f"[{self.low:g}, {self.high:g}]; extend the fit before asking "
                f"for a rule there"
            )


#: Widths the arms spanned. The target is wider than this on purpose: the
#: ladder is what extends the range, and until it runs a rule evaluated at the
#: target would be an extrapolation wearing a fit's clothes.
MODEL_DIM_RANGE = FittedRange("model_dim", 32, 128)

#: How long a run is, relative to its own capacity. The vehicle and the target
#: both sit at 800.
POSITIONS_PER_PARAMETER_RANGE = FittedRange("positions per parameter", 100, 1600)

#: Positions per optimizer step. Below the low end the step is launch bound and
#: above the high end nothing here measured.
BATCH_RANGE = FittedRange("positions per optimizer step", 4096, 65536)


def model_config_for_width(model_dim: int) -> MoveModelConfig:
    """Return the model one width produces under the fixed shape rules."""

    MODEL_DIM_RANGE.check(model_dim)
    if model_dim % ATTENTION_HEAD_DIM:
        raise ValueError(
            f"model_dim {model_dim} does not divide into {ATTENTION_HEAD_DIM}-wide "
            f"attention heads"
        )
    return MoveModelConfig(
        model_dim=model_dim,
        attention_heads=model_dim // ATTENTION_HEAD_DIM,
        layers=LAYERS,
        feedforward_dim=FEEDFORWARD_MULTIPLE * model_dim,
        geometric_bias_dim=model_dim // GEOMETRIC_BIAS_DIVISOR,
    )


@dataclass(frozen=True)
class TrainingScale:
    """Where on the two scale axes a run sits.

    Width is the model dial and positions per parameter is the horizon dial,
    so a run's absolute horizon is derived rather than declared. That is what
    makes a rung of a ladder one line.
    """

    model_dim: int
    positions_per_parameter: float

    def __post_init__(self) -> None:
        MODEL_DIM_RANGE.check(self.model_dim)
        POSITIONS_PER_PARAMETER_RANGE.check(self.positions_per_parameter)

    @property
    def model(self) -> MoveModelConfig:
        return model_config_for_width(self.model_dim)

    @property
    def parameters(self) -> int:
        return parameter_count(self.model)

    @property
    def positions(self) -> int:
        return round(self.positions_per_parameter * self.parameters)


#: The scale every rule is expressed relative to: the ablation vehicle, whose
#: rate ``0076`` swept at this exact point. Anchoring here rather than at an
#: arbitrary round number means the reference is a measurement rather than an
#: interpolation, and that the vehicle's own settings come back out of the
#: rules that were fitted through it.
REFERENCE_PARAMETERS = 1_422_662
REFERENCE_POSITIONS_PER_PARAMETER = 800
REFERENCE_POSITIONS = REFERENCE_POSITIONS_PER_PARAMETER * REFERENCE_PARAMETERS
REFERENCE_BATCH_POSITIONS = 16_384

# PROVISIONAL: replaced from the fit before this lands. See scratchpad PENDING.md.
LEARNING_RATE_AT_REFERENCE = 2.91e-3
SIZE_EXPONENT = 0.0
HORIZON_EXPONENT = 0.0
BATCH_EXPONENT = 0.0
BATCH_HORIZON_EXPONENT = 0.0
WARMUP_FRACTION = 0.01
WEIGHT_DECAY_HORIZONS = float("inf")
SECOND_MOMENT_POSITIONS = 1.6384e7


def batch_positions(positions: int) -> int:
    """Return the positions one optimizer step should span at this horizon.

    Batch is fitted against the horizon rather than against the model, which
    is the parameterization outside work uses and the one ``docs/research.md``
    records as disagreeing with a size-indexed fit. Snapped to a power of two,
    because that is the spacing the axis was measured on and the loss surface
    between two measured batches is flatter than the gap between them.
    """

    grown = (positions / REFERENCE_POSITIONS) ** BATCH_HORIZON_EXPONENT
    wanted = REFERENCE_BATCH_POSITIONS * grown
    snapped = 2 ** round(math.log2(wanted))
    BATCH_RANGE.check(snapped)
    return int(snapped)


def peak_learning_rate(parameters: int, positions: int, batch: int) -> float:
    """Return the rate the trunk holds, for a model, a horizon, and a batch.

    Three terms rather than one. The size and horizon exponents are separated
    by fitting the ray both the vehicle and the target sit on and then the
    horizon alone at one width: along that ray a size change is also a horizon
    change, so a single sweep cannot tell the two apart.
    """

    BATCH_RANGE.check(batch)
    rate: float = (
        LEARNING_RATE_AT_REFERENCE
        * (parameters / REFERENCE_PARAMETERS) ** SIZE_EXPONENT
        * (positions / REFERENCE_POSITIONS) ** HORIZON_EXPONENT
        * (batch / REFERENCE_BATCH_POSITIONS) ** BATCH_EXPONENT
    )
    return rate


def warmup_positions(positions: int) -> int:
    """Return how much data warmup spans, as a share of the horizon.

    A quantity of data rather than a step count, which is what the training
    configuration takes and what keeps a branch warming up over the same prefix
    as the trunk it resumes.
    """

    return round(WARMUP_FRACTION * positions)


def weight_decay(learning_rate: float, positions: int, batch: int) -> float:
    """Return the coefficient that puts the decay timescale at its rule.

    Decoupled decay shrinks a weight by ``learning_rate * weight_decay`` each
    step, so the coefficient means nothing on its own: what it sets is a
    timescale of ``1 / (learning_rate * weight_decay)`` steps, and a horizon
    change moves the coefficient that holds that timescale. The rule is stated
    over the timescale, which is the half of the pair a horizon change leaves
    alone.

    Zero is a legitimate answer and is what an infinite timescale means.
    """

    if WEIGHT_DECAY_HORIZONS == float("inf"):
        return 0.0
    steps = positions / batch
    return 1.0 / (learning_rate * WEIGHT_DECAY_HORIZONS * steps)


def second_moment_decay(batch: int) -> float:
    """Return Adam's second-moment decay for a batch, from its timescale.

    The second moment averages over ``1 / (1 - decay)`` optimizer steps, which
    is a number of steps rather than a quantity of data. Hold the constant and
    a batch change silently rescales how much data the average spans, so the
    rule is stated over positions and the constant is derived from the batch.
    """

    BATCH_RANGE.check(batch)
    return 1.0 - batch / SECOND_MOMENT_POSITIONS


@dataclass(frozen=True)
class ResolvedRun:
    """Every scale-dependent setting one run takes, and how it got there."""

    scale: TrainingScale
    parameters: int
    positions: int
    positions_per_batch: int
    gradient_accumulation_steps: int
    batch_positions: int
    steps: int
    learning_rate: float
    warmup_positions: int
    cooldown_fraction: float
    weight_decay: float
    second_moment_decay: float

    @property
    def weight_decay_steps(self) -> float:
        """The timescale the coefficient encodes, in optimizer steps."""

        if not self.weight_decay:
            return float("inf")
        return 1.0 / (self.learning_rate * self.weight_decay)

    @property
    def second_moment_positions(self) -> float:
        """The timescale the second-moment decay encodes, in positions."""

        return self.batch_positions / (1.0 - self.second_moment_decay)


def resolve(scale: TrainingScale) -> ResolvedRun:
    """Return every setting the rules produce at one scale.

    The order is forced rather than chosen: the batch follows from the horizon,
    the rate needs the batch, and the two timescales need the rate and the
    batch. Resolving in any other order would need a setting that does not
    exist yet.
    """

    parameters = scale.parameters
    positions = scale.positions
    batch = batch_positions(positions)
    rate = peak_learning_rate(parameters, positions, batch)
    return ResolvedRun(
        scale=scale,
        parameters=parameters,
        positions=positions,
        positions_per_batch=MICRO_BATCH_POSITIONS,
        gradient_accumulation_steps=batch // MICRO_BATCH_POSITIONS,
        batch_positions=batch,
        steps=positions // batch,
        learning_rate=rate,
        warmup_positions=warmup_positions(positions),
        cooldown_fraction=COOLDOWN_FRACTION,
        weight_decay=weight_decay(rate, positions, batch),
        second_moment_decay=second_moment_decay(batch),
    )


__all__ = [
    "BATCH_RANGE",
    "COOLDOWN_FRACTION",
    "MICRO_BATCH_POSITIONS",
    "MODEL_DIM_RANGE",
    "POSITIONS_PER_PARAMETER_RANGE",
    "FittedRange",
    "OutsideFittedRange",
    "ResolvedRun",
    "TrainingScale",
    "batch_positions",
    "model_config_for_width",
    "peak_learning_rate",
    "resolve",
    "second_moment_decay",
    "warmup_positions",
    "weight_decay",
]
