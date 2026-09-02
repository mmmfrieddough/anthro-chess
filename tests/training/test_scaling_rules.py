"""The scale-dependent settings, the range they hold over, and the refusal."""

from __future__ import annotations

from pathlib import Path

import pytest

from anthro_chess.config import load_config
from anthro_chess.models import MoveModel, parameter_count
from anthro_chess.training.config import TrainingConfig
from anthro_chess.training.scaling_rules import (
    MODEL_DIM_RANGE,
    POSITIONS_PER_PARAMETER_RANGE,
    POSITIONS_RANGE,
    REFERENCE_BATCH_POSITIONS,
    REFERENCE_PARAMETERS,
    REFERENCE_POSITIONS_PER_PARAMETER,
    OutsideFittedRange,
    TrainingScale,
    batch_positions,
    model_config_for_width,
    resolve,
    second_moment_decay,
)

VEHICLE_CONFIG_PATH = (
    Path(__file__).parents[2] / "configs/training/ablation-vehicle.toml"
)


@pytest.fixture(name="vehicle")
def _vehicle() -> TrainingConfig:
    return load_config(TrainingConfig, path=VEHICLE_CONFIG_PATH).value


def test_the_shape_rules_reproduce_the_vehicle_the_reference_was_fitted_at(
    vehicle: TrainingConfig,
) -> None:
    """The one width the shape rules can be checked against states them twice.

    ``configs/training/ablation-vehicle.toml`` writes the shape out because its
    digest is frozen and cannot be made to call anything. That leaves two
    statements of the same rules, and this is what keeps them from drifting: a
    change to either side fails here rather than in a ladder rung whose model
    is quietly a different shape from the vehicle it is read against.
    """

    assert model_config_for_width(vehicle.model.model_dim) == vehicle.model


def test_the_parameter_count_the_rules_are_expressed_against_holds_the_head() -> None:
    """Omitting the head is the confound the fitting protocol names first.

    Checked by construction rather than by a stored total: the count has to be
    every tensor the model owns, so the test asserts that the head's own
    tensors are inside it and that nothing else was dropped either.
    """

    config = model_config_for_width(64)
    model = MoveModel(config)
    head = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("action_head")
    )
    assert head > 0
    assert parameter_count(config) == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert parameter_count(config) > head


def test_the_reference_scale_is_the_vehicle_the_rules_were_anchored_at(
    vehicle: TrainingConfig,
) -> None:
    """Every rule is a ratio against this point, so it has to be this point."""

    assert REFERENCE_PARAMETERS == parameter_count(vehicle.model)
    loader = vehicle.train.loader
    assert (
        REFERENCE_BATCH_POSITIONS
        == loader.batch_extent * vehicle.gradient_accumulation_steps
    )


def test_the_vehicle_comes_back_out_of_the_rules_that_were_fitted_through_it(
    vehicle: TrainingConfig,
) -> None:
    """A fit anchored at a measured point should return that point.

    The rate is allowed the width of its own near-optimal band, since the
    anchor is the fitted vertex of a sweep rather than one of its arms, and the
    sweep's own grid did not contain the vertex.
    """

    resolved = resolve(
        TrainingScale(
            model_dim=vehicle.model.model_dim,
            positions_per_parameter=REFERENCE_POSITIONS_PER_PARAMETER,
        )
    )
    assert resolved.batch_positions == REFERENCE_BATCH_POSITIONS
    assert resolved.gradient_accumulation_steps == vehicle.gradient_accumulation_steps
    assert resolved.learning_rate == pytest.approx(vehicle.learning_rate, rel=0.5)
    assert resolved.warmup_positions == pytest.approx(
        vehicle.warmup_positions, rel=0.01
    )
    assert resolved.cooldown_fraction == vehicle.cooldown_fraction


def test_a_resolved_run_declares_a_schedule_its_own_horizon_can_carry(
    vehicle: TrainingConfig,
) -> None:
    """The settings have to survive the validator a real run passes through.

    A warmup rule and a horizon rule can each be reasonable and still produce a
    pair the schedule refuses, and that failure belongs here rather than in the
    first minute of a ladder rung.

    Scales the rules decline are skipped rather than asserted over: the width
    range and the ratio range are separate, and their corners multiply out to
    horizons no arm reached. The count at the end is what keeps this from
    passing vacuously if the rules ever declined everything.
    """

    carried = 0
    for model_dim in (32, 64, 96, 128):
        for ratio in (100, 400, 800):
            try:
                resolved = resolve(
                    TrainingScale(model_dim=model_dim, positions_per_parameter=ratio)
                )
            except OutsideFittedRange:
                continue
            configured = vehicle.model_copy(
                update={
                    "steps": resolved.steps,
                    "learning_rate": resolved.learning_rate,
                    "warmup_positions": resolved.warmup_positions,
                    "cooldown_fraction": resolved.cooldown_fraction,
                    "weight_decay": resolved.weight_decay,
                    "second_moment_decay": resolved.second_moment_decay,
                    "gradient_accumulation_steps": (
                        resolved.gradient_accumulation_steps
                    ),
                    "model": resolved.scale.model,
                }
            )
            schedule = configured.learning_rate_schedule()
            assert schedule.warmup_steps >= 1
            assert schedule.warmup_steps + schedule.cooldown_steps <= schedule.steps
            carried += 1
    assert carried >= 8


def test_the_second_moment_timescale_is_what_survives_a_batch_change() -> None:
    """The constant moves so that the data the average spans does not.

    This is the whole reason the rule is stated over positions: the decay
    averages over a number of steps, and holding it fixed while the batch
    changes silently changes the horizon of the average.
    """

    batches = (
        REFERENCE_BATCH_POSITIONS // 4,
        REFERENCE_BATCH_POSITIONS,
        REFERENCE_BATCH_POSITIONS * 4,
    )
    spans = {round(batch / (1.0 - second_moment_decay(batch))) for batch in batches}
    assert len(spans) == 1


def test_the_weight_decay_coefficient_encodes_the_timescale_the_rule_states() -> None:
    """A coefficient is meaningless alone; what it has to reproduce is a time."""

    resolved = resolve(TrainingScale(model_dim=64, positions_per_parameter=800))
    if not resolved.weight_decay:
        assert resolved.weight_decay_steps == float("inf")
        return
    assert resolved.weight_decay_steps == pytest.approx(
        1.0 / (resolved.learning_rate * resolved.weight_decay)
    )


@pytest.mark.parametrize(
    ("model_dim", "ratio"),
    [
        (MODEL_DIM_RANGE.low - 32, 800),
        (MODEL_DIM_RANGE.high + 32, 800),
        (128, POSITIONS_PER_PARAMETER_RANGE.low / 2),
        (128, POSITIONS_PER_PARAMETER_RANGE.high * 2),
    ],
)
def test_a_scale_outside_the_fit_is_refused_rather_than_extrapolated(
    model_dim: int, ratio: float
) -> None:
    """The single condition that keeps a fitted rule from becoming a guess.

    Nothing in a fit's residuals says where it stops holding, so the boundary
    is carried beside it and asking past it has to fail loudly. The target's
    own width is outside this range on purpose: the ladder is what extends it.
    """

    with pytest.raises(OutsideFittedRange):
        resolve(TrainingScale(model_dim=model_dim, positions_per_parameter=ratio))


def test_a_horizon_outside_the_measured_span_is_refused() -> None:
    """The horizon carries its own boundary, separate from the width's.

    A width inside the fit and a ratio inside the fit can still multiply to a
    horizon no arm reached, which is why the check is on the product rather than
    only on the two dials a caller sets.
    """

    with pytest.raises(OutsideFittedRange):
        batch_positions(int(POSITIONS_RANGE.high) * 4)
    with pytest.raises(OutsideFittedRange):
        batch_positions(int(POSITIONS_RANGE.low) // 4)


def test_the_batch_is_held_at_what_the_rate_rule_was_fitted_at() -> None:
    """The one setting that did not become a rule against scale.

    Sweeping it put the best batch at the same value at every horizon measured,
    so no exponent is supported. Holding it is what keeps the rate, which was
    measured at this batch and only at this batch, from being paired with one it
    was never measured at.
    """

    spans = {
        batch_positions(round(ratio * parameter_count(model_config_for_width(width))))
        for width in (32, 64, 128)
        for ratio in (100, 400, 800)
    }
    assert spans == {REFERENCE_BATCH_POSITIONS}
