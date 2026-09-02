"""The ablation vehicle's identity, and the horizon its digest cannot hold."""

from __future__ import annotations

from pathlib import Path

import pytest

from anthro_chess.config import load_config
from anthro_chess.evaluation.results.seed_dispersion import seed_dispersion_for
from anthro_chess.models import MoveModel
from anthro_chess.training.config import TrainingConfig
from anthro_chess.training.vehicle import (
    VEHICLE_TRAINING_SHA256,
    vehicle_training_sha256,
)

VEHICLE_CONFIG_PATH = (
    Path(__file__).parents[2] / "configs/training/ablation-vehicle.toml"
)


@pytest.fixture(name="vehicle")
def _vehicle() -> TrainingConfig:
    return load_config(TrainingConfig, path=VEHICLE_CONFIG_PATH).value


def test_the_vehicle_still_produces_the_identity_every_comparison_is_read_against(
    vehicle: TrainingConfig,
) -> None:
    """A change to the frozen configuration has to fail here and nowhere later.

    Every reading taken against the vehicle carries this digest, and the seed
    dispersion that qualifies those readings is stored under it. An edit that
    moved it would leave every earlier comparison describing a configuration
    that no longer exists, and no later reading would look wrong.

    A failure here is not a test to update. It says either that the edit was a
    mistake, or that a new vehicle is being designated, which invalidates the
    stored dispersion and every comparison read against the old one.
    """

    assert vehicle_training_sha256(vehicle) == VEHICLE_TRAINING_SHA256


def test_the_vehicle_trains_at_the_regime_its_size_was_derived_from(
    vehicle: TrainingConfig,
) -> None:
    """The horizon is outside the digest, so nothing above this checks it.

    ``training_sha256`` deliberately excludes the step budget, which is what
    lets a cooldown branch match its trunk. The consequence is that an edit to
    ``steps`` alone changes what the vehicle is while leaving its identity
    intact, and the dispersion stored against that identity describes the
    horizon it was measured at.
    """

    loader = vehicle.train.loader
    positions = (
        vehicle.steps * loader.batch_extent * vehicle.gradient_accumulation_steps
    )
    parameters = sum(
        parameter.numel() for parameter in MoveModel(vehicle.model).parameters()
    )
    assert positions / parameters == pytest.approx(800, rel=0.01)


def test_the_stored_seed_dispersion_describes_the_vehicle_it_is_filed_under(
    vehicle: TrainingConfig,
) -> None:
    """The floor and the configuration it qualifies have to be the same thing.

    Found by exact digest and by nothing else, so a characterization filed
    under a stale identity is not a floor that is slightly wrong: it is one no
    reading reaches. The horizon is checked too, because ``training_sha256``
    excludes the step budget, so a record could carry the right key and
    describe a run of another length.
    """

    dispersion = seed_dispersion_for(VEHICLE_TRAINING_SHA256)

    assert dispersion is not None
    assert dispersion.horizon_steps == vehicle.steps
    assert len(dispersion.seeds) >= 3
    assert dispersion.metrics
