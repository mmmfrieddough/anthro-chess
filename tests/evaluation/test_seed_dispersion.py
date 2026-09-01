"""The spread arms of one frozen configuration show, and how it is found."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from anthro_chess.evaluation.results import DEFAULT_COVERAGE, dispersion_bound
from anthro_chess.evaluation.seed_dispersion import (
    SEED_ARM_METHOD,
    ArmReading,
    SeedDispersion,
    SeedDispersionError,
    characterize,
    read_seed_dispersion,
    seed_dispersion_for,
    write_seed_dispersion,
)

MEASURED_AT = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
IDENTITY = "5a" * 32
OTHER_IDENTITY = "6b" * 32
MOVE_LOSS = "held_out.move_loss"
GRADIENT_NORM = "training_health.gradient_norm"


def _arm(
    run_id: str,
    seed: int,
    *,
    move_loss: float,
    health: Mapping[str, float] | None = None,
    metrics: Mapping[str, Mapping[str, float]] | None = None,
    training_seconds: float = 3600.0,
) -> ArmReading:
    return ArmReading(
        run_id=run_id,
        seed=seed,
        checkpoint=f"{run_id}-step-00069465",
        training_seconds=training_seconds,
        metrics=metrics if metrics is not None else {MOVE_LOSS: {"": move_loss}},
        health=health if health is not None else {GRADIENT_NORM: 0.5},
    )


def _characterized(
    *readings: ArmReading,
    training_sha256: str = IDENTITY,
    horizon_steps: int = 69465,
    scoring_seconds: float = 600.0,
) -> SeedDispersion:
    return characterize(
        readings,
        training_sha256=training_sha256,
        horizon_steps=horizon_steps,
        scoring_seconds=scoring_seconds,
        measured_at=MEASURED_AT,
    )


def test_the_spread_is_read_over_one_arm_per_seed() -> None:
    """A repeated seed measures nondeterminism, not another draw of the seed.

    Two arms at one seed differ by the run rather than by the initialization,
    so letting the second contribute to the spread would pull it toward a
    quantity the floor is not describing, and would claim a degree of freedom
    the measurement does not have.
    """

    dispersion = _characterized(
        _arm("a", 17, move_loss=1.40),
        _arm("b", 17, move_loss=1.41),
        _arm("c", 29, move_loss=1.50),
        _arm("d", 43, move_loss=1.60),
    )

    assert dispersion.seeds == (17, 29, 43)
    # The spread of 1.40, 1.50, 1.60 rather than of all four values.
    assert dispersion.metrics[MOVE_LOSS][""].value == pytest.approx(0.1)
    assert dispersion.metrics[MOVE_LOSS][""].estimator == SEED_ARM_METHOD
    assert dispersion.nondeterminism[MOVE_LOSS][""].value == pytest.approx(
        0.01 / 2**0.5
    )


def test_a_floor_is_the_delta_two_arms_of_the_configuration_could_produce() -> None:
    """One arm's spread is not the spread of a delta between two of them."""

    dispersion = _characterized(
        _arm("a", 17, move_loss=1.40),
        _arm("b", 29, move_loss=1.50),
        _arm("c", 43, move_loss=1.60),
    )

    spread = dispersion.metrics[MOVE_LOSS][""]
    expected = DEFAULT_COVERAGE * (2**0.5) * spread.bound
    assert dispersion.floor(MOVE_LOSS) == pytest.approx(expected)
    assert spread.bound == pytest.approx(
        dispersion_bound(spread.value, degrees_of_freedom=2)
    )


def test_a_metric_only_some_arms_reported_carries_no_spread() -> None:
    """Mixing replicate counts would make the bound depend on the row asked for."""

    dispersion = _characterized(
        _arm("a", 17, move_loss=1.40, metrics={MOVE_LOSS: {"": 1.40}}),
        _arm(
            "b",
            29,
            move_loss=1.50,
            metrics={MOVE_LOSS: {"": 1.50}, "legality.mask_penalty": {"": 0.1}},
        ),
        _arm("c", 43, move_loss=1.60, metrics={MOVE_LOSS: {"": 1.60}}),
    )

    assert set(dispersion.metrics) == {MOVE_LOSS}
    assert dispersion.floor("legality.mask_penalty") is None


def test_a_cell_every_arm_agreed_on_exactly_is_withheld() -> None:
    """A zero spread would produce a floor that clears every delta.

    The arms observed that they could not move the quantity, which is not the
    same as observing that nothing could, so the row reports no seed floor
    rather than one of zero.
    """

    settled = "legality.mask_penalty"
    dispersion = _characterized(
        _arm(
            "a", 17, move_loss=1.4, metrics={MOVE_LOSS: {"": 1.4}, settled: {"": 0.0}}
        ),
        _arm(
            "b", 29, move_loss=1.5, metrics={MOVE_LOSS: {"": 1.5}, settled: {"": 0.0}}
        ),
        _arm(
            "c", 43, move_loss=1.6, metrics={MOVE_LOSS: {"": 1.6}, settled: {"": 0.0}}
        ),
    )

    assert dispersion.floor(MOVE_LOSS) is not None
    assert settled not in dispersion.metrics
    assert dispersion.floor(settled) is None


def test_a_workload_scoped_metric_keeps_one_spread_per_cell() -> None:
    """A spread pooled over a matrix would describe none of its cells."""

    plies = "generated_play.mean_game_plies"
    dispersion = _characterized(
        _arm(
            "a",
            17,
            move_loss=1.4,
            metrics={plies: {"cell-1200": 60.0, "cell-1800": 80.0}},
        ),
        _arm(
            "b",
            29,
            move_loss=1.4,
            metrics={plies: {"cell-1200": 62.0, "cell-1800": 90.0}},
        ),
        _arm(
            "c",
            43,
            move_loss=1.4,
            metrics={plies: {"cell-1200": 64.0, "cell-1800": 100.0}},
        ),
    )

    assert dispersion.metrics[plies]["cell-1200"].value == pytest.approx(2.0)
    assert dispersion.metrics[plies]["cell-1800"].value == pytest.approx(10.0)
    narrow = dispersion.floor(plies, "cell-1200")
    wide = dispersion.floor(plies, "cell-1800")
    assert narrow is not None
    assert wide is not None
    assert wide > narrow


def test_arms_that_all_share_one_seed_characterize_nothing() -> None:
    """What they measure is nondeterminism, which is not the stored floor."""

    with pytest.raises(SeedDispersionError, match="two or more distinct seeds"):
        _characterized(
            _arm("a", 17, move_loss=1.40),
            _arm("b", 17, move_loss=1.41),
        )


def test_the_health_band_is_the_arms_own_spread_rather_than_their_range() -> None:
    """A range over a handful of arms is as wide as its widest arm and no wider."""

    dispersion = _characterized(
        _arm("a", 17, move_loss=1.40, health={GRADIENT_NORM: 0.50}),
        _arm("b", 29, move_loss=1.50, health={GRADIENT_NORM: 0.52}),
        _arm("c", 43, move_loss=1.60, health={GRADIENT_NORM: 0.54}),
    )

    band = dispersion.health[GRADIENT_NORM]
    assert band.center == pytest.approx(0.52)
    assert band.covers(0.52 + band.covered * 0.9)
    assert not band.covers(0.52 + band.covered * 1.1)
    assert dispersion.departures({GRADIENT_NORM: 0.52}) == ()
    assert dispersion.departures({GRADIENT_NORM: 9.0}) == (GRADIENT_NORM,)


def test_a_health_reading_the_arms_never_took_is_not_a_departure() -> None:
    """Silence about a quantity is not evidence that an arm is unstable."""

    dispersion = _characterized(
        _arm("a", 17, move_loss=1.40, health={GRADIENT_NORM: 0.50}),
        _arm("b", 29, move_loss=1.50, health={GRADIENT_NORM: 0.52}),
    )

    assert dispersion.departures({"training_health.clip_rate": 0.9}) == ()


def test_the_measurement_records_what_it_cost() -> None:
    """So a session replacing the vehicle can price re-characterizing it."""

    dispersion = _characterized(
        _arm("a", 17, move_loss=1.4, training_seconds=21000.0),
        _arm("b", 29, move_loss=1.5, training_seconds=21200.0),
        scoring_seconds=1800.0,
    )

    assert dispersion.training_seconds == pytest.approx(42200.0)
    assert dispersion.wall_clock_seconds == pytest.approx(44000.0)


def test_a_characterization_is_found_by_exact_digest_or_not_at_all(
    tmp_path: Path,
) -> None:
    """A floor findable approximately is one applicable to what it never measured."""

    dispersion = _characterized(
        _arm("a", 17, move_loss=1.40),
        _arm("b", 29, move_loss=1.50),
    )
    path = write_seed_dispersion(dispersion, directory=tmp_path)

    assert path.name == f"{IDENTITY}.json"
    assert seed_dispersion_for(IDENTITY, directory=tmp_path) == dispersion
    assert seed_dispersion_for(OTHER_IDENTITY, directory=tmp_path) is None
    # A prefix of the recorded digest is a different configuration, not a
    # near-enough one.
    assert seed_dispersion_for(IDENTITY[:-1] + "b", directory=tmp_path) is None


def test_a_characterization_filed_under_another_digest_is_refused(
    tmp_path: Path,
) -> None:
    """A file a lookup by its own identity could never reach is a broken record."""

    dispersion = _characterized(
        _arm("a", 17, move_loss=1.40),
        _arm("b", 29, move_loss=1.50),
    )
    misfiled = tmp_path / f"{OTHER_IDENTITY}.json"
    misfiled.write_text(json.dumps(dispersion.model_dump(mode="json")))

    with pytest.raises(SeedDispersionError, match="would never reach this file"):
        read_seed_dispersion(misfiled)
