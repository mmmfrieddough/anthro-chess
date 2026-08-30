"""What the resignation reading measures, and what it refuses to measure.

Everything here is one deterministic pass over frozen human games, so the
tests are about the join and the reductions rather than about generated play:
that both sides of a calibration band are read at the same plies, that the
error is weighted by how often a position comes up, and that a reading with no
population behind it says so instead of writing a zero a reader would take for
a measurement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from anthro_chess.chess import ACTION_VOCABULARY_SIZE, RESIGNATION_ACTION_ID
from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.data import DecisionContext
from anthro_chess.evaluation import PoolConfig, freeze_pool
from anthro_chess.evaluation.benchmarks import benchmark_registry, run_benchmark
from anthro_chess.evaluation.cost import BENCHMARK_COST_KIND
from anthro_chess.evaluation.results import (
    CheckpointReference,
    DetailStore,
    ResultsStore,
)
from anthro_chess.evaluation.results.metrics import (
    GAME_TERMINATION_FAMILY,
    TERMINATION_PREDICTION_PROJECTION,
    TERMINATION_RESIGNATION_CALIBRATION_ERROR,
    TERMINATION_RESIGNATION_CALIBRATION_GAP,
    TERMINATION_RESIGNATION_MASS_AT_MOVES,
    TERMINATION_RESIGNATION_MASS_AT_RESIGNATION,
    TERMINATION_RESIGNATION_MASS_SEPARATION,
    MetricDirection,
    registered_metrics,
)
from anthro_chess.evaluation.termination import (
    TERMINATION_KIND,
    CalibrationBucket,
    ResignationCalibration,
    TerminationBenchmarkConfig,
    TerminationBenchmarkError,
    TerminationBenchmarkResult,
)

CHECKPOINT = CheckpointReference(label="fixture-checkpoint", step=1)

#: White wins a pawn on move three and keeps it, so the scored plies span three
#: deficit bands instead of sitting level in one: the two before the capture,
#: the ones where Black is a pawn down, and the ones where White is a pawn up.
#: Six plies leaves White to move, which is what makes a White loss append the
#: resignation action rather than omit it for the opponent's turn.
PAWN_UP_MOVES = ("e2e4", "d7d5", "e4d5", "g8f6", "g1f3", "f6g4")


@dataclass
class MovePlayingRunner:
    """A stand-in policy that plays moves and cannot score whole batches."""

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def predict(self, context: DecisionContext) -> torch.Tensor:
        return torch.zeros(ACTION_VOCABULARY_SIZE)


@dataclass
class TargetAwareScorer:
    """A stand-in scorer whose resignation mass tracks the human's own action.

    It reads the batch's targets, which no real model can do. That is the
    point: it produces a reading whose direction is known in advance, so a test
    can assert the halves are separated the way they are defined to be rather
    than merely that both are floats.
    """

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    #: Zero makes the scorer blind to the target, which is what the collapsed
    #: separation case needs.
    separation: float = 6.0
    #: Applied at every scored ply, so a large negative reads as a policy that
    #: never wants to resign anywhere.
    baseline: float = 0.0

    def predict(self, context: DecisionContext) -> torch.Tensor:
        return torch.zeros(ACTION_VOCABULARY_SIZE)

    def action_logits(self, batch: Any) -> torch.Tensor:
        shape = (*batch.action_targets.shape, ACTION_VOCABULARY_SIZE)
        logits = torch.zeros(shape, dtype=torch.float32)
        resigned = batch.action_targets == RESIGNATION_ACTION_ID
        logits[..., RESIGNATION_ACTION_ID] = self.baseline + torch.where(
            resigned,
            torch.full_like(batch.action_targets, 0, dtype=torch.float32)
            + self.separation,
            torch.zeros_like(batch.action_targets, dtype=torch.float32),
        )
        return logits


def _freeze(
    directory: Path,
    rows: Sequence[dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    *,
    pool_id: str,
) -> Path:
    """Freeze one fixture pool from explicit rows."""

    normalized, manifest = write_corpus(directory / "corpus", list(rows))
    output = directory / "pool"
    freeze_pool(
        ResolvedConfig(
            value=PoolConfig.model_validate(
                {
                    "pool_id": pool_id,
                    "normalized": str(normalized),
                    "manifest": str(manifest),
                }
            ),
            provenance=ConfigProvenance(source=None, overrides=()),
        ),
        output,
    )
    return output


@pytest.fixture
def pool(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> Path:
    """Freeze a pool whose games span several material bands.

    A White loss on an even ply count leaves White to move at the end, which is
    what makes preparation attribute the resignation to the side to move and
    append the terminal action. Those are the games this reading has any
    positives at all from.
    """

    rows = [
        normalized_row(
            index,
            split="test",
            moves=PAWN_UP_MOVES,
            rating=1100 + (index % 10) * 100,
            result=("0-1", "1-0", "1/2-1/2")[index % 3],
        )
        for index in range(1, 41)
    ]
    return _freeze(tmp_path, rows, write_corpus, pool_id="fixture-termination")


def _config(pool: Path, **overrides: Any) -> ResolvedConfig[TerminationBenchmarkConfig]:
    """Return a resolved selection small enough for the CPU test suite."""

    fields: dict[str, Any] = {"pool": str(pool)}
    fields.update(overrides)
    return ResolvedConfig(
        value=TerminationBenchmarkConfig.model_validate(fields),
        provenance=ConfigProvenance(source=None, overrides=()),
    )


def _run(
    config: ResolvedConfig[TerminationBenchmarkConfig],
    *,
    runner: Any | None = None,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> TerminationBenchmarkResult:
    return cast(
        TerminationBenchmarkResult,
        run_benchmark(
            benchmark_registry()["termination"],
            config,
            store=store,
            detail=detail,
            runner=runner or TargetAwareScorer(),
            checkpoint=CHECKPOINT,
        ),
    )


# --- The two mass readings ------------------------------------------------


def test_the_reading_separates_resignation_plies_from_move_plies(pool: Path) -> None:
    """Both halves come out of one pass, and the separation is their difference."""

    held_out = _run(_config(pool)).held_out
    assert held_out.resignation_plies > 0
    assert held_out.move_plies > held_out.resignation_plies
    assert held_out.mass_at_resignation is not None
    assert held_out.mass_at_moves is not None
    assert held_out.mass_at_resignation > held_out.mass_at_moves
    assert held_out.separation == pytest.approx(
        held_out.mass_at_resignation - held_out.mass_at_moves
    )


def test_a_policy_blind_to_the_ending_shows_no_separation(pool: Path) -> None:
    """The reading has to be able to say the model learned nothing."""

    held_out = _run(_config(pool), runner=TargetAwareScorer(separation=0.0)).held_out
    assert held_out.separation == pytest.approx(0.0, abs=1e-9)


def test_the_two_halves_carry_their_own_sample_sizes(pool: Path) -> None:
    """The two populations differ by orders of magnitude and must say so."""

    result = _run(_config(pool))
    held_out = result.held_out
    envelope = _envelope(result)
    at_resignation = envelope.measurement(
        TERMINATION_RESIGNATION_MASS_AT_RESIGNATION.identifier
    )
    at_moves = envelope.measurement(TERMINATION_RESIGNATION_MASS_AT_MOVES.identifier)
    assert at_resignation is not None
    assert at_moves is not None
    assert at_resignation.sample_size == held_out.resignation_plies
    assert at_moves.sample_size == held_out.move_plies
    assert at_moves.sample_size != at_resignation.sample_size


# --- The deficit calibration ----------------------------------------------


def test_the_calibration_reads_both_sides_at_the_same_plies(pool: Path) -> None:
    """One position distribution, not one per side, is what makes it a comparison."""

    calibration = _run(_config(pool)).held_out.calibration
    assert len(calibration.buckets) > 1
    for bucket in calibration.buckets:
        assert bucket.plies > 0
        assert bucket.human_rate == pytest.approx(
            bucket.human_resignations / bucket.plies
        )
        assert bucket.gap == pytest.approx(bucket.model_mass - bucket.human_rate)
    assert calibration.plies == sum(bucket.plies for bucket in calibration.buckets)


def test_the_calibration_bands_run_in_deficit_order(pool: Path) -> None:
    """Sorting the names would file "below" first and "9-and-above" mid-range."""

    calibration = _run(_config(pool)).held_out.calibration
    names = [bucket.bucket for bucket in calibration.buckets]
    assert names[0] == "below-0"
    assert names == sorted(names, key=_declared_order.index)


def test_a_policy_that_never_resigns_reads_as_a_negative_gap(pool: Path) -> None:
    """A signed reading is what separates resigning too readily from too rarely."""

    calibration = _run(
        _config(pool), runner=TargetAwareScorer(separation=0.0, baseline=-40.0)
    ).held_out.calibration
    assert calibration.gap is not None
    assert calibration.error is not None
    assert calibration.gap < 0.0
    # Every band is short of the human rate, so the absolute reading is the
    # signed one negated rather than something larger.
    assert calibration.error == pytest.approx(-calibration.gap)


def test_the_calibration_error_weights_a_band_by_how_often_it_comes_up() -> None:
    """An unweighted mean would let thirty plies count for twenty thousand."""

    calibration = ResignationCalibration(
        buckets=(
            CalibrationBucket(
                bucket="0-to-1",
                plies=990,
                human_resignations=0,
                model_mass=0.0,
            ),
            CalibrationBucket(
                bucket="9-and-above",
                plies=10,
                human_resignations=5,
                model_mass=0.0,
            ),
        )
    )

    assert calibration.plies == 1000
    assert calibration.error == pytest.approx(0.005)
    assert calibration.gap == pytest.approx(-0.005)


def test_a_calibration_with_no_plies_reports_nothing_rather_than_zero() -> None:
    """Zero error would read as a policy that matched humans exactly."""

    calibration = ResignationCalibration(buckets=())

    assert calibration.plies == 0
    assert calibration.error is None
    assert calibration.gap is None


# --- Readings with nothing behind them ------------------------------------


def test_a_view_without_a_resignation_reports_unavailable(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """No positive to score against is a state, not a zero mass.

    This is also the shape a corpus whose vocabulary predates terminal actions
    takes here. Such a pool is rejected outright when it is loaded, because its
    action-vocabulary identity no longer matches; what survives that check and
    still has nothing to measure is a compatible pool holding no game that
    carries a terminal action, which is exactly this.
    """

    rows = [
        normalized_row(index, split="test", plies=6, rating=1500, result="1-0")
        for index in range(1, 13)
    ]
    output = _freeze(tmp_path, rows, write_corpus, pool_id="fixture-no-resignations")
    result = _run(_config(output))
    held_out = result.held_out
    assert held_out.resignation_plies == 0
    assert held_out.mass_at_resignation is None
    assert "resignation_mass_at_resignation" in held_out.unavailable
    reported = {
        found.metric for envelope in result.envelopes for found in envelope.measurements
    }
    assert TERMINATION_RESIGNATION_MASS_AT_RESIGNATION.identifier not in reported
    assert TERMINATION_RESIGNATION_MASS_SEPARATION.identifier not in reported
    # And the calibration with it. Every band's human rate is zero here, so any
    # mass the policy spends would be committed as spending more than humans
    # did, against a rate no human supplied.
    assert held_out.calibration.plies > 0
    assert held_out.calibration.human_resignations == 0
    assert held_out.calibration.error is None
    assert held_out.calibration.gap is None
    assert "resignation_calibration" in held_out.unavailable
    assert TERMINATION_RESIGNATION_CALIBRATION_ERROR.identifier not in reported
    assert TERMINATION_RESIGNATION_CALIBRATION_GAP.identifier not in reported
    # The half that does have a population is still reported, so one missing
    # reading does not take the rest of the family down with it.
    assert TERMINATION_RESIGNATION_MASS_AT_MOVES.identifier in reported


def test_a_runner_that_cannot_score_batches_says_so(pool: Path) -> None:
    """A stand-in that only plays games cannot produce this reading at all."""

    with pytest.raises(TerminationBenchmarkError, match="scores whole batches"):
        _run(_config(pool), runner=MovePlayingRunner())


def test_a_pool_this_reading_is_not_defined_over_is_refused(pool: Path) -> None:
    """A superseded pool left at the configured path is a different population."""

    config = _config(pool, expected_pool_game_ids_sha256="0" * 64)
    with pytest.raises(TerminationBenchmarkError, match="not the one this"):
        _run(config)


# --- What reaches the store -----------------------------------------------


def test_the_reading_is_scoped_by_the_content_it_scored(pool: Path) -> None:
    """Nothing was generated, so the human decisions are series identity."""

    envelope = _envelope(_run(_config(pool)))
    assert envelope.execution is None
    assert envelope.data is not None
    assert [digest.projection for digest in envelope.data.components] == [
        TERMINATION_PREDICTION_PROJECTION.name
    ]


def test_the_calibration_reaches_the_committed_tier(pool: Path) -> None:
    """A reading nothing records is a series that never starts."""

    result = _run(_config(pool))
    envelope = _envelope(result)
    reported = {found.metric for found in envelope.measurements}
    assert TERMINATION_RESIGNATION_CALIBRATION_ERROR.identifier in reported
    assert TERMINATION_RESIGNATION_CALIBRATION_GAP.identifier in reported
    assert (
        envelope.measurement(
            TERMINATION_RESIGNATION_CALIBRATION_ERROR.identifier
        ).sample_size
        == result.held_out.calibration.plies
    )


def test_a_calibration_with_no_human_resignation_reports_nothing() -> None:
    """Zero on both sides of every band is a missing reference, not agreement."""

    calibration = ResignationCalibration(
        buckets=(
            CalibrationBucket(
                bucket="9-and-above",
                plies=400,
                human_resignations=0,
                model_mass=0.02,
            ),
        )
    )

    assert calibration.plies == 400
    assert calibration.human_resignations == 0
    assert calibration.error is None
    assert calibration.gap is None


def test_every_registered_metric_of_the_family_is_reported(pool: Path) -> None:
    """A registered metric no benchmark writes is a series that never starts."""

    result = _run(_config(pool))
    reported = {found.metric for found in _envelope(result).measurements}
    assert reported == {
        metric.identifier
        for metric in registered_metrics(GAME_TERMINATION_FAMILY.identifier)
    }


def test_the_reading_is_recorded_under_its_own_kind(
    pool: Path,
    tmp_path: Path,
) -> None:
    """The cost of the run is filed apart from the reading it paid for."""

    store = ResultsStore(tmp_path / "store")
    result = _run(_config(pool), store=store, detail=DetailStore(tmp_path / "detail"))

    kinds = {envelope.kind for envelope in result.envelopes}
    assert kinds == {TERMINATION_KIND, BENCHMARK_COST_KIND}
    assert result.recorded_paths


def test_the_family_reports_its_defects_with_a_direction() -> None:
    """A reading with no direction cannot say whether a change was an improvement."""

    directions = {
        metric.identifier: metric.direction
        for metric in registered_metrics(GAME_TERMINATION_FAMILY.identifier)
    }

    assert (
        directions[TERMINATION_RESIGNATION_MASS_SEPARATION.identifier]
        is MetricDirection.HIGHER_IS_BETTER
    )
    assert (
        directions[TERMINATION_RESIGNATION_CALIBRATION_ERROR.identifier]
        is MetricDirection.LOWER_IS_BETTER
    )
    # Signed rather than directional: resigning too readily and too rarely are
    # different findings, and neither is the one this number is minimized at.
    assert (
        directions[TERMINATION_RESIGNATION_CALIBRATION_GAP.identifier]
        is MetricDirection.INFORMATIONAL
    )


_declared_order = [
    "below-0",
    "0-to-1",
    "1-to-3",
    "3-to-5",
    "5-to-9",
    "9-and-above",
]


def _envelope(result: TerminationBenchmarkResult) -> Any:
    """Return the one reading envelope, apart from the cost record."""

    envelopes = [
        envelope for envelope in result.envelopes if envelope.kind == TERMINATION_KIND
    ]
    assert len(envelopes) == 1
    return envelopes[0]
