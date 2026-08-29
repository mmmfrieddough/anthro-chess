"""What a rollout suite measures, and what makes two cells comparable."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chess
import pytest
import torch
from pydantic import ValidationError

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    DRAW_CLAIM_ACTION_ID,
    RESIGNATION_ACTION_ID,
    encode_move,
)
from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.data import DecisionContext
from anthro_chess.data.schema import NormalizedColumn
from anthro_chess.evaluation import PoolConfig, freeze_pool
from anthro_chess.evaluation import rollout as rollout_module
from anthro_chess.evaluation.benchmarks import benchmark_registry, run_benchmark
from anthro_chess.evaluation.cost import BENCHMARK_COST_KIND
from anthro_chess.evaluation.curves import (
    CurveQuantity,
)
from anthro_chess.evaluation.games import (
    GameTermination,
    analyze_games,
    analyze_trajectory,
)
from anthro_chess.evaluation.reference import (
    DECLARED_NEIGHBOURS,
    ComparedQuantity,
    ReferenceConfig,
    curve_spec,
    minimum_reference_games,
)
from anthro_chess.evaluation.results import (
    CheckpointReference,
    DetailStore,
    ResultEnvelope,
    ResultsStore,
)
from anthro_chess.evaluation.results.metrics import (
    DECISION_DECOMPOSITION_FAMILY,
    GENERATED_PLAY_CONDITIONAL_DISTANCE,
    GENERATED_PLAY_DECISIVE_GAME_RATE,
    GENERATED_PLAY_DISTINCT_GAME_FRACTION,
    GENERATED_PLAY_DIVERGENCE_HALF_DEPTH,
    GENERATED_PLAY_EXACT_REPERTOIRE_CONDITIONAL_DISTANCE,
    GENERATED_PLAY_EXACT_REPERTOIRE_POOLED_DISTANCE,
    GENERATED_PLAY_EXACT_REPERTOIRE_PRUNED_MASS,
    GENERATED_PLAY_EXACT_REPERTOIRE_WAYPOINT_MASS,
    GENERATED_PLAY_FAMILY,
    GENERATED_PLAY_MEAN_BOOK_PLY,
    GENERATED_PLAY_MEAN_CYCLE_PLY_FRACTION,
    GENERATED_PLAY_MEAN_GAME_PLIES,
    GENERATED_PLAY_RESIGNATION_RATE,
    GENERATED_PLAY_UNFINISHED_GAME_RATE,
    GENERATED_PLAY_WAYPOINT_GAME_RATE,
    GENERATED_PLAY_WHITE_SCORE,
    MetricDefinition,
    registered_metrics,
)
from anthro_chess.evaluation.rollout import (
    PREMATURE_MATERIAL_BALANCE,
    ROLLOUT_KIND,
    RepertoireWalkConfig,
    RolloutArm,
    RolloutBenchmarkConfig,
    RolloutBenchmarkError,
    RolloutBenchmarkResult,
)
from anthro_chess.inference import ModelRunnerConfig
from anthro_chess.runtime import RuntimeConfig

CHECKPOINT = CheckpointReference(label="fixture-checkpoint", step=1)


@dataclass
class TrajectoryRunner:
    """A stand-in policy that depends only on trajectory length and rating.

    Deterministic given its inputs, so a suite's reproducibility is a property
    of the harness and the seeds rather than of a lucky model.
    """

    calls: int = 0

    def predict(self, context: DecisionContext) -> torch.Tensor:
        self.calls += 1
        generator = torch.Generator().manual_seed(
            len(context.plies) * 1000 + (context.target_rating or 0)
        )
        return torch.randn(ACTION_VOCABULARY_SIZE, generator=generator)


@dataclass
class ResigningRunner:
    """A stand-in policy that always prefers to resign."""

    def predict(self, context: DecisionContext) -> torch.Tensor:
        logits = torch.zeros(ACTION_VOCABULARY_SIZE)
        logits[RESIGNATION_ACTION_ID] = 20.0
        return logits


#: One game of six plies per cell unless a test asks for more. Every count here
#: is a sample size rather than a measurement setting, so shrinking them for the
#: CPU suite leaves the measured quantities alone.
_BASE_GRID: dict[str, Any] = {
    "target_ratings": (1200,),
    "temperatures": (1.0,),
    "seeds": (0,),
}
_BASE_GENERATION: dict[str, Any] = {
    "games_per_position": 1,
    "maximum_generated_plies": 6,
    "swap_colors": False,
}
#: The walk enumerates rather than samples, so its cost is set by the shape of
#: the policy it walks, not by a sample size. The declared threshold is
#: affordable against a trained checkpoint because a trained policy concentrates
#: its mass; the stubs here spread it over every legal move, so the declared
#: value explores a tree no checkpoint would produce. Shrinking it changes what
#: these tests pay, not what they read: under a stub policy the walk prunes its
#: whole mass and quotes a distance of one at either setting. The depth still
#: reaches where the book names a destination rather than a waypoint, without
#: which the reading is skipped outright. The declared values themselves are
#: pinned by ``test_the_declared_walk_shape_is_the_one_a_suite_runs``.
_BASE_WALK: dict[str, Any] = {"plies": 5, "threshold": 0.01}


def _config(**overrides: Any) -> ResolvedConfig[RolloutBenchmarkConfig]:
    """Return a resolved suite small enough for the CPU test suite.

    Nested overrides merge into the small defaults, so a test naming one field
    of the grid does not silently inherit the production game counts.
    """

    fields: dict[str, Any] = {"runtime": RuntimeConfig()}
    # Off unless a test asks: these exercise the matrix, and a comparison needs
    # a human reference far larger than a fixture pool can hold.
    fields["reference"] = {"enabled": False, **overrides.pop("reference", {})}
    fields["grid"] = {**_BASE_GRID, **overrides.pop("grid", {})}
    fields["generation"] = {**_BASE_GENERATION, **overrides.pop("generation", {})}
    fields["walk"] = {**_BASE_WALK, **overrides.pop("walk", {})}
    fields.update(overrides)
    return ResolvedConfig(
        value=RolloutBenchmarkConfig.model_validate(fields),
        provenance=ConfigProvenance(source=None, overrides=()),
    )


def _run(
    config: ResolvedConfig[RolloutBenchmarkConfig],
    *,
    runner: Any | None = None,
    checkpoint: CheckpointReference | None = CHECKPOINT,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> RolloutBenchmarkResult:
    return cast(
        RolloutBenchmarkResult,
        run_benchmark(
            benchmark_registry()["rollout"],
            config,
            store=store,
            detail=detail,
            runner=runner or TrajectoryRunner(),
            checkpoint=checkpoint,
        ),
    )


def _readings(result: RolloutBenchmarkResult) -> tuple[ResultEnvelope, ...]:
    """Return the rollout's own records, without what the invocation cost."""

    return tuple(
        envelope for envelope in result.envelopes if envelope.kind == ROLLOUT_KIND
    )


def _curve_envelope(result: RolloutBenchmarkResult) -> ResultEnvelope:
    """Return the envelope carrying one reading's distances."""

    (reading,) = result.readings
    for envelope in result.envelopes:
        if (
            envelope.execution is not None
            and envelope.execution.workload_sha256 == reading.execution.workload_sha256
        ):
            return envelope
    raise AssertionError("no envelope was recorded for the curve reading")


def _sample(envelope: ResultEnvelope, metric: MetricDefinition) -> int | None:
    """Return one metric's recorded sample size."""

    found = envelope.measurement(metric.identifier)
    assert found is not None
    return found.sample_size


def _freeze(
    write_corpus: Callable[..., tuple[Path, Path]],
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    pool_id: str,
) -> Path:
    """Freeze fixture rows into a pool of their own.

    Several fixtures build one under the same ``tmp_path``, so the directory
    and the pool id come from the caller.
    """

    normalized, manifest = write_corpus(directory / "corpus", rows)
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
    """Freeze a tiny pool whose test games are long enough to prefix."""

    return _freeze(
        write_corpus,
        tmp_path / "prefix",
        [
            normalized_row(1, split="train"),
            normalized_row(2, split="test", plies=8),
            normalized_row(3, split="test", plies=10, result="0-1"),
        ],
        pool_id="fixture-test",
    )


@pytest.fixture
def reference_pool(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> Path:
    """Freeze a pool with enough rated games to estimate a curve from.

    Ratings are spread across the grid on purpose: a reference bunched at one
    rating would have nothing to say about how behavior varies with it, which
    is the whole point of the conditional reading.
    """

    return _freeze(
        write_corpus,
        tmp_path / "reference",
        [
            normalized_row(
                index,
                split="test",
                plies=4 + (index % 5) * 2,
                rating=1100 + (index % 12) * 100,
                result=("1-0", "0-1", "1/2-1/2")[index % 3],
            )
            for index in range(1, 61)
        ],
        pool_id="fixture-reference",
    )


@pytest.fixture
def mixed_speed_reference_pool(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> Path:
    """Freeze a rated pool holding two speed classes rather than one.

    Two classes in one pool is the shape a widened corpus has, and the one a
    reference must not average over.
    """

    return _freeze(
        write_corpus,
        tmp_path / "mixed-speed",
        [
            normalized_row(
                index,
                split="test",
                plies=4 + (index % 5) * 2,
                rating=1100 + (index % 12) * 100,
                result=("1-0", "0-1", "1/2-1/2")[index % 3],
                time_initial_ms=60_000 if index % 2 else 300_000,
            )
            for index in range(1, 61)
        ],
        pool_id="fixture-mixed-speed",
    )


@pytest.fixture
def mismatched_reference_pool(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> Path:
    """Freeze a pool of rated games whose players are mostly nowhere near each other.

    Every game carries ratings, so the view selects all of them; only a handful
    survives the rating gap. That is the shape a cap alone cannot rule out,
    because how many games are matched is a property of the pool rather than of
    configuration, and a reference of a handful is not empty either.
    """

    return _freeze(
        write_corpus,
        tmp_path / "mismatched",
        [
            normalized_row(
                index,
                split="test",
                plies=4 + (index % 5) * 2,
                ratings=(
                    (1200 + index * 100, 1200 + index * 100)
                    if index <= 6
                    else (1100, 2100)
                ),
                result=("1-0", "0-1", "1/2-1/2")[index % 3],
            )
            for index in range(1, 61)
        ],
        pool_id="fixture-mismatched",
    )


@pytest.fixture
def small_bandwidth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the declared bandwidth so a fixture reference can support it.

    The declared value is chosen from tens of thousands of real games and a
    fixture cannot hold that many. Shrinking it here keeps these tests about the
    wiring; the declared constant itself is asserted separately and exercised by
    a real run.
    """

    for quantity in ComparedQuantity:
        monkeypatch.setitem(DECLARED_NEIGHBOURS, quantity, 4)


def _compared(pool: Path, **overrides: Any) -> ResolvedConfig[RolloutBenchmarkConfig]:
    """Return a suite that compares its generated play against human play.

    Four games per position rather than the one the matrix tests play. A cell's
    games are one stream each and a stream is the unit a comparison resamples,
    so a suite below ``MINIMUM_BOOTSTRAP_STREAMS`` states no spread — however
    many ratings it played those streams at.
    """

    reference = {"enabled": True, "resamples": 8, **overrides.pop("reference", {})}
    generation = {"games_per_position": 4, **overrides.pop("generation", {})}
    return _config(
        pool=str(pool), reference=reference, generation=generation, **overrides
    )


def test_the_declared_bandwidth_is_one_frozen_value() -> None:
    """Re-selecting per run would measure two checkpoints differently.

    Pinned as a test because the constant is a benchmark-version commitment
    rather than a tuning knob: changing it ends every curve series, so it should
    not be possible to change quietly.
    """

    assert set(DECLARED_NEIGHBOURS) == set(ComparedQuantity)
    assert set(DECLARED_NEIGHBOURS.values()) == {1024}
    for quantity in ComparedQuantity:
        spec = curve_spec(quantity, (1200, 1800))
        assert spec.neighbours == 1024
        assert spec.quantity is quantity.kind
        # The evaluation points are the ratings played, not a declared grid.
        assert spec.grid == (1200.0, 1800.0)


def test_a_grid_point_needs_a_bandwidth_of_its_own() -> None:
    """The floor is one full bandwidth per point, and counts points once.

    Below it the neighbourhoods cannot be disjoint however the ratings fall, so
    the curve reports fewer independent points than it plots. Repeated ratings
    are one evaluation point rather than two, because the grid is a set.
    """

    assert minimum_reference_games((1200, 1800), 1024) == 2048
    assert minimum_reference_games((1100, 1300, 1500, 1700, 1900, 2100), 1024) == 6144
    assert minimum_reference_games((1200, 1200, 1800), 1024) == 2048


def test_a_reference_below_the_bandwidth_is_rejected_before_a_sweep_starts() -> None:
    """A sweep that cannot resolve its own grid should fail in the first second.

    The reference size is the neighbour-count bandwidth's radius rather than a
    sample size, so a cap this low does not make the reading noisier: it makes
    every grid point the same neighbourhood. Rejected on the configuration so a
    suite plan catches it, rather than an hour of generation later.
    """

    with pytest.raises(ValidationError, match="below the 2048 game"):
        _compared(
            Path("pool"),
            grid={"target_ratings": (1200, 1800)},
            reference={
                "view": {"name": "rollout-reference", "maximum_games": 2000},
            },
        )


def test_a_reference_the_rating_gap_guts_fails_before_a_game_is_played(
    mismatched_reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """The declared cap cannot promise what survives the rating-gap filter.

    That fraction is a property of the pool's rating composition, so a cap that
    clears the floor can still leave a reference that does not. It is read
    before the first game so a suite pays a pool pass rather than a whole
    generation matrix to learn it.
    """

    runner = TrajectoryRunner()

    with pytest.raises(RolloutBenchmarkError, match="usable human game"):
        _run(
            _compared(
                mismatched_reference_pool,
                grid={"target_ratings": (1200, 1800)},
            ),
            runner=runner,
        )

    assert runner.calls == 0


def test_the_reference_is_read_at_one_speed_class(
    mixed_speed_reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """The harness plays untimed, so the class slices the reference.

    Length and result are strong functions of the clock, so a reference drawn
    from a mixed pool would report its composition as a distance.
    """

    result = _run(
        _compared(
            mixed_speed_reference_pool,
            grid={"target_ratings": (1200, 1800)},
            reference={
                "view": {
                    "name": "rollout-reference",
                    "require_ratings": True,
                    "speed": "blitz",
                }
            },
        )
    )

    assert result.reference_view is not None
    assert result.reference_view.selected_games == 30
    assert result.reference_view.excluded_games == {"speed_mismatch": 30}
    assert result.readings[0].human_games == 30


def test_a_reference_class_the_pool_holds_none_of_fails_in_the_pool_pass(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A class with no reference game has nothing for a distance to be over.

    Read before the matrix is played, so the class is answered by a pool pass
    rather than by an hour of generation with nothing to compare it against.
    """

    runner = TrajectoryRunner()

    with pytest.raises(RolloutBenchmarkError, match="speed_mismatch"):
        _run(
            _compared(
                reference_pool,
                grid={"target_ratings": (1200, 1800)},
                reference={
                    "view": {
                        "name": "rollout-reference",
                        "require_ratings": True,
                        "speed": "bullet",
                    }
                },
            ),
            runner=runner,
        )

    assert runner.calls == 0


def test_the_human_reference_scopes_the_curve_series(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """Two references are two smoothings, so two quantities rather than two draws.

    The bandwidth is a neighbour count, so the reference decides the rating span
    every neighbourhood covers. Without this in identity a checkpoint read
    against a small reference would be plotted against one read against a large
    one, and the difference between the smoothings would render as movement.
    """

    workloads = []
    for maximum_games in (24, 48):
        result = _run(
            _compared(
                reference_pool,
                grid={"target_ratings": (1200, 1800)},
                reference={
                    "view": {
                        "name": "rollout-reference",
                        "require_ratings": True,
                        "maximum_games": maximum_games,
                    },
                },
            )
        )
        workloads.append(result.readings[0].execution.workload_sha256)

    assert workloads[0] != workloads[1]


def test_the_declared_walk_shape_is_the_one_a_suite_runs() -> None:
    """The walk's depth and threshold scope its series, like the bandwidth.

    Pinned here because the suites above run a cheaper walk: a stub policy
    spreads its mass over every legal move, so the declared threshold explores a
    tree no trained checkpoint would. That keeps the tests affordable but leaves
    nothing else asserting what a real run walks.
    """

    declared = RepertoireWalkConfig()

    assert declared.enabled
    assert declared.plies == 8
    assert declared.threshold == 0.001


def test_every_generated_play_metric_is_reported_by_the_benchmark(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A registered metric no benchmark writes is a series that never starts."""

    result = _run(
        _compared(
            reference_pool,
            grid={"target_ratings": (1200, 1800)},
            # Both terminal actions on, as the shipped selection has them: the
            # guardrails are counted over enabled actions, so a suite that
            # offered none reports them as absent rather than as zero and this
            # coverage check would pass while two series never started.
            runtime=RuntimeConfig(resignation_enabled=True, draw_claim_enabled=True),
        )
    )

    reported = {
        item.metric for envelope in _readings(result) for item in envelope.measurements
    }
    registered = {
        metric.identifier
        for metric in registered_metrics(GENERATED_PLAY_FAMILY.identifier)
    } | {
        metric.identifier
        for metric in registered_metrics(DECISION_DECOMPOSITION_FAMILY.identifier)
    }
    assert reported == registered


def test_a_curve_reading_spans_the_rating_grid_rather_than_one_cell(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A curve's axis is the rating, so a single cell has no curve at all."""

    result = _run(
        _compared(
            reference_pool,
            grid={"target_ratings": (1200, 1600, 2000), "temperatures": (0.7, 1.0)},
        )
    )

    assert len(result.cells) == 6
    # One reading per arm and temperature, each spanning all three ratings.
    assert len(result.readings) == 2
    for reading in result.readings:
        assert reading.ratings == (1200, 1600, 2000)
        assert reading.model_games == 12
    assert {reading.temperature for reading in result.readings} == {0.7, 1.0}


def test_a_curve_reading_is_its_own_series_apart_from_the_cells(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A distance and a raw scalar are different quantities, so different series."""

    result = _run(_compared(reference_pool, grid={"target_ratings": (1200, 1800)}))

    (reading,) = result.readings
    cell_workloads = {cell.execution.workload_sha256 for cell in result.cells}
    assert reading.execution.workload_sha256 not in cell_workloads
    # The grid is the axis, so it replaces the single rating a cell declares.
    assert reading.execution.workload["target_ratings"] == [1200, 1800]
    assert "target_rating" not in reading.execution.workload


def test_the_declared_curve_shape_scopes_the_comparison_series(
    reference_pool: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two bandwidths estimate different quantities, however alike they look.

    This is why the bandwidth is declared and frozen rather than configured: if
    it did not scope the series, a checkpoint measured at one smoothing would be
    plotted against one measured at another.
    """

    workloads = []
    for neighbours in (4, 6):
        for quantity in ComparedQuantity:
            monkeypatch.setitem(DECLARED_NEIGHBOURS, quantity, neighbours)
        result = _run(_compared(reference_pool, grid={"target_ratings": (1200, 1800)}))
        workloads.append(result.readings[0].execution.workload_sha256)

    assert workloads[0] != workloads[1]


def test_a_temperature_change_starts_a_new_curve_series(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """Temperature is a separate dial, not a point on the rating axis."""

    result = _run(
        _compared(
            reference_pool,
            grid={"target_ratings": (1200, 1800), "temperatures": (0.5, 1.0)},
        )
    )

    assert len({r.execution.workload_sha256 for r in result.readings}) == 2


#: Quantities every game has, whatever it played. The opening-derived ones do
#: not belong here: a game that stopped on a waypoint made no repertoire choice
#: and one the book never named has no depth, so both sides drop those games.
_UNIVERSAL_QUANTITIES = (
    ComparedQuantity.GAME_LENGTH,
    ComparedQuantity.RESULT,
    ComparedQuantity.REPETITION,
    ComparedQuantity.CYCLE,
    ComparedQuantity.WAYPOINT,
    ComparedQuantity.MOVE_DIVERSITY,
)


def test_every_compared_quantity_is_measured_against_the_human_reference(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A quantity with no comparison is a claim about human-likeness never made."""

    result = _run(_compared(reference_pool, grid={"target_ratings": (1200, 1800)}))

    (reading,) = result.readings
    assert set(reading.comparisons) | set(reading.unavailable) == set(ComparedQuantity)
    for quantity, comparison in reading.comparisons.items():
        assert comparison.spec.quantity is quantity.kind
        assert comparison.human_games <= reading.human_games
        assert comparison.model_games <= reading.model_games
        if quantity in _UNIVERSAL_QUANTITIES:
            assert comparison.human_games == reading.human_games
            assert comparison.model_games == reading.model_games


def test_a_distance_carries_the_floor_it_has_to_clear(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A distance shown without its floor invites reading noise as a finding.

    The floor is evaluation noise rather than data-sampling noise, because what
    it qualifies is a delta between two checkpoints measured against the same
    human reference: re-measuring means generating another draw of games.
    """

    result = _run(_compared(reference_pool, grid={"target_ratings": (1200, 1800)}))

    envelope = _curve_envelope(result)
    for quantity in ComparedQuantity:
        conditional = envelope.measurement(
            GENERATED_PLAY_CONDITIONAL_DISTANCE[quantity.value].identifier
        )
        assert conditional is not None
        assert conditional.value >= 0.0
        assert conditional.dispersion is not None
        assert conditional.dispersion.bound >= conditional.dispersion.value


def test_a_greedy_temperature_is_measured_but_not_compared_to_humans(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """Greedy play is a point mass, and a distance against one says nothing.

    Both seats greedy means one game per position, so the model side carries a
    single category per rating while the human side carries a distribution. The
    distance between them is one minus the human mass of that category whatever
    the model plays, which moves with how popular an opening is rather than with
    how well it was chosen. The cells still record what greedy played.
    """

    result = _run(
        _compared(
            reference_pool,
            grid={"target_ratings": (1200, 1800), "temperatures": (0.0, 1.0)},
            generation={"swap_colors": True},
        )
    )

    assert [reading.temperature for reading in result.readings] == [1.0]
    with pytest.raises(RolloutBenchmarkError, match="no curve reading"):
        result.reading(RolloutArm.STANDARD_START, 0.0)
    greedy_cells = [cell for cell in result.cells if cell.temperature == 0.0]
    assert len(greedy_cells) == 2
    assert all(cell.distribution.games for cell in greedy_cells)


def test_the_seeds_re_measure_each_distance_independently(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """The bootstrap floor is an estimate; the seeds are the thing itself.

    Recording both is what makes the floor checkable. A bootstrap can only
    reshuffle the games one run produced, so it can understate the spread a
    genuinely fresh draw would show.
    """

    result = _run(
        _compared(
            reference_pool,
            grid={"target_ratings": (1200, 1800), "seeds": (0, 1, 2)},
        )
    )

    (reading,) = result.readings
    assert set(reading.seed_spread) == set(ComparedQuantity)
    for spread in reading.seed_spread.values():
        assert [seed for seed, _ in spread.distances] == [0, 1, 2]
        # A floor exactly where the seeds moved. Three that agreed observed that
        # this draw of them could not move the distance, which is not a spread
        # of zero for every later delta to clear.
        moved = len({value for _, value in spread.distances}) > 1
        assert (spread.floor is not None) == moved
    # A stub policy saturates several of these quantities whatever the seed, so
    # the check above passes vacuously if the play ever shrinks to where none of
    # them move.
    assert any(spread.floor is not None for spread in reading.seed_spread.values())


def test_the_committed_floor_is_the_bootstrap_rather_than_the_seed_spread(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A floor has to be estimated at the sample size the reading was taken at.

    Each seed plays a fraction of the suite's games, so the spread across seeds
    measures the noise of a much smaller reading and runs roughly the square
    root of the seed count too wide. The bootstrap resamples the games this
    reading actually generated, which is the right size, so it is what gets
    committed and the seeds stay a diagnostic.
    """

    result = _run(
        _compared(
            reference_pool,
            grid={"target_ratings": (1200, 1800), "seeds": (0, 1, 2)},
        )
    )

    (reading,) = result.readings
    envelope = _curve_envelope(result)
    for quantity, comparison in reading.comparisons.items():
        item = envelope.measurement(
            GENERATED_PLAY_CONDITIONAL_DISTANCE[quantity.value].identifier
        )
        assert item is not None and item.dispersion is not None
        assert comparison.dispersions is not None
        assert item.dispersion == comparison.dispersions.conditional


def test_a_reading_reports_how_far_its_smoother_actually_reached(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """The declared neighbour count says nothing about a particular reading.

    It is the same number whatever the reference holds. What decides whether
    the grid resolves the points it plots is the rating span those neighbours
    occupy, so that is what the reading prints — the widest across quantities,
    because a quantity some games lack reaches furthest for its neighbours.
    """

    from anthro_chess.interfaces.cli import _render_rollout

    result = _run(_compared(reference_pool, grid={"target_ratings": (1200, 1800)}))

    (reading,) = result.readings
    widest = [
        max(comparison.points[index].bandwidth for comparison in comparisons)
        for comparisons in [list(reading.comparisons.values())]
        for index in range(len(reading.ratings))
    ]
    assert len(widest) == 2
    line = f"  bandwidth      reaches ±{widest[0]:.0f} ±{widest[1]:.0f} rating points"
    assert line in _render_rollout(result)


def _comparison_table(rendered: str) -> tuple[str, dict[str, str]]:
    """Return the comparison table's heading line and its rows by quantity."""

    lines = rendered.splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.strip().startswith("quantity ")
    )
    rows = {}
    for line in lines[start + 1 :]:
        if line.strip().startswith("null:"):
            break
        rows[line.split()[0]] = line
    return lines[start], rows


def _dashed(value: float | None) -> str:
    """Return the cell the table prints for an optional qualifier."""

    return "-" if value is None else f"{value:.4f}"


def test_the_comparison_table_qualifies_each_arm_with_its_own_numbers(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A qualifier beside the wrong distance is read as if it belonged to it.

    The first full suite reading drew a wrong conclusion from this table with
    the source available: the conditional delta floor sat next to the pooled
    distance, and the null level the verdict is actually computed against was
    not in the output at all. So every arm now carries its own distance, null,
    floor, and seed spread, and nothing is shared between the two.
    """

    from anthro_chess.interfaces.cli import _render_rollout

    result = _run(
        _compared(
            reference_pool,
            grid={"target_ratings": (1200, 1800), "seeds": (0, 1, 2)},
        )
    )

    _, rows = _comparison_table(_render_rollout(result))
    (reading,) = result.readings
    assert set(rows) == {quantity.value for quantity in reading.comparisons}
    for quantity, comparison in reading.comparisons.items():
        assert comparison.references is not None
        assert comparison.dispersions is not None
        spread = reading.seed_spread[quantity]
        assert rows[quantity.value].split() == [
            quantity.value,
            f"{comparison.conditional_distance:.4f}",
            f"{comparison.references.conditional:.4f}",
            f"{comparison.dispersions.conditional_floor:.4f}",
            _dashed(spread.floor),
            f"{comparison.pooled_distance:.4f}",
            f"{comparison.references.pooled:.4f}",
            f"{comparison.dispersions.pooled_floor:.4f}",
            _dashed(spread.pooled_floor),
            comparison.response.value,
        ]


def test_the_comparison_table_lines_its_rows_up_under_its_own_headings(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A heading its values do not sit under names the wrong column.

    The table that produced the misreading was also misaligned: its heading row
    and its value rows were built from different widths, so the last four
    headings stood over a neighbouring column's numbers. The quantity column is
    sized from the names present for the same reason — the longest runs to
    twenty-odd characters and a fixed column pushes that row out on its own.
    """

    from anthro_chess.interfaces.cli import _render_rollout

    result = _run(_compared(reference_pool, grid={"target_ratings": (1200, 1800)}))

    heading, rows = _comparison_table(_render_rollout(result))
    columns = [match.end() for match in re.finditer(r"\S+", heading)]
    assert rows
    for row in rows.values():
        # The first column is left-aligned, so its values share a start rather
        # than an end; every column right of it is compared where it ends. The
        # verdict is checked on its own because "reads as" is two heading words
        # over one value.
        ends = [match.end() for match in re.finditer(r"\S+", row)]
        assert ends[1:-1] == columns[1:-2]
        assert ends[-1] == columns[-1]
        assert row.index(row.split()[0]) == heading.index("quantity")
        assert max(len(row), len(heading)) <= 120


def test_two_seeds_report_their_distances_without_inventing_a_floor(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A floor from two replicates would license almost any delta."""

    result = _run(
        _compared(
            reference_pool,
            grid={"target_ratings": (1200, 1800), "seeds": (0, 1)},
        )
    )

    (reading,) = result.readings
    for spread in reading.seed_spread.values():
        assert len(spread.distances) == 2
        assert spread.floor is None


def test_a_model_matching_the_reference_reads_closer_than_one_that_does_not(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """The distance has to actually rank two checkpoints, or it says nothing.

    This is the property that makes the family directional: unlike a draw rate,
    a smaller distance to matched human play is unambiguously better, and that
    only holds if a worse model measures further away.
    """

    close = _run(
        _compared(
            reference_pool,
            grid={"target_ratings": (1200, 1800)},
            generation={"maximum_generated_plies": 8},
        )
    )
    # A model stopped after two plies produces games nothing like the human
    # ones, which have to read as further away on length.
    far = _run(
        _compared(
            reference_pool,
            grid={"target_ratings": (1200, 1800)},
            generation={"maximum_generated_plies": 2},
        )
    )

    metric = ComparedQuantity.GAME_LENGTH
    assert (
        far.readings[0].comparisons[metric].pooled_distance
        > close.readings[0].comparisons[metric].pooled_distance
    )


def test_the_comparison_needs_a_pool_to_read_its_reference_from() -> None:
    """A comparison with no human side is a verdict with nothing behind it."""

    with pytest.raises(ValidationError, match="needs pool"):
        _config(reference={"enabled": True})


def test_curve_points_stay_in_the_detail_tier(
    tmp_path: Path,
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """Points are data a later chart queries, not a number for the summary."""

    result = _run(
        _compared(reference_pool, grid={"target_ratings": (1200, 1800)}),
        store=ResultsStore(tmp_path / "results"),
        detail=DetailStore(tmp_path / "detail"),
    )

    envelope = _curve_envelope(result)
    assert envelope.detail is not None
    payload = json.loads(
        next(path for path in result.detail_paths if "curves" in path.name).read_text()
    )
    assert set(payload["comparisons"]) == {q.value for q in ComparedQuantity}
    length = payload["comparisons"][ComparedQuantity.GAME_LENGTH.value]
    assert [point["rating"] for point in length["points"]] == [1200.0, 1800.0]
    assert length["response"] in {
        "matches",
        "average_human",
        "divergent_response",
        "mismatch",
        "unknown",
    }
    assert payload["reference"]["games"] > 0


def test_the_reference_excludes_lopsided_games_and_says_so(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A mismatch belongs to neither player's rating, so it is not averaged in."""

    result = _run(
        _compared(
            reference_pool,
            grid={"target_ratings": (1200, 1800)},
            reference={"enabled": True, "resamples": 8, "maximum_rating_gap": 0},
        )
    )

    assert result.reference is not None
    assert result.reference.games
    record = result.reference.as_record()
    assert record["rating_range"][0] <= record["rating_range"][1]


def test_the_matrix_produces_one_result_per_cell() -> None:
    """Each cell is its own measurement, so each gets its own envelope."""

    result = _run(
        _config(
            grid={
                "target_ratings": (1000, 1800),
                "temperatures": (0.5, 1.0),
                "seeds": (0,),
            }
        )
    )

    assert len(result.cells) == 4
    assert len(_readings(result)) == 4
    assert {
        (cell.arm, cell.target_rating, cell.temperature) for cell in result.cells
    } == {
        (RolloutArm.STANDARD_START, 1000, 0.5),
        (RolloutArm.STANDARD_START, 1000, 1.0),
        (RolloutArm.STANDARD_START, 1800, 0.5),
        (RolloutArm.STANDARD_START, 1800, 1.0),
    }


def test_a_dial_change_starts_a_new_series_and_a_seed_change_does_not() -> None:
    """Rating and temperature decide what was measured; seeds decide precision.

    This is the whole comparability contract for this benchmark. If seeds
    entered identity, adding replicates to narrow a noisy metric would silently
    abandon its history; if the dials did not, two temperatures would be plotted
    as though one had improved on the other.
    """

    graded = _run(
        _config(
            grid={
                "target_ratings": (1000, 1800),
                "temperatures": (0.5, 1.0),
                "seeds": (0,),
            }
        )
    )
    fingerprints = {
        cell.execution.workload_sha256: (cell.target_rating, cell.temperature)
        for cell in graded.cells
    }
    assert len(fingerprints) == 4

    one_seed = _run(_config(grid={"seeds": (0,)}))
    three_seeds = _run(_config(grid={"seeds": (0, 1, 2)}))
    assert (
        one_seed.cells[0].execution.workload_sha256
        == three_seeds.cells[0].execution.workload_sha256
    )
    assert three_seeds.games > one_seed.games


def test_a_ply_limit_change_starts_a_new_series() -> None:
    """A game cut off at six plies is not the same quantity as one at sixty."""

    short = _run(_config(generation={"maximum_generated_plies": 6}))
    longer = _run(_config(generation={"maximum_generated_plies": 8}))

    assert (
        short.cells[0].execution.workload_sha256
        != longer.cells[0].execution.workload_sha256
    )


def test_concurrency_does_not_start_a_new_series() -> None:
    """Batching is a throughput setting, so it must not end a series."""

    sequential = _run(_config(generation={"maximum_generated_plies": 6}))
    batched = _run(_config(generation={"maximum_generated_plies": 6, "concurrency": 4}))

    assert (
        sequential.cells[0].execution.workload_sha256
        == batched.cells[0].execution.workload_sha256
    )


def test_explicit_seeds_reproduce_the_same_games() -> None:
    """A recorded seed has to reproduce its suite, or nothing else is checkable."""

    first = _run(_config(grid={"seeds": (0, 1)}))
    second = _run(_config(grid={"seeds": (0, 1)}))

    assert [record.game_id for record in first.cells[0].records] == [
        record.game_id for record in second.cells[0].records
    ]
    assert first.cells[0].distribution.as_record() == (
        second.cells[0].distribution.as_record()
    )


def test_a_different_seed_produces_a_different_suite() -> None:
    """Reproducibility must not come from the seed being ignored."""

    baseline = _run(_config(grid={"seeds": (0,)}, detail={"retain_games": True}))
    other = _run(_config(grid={"seeds": (7,)}, detail={"retain_games": True}))

    assert [record.game_id for record in baseline.cells[0].records] != [
        record.game_id for record in other.cells[0].records
    ]


def test_seeds_are_kept_apart_so_their_spread_stays_readable() -> None:
    """Pooling the cell must not lose the per-seed readings behind it.

    The spread across seeds is this benchmark's evaluation noise, and it is
    unrecoverable from a pooled number alone.
    """

    result = _run(_config(grid={"seeds": (0, 1, 2)}))

    cell = result.cells[0]
    assert [seed for seed, _ in cell.per_seed] == [0, 1, 2]
    assert cell.distribution.games == sum(
        distribution.games for _, distribution in cell.per_seed
    )


def test_the_ply_limit_is_reported_as_unfinished_rather_than_as_a_result() -> None:
    """An adjudicated game has no result, so it must not read as a draw."""

    result = _run(_config(generation={"maximum_generated_plies": 2}))

    cell = result.cells[0]
    assert cell.distribution.termination_counts == {GameTermination.PLY_LIMIT.value: 1}
    assert cell.distribution.result_counts == {"*": 1}
    envelope = result.envelopes[0]
    unfinished = envelope.measurement(GENERATED_PLAY_UNFINISHED_GAME_RATE.identifier)
    assert unfinished is not None
    assert unfinished.value == pytest.approx(1.0)
    # With nothing finished there is no white score to report, and inventing a
    # half would read as a balanced suite.
    white = envelope.measurement(GENERATED_PLAY_WHITE_SCORE.identifier)
    assert white is not None
    assert white.value == pytest.approx(0.0)


def test_a_metric_averaged_over_a_subset_reports_that_subset_as_its_sample() -> None:
    """Sample size is per metric, because three of them average over a subset.

    Reporting the suite's game count for a cycle depth estimated from the games
    that repeated would overstate its precision to every reader, the noise-floor
    layer included. An empty subset reports no sample size rather than one.
    """

    result = _run(
        _config(generation={"games_per_position": 2, "maximum_generated_plies": 6})
    )

    (envelope,) = _readings(result)
    distribution = result.cells[0].distribution
    assert distribution.games == 2
    # Nothing repeated and nothing finished inside six plies, so both subsets
    # are empty and neither reports a sample size it does not have.
    assert distribution.repeated_games == 0
    assert _sample(envelope, GENERATED_PLAY_MEAN_CYCLE_PLY_FRACTION) is None
    assert _sample(envelope, GENERATED_PLAY_DECISIVE_GAME_RATE) is None
    # A rate over every game keeps the suite's count.
    assert _sample(envelope, GENERATED_PLAY_UNFINISHED_GAME_RATE) == 2
    assert _sample(envelope, GENERATED_PLAY_MEAN_GAME_PLIES) == 2


def test_a_finished_game_counts_toward_the_rates_computed_over_results() -> None:
    """The denominator for a decisive rate is the games that produced a result."""

    result = _run(
        _config(runtime=RuntimeConfig(resignation_enabled=True)),
        runner=ResigningRunner(),
    )

    (envelope,) = _readings(result)
    assert _sample(envelope, GENERATED_PLAY_DECISIVE_GAME_RATE) == 1
    decisive = envelope.measurement(GENERATED_PLAY_DECISIVE_GAME_RATE.identifier)
    assert decisive is not None
    # A resignation is a decided game, so it is decisive rather than unfinished.
    assert decisive.value == pytest.approx(1.0)


def test_a_resigning_model_is_recorded_as_resigning() -> None:
    """Resignation is a model ending, so it needs its own visible rate."""

    result = _run(
        _config(runtime=RuntimeConfig(resignation_enabled=True)),
        runner=ResigningRunner(),
    )

    cell = result.cells[0]
    assert cell.distribution.termination_counts == {
        GameTermination.RESIGNATION.value: 1
    }
    rate = result.envelopes[0].measurement(GENERATED_PLAY_RESIGNATION_RATE.identifier)
    assert rate is not None
    assert rate.value == pytest.approx(1.0)


def test_resignation_enablement_starts_a_new_series() -> None:
    """Whether the model may resign changes what every ending means."""

    disabled = _run(_config(runtime=RuntimeConfig(resignation_enabled=False)))
    enabled = _run(_config(runtime=RuntimeConfig(resignation_enabled=True)))

    assert (
        disabled.cells[0].execution.workload_sha256
        != enabled.cells[0].execution.workload_sha256
    )


def test_temperature_zero_collapses_the_suite_to_one_trajectory() -> None:
    """The diversity metric has to be able to see a collapsed policy.

    Greedy selection makes every game from one position identical, which is the
    failure mode a single-seed rollout cannot distinguish from stable behavior.
    The cell plays the color assignments it declared rather than the replicates
    it configured, so the collapse is read against the smallest sample that can
    still show it.
    """

    result = _run(
        _config(
            grid={
                "target_ratings": (1200,),
                "temperatures": (0.0,),
                "seeds": (0, 1, 2),
            },
            generation={
                "games_per_position": 2,
                "maximum_generated_plies": 6,
                "swap_colors": True,
            },
        )
    )

    fraction = result.envelopes[0].measurement(
        GENERATED_PLAY_DISTINCT_GAME_FRACTION.identifier
    )
    assert fraction is not None
    assert result.cells[0].distribution.games == 2
    assert fraction.value == pytest.approx(0.5)


def test_a_greedy_cell_plays_one_replicate_of_the_game_it_can_play() -> None:
    """Seeds and games per position buy no precision over a point mass.

    Both are sample counts, and at temperature zero every game they buy is the
    same game, so the cell plays one replicate and reports the sample it
    realized rather than the sample it was configured for.
    """

    result = _run(
        _config(
            grid={
                "target_ratings": (1200,),
                "temperatures": (0.0, 1.0),
                "seeds": (0, 1, 2),
            },
            generation={"games_per_position": 2, "maximum_generated_plies": 6},
        )
    )

    greedy = result.cell(RolloutArm.STANDARD_START, 1200, 0.0)
    sampled = result.cell(RolloutArm.STANDARD_START, 1200, 1.0)
    assert greedy.seeds == (0,)
    assert greedy.distribution.games == 1
    assert sampled.seeds == (0, 1, 2)
    assert sampled.distribution.games == 6


def test_a_collapsed_cell_reads_as_the_suite_it_collapsed_to() -> None:
    """Collapsing changes what a greedy cell cost, not what it read."""

    collapsed = _run(
        _config(
            grid={"temperatures": (0.0,), "seeds": (0, 1, 2)},
            generation={"games_per_position": 2},
        )
    )
    single = _run(_config(grid={"temperatures": (0.0,), "seeds": (0,)}))

    assert collapsed.cells[0].distribution.as_record() == (
        single.cells[0].distribution.as_record()
    )


def test_multiple_games_aggregate_into_one_cell_reading() -> None:
    """A cell's reading is over every game it played, not the last one."""

    result = _run(
        _config(
            generation={
                "games_per_position": 3,
                "maximum_generated_plies": 6,
                "swap_colors": True,
            }
        )
    )

    cell = result.cells[0]
    assert cell.distribution.games == 6
    mean = result.envelopes[0].measurement(GENERATED_PLAY_MEAN_GAME_PLIES.identifier)
    assert mean is not None
    assert mean.value == pytest.approx(cell.distribution.mean_ply_count)
    assert mean.sample_size == 6


def test_prefix_continuations_start_from_the_human_games(
    pool: Path,
    fixture_game_id: Callable[[int], int],
) -> None:
    """A prefix arm has to actually replay the pool's openings."""

    result = _run(
        _config(
            arms=(RolloutArm.HUMAN_PREFIX,),
            pool=str(pool),
            prefix={"plies": 4},
        )
    )

    cell = result.cells[0]
    assert cell.arm is RolloutArm.HUMAN_PREFIX
    assert cell.positions == 2
    for record in cell.records:
        assert record.prefix_plies == 4
        assert record.source_game_id in {fixture_game_id(2), fixture_game_id(3)}
        board = chess.Board()
        for move_text in ("e2e4", "e7e5", "g1f3", "b8c6"):
            move = chess.Move.from_uci(move_text)
            assert record.action_ids[board.ply()] == encode_move(move)
            board.push(move)
        # The decisions start where the prefix stops, so a continuation is
        # measured on what the model added rather than on the human opening.
        assert record.decisions[0].ply_index == 4


def test_prefix_depth_and_pool_identity_scope_the_series(pool: Path) -> None:
    """Continuing a different opening, or a different depth, is a new quantity."""

    shallow = _run(
        _config(
            arms=(RolloutArm.HUMAN_PREFIX,),
            pool=str(pool),
            prefix={"plies": 4},
        )
    )
    deeper = _run(
        _config(
            arms=(RolloutArm.HUMAN_PREFIX,),
            pool=str(pool),
            prefix={"plies": 6},
        )
    )

    assert (
        shallow.cells[0].execution.workload_sha256
        != deeper.cells[0].execution.workload_sha256
    )
    workload = shallow.cells[0].execution.workload["positions"]
    assert workload["prefix_plies"] == 4
    assert workload["pool_id"] == "fixture-test"
    assert workload["game_ids_sha256"]


def test_the_two_arms_are_separate_series_over_one_run(pool: Path) -> None:
    """One run, two position sources, two series rather than a blended one."""

    result = _run(
        _config(
            arms=(RolloutArm.STANDARD_START, RolloutArm.HUMAN_PREFIX),
            pool=str(pool),
            prefix={"plies": 4},
        )
    )

    assert {cell.arm for cell in result.cells} == {
        RolloutArm.STANDARD_START,
        RolloutArm.HUMAN_PREFIX,
    }
    assert len({cell.execution.workload_sha256 for cell in result.cells}) == 2


def test_the_prefix_arm_records_its_human_games_as_provenance(pool: Path) -> None:
    """A reader has to be able to tell which openings a rollout continued.

    The digest is provenance rather than identity: a generated-play metric
    declares no projection, so this never enters a fingerprint.
    """

    result = _run(
        _config(
            arms=(RolloutArm.STANDARD_START, RolloutArm.HUMAN_PREFIX),
            pool=str(pool),
            prefix={"plies": 4},
        )
    )

    assert result.dataset is not None
    assert result.dataset.selected_games == 2
    by_arm = {
        envelope.execution.workload["positions"]["kind"]: envelope
        for envelope in _readings(result)
        if envelope.execution is not None
    }
    assert by_arm[RolloutArm.HUMAN_PREFIX.value].data is not None
    assert by_arm[RolloutArm.STANDARD_START.value].data is None
    for envelope in result.envelopes:
        for item in envelope.measurements:
            assert item.fingerprint == envelope.expected_fingerprint(item.metric)


def test_the_prefix_view_excludes_games_shorter_than_the_prefix(pool: Path) -> None:
    """A truncated prefix is a different measurement, so it is excluded loudly."""

    result = _run(
        _config(
            arms=(RolloutArm.HUMAN_PREFIX,),
            pool=str(pool),
            prefix={"plies": 10},
        )
    )

    assert result.view is not None
    assert result.view.selected_games == 1
    assert result.view.excluded_games == {"shorter_than_prefix": 1}


def test_a_prefix_deeper_than_every_game_fails_rather_than_measuring_nothing(
    pool: Path,
) -> None:
    result = _config(
        arms=(RolloutArm.HUMAN_PREFIX,),
        pool=str(pool),
        prefix={"plies": 40},
    )

    with pytest.raises(RolloutBenchmarkError, match="selected no games"):
        _run(result)


def test_the_prefix_arm_needs_a_pool() -> None:
    """A missing pool is a configuration error, not a silently skipped arm."""

    with pytest.raises(ValidationError, match="needs pool"):
        _config(arms=(RolloutArm.HUMAN_PREFIX,))


def test_a_pinned_generation_without_a_pool_is_a_configuration_error() -> None:
    """A pin that protects nothing reads exactly like one that protects."""

    with pytest.raises(ValidationError, match="does not read"):
        _config(expected_pool_game_ids_sha256="0" * 64)


def test_a_pool_this_suite_is_not_defined_over_is_refused(pool: Path) -> None:
    """The generation a selection pins reaches the loader from here."""

    config = _config(
        arms=(RolloutArm.HUMAN_PREFIX,),
        pool=str(pool),
        expected_pool_game_ids_sha256="0" * 64,
    )

    with pytest.raises(RolloutBenchmarkError, match="expected 0{64}"):
        _run(config)


def test_games_stay_in_the_detail_tier(tmp_path: Path) -> None:
    """Records are bulk diagnostics: referenced from the summary, never in it."""

    store = ResultsStore(tmp_path / "results")
    detail = DetailStore(tmp_path / "detail")

    result = _run(
        _config(
            generation={"games_per_position": 2, "maximum_generated_plies": 6},
            detail={"retain_games": True},
        ),
        store=store,
        detail=detail,
    )

    (envelope,) = _readings(result)
    # The rollout's own record, and one saying what the invocation cost.
    assert {item.kind for item in result.envelopes} == {
        ROLLOUT_KIND,
        BENCHMARK_COST_KIND,
    }
    assert envelope.detail is not None
    assert envelope.execution is not None
    (path,) = result.detail_paths
    payload = json.loads(path.read_text())
    assert len(payload["games_detail"]) == 2
    assert payload["workload_sha256"] == envelope.execution.workload_sha256
    # The committed record carries scalars and a reference, nothing bulk.
    committed = json.loads(result.recorded_paths[0].read_text())
    assert "games_detail" not in json.dumps(committed)


def test_retaining_games_can_be_turned_off(tmp_path: Path) -> None:
    """A long suite must be runnable without keeping every game on disk."""

    detail = DetailStore(tmp_path / "detail")

    result = _run(
        _config(detail={"retain_games": False}),
        store=ResultsStore(tmp_path / "results"),
        detail=detail,
    )

    assert result.cells[0].records == ()
    payload = json.loads(result.detail_paths[0].read_text())
    assert payload["games_detail"] == []
    assert payload["distribution"]["games"] == 1


def test_recording_can_be_skipped_entirely() -> None:
    """An exploratory reading is real but does not belong in the history."""

    result = _run(_config())

    assert result.envelopes
    assert result.recorded_paths == ()
    assert result.detail_paths == ()


def test_a_supplied_runner_needs_a_checkpoint_reference() -> None:
    """A result with no checkpoint identity cannot be compared to anything."""

    with pytest.raises(RolloutBenchmarkError, match="checkpoint reference"):
        _run(_config(), runner=TrajectoryRunner(), checkpoint=None)


def test_a_grid_axis_cannot_be_empty_or_repeat_a_value() -> None:
    with pytest.raises(ValidationError, match="at least one target_ratings"):
        _config(grid={"target_ratings": ()})
    with pytest.raises(ValidationError, match="must not repeat a seeds"):
        _config(grid={"seeds": (0, 0)})


def test_a_cell_reports_the_repertoire_apart_from_the_waypoint_rate(
    pool: Path,
) -> None:
    """Two readings out of one classification pass, kept separate on purpose."""

    result = _run(_config(arms=("standard-start",)))

    distribution = result.cells[0].distribution
    assert set(distribution.repertoire_counts) <= set(distribution.opening_counts)
    chosen = sum(distribution.repertoire_counts.values())
    assert (
        chosen + round(distribution.waypoint_game_rate * distribution.games)
        == distribution.games
    )


def test_a_cell_reports_book_depth_in_three_parts(pool: Path) -> None:
    """Raw depth alone cannot tell a deep line abandoned from an offbeat one."""

    distribution = _run(_config()).cells[0].distribution

    assert distribution.classified_games > 0
    assert distribution.mean_book_ply <= distribution.mean_available_ply
    assert distribution.mean_consumed_fraction == pytest.approx(
        distribution.mean_book_ply / distribution.mean_available_ply, rel=0.5
    )


def test_the_depth_readings_are_averaged_over_named_games_only(
    pool: Path,
) -> None:
    """An unnamed game has no depth, so counting it would dilute the average."""

    result = _run(_config(), store=None)
    envelope = next(
        envelope
        for envelope in result.envelopes
        if envelope.measurement(GENERATED_PLAY_MEAN_BOOK_PLY.identifier) is not None
    )
    distribution = result.cells[0].distribution

    assert _sample(envelope, GENERATED_PLAY_MEAN_BOOK_PLY) == (
        distribution.classified_games
    )
    assert _sample(envelope, GENERATED_PLAY_WAYPOINT_GAME_RATE) == distribution.games


def test_the_repertoire_curve_leaves_waypoint_games_out(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A game that chose nothing cannot contribute to a distribution of choices."""

    result = _run(_compared(reference_pool, grid={"target_ratings": (1200, 1800)}))

    (reading,) = result.readings
    repertoire = reading.comparisons[ComparedQuantity.REPERTOIRE]
    waypoints = reading.comparisons[ComparedQuantity.WAYPOINT]

    # The waypoint rate is measured over every game; the repertoire only over
    # the games that reached a destination.
    assert waypoints.human_games == reading.human_games
    assert repertoire.human_games <= reading.human_games
    assert repertoire.spec.quantity is CurveQuantity.CATEGORICAL


def test_the_repertoire_comparison_carries_its_category_drilldown(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """Uneven family granularity makes a bare delta unreadable."""

    result = _run(_compared(reference_pool, grid={"target_ratings": (1200, 1800)}))

    (reading,) = result.readings
    record = reading.comparisons[ComparedQuantity.REPERTOIRE].as_detail_record()
    categories = record["categories"]

    assert categories
    assert set(categories[0]) == {"category", "human", "model", "delta", "mass"}
    masses = [row["mass"] for row in categories]
    assert masses == sorted(masses, reverse=True)


def test_divergence_by_depth_never_arrives_without_its_floor(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """Category count grows with depth, so the raw distance does too."""

    result = _run(_compared(reference_pool, grid={"target_ratings": (1200, 1800)}))

    (reading,) = result.readings
    assert reading.divergence
    plies = [point.ply for point in reading.divergence]
    assert plies == sorted(plies)
    for point in reading.divergence:
        assert point.conditional_floor is not None
        assert point.pooled_floor is not None
        assert point.categories >= 1


def test_the_depth_sweep_commits_a_depth_and_keeps_its_curve_in_detail(
    reference_pool: Path,
    small_bandwidth: None,
    tmp_path: Path,
) -> None:
    """One location is committed; the curve behind it stays a diagnostic.

    Category count grows with depth, so a per-ply distance read against another
    checkpoint's compares two different numbers of categories. The depth the
    curve reaches half its excess at does not, which is why that is the half
    that travels.
    """

    store = ResultsStore(tmp_path / "results")
    detail = DetailStore(tmp_path / "detail")

    result = _run(
        _compared(reference_pool, grid={"target_ratings": (1200, 1800)}),
        store=store,
        detail=detail,
    )

    (sweep,) = [
        envelope
        for envelope in _readings(result)
        if any(
            item.metric == GENERATED_PLAY_DIVERGENCE_HALF_DEPTH.identifier
            for item in envelope.measurements
        )
    ]
    curves = _curve_envelope(result)
    assert sweep.execution is not None
    assert curves.execution is not None
    assert sweep.execution.workload_sha256 != curves.execution.workload_sha256
    assert [item.metric for item in sweep.measurements] == [
        GENERATED_PLAY_DIVERGENCE_HALF_DEPTH.identifier
    ]
    assert sweep.detail is not None
    payload = json.loads((detail.root / sweep.detail.path).read_text(encoding="utf-8"))
    assert payload["points"]
    assert all(point["conditional_null"] is not None for point in payload["points"])


def test_the_depth_sweep_can_be_switched_off(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    result = _run(
        _compared(
            reference_pool,
            grid={"target_ratings": (1200, 1800)},
            divergence={"enabled": False},
        )
    )

    (reading,) = result.readings
    assert reading.divergence == ()


def test_the_exact_walk_is_its_own_series_apart_from_the_curves(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """Changing the walk's depth must not end the sampled curve series."""

    result = _run(_compared(reference_pool, grid={"target_ratings": (1200, 1800)}))

    (reading,) = result.readings
    assert reading.exact is not None
    assert reading.exact.execution.workload_sha256 != reading.execution.workload_sha256
    assert reading.exact.execution.workload["walk_plies"] == reading.exact.plies


def test_the_exact_walk_reports_the_bound_its_pruning_implies(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A distance quoted without the pruned mass overstates its own precision."""

    result = _run(_compared(reference_pool, grid={"target_ratings": (1200, 1800)}))

    exact = result.readings[0].exact
    assert exact is not None
    assert [rating for rating, _ in exact.walks] == [1200, 1800]
    assert 0.0 <= exact.pruned_mass <= 1.0
    assert exact.pruned_mass == max(walk.pruned_mass for _, walk in exact.walks)
    assert exact.conditional_distance == pytest.approx(
        sum(value for _, value in exact.distances) / len(exact.distances)
    )


def test_the_exact_walk_records_its_own_measurements(
    reference_pool: Path,
    small_bandwidth: None,
    tmp_path: Path,
) -> None:
    store = ResultsStore(tmp_path / "results")
    detail = DetailStore(tmp_path / "detail")

    result = _run(
        _compared(reference_pool, grid={"target_ratings": (1200, 1800)}),
        store=store,
        detail=detail,
    )

    exact = result.readings[0].exact
    assert exact is not None
    envelope = next(
        envelope
        for envelope in result.envelopes
        if envelope.execution is not None
        and envelope.execution.workload_sha256 == exact.execution.workload_sha256
    )
    for metric in (
        GENERATED_PLAY_EXACT_REPERTOIRE_CONDITIONAL_DISTANCE,
        GENERATED_PLAY_EXACT_REPERTOIRE_POOLED_DISTANCE,
        GENERATED_PLAY_EXACT_REPERTOIRE_PRUNED_MASS,
        GENERATED_PLAY_EXACT_REPERTOIRE_WAYPOINT_MASS,
    ):
        found = envelope.measurement(metric.identifier)
        assert found is not None
        # Enumerated rather than drawn, so there is no game count behind it that
        # a precision estimate could use.
        assert found.sample_size is None
    assert envelope.detail is not None
    payload = json.loads(
        (detail.root / envelope.detail.path).read_text(encoding="utf-8")
    )
    assert len(payload["ratings"]) == 2


def test_the_exact_walk_is_skipped_on_the_prefix_arm(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """A prefix decided the opening before the model moved."""

    result = _run(
        _compared(
            reference_pool,
            arms=("human-prefix",),
            grid={"target_ratings": (1200, 1800)},
            prefix={"plies": 4, "view": {"name": "rollout-prefix", "maximum_games": 2}},
        )
    )

    (reading,) = result.readings
    assert reading.arm is RolloutArm.HUMAN_PREFIX
    assert reading.exact is None


def test_the_exact_walk_can_be_switched_off(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    result = _run(
        _compared(
            reference_pool,
            grid={"target_ratings": (1200, 1800)},
            walk={"enabled": False},
        )
    )

    assert result.readings[0].exact is None


def test_the_rendered_walk_names_the_bound_that_matters(
    reference_pool: Path,
    small_bandwidth: None,
) -> None:
    """The assumption-free bound alone would read as an unusable measurement."""

    from anthro_chess.interfaces.cli import _render_rollout

    result = _run(_compared(reference_pool, grid={"target_ratings": (1200, 1800)}))

    rendered = _render_rollout(result)

    assert "exact repertoire to" in rendered
    assert "uncommitted mass at most" in rendered
    assert "pruned in all" in rendered
    assert "reached ply" in rendered
    assert max(len(line) for line in rendered.splitlines()) <= 120


def test_a_parallel_matrix_reads_the_same_as_a_serial_one(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """Which process played a cell must not reach anything the reading takes.

    Every other test here supplies a runner, which holds the matrix to one
    process, so this is the only place the worker path runs at all. It loads a
    real checkpoint because that is what a worker loads for itself.
    """

    checkpoint = inference_run(tmp_path / "run", seed=11)
    model = ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu")

    def read(workers: int) -> RolloutBenchmarkResult:
        return cast(
            RolloutBenchmarkResult,
            run_benchmark(
                benchmark_registry()["rollout"],
                _config(
                    model=model,
                    workers=workers,
                    grid={"target_ratings": (1200, 1800), "temperatures": (1.0,)},
                ),
                runner=None,
                checkpoint=None,
            ),
        )

    serial = read(1)
    parallel = read(2)

    assert [cell.as_record() for cell in parallel.cells] == [
        cell.as_record() for cell in serial.cells
    ]


def test_a_worker_that_stops_fails_this_reading_rather_than_the_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inference_run: Callable[..., Path],
) -> None:
    """A broken pool raises `RuntimeError`, which no sweep converts to a failure.

    An initializer's own exception never reaches the caller either, so a worker
    that could not load its checkpoint arrives here the same way.
    """

    checkpoint = inference_run(tmp_path / "run", seed=13)

    class _BrokenPool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> _BrokenPool:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def map(self, *args: Any, **kwargs: Any) -> Any:
            raise BrokenProcessPool("a worker exited unexpectedly")

    monkeypatch.setattr(rollout_module, "ProcessPoolExecutor", _BrokenPool)

    with pytest.raises(RolloutBenchmarkError, match="worker stopped"):
        run_benchmark(
            benchmark_registry()["rollout"],
            _config(
                model=ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu"),
                workers=2,
            ),
            runner=None,
            checkpoint=None,
        )


# --- Terminal-action guardrails -------------------------------------------


def _guardrails(**overrides: Any) -> Any:
    """Return the guardrails of a one-cell suite's only cell."""

    runtime = overrides.pop(
        "runtime", RuntimeConfig(resignation_enabled=True, draw_claim_enabled=True)
    )
    result = _run(_config(runtime=runtime, **overrides), runner=ResigningRunner())
    return result.cells[0].guardrails


def test_a_seat_that_always_resigns_is_measured_as_premature() -> None:
    """Resigning from the opening is the failure the guardrail exists to catch."""

    guardrails = _guardrails()

    assert guardrails.resignations > 0
    assert guardrails.premature_resignations == guardrails.resignations
    assert guardrails.premature_rate == pytest.approx(1.0)


def test_a_disabled_terminal_action_is_absent_from_the_silent_count() -> None:
    """An action nothing offered cannot be one the model declined to use."""

    guardrails = _guardrails(
        runtime=RuntimeConfig(resignation_enabled=True, draw_claim_enabled=False)
    )

    assert guardrails.enabled_terminal_actions == (RESIGNATION_ACTION_ID,)
    assert guardrails.silent_terminal_actions == ()


def test_an_offered_action_nobody_selects_is_named() -> None:
    """A capability the runtime offers and the model never uses is silent."""

    guardrails = _guardrails()

    assert DRAW_CLAIM_ACTION_ID in guardrails.enabled_terminal_actions
    assert guardrails.silent_terminal_actions == (DRAW_CLAIM_ACTION_ID,)


def test_a_suite_that_finished_every_game_reports_no_non_termination() -> None:
    """The rate counts games the ply limit stopped, not games that ended."""

    guardrails = _guardrails()

    assert guardrails.claimable_unfinished_games == 0


def test_the_premature_threshold_is_declared_rather_than_configured() -> None:
    """A per-run dial would end every series sharing a cell's workload.

    Only the premature rate reads the threshold, and a cell's workload is
    shared by every metric on it, so a knob that moved it would re-baseline
    twenty-odd readings that never look at it.
    """

    cell = _run(_config(), runner=ResigningRunner()).cells[0]

    assert PREMATURE_MATERIAL_BALANCE == 0.0
    assert "premature_material_balance" not in cell.execution.workload
    assert not hasattr(_config().value, "guardrails")


def test_the_human_premature_rate_is_read_from_the_losing_players_side() -> None:
    """A player may resign on the opponent's turn, so the mover is the wrong seat."""

    from anthro_chess.evaluation.reference import ComparableGame, HumanReference
    from anthro_chess.evaluation.rollout import _human_premature_rate

    # White is a queen up and Black is to move. Black resigning is hopeless and
    # White resigning is the premature case, and both share this final position.
    ahead = analyze_trajectory(
        chess.STARTING_FEN,
        [
            encode_move(chess.Move.from_uci(uci))
            for uci in ("e2e4", "e7e5", "d1h5", "e8e7", "h5e5")
        ],
    )
    config = _config().value
    losses = {
        "0-1": ComparableGame(
            rating=1500.0, result="0-1", trajectory=ahead, termination="resignation"
        ),
        "1-0": ComparableGame(
            rating=1500.0, result="1-0", trajectory=ahead, termination="resignation"
        ),
    }

    assert ahead.final_turn_white is False
    assert _human_premature_rate(
        config, HumanReference(games=(losses["1-0"],), excluded={})
    ) == (pytest.approx(0.0), 1)
    assert _human_premature_rate(
        config, HumanReference(games=(losses["0-1"],), excluded={})
    ) == (pytest.approx(1.0), 1)


def test_a_reference_with_no_resignation_has_no_premature_rate() -> None:
    """Zero would read as a population that never resigned early."""

    from anthro_chess.evaluation.reference import ComparableGame, HumanReference
    from anthro_chess.evaluation.rollout import _human_premature_rate

    trajectory = analyze_trajectory(
        chess.STARTING_FEN,
        [encode_move(chess.Move.from_uci(uci)) for uci in ("e2e4", "e7e5")],
    )
    reference = HumanReference(
        games=(
            ComparableGame(
                rating=1500.0,
                result="1-0",
                trajectory=trajectory,
                termination="checkmate",
            ),
        ),
        excluded={},
    )

    assert _human_premature_rate(_config().value, reference) == (None, 0)


# --- The termination quantity ---------------------------------------------


def _human_game(termination: str) -> Any:
    """Return one human game that ended the named way."""

    from anthro_chess.evaluation.reference import _comparable_game

    row = {
        NormalizedColumn.WHITE_NORMALIZED_RATING.value: 1500,
        NormalizedColumn.BLACK_NORMALIZED_RATING.value: 1500,
        NormalizedColumn.ACTION_IDS.value: [
            encode_move(chess.Move.from_uci(uci)) for uci in ("e2e4", "e7e5")
        ],
        NormalizedColumn.INITIAL_POSITION.value: chess.STARTING_FEN,
        NormalizedColumn.RESULT.value: "1-0",
        NormalizedColumn.TERMINATION_CATEGORY.value: termination,
    }
    game, reason = _comparable_game(row, ReferenceConfig(), book=None)
    assert game is not None, reason
    return game


@pytest.mark.parametrize(
    "termination",
    ("clock_expiry", "abandonment", "draw_agreement", "unknown"),
)
def test_a_human_ending_the_model_cannot_reach_leaves_the_termination_quantity(
    termination: str,
) -> None:
    """Its mass is what pinned the distance and flattened it against the model."""

    game = _human_game(termination)

    assert game.value(ComparedQuantity.TERMINATION) is None
    assert game.observation(ComparedQuantity.TERMINATION) is None
    # Only that quantity. The game was still played, so its length, result and
    # opening are ordinary observations of the reference.
    assert game.observation(ComparedQuantity.GAME_LENGTH) is not None
    assert game.observation(ComparedQuantity.RESULT) is not None


def test_a_human_ending_the_model_can_reach_stays_in_the_comparison() -> None:
    """Dropping every category would leave nothing to be close to."""

    game = _human_game("checkmate")

    assert game.value(ComparedQuantity.TERMINATION) == "checkmate"


def test_the_ply_limit_stays_on_the_model_side_with_no_counterpart() -> None:
    """A game the harness stopped is charged for here as well as as unfinished."""

    from anthro_chess.evaluation.reference import generated_games

    features = analyze_games(
        [
            _record_for_termination(GameTermination.PLY_LIMIT),
            _record_for_termination(GameTermination.CHECKMATE),
        ]
    )
    values = [
        game.value(ComparedQuantity.TERMINATION)
        for game in generated_games(features, rating=1500)
    ]

    assert values == [GameTermination.PLY_LIMIT.value, GameTermination.CHECKMATE.value]


def _record_for_termination(termination: GameTermination) -> Any:
    """Return one generated record that ended the named way."""

    from anthro_chess.evaluation.games import (
        DecisionRecord,
        GameOutcome,
        SeatRecord,
        build_game_record,
    )

    action_ids = [
        encode_move(chess.Move.from_uci(uci)) for uci in ("e2e4", "e7e5", "g1f3")
    ]
    seat = SeatRecord(kind="model", label="fixture", seed=0)
    return build_game_record(
        initial_position=chess.STARTING_FEN,
        prefix_plies=0,
        action_ids=action_ids,
        white=seat,
        black=seat,
        seed=0,
        decisions=[
            DecisionRecord(
                ply_index=index,
                slot="white" if index % 2 == 0 else "black",
                action_id=action_id,
            )
            for index, action_id in enumerate(action_ids)
        ],
        outcome=GameOutcome(
            result="*" if termination is GameTermination.PLY_LIMIT else "1-0",
            termination=termination,
            adjudicated=termination is GameTermination.PLY_LIMIT,
        ),
    )


def test_the_unreachable_set_tracks_the_two_ending_vocabularies() -> None:
    """A category one side gains has to move the comparison with it.

    Derived rather than listed, so the pin worth having is on the consequence:
    a clock the harness one day runs makes clock expiry reachable and takes it
    out of the set, which should be a deliberate change rather than a surprise.
    """

    from anthro_chess.data.termination import TerminationCategory
    from anthro_chess.evaluation.reference import UNREACHABLE_HUMAN_TERMINATIONS

    assert TerminationCategory.CLOCK_EXPIRY.value in UNREACHABLE_HUMAN_TERMINATIONS
    assert TerminationCategory.ABANDONMENT.value in UNREACHABLE_HUMAN_TERMINATIONS
    assert TerminationCategory.CHECKMATE.value not in UNREACHABLE_HUMAN_TERMINATIONS
    # The mirror image stays on the model's side rather than being dropped with
    # them, because the ply limit is a gap a checkpoint can close.
    assert GameTermination.PLY_LIMIT.value not in UNREACHABLE_HUMAN_TERMINATIONS
