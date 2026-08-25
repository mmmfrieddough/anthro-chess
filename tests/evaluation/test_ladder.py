"""What a rating ladder measures, and what it reports when it cannot."""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    RESIGNATION_ACTION_ID,
)
from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.data import DecisionContext, Speed
from anthro_chess.evaluation import PoolConfig, freeze_pool
from anthro_chess.evaluation import ladder as ladder_module
from anthro_chess.evaluation.benchmarks import benchmark_registry, run_benchmark
from anthro_chess.evaluation.cost import BENCHMARK_COST_KIND
from anthro_chess.evaluation.dependency import ConditioningKind
from anthro_chess.evaluation.ladder import (
    LADDER_BOOTSTRAP_METHOD,
    LADDER_DETERMINISTIC_METHOD,
    LADDER_KIND,
    LADDER_UNRESOLVED_METHOD,
    RESPONSE_SCOPE,
    LadderBenchmarkConfig,
    LadderBenchmarkError,
    LadderBenchmarkResult,
    LadderPairing,
    LadderSeat,
    SeatConditioning,
    SeatKey,
    fit_ratings,
    seat_keys,
)
from anthro_chess.evaluation.results import (
    CheckpointReference,
    DetailStore,
    ResultEnvelope,
    ResultsStore,
    dispersion_bound,
)
from anthro_chess.evaluation.results.metrics import (
    LADDER_ABLATED_TEMPERATURE_RESPONSE,
    LADDER_ADJACENT_RATING_ORDER_ACCURACY,
    LADDER_DEPARTURE_POLICY_REGRET,
    LADDER_FITTED_RATING,
    LADDER_FITTED_RATING_SLOPE,
    LADDER_FITTED_RATING_SPAN,
    LADDER_POLICY_REGRET,
    LADDER_PREFERRED_SELECTION_RATE,
    LADDER_RATING_ERROR,
    LADDER_RATING_ORDER_ACCURACY,
    LADDER_SCORE_RATE,
    LADDER_SCORED_GAME_RATE,
    LADDER_SELECTED_RANK,
    LADDER_TEMPERATURE_RESPONSE,
    RATING_BEHAVIOR_FAMILY,
    MetricDefinition,
    registered_metrics,
)
from anthro_chess.inference import ModelRunnerConfig
from anthro_chess.runtime import RuntimeConfig

CHECKPOINT = CheckpointReference(label="fixture-checkpoint", step=1)


@dataclass
class GradedRunner:
    """A stand-in policy whose willingness to resign falls with its rating.

    Real chess strength cannot be faked in a fixture, and a ladder does not
    need it to be: what it needs is seats that are genuinely orderable by their
    results. Resignation is the lever, because it is the one action that ends a
    game inside a short ply limit. A seat configured low rates every move below
    resigning and gives up sooner, so it loses more, and both dials reach the
    outcome through the ordinary sampling path rather than through anything
    this benchmark knows about.

    The direction temperature moves the fixture in is an artifact of that lever
    rather than a prediction, so the tests read that a response was measured
    and never that it has a particular sign.
    """

    def predict(self, context: DecisionContext) -> torch.Tensor:
        rating = 1000 if context.target_rating is None else context.target_rating
        logits = torch.full((ACTION_VOCABULARY_SIZE,), (rating - 2000) / 500.0)
        logits[RESIGNATION_ACTION_ID] = 0.0
        return logits


@dataclass
class SweepingRunner:
    """A stand-in policy that resigns at once below 2000 and never above it.

    The top seat therefore wins every game it plays, which is the record whose
    maximum-likelihood rating is unbounded — a bound rather than an estimate,
    and the one case a resample cannot say anything about. The two weak seats
    still split their own pairing, so exactly one seat sweeps.
    """

    def predict(self, context: DecisionContext) -> torch.Tensor:
        weak = (context.target_rating or 0) < 2000
        logits = torch.full((ACTION_VOCABULARY_SIZE,), -30.0 if weak else 0.0)
        logits[RESIGNATION_ACTION_ID] = 0.0 if weak else -30.0
        return logits


@dataclass
class NeverEndingRunner:
    """A stand-in policy that plays on and never resigns.

    Every game it plays reaches the ply limit, so nothing it plays is scored.
    That is the input a ladder cannot fit, as opposed to one it fits badly.
    """

    def predict(self, context: DecisionContext) -> torch.Tensor:
        logits = torch.zeros(ACTION_VOCABULARY_SIZE)
        logits[RESIGNATION_ACTION_ID] = -30.0
        return logits


_BASE_GRID: dict[str, Any] = {
    "target_ratings": (1200, 2000),
    "temperatures": (1.0,),
    "reference_temperature": 1.0,
    "seeds": (0, 1),
}
_BASE_GENERATION: dict[str, Any] = {
    "games_per_position": 4,
    "maximum_generated_plies": 20,
    "swap_colors": True,
}


def _config(**overrides: Any) -> ResolvedConfig[LadderBenchmarkConfig]:
    """Return a resolved ladder small enough for the CPU test suite.

    Nested overrides merge into the small defaults, so a test naming one field
    of the grid does not silently inherit the production game counts.
    """

    fields: dict[str, Any] = {
        # Resignation is what ends a fixture game, so it is enabled here even
        # though the shipped selection leaves it off.
        "runtime": RuntimeConfig(resignation_enabled=True),
    }
    fields["grid"] = {**_BASE_GRID, **overrides.pop("grid", {})}
    fields["generation"] = {**_BASE_GENERATION, **overrides.pop("generation", {})}
    # Resamples buy precision on the floor rather than deciding its shape, and
    # the shape is what these tests read, so a fixture pays for a hundred rather
    # than the shipped thousand.
    fields["noise"] = {"resamples": 100, **overrides.pop("noise", {})}
    fields.update(overrides)
    return ResolvedConfig(
        value=LadderBenchmarkConfig.model_validate(fields),
        provenance=ConfigProvenance(source=None, overrides=()),
    )


def _run(
    config: ResolvedConfig[LadderBenchmarkConfig],
    *,
    runner: Any | None = None,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> LadderBenchmarkResult:
    return cast(
        LadderBenchmarkResult,
        run_benchmark(
            benchmark_registry()["ladder"],
            config,
            store=store,
            detail=detail,
            runner=runner or GradedRunner(),
            checkpoint=CHECKPOINT,
        ),
    )


def _seats(*, ratings: tuple[int, ...], temperature: float) -> tuple[SeatKey, ...]:
    return tuple(
        SeatKey(SeatConditioning.CONDITIONED, rating, temperature) for rating in ratings
    )


def _round_robin(
    seats: tuple[SeatKey, ...],
    strengths: dict[SeatKey, float],
    *,
    games: int = 400,
) -> tuple[LadderPairing, ...]:
    """Return exact expected results for a ladder of known true ratings."""

    pairings = []
    for first, second in itertools.combinations(seats, 2):
        expected = 1.0 / (
            1.0 + 10.0 ** ((strengths[second] - strengths[first]) / 400.0)
        )
        wins = round(expected * games)
        pairings.append(
            LadderPairing(
                first=first,
                second=second,
                first_wins=wins,
                draws=0,
                first_losses=games - wins,
            )
        )
    return tuple(pairings)


def test_grid_schedules_every_seat_and_one_ablated_arm_per_temperature() -> None:
    config = LadderBenchmarkConfig.model_validate(
        {
            "grid": {
                "target_ratings": (1200, 1800),
                "temperatures": (0.0, 1.0),
                "reference_temperature": 1.0,
            }
        }
    )

    seats = seat_keys(config)

    assert [seat.label for seat in seats] == [
        "1200@t0",
        "1800@t0",
        "1200@t1",
        "1800@t1",
        "ablated@t0",
        "ablated@t1",
    ]
    assert [
        seat.target_rating
        for seat in seats
        if seat.conditioning is SeatConditioning.ABLATED
    ] == [None, None]


def test_disabled_ablation_fields_no_control_arm() -> None:
    config = LadderBenchmarkConfig.model_validate(
        {"grid": {"target_ratings": (1200, 1800)}, "ablation": {"enabled": False}}
    )

    assert all(
        seat.conditioning is SeatConditioning.CONDITIONED for seat in seat_keys(config)
    )


def test_reference_temperature_must_be_on_the_grid() -> None:
    with pytest.raises(ValidationError, match="reference_temperature"):
        LadderBenchmarkConfig.model_validate(
            {"grid": {"temperatures": (0.5, 1.0), "reference_temperature": 0.7}}
        )


def test_a_ladder_needs_two_configured_ratings() -> None:
    with pytest.raises(ValidationError, match="two configured ratings"):
        LadderBenchmarkConfig.model_validate({"grid": {"target_ratings": (1500,)}})


def test_fit_recovers_the_ratings_that_generated_the_results() -> None:
    """The fit is checked against a ladder whose true ratings are known."""

    seats = _seats(ratings=(1200, 1500, 1800, 2100), temperature=1.0)
    strengths = {seat: float(seat.target_rating or 0) for seat in seats}

    fit = fit_ratings(
        seats,
        _round_robin(seats, strengths),
        anchor_seats=seats,
        anchor_rating=1650.0,
    )

    assert fit.converged
    assert not fit.clamped
    for seat in seats:
        assert fit.rating(seat) == pytest.approx(strengths[seat], abs=5.0)


def test_an_indistinguishable_ladder_is_reported_rather_than_failed() -> None:
    """A flat ladder is a reading about the checkpoint, not a broken fit."""

    seats = _seats(ratings=(1200, 1500, 1800), temperature=1.0)
    strengths = dict.fromkeys(seats, 1500.0)

    fit = fit_ratings(
        seats,
        _round_robin(seats, strengths),
        anchor_seats=seats,
        anchor_rating=1500.0,
    )

    assert fit.converged
    spread = max(fit.ratings.values()) - min(fit.ratings.values())
    assert spread == pytest.approx(0.0, abs=1.0)


def test_a_seat_that_never_loses_is_clamped_and_named() -> None:
    """An unbounded maximum likelihood becomes a declared extreme."""

    seats = _seats(ratings=(1200, 1500, 1800), temperature=1.0)
    perfect = seats[-1]
    pairings = tuple(
        LadderPairing(
            first=first,
            second=second,
            first_wins=0 if second == perfect else 10,
            draws=0,
            first_losses=20 if second == perfect else 10,
        )
        for first, second in itertools.combinations(seats, 2)
    )

    fit = fit_ratings(
        seats,
        pairings,
        anchor_seats=seats,
        anchor_rating=1500.0,
        maximum_spread=800.0,
    )

    assert fit.clamped == (perfect,)
    assert fit.rating(perfect) is not None


def test_a_fit_that_runs_out_of_iterations_still_reports() -> None:
    seats = _seats(ratings=(1200, 2100), temperature=1.0)
    strengths = {seat: float(seat.target_rating or 0) for seat in seats}

    fit = fit_ratings(
        seats,
        _round_robin(seats, strengths),
        anchor_seats=seats,
        anchor_rating=1650.0,
        maximum_iterations=1,
    )

    assert not fit.converged
    assert fit.iterations == 1
    assert set(fit.ratings) == set(seats)


def test_a_ladder_with_no_scored_game_is_a_generation_failure() -> None:
    """No result at all is different from a fit that resolved nothing."""

    with pytest.raises(LadderBenchmarkError, match="generation failure"):
        _run(
            _config(generation={"maximum_generated_plies": 4}),
            runner=NeverEndingRunner(),
        )


def test_a_pinned_generation_without_a_pool_is_a_configuration_error() -> None:
    """A pin that protects nothing reads exactly like one that protects."""

    with pytest.raises(ValidationError, match="does not read"):
        _config(openings={"expected_pool_game_ids_sha256": "0" * 64})


def _freeze(
    write_corpus: Callable[..., tuple[Path, Path]],
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Freeze fixture rows into a pool the openings can be projected out of."""

    normalized, manifest = write_corpus(directory / "corpus", rows)
    output = directory / "pool"
    freeze_pool(
        ResolvedConfig(
            value=PoolConfig.model_validate(
                {
                    "pool_id": "ladder-fixture",
                    "normalized": str(normalized),
                    "manifest": str(manifest),
                }
            ),
            provenance=ConfigProvenance(source=None, overrides=()),
        ),
        output,
    )
    return output


def test_the_openings_are_drawn_at_one_speed_class(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """Every pairing plays the same roots, so their composition reaches every seat.

    Which openings people play is a strong function of the clock, so a draw from
    a mixed pool would put that mixture into every fitted rating.
    """

    pool = _freeze(
        write_corpus,
        tmp_path,
        [
            normalized_row(
                index,
                split="test",
                plies=10,
                time_initial_ms=60_000 if index % 2 else 300_000,
            )
            for index in range(1, 9)
        ],
    )

    result = _run(
        _config(
            openings={
                "pool": str(pool),
                "view": {"name": "ladder-openings", "speed": "blitz"},
            }
        )
    )

    assert result.view is not None
    assert result.view.speed is Speed.BLITZ
    assert result.view.selected_games == 4

    # Which filter emptied the view is what a reader has to fix, so the reason
    # is matched rather than the error type.
    with pytest.raises(LadderBenchmarkError, match="8 speed_mismatch"):
        _run(
            _config(
                openings={
                    "pool": str(pool),
                    "view": {"name": "ladder-openings", "speed": "rapid"},
                }
            )
        )


def test_openings_from_a_pool_this_ladder_is_not_defined_over_are_refused(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """The generation a selection pins reaches the loader from here."""

    pool = _freeze(write_corpus, tmp_path, [normalized_row(1, split="test", plies=10)])
    config = _config(
        openings={
            "pool": str(pool),
            "expected_pool_game_ids_sha256": "0" * 64,
        }
    )

    with pytest.raises(LadderBenchmarkError, match="expected 0{64}"):
        _run(config)


def test_every_pair_of_seats_meets_and_both_colors_are_played() -> None:
    result = _run(_config())

    seats = result.seats
    assert len(seats) == 3  # two configured ratings plus one ablated seat
    assert len(result.pairings) == 3
    assert {(pairing.first, pairing.second) for pairing in result.pairings} == set(
        itertools.combinations([seat.key for seat in seats], 2)
    )
    # Every seat's games are the ones its own pairings scored, and no game is
    # counted for a seat that did not play it.
    for seat in seats:
        assert seat.games == sum(
            pairing.games
            for pairing in result.pairings
            if seat.key in (pairing.first, pairing.second)
        )
    assert result.games > 0


def test_two_greedy_seats_play_one_replicate_of_the_game_they_can_play() -> None:
    """The same game three times is not three results for the joint fit.

    Two greedy seats replay one game per opening however many replicates they
    are asked for, so that pairing plays one. A pairing with a sampling seat
    still needs every replicate it was given, which is what keeps this a
    collapse rather than a cut.
    """

    result = _run(
        _config(
            grid={
                "target_ratings": (1200, 2000),
                "temperatures": (0.0, 1.0),
                "reference_temperature": 1.0,
                "seeds": (0, 1, 2),
            },
            generation={"games_per_position": 1},
            ablation={"enabled": False},
        )
    )

    greedy = [
        pairing
        for pairing in result.pairings
        if pairing.first.temperature == 0.0 and pairing.second.temperature == 0.0
    ]
    sampling = [pairing for pairing in result.pairings if pairing not in greedy]
    assert len(greedy) == 1
    assert {pairing.seeds for pairing in greedy} == {(0,)}
    assert {pairing.seeds for pairing in sampling} == {(0, 1, 2)}
    # One opening played from both sides: the greedy pairing plays that pair of
    # games once, where a sampling one plays it at every seed and replicate.
    assert {pairing.games + pairing.unfinished for pairing in greedy} == {2}
    assert {pairing.games + pairing.unfinished for pairing in sampling} == {6}


def test_the_ladder_orders_configured_ratings_and_reports_its_shape() -> None:
    result = _run(_config())

    reading = result.reading(1.0)
    assert reading.ratings == (1200, 2000)
    assert reading.order_accuracy == 1.0
    assert reading.adjacent_order_accuracy == 1.0
    assert reading.slope > 0.0
    assert reading.span > 0.0
    assert not reading.inversions
    assert reading.ladder_error >= 0.0


def test_a_temperature_response_needs_more_than_one_temperature() -> None:
    result = _run(_config())

    assert result.response is None
    assert "temperature-response" in result.unavailable


def test_temperature_response_is_measured_against_an_ablated_control() -> None:
    result = _run(
        _config(
            grid={
                "target_ratings": (1200, 2000),
                "temperatures": (0.7, 1.0),
                "reference_temperature": 1.0,
            }
        )
    )

    response = result.response
    assert response is not None
    assert response.temperatures == (0.7, 1.0)
    assert [rating for rating, _ in response.per_rating] == [1200, 2000]
    # Both arms come out of one fit, so the ablated response is on the same
    # scale as the conditioned one rather than on a scale of its own.
    assert response.ablated_response is not None
    assert response.attenuation is not None or (
        response.attenuation_unavailable is not None
    )


def test_disabling_ablation_leaves_the_response_uncontrolled() -> None:
    result = _run(
        _config(
            grid={
                "target_ratings": (1200, 2000),
                "temperatures": (0.7, 1.0),
                "reference_temperature": 1.0,
            },
            ablation={"enabled": False},
        )
    )

    response = result.response
    assert response is not None
    assert response.ablated_response is None
    assert response.attenuation is None
    assert "ablation was disabled" in (response.attenuation_unavailable or "")
    assert "temperature-response-attenuation" in result.unavailable


def test_the_fit_is_anchored_on_the_reference_temperature_row() -> None:
    result = _run(
        _config(
            grid={
                "target_ratings": (1200, 2000),
                "temperatures": (0.7, 1.0),
                "reference_temperature": 1.0,
            }
        )
    )

    assert result.fit.anchor_basis == "reference-temperature"
    anchored = [
        result.seat(SeatKey(SeatConditioning.CONDITIONED, rating, 1.0)).fitted_rating
        for rating in (1200, 2000)
    ]
    assert sum(value or 0.0 for value in anchored) / len(anchored) == pytest.approx(
        1600.0, abs=1e-6
    )


def test_the_error_profile_is_reported_beside_each_seat_strength() -> None:
    result = _run(_config())

    for seat in result.seats:
        profile = seat.decisions
        assert profile is not None
        assert profile.decisions > 0
        assert 0.0 <= profile.preferred_selection_rate <= 1.0
        assert profile.selected_rank >= 1.0


def test_a_suite_reproduces_from_its_seeds() -> None:
    first = _run(_config())
    second = _run(_config())

    assert [pairing.as_record() for pairing in first.pairings] == [
        pairing.as_record() for pairing in second.pairings
    ]
    assert [seat.fitted_rating for seat in first.seats] == [
        seat.fitted_rating for seat in second.seats
    ]


def test_a_parallel_ladder_reads_the_same_as_a_serial_one(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    """Which process played a pairing must not reach anything the fit takes.

    Every other test here supplies a runner, which holds the benchmark to one
    process, so this is the only place the worker path runs at all. It loads a
    real checkpoint because that is what a worker loads for itself.

    Exact equality is available because this model is too small for the intra-op
    thread count to reach a result, and a worker pins that to one where this
    process does not. A model large enough to be reduced in parallel can differ
    in the last bits between the two.
    """

    checkpoint = inference_run(tmp_path / "run", seed=11)
    model = ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu")

    def read(workers: int) -> LadderBenchmarkResult:
        return cast(
            LadderBenchmarkResult,
            run_benchmark(
                benchmark_registry()["ladder"],
                _config(model=model, workers=workers),
            ),
        )

    serial = read(1)
    parallel = read(2)

    assert serial.games > 0
    assert parallel.as_record() == serial.as_record()
    assert [record.as_record() for record in parallel.records] == [
        record.as_record() for record in serial.records
    ]


def test_a_worker_that_stops_fails_this_reading_rather_than_the_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inference_run: Callable[..., Path],
) -> None:
    """A broken pool raises `RuntimeError`, which no sweep converts to a step failure.

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

    monkeypatch.setattr(ladder_module, "ProcessPoolExecutor", _BrokenPool)

    with pytest.raises(LadderBenchmarkError, match="worker stopped"):
        run_benchmark(
            benchmark_registry()["ladder"],
            _config(
                model=ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu"),
                workers=2,
            ),
        )


def test_a_supplied_runner_keeps_the_ladder_in_one_process() -> None:
    """A loaded runner is what the reading is of, and it cannot be sent to a worker.

    The selection here names no checkpoint, so a run that reached for workers
    would fail resolving one rather than quietly measuring something else.
    """

    result = _run(_config(workers=4))

    assert result.games > 0


def test_worker_count_stays_out_of_what_a_reading_declares(tmp_path: Path) -> None:
    """Spreading the round robin changes no number, so it ends no series."""

    result = _run(_config(workers=1), store=ResultsStore(tmp_path / "store"))
    spread = _run(_config(workers=3), store=ResultsStore(tmp_path / "spread"))

    for envelope, other in zip(_readings(result), _readings(spread), strict=True):
        assert envelope.execution is not None
        assert other.execution is not None
        assert "workers" not in envelope.execution.workload
        assert envelope.execution.workload_sha256 == other.execution.workload_sha256


def test_a_ladder_needs_at_least_one_worker() -> None:
    with pytest.raises(ValidationError):
        _config(workers=0)


def test_recorded_results_carry_one_series_per_unit(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "store")
    detail = DetailStore(tmp_path / "detail")

    result = _run(_config(), store=store, detail=detail)

    envelopes = _readings(result)
    # One per seat, one per temperature row, and none for the response, which a
    # single-temperature grid cannot measure.
    assert len(envelopes) == len(result.seats) + len(result.readings)
    # The ladder's own records, and one saying what the invocation cost.
    assert {envelope.kind for envelope in result.envelopes} == {
        LADDER_KIND,
        BENCHMARK_COST_KIND,
    }
    assert all(envelope.execution is not None for envelope in envelopes)
    # Every reading is committed, and what the invocation cost beside them.
    assert len(result.recorded_paths) == len(envelopes) + 1
    assert len(result.detail_paths) == len(envelopes)
    for envelope in envelopes:
        envelope.verify()

    seat_envelope = _envelope_with(envelopes, LADDER_FITTED_RATING)
    reading_envelope = _envelope_with(envelopes, LADDER_RATING_ORDER_ACCURACY)
    assert seat_envelope.execution is not None
    assert reading_envelope.execution is not None
    # A seat and the row it belongs to measure different quantities, so they
    # must not share a series.
    assert (
        seat_envelope.execution.workload_sha256
        != reading_envelope.execution.workload_sha256
    )
    assert {measurement.metric for measurement in reading_envelope.measurements} == {
        LADDER_RATING_ORDER_ACCURACY.identifier,
        LADDER_ADJACENT_RATING_ORDER_ACCURACY.identifier,
        LADDER_RATING_ERROR.identifier,
        LADDER_FITTED_RATING_SLOPE.identifier,
        LADDER_FITTED_RATING_SPAN.identifier,
    }
    assert {
        LADDER_SCORE_RATE.identifier,
        LADDER_PREFERRED_SELECTION_RATE.identifier,
        LADDER_POLICY_REGRET.identifier,
        LADDER_SELECTED_RANK.identifier,
    } <= {measurement.metric for measurement in seat_envelope.measurements}


def test_two_seats_of_one_row_are_different_series(tmp_path: Path) -> None:
    result = _run(
        _config(),
        store=ResultsStore(tmp_path / "store"),
        detail=DetailStore(tmp_path / "detail"),
    )

    fingerprints = {
        measurement.fingerprint
        for envelope in result.envelopes
        for measurement in envelope.measurements
        if measurement.metric == LADDER_FITTED_RATING.identifier
    }

    assert len(fingerprints) == len(result.seats)


def test_the_response_is_recorded_as_its_own_series(tmp_path: Path) -> None:
    result = _run(
        _config(
            grid={
                "target_ratings": (1200, 2000),
                "temperatures": (0.7, 1.0),
                "reference_temperature": 1.0,
            }
        ),
        store=ResultsStore(tmp_path / "store"),
        detail=DetailStore(tmp_path / "detail"),
    )

    envelope = _envelope_with(result.envelopes, LADDER_TEMPERATURE_RESPONSE)
    metrics = {measurement.metric for measurement in envelope.measurements}

    assert LADDER_ABLATED_TEMPERATURE_RESPONSE.identifier in metrics
    assert envelope.execution is not None
    # The response spans the grid, so its workload names both axes rather than
    # a single temperature.
    workload = envelope.execution.workload
    assert workload["temperatures"] == [0.7, 1.0]
    assert "temperature" not in workload


def test_every_result_declares_the_reference_temperature(tmp_path: Path) -> None:
    """A calibration figure that does not say what it was measured at is unreadable."""

    result = _run(
        _config(
            grid={
                "target_ratings": (1200, 2000),
                "temperatures": (0.7, 1.0),
                "reference_temperature": 1.0,
            }
        ),
        store=ResultsStore(tmp_path / "store"),
        detail=DetailStore(tmp_path / "detail"),
    )

    for envelope in _readings(result):
        assert envelope.execution is not None
        assert envelope.execution.workload["reference_temperature"] == 1.0


def test_a_seat_names_its_treatment_in_the_shared_conditioning_vocabulary(
    tmp_path: Path,
) -> None:
    """The ablated arm and a dependency test's absent arm are one treatment."""

    result = _run(
        _config(),
        store=ResultsStore(tmp_path / "store"),
        detail=DetailStore(tmp_path / "detail"),
    )

    treatments = {seat.execution.workload["conditioning_kind"] for seat in result.seats}

    assert treatments == {ConditioningKind.TRUE.value, ConditioningKind.ABSENT.value}


def test_retained_games_are_written_once_under_the_seat_that_held_white(
    tmp_path: Path,
) -> None:
    detail = DetailStore(tmp_path / "detail")

    result = _run(
        _config(),
        store=ResultsStore(tmp_path / "store"),
        detail=detail,
    )

    written = [
        json.loads(path.read_text(encoding="utf-8")) for path in result.detail_paths
    ]
    game_ids = [
        game["game_id"]
        for payload in written
        for game in payload.get("games_detail", [])
    ]

    assert len(game_ids) == len(set(game_ids))
    assert len(game_ids) == result.games + result.unfinished


def test_a_grid_change_ends_every_seat_series(tmp_path: Path) -> None:
    """A seat's fitted rating is an output of the whole fit, not of its games."""

    narrow = _run(
        _config(),
        store=ResultsStore(tmp_path / "narrow"),
        detail=DetailStore(tmp_path / "narrow-detail"),
    )
    wide = _run(
        _config(grid={"target_ratings": (1200, 1600, 2000)}),
        store=ResultsStore(tmp_path / "wide"),
        detail=DetailStore(tmp_path / "wide-detail"),
    )

    seat = SeatKey(SeatConditioning.CONDITIONED, 1200, 1.0)
    assert _fingerprint(narrow, seat) != _fingerprint(wide, seat)


def test_every_ladder_metric_is_registered_in_the_rating_behavior_family() -> None:
    identifiers = {
        definition.identifier
        for definition in registered_metrics(RATING_BEHAVIOR_FAMILY.identifier)
    }

    assert {
        LADDER_FITTED_RATING.identifier,
        LADDER_RATING_ORDER_ACCURACY.identifier,
        LADDER_SCORED_GAME_RATE.identifier,
        LADDER_TEMPERATURE_RESPONSE.identifier,
        LADDER_DEPARTURE_POLICY_REGRET.identifier,
    } <= identifiers


def test_every_fitted_quantity_carries_what_the_reading_can_resolve() -> None:
    """A floor on the seats alone would leave the headline numbers bare.

    Ordering, slope, span, ladder error and both temperature responses are
    functions of the fitted ratings, so each inherits their sampling error and
    each has to say what it is.
    """

    result = _run(
        _config(
            grid={
                "target_ratings": (1200, 2000),
                "temperatures": (0.7, 1.0),
                "reference_temperature": 1.0,
            }
        )
    )
    resolution = result.resolution

    assert resolution is not None
    assert resolution.method == LADDER_BOOTSTRAP_METHOD
    # Nothing was silently dropped, and the floors say what they claim.
    assert resolution.fitted_resamples == resolution.resamples
    settings = LadderBenchmarkConfig().noise
    assert resolution.confidence == settings.confidence
    reading = result.reading(1.0)
    for definition in (
        LADDER_RATING_ERROR,
        LADDER_FITTED_RATING_SLOPE,
        LADDER_FITTED_RATING_SPAN,
    ):
        spread = resolution.dispersion(reading.label, definition.identifier)
        assert spread is not None, definition.identifier
    for seat in result.seats:
        assert (
            resolution.dispersion(seat.label, LADDER_FITTED_RATING.identifier)
            is not None
        )
    for definition in (
        LADDER_TEMPERATURE_RESPONSE,
        LADDER_ABLATED_TEMPERATURE_RESPONSE,
    ):
        assert resolution.dispersion(RESPONSE_SCOPE, definition.identifier) is not None
    # An ordering over two configured ratings is one binary comparison, and a
    # fixture that finishes every game has a scored share the redraw cannot
    # move. Either answer is allowed for those; a bare number is not.
    named = [
        (reading.label, LADDER_RATING_ORDER_ACCURACY.identifier),
        (reading.label, LADDER_ADJACENT_RATING_ORDER_ACCURACY.identifier),
        *((seat.label, LADDER_SCORED_GAME_RATE.identifier) for seat in result.seats),
    ]
    for key in named:
        assert (key in resolution.dispersions) != (key in resolution.unqualifiable), key


def test_a_recorded_ladder_measurement_carries_its_own_spread(tmp_path: Path) -> None:
    """The spread travels on the measurement, not on the series.

    Seeds and games per pairing are deliberately outside a ladder's identity,
    so a spread filed against the series would later be applied to a reading
    taken at a different sample size.
    """

    result = _run(
        _config(),
        store=ResultsStore(tmp_path / "store"),
        detail=DetailStore(tmp_path / "detail"),
    )

    qualified = {
        measurement.metric
        for envelope in _readings(result)
        for measurement in envelope.measurements
        if measurement.dispersion is not None
    }

    assert LADDER_FITTED_RATING.identifier in qualified
    assert LADDER_SCORED_GAME_RATE.identifier in qualified
    assert LADDER_RATING_ERROR.identifier in qualified
    assert LADDER_FITTED_RATING_SLOPE.identifier in qualified
    # The error profile is a mean over decisions rather than an output of the
    # fit, so the refit does not reach it and its delta reports the noise as
    # unknown — which is what a floor somebody could still produce reads as.
    assert LADDER_POLICY_REGRET.identifier not in qualified


def test_a_ladder_nothing_would_redraw_states_a_floor_of_zero() -> None:
    """A grid of greedy seats replays, so its evaluation noise is exactly zero.

    Bootstrapping it would report the spread of a draw that another seed was
    never going to take, and a floor built from that hides real movement.
    """

    result = _run(
        _config(
            grid={
                "target_ratings": (1200, 2000),
                "temperatures": (0.0,),
                "reference_temperature": 0.0,
            }
        ),
        runner=SweepingRunner(),
    )
    resolution = result.resolution

    assert resolution is not None
    assert resolution.method == LADDER_DETERMINISTIC_METHOD
    assert resolution.resamples == 0
    assert resolution.redrawn_games == 0
    assert resolution.replayed_pairings == len(result.pairings)
    assert resolution.dispersions
    assert all(spread.value == 0.0 for spread in resolution.dispersions.values())
    # A seat that swept is pinned at the declared spread, and a replay pins it
    # there identically both times. Withholding its floor would state that a
    # reading which cannot move might have.
    assert not resolution.unqualifiable
    swept = _swept(result)
    assert swept
    for seat in swept:
        assert (
            resolution.dispersion(seat.label, LADDER_FITTED_RATING.identifier)
            is not None
        )


def test_a_pairing_that_replays_is_held_fixed_while_the_rest_redraw() -> None:
    """One greedy pairing in a mixed grid contributes no spread to bound."""

    result = _run(
        _config(
            grid={
                "target_ratings": (1200, 2000),
                "temperatures": (0.0, 1.0),
                "reference_temperature": 1.0,
            },
            ablation={"enabled": False},
        )
    )
    resolution = result.resolution
    greedy = [
        pairing
        for pairing in result.pairings
        if pairing.first.temperature == 0.0 and pairing.second.temperature == 0.0
    ]

    assert resolution is not None
    assert resolution.method == LADDER_BOOTSTRAP_METHOD
    assert resolution.replayed_pairings == len(greedy)
    assert resolution.redrawn_games == sum(
        pairing.played for pairing in result.pairings if pairing not in greedy
    )


def test_a_seat_rate_is_bounded_for_its_own_games_rather_than_the_grid_s() -> None:
    """A seat's score rate is computed from the games that seat played.

    Bounding it for every game in the grid would claim replicates it never had,
    and the claim grows with the grid: a three-seat grid gives each seat two
    thirds of the games, and a wider one gives it less. The fitted rating is
    keyed by seat too and is not the same case, because the fit is joint.
    """

    result = _run(_config(grid={"target_ratings": (1200, 1600, 2000)}))
    resolution = result.resolution
    seat = SeatKey(SeatConditioning.CONDITIONED, 1600, 1.0)

    assert resolution is not None
    seat_games = sum(
        pairing.played
        for pairing in result.pairings
        if seat in (pairing.first, pairing.second)
    )
    # Both counts have to stay under the surviving refits, or the refits are
    # what the bound rests on and this measures nothing.
    assert seat_games < resolution.redrawn_games < resolution.fitted_resamples

    rate = resolution.dispersion(seat.label, LADDER_SCORE_RATE.identifier)
    rating = resolution.dispersion(seat.label, LADDER_FITTED_RATING.identifier)
    assert rate is not None
    assert rating is not None
    assert rate.bound == pytest.approx(
        dispersion_bound(rate.value, degrees_of_freedom=seat_games - 1)
    )
    assert rating.bound == pytest.approx(
        dispersion_bound(rating.value, degrees_of_freedom=resolution.redrawn_games - 1)
    )


def test_a_thicker_sample_resolves_a_ladder_more_finely() -> None:
    """The floor has to answer to the games behind it, or it measures nothing.

    Same grid, same seats, four times the replicates. A floor that did not
    tighten would be describing the configuration rather than the sample.
    """

    thin = _run(_config(grid={"seeds": (0,)}))
    thick = _run(_config(grid={"seeds": (0, 1, 2, 3)}))

    assert thin.resolution is not None
    assert thick.resolution is not None
    assert thick.games > thin.games
    seat = SeatKey(SeatConditioning.CONDITIONED, 1200, 1.0)
    thin_floor = thin.resolution.floor(seat.label, LADDER_FITTED_RATING.identifier)
    thick_floor = thick.resolution.floor(seat.label, LADDER_FITTED_RATING.identifier)
    assert thin_floor is not None
    assert thick_floor is not None
    assert thick_floor < thin_floor


def test_a_seat_with_no_finite_rating_is_named_rather_than_given_a_zero() -> None:
    """A bound is not an estimate, and a resample of one reproduces the bound.

    Every resample of a seat that lost every game loses every game again, so its
    spread is zero and a floor built from it would license any delta at all.
    """

    result = _run(_config(), runner=SweepingRunner())
    resolution = result.resolution

    assert resolution is not None
    swept = _swept(result)
    assert swept
    for seat in swept:
        key = (seat.label, LADDER_FITTED_RATING.identifier)
        assert key not in resolution.dispersions
        assert "no finite maximum-likelihood rating" in resolution.unqualifiable[key]


def test_a_spread_that_merely_binds_still_qualifies_its_seat() -> None:
    """`clamped` is not the same question as `has no finite estimate`.

    A narrow declared spread pins a seat with an ordinary win-and-loss record.
    That rating still moves under resampling, so withholding its floor would
    throw away a real one and say something false about why.
    """

    result = _run(_config(fit={"maximum_spread": 30.0}))
    resolution = result.resolution

    assert resolution is not None
    assert result.fit.clamped
    unbounded = {seat.key for seat in _swept(result)}
    for seat in result.fit.clamped:
        if seat in unbounded:
            continue
        assert (
            resolution.dispersion(seat.label, LADDER_FITTED_RATING.identifier)
            is not None
        )


def test_a_quantity_the_redraw_could_not_move_is_named_rather_than_zeroed() -> None:
    """An estimated zero and a stated zero are different claims.

    A floor of zero from a bootstrap reads as perfect resolution and clears
    every delta, so a quantity every resample reproduced exactly reports no
    floor and says why. Only a ladder that replays states a zero.
    """

    result = _run(_config(), runner=SweepingRunner())
    resolution = result.resolution

    assert resolution is not None
    assert resolution.method == LADDER_BOOTSTRAP_METHOD
    assert all(spread.value > 0.0 for spread in resolution.dispersions.values())
    reading = result.reading(1.0)
    key = (reading.label, LADDER_RATING_ORDER_ACCURACY.identifier)
    assert key not in resolution.dispersions
    assert "every resample returned the same value" in resolution.unqualifiable[key]


def test_a_single_redrawn_game_reports_no_floor_rather_than_failing() -> None:
    """One game shows no spread, and a bound needs a degree of freedom.

    The games are already played by the time this is asked, so the answer is a
    reading that says it resolves nothing rather than a raised error.
    """

    result = _run(
        _config(
            grid={"seeds": (0,)},
            generation={"games_per_position": 1, "swap_colors": False},
            ablation={"enabled": False},
        ),
        runner=SweepingRunner(),
    )
    resolution = result.resolution
    reading = result.reading(1.0)

    assert resolution is not None
    assert resolution.method == LADDER_UNRESOLVED_METHOD
    assert resolution.redrawn_games == 1
    assert not resolution.dispersions
    assert (
        "no spread to bound"
        in resolution.unqualifiable[
            (reading.label, LADDER_FITTED_RATING_SLOPE.identifier)
        ]
    )


def _swept(result: LadderBenchmarkResult) -> list[LadderSeat]:
    """Return the seats that scored nothing or scored everything.

    The distinction these tests exist to pin: a seat with no finite
    maximum-likelihood rating, whose number is a bound rather than an
    estimate.
    """

    return [seat for seat in result.seats if seat.points in (0.0, float(seat.games))]


def _readings(result: LadderBenchmarkResult) -> tuple[ResultEnvelope, ...]:
    """Return the ladder's own records, without what the invocation cost."""

    return tuple(
        envelope for envelope in result.envelopes if envelope.kind == LADDER_KIND
    )


def _envelope_with(
    envelopes: tuple[ResultEnvelope, ...],
    definition: MetricDefinition,
) -> ResultEnvelope:
    for envelope in envelopes:
        if any(
            measurement.metric == definition.identifier
            for measurement in envelope.measurements
        ):
            return envelope
    raise AssertionError(f"no result reported {definition.identifier}")


def _fingerprint(result: LadderBenchmarkResult, seat: SeatKey) -> str:
    measured = result.seat(seat)
    for envelope in result.envelopes:
        if envelope.execution is None:
            continue
        if envelope.execution.workload_sha256 != measured.execution.workload_sha256:
            continue
        for measurement in envelope.measurements:
            if measurement.metric == LADDER_FITTED_RATING.identifier:
                return measurement.fingerprint
    raise AssertionError(f"no fitted rating was recorded for {seat.label}")
