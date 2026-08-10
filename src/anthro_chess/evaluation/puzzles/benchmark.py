"""Checkpoint rating response against the owned Lichess puzzle set."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple, Protocol

import chess
import numpy as np
import torch
from pydantic import Field, StrictInt, model_validator
from torch import Tensor

from anthro_chess.chess import ACTION_VOCABULARY_SIZE, encode_move, legal_action_ids
from anthro_chess.config import ResolvedConfig
from anthro_chess.data import DecisionContext, DecisionHistory
from anthro_chess.data.artifacts import read_normalized_rows
from anthro_chess.data.schema import NormalizedColumn
from anthro_chess.evaluation.curves import (
    CurveComparison,
    CurveQuantity,
    CurveSpec,
    Observation,
    compare_curves,
)
from anthro_chess.evaluation.noise import NoiseConfig
from anthro_chess.evaluation.puzzles.dataset import (
    Puzzle,
    PuzzleSelection,
    PuzzleSet,
    PuzzleSetError,
    load_puzzle_set,
    select_puzzles,
)
from anthro_chess.evaluation.recording import ResultRecording, checkpoint_reference
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
    DatasetReference,
    Measurement,
    ResultEnvelope,
    ResultRecordError,
    ResultsStoreError,
    dataset_reference,
    measurement,
    projection_content_digest,
)
from anthro_chess.evaluation.results.metrics import (
    PUZZLE_GREEDY_CURVE_DISTANCE,
    PUZZLE_GREEDY_FIRST_MOVE_ACCURACY,
    PUZZLE_GREEDY_LINE_COMPLETION,
    PUZZLE_GREEDY_RATING_ORDER_ACCURACY,
    PUZZLE_GREEDY_RATING_SLOPE,
    PUZZLE_RESPONSE_PROJECTION,
    PUZZLE_SAMPLED_CURVE_DISTANCE,
    PUZZLE_SAMPLED_FIRST_MOVE_SOLVE_RATE,
    PUZZLE_SAMPLED_LINE_COMPLETION,
    PUZZLE_SAMPLED_RATING_ORDER_ACCURACY,
    PUZZLE_SAMPLED_RATING_SLOPE,
    PUZZLE_TRAINING_OVERLAP_RATE,
)
from anthro_chess.evaluation.results.noise import bounded_spread
from anthro_chess.evaluation.selection import CheckpointSelection
from anthro_chess.inference import (
    CheckpointModelRunner,
    ModelRunnerError,
)

logger = logging.getLogger(__name__)

PUZZLE_BENCHMARK_VERSION = 1
PUZZLE_CURVE_VERSION = 1
PUZZLE_CURVE_NEIGHBOURS = 4000
PUZZLE_CURVE_GRID = tuple(float(rating) for rating in range(850, 2800, 100))
PUZZLE_KIND = "puzzle-rating-response"
#: How far past the scored puzzle ratings a fitted rating may land, in rating
#: points. Shared so the tabulated inverse the resolution reads covers exactly
#: the range the fit searches.
_FIT_SEARCH_MARGIN = 2400.0
#: The rating difference worth a factor of ten in the odds of scoring. The
#: tabulated inverse has to use the same scale as the fit it inverts, or it
#: would report a spread belonging to a different quantity.
_RATING_SCALE = 400.0
PUZZLE_BENCHMARK = BenchmarkReference(
    name=PUZZLE_KIND,
    version=PUZZLE_BENCHMARK_VERSION,
)


class PuzzleBenchmarkError(ValueError):
    """Raised when puzzle response cannot be measured safely."""


class PuzzlePredictionRunner(Protocol):
    """The target-free batch inference boundary used by the benchmark."""

    def predict_batch(
        self,
        contexts: Sequence[DecisionContext],
    ) -> tuple[Tensor, ...]: ...


class PuzzleBenchmarkConfig(CheckpointSelection):
    """Code-owned schema for ``anthro eval puzzles``."""

    puzzle_set: Path
    training_normalized: Path
    target_ratings: tuple[StrictInt, ...] = (1000, 1400, 1800, 2200)
    #: How many puzzles to score at each exact puzzle rating, or every one of
    #: them. Two rather than one is the floor because the response redraw
    #: stratifies by exact rating and takes one fewer than each stratum holds,
    #: so a stratum of one leaves nothing to draw and the reading would come
    #: back with no resolution beside it at all.
    puzzles_per_rating: StrictInt | None = Field(default=None, ge=2)
    reference_temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=3.0,
        allow_inf_nan=False,
    )
    inference_batch_size: StrictInt = Field(default=32, ge=1)
    noise: NoiseConfig = NoiseConfig()

    @model_validator(mode="after")
    def _validate_ratings(self) -> PuzzleBenchmarkConfig:
        ratings = self.target_ratings
        if len(ratings) < 2:
            raise ValueError("target_ratings needs at least two configured ratings")
        if any(rating < 0 for rating in ratings):
            raise ValueError("target ratings must be nonnegative")
        if tuple(sorted(set(ratings))) != ratings:
            raise ValueError("target_ratings must be sorted and unique")
        return self


@dataclass(frozen=True)
class PuzzleBandResult:
    """Solve rates in one fixed puzzle-rating band."""

    name: str
    lower: int
    upper: int
    puzzles: int
    human_expected_score: float
    greedy_first_move_accuracy: float
    greedy_line_completion: float
    sampled_first_move_solve_rate: float
    sampled_line_completion: float

    def as_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "puzzles": self.puzzles,
            "human_expected_score": self.human_expected_score,
            "greedy_first_move_accuracy": self.greedy_first_move_accuracy,
            "greedy_line_completion": self.greedy_line_completion,
            "sampled_first_move_solve_rate": self.sampled_first_move_solve_rate,
            "sampled_line_completion": self.sampled_line_completion,
        }


@dataclass(frozen=True)
class PuzzleCurvePoint:
    """One continuous local estimate over exact puzzle ratings."""

    puzzle_rating: float
    bandwidth: float
    effective_sample_size: float
    human_expected_score: float
    greedy_first_move_accuracy: float
    greedy_line_completion: float
    sampled_first_move_solve_rate: float
    sampled_line_completion: float

    def as_record(self) -> dict[str, object]:
        return {
            "puzzle_rating": self.puzzle_rating,
            "bandwidth": self.bandwidth,
            "effective_sample_size": self.effective_sample_size,
            "human_expected_score": self.human_expected_score,
            "greedy_first_move_accuracy": self.greedy_first_move_accuracy,
            "greedy_line_completion": self.greedy_line_completion,
            "sampled_first_move_solve_rate": self.sampled_first_move_solve_rate,
            "sampled_line_completion": self.sampled_line_completion,
        }


@dataclass(frozen=True)
class PuzzleRatingResult:
    """The response at one configured model rating."""

    target_rating: int
    human_expected_score: float
    greedy_first_move_accuracy: float
    greedy_line_completion: float
    greedy_fitted_puzzle_rating: float
    sampled_first_move_solve_rate: float
    sampled_line_completion: float
    sampled_fitted_puzzle_rating: float
    greedy_curve_distance: float
    sampled_curve_distance: float
    curve: tuple[PuzzleCurvePoint, ...]
    bands: tuple[PuzzleBandResult, ...]

    def as_record(self) -> dict[str, object]:
        return {
            "target_rating": self.target_rating,
            "human_expected_score": self.human_expected_score,
            "greedy_first_move_accuracy": self.greedy_first_move_accuracy,
            "greedy_line_completion": self.greedy_line_completion,
            "greedy_fitted_puzzle_rating": self.greedy_fitted_puzzle_rating,
            "sampled_first_move_solve_rate": self.sampled_first_move_solve_rate,
            "sampled_line_completion": self.sampled_line_completion,
            "sampled_fitted_puzzle_rating": self.sampled_fitted_puzzle_rating,
            "greedy_curve_distance": self.greedy_curve_distance,
            "sampled_curve_distance": self.sampled_curve_distance,
            "curve": [point.as_record() for point in self.curve],
            "bands": [band.as_record() for band in self.bands],
        }


@dataclass(frozen=True)
class PuzzleRatingResolution:
    """How far a redraw moves the fit at one configured rating.

    ``None`` where no redraw moved the quantity at all. A spread of zero would
    read as perfect resolution, which is the opposite of what was observed.
    """

    target_rating: int
    greedy_fitted_puzzle_rating: float | None
    sampled_fitted_puzzle_rating: float | None

    def as_record(self) -> dict[str, object]:
        return {
            "target_rating": self.target_rating,
            "greedy_fitted_puzzle_rating": self.greedy_fitted_puzzle_rating,
            "sampled_fitted_puzzle_rating": self.sampled_fitted_puzzle_rating,
        }


@dataclass(frozen=True)
class PuzzleResponseResolution:
    """What the scored puzzles can resolve of the response the fit yields.

    Every configured rating is scored on the same puzzles, so the response is a
    comparison within one reading and a redraw of those puzzles moves all of it
    together. Each replicate therefore refits every configured rating from one
    draw, and the spreads below are of the reduction the reading itself ran.
    """

    resamples: int
    puzzles: int
    coverage: float
    confidence: float
    ratings: tuple[PuzzleRatingResolution, ...]
    greedy_rating_slope: float | None
    sampled_rating_slope: float | None
    greedy_order_accuracy: float | None
    sampled_order_accuracy: float | None

    @property
    def widest_greedy_fit(self) -> float | None:
        return _widest(rating.greedy_fitted_puzzle_rating for rating in self.ratings)

    @property
    def widest_sampled_fit(self) -> float | None:
        return _widest(rating.sampled_fitted_puzzle_rating for rating in self.ratings)

    def as_record(self) -> dict[str, object]:
        return {
            "resamples": self.resamples,
            "puzzles": self.puzzles,
            "coverage": self.coverage,
            "confidence": self.confidence,
            "ratings": [rating.as_record() for rating in self.ratings],
            "greedy_rating_slope": self.greedy_rating_slope,
            "sampled_rating_slope": self.sampled_rating_slope,
            "greedy_order_accuracy": self.greedy_order_accuracy,
            "sampled_order_accuracy": self.sampled_order_accuracy,
        }


@dataclass(frozen=True)
class PuzzleBenchmarkResult:
    """The response grid, provenance, and durable result envelope."""

    checkpoint: CheckpointReference
    dataset: DatasetReference
    puzzle_set: Mapping[str, object]
    selection: PuzzleSelection
    reference_temperature: float
    ratings: tuple[PuzzleRatingResult, ...]
    greedy_rating_slope: float
    sampled_rating_slope: float
    greedy_order_accuracy: float
    sampled_order_accuracy: float
    training_games: int
    overlapping_puzzles: int
    overlap_rate: float
    resolution: PuzzleResponseResolution | None = None
    envelopes: tuple[ResultEnvelope, ...] = ()
    recorded_paths: tuple[Path, ...] = ()
    detail_paths: tuple[Path, ...] = ()

    def as_record(self) -> dict[str, object]:
        return {
            "version": PUZZLE_BENCHMARK_VERSION,
            "checkpoint": self.checkpoint.model_dump(mode="json"),
            "dataset": self.dataset.model_dump(mode="json"),
            "puzzle_set": dict(self.puzzle_set),
            "selection": self.selection.as_record(),
            "reference_temperature": self.reference_temperature,
            "ratings": [rating.as_record() for rating in self.ratings],
            "greedy_rating_slope": self.greedy_rating_slope,
            "sampled_rating_slope": self.sampled_rating_slope,
            "greedy_order_accuracy": self.greedy_order_accuracy,
            "sampled_order_accuracy": self.sampled_order_accuracy,
            "response_resolution": (
                None if self.resolution is None else self.resolution.as_record()
            ),
            "training_overlap": {
                "training_games": self.training_games,
                "overlapping_puzzles": self.overlapping_puzzles,
                "rate": self.overlap_rate,
            },
            "recorded": [str(path) for path in self.recorded_paths],
        }


@dataclass(frozen=True)
class _DecisionTask:
    """One puzzle decision, derived once and scored at every target rating.

    ``accepted_indices`` positions the accepted actions within
    ``legal_action_ids``, which is the coordinate the scored logits arrive in.
    """

    puzzle_id: str
    accepted_indices: tuple[int, ...]
    legal_action_ids: tuple[int, ...]
    rating_free_context: DecisionContext


@dataclass(frozen=True)
class _DecisionScore:
    greedy_correct: float
    sampled_probability: float


@dataclass(frozen=True)
class _PuzzleScore:
    puzzle: Puzzle
    greedy_first: float
    greedy_line: float
    sampled_first: float
    sampled_line: float


@dataclass(frozen=True)
class _ScoredRating:
    result: PuzzleRatingResult
    scores: tuple[_PuzzleScore, ...]


def benchmark_puzzles(
    resolved_config: ResolvedConfig[PuzzleBenchmarkConfig],
    *,
    run_root: Path | None = None,
    recording: ResultRecording,
) -> PuzzleBenchmarkResult:
    """Measure and optionally record puzzle response for one checkpoint."""

    config = resolved_config.value
    try:
        puzzle_set = load_puzzle_set(config.puzzle_set)
        selection = select_puzzles(puzzle_set, config.puzzles_per_rating)
        runner = CheckpointModelRunner.load(config.model, run_root=run_root)
        training_games, overlapping = _training_overlap(
            selection.puzzles,
            config.training_normalized,
        )
        tasks = _decision_tasks(selection.puzzles)
        scored_ratings = tuple(
            _score_rating(
                puzzle_set,
                selection.puzzles,
                runner,
                tasks,
                target_rating=target_rating,
                temperature=config.reference_temperature,
                batch_size=config.inference_batch_size,
            )
            for target_rating in config.target_ratings
        )
        ratings = tuple(scored.result for scored in scored_ratings)
        component = projection_content_digest(
            (puzzle.as_projection_record() for puzzle in selection.puzzles),
            PUZZLE_RESPONSE_PROJECTION,
        )
        checkpoint = checkpoint_reference(runner, label=config.checkpoint_label)
        data = _dataset_reference(puzzle_set, selection, component)
    except (
        ModelRunnerError,
        OSError,
        PuzzleSetError,
        ResultRecordError,
        ResultsStoreError,
        ValueError,
    ) as error:
        if isinstance(error, PuzzleBenchmarkError):
            raise
        raise PuzzleBenchmarkError(str(error)) from error

    greedy_fitted = [item.greedy_fitted_puzzle_rating for item in ratings]
    sampled_fitted = [item.sampled_fitted_puzzle_rating for item in ratings]
    greedy_slope = _slope(config.target_ratings, greedy_fitted)
    sampled_slope = _slope(config.target_ratings, sampled_fitted)
    greedy_order = _order_accuracy(greedy_fitted)
    sampled_order = _order_accuracy(sampled_fitted)
    overlap_rate = overlapping / selection.selected_puzzles
    resolution = _response_resolution(scored_ratings, config.noise)
    result = PuzzleBenchmarkResult(
        checkpoint=checkpoint,
        dataset=data,
        puzzle_set=puzzle_set.identity(),
        selection=selection,
        reference_temperature=config.reference_temperature,
        ratings=ratings,
        greedy_rating_slope=greedy_slope,
        sampled_rating_slope=sampled_slope,
        greedy_order_accuracy=greedy_order,
        sampled_order_accuracy=sampled_order,
        training_games=training_games,
        overlapping_puzzles=overlapping,
        overlap_rate=overlap_rate,
        resolution=resolution,
    )
    recorder = recording.measuring(
        checkpoint,
        kind=PUZZLE_KIND,
        benchmark=PUZZLE_BENCHMARK,
    )
    recorder.add(
        _measurements(result, component),
        payload=result.as_record,
        description=(
            "Puzzle-rating grid, human reference curve, rating-band "
            "response, and source-game overlap provenance."
        ),
        data=data,
    )
    return result


def score_puzzle_set(
    puzzle_set: PuzzleSet,
    runner: PuzzlePredictionRunner,
    *,
    target_ratings: Sequence[int],
    temperature: float,
    batch_size: int = 32,
) -> tuple[PuzzleRatingResult, ...]:
    """Score an injected set and runner for fixtures and library consumers."""

    tasks = _decision_tasks(puzzle_set.puzzles)
    return tuple(
        _score_rating(
            puzzle_set,
            puzzle_set.puzzles,
            runner,
            tasks,
            target_rating=rating,
            temperature=temperature,
            batch_size=batch_size,
        ).result
        for rating in target_ratings
    )


def expected_score(player_rating: float, puzzle_rating: float) -> float:
    """Return the Glicko/Elo expected score used as the human reference."""

    exponent = (puzzle_rating - player_rating) / _RATING_SCALE
    if exponent > 15:
        return 0.0
    if exponent < -15:
        return 1.0
    return float(1.0 / (1.0 + math.pow(10.0, exponent)))


def fitted_rating(
    puzzle_ratings: Sequence[int],
    outcomes: Sequence[float],
) -> float:
    """Fit the puzzle rating whose expected successes match observed outcomes."""

    if not puzzle_ratings or len(puzzle_ratings) != len(outcomes):
        raise ValueError("fitted rating needs aligned, nonempty observations")
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0 for value in outcomes
    ):
        raise ValueError("puzzle outcomes must be finite probabilities")
    target = sum(outcomes)
    lower = min(puzzle_ratings) - _FIT_SEARCH_MARGIN
    upper = max(puzzle_ratings) + _FIT_SEARCH_MARGIN
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        predicted = sum(expected_score(midpoint, rating) for rating in puzzle_ratings)
        if predicted < target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _score_rating(
    puzzle_set: PuzzleSet,
    puzzles: Sequence[Puzzle],
    runner: PuzzlePredictionRunner,
    tasks: Sequence[_DecisionTask],
    *,
    target_rating: int,
    temperature: float,
    batch_size: int,
) -> _ScoredRating:
    decisions: dict[str, list[_DecisionScore]] = {
        puzzle.puzzle_id: [] for puzzle in puzzles
    }
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        logits = runner.predict_batch(
            [
                replace(task.rating_free_context, target_rating=target_rating)
                for task in batch
            ]
        )
        if len(logits) != len(batch):
            raise PuzzleBenchmarkError(
                "model runner returned the wrong number of puzzle predictions"
            )
        for task, predicted in zip(batch, logits, strict=True):
            decisions[task.puzzle_id].append(
                _score_decision(task, predicted, temperature)
            )

    scores = tuple(
        _puzzle_score(puzzle, decisions[puzzle.puzzle_id]) for puzzle in puzzles
    )
    puzzle_ratings = [score.puzzle.rating for score in scores]
    greedy_lines = [score.greedy_line for score in scores]
    sampled_lines = [score.sampled_line for score in scores]
    curve, greedy_curve_distance, sampled_curve_distance = _continuous_curve(
        puzzle_set.puzzles,
        scores,
        target_rating,
    )
    return _ScoredRating(
        result=PuzzleRatingResult(
            target_rating=target_rating,
            human_expected_score=_mean(
                [expected_score(target_rating, rating) for rating in puzzle_ratings]
            ),
            greedy_first_move_accuracy=_mean([score.greedy_first for score in scores]),
            greedy_line_completion=_mean(greedy_lines),
            greedy_fitted_puzzle_rating=fitted_rating(puzzle_ratings, greedy_lines),
            sampled_first_move_solve_rate=_mean(
                [score.sampled_first for score in scores]
            ),
            sampled_line_completion=_mean(sampled_lines),
            sampled_fitted_puzzle_rating=fitted_rating(puzzle_ratings, sampled_lines),
            greedy_curve_distance=greedy_curve_distance,
            sampled_curve_distance=sampled_curve_distance,
            curve=curve,
            bands=_band_results(puzzle_set, scores, target_rating),
        ),
        scores=scores,
    )


def _continuous_curve(
    puzzles: Sequence[Puzzle],
    scores: Sequence[_PuzzleScore],
    target_rating: int,
) -> tuple[tuple[PuzzleCurvePoint, ...], float, float]:
    """Estimate continuous response with the shared frozen curve machinery.

    The reference is the whole set rather than the puzzles scored, which is the
    rule the generated-play and termination curves already follow: at a
    neighbour-count bandwidth the reference's size is a smoothing radius rather
    than a sample size, so shrinking it would re-smooth the curve instead of
    sampling it. Holding it costs nothing here, because this reference is
    analytic rather than played, and it leaves a subsampled reading estimated
    at exactly the radii a full one uses.
    """

    neighbours = min(PUZZLE_CURVE_NEIGHBOURS, len(puzzles))
    if neighbours < 2:
        raise PuzzleBenchmarkError("a puzzle response curve needs at least two puzzles")
    ratings = [puzzle.rating for puzzle in puzzles]
    low = min(ratings)
    high = max(ratings)
    grid = tuple(rating for rating in PUZZLE_CURVE_GRID if low <= rating <= high)
    if len(grid) < 2:
        grid = (
            (float(low), float(high))
            if low < high
            else (float(low) - 0.5, float(high) + 0.5)
        )
    spec = CurveSpec(
        name="puzzle-solve-response",
        version=PUZZLE_CURVE_VERSION,
        quantity=CurveQuantity.SCALAR,
        neighbours=neighbours,
        grid=grid,
    )
    human = [
        Observation(
            rating=float(puzzle.rating),
            value=expected_score(target_rating, puzzle.rating),
        )
        for puzzle in puzzles
    ]

    def comparison(values: Sequence[float]) -> CurveComparison:
        return compare_curves(
            spec=spec,
            human=human,
            model=[
                Observation(rating=float(score.puzzle.rating), value=value)
                for score, value in zip(scores, values, strict=True)
            ],
            # The human curve is analytic rather than a sampled corpus. The
            # shared curve estimator is useful here, but its two-sample
            # bootstrap would invent uncertainty on the human side.
            resamples=0,
        )

    greedy_first = comparison([score.greedy_first for score in scores])
    greedy_line = comparison([score.greedy_line for score in scores])
    sampled_first = comparison([score.sampled_first for score in scores])
    sampled_line = comparison([score.sampled_line for score in scores])
    points: list[PuzzleCurvePoint] = []
    for (
        human_point,
        greedy_first_point,
        greedy_line_point,
        sampled_first_point,
        sampled_line_point,
    ) in zip(
        greedy_line.points,
        greedy_first.points,
        greedy_line.points,
        sampled_first.points,
        sampled_line.points,
        strict=True,
    ):
        points.append(
            PuzzleCurvePoint(
                puzzle_rating=human_point.rating,
                bandwidth=human_point.bandwidth,
                effective_sample_size=greedy_line_point.model.effective_sample_size,
                human_expected_score=_curve_value(human_point.human.value),
                greedy_first_move_accuracy=_curve_value(greedy_first_point.model.value),
                greedy_line_completion=_curve_value(greedy_line_point.model.value),
                sampled_first_move_solve_rate=_curve_value(
                    sampled_first_point.model.value
                ),
                sampled_line_completion=_curve_value(sampled_line_point.model.value),
            )
        )
    return (
        tuple(points),
        greedy_line.conditional_distance,
        sampled_line.conditional_distance,
    )


def _curve_value(value: float | None) -> float:
    if value is None:
        raise PuzzleBenchmarkError("puzzle response curve has an unsupported point")
    return value


def _decision_tasks(puzzles: Sequence[Puzzle]) -> tuple[_DecisionTask, ...]:
    tasks: list[_DecisionTask] = []
    for puzzle in puzzles:
        history = DecisionHistory(initial_fen=puzzle.initial_fen)
        for ply, move in enumerate(puzzle.moves):
            history.push(move)
            if ply % 2 != 0:
                continue
            accepted = _accepted_actions(history.board, puzzle.moves[ply + 1])
            legal = legal_action_ids(history.board)
            tasks.append(
                _DecisionTask(
                    puzzle_id=puzzle.puzzle_id,
                    accepted_indices=tuple(
                        legal.index(action_id) for action_id in accepted
                    ),
                    legal_action_ids=legal,
                    rating_free_context=history.context(target_rating=None),
                )
            )
    return tuple(tasks)


def _score_decision(
    task: _DecisionTask,
    logits: Tensor,
    temperature: float,
) -> _DecisionScore:
    observed = logits.detach().to(device="cpu", dtype=torch.float64)
    if (
        observed.shape != (ACTION_VOCABULARY_SIZE,)
        or not torch.isfinite(observed).all()
    ):
        raise PuzzleBenchmarkError(
            f"model returned invalid logits for puzzle {task.puzzle_id}"
        )
    candidates = torch.as_tensor(task.legal_action_ids, dtype=torch.long)
    candidate_logits = observed[candidates]
    greedy_index = int(torch.argmax(candidate_logits).item())
    greedy_correct = float(greedy_index in task.accepted_indices)
    if temperature == 0.0:
        probability = greedy_correct
    else:
        accepted = torch.as_tensor(task.accepted_indices, dtype=torch.long)
        probability = float(
            torch.softmax(candidate_logits / temperature, dim=0)[accepted].sum().item()
        )
    return _DecisionScore(
        greedy_correct=greedy_correct,
        sampled_probability=probability,
    )


def _accepted_actions(
    board: chess.Board,
    recorded: chess.Move,
) -> tuple[int, ...]:
    """Return the verified move, or every mate when the puzzle is mate in one."""

    after = board.copy(stack=False)
    after.push(recorded)
    if not after.is_checkmate():
        return (encode_move(recorded),)
    accepted: list[int] = []
    for candidate in board.legal_moves:
        result = board.copy(stack=False)
        result.push(candidate)
        if result.is_checkmate():
            accepted.append(encode_move(candidate))
    return tuple(sorted(accepted))


def _puzzle_score(
    puzzle: Puzzle,
    decisions: Sequence[_DecisionScore],
) -> _PuzzleScore:
    if len(decisions) != len(puzzle.solution_moves):
        raise PuzzleBenchmarkError(
            f"puzzle {puzzle.puzzle_id} has incomplete decision scores"
        )
    return _PuzzleScore(
        puzzle=puzzle,
        greedy_first=decisions[0].greedy_correct,
        greedy_line=float(all(score.greedy_correct == 1.0 for score in decisions)),
        sampled_first=decisions[0].sampled_probability,
        sampled_line=math.prod(score.sampled_probability for score in decisions),
    )


def _band_results(
    puzzle_set: PuzzleSet,
    scores: Sequence[_PuzzleScore],
    target_rating: int,
) -> tuple[PuzzleBandResult, ...]:
    selection = puzzle_set.selection
    try:
        minimum = int(selection["minimum_rating"])
        maximum = int(selection["maximum_rating_exclusive"])
        width = int(selection["local_precision_span"])
    except (KeyError, TypeError, ValueError) as error:
        raise PuzzleBenchmarkError(
            f"puzzle set has invalid rating-band metadata: {error}"
        ) from error
    results: list[PuzzleBandResult] = []
    for lower in range(minimum, maximum, width):
        upper = lower + width
        band = [score for score in scores if lower <= score.puzzle.rating < upper]
        if not band:
            continue
        results.append(
            PuzzleBandResult(
                name=f"{lower}_to_{upper - 1}",
                lower=lower,
                upper=upper,
                puzzles=len(band),
                human_expected_score=_mean(
                    [
                        expected_score(target_rating, score.puzzle.rating)
                        for score in band
                    ]
                ),
                greedy_first_move_accuracy=_mean(
                    [score.greedy_first for score in band]
                ),
                greedy_line_completion=_mean([score.greedy_line for score in band]),
                sampled_first_move_solve_rate=_mean(
                    [score.sampled_first for score in band]
                ),
                sampled_line_completion=_mean([score.sampled_line for score in band]),
            )
        )
    return tuple(results)


def _training_overlap(puzzles: Sequence[Puzzle], path: Path) -> tuple[int, int]:
    keys: set[str] = set()
    games = 0
    columns = (
        NormalizedColumn.SOURCE_GAME_KEY.value,
        NormalizedColumn.SOURCE_ID.value,
        NormalizedColumn.SPLIT.value,
    )
    for row in read_normalized_rows(path, columns=columns):
        if (
            row[NormalizedColumn.SOURCE_ID] == "lichess"
            and row[NormalizedColumn.SPLIT] != "test"
        ):
            games += 1
            keys.add(str(row[NormalizedColumn.SOURCE_GAME_KEY]))
    if games == 0:
        raise PuzzleBenchmarkError(
            "training selection contains no non-test Lichess games"
        )
    overlapping = sum(puzzle.source_game_key in keys for puzzle in puzzles)
    return games, overlapping


def _dataset_reference(
    puzzle_set: PuzzleSet,
    selection: PuzzleSelection,
    component: DataComponent,
) -> DatasetReference:
    digest = sha256()
    for puzzle in selection.puzzles:
        digest.update(f"{puzzle.puzzle_id}\n".encode())
    return dataset_reference(
        pool_id=puzzle_set.name,
        pool_version=puzzle_set.version,
        view=(
            f"per-rating-{selection.puzzles_per_rating}"
            if selection.subsampled
            else "canonical"
        ),
        selected_games=selection.selected_puzzles,
        game_ids_sha256=digest.hexdigest(),
        components=[component],
    )


def _measurements(
    result: PuzzleBenchmarkResult,
    component: DataComponent,
) -> tuple[Measurement, ...]:
    ratings = result.ratings
    sample_size = len(ratings) * result.dataset.selected_games
    values = (
        (
            PUZZLE_GREEDY_FIRST_MOVE_ACCURACY,
            _mean([item.greedy_first_move_accuracy for item in ratings]),
            sample_size,
        ),
        (
            PUZZLE_GREEDY_LINE_COMPLETION,
            _mean([item.greedy_line_completion for item in ratings]),
            sample_size,
        ),
        (
            PUZZLE_SAMPLED_FIRST_MOVE_SOLVE_RATE,
            _mean([item.sampled_first_move_solve_rate for item in ratings]),
            sample_size,
        ),
        (
            PUZZLE_SAMPLED_LINE_COMPLETION,
            _mean([item.sampled_line_completion for item in ratings]),
            sample_size,
        ),
        (
            PUZZLE_GREEDY_RATING_SLOPE,
            result.greedy_rating_slope,
            len(ratings),
        ),
        (
            PUZZLE_SAMPLED_RATING_SLOPE,
            result.sampled_rating_slope,
            len(ratings),
        ),
        (
            PUZZLE_GREEDY_RATING_ORDER_ACCURACY,
            result.greedy_order_accuracy,
            len(ratings),
        ),
        (
            PUZZLE_SAMPLED_RATING_ORDER_ACCURACY,
            result.sampled_order_accuracy,
            len(ratings),
        ),
        (
            PUZZLE_GREEDY_CURVE_DISTANCE,
            _mean([item.greedy_curve_distance for item in ratings]),
            sample_size,
        ),
        (
            PUZZLE_SAMPLED_CURVE_DISTANCE,
            _mean([item.sampled_curve_distance for item in ratings]),
            sample_size,
        ),
        (
            PUZZLE_TRAINING_OVERLAP_RATE,
            result.overlap_rate,
            result.dataset.selected_games,
        ),
    )
    return tuple(
        measurement(
            definition.identifier,
            value,
            data=component,
            sample_size=measurement_size,
        )
        for definition, value, measurement_size in values
    )


def _stratum_buckets(
    strata: Sequence[Hashable],
) -> tuple[tuple[int, np.ndarray], ...]:
    """Group equal-size strata so one bootstrap draw remains vectorized."""

    grouped: dict[Hashable, list[int]] = {}
    for index, stratum in enumerate(strata):
        grouped.setdefault(stratum, []).append(index)
    by_size: dict[int, list[list[int]]] = {}
    for indices in grouped.values():
        by_size.setdefault(len(indices), []).append(indices)
    return tuple(
        (size, np.asarray(groups, dtype=np.int64))
        for size, groups in sorted(by_size.items())
    )


def _draw_multiplicity(
    generator: np.random.Generator,
    *,
    units: int,
    buckets: Sequence[tuple[int, np.ndarray]],
) -> np.ndarray:
    """Redraw within every stratum and return each unit's weight in the draw.

    Holding every stratum at its own weight is what keeps the design of a
    stratified selection fixed across replicates. Each stratum draws one fewer
    than it holds and the counts are scaled back up, which removes the
    ``(n-1)/n`` understatement a plug-in draw carries and is exact for a mean.
    `docs/decisions/0039-stratifying-the-ladder-redraw-costs-more-than-it-removes.md`
    measures what leaving it in costs: at a three-unit stratum the plug-in draw
    reported 83% of the spread it should. Taking one fewer is also why a
    stratum has to hold at least two; the caller checks that, since it is the
    one that can report the reading without a resolution instead of failing.
    """

    multiplicity = np.zeros(units, dtype=np.float64)
    for size, grouped_indices in buckets:
        taken = size - 1
        offsets = generator.integers(0, size, (grouped_indices.shape[0], taken))
        drawn = np.take_along_axis(grouped_indices, offsets, axis=1).ravel()
        multiplicity += np.bincount(drawn, minlength=units) * (size / taken)
    return multiplicity


def _response_resolution(
    scored_ratings: Sequence[_ScoredRating],
    config: NoiseConfig,
) -> PuzzleResponseResolution | None:
    """Refit resampled puzzles and read the spread of the response.

    The draw is stratified by exact puzzle rating, matching the selection
    design. That is also what makes the refit cheap: holding each stratum at
    its own size holds the scored rating composition fixed, so the fit's
    expected-score sum is one curve every replicate inverts rather than a
    bisection each has to run for itself.
    """

    if not config.enabled or len(scored_ratings) < 2:
        return None
    scores = scored_ratings[0].scores
    if len(scores) < 2:
        return None
    target_ratings = [scored.result.target_rating for scored in scored_ratings]
    puzzle_ratings = [score.puzzle.rating for score in scores]
    buckets = _stratum_buckets(puzzle_ratings)
    thinnest = min(size for size, _ in buckets)
    if thinnest < 2:
        # A fact about the set rather than a failed reading: only an artifact
        # built at one puzzle per rating reaches this, since the reading dial
        # already floors puzzles_per_rating at two.
        logger.warning(
            "Skipping puzzle response resolution: %d puzzle(s) at some exact "
            "rating, and a stratified redraw needs at least two",
            thinnest,
        )
        return None
    greedy = np.array(
        [[score.greedy_line for score in scored.scores] for scored in scored_ratings],
        dtype=np.float64,
    ).T
    sampled = np.array(
        [[score.sampled_line for score in scored.scores] for scored in scored_ratings],
        dtype=np.float64,
    ).T
    fitted, totals = _fit_curve(puzzle_ratings)
    generator = np.random.default_rng(config.seed)
    greedy_fits = np.empty((config.resamples, len(scored_ratings)), dtype=np.float64)
    sampled_fits = np.empty_like(greedy_fits)
    for index in range(config.resamples):
        multiplicity = _draw_multiplicity(
            generator,
            units=len(scores),
            buckets=buckets,
        )
        greedy_fits[index] = np.interp(multiplicity @ greedy, totals, fitted)
        sampled_fits[index] = np.interp(multiplicity @ sampled, totals, fitted)

    spread = partial(_bounded_spread, units=len(scores), config=config)
    greedy_response = [_reductions(target_ratings, row) for row in greedy_fits]
    sampled_response = [_reductions(target_ratings, row) for row in sampled_fits]
    return PuzzleResponseResolution(
        resamples=config.resamples,
        puzzles=len(scores),
        coverage=config.coverage,
        confidence=config.confidence,
        ratings=tuple(
            PuzzleRatingResolution(
                target_rating=target_rating,
                greedy_fitted_puzzle_rating=spread(greedy_fits[:, column]),
                sampled_fitted_puzzle_rating=spread(sampled_fits[:, column]),
            )
            for column, target_rating in enumerate(target_ratings)
        ),
        greedy_rating_slope=spread(_column(item.slope for item in greedy_response)),
        sampled_rating_slope=spread(_column(item.slope for item in sampled_response)),
        greedy_order_accuracy=spread(
            _column(item.order_accuracy for item in greedy_response)
        ),
        sampled_order_accuracy=spread(
            _column(item.order_accuracy for item in sampled_response)
        ),
    )


#: How finely the expected-score sum is tabulated before replicate fits are read
#: off it by interpolation, in puzzle-rating points. The sum is a smooth
#: logistic mixture, so this resolves a fit to far below the tenth of a rating
#: point the reading reports.
_FIT_CURVE_STEP = 0.5


def _fit_curve(puzzle_ratings: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate the fit's expected-score sum over the range it searches.

    ``fitted_rating`` bisects this sum for one target. A stratified redraw
    scores the same rating composition every time, so the sum is the same
    increasing function for every replicate and inverting it once serves all of
    them.
    """

    lower = float(min(puzzle_ratings)) - _FIT_SEARCH_MARGIN
    upper = float(max(puzzle_ratings)) + _FIT_SEARCH_MARGIN
    fitted = np.arange(lower, upper + _FIT_CURVE_STEP, _FIT_CURVE_STEP)
    totals = np.zeros_like(fitted)
    for rating, count in Counter(puzzle_ratings).items():
        totals += count / (1.0 + np.power(10.0, (rating - fitted) / _RATING_SCALE))
    return fitted, totals


def _bounded_spread(
    replicates: np.ndarray,
    *,
    units: int,
    config: NoiseConfig,
) -> float | None:
    """Return the conservative spread of one resampled quantity.

    The matched puzzles are the independent replicates rather than the draws
    taken from them, which is what the dispersion bound's degrees of freedom
    count. A quantity no draw moved returns ``None``: the resample observed
    that it could not move this number, not that a wider sample could not.
    """

    dispersion = float(np.std(replicates, ddof=1))
    if dispersion == 0.0:
        return None
    return bounded_spread(
        dispersion,
        degrees_of_freedom=units - 1,
        coverage=config.coverage,
        confidence=config.confidence,
    )


def _widest(spreads: Iterable[float | None]) -> float | None:
    """Return the widest estimated spread, or ``None`` if none was estimated.

    An unmoved quantity is narrower than any estimate rather than wider, so it
    cannot be the widest and only decides the answer when it is the only one.
    """

    estimated = [spread for spread in spreads if spread is not None]
    return max(estimated) if estimated else None


class _Reduction(NamedTuple):
    """What one replicate's fitted ratings reduce to."""

    slope: float
    order_accuracy: float


def _reductions(target_ratings: Sequence[int], fits: np.ndarray) -> _Reduction:
    """Reduce one replicate through the reading's own reductions.

    Rather than a vectorized restatement of them, so a spread cannot end up
    describing a slightly different quantity from the number it is printed
    beside.
    """

    fitted = fits.tolist()
    return _Reduction(_slope(target_ratings, fitted), _order_accuracy(fitted))


def _column(values: Iterable[float]) -> np.ndarray:
    """Return one reduced quantity across every replicate."""

    return np.fromiter(values, dtype=np.float64)


def _slope(x_values: Sequence[int], y_values: Sequence[float]) -> float:
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0.0:
        raise PuzzleBenchmarkError("rating slope needs distinct target ratings")
    return (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / denominator
    )


def _order_accuracy(values: Sequence[float]) -> float:
    outcomes: list[float] = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            difference = values[right] - values[left]
            outcomes.append(1.0 if difference > 0 else 0.5 if difference == 0 else 0.0)
    return _mean(outcomes)


def _mean(values: Sequence[float] | Sequence[int]) -> float:
    if not values:
        raise PuzzleBenchmarkError("cannot average an empty puzzle measurement")
    return float(sum(values) / len(values))
