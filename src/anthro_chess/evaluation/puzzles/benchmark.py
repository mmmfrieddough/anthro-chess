"""Checkpoint rating response against the owned Lichess puzzle set."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import chess
import torch
from pydantic import Field, StrictInt, model_validator
from torch import Tensor

from anthro_chess.chess import ACTION_VOCABULARY_SIZE, encode_move, legal_action_ids
from anthro_chess.config import ConfigModel, ResolvedConfig
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
from anthro_chess.evaluation.noise import (
    GameTotals,
    MetricTotal,
    NoiseConfig,
    bootstrap_floors,
)
from anthro_chess.evaluation.puzzles.dataset import Puzzle, PuzzleSet, load_puzzle_set
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
    DatasetReference,
    DetailReference,
    DetailStore,
    Measurement,
    NoiseFloor,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    ResultsStoreError,
    build_result,
    configuration_reference,
    dataset_reference,
    default_checkpoint_label,
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
from anthro_chess.inference import (
    CheckpointModelRunner,
    ModelRunnerConfig,
    ModelRunnerError,
)

PUZZLE_BENCHMARK_VERSION = 1
PUZZLE_CURVE_VERSION = 1
PUZZLE_CURVE_NEIGHBOURS = 4000
PUZZLE_CURVE_GRID = tuple(float(rating) for rating in range(850, 2800, 100))
PUZZLE_KIND = "puzzle-rating-response"
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


class PuzzleBenchmarkConfig(ConfigModel):
    """Code-owned schema for ``anthro eval puzzles``."""

    model: ModelRunnerConfig = ModelRunnerConfig()
    checkpoint_label: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    puzzle_set: Path
    training_normalized: Path
    target_ratings: tuple[StrictInt, ...] = (1000, 1400, 1800, 2200)
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
class PuzzleBenchmarkResult:
    """The response grid, provenance, and durable result envelope."""

    checkpoint: CheckpointReference
    dataset: DatasetReference
    puzzle_set: Mapping[str, object]
    reference_temperature: float
    ratings: tuple[PuzzleRatingResult, ...]
    greedy_rating_slope: float
    sampled_rating_slope: float
    greedy_order_accuracy: float
    sampled_order_accuracy: float
    training_games: int
    overlapping_puzzles: int
    overlap_rate: float
    envelope: ResultEnvelope | None
    recorded_path: Path | None
    detail_path: Path | None

    def as_record(self) -> dict[str, object]:
        return {
            "version": PUZZLE_BENCHMARK_VERSION,
            "checkpoint": self.checkpoint.model_dump(mode="json"),
            "dataset": self.dataset.model_dump(mode="json"),
            "puzzle_set": dict(self.puzzle_set),
            "reference_temperature": self.reference_temperature,
            "ratings": [rating.as_record() for rating in self.ratings],
            "greedy_rating_slope": self.greedy_rating_slope,
            "sampled_rating_slope": self.sampled_rating_slope,
            "greedy_order_accuracy": self.greedy_order_accuracy,
            "sampled_order_accuracy": self.sampled_order_accuracy,
            "training_overlap": {
                "training_games": self.training_games,
                "overlapping_puzzles": self.overlapping_puzzles,
                "rate": self.overlap_rate,
            },
            "recorded": (
                None if self.recorded_path is None else str(self.recorded_path)
            ),
        }


@dataclass(frozen=True)
class _DecisionTask:
    puzzle_id: str
    solution_index: int
    accepted_action_ids: tuple[int, ...]
    legal_action_ids: tuple[int, ...]
    context: DecisionContext


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
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> PuzzleBenchmarkResult:
    """Measure and optionally record puzzle response for one checkpoint."""

    config = resolved_config.value
    puzzle_set = load_puzzle_set(config.puzzle_set)
    try:
        runner = CheckpointModelRunner.load(config.model, run_root=run_root)
        training_games, overlapping = _training_overlap(
            puzzle_set,
            config.training_normalized,
        )
        scored_ratings = tuple(
            _score_rating(
                puzzle_set,
                runner,
                target_rating=target_rating,
                temperature=config.reference_temperature,
                batch_size=config.inference_batch_size,
            )
            for target_rating in config.target_ratings
        )
        ratings = tuple(scored.result for scored in scored_ratings)
        component = projection_content_digest(
            (puzzle.as_projection_record() for puzzle in puzzle_set.puzzles),
            PUZZLE_RESPONSE_PROJECTION,
        )
        checkpoint = _checkpoint_reference(config, runner)
        data = _dataset_reference(puzzle_set, component)
    except (
        ModelRunnerError,
        OSError,
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
    overlap_rate = overlapping / len(puzzle_set.puzzles)
    result = PuzzleBenchmarkResult(
        checkpoint=checkpoint,
        dataset=data,
        puzzle_set=puzzle_set.identity(),
        reference_temperature=config.reference_temperature,
        ratings=ratings,
        greedy_rating_slope=greedy_slope,
        sampled_rating_slope=sampled_slope,
        greedy_order_accuracy=greedy_order,
        sampled_order_accuracy=sampled_order,
        training_games=training_games,
        overlapping_puzzles=overlapping,
        overlap_rate=overlap_rate,
        envelope=None,
        recorded_path=None,
        detail_path=None,
    )
    configuration = configuration_reference(
        resolved_config.as_record(),
        source=resolved_config.provenance.source,
        overrides=resolved_config.provenance.overrides,
    )
    recorded_at = datetime.now(tz=UTC)
    detail_reference = _write_detail(
        detail,
        result,
        recorded_at=recorded_at,
    )
    measurements = _measurements(
        result,
        component,
        scored_ratings,
        noise=config.noise,
    )
    envelope = build_result(
        kind=PUZZLE_KIND,
        benchmark=PUZZLE_BENCHMARK,
        checkpoint=checkpoint,
        configuration=configuration,
        data=data,
        measurements=measurements,
        detail=detail_reference,
        recorded_at=recorded_at,
    )
    recorded_path = store.append(envelope) if store is not None else None
    return replace(
        result,
        envelope=envelope,
        recorded_path=recorded_path,
        detail_path=(None if detail_reference is None else Path(detail_reference.path)),
    )


def score_puzzle_set(
    puzzle_set: PuzzleSet,
    runner: PuzzlePredictionRunner,
    *,
    target_ratings: Sequence[int],
    temperature: float,
    batch_size: int = 32,
) -> tuple[PuzzleRatingResult, ...]:
    """Score an injected set and runner for fixtures and library consumers."""

    return tuple(
        _score_rating(
            puzzle_set,
            runner,
            target_rating=rating,
            temperature=temperature,
            batch_size=batch_size,
        ).result
        for rating in target_ratings
    )


def expected_score(player_rating: float, puzzle_rating: float) -> float:
    """Return the Glicko/Elo expected score used as the human reference."""

    exponent = (puzzle_rating - player_rating) / 400.0
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
    lower = min(puzzle_ratings) - 2400.0
    upper = max(puzzle_ratings) + 2400.0
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
    runner: PuzzlePredictionRunner,
    *,
    target_rating: int,
    temperature: float,
    batch_size: int,
) -> _ScoredRating:
    tasks = _decision_tasks(puzzle_set, target_rating)
    decisions: dict[str, list[_DecisionScore]] = {
        puzzle.puzzle_id: [] for puzzle in puzzle_set.puzzles
    }
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        logits = runner.predict_batch([task.context for task in batch])
        if len(logits) != len(batch):
            raise PuzzleBenchmarkError(
                "model runner returned the wrong number of puzzle predictions"
            )
        for task, predicted in zip(batch, logits, strict=True):
            decisions[task.puzzle_id].append(
                _score_decision(task, predicted, temperature)
            )

    scores = tuple(
        _puzzle_score(puzzle, decisions[puzzle.puzzle_id])
        for puzzle in puzzle_set.puzzles
    )
    puzzle_ratings = [score.puzzle.rating for score in scores]
    greedy_lines = [score.greedy_line for score in scores]
    sampled_lines = [score.sampled_line for score in scores]
    curve, greedy_curve_distance, sampled_curve_distance = _continuous_curve(
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
    scores: Sequence[_PuzzleScore],
    target_rating: int,
) -> tuple[tuple[PuzzleCurvePoint, ...], float, float]:
    """Estimate continuous response with the shared frozen curve machinery."""

    neighbours = min(PUZZLE_CURVE_NEIGHBOURS, len(scores))
    if neighbours < 2:
        raise PuzzleBenchmarkError("a puzzle response curve needs at least two puzzles")
    ratings = [score.puzzle.rating for score in scores]
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
            rating=float(score.puzzle.rating),
            value=expected_score(target_rating, score.puzzle.rating),
        )
        for score in scores
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


def _decision_tasks(
    puzzle_set: PuzzleSet,
    target_rating: int,
) -> tuple[_DecisionTask, ...]:
    tasks: list[_DecisionTask] = []
    for puzzle in puzzle_set.puzzles:
        history = DecisionHistory(initial_fen=puzzle.initial_fen)
        solution_index = 0
        for ply, move in enumerate(puzzle.moves):
            history.push(move)
            if ply % 2 != 0:
                continue
            target = puzzle.moves[ply + 1]
            tasks.append(
                _DecisionTask(
                    puzzle_id=puzzle.puzzle_id,
                    solution_index=solution_index,
                    accepted_action_ids=_accepted_actions(history.board, target),
                    legal_action_ids=legal_action_ids(history.board),
                    context=history.context(target_rating=target_rating),
                )
            )
            solution_index += 1
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
    greedy_action = task.legal_action_ids[greedy_index]
    accepted_indices = torch.as_tensor(
        [
            task.legal_action_ids.index(action_id)
            for action_id in task.accepted_action_ids
        ],
        dtype=torch.long,
    )
    if temperature == 0.0:
        probability = 1.0 if greedy_action in task.accepted_action_ids else 0.0
    else:
        probability = float(
            torch.softmax(candidate_logits / temperature, dim=0)[accepted_indices]
            .sum()
            .item()
        )
    return _DecisionScore(
        greedy_correct=float(greedy_action in task.accepted_action_ids),
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


def _training_overlap(puzzle_set: PuzzleSet, path: Path) -> tuple[int, int]:
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
    overlapping = sum(puzzle.source_game_key in keys for puzzle in puzzle_set.puzzles)
    return games, overlapping


def _checkpoint_reference(
    config: PuzzleBenchmarkConfig,
    runner: CheckpointModelRunner,
) -> CheckpointReference:
    run_id = runner.selection.run_path.name
    label = config.checkpoint_label or default_checkpoint_label(
        run_id,
        runner.global_step,
    )
    return CheckpointReference(
        label=label,
        step=runner.global_step,
        run_id=run_id,
        parameter_sha256=runner.parameter_sha256(),
    )


def _dataset_reference(
    puzzle_set: PuzzleSet,
    component: DataComponent,
) -> DatasetReference:
    digest = sha256()
    for puzzle in puzzle_set.puzzles:
        digest.update(f"{puzzle.puzzle_id}\n".encode())
    return dataset_reference(
        pool_id=puzzle_set.name,
        pool_version=puzzle_set.version,
        view="canonical",
        selected_games=len(puzzle_set.puzzles),
        game_ids_sha256=digest.hexdigest(),
        components=[component],
    )


def _measurements(
    result: PuzzleBenchmarkResult,
    component: DataComponent,
    scored_ratings: Sequence[_ScoredRating],
    *,
    noise: NoiseConfig,
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
    floors = _sampling_floors(scored_ratings, component, noise) if noise.enabled else {}
    return tuple(
        measurement(
            definition.identifier,
            value,
            data=component,
            sample_size=measurement_size,
            noise_floor=floors.get(definition.identifier),
        )
        for definition, value, measurement_size in values
    )


def _sampling_floors(
    scored_ratings: Sequence[_ScoredRating],
    component: DataComponent,
    config: NoiseConfig,
) -> dict[str, NoiseFloor]:
    if not scored_ratings:
        return {}
    puzzles = scored_ratings[0].scores
    metric_values = (
        (PUZZLE_GREEDY_FIRST_MOVE_ACCURACY, "greedy_first"),
        (PUZZLE_GREEDY_LINE_COMPLETION, "greedy_line"),
        (PUZZLE_SAMPLED_FIRST_MOVE_SOLVE_RATE, "sampled_first"),
        (PUZZLE_SAMPLED_LINE_COMPLETION, "sampled_line"),
    )
    totals: list[GameTotals] = []
    for index, puzzle_score in enumerate(puzzles):
        puzzle_id = puzzle_score.puzzle.puzzle_id
        ratings = [scored.scores[index] for scored in scored_ratings]
        if any(score.puzzle.puzzle_id != puzzle_id for score in ratings):
            raise PuzzleBenchmarkError("puzzle score grids are not aligned")
        totals.append(
            GameTotals(
                game_id=int.from_bytes(
                    sha256(puzzle_id.encode()).digest()[:8],
                    "big",
                ),
                metrics={
                    definition.identifier: MetricTotal(
                        total=sum(
                            float(getattr(score, attribute)) for score in ratings
                        ),
                        positions=len(ratings),
                    )
                    for definition, attribute in metric_values
                },
            )
        )
    entries = bootstrap_floors(
        totals,
        component=component,
        seed=config.seed,
        resamples=config.resamples,
        coverage=config.coverage,
    )
    return {
        entry.metric: NoiseFloor(
            value=entry.floor,
            kind="data-sampling",
            source=(
                f"{config.resamples} bootstrap resamples of "
                f"{len(totals)} puzzle source games"
            ),
        )
        for entry in entries
    }


def _write_detail(
    detail: DetailStore | None,
    result: PuzzleBenchmarkResult,
    *,
    recorded_at: datetime,
) -> DetailReference | None:
    if detail is None:
        return None
    stamp = recorded_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return detail.write(
        Path(PUZZLE_KIND) / f"{result.checkpoint.label}-{stamp}.json",
        result.as_record(),
        description=(
            "Puzzle-rating grid, human reference curve, rating-band response, "
            "and source-game overlap provenance."
        ),
    )


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
