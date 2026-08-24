"""Tests for the owned puzzle set and rating-response measurement."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from collections.abc import Callable, Sequence
from hashlib import sha256
from operator import attrgetter
from pathlib import Path
from typing import cast

import chess
import numpy as np
import pytest
import torch
import zstandard
from pydantic import ValidationError
from torch import Tensor

from anthro_chess.chess import ACTION_VOCABULARY_SIZE, encode_move, legal_action_ids
from anthro_chess.config import ConfigProvenance, ResolvedConfig, load_config
from anthro_chess.data import DecisionContext, DecisionHistory
from anthro_chess.evaluation.benchmarks import benchmark_registry, run_benchmark
from anthro_chess.evaluation.cost import BENCHMARK_COST_KIND
from anthro_chess.evaluation.noise import NoiseConfig
from anthro_chess.evaluation.puzzles import (
    Puzzle,
    PuzzleSet,
    PuzzleSetBuildConfig,
    PuzzleSetError,
    conservative_detectable_difference,
    dataset,
    expected_score,
    fitted_rating,
    load_puzzle_set,
    prepare_puzzle_set,
    puzzle_set_identity,
)
from anthro_chess.evaluation.puzzles.benchmark import (
    PUZZLE_KIND,
    PuzzleBenchmarkConfig,
    PuzzleBenchmarkError,
    PuzzleBenchmarkResult,
    _accepted_actions,
    _decision_tasks,
    _draw_multiplicity,
    _fit_curve,
    _response_resolution,
    _score_rating,
    _ScoredRating,
    _stratum_buckets,
    score_puzzle_set,
)
from anthro_chess.evaluation.puzzles.dataset import (
    PUZZLE_FILE_NAME,
    PUZZLE_METADATA_FILE_NAME,
    PUZZLE_SELECTION_ALGORITHM,
    VENDORED_RECORD_FILE_NAME,
    build_vendored_puzzle_set,
    load_vendored_puzzle_set,
    select_puzzles,
)
from anthro_chess.evaluation.results import (
    DetailStore,
    ResultsStore,
)
from anthro_chess.evaluation.results.noise import dispersion_bound

REPOSITORY_ROOT = Path(__file__).parents[2]
PUZZLE_SET_CONFIG = REPOSITORY_ROOT / "configs/evaluation/lichess-puzzles-v1.toml"


def _measure(
    resolved_config: ResolvedConfig[PuzzleBenchmarkConfig],
    *,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> PuzzleBenchmarkResult:
    """Measure the benchmark the way both callers do, through the driver."""

    return cast(
        PuzzleBenchmarkResult,
        run_benchmark(
            benchmark_registry()["puzzles"],
            resolved_config,
            store=store,
            detail=detail,
        ),
    )


def _context_key(context: DecisionContext) -> tuple[object, ...]:
    board = context.plies[-1].board
    return (
        board.piece_ids,
        board.side_to_move,
        board.castling_rights,
        board.en_passant_square,
        board.halfmove_clock,
        board.fullmove_number,
        board.repetition_count,
    )


class _ControlledRunner:
    """Prefer wrong moves at low ratings and puzzle moves at high ratings."""

    def __init__(self, puzzles: Sequence[Puzzle], *, fail_continuations: bool) -> None:
        self._decisions: dict[tuple[object, ...], tuple[int, int, int]] = {}
        for puzzle in puzzles:
            history = DecisionHistory(initial_fen=puzzle.initial_fen)
            solution_index = 0
            for ply, move in enumerate(puzzle.moves):
                history.push(move)
                if ply % 2:
                    continue
                target = encode_move(puzzle.moves[ply + 1])
                legal = legal_action_ids(history.board)
                alternative = next(action for action in legal if action != target)
                self._decisions[_context_key(history.context(target_rating=0))] = (
                    target,
                    alternative,
                    solution_index,
                )
                solution_index += 1
        self._fail_continuations = fail_continuations
        self.contexts: list[DecisionContext] = []

    def predict_batch(
        self,
        contexts: Sequence[DecisionContext],
    ) -> tuple[Tensor, ...]:
        predictions: list[Tensor] = []
        for context in contexts:
            self.contexts.append(context)
            target, alternative, solution_index = self._decisions[_context_key(context)]
            high = int(context.target_rating or 0) >= 1600
            preferred = (
                alternative
                if not high or (self._fail_continuations and solution_index > 0)
                else target
            )
            logits = torch.full((ACTION_VOCABULARY_SIZE,), -4.0)
            logits[preferred] = 4.0
            predictions.append(logits)
        return tuple(predictions)


def _fixture_set() -> PuzzleSet:
    puzzles = tuple(
        sorted(
            (
                _puzzle(
                    "0db6n",
                    "N1bk2nr/1p1p1ppp/p2Qp3/8/4P3/6P1/1Pn1KP1P/2qN1B1R b - - 1 14",
                    "c2a1 d6f8",
                    1379,
                    "zotX7Zc3",
                ),
                _puzzle(
                    "01k4m",
                    "r1bqr1k1/1p2bppp/p4n2/3p2B1/8/2PB1N1P/PP2Q1P1/RN2R1K1 w - - 4 15",
                    "b1d2 e7c5 g1h1 e8e2",
                    1350,
                    "NGD3XQIZ",
                ),
            ),
            key=lambda puzzle: puzzle.puzzle_id,
        )
    )
    return PuzzleSet(
        name="fixture-puzzles",
        version=1,
        content_sha256="0" * 64,
        source={},
        license={},
        selection={
            "minimum_rating": 800,
            "maximum_rating_exclusive": 2800,
            "local_precision_span": 400,
        },
        sizing={},
        coverage={},
        puzzles=puzzles,
    )


def _puzzle(
    puzzle_id: str,
    initial_fen: str,
    moves: str,
    rating: int,
    source_game_key: str,
) -> Puzzle:
    return Puzzle(
        puzzle_id=puzzle_id,
        initial_fen=initial_fen,
        moves=tuple(chess.Move.from_uci(move) for move in moves.split()),
        rating=rating,
        source_game_key=source_game_key,
    )


def _write_fixture_artifact(path: Path) -> Path:
    puzzle_set = _fixture_set()
    rows = [
        "puzzle_id,initial_fen,moves,rating,source_game_key",
        *[
            ",".join(
                (
                    puzzle.puzzle_id,
                    puzzle.initial_fen,
                    " ".join(move.uci() for move in puzzle.moves),
                    str(puzzle.rating),
                    puzzle.source_game_key,
                )
            )
            for puzzle in puzzle_set.puzzles
        ],
    ]
    content = "\n".join(rows) + "\n"
    path.mkdir()
    (path / PUZZLE_FILE_NAME).write_text(content)
    (path / PUZZLE_METADATA_FILE_NAME).write_text(
        json.dumps(
            {
                "name": puzzle_set.name,
                "version": puzzle_set.version,
                "entries": len(puzzle_set.puzzles),
                "puzzles_sha256": sha256(content.encode()).hexdigest(),
                "source": {"url": "https://example.test/puzzles"},
                "license": {"spdx_id": "CC0-1.0"},
                "selection": puzzle_set.selection,
                "sizing": {},
                "coverage": {},
            }
        )
    )
    return path


def test_external_set_matches_its_identity_and_validates_lines(
    tmp_path: Path,
) -> None:
    path = _write_fixture_artifact(tmp_path / "puzzles")
    puzzle_set = load_puzzle_set(path)
    content = (path / PUZZLE_FILE_NAME).read_text()

    assert puzzle_set_identity(path) == {
        "name": puzzle_set.name,
        "version": puzzle_set.version,
        "entries": 2,
        "sha256": sha256(content.encode()).hexdigest(),
    }
    assert puzzle_set.license["spdx_id"] == "CC0-1.0"
    assert len(puzzle_set.puzzles) == 2


def test_statistical_size_resolves_small_overall_and_local_differences() -> None:
    assert conservative_detectable_difference(20_000) == pytest.approx(0.0140, abs=1e-4)
    assert conservative_detectable_difference(4_000) == pytest.approx(0.0313, abs=1e-4)


def test_a_subsample_is_the_set_a_smaller_build_would_have_written(
    tmp_path: Path,
    write_puzzle_artifact: Callable[..., Path],
) -> None:
    """The dial reads the same design at a smaller size, not a slice of it.

    A flat count over the whole set would leave some exact ratings unscored
    and overweight others, which is the source-population bias the build
    removed. Ranking within each rating by the hash the build ranks by keeps
    the design, and keeps smaller readings nested inside larger ones.
    """

    ratings = (1200, 1300, 1400)
    puzzle_set = load_puzzle_set(
        write_puzzle_artifact(tmp_path / "set", ratings=ratings, puzzles_per_rating=6)
    )

    full = select_puzzles(puzzle_set, None)
    quad = select_puzzles(puzzle_set, 4)
    pair = select_puzzles(puzzle_set, 2)

    assert full.puzzles == puzzle_set.puzzles
    assert full.puzzles_per_rating is None
    assert Counter(puzzle.rating for puzzle in pair.puzzles) == dict.fromkeys(
        ratings, 2
    )
    assert set(pair.puzzles) < set(quad.puzzles) < set(full.puzzles)
    for rating in ratings:
        ranked = sorted(
            (puzzle for puzzle in puzzle_set.puzzles if puzzle.rating == rating),
            key=lambda puzzle: sha256(puzzle.puzzle_id.encode()).digest(),
        )
        kept = [puzzle for puzzle in pair.puzzles if puzzle.rating == rating]
        assert set(kept) == set(ranked[:2])

    assert pair.as_record() == {
        # The build's own algorithm identity, because this is that selection.
        "algorithm": PUZZLE_SELECTION_ALGORITHM,
        "puzzles_per_rating": 2,
        "selected_puzzles": 6,
        "eligible_puzzles": 18,
    }
    assert pair.minimum_detectable_difference == conservative_detectable_difference(6)
    assert pair.subsampled and not full.subsampled


def test_a_dial_that_leaves_one_puzzle_per_rating_is_refused() -> None:
    """A stratum of one leaves the redraw nothing to take.

    The response redraw stratifies by exact puzzle rating and draws one fewer
    than each stratum holds, so a setting that leaves one puzzle at each rating
    would produce a reading with no response resolution beside it at all.
    """

    with pytest.raises(ValidationError):
        PuzzleBenchmarkConfig.model_validate(
            {
                "puzzle_set": "puzzles",
                "puzzles_per_rating": 1,
            }
        )


def test_vendoring_selects_uniform_exact_ratings_and_records_coverage(
    tmp_path: Path,
) -> None:
    source_rows = [
        (
            "candidate-a",
            "N1bk2nr/1p1p1ppp/p2Qp3/8/4P3/6P1/1Pn1KP1P/2qN1B1R b - - 1 14",
            "c2a1 d6f8",
            1000,
            80,
            90,
            500,
            "https://lichess.org/game0001#1",
        ),
        (
            "candidate-b",
            "N1bk2nr/1p1p1ppp/p2Qp3/8/4P3/6P1/1Pn1KP1P/2qN1B1R b - - 1 14",
            "c2a1 d6f8",
            1000,
            80,
            90,
            500,
            "https://lichess.org/game0002#1",
        ),
        (
            "candidate-c",
            "r1bqr1k1/1p2bppp/p4n2/3p2B1/8/2PB1N1P/PP2Q1P1/RN2R1K1 w - - 4 15",
            "b1d2 e7c5 g1h1 e8e2",
            1001,
            75,
            91,
            700,
            "https://lichess.org/game0003#1",
        ),
        (
            "filtered-rd",
            "r1bqr1k1/1p2bppp/p4n2/3p2B1/8/2PB1N1P/PP2Q1P1/RN2R1K1 w - - 4 15",
            "b1d2 e7c5 g1h1 e8e2",
            1001,
            101,
            91,
            700,
            "https://lichess.org/game0004#1",
        ),
    ]
    source_text = _source_csv(source_rows)
    compressed = zstandard.ZstdCompressor().compress(source_text.encode())
    source = tmp_path / "source.csv.zst"
    source.write_bytes(compressed)

    selected = [
        min(
            source_rows[:2],
            key=lambda row: sha256(str(row[0]).encode()).digest(),
        ),
        source_rows[2],
    ]
    output_text = _selected_csv(sorted(selected, key=lambda row: str(row[0])))
    config = PuzzleSetBuildConfig.model_validate(
        {
            "artifact_name": "fixture-puzzles",
            "name": "fixture-puzzles",
            "version": 1,
            "source_retrieved": "2026-07-29",
            "source_last_modified": "2026-07-28T00:00:00Z",
            "expected_entries": 2,
            "expected_puzzles_sha256": sha256(output_text.encode()).hexdigest(),
            "archive": {
                "url": "https://example.test/puzzles.csv.zst",
                "file_name": "source.csv.zst",
                "sha256": sha256(compressed).hexdigest(),
            },
            "selection": {
                "minimum_rating": 1000,
                "maximum_rating_exclusive": 1002,
                "puzzles_per_rating": 1,
                "minimum_plays": 100,
                "minimum_popularity": 0,
                "maximum_rating_deviation": 100,
                "local_precision_span": 2,
            },
        }
    )
    vendored = build_vendored_puzzle_set(config, source)

    assert vendored.entries == 2
    assert vendored.content == output_text
    assert vendored.puzzles_sha256 == config.expected_puzzles_sha256
    assert vendored.coverage["eligible_candidates"] == 3
    assert vendored.coverage["minimum_candidates_per_rating"] == 1
    assert vendored.source["last_modified"] == "2026-07-28T00:00:00+00:00"


def test_the_committed_selection_builds_the_artifact_its_configuration_pins(
    tmp_path: Path,
) -> None:
    """Rebuild the canonical artifact with no archive and no network.

    Upstream rotates its puzzle export every few days and keeps no history,
    so this is the only check that the pinned identity is still reachable on
    a machine that has never downloaded it.
    """

    resolved = load_config(PuzzleSetBuildConfig, path=PUZZLE_SET_CONFIG)
    result = prepare_puzzle_set(resolved, tmp_path / "artifact")
    loaded = load_puzzle_set(result.artifact_path)
    vendored = load_vendored_puzzle_set()

    assert result.entries == resolved.value.expected_entries
    assert result.puzzles_sha256 == resolved.value.expected_puzzles_sha256
    assert loaded.identity() == {
        "name": resolved.value.name,
        "version": resolved.value.version,
        "entries": resolved.value.expected_entries,
        "sha256": resolved.value.expected_puzzles_sha256,
    }
    assert loaded.coverage == vendored.coverage
    assert loaded.selection["algorithm"] == PUZZLE_SELECTION_ALGORITHM
    assert loaded.sizing["overall_puzzles"] == resolved.value.expected_entries
    assert loaded.sizing["local_rating_span"] == (
        resolved.value.selection.local_precision_span
    )


def test_a_repin_that_skips_revendoring_is_refused(tmp_path: Path) -> None:
    repinned = load_config(
        PuzzleSetBuildConfig,
        path=PUZZLE_SET_CONFIG,
        overrides=[f'archive.sha256="{"0" * 64}"'],
    )

    with pytest.raises(PuzzleSetError, match="disagrees with this configuration"):
        prepare_puzzle_set(repinned, tmp_path / "artifact")


@pytest.mark.parametrize(
    ("file_name", "content", "expected"),
    [
        (PUZZLE_FILE_NAME, "edited\n", "does not match the checksum"),
        (VENDORED_RECORD_FILE_NAME, "{", "is unusable"),
        (VENDORED_RECORD_FILE_NAME, '{"name": "x"}', "is unusable"),
    ],
)
def test_a_committed_pair_that_disagrees_with_itself_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
    content: str,
    expected: str,
) -> None:
    original = dataset._read_data_file
    monkeypatch.setattr(
        dataset,
        "_read_data_file",
        lambda name: content if name == file_name else original(name),
    )

    with pytest.raises(PuzzleSetError, match=expected):
        load_vendored_puzzle_set()


def _source_csv(rows: Sequence[tuple[object, ...]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "PuzzleId",
            "FEN",
            "Moves",
            "Rating",
            "RatingDeviation",
            "Popularity",
            "NbPlays",
            "Themes",
            "GameUrl",
            "OpeningTags",
        )
    )
    for row in rows:
        writer.writerow((*row[:7], "fixture", row[7], ""))
    return output.getvalue()


def _selected_csv(rows: Sequence[tuple[object, ...]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("puzzle_id", "initial_fen", "moves", "rating", "source_game_key"))
    for row in rows:
        game_key = str(row[7]).split("/")[3].split("#")[0][:8]
        writer.writerow((row[0], row[1], row[2], row[3], game_key))
    return output.getvalue()


def test_reference_curve_and_fit_share_the_expected_score_definition() -> None:
    ratings = [1000, 1400, 1800, 2200]
    outcomes = [expected_score(1750, rating) for rating in ratings]

    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1800, 1500) > expected_score(1200, 1500)
    assert fitted_rating(ratings, outcomes) == pytest.approx(1750)


def test_every_checkmate_is_accepted_for_a_mate_in_one() -> None:
    puzzle = _puzzle(
        "90Yss",
        "4r3/6pk/1b1P4/5p1p/7P/5p2/2QRK1B1/4R1q1 w - - 0 37",
        "e2d1 g1e1",
        996,
        "DBhDAhGJ",
    )
    history = DecisionHistory(initial_fen=puzzle.initial_fen)
    history.push(puzzle.moves[0])

    accepted = _accepted_actions(history.board, puzzle.moves[1])

    assert accepted == tuple(
        sorted(
            (
                encode_move(chess.Move.from_uci("e8e1")),
                encode_move(chess.Move.from_uci("g1e1")),
            )
        )
    )


def test_first_move_and_full_line_stay_separate_and_output_is_deterministic() -> None:
    puzzle_set = _fixture_set()
    runner = _ControlledRunner(puzzle_set.puzzles, fail_continuations=True)

    first = score_puzzle_set(
        puzzle_set,
        runner,
        target_ratings=(1000, 2000),
        temperature=0.0,
        batch_size=1,
    )
    second = score_puzzle_set(
        puzzle_set,
        runner,
        target_ratings=(1000, 2000),
        temperature=0.0,
        batch_size=2,
    )

    assert first == second
    low, high = first
    assert low.greedy_first_move_accuracy == 0.0
    assert low.greedy_line_completion == 0.0
    assert high.greedy_first_move_accuracy == 1.0
    assert high.greedy_line_completion == 0.5
    assert high.sampled_first_move_solve_rate == high.greedy_first_move_accuracy
    assert high.sampled_line_completion == high.greedy_line_completion
    assert high.greedy_fitted_puzzle_rating > low.greedy_fitted_puzzle_rating
    assert sum(band.puzzles for band in high.bands) == 2


def test_one_replay_serves_every_configured_rating() -> None:
    # Object identity rather than equality, because that is the whole claim:
    # a rebuilt derivation produces contexts that compare equal to these and
    # cost the replay, the legal actions and the accepted actions again.
    puzzle_set = _fixture_set()
    runner = _ControlledRunner(puzzle_set.puzzles, fail_continuations=True)
    ratings = (1000, 1400, 2000)

    score_puzzle_set(
        puzzle_set,
        runner,
        target_ratings=ratings,
        temperature=0.0,
        batch_size=1,
    )

    passes = [
        [context for context in runner.contexts if context.target_rating == rating]
        for rating in ratings
    ]
    decisions = sum(len(puzzle.solution_moves) for puzzle in puzzle_set.puzzles)
    assert [len(contexts) for contexts in passes] == [decisions] * len(ratings)
    for shared in zip(*passes, strict=True):
        first = shared[0]
        assert all(context.plies is first.plies for context in shared)
        assert all(context.columns is first.columns for context in shared)


def _scored_over_strata(
    tmp_path: Path,
    write_puzzle_artifact: Callable[..., Path],
    target_ratings: Sequence[int],
    puzzles_per_rating: int = 4,
) -> tuple[PuzzleSet, tuple[_ScoredRating, ...]]:
    """Score a set holding ``puzzles_per_rating`` puzzles at each exact rating."""

    puzzle_set = load_puzzle_set(
        write_puzzle_artifact(
            tmp_path / "puzzles",
            ratings=(1200, 1400),
            puzzles_per_rating=puzzles_per_rating,
        )
    )
    runner = _ControlledRunner(puzzle_set.puzzles, fail_continuations=True)
    tasks = _decision_tasks(puzzle_set.puzzles)
    return puzzle_set, tuple(
        _score_rating(
            puzzle_set,
            puzzle_set.puzzles,
            runner,
            tasks,
            target_rating=rating,
            temperature=0.0,
            batch_size=4,
        )
        for rating in target_ratings
    )


def test_the_response_resolution_refits_puzzles_redrawn_within_rating_strata(
    tmp_path: Path,
    write_puzzle_artifact: Callable[..., Path],
) -> None:
    """A rating response is a comparison within one reading over one draw.

    Every configured rating is scored on the same puzzles, so one redraw has to
    move all of them together for the spread to describe the response rather
    than several unrelated readings.
    """

    target_ratings = (1000, 1800)
    _, scored = _scored_over_strata(tmp_path, write_puzzle_artifact, target_ratings)

    resolution = _response_resolution(scored, NoiseConfig())

    assert resolution is not None
    assert (resolution.puzzles, resolution.resamples) == (8, 1000)
    low, high = resolution.ratings
    # The controlled runner answers every low-rating decision wrongly, so no
    # redraw of it can move that fit, and a spread of zero there would claim
    # the opposite of what was observed.
    assert low.greedy_fitted_puzzle_rating is None
    assert high.greedy_fitted_puzzle_rating is not None
    assert high.greedy_fitted_puzzle_rating.bound > 0.0
    assert resolution.greedy_rating_slope is not None
    assert resolution.greedy_rating_slope.bound > 0.0


def test_a_reading_that_estimates_no_noise_reports_no_response_resolution(
    tmp_path: Path,
    write_puzzle_artifact: Callable[..., Path],
) -> None:
    target_ratings = (1000, 1800)
    _, scored = _scored_over_strata(tmp_path, write_puzzle_artifact, target_ratings)

    assert _response_resolution(scored, NoiseConfig(enabled=False)) is None


def test_the_tabulated_fit_inverse_reproduces_the_bisected_one(
    tmp_path: Path,
    write_puzzle_artifact: Callable[..., Path],
) -> None:
    """The resolution refits by inverting a curve rather than by bisecting.

    That is sound only while the two agree, since the spread it reports has to
    belong to the quantity printed beside it.
    """

    target_ratings = (1000, 1800)
    puzzle_set, scored = _scored_over_strata(
        tmp_path,
        write_puzzle_artifact,
        target_ratings,
    )
    puzzle_ratings = [puzzle.rating for puzzle in puzzle_set.puzzles]
    fitted, totals = _fit_curve(puzzle_ratings)

    for scored_rating in scored:
        outcomes = [score.greedy_line for score in scored_rating.scores]
        assert float(np.interp(sum(outcomes), totals, fitted)) == pytest.approx(
            scored_rating.result.greedy_fitted_puzzle_rating,
            abs=1e-2,
        )


def test_the_benchmark_records_every_envelope_and_payload_it_produced(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    # The only end-to-end reading of this benchmark. Its result carried a
    # single envelope and a relative detail path where every other benchmark
    # carried tuples of absolute ones, and nothing here noticed for as long as
    # the drift existed.
    config = _benchmark_config(tmp_path, inference_run)
    store = ResultsStore(tmp_path / "results")
    detail = DetailStore(tmp_path / "detail")

    result = _measure(config, store=store, detail=detail)

    # The reading, and what the invocation cost recorded beside it. Sorted
    # because the store reads its records back in its own order.
    key = attrgetter("result_id")
    assert sorted(result.envelopes, key=key) == sorted(store.results(), key=key)
    assert {item.kind for item in result.envelopes} == {
        PUZZLE_KIND,
        BENCHMARK_COST_KIND,
    }
    assert len(result.recorded_paths) == len(result.envelopes) == 2
    (written,) = result.detail_paths
    assert written.is_absolute()
    assert written.parent == detail.root / PUZZLE_KIND / result.checkpoint.label
    assert result.as_record()["recorded"] == [
        str(path) for path in result.recorded_paths
    ]


def test_the_benchmark_measures_without_recording_anything(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    config = _benchmark_config(tmp_path, inference_run)

    result = _measure(config)

    assert len(result.envelopes) == 2
    assert result.recorded_paths == ()
    assert result.detail_paths == ()


def test_every_solve_rate_stores_a_floor_the_comparison_can_cover(
    tmp_path: Path,
    write_puzzle_artifact: Callable[..., Path],
    inference_run: Callable[..., Path],
) -> None:
    """A stored bound carries no coverage, because a delta floor applies it.

    Storing the printed spread instead would cover the reading once here and
    again in ``combined_floor``, and the two factors are invisible in the
    output: the floor is simply wider, and real movement reads as noise.
    """

    artifact = write_puzzle_artifact(
        tmp_path / "puzzles",
        ratings=(1200, 1400),
        puzzles_per_rating=4,
    )
    result = _measure(
        _benchmark_config(
            tmp_path,
            inference_run,
            puzzle_set=artifact,
        )
    )
    resolution = result.resolution
    assert resolution is not None
    stored = {
        measurement.metric: measurement
        for envelope in result.envelopes
        for measurement in envelope.measurements
    }

    rates = {
        "puzzle.greedy_first_move_accuracy": resolution.greedy_first_move_accuracy,
        "puzzle.greedy_line_completion": resolution.greedy_line_completion,
        "puzzle.sampled_first_move_solve_rate": (
            resolution.sampled_first_move_solve_rate
        ),
        "puzzle.sampled_line_completion": resolution.sampled_line_completion,
    }
    # A fixture runner answers every puzzle identically, so a rate no redraw
    # moves stores no dispersion at all; the ones that move carry the relation.
    moved = {metric: spread for metric, spread in rates.items() if spread is not None}
    assert moved
    for metric, spread in moved.items():
        dispersion = stored[metric].dispersion
        assert dispersion is not None, metric
        assert dispersion.value == pytest.approx(spread.dispersion)
        assert dispersion.units == resolution.puzzles
        assert dispersion.bound == pytest.approx(
            dispersion_bound(
                spread.dispersion,
                degrees_of_freedom=resolution.puzzles - 1,
                confidence=resolution.confidence,
            )
        )
        # The printed spread is the covered one, and the stored bound is not.
        assert spread.bound > dispersion.bound
    for metric in rates.keys() - moved.keys():
        assert stored[metric].dispersion is None, metric


def test_a_configured_subsample_measures_and_records_only_what_it_selected(
    tmp_path: Path,
    write_puzzle_artifact: Callable[..., Path],
    inference_run: Callable[..., Path],
) -> None:
    """A smaller reading is its own series, not a partial full one.

    The puzzles scored are the data component, so a subsample has to carry a
    different component and view from the full reading; sharing either would
    let a smaller reading continue a larger one's line.
    """

    artifact = write_puzzle_artifact(
        tmp_path / "puzzles",
        ratings=(1200, 1400),
        puzzles_per_rating=4,
    )
    arguments = (tmp_path, inference_run)

    full = _measure(_benchmark_config(*arguments, puzzle_set=artifact))
    reduced = _measure(
        _benchmark_config(*arguments, puzzle_set=artifact, puzzles_per_rating=2)
    )
    # A dial that drops nothing selected the canonical set, whatever it asked
    # for, and has to land on the series that reading already has.
    undialled = _measure(
        _benchmark_config(*arguments, puzzle_set=artifact, puzzles_per_rating=4)
    )

    assert full.selection.selected_puzzles == 8
    assert reduced.selection.selected_puzzles == 4
    assert reduced.selection.eligible_puzzles == 8
    assert reduced.dataset.selected_games == 4
    assert reduced.dataset.view == "per-rating-2"
    assert full.dataset.view == "canonical"
    assert undialled.dataset == full.dataset
    assert reduced.dataset.components != full.dataset.components
    assert reduced.dataset.game_ids_sha256 != full.dataset.game_ids_sha256
    assert reduced.as_record()["selection"] == reduced.selection.as_record()
    # The puzzle count every measurement reports is the realized one, not the
    # artifact's, or a smaller reading would claim a resolution it never had.
    (reading,) = [item for item in reduced.envelopes if item.kind == PUZZLE_KIND]
    sizes = {item.sample_size for item in reading.measurements}
    assert sizes == {len(reduced.ratings), 4 * len(reduced.ratings)}


def test_a_missing_puzzle_artifact_raises_the_error_the_suite_declares(
    tmp_path: Path,
    inference_run: Callable[..., Path],
) -> None:
    # A host without the pinned artifact is the ordinary partial failure the
    # sweep is built to survive. Raising anything the suite has not declared
    # for this step ends the whole sweep and discards the readings before it.
    config = _benchmark_config(
        tmp_path,
        inference_run,
        puzzle_set=tmp_path / "absent",
    )

    with pytest.raises(
        PuzzleBenchmarkError,
        match="puzzle artifact is missing",
    ) as raised:
        _measure(config)

    assert isinstance(raised.value, benchmark_registry()["puzzles"].errors)


def _benchmark_config(
    tmp_path: Path,
    inference_run: Callable[..., Path],
    *,
    puzzle_set: Path | None = None,
    puzzles_per_rating: int | None = None,
) -> ResolvedConfig[PuzzleBenchmarkConfig]:
    """Write a puzzle artifact and a checkpoint, and select them."""

    artifact = puzzle_set or _write_fixture_artifact(tmp_path / "puzzles")
    checkpoint = inference_run(tmp_path / "run")
    return ResolvedConfig(
        value=PuzzleBenchmarkConfig.model_validate(
            {
                "puzzle_set": str(artifact),
                "model": {"checkpoint_path": str(checkpoint), "device": "cpu"},
                "target_ratings": [1000, 1800],
                "inference_batch_size": 4,
                "puzzles_per_rating": puzzles_per_rating,
            }
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )


def test_a_rating_holding_one_puzzle_gets_no_response_resolution(
    tmp_path: Path,
    write_puzzle_artifact: Callable[..., Path],
) -> None:
    """A stratum of one leaves nothing for the rescaled draw to take."""

    _, scored = _scored_over_strata(
        tmp_path,
        write_puzzle_artifact,
        (1000, 1800),
        puzzles_per_rating=1,
    )

    assert _response_resolution(scored, NoiseConfig()) is None


def test_a_rescaled_stratified_redraw_removes_the_plug_in_understatement() -> None:
    """A plug-in draw of `n` reports `(n-1)/n` of the variance it should.

    Decision 0039 measures what that costs where a stratum is small, and names
    the correction: take one fewer and scale the counts back up. Two units to a
    stratum is where it is worst and is what the reduced puzzle sweep scores.
    """

    values = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    buckets = _stratum_buckets(["low", "low", "mid", "mid", "high", "high"])
    means = np.asarray(
        [
            _draw_multiplicity(
                np.random.default_rng(seed),
                units=len(values),
                buckets=buckets,
            )
            @ values
            / len(values)
            for seed in range(4000)
        ]
    )

    # Each stratum holds one zero and one one, so a stratified mean has
    # variance 1/12 exactly; a plug-in draw would report half of it.
    assert float(np.var(means)) == pytest.approx(1.0 / 12.0, rel=0.05)
