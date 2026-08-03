"""Tests for the owned puzzle set and rating-response measurement."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path

import chess
import pytest
import torch
import zstandard
from torch import Tensor

from anthro_chess.chess import ACTION_VOCABULARY_SIZE, encode_move, legal_action_ids
from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.data import DecisionContext, DecisionHistory
from anthro_chess.evaluation.noise import NoiseConfig
from anthro_chess.evaluation.puzzles import (
    Puzzle,
    PuzzleSet,
    PuzzleSetBuildConfig,
    conservative_detectable_difference,
    expected_score,
    fitted_rating,
    load_puzzle_set,
    prepare_puzzle_set,
    puzzle_set_identity,
)
from anthro_chess.evaluation.puzzles.benchmark import (
    PUZZLE_KIND,
    PuzzleBenchmarkConfig,
    _accepted_actions,
    _paired_contributions,
    _score_rating,
    _training_overlap,
    benchmark_puzzles,
    score_puzzle_set,
)
from anthro_chess.evaluation.puzzles.dataset import (
    PUZZLE_FILE_NAME,
    PUZZLE_METADATA_FILE_NAME,
)
from anthro_chess.evaluation.results import (
    DetailStore,
    PairedContributions,
    ResultsStore,
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
        context.plies[-1].previous_action_id,
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

    def predict_batch(
        self,
        contexts: Sequence[DecisionContext],
    ) -> tuple[Tensor, ...]:
        predictions: list[Tensor] = []
        for context in contexts:
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


def test_preparation_selects_uniform_exact_ratings_and_records_coverage(
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
    result = prepare_puzzle_set(
        ResolvedConfig(
            value=config,
            provenance=ConfigProvenance(source=None, overrides=()),
        ),
        tmp_path / "artifact",
        source_path=source,
    )
    loaded = load_puzzle_set(result.artifact_path)

    assert result.entries == 2
    assert [puzzle.rating for puzzle in loaded.puzzles] == [1000, 1001]
    assert loaded.coverage["eligible_candidates"] == 3
    assert loaded.coverage["minimum_candidates_per_rating"] == 1
    assert loaded.sizing["overall_puzzles"] == 2


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


def test_puzzle_details_retain_source_game_aligned_checkpoint_contributions() -> None:
    puzzle_set = _fixture_set()
    runner = _ControlledRunner(puzzle_set.puzzles, fail_continuations=True)
    scored = tuple(
        _score_rating(
            puzzle_set,
            runner,
            target_rating=rating,
            temperature=0.0,
            batch_size=2,
        )
        for rating in (1000, 2000)
    )

    raw = _paired_contributions(scored, NoiseConfig())

    assert raw is not None
    retained = PairedContributions.model_validate(raw)
    assert retained.unit == "puzzle-source-game"
    assert retained.stratum == "puzzle-rating"
    assert retained.strata == tuple(str(puzzle.rating) for puzzle in puzzle_set.puzzles)
    assert retained.unit_ids == tuple(
        puzzle.source_game_key for puzzle in puzzle_set.puzzles
    )
    assert retained.metrics["puzzle.greedy_line_completion"] == (0.0, 0.5)
    assert sum(retained.metrics["puzzle.greedy_first_move_accuracy"]) / 2 == (
        pytest.approx(
            sum(item.result.greedy_first_move_accuracy for item in scored) / 2
        )
    )


def test_training_overlap_joins_source_keys_and_excludes_test_games(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, object]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    puzzle_set = _fixture_set()
    matching = puzzle_set.puzzles[0].source_game_key
    test_only = puzzle_set.puzzles[1].source_game_key
    rows = [
        {
            **normalized_row(1, split="train"),
            "source_id": "lichess",
            "source_game_key": matching,
        },
        {
            **normalized_row(2, split="validation"),
            "source_id": "lichess",
            "source_game_key": "not-a-puzzle",
        },
        {
            **normalized_row(3, split="test"),
            "source_id": "lichess",
            "source_game_key": test_only,
        },
        {
            **normalized_row(4, split="train"),
            "source_id": "other",
            "source_game_key": test_only,
        },
    ]
    normalized, _ = write_corpus(tmp_path, rows)

    training_games, overlapping = _training_overlap(puzzle_set, normalized)

    assert training_games == 2
    assert overlapping == 1


def test_the_benchmark_records_every_envelope_and_payload_it_produced(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, object]],
    write_corpus: Callable[..., tuple[Path, Path]],
    inference_run: Callable[..., Path],
) -> None:
    # The only end-to-end reading of this benchmark. Its result carried a
    # single envelope and a relative detail path where every other benchmark
    # carried tuples of absolute ones, and nothing here noticed for as long as
    # the drift existed.
    artifact = _write_fixture_artifact(tmp_path / "puzzles")
    rows = [
        {
            **normalized_row(1, split="train"),
            "source_id": "lichess",
            "source_game_key": _fixture_set().puzzles[0].source_game_key,
        }
    ]
    normalized, _ = write_corpus(tmp_path / "corpus", rows)
    checkpoint = inference_run(tmp_path / "run")
    store = ResultsStore(tmp_path / "results")
    detail = DetailStore(tmp_path / "detail")

    result = benchmark_puzzles(
        _benchmark_config(artifact, normalized, checkpoint),
        store=store,
        detail=detail,
    )

    assert result.envelopes == store.results()
    assert len(result.recorded_paths) == len(result.envelopes) == 1
    (written,) = result.detail_paths
    assert written.is_absolute()
    assert written.parent == detail.root / PUZZLE_KIND / result.checkpoint.label
    assert result.as_record()["recorded"] == [str(result.recorded_paths[0])]


def test_the_benchmark_measures_without_recording_anything(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, object]],
    write_corpus: Callable[..., tuple[Path, Path]],
    inference_run: Callable[..., Path],
) -> None:
    artifact = _write_fixture_artifact(tmp_path / "puzzles")
    rows = [
        {
            **normalized_row(1, split="train"),
            "source_id": "lichess",
            "source_game_key": "not-a-puzzle",
        }
    ]
    normalized, _ = write_corpus(tmp_path / "corpus", rows)
    checkpoint = inference_run(tmp_path / "run")

    result = benchmark_puzzles(_benchmark_config(artifact, normalized, checkpoint))

    assert len(result.envelopes) == 1
    assert result.recorded_paths == ()
    assert result.detail_paths == ()
    assert result.overlapping_puzzles == 0


def _benchmark_config(
    puzzle_set: Path,
    training_normalized: Path,
    checkpoint: Path,
) -> ResolvedConfig[PuzzleBenchmarkConfig]:
    return ResolvedConfig(
        value=PuzzleBenchmarkConfig.model_validate(
            {
                "puzzle_set": str(puzzle_set),
                "training_normalized": str(training_normalized),
                "model": {"checkpoint_path": str(checkpoint), "device": "cpu"},
                "target_ratings": [1000, 1800],
                "inference_batch_size": 4,
            }
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )
