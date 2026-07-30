"""Tests for the owned puzzle set and rating-response measurement."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from hashlib import sha256
from importlib.resources import files
from pathlib import Path

import chess
import pytest
import torch
from torch import Tensor

from anthro_chess.chess import ACTION_VOCABULARY_SIZE, encode_move, legal_action_ids
from anthro_chess.data import DecisionContext, DecisionHistory
from anthro_chess.evaluation.puzzles import (
    Puzzle,
    PuzzleSet,
    expected_score,
    fitted_rating,
    load_puzzle_set,
    puzzle_set_identity,
)
from anthro_chess.evaluation.puzzles.benchmark import (
    _accepted_actions,
    _training_overlap,
    score_puzzle_set,
)
from anthro_chess.evaluation.puzzles.dataset import PUZZLE_FILE_NAME


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
    packaged = load_puzzle_set()
    single = next(
        puzzle for puzzle in packaged.puzzles if len(puzzle.solution_moves) == 1
    )
    multi = next(
        puzzle for puzzle in packaged.puzzles if len(puzzle.solution_moves) > 1
    )
    puzzles = tuple(sorted((single, multi), key=lambda puzzle: puzzle.puzzle_id))
    return PuzzleSet(
        name="fixture-puzzles",
        version=1,
        content_sha256="0" * 64,
        source={},
        license={},
        selection={
            "minimum_rating": 800,
            "maximum_rating_exclusive": 2800,
            "band_width": 400,
        },
        puzzles=puzzles,
    )


def test_packaged_set_matches_its_identity_license_and_balanced_selection() -> None:
    puzzle_set = load_puzzle_set()
    packaged = (
        files("anthro_chess.evaluation.puzzles")
        .joinpath("data", PUZZLE_FILE_NAME)
        .read_text(encoding="utf-8")
    )

    assert puzzle_set_identity() == {
        "name": puzzle_set.name,
        "version": puzzle_set.version,
        "entries": 320,
        "sha256": sha256(packaged.encode()).hexdigest(),
    }
    assert puzzle_set.license["spdx_id"] == "CC0-1.0"
    assert puzzle_set.source["url"] == (
        "https://database.lichess.org/lichess_db_puzzle.csv.zst"
    )
    minimum = int(puzzle_set.selection["minimum_rating"])
    width = int(puzzle_set.selection["band_width"])
    bands = Counter(
        ((puzzle.rating - minimum) // width) * width + minimum
        for puzzle in puzzle_set.puzzles
    )
    assert set(bands.values()) == {64}


def test_reference_curve_and_fit_share_the_expected_score_definition() -> None:
    ratings = [1000, 1400, 1800, 2200]
    outcomes = [expected_score(1750, rating) for rating in ratings]

    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1800, 1500) > expected_score(1200, 1500)
    assert fitted_rating(ratings, outcomes) == pytest.approx(1750)


def test_every_checkmate_is_accepted_for_a_mate_in_one() -> None:
    puzzle = next(
        puzzle for puzzle in load_puzzle_set().puzzles if puzzle.puzzle_id == "90Yss"
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
