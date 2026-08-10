"""Perturbation supplies novelty at a known dose; retention is what is read."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import chess
import pytest

from anthro_chess.chess import decode_move, is_terminal_action
from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.data.schema import row_game_id
from anthro_chess.evaluation import PoolConfig, freeze_pool
from anthro_chess.evaluation.benchmarks import benchmark_registry, run_benchmark
from anthro_chess.evaluation.cost import BENCHMARK_COST_KIND
from anthro_chess.evaluation.novelty import (
    NOVELTY_KIND,
    NOVELTY_RECIPE_VERSION,
    NoveltyBenchmarkConfig,
    NoveltyBenchmarkError,
    NoveltyBenchmarkResult,
    PerturbationConfig,
    PerturbationRecipe,
    derive_arm,
)
from anthro_chess.evaluation.results import DetailStore, ResultsStore
from anthro_chess.evaluation.slices import PositionPredicate
from anthro_chess.interfaces.cli import main

#: A long, quiet line. The window has to open and then run for several of the
#: opponent's moves, so the fixture games are longer than the shared opening.
LONG_LINE = (
    "e2e4",
    "e7e5",
    "g1f3",
    "b8c6",
    "f1b5",
    "a7a6",
    "b5a4",
    "g8f6",
    "e1g1",
    "f8e7",
    "f1e1",
    "b7b5",
    "a4b3",
    "d7d6",
    "c2c3",
    "e8g8",
    "h2h3",
    "c6a5",
    "b3c2",
    "c7c5",
    "d2d4",
    "d8c7",
    "b1d2",
    "c5d4",
)

#: Small enough that a fixture game reaches it, large enough that the window is
#: a genuine suffix rather than the whole game.
ONSET = 6
WINDOW = 4


def _measure(
    resolved_config: ResolvedConfig[NoveltyBenchmarkConfig],
    *,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> NoveltyBenchmarkResult:
    """Measure the benchmark the way both callers do, through the driver."""

    return cast(
        NoveltyBenchmarkResult,
        run_benchmark(
            benchmark_registry()["novelty"],
            resolved_config,
            store=store,
            detail=detail,
        ),
    )


def _rows(normalized_row: Callable[..., dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        normalized_row(
            game_id,
            split="test",
            rating=rating,
            moves=LONG_LINE[:length],
            result="*",
        )
        # The perturbation draws which colour the model plays from the game
        # id, so these indices are chosen to span both draws: a pool scoring
        # one side only reaches whichever predicates that side happens to meet.
        for game_id, rating, length in (
            (11, 1100, 24),
            (12, 1500, 22),
            (15, 2100, 20),
            (16, None, 18),
        )
    ]


def _corpus_rows(
    normalized_row: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the pool games plus the train split the leakage check reads."""

    return [
        normalized_row(1, split="train", plies=10, rating=1500),
        normalized_row(2, split="validation", plies=8, rating=1100),
        *_rows(normalized_row),
    ]


def _perturbation(**overrides: Any) -> PerturbationConfig:
    settings: dict[str, Any] = {
        "onset_plies": ONSET,
        "window_moves": WINDOW,
        "doses": [0.0, 1.0],
    }
    settings.update(overrides)
    return PerturbationConfig.model_validate(settings)


def test_the_control_arm_is_the_source_game_and_the_window_is_paired(
    normalized_row: Callable[..., dict[str, Any]],
    fixture_game_id: Callable[[int], int],
) -> None:
    rows = _rows(normalized_row)
    config = _perturbation()

    control = derive_arm(rows, dose=0.0, config=config)
    perturbed = derive_arm(rows, dose=1.0, config=config)

    assert {game.game_id for game in control} == {
        fixture_game_id(index) for index in (11, 12, 15, 16)
    }
    by_id = {game.game_id: game for game in perturbed}
    for game in control:
        source = next(row for row in rows if row_game_id(row) == game.game_id)
        # Dose zero replaces nothing, so the control arm's actions are the
        # human game itself over the window it measures.
        derived = game.row["action_ids"]
        assert derived == list(source["action_ids"])[: len(derived)]
        assert game.perturbed_opponent_moves == 0
        assert not game.truncated
        assert len(game.measured_plies) == WINDOW

        # Every arm measures the same color of the same game, or the retention
        # comparison would be between two different players.
        assert by_id[game.game_id].player_color == game.player_color
        # The perturbed arm's measured plies are a prefix of the control's:
        # they are the same positions until the human's reply stops being legal.
        other = by_id[game.game_id].measured_plies
        assert other == game.measured_plies[: len(other)]


def test_only_the_opponent_is_perturbed_and_divergence_is_absorbing(
    normalized_row: Callable[..., dict[str, Any]],
    fixture_game_id: Callable[[int], int],
) -> None:
    rows = _rows(normalized_row)
    games = derive_arm(rows, dose=1.0, config=_perturbation())

    assert games
    departures = 0
    for game in games:
        source = next(row for row in rows if row_game_id(row) == game.game_id)
        source_actions = list(source["action_ids"])
        board = chess.Board()
        opponent_plies_in_window = 0
        for ply_index, action_id in enumerate(game.row["action_ids"]):
            if is_terminal_action(action_id):
                break
            move = decode_move(action_id)
            # Every derived action is legal in the position it is played from,
            # so the arm is a reachable game rather than a spliced one.
            assert move in board.legal_moves
            if board.turn == game.player_color:
                # The player's side is always the human's own move. This is the
                # one-sidedness the benchmark depends on.
                assert action_id == source_actions[ply_index]
            elif ply_index >= ONSET:
                opponent_plies_in_window += 1
                # A uniform draw can land on the move the human played, so a
                # replaced move is not required to differ from it.
                departures += int(action_id != source_actions[ply_index])
            board.push(move)
        # At dose one, divergence starts at the window's first opponent move
        # and every later one is drawn too.
        assert opponent_plies_in_window == game.window_opponent_moves
        assert game.perturbed_opponent_moves == game.window_opponent_moves
        assert game.window_opponent_moves >= 1

    # Coinciding with the human on every draw across every game would mean
    # nothing was actually randomized.
    assert departures > 0


def test_derivation_is_deterministic_and_scoped_to_its_seed_and_recipe(
    normalized_row: Callable[..., dict[str, Any]],
) -> None:
    rows = _rows(normalized_row)

    first = derive_arm(rows, dose=0.5, config=_perturbation())
    again = derive_arm(rows, dose=0.5, config=_perturbation())
    reordered = derive_arm(list(reversed(rows)), dose=0.5, config=_perturbation())
    reseeded = derive_arm(rows, dose=0.5, config=_perturbation(seed="other-seed"))

    assert [game.row["action_ids"] for game in first] == [
        game.row["action_ids"] for game in again
    ]
    # Row order is not an input to the derivation, because every draw is keyed
    # by the game rather than by where it appeared.
    assert [game.row["action_ids"] for game in reordered] == [
        game.row["action_ids"] for game in first
    ]
    assert [game.row["action_ids"] for game in reseeded] != [
        game.row["action_ids"] for game in first
    ]


def test_a_higher_dose_diverges_at_least_as_early(
    normalized_row: Callable[..., dict[str, Any]],
) -> None:
    rows = _rows(normalized_row)
    config = _perturbation(doses=[0.0, 0.25, 1.0])

    light = {
        game.game_id: game.perturbed_opponent_moves
        for game in derive_arm(rows, dose=0.25, config=config)
    }
    heavy = {
        game.game_id: game.perturbed_opponent_moves
        for game in derive_arm(rows, dose=1.0, config=config)
    }

    assert sum(heavy.values()) > sum(light.values())


def test_a_sweep_without_a_control_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="control arm"):
        PerturbationConfig.model_validate({"doses": [0.25, 1.0]})
    with pytest.raises(ValueError, match="beside the control"):
        PerturbationConfig.model_validate({"doses": [0.0]})
    with pytest.raises(ValueError, match="between zero and one"):
        PerturbationConfig.model_validate({"doses": [0.0, 1.5]})


def test_derive_arm_rejects_a_dose_outside_the_rate(
    normalized_row: Callable[..., dict[str, Any]],
) -> None:
    with pytest.raises(NoveltyBenchmarkError, match="between zero and one"):
        derive_arm(_rows(normalized_row), dose=2.0, config=_perturbation())


def test_the_benchmark_reports_retention_against_its_own_control(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    training_run: Callable[..., Path],
    fixture_game_id: Callable[[int], int],
) -> None:
    normalized, manifest = write_corpus(
        tmp_path / "corpus", _corpus_rows(normalized_row)
    )
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )
    store = ResultsStore(tmp_path / "results")
    detail = DetailStore(tmp_path / "detail")

    result = _measure(
        _config(pool, checkpoint, doses=[0.0, 0.5, 1.0]),
        store=store,
        detail=detail,
    )

    assert [arm.dose for arm in result.arms] == [0.0, 0.5, 1.0]
    control = result.control
    assert control.is_control
    assert control.realized_dose == 0.0
    assert result.arms[-1].realized_dose > 0.0

    assert {envelope.kind for envelope in store.results()} == {
        NOVELTY_KIND,
        BENCHMARK_COST_KIND,
    }
    recorded = [item for item in store.results() if item.kind == NOVELTY_KIND]
    # One result per dose cell. Showing the most recent would present one
    # arbitrary cell as the checkpoint's reading and hide the rest.
    assert len(recorded) == 3

    by_dose = {
        envelope.execution.workload["dose"]: envelope
        for envelope in recorded
        if envelope.execution is not None
    }
    assert set(by_dose) == {0.0, 0.5, 1.0}

    control_envelope = by_dose[0.0]
    perturbed_envelope = by_dose[1.0]
    control_metrics = {item.metric for item in control_envelope.measurements}
    perturbed_metrics = {item.metric for item in perturbed_envelope.measurements}

    # Legality survives out of distribution and is reported at every dose,
    # held fixed per phase.
    assert "novelty.legal_mass" in control_metrics
    assert "novelty.mask_penalty" in control_metrics
    assert any(metric.startswith("novelty.mask_penalty_") for metric in control_metrics)
    # Retention exists only where there is something to retain against, so the
    # control arm carries no retention of its own.
    assert "novelty.legal_mass_retention" not in control_metrics
    assert "novelty.legal_mass_retention" in perturbed_metrics
    assert "novelty.mask_penalty_ratio" in perturbed_metrics
    # Held-out prediction is undefined once the prefix diverges, so it is
    # absent from every arm of this benchmark.
    assert not any(metric.startswith("held_out.") for metric in perturbed_metrics)

    retention = perturbed_envelope.measurement("novelty.legal_mass_retention")
    control_mass = control_envelope.measurement("novelty.legal_mass")
    perturbed_mass = perturbed_envelope.measurement("novelty.legal_mass")
    assert retention is not None and control_mass is not None
    assert perturbed_mass is not None

    # Retention is paired on position rather than being a ratio of the two
    # arms' overall means. A perturbed arm ends where the human's reply stopped
    # being legal, so its positions are a subset of the control's, and the
    # unpaired ratio reports that composition difference as a novelty effect.
    perturbed_arm = result.arms[-1]
    keys = perturbed_arm.measured_keys
    assert keys < control.measured_keys
    assert retention.value == pytest.approx(
        perturbed_arm.paired_legality(keys).legal_mass
        / control.paired_legality(keys).legal_mass
    )
    assert retention.value != pytest.approx(perturbed_mass.value / control_mass.value)

    # Two doses are two series: a delta across them would compare different
    # measurements rather than the same one over time.
    assert control_mass.fingerprint != perturbed_mass.fingerprint

    for envelope in recorded:
        envelope.verify()
        assert envelope.execution is not None
        workload = envelope.execution.workload
        assert workload["recipe"] == PerturbationRecipe.RANDOM_LEGAL_OPPONENT.value
        assert workload["recipe_version"] == NOVELTY_RECIPE_VERSION
        assert workload["onset_plies"] == ONSET
        assert workload["window_moves"] == WINDOW


def test_predicates_and_phase_slices_reach_the_detail_tier(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    training_run: Callable[..., Path],
    fixture_game_id: Callable[[int], int],
) -> None:
    normalized, manifest = write_corpus(
        tmp_path / "corpus", _corpus_rows(normalized_row)
    )
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )
    detail = DetailStore(tmp_path / "detail")

    result = _measure(_config(pool, checkpoint), detail=detail)

    control = result.control
    # Material gain is the predicate common enough to appear on quiet play, and
    # it is what carries the capability question onto the perturbed arms.
    assert PositionPredicate.MATERIAL_GAIN in control.predicates
    reading = control.predicates[PositionPredicate.MATERIAL_GAIN]
    assert reading.opportunities >= 1
    assert 0.0 <= reading.selected_rate <= 1.0
    assert 0.0 <= reading.policy_mass <= 1.0
    if reading.mean_best_rank is not None:
        assert reading.mean_best_rank >= 1.0

    assert result.detail_paths
    payload = json.loads(Path(result.detail_paths[0]).read_text(encoding="utf-8"))
    assert payload["dose"] == 0.0
    assert payload["realized_dose"] == 0.0
    assert "phase" in payload["slices"]["dimensions"]
    assert payload["games"]
    assert set(payload["games"][0]) >= {
        "game_id",
        "player_color",
        "measured_plies",
        "perturbed_opponent_moves",
        "truncated",
    }
    # Per-position records are opt-in, exactly as they are for the checkpoint
    # runner, so a routine sweep does not grow by an order of magnitude.
    assert "positions" not in payload


def test_nothing_is_recorded_without_a_store(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = write_corpus(
        tmp_path / "corpus", _corpus_rows(normalized_row)
    )
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )

    result = _measure(_config(pool, checkpoint))

    # A shakedown reading is evidence about the instrument rather than about
    # the model, so it computes everything and commits nothing.
    assert result.recorded_paths == ()
    assert result.envelopes


def test_a_view_too_short_for_the_window_fails_loudly(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = write_corpus(
        tmp_path / "corpus", _corpus_rows(normalized_row)
    )
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )

    with pytest.raises(NoveltyBenchmarkError, match="no measurable position"):
        _measure(_config(pool, checkpoint, onset_plies=500))


def test_cli_reads_a_sweep_without_recording_it(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    training_run: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    normalized, manifest = write_corpus(
        tmp_path / "corpus", _corpus_rows(normalized_row)
    )
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )
    config_path = tmp_path / "novelty.toml"
    config_path.write_text(
        "\n".join(
            [
                f'pool = "{pool}"',
                "",
                "[model]",
                f'checkpoint_path = "{checkpoint}"',
                'device = "cpu"',
                "",
                "[perturbation]",
                f"onset_plies = {ONSET}",
                f"window_moves = {WINDOW}",
                "doses = [0.0, 1.0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status = main(["eval", "novelty", "--config", str(config_path), "--no-record"])

    output = capsys.readouterr().out
    assert status == 0
    assert "Dose response" in output
    assert "retention is against this checkpoint's own control" in output
    assert "Predicate retention by dose:" in output
    assert "material_gain" in output
    # A shakedown reading commits nothing, so no recorded files are named.
    assert "Recorded" not in output


def test_cli_reports_an_unusable_sweep_as_a_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "novelty.toml"
    missing = tmp_path / "missing"
    config_path.write_text(f'pool = "{missing}"\n', encoding="utf-8")

    status = main(["eval", "novelty", "--config", str(config_path), "--no-record"])

    assert status == 2
    assert "anthro eval novelty:" in capsys.readouterr().err


def _config(
    pool: Path,
    checkpoint: Path,
    *,
    doses: list[float] | None = None,
    onset_plies: int = ONSET,
) -> ResolvedConfig[NoveltyBenchmarkConfig]:
    return ResolvedConfig(
        value=NoveltyBenchmarkConfig.model_validate(
            {
                "pool": str(pool),
                "view": {"name": "novelty"},
                "model": {"checkpoint_path": str(checkpoint), "device": "cpu"},
                "loader": {"batch_size": 2},
                "perturbation": {
                    "onset_plies": onset_plies,
                    "window_moves": WINDOW,
                    "doses": doses or [0.0, 1.0],
                },
                "noise": {"resamples": 100},
            }
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )


def _freeze(tmp_path: Path, normalized: Path, manifest: Path) -> Path:
    output = tmp_path / "pool"
    freeze_pool(
        ResolvedConfig(
            value=PoolConfig.model_validate(
                {
                    "pool_id": "novelty-fixture",
                    "normalized": str(normalized),
                    "manifest": str(manifest),
                }
            ),
            provenance=ConfigProvenance(source=None, overrides=()),
        ),
        output,
    )
    return output
