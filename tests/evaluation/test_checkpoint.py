"""Tests for the offline checkpoint evaluation runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.evaluation import (
    CheckpointEvaluationConfig,
    CheckpointEvaluationError,
    LeakageError,
    PoolConfig,
    evaluate_checkpoint,
    freeze_pool,
)
from anthro_chess.evaluation.aggregation import PHASE_DIMENSION, RULE_CASE_DIMENSION
from anthro_chess.evaluation.checkpoint import (
    ADJUDICATION_KIND,
    DEPENDENCY_KIND,
    HELD_OUT_KIND,
    _batch_keys,
)
from anthro_chess.evaluation.dependency import ConditioningKind
from anthro_chess.evaluation.results import DetailStore, ResultsStore
from anthro_chess.evaluation.slices import (
    GamePhase,
    PositionCharacteristic,
    PositionPredicate,
)
from anthro_chess.interfaces.cli import main
from anthro_chess.models import MoveModelBatch

#: A middlegame position where the side to move has a promotion available and
#: the shared opening line never reaches, so the rule-case slices are exercised
#: against a real position rather than a hand-authored label.
PROMOTION_FEN = "8/5P2/8/7k/8/8/8/K7 w - - 0 40"
PROMOTION_MOVES = ("f7f8q", "h5g5", "f8f3", "g5h6")

#: A position with an en-passant capture available to the side to move.
EN_PASSANT_FEN = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 20"
EN_PASSANT_MOVES = ("e5d6", "e8d8", "d6d7", "d8d7")


@pytest.fixture
def corpus(
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> Callable[[Path], tuple[Path, Path]]:
    """Return a factory writing a mixed-split corpus with varied ratings."""

    def build(directory: Path) -> tuple[Path, Path]:
        rows = [
            normalized_row(1, split="train", plies=10, rating=1500),
            normalized_row(2, split="train", plies=8, rating=2100),
            normalized_row(3, split="validation", plies=6, rating=1100),
            normalized_row(4, split="test", plies=10, rating=1100),
            normalized_row(5, split="test", plies=10, rating=1500),
            normalized_row(6, split="test", plies=8, rating=2100),
            normalized_row(7, split="test", plies=6, rating=None),
            normalized_row(
                8,
                split="test",
                rating=1500,
                moves=PROMOTION_MOVES,
                initial_position=PROMOTION_FEN,
            ),
            normalized_row(
                9,
                split="test",
                rating=2100,
                moves=EN_PASSANT_MOVES,
                initial_position=EN_PASSANT_FEN,
            ),
        ]
        return write_corpus(directory, rows)

    return build


def test_evaluation_records_sliced_results_over_the_frozen_pool(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )
    store = ResultsStore(tmp_path / "results")
    detail = DetailStore(tmp_path / "detail")

    result = evaluate_checkpoint(
        _config(pool, checkpoint),
        store=store,
        detail=detail,
    )

    recorded = store.results()
    kinds = {envelope.kind for envelope in recorded}
    held_out = next(item for item in recorded if item.kind == HELD_OUT_KIND)
    metrics = {item.metric: item for item in held_out.measurements}
    # Material gain is the one predicate common enough to fire on ordinary
    # play, so an adjudication record appears where no forced outcome would
    # have produced one.
    assert kinds == {HELD_OUT_KIND, DEPENDENCY_KIND, ADJUDICATION_KIND}
    adjudicated = next(item for item in recorded if item.kind == ADJUDICATION_KIND)
    assert {item.metric for item in adjudicated.measurements} >= {
        "adjudicated.material_gain_human_rate",
        "adjudicated.material_gain_selected_rate",
        "adjudicated.material_gain_best_rank",
    }
    assert result.adjudication is not None
    assert PositionPredicate.MATERIAL_GAIN in result.adjudication.predicates
    # Three result envelopes plus the data-sampling floors bootstrapped from
    # the same pass, all committed.
    assert len(result.recorded_paths) == 4
    assert result.checkpoint.step == 1
    assert result.checkpoint.parameter_sha256 is not None
    assert result.dataset.pool_id == "fixture-test"
    assert result.dataset.selected_games == 6
    assert result.view.selected_games == 6

    overall = result.slices.overall
    # Six pool games of 10, 10, 8, 6, 4, and 4 plies.
    assert overall.position_count == 42
    assert metrics["held_out.move_loss"].value == pytest.approx(overall.move_loss)
    assert metrics["held_out.move_loss"].sample_size == overall.position_count
    assert metrics["legality.mask_penalty"].value == pytest.approx(overall.mask_penalty)
    assert 0.0 <= metrics["held_out.top1_accuracy"].value <= 1.0

    # Legality and prediction are held fixed per phase, per rating band, and
    # per rule case, so a composition shift cannot masquerade as a change.
    assert "legality.mask_penalty_opening" in metrics
    assert "held_out.move_loss_opening" in metrics
    assert "held_out.move_loss_1200_to_1599" in metrics
    assert "held_out.move_loss_unrated" in metrics
    assert "legality.mask_penalty_promotion" in metrics
    assert "legality.mask_penalty_en_passant" in metrics
    assert metrics["legality.mask_penalty_en_passant"].sample_size == 1

    phases = set(result.slices.dimensions[PHASE_DIMENSION])
    rule_cases = set(result.slices.dimensions[RULE_CASE_DIMENSION])
    assert phases == {str(GamePhase.OPENING), str(GamePhase.ENDGAME)}
    assert str(PositionCharacteristic.PROMOTION) in rule_cases
    assert str(PositionCharacteristic.TERMINAL) not in rule_cases

    assert held_out.detail is not None
    payload = detail.read(held_out.detail)
    assert payload["leakage"]["overlapping_games"] == 0
    assert payload["view"]["selected_games"] == 6
    assert payload["positions"] is None
    held_out.verify()


def test_repeated_evaluation_reproduces_every_measurement(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )

    first = evaluate_checkpoint(_config(pool, checkpoint))
    second = evaluate_checkpoint(_config(pool, checkpoint))

    assert first.slices.as_record() == second.slices.as_record()
    assert first.dependency is not None
    assert second.dependency is not None
    assert first.dependency.as_record() == second.dependency.as_record()
    assert [item.fingerprint for item in first.envelopes[0].measurements] == [
        item.fingerprint for item in second.envelopes[0].measurements
    ]
    assert first.recorded_paths == ()


def test_evaluation_records_human_referenced_forced_outcomes(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    forced_fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [
            normalized_row(1, split="train", plies=4, rating=1500),
            normalized_row(
                2,
                split="test",
                rating=1100,
                initial_position=forced_fen,
                moves=("f7f8",),
            ),
            normalized_row(
                3,
                split="test",
                rating=2100,
                initial_position=forced_fen,
                moves=("f7f8",),
            ),
        ],
    )
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )
    store = ResultsStore(tmp_path / "results")
    detail = DetailStore(tmp_path / "detail")

    result = evaluate_checkpoint(
        _config(pool, checkpoint),
        store=store,
        detail=detail,
    )

    assert result.adjudication is not None
    mate = result.adjudication.predicates[PositionPredicate.MATE_AVAILABLE]
    assert mate.overall.opportunities == 2
    assert set(mate.rating_bands) == {"under_1200", "2000_plus"}

    envelope = next(item for item in result.envelopes if item.kind == ADJUDICATION_KIND)
    metrics = {item.metric: item for item in envelope.measurements}
    assert metrics["adjudicated.mate_available_human_rate"].value == pytest.approx(1.0)
    assert metrics["adjudicated.mate_available_selected_rate"].sample_size == 2
    assert envelope.detail is not None
    payload = detail.read(envelope.detail)
    assert payload["predicates"]["mate_available"]["overall"]["opportunities"] == 2
    envelope.verify()

    assert result.noise is not None
    floor_metrics = {entry.metric for entry in result.noise.floors}
    assert "adjudicated.mate_available_human_rate" in floor_metrics


def test_evaluation_bootstraps_a_floor_for_every_series_it_reports(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )
    store = ResultsStore(tmp_path / "results")

    result = evaluate_checkpoint(_config(pool, checkpoint), store=store)

    noise = result.noise
    assert noise is not None
    assert noise.kind == "data-sampling"

    # One pass bootstraps the held-out and adjudicated series together, so the
    # floors are checked against everything that pass recorded rather than
    # against one envelope of it.
    reported = {
        item.metric: item.fingerprint
        for envelope in result.envelopes
        for item in envelope.measurements
    }
    floors = {entry.metric: entry for entry in noise.floors}
    # A floor is only readable beside the value it qualifies, so every floor
    # has to land on the same series as the measurement it describes.
    assert set(floors) <= set(reported)
    assert "held_out.move_loss" in floors
    assert "adjudicated.material_gain_selected_rate" in floors
    for metric, entry in floors.items():
        assert entry.fingerprint == reported[metric]
        assert entry.sampling_units == result.view.selected_games

    assert store.characterizations() == (noise,)


def test_a_noise_floor_is_reproducible_and_can_be_declined(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )

    first = evaluate_checkpoint(_config(pool, checkpoint))
    second = evaluate_checkpoint(_config(pool, checkpoint))
    disabled = evaluate_checkpoint(
        _config(pool, checkpoint, noise={"enabled": False}),
    )

    assert first.noise is not None
    assert second.noise is not None
    assert first.noise.floors == second.noise.floors
    assert disabled.noise is None


def test_dependency_tests_report_degradation_without_a_verdict(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )

    result = evaluate_checkpoint(_config(pool, checkpoint))

    dependency = result.dependency
    assert dependency is not None
    assert {item.conditioning.name for item in dependency.corruptions} == {
        "shuffled",
        "constant",
        "absent",
    }
    for item in dependency.corruptions:
        assert item.position_count == dependency.rated_position_count
        assert item.degradation == pytest.approx(
            item.move_loss - dependency.true_move_loss
        )
    cells = dependency.cross_conditioning.cells
    assert {cell.conditioning_rating for cell in cells} == {1000, 1400, 1800, 2200}
    assert {cell.rating_band for cell in cells} == {
        "under_1200",
        "1200_to_1599",
        "2000_plus",
    }
    assert dependency.maturity.step == 1
    assert 0.0 <= dependency.anchor_agreement_rate <= 1.0

    measurements = {
        item.metric
        for envelope in result.envelopes
        if envelope.kind == DEPENDENCY_KIND
        for item in envelope.measurements
    }
    assert "dependency.rating_shuffled_degradation" in measurements
    assert "dependency.rating_absent_degradation" in measurements
    assert "dependency.rating_anchor_policy_divergence" in measurements


def test_absent_conditioning_changes_what_the_model_is_shown(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )

    result = evaluate_checkpoint(_config(pool, checkpoint))

    dependency = result.dependency
    assert dependency is not None
    absent = dependency.corruption(ConditioningKind.ABSENT)
    assert absent is not None
    # An untrained fixture model need not degrade, but the treatments must be
    # genuinely different inputs rather than three copies of one pass.
    assert absent.move_loss != pytest.approx(dependency.true_move_loss, abs=1e-12)


def test_a_prefix_view_scores_fewer_plies_and_starts_its_own_series(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )

    full = evaluate_checkpoint(_config(pool, checkpoint))
    prefix = evaluate_checkpoint(
        _config(pool, checkpoint, view={"name": "prefix", "prefix_plies": 4})
    )

    full_fingerprints = {
        item.metric: item.fingerprint for item in full.envelopes[0].measurements
    }
    prefix_fingerprints = {
        item.metric: item.fingerprint for item in prefix.envelopes[0].measurements
    }
    assert prefix.slices.overall.position_count < full.slices.overall.position_count
    assert prefix.slices.overall.position_count == 4 * prefix.view.selected_games
    assert (
        full_fingerprints["held_out.move_loss"]
        != prefix_fingerprints["held_out.move_loss"]
    )
    assert prefix.dataset.view == "prefix"


def test_leakage_check_refuses_a_checkpoint_trained_on_pool_games(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    rows = [
        normalized_row(1, split="train", plies=8),
        normalized_row(2, split="test", plies=8),
    ]
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    pool = _freeze(tmp_path, normalized, manifest)
    leaked = [
        normalized_row(1, split="train", plies=8),
        normalized_row(2, split="train", plies=8),
    ]
    leaked_normalized, leaked_manifest = write_corpus(tmp_path / "leaked", leaked)
    checkpoint = training_run(
        tmp_path / "run",
        normalized=leaked_normalized,
        manifest=leaked_manifest,
    )

    with pytest.raises(LeakageError, match="appear in the checkpoint's train split"):
        evaluate_checkpoint(_config(pool, checkpoint))


def test_leakage_compares_content_when_the_corpora_differ(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [
            normalized_row(1, split="train", plies=8),
            normalized_row(2, split="test", plies=8),
        ],
    )
    pool = _freeze(tmp_path, normalized, manifest)
    # The same games, renumbered by a separate preparation run. Ids no longer
    # mean the same thing, so only recorded content can answer the question.
    renumbered, renumbered_manifest = write_corpus(
        tmp_path / "renumbered",
        [
            normalized_row(11, split="train", plies=8),
            normalized_row(12, split="validation", plies=8),
        ],
        source_id="renumbered",
    )
    disjoint, disjoint_manifest = write_corpus(
        tmp_path / "disjoint",
        [
            normalized_row(21, split="train", plies=4, result="0-1"),
            normalized_row(22, split="validation", plies=8),
        ],
        source_id="disjoint",
    )
    overlapping_checkpoint = training_run(
        tmp_path / "overlapping",
        normalized=renumbered,
        manifest=renumbered_manifest,
    )
    clean_checkpoint = training_run(
        tmp_path / "clean",
        normalized=disjoint,
        manifest=disjoint_manifest,
    )

    result = evaluate_checkpoint(_config(pool, clean_checkpoint))

    assert result.leakage.algorithm == "content-hash-intersection-v1"
    assert result.leakage.same_source_corpus is False
    assert result.leakage.overlapping_games == 0
    with pytest.raises(LeakageError, match="content-hash-intersection-v1"):
        evaluate_checkpoint(_config(pool, overlapping_checkpoint))


def test_leakage_check_reports_a_training_corpus_this_machine_cannot_read(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["metadata"]["data"]["train"]["normalized_paths"] = [
        str(tmp_path / "moved" / "games.parquet")
    ]
    torch.save(payload, checkpoint)

    with pytest.raises(LeakageError, match="leakage.training_normalized"):
        evaluate_checkpoint(_config(pool, checkpoint))


def test_evaluation_rejects_an_incompatible_checkpoint(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["metadata"]["encoding"] = {"name": "other-encoding"}
    torch.save(payload, checkpoint)

    with pytest.raises(CheckpointEvaluationError, match="encoding is incompatible"):
        evaluate_checkpoint(_config(pool, checkpoint))


def test_cli_runs_an_evaluation_without_recording_it(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    capsys: pytest.CaptureFixture[str],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )
    config_path = tmp_path / "evaluation.toml"
    config_path.write_text(
        "\n".join(
            [
                f'pool = "{pool}"',
                "",
                "[model]",
                f'checkpoint_path = "{checkpoint}"',
                'device = "cpu"',
                "",
                "[dependency]",
                "enabled = false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status = main(["eval", "run", "--config", str(config_path), "--no-record"])

    output = capsys.readouterr().out
    assert status == 0
    assert "move_loss" in output
    assert "Legality and move loss by phase:" in output
    assert "Recorded: nothing" in output


def test_cli_reports_a_leaking_checkpoint_as_a_failure(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    capsys: pytest.CaptureFixture[str],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [
            normalized_row(1, split="train", plies=8),
            normalized_row(2, split="test", plies=8),
        ],
    )
    pool = _freeze(tmp_path, normalized, manifest)
    leaked, leaked_manifest = write_corpus(
        tmp_path / "leaked",
        [
            normalized_row(1, split="train", plies=8),
            normalized_row(2, split="train", plies=8),
        ],
    )
    checkpoint = training_run(
        tmp_path / "run",
        normalized=leaked,
        manifest=leaked_manifest,
    )
    config_path = tmp_path / "evaluation.toml"
    config_path.write_text(
        f'pool = "{pool}"\n\n[model]\ncheckpoint_path = "{checkpoint}"\n'
        'device = "cpu"\n',
        encoding="utf-8",
    )

    status = main(["eval", "run", "--config", str(config_path), "--no-record"])

    assert status == 2
    assert "anthro eval run:" in capsys.readouterr().err


def test_position_keys_are_read_off_the_device_in_one_pass(
    sequence_batch: Callable[..., Any],
    device_read_trap: Callable[[Any], Any],
) -> None:
    """The dependency pass identifies a whole batch without a scalar read.

    It keys three conditioning treatments per batch, so asking the device for
    each game id and ply index separately costs a stall per position per arm.
    CPU hides that, which is why this asserts on the access pattern.
    """

    batch = MoveModelBatch.from_sequence_batch(
        sequence_batch((("e2e4", "e7e5"), 1500, 1600))
    )

    assert _batch_keys(device_read_trap(batch)) == _batch_keys(batch)


def test_position_keys_preserve_a_game_id_past_the_signed_maximum(
    sequence_batch: Callable[..., Any],
) -> None:
    """The unsigned id has to survive the read; see the scoring path's own test."""

    batch = MoveModelBatch.from_sequence_batch(sequence_batch((("e2e4",), None, None)))
    identifier = 2**64 - 1234567
    widened = replace(
        batch,
        game_ids=torch.full_like(batch.game_ids, identifier, dtype=torch.uint64),
    )

    assert _batch_keys(widened) == ((identifier, 0),)


def _config(
    pool: Path,
    checkpoint: Path,
    *,
    view: dict[str, Any] | None = None,
    noise: dict[str, Any] | None = None,
) -> ResolvedConfig[CheckpointEvaluationConfig]:
    return ResolvedConfig(
        value=CheckpointEvaluationConfig.model_validate(
            {
                "pool": str(pool),
                "view": view or {"name": "canonical"},
                "model": {"checkpoint_path": str(checkpoint), "device": "cpu"},
                "loader": {"batch_size": 4},
                "dependency": {
                    "minimum_slice_positions": 1,
                    "minimum_prefix_decisions": 1,
                },
                "noise": noise or {"resamples": 100},
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
                    "pool_id": "fixture-test",
                    "normalized": str(normalized),
                    "manifest": str(manifest),
                }
            ),
            provenance=ConfigProvenance(source=None, overrides=()),
        ),
        output,
    )
    return output
