"""Tests for the offline checkpoint evaluation runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.evaluation import (
    CheckpointEvaluationConfig,
    CheckpointEvaluationError,
    CheckpointEvaluationResult,
    LeakageError,
    PoolConfig,
    freeze_pool,
)
from anthro_chess.evaluation import leakage as leakage_module
from anthro_chess.evaluation.aggregation import (
    OPENING_FAMILY_DIMENSION,
    OPENING_TIER_DIMENSION,
    PHASE_DIMENSION,
    RULE_CASE_DIMENSION,
)
from anthro_chess.evaluation.benchmarks import benchmark_registry, run_benchmark
from anthro_chess.evaluation.checkpoint import (
    ADJUDICATION_KIND,
    DEPENDENCY_KIND,
    HELD_OUT_KIND,
)
from anthro_chess.evaluation.cost import BENCHMARK_COST_KIND
from anthro_chess.evaluation.dependency import ConditioningKind
from anthro_chess.evaluation.opening_frequency import UNCLASSIFIED_TIER
from anthro_chess.evaluation.results import (
    DetailStore,
    ResultsStore,
)
from anthro_chess.evaluation.results.metrics import (
    HELD_OUT_MOVE_LOSS_BY_OPENING_TIER,
)
from anthro_chess.evaluation.slices import (
    GamePhase,
    PositionCharacteristic,
    PositionPredicate,
)
from anthro_chess.inference import CheckpointModelRunner
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


def _evaluate(
    resolved_config: ResolvedConfig[CheckpointEvaluationConfig],
    *,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> CheckpointEvaluationResult:
    """Evaluate one checkpoint the way both callers do, through the driver."""

    return cast(
        CheckpointEvaluationResult,
        run_benchmark(
            benchmark_registry()["run"],
            resolved_config,
            store=store,
            detail=detail,
        ),
    )


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

    result = _evaluate(
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
    # Three readings from one invocation, and one record of what that
    # invocation cost, which belongs to none of them.
    assert kinds == {
        HELD_OUT_KIND,
        DEPENDENCY_KIND,
        ADJUDICATION_KIND,
        BENCHMARK_COST_KIND,
    }
    adjudicated = next(item for item in recorded if item.kind == ADJUDICATION_KIND)
    assert {item.metric for item in adjudicated.measurements} >= {
        "adjudicated.material_gain_human_rate",
        "adjudicated.material_gain_selected_rate",
        "adjudicated.material_gain_best_rank",
    }
    assert result.adjudication is not None
    assert PositionPredicate.MATERIAL_GAIN in result.adjudication.predicates
    # Three result envelopes and what the invocation cost.
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


def test_an_ordinary_reading_never_classifies_an_opening(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    """Classifying costs a replay per game, and the table needs the axis."""

    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )
    store = ResultsStore(tmp_path / "results")

    result = _evaluate(_config(pool, checkpoint), store=store)

    assert result.slices.dimensions[OPENING_FAMILY_DIMENSION] == {}
    assert result.slices.dimensions[OPENING_TIER_DIMENSION] == {}
    assert result.opening_frequency is None
    assert result.opening_tail is None
    held_out = next(item for item in store.results() if item.kind == HELD_OUT_KIND)
    committed = {item.metric for item in held_out.measurements}
    assert not committed & {
        definition.identifier
        for definition in HELD_OUT_MOVE_LOSS_BY_OPENING_TIER.values()
    }


def test_counting_training_frequency_commits_the_tier_series(
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

    result = _evaluate(
        _config(pool, checkpoint, openings={"training_frequency": True}),
        store=store,
        detail=detail,
    )

    assert result.opening_frequency is not None
    # Both training games play the shared opening line, so the one family the
    # corpus holds is every training game and the test pool's other games are
    # unnamed.
    assert result.opening_frequency.family_games == {"Ruy Lopez": 2}
    assert set(result.slices.dimensions[OPENING_TIER_DIMENSION]) == {
        "common_opening",
        UNCLASSIFIED_TIER,
    }

    held_out = next(item for item in store.results() if item.kind == HELD_OUT_KIND)
    metrics = {item.metric: item for item in held_out.measurements}
    common = result.slices.slice_summary(OPENING_TIER_DIMENSION, "common_opening")
    assert common is not None
    assert metrics["held_out.move_loss_common_opening"].value == pytest.approx(
        common.move_loss
    )
    assert metrics["held_out.move_loss_common_opening"].sample_size == (
        common.position_count
    )
    # A tier series is an ordinary slice of the same pass, so the bootstrap
    # qualifies it like every other one.
    assert metrics["held_out.move_loss_common_opening"].dispersion is not None
    held_out.verify()

    assert result.opening_tail is not None
    assert [row.family for row in result.opening_tail.families] == ["Ruy Lopez"]
    assert held_out.detail is not None
    payload = detail.read(held_out.detail)
    tail = payload["opening_tail"]
    assert isinstance(tail, Mapping)
    assert tail["families"][0]["training_share"] == pytest.approx(1.0)
    assert payload["opening_frequency"]["split"] == "train"


def test_a_foreign_training_corpus_refuses_the_frequency_axis(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    """A tier is a share of a corpus the scored games' digest cannot pin."""

    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    # Ratings no pool game carries, so the content comparison the leakage check
    # falls back to across corpora finds nothing shared.
    other_normalized, other_manifest = write_corpus(
        tmp_path / "other",
        [
            normalized_row(101, split="train", plies=10, rating=1234),
            normalized_row(102, split="test", plies=6, rating=1345),
        ],
        source_id="other",
    )
    checkpoint = training_run(
        tmp_path / "run",
        normalized=other_normalized,
        manifest=other_manifest,
    )

    with pytest.raises(CheckpointEvaluationError, match="the same one"):
        _evaluate(_config(pool, checkpoint, openings={"training_frequency": True}))


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

    first = _evaluate(_config(pool, checkpoint))
    second = _evaluate(_config(pool, checkpoint))

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

    result = _evaluate(
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

    # The human rate carries none because the human took the mate at both
    # opportunities, so no resample of these games can move it. A spread of zero
    # there would clear every later delta rather than describe one.
    assert metrics["adjudicated.mate_available_policy_mass"].dispersion is not None
    assert metrics["adjudicated.mate_available_human_rate"].dispersion is None


def test_evaluation_bootstraps_a_spread_for_every_series_it_reports(
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

    result = _evaluate(_config(pool, checkpoint), store=store)

    # One pass bootstraps the held-out and adjudicated series together, so the
    # spreads are checked against everything that pass recorded rather than
    # against one envelope of it.
    reported = {
        item.metric: item
        for envelope in result.envelopes
        for item in envelope.measurements
    }
    spread = {
        metric: item.dispersion
        for metric, item in reported.items()
        if item.dispersion is not None
    }
    assert "held_out.move_loss" in spread
    # A rate the fixture's games all agree on carries none rather than a zero,
    # since a redraw of those games observed that it could not move the rate
    # rather than that nothing could.
    assert "adjudicated.material_gain_best_rank" in spread
    assert "adjudicated.material_gain_selected_rate" not in spread
    for dispersion in spread.values():
        assert dispersion.kind == "data-sampling"
        assert dispersion.bound >= dispersion.value

    # Nothing is filed against the series, so nothing can collide there when
    # the next reading of the same series measures its own spread.
    assert store.characterizations() == ()


def test_a_sampled_spread_is_reproducible_and_can_be_declined(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )

    first = _evaluate(_config(pool, checkpoint))
    second = _evaluate(_config(pool, checkpoint))
    disabled = _evaluate(
        _config(pool, checkpoint, noise={"enabled": False}),
    )

    assert first.dispersions
    assert first.dispersions == second.dispersions
    assert disabled.dispersions == {}


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

    result = _evaluate(_config(pool, checkpoint))

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
    # The checkpoint's own count, not the larger one its finished run reports.
    assert dependency.maturity.processed_positions == 64
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


def test_the_dependency_tests_score_each_conditioning_once(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nine passes over the view, not eleven.

    The anchor comparison scores two fixed conditionings the
    cross-conditioning table also wants, and re-scoring them cost two of the
    eleven passes this reading used to make. Counted rather than asserted
    structurally, because the passes are what the reading costs: the same
    evaluation with the dependency block off is exactly one pass, which is
    what turns a call count into a pass count on any fixture.

    That the retained scores equal a standalone pass' is
    ``ActiveBatch.rescored``'s guarantee, which ``test_policy`` pins.

    ``configs/evaluation/checkpoint-suite.toml`` states this same count to
    argue what a reduced sweep shrinks, and nothing else here would catch it
    going stale, so a change that moves this number belongs there too.
    """

    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )
    calls = 0
    scored = CheckpointModelRunner.action_logits

    def counted(self: CheckpointModelRunner, batch: MoveModelBatch) -> Tensor:
        nonlocal calls
        calls += 1
        return scored(self, batch)

    monkeypatch.setattr(CheckpointModelRunner, "action_logits", counted)

    _evaluate(_config(pool, checkpoint, dependency={"enabled": False}))
    batches = calls
    calls = 0
    _evaluate(_config(pool, checkpoint))

    assert batches > 0
    assert calls == 9 * batches


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

    result = _evaluate(_config(pool, checkpoint))

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

    full = _evaluate(_config(pool, checkpoint))
    prefix = _evaluate(
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
        _evaluate(_config(pool, checkpoint))


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

    result = _evaluate(_config(pool, clean_checkpoint))

    assert result.leakage.algorithm == "content-hash-intersection-v1"
    assert result.leakage.same_source_corpus is False
    assert result.leakage.overlapping_games == 0
    with pytest.raises(LeakageError, match="content-hash-intersection-v1"):
        _evaluate(_config(pool, overlapping_checkpoint))


def test_a_repeated_leakage_check_reuses_the_scan_it_already_made(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sweep checks one checkpoint against one pool once per benchmark."""

    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )
    first = _evaluate(_config(pool, checkpoint))

    def unreadable(*args: object, **kwargs: object) -> list[dict[str, Any]]:
        raise AssertionError("a repeated leakage check re-read the corpus")

    monkeypatch.setattr(leakage_module, "read_normalized_rows", unreadable)
    second = _evaluate(_config(pool, checkpoint))

    assert second.leakage.algorithm == "game-id-intersection-v1"
    assert second.leakage.as_record() == first.leakage.as_record()


def test_a_repeated_content_comparison_reuses_its_scans_too(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    training_run: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The branch that costs a full corpus read is the one worth reusing."""

    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [
            normalized_row(1, split="train", plies=8),
            normalized_row(2, split="test", plies=10, rating=1500),
            normalized_row(3, split="test", plies=8, rating=2100),
        ],
    )
    pool = _freeze(tmp_path, normalized, manifest)
    # A separate preparation of unrelated games, so ids mean nothing across the
    # two and the comparison has to read what each side contains.
    disjoint, disjoint_manifest = write_corpus(
        tmp_path / "disjoint",
        [
            normalized_row(21, split="train", plies=4, result="0-1"),
            normalized_row(22, split="validation", plies=8),
        ],
        source_id="disjoint",
    )
    checkpoint = training_run(
        tmp_path / "run", normalized=disjoint, manifest=disjoint_manifest
    )
    first = _evaluate(_config(pool, checkpoint))

    def unreadable(*args: object, **kwargs: object) -> list[dict[str, Any]]:
        raise AssertionError("a repeated content comparison re-read a corpus")

    monkeypatch.setattr(leakage_module, "read_normalized_rows", unreadable)
    second = _evaluate(_config(pool, checkpoint))

    assert second.leakage.algorithm == "content-hash-intersection-v1"
    assert second.leakage.as_record() == first.leakage.as_record()


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
        _evaluate(_config(pool, checkpoint))


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
        _evaluate(_config(pool, checkpoint))


def test_evaluation_refuses_a_pool_this_reading_is_not_defined_over(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    """The generation a selection pins reaches the loader from here.

    `test_pool` owns what the refusal compares; this owns that the canonical
    reading asks for it at all, which is the half that fails silently.
    """

    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )

    with pytest.raises(CheckpointEvaluationError, match="expected 0{64}"):
        _evaluate(_config(pool, checkpoint, expected_pool_game_ids_sha256="0" * 64))


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


def _config(
    pool: Path,
    checkpoint: Path,
    *,
    view: dict[str, Any] | None = None,
    noise: dict[str, Any] | None = None,
    dependency: dict[str, Any] | None = None,
    openings: dict[str, Any] | None = None,
    expected_pool_game_ids_sha256: str | None = None,
) -> ResolvedConfig[CheckpointEvaluationConfig]:
    return ResolvedConfig(
        value=CheckpointEvaluationConfig.model_validate(
            {
                "pool": str(pool),
                "expected_pool_game_ids_sha256": expected_pool_game_ids_sha256,
                "view": view or {"name": "canonical"},
                "model": {"checkpoint_path": str(checkpoint), "device": "cpu"},
                "loader": {"batch_size": 4},
                "dependency": {
                    "minimum_slice_positions": 1,
                    "minimum_prefix_decisions": 1,
                    **(dependency or {}),
                },
                "noise": noise or {"resamples": 100},
                "openings": openings or {},
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
