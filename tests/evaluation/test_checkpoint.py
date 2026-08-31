"""Tests for the offline checkpoint evaluation runner."""

from __future__ import annotations

import json
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
    DependencyBenchmarkConfig,
    DependencyBenchmarkResult,
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
    _sessions,
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
    # Two readings from one invocation, and one record of what that invocation
    # cost, which belongs to neither. Rating dependency is absent because it is
    # its own benchmark over its own view.
    assert kinds == {
        HELD_OUT_KIND,
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
    # Two result envelopes and what the invocation cost.
    assert len(result.recorded_paths) == 3
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
    first_dependency = _measure_dependency(_dependency_config(pool, checkpoint))
    second_dependency = _measure_dependency(_dependency_config(pool, checkpoint))

    assert first.slices.as_record() == second.slices.as_record()
    assert (
        first_dependency.dependency.as_record()
        == second_dependency.dependency.as_record()
    )
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
    assert "adjudicated.material_gain_best_rank" in spread
    for dispersion in spread.values():
        assert dispersion.bound >= dispersion.value
        # A quantity the fixture's games all agree on carries no spread rather
        # than a zero one, since a redraw of those games observed that it could
        # not move rather than that nothing could. Which quantity that is
        # depends on the fixture model's weights, so the rule is stated over
        # every spread reported instead of over the one that happens to be flat.
        assert dispersion.value > 0.0


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

    result = _measure_dependency(_dependency_config(pool, checkpoint))

    dependency = result.dependency
    assert {item.conditioning.name for item in dependency.corruptions} == {
        "shuffled",
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
    assert "dependency.rating_cross_conditioning_penalty" in measurements


def test_the_dependency_reading_carries_a_spread_for_what_it_can_resample(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    """The quantities that count something other than games carry none.

    The cross-conditioning match rate and the within-game response declare why
    in the registry, so a report renders them ``unqualifiable`` rather than
    sending a reader after a spread nothing can estimate.

    The rest are asserted as a property rather than as a list. Whether a given
    dependency metric carries a spread also depends on whether the fixture's
    weights make it vary at all, so naming which ones do would pin the model
    this fixture happens to build rather than anything the reading guarantees.
    """

    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )

    result = _measure_dependency(_dependency_config(pool, checkpoint))

    reported = {
        item.metric: item
        for envelope in result.envelopes
        if envelope.kind == DEPENDENCY_KIND
        for item in envelope.measurements
    }
    undispersed = {
        metric for metric, item in reported.items() if item.dispersion is None
    }
    assert undispersed >= {
        "dependency.rating_cross_conditioning_match_rate",
        "dependency.rating_within_game_response",
    }
    dispersed = {
        metric: item.dispersion
        for metric, item in reported.items()
        if item.dispersion is not None
    }
    assert dispersed
    for dispersion in dispersed.values():
        # A quantity that could not move across the fixture's games is absent
        # above rather than present with a zero, which would clear every later
        # delta it was ever combined into.
        assert dispersion.value > 0.0
    divergence = reported["dependency.rating_anchor_policy_divergence"].dispersion
    assert divergence is not None
    assert divergence.units == len(result.dependency.per_game_totals)


def test_the_dependency_tests_score_each_conditioning_once(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seven passes over the view, one per distinct conditioning.

    The anchor comparison scores two fixed conditionings the cross-conditioning
    table also wants, and the trajectory needs the true-conditioning policy the
    primary pass already computed, so all three are carried rather than
    re-scored. Counted rather than asserted structurally, because the passes
    are what the reading costs: the checkpoint reading over the same view is
    exactly one pass, which is what turns a call count into a pass count on any
    fixture.

    That the carried scores equal a standalone pass' is
    ``ActiveBatch.rescored``'s guarantee, which ``test_policy`` pins.

    This is the only place the count itself is written down. The configuration
    files, the schema docstring and ``docs/evaluation.md`` state the rule that
    produces it, so a change to the treatments fails here and nowhere else.
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

    view = {"name": "canonical"}
    _evaluate(_config(pool, checkpoint, view=view))
    batches = calls
    calls = 0
    _measure_dependency(_dependency_config(pool, checkpoint, view=view))

    assert batches > 0
    assert calls == 7 * batches


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

    result = _measure_dependency(_dependency_config(pool, checkpoint))

    dependency = result.dependency
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


def test_leakage_check_refuses_a_checkpoint_trained_on_the_pool_split(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    """Reading the split a pool was cut from puts every pool game in the run."""

    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run",
        normalized=normalized,
        manifest=manifest,
        split="test",
    )

    with pytest.raises(LeakageError, match="which is the split this pool was cut from"):
        _evaluate(_config(pool, checkpoint))


def test_leakage_check_refuses_a_pool_holding_games_the_recipe_puts_in_training(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    """A pool whose games the corpus' own recipe assigns elsewhere is refused.

    Split names agreeing is what makes disjointness structural. This is the case
    the names alone cannot see: the pool does not hold the split it claims, and
    putting its ids back through the recipe is what notices.
    """

    normalized, manifest = corpus(tmp_path / "corpus")
    # Declared before either side reads it, so the pool and the checkpoint agree
    # on the corpus. Nothing lands in test under these fractions, so every game
    # the pool holds is one the recipe puts in the training split.
    record = json.loads(manifest.read_text())
    record["split"] = {
        "algorithm": "sha256-threshold-v2",
        "seed": "fixture",
        "test_fraction": 0.0,
        "validation_fraction": 0.0,
    }
    manifest.write_text(json.dumps(record))
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )

    with pytest.raises(LeakageError, match="under the corpus' own split recipe"):
        _evaluate(_config(pool, checkpoint))


def test_a_grown_corpus_is_still_settled_by_the_shared_split_recipe(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    """A game keeps its id as a corpus grows, so its split survives with it.

    Widening the corpus and re-cutting the pool is this project's own workflow,
    and it leaves a checkpoint trained on one generation scored against a pool
    cut from the next. The manifests differ; the recipe does not, which is what
    still settles the question.
    """

    # Everything recomputes to test under these fractions, so no game the pool
    # holds is one the recipe puts in the split the checkpoint read.
    split = {
        "algorithm": "sha256-threshold-v2",
        "seed": "fixture",
        "test_fraction": 1.0,
        "validation_fraction": 0.0,
    }
    earlier, earlier_manifest = write_corpus(
        tmp_path / "earlier",
        [
            normalized_row(1, split="train", plies=8),
            normalized_row(2, split="test", plies=8),
        ],
    )
    _declare_split(earlier_manifest, split)
    later, later_manifest = write_corpus(
        tmp_path / "later",
        [
            normalized_row(1, split="train", plies=8),
            normalized_row(2, split="test", plies=8),
            normalized_row(3, split="test", plies=10),
            normalized_row(4, split="train", plies=6),
        ],
    )
    _declare_split(later_manifest, split)
    pool = _freeze(tmp_path, later, later_manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=earlier, manifest=earlier_manifest
    )

    result = _evaluate(_config(pool, checkpoint))

    assert result.leakage.same_source_corpus is False
    assert result.leakage.split_recipe_matches is True
    assert result.leakage.verified is True
    assert result.leakage.recipe_recomputed is True
    assert result.leakage.overlapping_games == 0


def test_a_different_corpus_records_the_check_as_unverified(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    """Splits of unrelated corpora say nothing, so the reading says so too.

    The reading still happens. What it must not do is carry an assurance it
    never established, so the outcome is recorded as unverified with the reason
    on it rather than refused or quietly passed.
    """

    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [
            normalized_row(1, split="train", plies=8),
            normalized_row(2, split="test", plies=8),
        ],
    )
    pool = _freeze(tmp_path, normalized, manifest)
    separate, separate_manifest = write_corpus(
        tmp_path / "separate",
        [
            normalized_row(21, split="train", plies=4, result="0-1"),
            normalized_row(22, split="validation", plies=8),
        ],
        source_id="separate",
    )
    checkpoint = training_run(
        tmp_path / "run", normalized=separate, manifest=separate_manifest
    )

    result = _evaluate(_config(pool, checkpoint))

    assert result.leakage.verified is False
    assert result.leakage.algorithm == "unverified-v1"
    assert result.leakage.same_source_corpus is False
    assert result.leakage.unverified_reason is not None
    assert "different normalized corpus" in result.leakage.unverified_reason
    assert result.leakage.as_record()["verified"] is False


def test_the_same_corpus_verifies_without_reading_any_of_it(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disjointness is settled by the splits, not by the games.

    The corpus this project trains on holds nearly two billion games in its
    training split, so a check that reads them cannot finish on the host it
    protects. Refusing the read outright is what keeps that true.
    """

    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )

    def unreadable(*args: object, **kwargs: object) -> list[dict[str, Any]]:
        raise AssertionError("the leakage check read the training corpus")

    monkeypatch.setattr(leakage_module, "normalized_shard_paths", unreadable)
    result = _evaluate(_config(pool, checkpoint))

    assert result.leakage.verified is True
    assert result.leakage.algorithm == "split-disjoint-v1"
    assert result.leakage.pool_split == "test"
    assert result.leakage.training_split == "train"
    assert result.leakage.overlapping_games == 0


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
    checkpoint = training_run(
        tmp_path / "run",
        normalized=normalized,
        manifest=manifest,
        split="test",
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
    openings: dict[str, Any] | None = None,
    detail: dict[str, Any] | None = None,
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
                "noise": noise or {"resamples": 100},
                "openings": openings or {},
                "detail": detail or {},
            }
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )


def _dependency_config(
    pool: Path,
    checkpoint: Path,
    *,
    view: dict[str, Any] | None = None,
    noise: dict[str, Any] | None = None,
    dependency: dict[str, Any] | None = None,
) -> ResolvedConfig[DependencyBenchmarkConfig]:
    return ResolvedConfig(
        value=DependencyBenchmarkConfig.model_validate(
            {
                "pool": str(pool),
                "view": view or {"name": "rating-dependency"},
                "model": {"checkpoint_path": str(checkpoint), "device": "cpu"},
                "loader": {"batch_size": 4},
                "dependency": {
                    "minimum_slice_positions": 1,
                    "minimum_prefix_decisions": 1,
                    **(dependency or {}),
                },
                "noise": noise or {"resamples": 100},
            }
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )


def _measure_dependency(
    resolved_config: ResolvedConfig[DependencyBenchmarkConfig],
    *,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> DependencyBenchmarkResult:
    """Read the dependency benchmark the way both callers do, through the driver."""

    return cast(
        DependencyBenchmarkResult,
        run_benchmark(
            benchmark_registry()["dependency"],
            resolved_config,
            store=store,
            detail=detail,
        ),
    )


def _declare_split(manifest: Path, split: dict[str, Any]) -> None:
    """Give a fixture corpus a complete split recipe, as preparation writes one."""

    record = json.loads(manifest.read_text())
    record["split"] = split
    manifest.write_text(json.dumps(record))


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


def test_the_batch_plan_matches_the_batches_the_loader_would_build(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """The reading plans batches from lengths; the loader builds them from games.

    Both have to describe the same batches, because the forward pass is not
    reproducible across batch shapes. Planning is shared rather than copied,
    and this is what says the sharing still holds end to end.
    """

    from anthro_chess.data import SequenceDataLoader
    from anthro_chess.evaluation.checkpoint import _open_reading

    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    resolved = _config(pool, tmp_path / "unused.pt")
    reading = _open_reading(resolved.value)

    whole = reading.inputs(reading.game_ids)
    planned = [list(batch) for batch in reading.batches]
    # Column zero of each row, so a padded timestep's absent id stays out.
    built = [
        sorted(int(row[0]) for row in batch.game_ids)
        for batch in SequenceDataLoader(whole.dataset, whole.loader_config)
    ]
    assert planned == built


def test_the_batch_plan_counts_a_terminal_decision_as_a_scored_ply(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """A resignation is a decision, and the pool's ply count does not hold it.

    Planning off that column would bucket every game that ended in a terminal
    action one length short, so the plan and the loader would disagree about a
    third of a real pool.
    """

    from anthro_chess.evaluation.checkpoint import _open_reading

    rows = [normalized_row(index, split="test", plies=8) for index in (1, 2)]
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    pool = _freeze(tmp_path, normalized, manifest)
    reading = _open_reading(_config(pool, tmp_path / "unused.pt").value)

    inputs = reading.inputs(reading.game_ids)
    for game_id in reading.game_ids:
        encoded = sum(1 for key in inputs.plies if key[0] == game_id)
        assert reading.encoded_plies[game_id] == encoded


def test_the_adjudicated_decisions_stay_out_of_the_payload_unless_asked_for(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    training_run: Callable[..., Path],
) -> None:
    """One record per realized opportunity is millions over the canonical pool.

    Every reported quantity is computed from the summary beside them, so they
    are retained only where a session asked to look at the decisions.
    """

    normalized, manifest = corpus(tmp_path / "corpus")
    pool = _freeze(tmp_path, normalized, manifest)
    checkpoint = training_run(
        tmp_path / "run", normalized=normalized, manifest=manifest
    )

    default = _evaluate(_config(pool, checkpoint))
    assert default.adjudication is not None
    assert default.adjudication.positions is None
    assert default.adjudication.as_record()["positions"] is None
    assert default.adjudication.per_game_totals

    retained = _evaluate(
        _config(pool, checkpoint, detail={"per_position": True}),
    )
    assert retained.adjudication is not None
    assert retained.adjudication.positions
    assert retained.adjudication.as_record()["positions"]
    assert retained.adjudication.predicates == default.adjudication.predicates


def test_a_pool_pass_replicates_only_onto_the_devices_it_declares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spare card would otherwise halve a cost the record cannot distinguish."""

    class _Replicating:
        def __init__(self, device: torch.device) -> None:
            self.device = device

        def replicated(self, device: torch.device) -> _Replicating:
            return _Replicating(device)

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    runner = cast(Any, _Replicating(torch.device("cuda", 0)))

    assert CheckpointEvaluationConfig(pool=Path("artifacts/pool")).devices == 1
    assert len(_sessions(runner, None, None, 1)) == 1
    assert len(_sessions(runner, None, None, 2)) == 2
    assert len(_sessions(runner, None, None, "all")) == 4
    assert len(_sessions(runner, None, None, 8)) == 4
