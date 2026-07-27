from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from anthro_chess.data import SequenceDataConfig, SequenceLoaderConfig
from anthro_chess.data.schema import SplitName
from anthro_chess.evaluation.results import (
    CheckpointReference,
    ResultsStore,
    series_fingerprint,
)
from anthro_chess.models import CausalMoveModel, MoveModelConfig
from anthro_chess.training.cadence import (
    HEALTH_KIND,
    PREVIEW_KIND,
    CadenceConfig,
    CadenceError,
    PreviewViewConfig,
    TrainingEvaluationConfig,
    prepare_schedule,
    score_preview,
)
from anthro_chess.training.health import StepHealth

RowFactory = Callable[..., dict[str, Any]]
CorpusFactory = Callable[..., tuple[Path, Path]]

HELD_OUT_METRICS = ("held_out.move_loss", "legality.mask_penalty")


def _selection(
    normalized: Path,
    manifest: Path,
    split: SplitName,
) -> SequenceDataConfig:
    return SequenceDataConfig(
        normalized=normalized,
        manifest=manifest,
        loader=SequenceLoaderConfig(split=split, batch_size=2, shuffle=False),
    )


def _corpus(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
    *,
    validation_games: int = 6,
) -> SequenceDataConfig:
    rows = [normalized_row(game_id, split="train", plies=6) for game_id in range(1, 4)]
    rows.extend(
        normalized_row(game_id, split="validation", plies=6)
        for game_id in range(100, 100 + validation_games)
    )
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    return _selection(normalized, manifest, "validation")


def _model() -> CausalMoveModel:
    torch.manual_seed(5)
    return CausalMoveModel(
        MoveModelConfig(
            piece_embedding_dim=2,
            action_embedding_dim=4,
            model_dim=16,
            attention_heads=2,
            transformer_layers=1,
            feedforward_dim=24,
            dropout=0.0,
        )
    )


def _evaluation(
    *,
    metrics: tuple[str, ...] = HELD_OUT_METRICS,
    every_steps: int = 2,
    maximum_games: int | None = 2,
    budget: int = 4096,
) -> TrainingEvaluationConfig:
    return TrainingEvaluationConfig(
        position_budget_per_step=budget,
        cadences=(
            CadenceConfig(
                name="preview",
                every_steps=every_steps,
                metrics=metrics,
                view=(
                    None
                    if maximum_games is None
                    else PreviewViewConfig(
                        name="preview-small",
                        maximum_games=maximum_games,
                    )
                ),
            ),
        ),
    )


def test_a_preview_view_subsamples_the_validation_split_deterministically(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    validation = _corpus(tmp_path, normalized_row, write_corpus)

    first = prepare_schedule(_evaluation(), validation).entries[0]
    second = prepare_schedule(_evaluation(), validation).entries[0]

    assert first.selection is not None
    assert first.selection.selected_games == 2
    assert first.selection.eligible_games == 6
    # A preview subsamples and never filters, so nothing is excluded.
    assert first.selection.excluded_games == {}
    assert second.selection is not None
    assert second.selection.game_ids == first.selection.game_ids
    # Every selected game belongs to the validation split.
    assert set(first.selection.game_ids) <= set(range(100, 106))


def test_a_preview_reading_cannot_share_a_series_with_a_wider_reading(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    validation = _corpus(tmp_path, normalized_row, write_corpus)

    small = prepare_schedule(_evaluation(maximum_games=2), validation).entries[0]
    wide = prepare_schedule(_evaluation(maximum_games=6), validation).entries[0]

    assert small.component is not None
    assert wide.component is not None
    assert series_fingerprint("held_out.move_loss", small.component) != (
        series_fingerprint("held_out.move_loss", wide.component)
    )


def test_an_unaffordable_pairing_fails_before_the_first_step(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    validation = _corpus(tmp_path, normalized_row, write_corpus)

    with pytest.raises(CadenceError, match="position\\(s\\) per optimizer step"):
        prepare_schedule(
            _evaluation(maximum_games=6, every_steps=1, budget=4),
            validation,
        )


def test_cost_is_amortized_over_the_declared_interval(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    """Widening the interval is what makes the same view affordable."""

    validation = _corpus(tmp_path, normalized_row, write_corpus)

    frequent = prepare_schedule(_evaluation(every_steps=2), validation).entries[0]
    occasional = prepare_schedule(_evaluation(every_steps=8), validation).entries[0]

    assert frequent.view_passes == 1
    assert frequent.inputs is not None
    assert frequent.positions_per_step == pytest.approx(
        frequent.inputs.position_count / 2
    )
    assert occasional.positions_per_step == pytest.approx(
        frequent.positions_per_step / 4
    )


def test_a_metric_no_in_training_measurement_computes_is_rejected(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    validation = _corpus(tmp_path, normalized_row, write_corpus)

    with pytest.raises(CadenceError, match="no in-training measurement computes"):
        prepare_schedule(
            _evaluation(metrics=("dependency.rating_shuffled_degradation",)),
            validation,
        )


def test_an_unknown_metric_is_rejected_by_name(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    validation = _corpus(tmp_path, normalized_row, write_corpus)

    with pytest.raises(CadenceError, match="unknown metric: held_out.invented"):
        prepare_schedule(_evaluation(metrics=("held_out.invented",)), validation)


def test_a_data_metric_without_a_view_is_rejected(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    validation = _corpus(tmp_path, normalized_row, write_corpus)

    with pytest.raises(CadenceError, match="must declare a view"):
        prepare_schedule(_evaluation(maximum_games=None), validation)


def test_a_preview_needs_a_validation_selection() -> None:
    with pytest.raises(CadenceError, match="needs a validation data selection"):
        prepare_schedule(_evaluation(), None)


def test_a_preview_never_reads_the_held_out_test_split(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    rows = [normalized_row(game_id, split="test", plies=6) for game_id in range(1, 4)]
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)

    with pytest.raises(CadenceError, match="never read the held-out test split"):
        prepare_schedule(
            _evaluation(),
            _selection(normalized, manifest, "test"),
        )


def test_health_only_cadences_need_no_view_and_cost_nothing(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    validation = _corpus(tmp_path, normalized_row, write_corpus)

    entry = prepare_schedule(
        _evaluation(
            metrics=("training_health.gradient_norm",),
            maximum_games=None,
            every_steps=1,
            budget=1,
        ),
        validation,
    ).entries[0]

    assert entry.slice_metrics == ()
    assert entry.health_metrics == ("training_health.gradient_norm",)
    assert entry.inputs is None
    assert entry.positions_per_step == 0.0


def test_two_cadences_over_one_view_share_its_materialization(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    validation = _corpus(tmp_path, normalized_row, write_corpus)
    view = PreviewViewConfig(name="preview-small", maximum_games=2)
    config = TrainingEvaluationConfig(
        cadences=(
            CadenceConfig(
                name="frequent",
                every_steps=4,
                metrics=("held_out.move_loss",),
                view=view,
            ),
            CadenceConfig(
                name="occasional",
                every_steps=8,
                metrics=("legality.mask_penalty",),
                view=view,
            ),
        )
    )

    entries = prepare_schedule(config, validation).entries

    assert entries[0].inputs is entries[1].inputs


def test_a_duplicate_cadence_name_is_rejected() -> None:
    config = TrainingEvaluationConfig(
        cadences=(
            CadenceConfig(
                name="preview",
                every_steps=2,
                metrics=("training_health.gradient_norm",),
            ),
            CadenceConfig(
                name="preview",
                every_steps=4,
                metrics=("training_health.gradient_norm",),
            ),
        )
    )

    with pytest.raises(CadenceError, match="distinct name"):
        prepare_schedule(config, None)


def test_entries_run_only_on_their_declared_steps(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    validation = _corpus(tmp_path, normalized_row, write_corpus)
    schedule = prepare_schedule(_evaluation(every_steps=3), validation)

    assert schedule.due(1) == ()
    assert [entry.config.name for entry in schedule.due(3)] == ["preview"]
    assert [entry.config.name for entry in schedule.due(6)] == ["preview"]


def test_a_run_without_cadences_resolves_to_an_empty_schedule() -> None:
    assert not prepare_schedule(TrainingEvaluationConfig(), None)


def test_a_firing_records_previews_and_health_as_separate_results(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    validation = _corpus(tmp_path, normalized_row, write_corpus)
    store = ResultsStore(tmp_path / "results")
    schedule = prepare_schedule(
        _evaluation(metrics=(*HELD_OUT_METRICS, "training_health.gradient_norm")),
        validation,
        store=store,
    )
    entry = schedule.entries[0]
    reading = schedule.run(
        entry,
        _model(),
        device=torch.device("cpu"),
        global_step=2,
        checkpoint=CheckpointReference(label="run-step-00000002", step=2),
        health=StepHealth(
            steps=2,
            global_step=2,
            gradient_norm=1.25,
            gradient_norm_interval_maximum=1.5,
            update_to_weight_ratio=None,
        ),
    )

    kinds = {envelope.kind for envelope in reading.envelopes}
    assert kinds == {PREVIEW_KIND, HEALTH_KIND}
    preview = next(item for item in reading.envelopes if item.kind == PREVIEW_KIND)
    health = next(item for item in reading.envelopes if item.kind == HEALTH_KIND)
    assert {item.metric for item in preview.measurements} == set(HELD_OUT_METRICS)
    assert preview.data is not None
    assert preview.data.view == "preview-small"
    # A training-batch statistic never carries evaluation inputs at all.
    assert health.data is None
    assert len(reading.recorded_paths) == 2
    assert len(store.results()) == 2
    assert reading.as_record()["record"] == "evaluation"
    assert reading.as_record()["cadence"] == "preview"


def test_a_firing_records_nothing_without_a_store(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    validation = _corpus(tmp_path, normalized_row, write_corpus)
    schedule = prepare_schedule(_evaluation(), validation)
    reading = schedule.run(
        schedule.entries[0],
        _model(),
        device=torch.device("cpu"),
        global_step=2,
        checkpoint=CheckpointReference(label="run-step-00000002", step=2),
        health=None,
    )

    assert reading.recorded_paths == ()
    assert reading.envelopes


def test_scoring_a_preview_is_deterministic_and_restores_training_mode(
    tmp_path: Path,
    normalized_row: RowFactory,
    write_corpus: CorpusFactory,
) -> None:
    validation = _corpus(tmp_path, normalized_row, write_corpus)
    entry = prepare_schedule(_evaluation(), validation).entries[0]
    assert entry.inputs is not None
    model = _model()
    model.train()

    first = score_preview(model, entry.inputs, device=torch.device("cpu"))
    second = score_preview(model, entry.inputs, device=torch.device("cpu"))

    assert model.training
    assert first.overall.move_loss == pytest.approx(second.overall.move_loss)
    assert first.overall.position_count == entry.inputs.position_count
