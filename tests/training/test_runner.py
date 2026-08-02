from __future__ import annotations

import json
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore[import-untyped]
    EventAccumulator,
)

from anthro_chess.application_logging import configure_application_logging
from anthro_chess.config import load_config
from anthro_chess.data import PrepareConfig, prepare_pgn
from anthro_chess.data.schema import SCHEMA_VERSION
from anthro_chess.evaluation.results import ResultsStore
from anthro_chess.evaluation.results.budget import build_budget_report
from anthro_chess.training import (
    CHECKPOINT_VERSION,
    TrainingConfig,
    TrainingError,
    load_training_checkpoint,
    run_training,
)
from anthro_chess.training.devices import DeviceCapabilities
from anthro_chess.training.runner import _training_device
from anthro_chess.training.tensorboard import TENSORBOARD_DIRECTORY

from accelerators import requires_training_accelerator

REPOSITORY_ROOT = Path(__file__).parents[2]
SAMPLE_PGN = REPOSITORY_ROOT / "samples/lichess/standard-export-sample.pgn"
SAMPLE_DATA_CONFIG = REPOSITORY_ROOT / "configs/data/lichess-sample.toml"


def test_ordinary_runner_updates_model_and_writes_reproducible_records(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    config_path = _write_training_config(
        tmp_path,
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "run",
        validation=True,
    )

    resolved = load_config(TrainingConfig, path=config_path)
    log_output = StringIO()
    configure_application_logging(level="INFO", stream=log_output)
    result = run_training(resolved)

    assert result.steps == 2
    assert result.initial_parameter_sha256 != result.final_parameter_sha256
    assert result.checkpoint_path.name == "step-00000002.pt"
    assert result.validation is not None
    assert result.validation.position_count == 26
    metric_records = [
        json.loads(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["global_step"] for record in metric_records] == [1, 2]
    assert all(record["move_loss"] > 0.0 for record in metric_records)
    assert all(
        record["learning_rate"] == pytest.approx(0.003) for record in metric_records
    )
    assert all(record["interval_active_positions"] == 26 for record in metric_records)

    run_record = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run_record["resolved_config"] == resolved.as_record()
    assert run_record["seed"] == 23
    assert run_record["code"]["package_version"]
    assert run_record["data"]["train"]["manifest"]["schema_version"] == SCHEMA_VERSION
    assert run_record["data"]["train"]["manifest_sha256"]
    assert run_record["action_vocabulary"] == run_record["model"]["action_vocabulary"]
    assert run_record["encoding"] == run_record["model"]["encoding"]
    assert run_record["optimization"]["completed_steps"] == 2
    assert run_record["optimization"]["starting_step"] == 0
    assert run_record["optimization"]["checkpoint"] == str(
        result.checkpoint_path.resolve()
    )
    assert run_record["execution"] == {
        "backend": "cpu",
        "device": "cpu",
        "precision": "float32",
        "parameter_dtype": "float32",
        "determinism": "strict",
        "gradient_accumulation_steps": 1,
        "phase_profiling": False,
    }
    assert run_record["validation"]["position_count"] == 26
    assert "step=1 move_loss=" in log_output.getvalue()
    assert "step=1 move_loss=" not in capsys.readouterr().out


def test_resume_latest_restores_exact_training_state(tmp_path: Path) -> None:
    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    uninterrupted_config = _write_training_config(
        tmp_path / "uninterrupted-config",
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "uninterrupted",
        validation=False,
        steps=4,
        checkpoint_every_steps=2,
    )
    uninterrupted = run_training(load_config(TrainingConfig, path=uninterrupted_config))

    resumable_config_directory = tmp_path / "resumable-config"
    initial_config = _write_training_config(
        resumable_config_directory,
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "resumable",
        validation=False,
        steps=2,
        checkpoint_every_steps=2,
    )
    initial = run_training(load_config(TrainingConfig, path=initial_config))
    resumed_config = _write_training_config(
        resumable_config_directory,
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "resumable",
        validation=False,
        steps=4,
        checkpoint_every_steps=2,
        resume_from="latest",
    )
    resumed = run_training(load_config(TrainingConfig, path=resumed_config))

    assert initial.checkpoint_path.name == "step-00000002.pt"
    assert resumed.checkpoint_path.name == "step-00000004.pt"
    assert resumed.final_parameter_sha256 == uninterrupted.final_parameter_sha256
    assert resumed.initial_parameter_sha256 == initial.final_parameter_sha256
    records = [
        json.loads(line)
        for line in resumed.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["global_step"] for record in records] == [1, 2, 3, 4]
    events = EventAccumulator(str(resumed.run_path.parent / TENSORBOARD_DIRECTORY))
    events.Reload()
    assert [item.step for item in events.Scalars("training/move_loss")] == [
        1,
        2,
        3,
        4,
    ]

    checkpoint = load_training_checkpoint(resumed.checkpoint_path)
    assert checkpoint["version"] == CHECKPOINT_VERSION
    assert checkpoint["global_step"] == 4
    assert checkpoint["counters"]["processed_positions"] == 104
    assert checkpoint["optimizer_state"]["state"]
    assert checkpoint["scheduler_state"] is None
    assert checkpoint["scaler_state"] is None
    assert set(checkpoint["rng_state"]) == {"python", "torch_cpu"}
    assert checkpoint["loader_state"]["epoch"] == 3
    assert checkpoint["metadata"]["resolved_config"]["config"]["steps"] == 4
    assert checkpoint["metadata"]["code"]["git_revision"]
    assert checkpoint["metadata"]["data"]["train"]["manifest_sha256"]
    assert checkpoint["metadata"]["action_vocabulary"]["sha256"]
    assert checkpoint["metadata"]["encoding"]["schema_sha256"]
    assert checkpoint["metadata"]["execution"] == {
        "backend": "cpu",
        "determinism": "strict",
        "device": "cpu",
        "gradient_accumulation_steps": 1,
        "parameter_dtype": "float32",
        "phase_profiling": False,
        "precision": "float32",
    }

    run_record = json.loads(resumed.run_path.read_text(encoding="utf-8"))
    assert run_record["version"] == 5
    assert run_record["optimization"]["starting_step"] == 2
    assert run_record["optimization"]["processed_positions"] == 104
    assert run_record["optimization"]["resumed_from"] == str(
        initial.checkpoint_path.resolve()
    )


def test_explicit_resume_rejects_incompatible_state_identities(
    tmp_path: Path,
) -> None:
    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    initial_config = _write_training_config(
        tmp_path / "initial-config",
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "initial",
        validation=False,
    )
    initial = run_training(load_config(TrainingConfig, path=initial_config))
    incompatible_config = _write_training_config(
        tmp_path / "incompatible-config",
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "incompatible",
        validation=False,
        steps=3,
        learning_rate=0.004,
        resume_from=initial.checkpoint_path,
    )

    with pytest.raises(
        TrainingError,
        match="checkpoint training configuration is incompatible",
    ):
        run_training(load_config(TrainingConfig, path=incompatible_config))

    incompatible_model_config = _write_training_config(
        tmp_path / "incompatible-model-config",
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "incompatible-model",
        validation=False,
        steps=3,
        model_dim=18,
        resume_from=initial.checkpoint_path,
    )
    with pytest.raises(TrainingError, match="checkpoint model is incompatible"):
        run_training(load_config(TrainingConfig, path=incompatible_model_config))

    incompatible_data_config = _write_training_config(
        tmp_path / "incompatible-data-config",
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "incompatible-data",
        validation=False,
        steps=3,
        shuffle=True,
        resume_from=initial.checkpoint_path,
    )
    with pytest.raises(TrainingError, match="checkpoint data is incompatible"):
        run_training(load_config(TrainingConfig, path=incompatible_data_config))


def test_training_device_rejects_unavailable_or_strict_mps(
    tmp_path: Path,
) -> None:
    config_path = _write_training_config(
        tmp_path,
        normalized=tmp_path / "missing.parquet",
        manifest=tmp_path / "missing-manifest.json",
        output=tmp_path / "run",
        validation=False,
        device="mps",
    )
    config = load_config(TrainingConfig, path=config_path).value

    with pytest.raises(
        TrainingError,
        match="training device mps was requested but",
    ):
        _training_device(
            config,
            capabilities=DeviceCapabilities(
                mps_built=True,
                mps_available=False,
            ),
        )

    with pytest.raises(
        TrainingError,
        match="strict determinism is not supported",
    ):
        _training_device(
            config,
            capabilities=DeviceCapabilities(
                mps_built=True,
                mps_available=True,
            ),
        )


@pytest.mark.gpu
@requires_training_accelerator("mps")
def test_mps_checkpoint_cross_backend_and_original_device_resume(
    tmp_path: Path,
) -> None:
    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    cpu_config = _write_training_config(
        tmp_path / "cpu-config",
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "cpu",
        validation=True,
        steps=1,
        device="cpu",
        determinism="relaxed",
    )
    cpu_result = run_training(load_config(TrainingConfig, path=cpu_config))

    mps_config_directory = tmp_path / "mps-config"
    cross_backend_config = _write_training_config(
        mps_config_directory,
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "mps",
        validation=True,
        steps=2,
        resume_from=cpu_result.checkpoint_path,
        device="mps",
        determinism="relaxed",
    )
    cross_backend = run_training(load_config(TrainingConfig, path=cross_backend_config))

    assert cross_backend.initial_parameter_sha256 == (cpu_result.final_parameter_sha256)
    assert cross_backend.initial_parameter_sha256 != (
        cross_backend.final_parameter_sha256
    )
    assert cross_backend.validation is not None
    mps_checkpoint = load_training_checkpoint(cross_backend.checkpoint_path)
    assert all(
        value.device.type == "cpu"
        for value in mps_checkpoint["model_state"].values()
        if isinstance(value, torch.Tensor)
    )
    assert mps_checkpoint["metadata"]["execution"] == {
        "backend": "mps",
        "determinism": "relaxed",
        "device": "mps",
        "gradient_accumulation_steps": 1,
        "parameter_dtype": "float32",
        "phase_profiling": False,
        "precision": "float32",
    }
    assert set(mps_checkpoint["rng_state"]) == {
        "python",
        "torch_cpu",
        "torch_mps",
    }

    original_device_config = _write_training_config(
        mps_config_directory,
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "mps",
        validation=True,
        steps=3,
        resume_from="latest",
        device="mps",
        determinism="relaxed",
    )
    original_device = run_training(
        load_config(TrainingConfig, path=original_device_config)
    )

    assert original_device.initial_parameter_sha256 == (
        cross_backend.final_parameter_sha256
    )
    assert original_device.initial_parameter_sha256 != (
        original_device.final_parameter_sha256
    )
    assert original_device.validation is not None
    assert original_device.checkpoint_path.name == "step-00000003.pt"

    return_to_cpu_config = _write_training_config(
        tmp_path / "return-to-cpu-config",
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "return-to-cpu",
        validation=True,
        steps=4,
        resume_from=original_device.checkpoint_path,
        device="cpu",
        determinism="relaxed",
    )
    return_to_cpu = run_training(load_config(TrainingConfig, path=return_to_cpu_config))

    assert return_to_cpu.initial_parameter_sha256 == (
        original_device.final_parameter_sha256
    )
    assert return_to_cpu.initial_parameter_sha256 != (
        return_to_cpu.final_parameter_sha256
    )
    assert return_to_cpu.validation is not None


def test_runner_rejects_manifest_and_normalized_data_mismatch(
    tmp_path: Path,
) -> None:
    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    other_path = tmp_path / "other.parquet"
    other_path.write_bytes(prepared.normalized_path.read_bytes())
    config_path = _write_training_config(
        tmp_path,
        normalized=other_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "run",
        validation=False,
    )

    with pytest.raises(TrainingError, match="do not match"):
        run_training(load_config(TrainingConfig, path=config_path))


def test_two_selections_of_one_corpus_are_told_apart_by_the_run_record(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """The comparison this dial exists for: one corpus, two training runs.

    Both runs read the same manifest, so a later benchmark scores them against
    one reference. What separates them is the realized selection recorded in
    each run, which is what a reader traces a result back to.
    """

    rows = [
        normalized_row(game_id, split="train", plies=6, time_initial_ms=60_000)
        for game_id in range(1, 5)
    ]
    rows.extend(
        normalized_row(game_id, split="train", plies=6, time_initial_ms=300_000)
        for game_id in range(5, 9)
    )
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)

    broad = run_training(
        load_config(
            TrainingConfig,
            path=_write_training_config(
                tmp_path / "broad",
                normalized=normalized,
                manifest=manifest,
                output=tmp_path / "runs" / "broad",
                validation=False,
            ),
        )
    )
    narrow = run_training(
        load_config(
            TrainingConfig,
            path=_write_training_config(
                tmp_path / "narrow",
                normalized=normalized,
                manifest=manifest,
                output=tmp_path / "runs" / "narrow",
                validation=False,
                train_selection=(
                    "\n[train.loader.selection]\nminimum_time_initial_ms = 300000\n"
                ),
            ),
        )
    )

    broad_data = json.loads(broad.run_path.read_text(encoding="utf-8"))["data"]["train"]
    narrow_data = json.loads(narrow.run_path.read_text(encoding="utf-8"))["data"][
        "train"
    ]

    assert broad_data["manifest_sha256"] == narrow_data["manifest_sha256"]
    assert broad_data["selection"]["selected_games"] == 8
    assert broad_data["selection"]["excluded_games"] == {}
    assert narrow_data["selection"]["selected_games"] == 4
    assert narrow_data["selection"]["excluded_games"] == {
        "below_minimum_time_initial": 4
    }
    assert narrow_data["selection"]["spec"]["minimum_time_initial_ms"] == 300_000
    assert (
        broad_data["selection"]["game_ids_sha256"]
        != narrow_data["selection"]["game_ids_sha256"]
    )
    assert broad_data["dataset_sha256"] != narrow_data["dataset_sha256"]


def test_a_selection_cannot_reach_the_held_out_test_split(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    rows = [normalized_row(game_id, split="test", plies=6) for game_id in range(1, 5)]
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    config_path = _write_training_config(
        tmp_path,
        normalized=normalized,
        manifest=manifest,
        output=tmp_path / "run",
        validation=False,
        train_selection="\n[train.loader.selection]\nfraction = 1.0\n",
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'split = "train"', 'split = "test"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not use the held-out test split"):
        load_config(TrainingConfig, path=config_path)


def test_gradient_accumulation_uses_multiple_batches_per_optimizer_step(
    tmp_path: Path,
) -> None:
    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    config_path = _write_training_config(
        tmp_path,
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "run",
        validation=False,
        extra="gradient_accumulation_steps = 2\nprofile_phases = true\n",
    )

    result = run_training(load_config(TrainingConfig, path=config_path))

    records = [
        json.loads(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["processed_positions"] for record in records] == [52, 104]
    assert all(record["interval_active_positions"] == 52 for record in records)
    assert all(record["data_seconds"] >= 0.0 for record in records)
    assert all(record["transfer_seconds"] >= 0.0 for record in records)
    assert all(record["compute_seconds"] >= 0.0 for record in records)
    assert all(
        record["peak_sampled_allocated_memory_bytes"] is None for record in records
    )


@pytest.mark.gpu
@requires_training_accelerator("mps")
def test_real_mps_forward_backward_update_and_validation(tmp_path: Path) -> None:
    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    config_path = _write_training_config(
        tmp_path,
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "run",
        validation=True,
        device="mps",
        determinism="relaxed",
        extra="profile_phases = true\n",
    )

    result = run_training(load_config(TrainingConfig, path=config_path))

    run_record = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert result.initial_parameter_sha256 != result.final_parameter_sha256
    assert result.checkpoint_path.is_file()
    assert result.validation is not None
    assert run_record["execution"]["backend"] == "mps"
    metric = json.loads(
        result.metrics_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    assert metric["peak_sampled_allocated_memory_bytes"] > 0
    assert metric["peak_sampled_driver_memory_bytes"] > 0


def test_declared_cadences_report_a_run_before_it_finishes(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """A preview, per-step health, and their records all land mid-run."""

    rows = [normalized_row(game_id, split="train", plies=6) for game_id in range(1, 5)]
    rows.extend(
        normalized_row(game_id, split="validation", plies=6)
        for game_id in range(100, 106)
    )
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    config_path = _write_training_config(
        tmp_path,
        normalized=normalized,
        manifest=manifest,
        output=tmp_path / "run",
        validation=True,
        validation_split="validation",
        steps=4,
        extra="""
[evaluation]
position_budget_per_step = 4096

[[evaluation.cadences]]
name = "preview"
every_steps = 2
metrics = [
  "held_out.move_loss",
  "legality.mask_penalty",
  "training_health.gradient_norm",
  "training_health.update_to_weight_ratio",
]

[evaluation.cadences.view]
name = "preview-small"
maximum_games = 2
""",
    )
    store = ResultsStore(tmp_path / "results")

    result = run_training(load_config(TrainingConfig, path=config_path), store=store)

    assert [reading.global_step for reading in result.readings] == [2, 4]
    records = [
        json.loads(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    steps = [record for record in records if record["record"] == "step"]
    readings = [record for record in records if record["record"] == "evaluation"]
    assert [record["global_step"] for record in readings] == [2, 4]
    assert set(readings[0]["measurements"]) == {
        "held_out.move_loss",
        "legality.mask_penalty",
        "training_health.gradient_norm",
        "training_health.update_to_weight_ratio",
    }
    health = steps[-1]["training_health"]
    assert health["gradient_norm"] > 0.0
    assert health["gradient_norm_interval_maximum"] >= health["gradient_norm"]
    assert health["update_to_weight_ratio"] > 0.0
    assert steps[-1]["health_instrumentation_seconds"] > 0.0

    recorded = store.results()
    assert {envelope.kind for envelope in recorded} == {
        "held-out-preview",
        "training-health",
        "training-efficiency",
    }
    assert {envelope.checkpoint.label for envelope in recorded} == {
        "run-step-00000002",
        "run-step-00000004",
    }
    # Every cadence firing leaves a budget point under the same label as the
    # preview taken beside it, which is the join a budget report reads.
    efficiency = [item for item in recorded if item.kind == "training-efficiency"]
    assert {item.checkpoint.label for item in efficiency} == {
        "run-step-00000002",
        "run-step-00000004",
    }
    budget = build_budget_report(recorded)
    assert [point.checkpoint for point in budget.points] == [
        "run-step-00000002",
        "run-step-00000004",
    ]
    assert budget.points[0].processed_positions < budget.points[1].processed_positions
    assert budget.points[0].training_seconds < budget.points[1].training_seconds
    assert budget.points[0].view == "preview-small"
    # A preview reads the validation split, so it can never score a game the
    # training loop consumed.
    preview = next(item for item in recorded if item.kind == "held-out-preview")
    assert preview.data is not None
    assert preview.data.selected_games == 2

    run_record = json.loads(result.run_path.read_text(encoding="utf-8"))
    evaluation = run_record["evaluation"]
    assert evaluation["cadences"][0]["name"] == "preview"
    assert evaluation["cadences"][0]["view"]["selected_games"] == 2
    assert evaluation["cadences"][0]["positions_per_step"] > 0.0
    assert len(evaluation["readings"]) == 2
    assert evaluation["instrumentation_seconds"] > 0.0


def test_tensorboard_projects_training_health_and_evaluation_by_step(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    rows = [normalized_row(game_id, split="train", plies=6) for game_id in range(1, 5)]
    rows.extend(
        normalized_row(game_id, split="validation", plies=6)
        for game_id in range(100, 106)
    )
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    config_path = _write_training_config(
        tmp_path,
        normalized=normalized,
        manifest=manifest,
        output=tmp_path / "runs" / "tensorboard-test",
        validation=True,
        validation_split="validation",
        steps=2,
        extra="""
[evaluation]
position_budget_per_step = 4096

[[evaluation.cadences]]
name = "preview"
every_steps = 2
metrics = ["held_out.move_loss"]

[evaluation.cadences.view]
name = "preview-small"
maximum_games = 2
""",
    )

    result = run_training(load_config(TrainingConfig, path=config_path))

    event_directory = result.run_path.parent / TENSORBOARD_DIRECTORY
    event_files = tuple(event_directory.glob("events.out.tfevents.*"))
    assert len(event_files) == 1
    events = EventAccumulator(str(event_directory))
    events.Reload()
    assert set(events.Tags()["scalars"]) >= {
        "training/move_loss",
        "training/learning_rate",
        "training_health/gradient_norm",
        "training_health/gradient_norm_interval_maximum",
        "training_health/update_to_weight_ratio",
        "evaluation/held_out.move_loss",
    }
    assert [item.step for item in events.Scalars("training/move_loss")] == [1, 2]
    assert [item.step for item in events.Scalars("evaluation/held_out.move_loss")] == [
        2
    ]


def test_training_continues_when_tensorboard_writer_cannot_be_constructed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    config_path = _write_training_config(
        tmp_path,
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "run",
        validation=False,
    )

    def reject_writer(*args: object, **kwargs: object) -> None:
        raise OSError("read-only event directory")

    monkeypatch.setattr(
        "anthro_chess.training.tensorboard.SummaryWriter",
        reject_writer,
    )

    result = run_training(load_config(TrainingConfig, path=config_path))

    assert result.run_path.is_file()
    assert result.metrics_path.is_file()
    assert "TensorBoard output is unavailable" in caplog.text


def test_reported_throughput_excludes_the_time_a_cadence_spent_measuring(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """A run that measures itself must not report itself as slower for it."""

    rows = [normalized_row(game_id, split="train", plies=6) for game_id in range(1, 5)]
    rows.extend(
        normalized_row(game_id, split="validation", plies=6)
        for game_id in range(100, 106)
    )
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    config_path = _write_training_config(
        tmp_path,
        normalized=normalized,
        manifest=manifest,
        output=tmp_path / "run",
        validation=True,
        validation_split="validation",
        steps=4,
        extra="""
[efficiency]
warmup_steps = 0
synchronization_probe_every_intervals = 0

[evaluation]
position_budget_per_step = 4096

[[evaluation.cadences]]
name = "preview"
every_steps = 2
metrics = ["held_out.move_loss"]

[evaluation.cadences.view]
name = "preview-small"
maximum_games = 6
""",
    )

    result = run_training(load_config(TrainingConfig, path=config_path))

    records = [
        json.loads(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    steps = [record for record in records if record["record"] == "step"]
    readings = [record for record in records if record["record"] == "evaluation"]
    final = steps[-1]

    # The first reading has already been charged by the time the last step is
    # reported, so it is visible in the record and out of the denominator.
    assert final["evaluation_seconds"] == pytest.approx(readings[0]["seconds"])
    assert final["evaluation_seconds"] > 0.0
    assert final["elapsed_seconds"] > final["evaluation_seconds"]
    # Warmup is disabled and the probe is off, so every step is in the window
    # and the headline reduces to the whole run's training time.
    assert final["positions_per_second"] == pytest.approx(
        final["processed_positions"] / final["training_seconds"]
    )
    # Startup and checkpoint writes come out of the numerator too, so the
    # measured training time is strictly less than the run minus evaluation.
    assert final["training_seconds"] < (
        final["elapsed_seconds"] - final["evaluation_seconds"]
    )
    assert final["positions_per_second"] > (
        final["processed_positions"] / final["elapsed_seconds"]
    )
    assert result.efficiency is not None
    assert result.efficiency.evaluation_seconds > 0.0
    assert result.efficiency.startup_seconds > 0.0
    assert result.efficiency.checkpoint_seconds > 0.0


def test_a_run_without_declared_cadences_records_only_what_it_cost(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """Efficiency is not a cadence reading, so it does not need one declared.

    A run's cost cannot be measured after the fact, so it is recorded whether
    or not the run also chose to evaluate itself on the way past.
    """

    rows = [normalized_row(game_id, split="train", plies=6) for game_id in range(1, 5)]
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    config_path = _write_training_config(
        tmp_path,
        normalized=normalized,
        manifest=manifest,
        output=tmp_path / "run",
        validation=False,
    )
    store = ResultsStore(tmp_path / "results")

    result = run_training(load_config(TrainingConfig, path=config_path), store=store)

    assert result.readings == ()
    recorded = store.results()
    assert [envelope.kind for envelope in recorded] == ["training-efficiency"]
    assert recorded[0].checkpoint.label == "run-step-00000002"
    assert recorded[0].execution is not None
    assert result.efficiency is not None
    assert result.efficiency.processed_positions == 12
    # Two steps against a three-step warmup, so the run never reaches steady
    # state and reports no throughput rather than a figure taken from warmup.
    assert result.efficiency.active_positions_per_second is None


def test_declining_to_record_keeps_a_run_cost_out_of_committed_history(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    rows = [normalized_row(game_id, split="train", plies=6) for game_id in range(1, 5)]
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    config_path = _write_training_config(
        tmp_path,
        normalized=normalized,
        manifest=manifest,
        output=tmp_path / "run",
        validation=False,
        extra="""
[efficiency]
record = false
""",
    )
    store = ResultsStore(tmp_path / "results")

    result = run_training(load_config(TrainingConfig, path=config_path), store=store)

    assert store.results() == ()
    assert not (tmp_path / "results" / "records").exists()
    # Measured anyway, so the run still reports what it cost.
    assert result.efficiency is not None
    assert result.efficiency_paths == ()


def test_an_unaffordable_cadence_fails_before_training_starts(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    rows = [normalized_row(game_id, split="train", plies=6) for game_id in range(1, 5)]
    rows.extend(
        normalized_row(game_id, split="validation", plies=6)
        for game_id in range(100, 106)
    )
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    config_path = _write_training_config(
        tmp_path,
        normalized=normalized,
        manifest=manifest,
        output=tmp_path / "run",
        validation=True,
        validation_split="validation",
        extra="""
[evaluation]
position_budget_per_step = 2

[[evaluation.cadences]]
name = "preview"
every_steps = 1
metrics = ["held_out.move_loss"]

[evaluation.cadences.view]
name = "preview-small"
maximum_games = 6
""",
    )

    with pytest.raises(TrainingError, match="position\\(s\\) per optimizer step"):
        run_training(load_config(TrainingConfig, path=config_path))

    assert not (tmp_path / "run" / "run.json").exists()


def _write_training_config(
    tmp_path: Path,
    *,
    normalized: Path,
    manifest: Path,
    output: Path,
    validation: bool,
    validation_split: str = "train",
    steps: int = 2,
    learning_rate: float = 0.003,
    checkpoint_every_steps: int = 100,
    resume_from: str | Path | None = None,
    model_dim: int = 16,
    shuffle: bool = False,
    device: str = "cpu",
    precision: str = "float32",
    determinism: str = "strict",
    extra: str = "",
    train_selection: str = "",
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "training.toml"
    resume_selection = (
        f"resume_from = {json.dumps(str(resume_from))}\n"
        if resume_from is not None
        else ""
    )
    validation_selection = ""
    if validation:
        validation_selection = f"""
[validation]
normalized = {json.dumps(str(normalized))}
manifest = {json.dumps(str(manifest))}

[validation.loader]
split = {json.dumps(validation_split)}
batch_size = 1
shuffle = false
"""
    config_path.write_text(
        f"""
output_directory = {json.dumps(str(output))}
seed = 23
steps = {steps}
learning_rate = {learning_rate}
log_every_steps = 1
checkpoint_every_steps = {checkpoint_every_steps}
{resume_selection}
device = {json.dumps(device)}
precision = {json.dumps(precision)}
determinism = {json.dumps(determinism)}
{extra}

[model]
piece_embedding_dim = 2
action_embedding_dim = 4
model_dim = {model_dim}
attention_heads = 2
transformer_layers = 1
feedforward_dim = 24
dropout = 0.0

[train]
normalized = {json.dumps(str(normalized))}
manifest = {json.dumps(str(manifest))}

[train.loader]
split = "train"
batch_size = 1
shuffle = {str(shuffle).lower()}
{train_selection}{validation_selection}
""",
        encoding="utf-8",
    )
    return config_path
