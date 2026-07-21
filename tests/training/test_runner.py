from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from anthro_chess.config import load_config
from anthro_chess.data import PrepareConfig, prepare_pgn
from anthro_chess.training import (
    CHECKPOINT_VERSION,
    TrainingConfig,
    TrainingError,
    load_training_checkpoint,
    run_training,
)
from anthro_chess.training.devices import DeviceCapabilities
from anthro_chess.training.runner import _training_device

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
    assert all(record["batch_positions"] == 13 for record in metric_records)

    run_record = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run_record["resolved_config"] == resolved.as_record()
    assert run_record["seed"] == 23
    assert run_record["code"]["package_version"]
    assert run_record["data"]["train"]["manifest"]["schema_version"] == 1
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
    assert "step=1 move_loss=" in capsys.readouterr().out


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

    checkpoint = load_training_checkpoint(resumed.checkpoint_path)
    assert checkpoint["version"] == CHECKPOINT_VERSION
    assert checkpoint["global_step"] == 4
    assert checkpoint["counters"]["processed_positions"] == 52
    assert checkpoint["optimizer_state"]["state"]
    assert checkpoint["scheduler_state"] is None
    assert checkpoint["scaler_state"] is None
    assert set(checkpoint["rng_state"]) == {"python", "torch_cpu"}
    assert checkpoint["loader_state"]["epoch"] == 1
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
    assert run_record["version"] == 3
    assert run_record["optimization"]["starting_step"] == 2
    assert run_record["optimization"]["processed_positions"] == 52
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
@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS smoke verification requires Apple silicon",
)
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
    assert [record["processed_positions"] for record in records] == [26, 52]
    assert all(record["batch_positions"] == 26 for record in records)
    assert all(record["data_seconds"] >= 0.0 for record in records)
    assert all(record["transfer_seconds"] >= 0.0 for record in records)
    assert all(record["compute_seconds"] >= 0.0 for record in records)
    assert all(
        record["peak_sampled_allocated_memory_bytes"] is None for record in records
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an available PyTorch MPS backend",
)
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


def _write_training_config(
    tmp_path: Path,
    *,
    normalized: Path,
    manifest: Path,
    output: Path,
    validation: bool,
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
split = "train"
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
{validation_selection}
""",
        encoding="utf-8",
    )
    return config_path
