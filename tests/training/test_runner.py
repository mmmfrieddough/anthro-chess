from __future__ import annotations

import json
from pathlib import Path

import pytest

from anthro_chess.config import load_config
from anthro_chess.data import PrepareConfig, prepare_pgn
from anthro_chess.training import (
    TrainingConfig,
    TrainingError,
    run_training,
)

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
    assert all(record["batch_positions"] == 26 for record in metric_records)

    run_record = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run_record["resolved_config"] == resolved.as_record()
    assert run_record["seed"] == 23
    assert run_record["code"]["package_version"]
    assert run_record["data"]["train"]["manifest"]["schema_version"] == 1
    assert run_record["data"]["train"]["manifest_sha256"]
    assert run_record["action_vocabulary"] == run_record["model"]["action_vocabulary"]
    assert run_record["encoding"] == run_record["model"]["encoding"]
    assert run_record["optimization"]["completed_steps"] == 2
    assert run_record["validation"]["position_count"] == 26
    assert "step=1 move_loss=" in capsys.readouterr().out


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


def _write_training_config(
    tmp_path: Path,
    *,
    normalized: Path,
    manifest: Path,
    output: Path,
    validation: bool,
) -> Path:
    config_path = tmp_path / "training.toml"
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
steps = 2
learning_rate = 0.003
log_every_steps = 1

[model]
piece_embedding_dim = 2
action_embedding_dim = 4
model_dim = 16
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
shuffle = false
{validation_selection}
""",
        encoding="utf-8",
    )
    return config_path
