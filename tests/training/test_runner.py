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
from anthro_chess.data import PrepareConfig, SequenceDataLoader, prepare_pgn
from anthro_chess.data.schema import SCHEMA_VERSION
from anthro_chess.evaluation.results import ResultsStore
from anthro_chess.evaluation.results.budget import build_budget_report
from anthro_chess.models import CausalMoveModel, MoveModelBatch
from anthro_chess.training import (
    CHECKPOINT_VERSION,
    RUN_ARTIFACT_VERSION,
    TrainingConfig,
    TrainingError,
    load_training_checkpoint,
    run_training,
)
from anthro_chess.training.devices import (
    STRICT_DETERMINISM_BACKENDS,
    DeviceCapabilities,
)
from anthro_chess.training.runner import (
    _EXECUTION_COMPATIBILITY_KEYS,
    _EXECUTION_PROVENANCE_KEYS,
    _execution_record,
    _training_device,
)
from anthro_chess.training.tensorboard import TENSORBOARD_DIRECTORY

from accelerators import (
    ACCELERATOR_RNG_KEYS,
    training_accelerator_parameters,
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
        "fused_optimizer": False,
        "matmul_precision": "highest",
        "phase_profiling": False,
    }
    # Which machine, kept apart from what was selected on it. Nothing compares
    # these across runs, which is exactly why they can name the host.
    assert run_record["hardware"]["backend"] == "cpu"
    assert run_record["hardware"]["torch_version"]
    assert run_record["hardware"]["cpu"]["torch_threads"] >= 1
    assert "cuda" not in run_record["hardware"]
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
    # Sorted, because the property under test is that resuming leaves each step
    # projected exactly once — neither purged away nor written twice. The order
    # the accumulator returns them in is not that property: it reads a
    # directory's event files in plain filename order, and their names end in a
    # process-global counter that is not zero padded, so a run whose two writers
    # straddle `...9` and `...10` is read second file first. Which side of that
    # boundary a test lands on depends on how many writers the process built
    # before it, which under sharding depends on how the suite was distributed.
    assert sorted(item.step for item in events.Scalars("training/move_loss")) == [
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
        "fused_optimizer": False,
        "gradient_accumulation_steps": 1,
        "matmul_precision": "highest",
        "parameter_dtype": "float32",
        "phase_profiling": False,
        "precision": "float32",
    }

    run_record = json.loads(resumed.run_path.read_text(encoding="utf-8"))
    assert run_record["version"] == RUN_ARTIFACT_VERSION
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


def test_every_recorded_execution_setting_has_exactly_one_declared_role(
    tmp_path: Path,
) -> None:
    config = load_config(
        TrainingConfig,
        path=_write_training_config(
            tmp_path,
            normalized=tmp_path / "missing.parquet",
            manifest=tmp_path / "missing-manifest.json",
            output=tmp_path / "run",
            validation=False,
        ),
    ).value

    record = _execution_record(config, torch.device("cpu"))

    compatibility = set(_EXECUTION_COMPATIBILITY_KEYS)
    provenance = set(_EXECUTION_PROVENANCE_KEYS)
    assert compatibility.isdisjoint(provenance)
    assert set(record) == compatibility | provenance


def test_matmul_precision_is_applied_for_the_run_and_restored_after_it(
    tmp_path: Path,
) -> None:
    """The dial is a process-wide Torch setting, so the run has to hand it back.

    Leaving it set would make every later benchmark in the same process inherit
    a training run's arithmetic, which is the kind of cross-contamination no
    reading would report.
    """

    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    torch.set_float32_matmul_precision("highest")
    config_path = _write_training_config(
        tmp_path / "config",
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "run",
        validation=False,
        extra='matmul_precision = "high"\n',
    )

    result = run_training(load_config(TrainingConfig, path=config_path))

    assert torch.get_float32_matmul_precision() == "highest"
    checkpoint = load_training_checkpoint(result.checkpoint_path)
    assert checkpoint["metadata"]["execution"]["matmul_precision"] == "high"


def test_resume_refuses_a_changed_matmul_precision(tmp_path: Path) -> None:
    """A declared arithmetic change is one a continuation has to match.

    Unlike the fused optimizer, which arrives with the backend and describes
    where a run happened, this is chosen — and what it chooses is the precision
    every gradient is computed in. A run that changed it partway would have no
    way to say which half produced its weights.
    """

    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    initial = run_training(
        load_config(
            TrainingConfig,
            path=_write_training_config(
                tmp_path / "initial-config",
                normalized=prepared.normalized_path,
                manifest=prepared.manifest_path,
                output=tmp_path / "initial",
                validation=False,
            ),
        )
    )
    continuation = _write_training_config(
        tmp_path / "continuation-config",
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "continuation",
        validation=False,
        steps=3,
        resume_from=initial.checkpoint_path,
        extra='matmul_precision = "high"\n',
    )

    with pytest.raises(TrainingError, match="incompatible"):
        run_training(load_config(TrainingConfig, path=continuation))


def test_resume_reads_execution_provenance_it_does_not_recognize(
    tmp_path: Path,
) -> None:
    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    initial = run_training(
        load_config(
            TrainingConfig,
            path=_write_training_config(
                tmp_path / "initial-config",
                normalized=prepared.normalized_path,
                manifest=prepared.manifest_path,
                output=tmp_path / "initial",
                validation=False,
            ),
        )
    )
    # A checkpoint from a version that recorded distributed provenance this one
    # knows nothing about, and had not yet recorded one this one writes.
    foreign = _rewrite_checkpoint_execution(
        initial.checkpoint_path,
        tmp_path / "foreign.pt",
        add={"world_size": 4},
        drop=("fused_optimizer",),
    )
    continuation = _write_training_config(
        tmp_path / "continuation-config",
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "continuation",
        validation=False,
        steps=3,
        resume_from=foreign,
    )

    resumed = run_training(load_config(TrainingConfig, path=continuation))

    assert resumed.initial_parameter_sha256 == initial.final_parameter_sha256
    assert resumed.final_parameter_sha256 != initial.final_parameter_sha256


def test_resume_rejects_a_changed_or_absent_execution_identity(
    tmp_path: Path,
) -> None:
    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    initial = run_training(
        load_config(
            TrainingConfig,
            path=_write_training_config(
                tmp_path / "initial-config",
                normalized=prepared.normalized_path,
                manifest=prepared.manifest_path,
                output=tmp_path / "initial",
                validation=False,
            ),
        )
    )
    changed = _rewrite_checkpoint_execution(
        initial.checkpoint_path,
        tmp_path / "changed.pt",
        add={"parameter_dtype": "bfloat16"},
    )
    absent = _rewrite_checkpoint_execution(
        initial.checkpoint_path,
        tmp_path / "absent.pt",
        drop=("determinism",),
    )

    with pytest.raises(
        TrainingError,
        match=(
            "checkpoint execution parameter_dtype is incompatible with the "
            "current run: checkpoint 'bfloat16', current 'float32'"
        ),
    ):
        run_training(
            load_config(
                TrainingConfig,
                path=_write_training_config(
                    tmp_path / "changed-config",
                    normalized=prepared.normalized_path,
                    manifest=prepared.manifest_path,
                    output=tmp_path / "changed",
                    validation=False,
                    steps=3,
                    resume_from=changed,
                ),
            )
        )

    with pytest.raises(
        TrainingError,
        match="checkpoint execution metadata has no determinism",
    ):
        run_training(
            load_config(
                TrainingConfig,
                path=_write_training_config(
                    tmp_path / "absent-config",
                    normalized=prepared.normalized_path,
                    manifest=prepared.manifest_path,
                    output=tmp_path / "absent",
                    validation=False,
                    steps=3,
                    resume_from=absent,
                ),
            )
        )


def _rewrite_checkpoint_execution(
    source: Path,
    destination: Path,
    *,
    add: dict[str, object] | None = None,
    drop: tuple[str, ...] = (),
) -> Path:
    payload = load_training_checkpoint(source)
    execution = {
        key: value
        for key, value in payload["metadata"]["execution"].items()
        if key not in drop
    }
    execution.update(add or {})
    payload["metadata"] = {**payload["metadata"], "execution": execution}
    torch.save(payload, destination)
    return destination


_SHARD_BACKED = """
[train.streaming]
planning_window_examples = 8
workers = 0
prefetch_batches = 2
"""


def test_shard_backed_training_runs_and_records_which_loader_read_the_corpus(
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
        validation=True,
        train_streaming=_SHARD_BACKED,
    )

    result = run_training(load_config(TrainingConfig, path=config_path))

    assert result.initial_parameter_sha256 != result.final_parameter_sha256
    run_record = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run_record["data"]["train"]["loader"] == "shard-backed"
    # The eager loader still reads the validation selection, because only the
    # train selection declared a streaming section.
    assert run_record["data"]["validation"]["loader"] == "eager"
    assert run_record["data"]["train"]["selection"]["selected_games"] > 0
    # Absolute resolved paths, so a run record names the corpus it read rather
    # than a configured path whose meaning depends on where it was read from.
    assert run_record["data"]["train"]["normalized_paths"] == [
        str(prepared.normalized_path.resolve())
    ]


def test_shard_backed_training_resumes_from_its_own_checkpoint(tmp_path: Path) -> None:
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
        train_streaming=_SHARD_BACKED,
    )
    uninterrupted = run_training(load_config(TrainingConfig, path=uninterrupted_config))

    resumable = tmp_path / "resumable-config"
    initial = run_training(
        load_config(
            TrainingConfig,
            path=_write_training_config(
                resumable,
                normalized=prepared.normalized_path,
                manifest=prepared.manifest_path,
                output=tmp_path / "resumable",
                validation=False,
                steps=2,
                checkpoint_every_steps=2,
                train_streaming=_SHARD_BACKED,
            ),
        )
    )
    resumed = run_training(
        load_config(
            TrainingConfig,
            path=_write_training_config(
                resumable,
                normalized=prepared.normalized_path,
                manifest=prepared.manifest_path,
                output=tmp_path / "resumable",
                validation=False,
                steps=4,
                checkpoint_every_steps=2,
                resume_from="latest",
                train_streaming=_SHARD_BACKED,
            ),
        )
    )

    assert resumed.final_parameter_sha256 == uninterrupted.final_parameter_sha256
    assert resumed.initial_parameter_sha256 == initial.final_parameter_sha256
    checkpoint = load_training_checkpoint(resumed.checkpoint_path)
    assert checkpoint["loader_state"]["position"] >= 0
    assert (
        checkpoint["metadata"]["data"]["train"]["dataset_sha256"]
        == json.loads(resumed.run_path.read_text(encoding="utf-8"))["data"]["train"][
            "dataset_sha256"
        ]
    )


def test_a_run_cannot_change_loader_and_continue_the_same_checkpoint(
    tmp_path: Path,
) -> None:
    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    initial = run_training(
        load_config(
            TrainingConfig,
            path=_write_training_config(
                tmp_path / "initial-config",
                normalized=prepared.normalized_path,
                manifest=prepared.manifest_path,
                output=tmp_path / "initial",
                validation=False,
                train_streaming=_SHARD_BACKED,
            ),
        )
    )
    eager_continuation = _write_training_config(
        tmp_path / "eager-config",
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "eager",
        validation=False,
        steps=3,
        resume_from=initial.checkpoint_path,
    )

    with pytest.raises(TrainingError, match="checkpoint data is incompatible"):
        run_training(load_config(TrainingConfig, path=eager_continuation))


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
                cuda_built=False,
                cuda_available=False,
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
                cuda_built=False,
                cuda_available=False,
            ),
        )


def test_mixed_precision_keeps_full_precision_parameters_and_no_scaler(
    tmp_path: Path,
) -> None:
    """Autocast changes how the forward pass computes, not what is stored.

    The checkpoint is the thing worth pinning: master weights stay float32 and
    the gradient-scaler slot stays empty, which is what keeps a mixed-precision
    checkpoint loadable everywhere a full-precision one is.
    """

    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    config_path = _write_training_config(
        tmp_path / "config",
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "run",
        validation=True,
        precision="bfloat16-mixed",
        determinism="relaxed",
    )

    result = run_training(load_config(TrainingConfig, path=config_path))

    assert result.initial_parameter_sha256 != result.final_parameter_sha256
    assert result.validation is not None
    checkpoint = load_training_checkpoint(result.checkpoint_path)
    assert checkpoint["scaler_state"] is None
    assert all(
        value.dtype == torch.float32
        for value in checkpoint["model_state"].values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )
    execution = checkpoint["metadata"]["execution"]
    assert execution["precision"] == "bfloat16-mixed"
    assert execution["parameter_dtype"] == "float32"


@pytest.mark.gpu
@pytest.mark.parametrize("backend", training_accelerator_parameters())
def test_accelerator_checkpoint_cross_backend_and_original_device_resume(
    backend: str,
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

    accelerator_config_directory = tmp_path / f"{backend}-config"
    cross_backend_config = _write_training_config(
        accelerator_config_directory,
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / backend,
        validation=True,
        steps=2,
        resume_from=cpu_result.checkpoint_path,
        device=backend,
        determinism="relaxed",
    )
    cross_backend = run_training(load_config(TrainingConfig, path=cross_backend_config))

    assert cross_backend.initial_parameter_sha256 == (cpu_result.final_parameter_sha256)
    assert cross_backend.initial_parameter_sha256 != (
        cross_backend.final_parameter_sha256
    )
    assert cross_backend.validation is not None
    accelerator_checkpoint = load_training_checkpoint(cross_backend.checkpoint_path)
    assert all(
        value.device.type == "cpu"
        for value in accelerator_checkpoint["model_state"].values()
        if isinstance(value, torch.Tensor)
    )
    assert accelerator_checkpoint["metadata"]["execution"] == {
        "backend": backend,
        "determinism": "relaxed",
        "device": backend,
        "fused_optimizer": backend == "cuda",
        "gradient_accumulation_steps": 1,
        "matmul_precision": "highest",
        "parameter_dtype": "float32",
        "phase_profiling": False,
        "precision": "float32",
    }
    assert set(accelerator_checkpoint["rng_state"]) == {
        "python",
        "torch_cpu",
        ACCELERATOR_RNG_KEYS[backend],
    }

    original_device_config = _write_training_config(
        accelerator_config_directory,
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / backend,
        validation=True,
        steps=3,
        resume_from="latest",
        device=backend,
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


def test_a_corpus_past_the_declared_context_refuses_before_the_first_step(
    tmp_path: Path,
) -> None:
    """The mid-run death this replaces is reachable from the same corpus."""

    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    output = tmp_path / "run"
    config_path = _write_training_config(
        tmp_path,
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=output,
        validation=False,
        maximum_context_plies=8,
    )
    resolved = load_config(TrainingConfig, path=config_path)

    with pytest.raises(TrainingError) as refusal:
        run_training(resolved)

    assert "longest game of 26 plies" in str(refusal.value)
    assert "reaches ply index 26" in str(refusal.value)
    assert "8 plies the model declares as its context" in str(refusal.value)
    # Nothing ran: the run directory is created after the selections load.
    assert not output.exists()

    # Without the refusal this corpus reaches the model from inside the
    # micro-batch loop, which handles only an exhausted loader.
    loader = SequenceDataLoader.from_parquet(
        prepared.normalized_path,
        resolved.value.train.loader,
        legal_actions=False,
    )
    batch = MoveModelBatch.from_sequence_batch(next(loader))
    with pytest.raises(ValueError, match="past the 8 plies"):
        CausalMoveModel(resolved.value.model)(batch)


def test_the_declared_context_is_compared_against_chunked_reach(
    tmp_path: Path,
) -> None:
    """A chunk keeps its game's ply indices, so it reaches past its own width."""

    # The 26-ply game encodes to at most 27 plies, so it starts its last chunk
    # at ply 24 and no chunk of it is wider than 8: its batches reach ply index
    # 31 rather than 26.
    chunked = "chunk_length = 8\n"
    with pytest.raises(TrainingError, match="chunked at 8, and reaches ply index 31"):
        _train_at_declared_context(tmp_path, 30, train_selection=chunked)
    _train_at_declared_context(tmp_path, 32, train_selection=chunked)


def test_the_declared_context_leaves_room_for_an_appended_terminal_action(
    tmp_path: Path,
) -> None:
    """A manifest counts moves; the encoding may append one action past them."""

    with pytest.raises(TrainingError, match="encodes to at most 27"):
        _train_at_declared_context(tmp_path, 26)
    _train_at_declared_context(tmp_path, 27)


@pytest.mark.parametrize("recorded", [None, 0, "26"])
def test_a_manifest_without_a_usable_longest_game_is_refused_rather_than_assumed(
    tmp_path: Path,
    recorded: object,
) -> None:
    """An unreadable length must not resolve to a bound that admits anything."""

    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    if recorded is None:
        del manifest["games"]["plies"]
    else:
        manifest["games"]["plies"]["maximum_per_game"] = recorded
    prepared.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = _write_training_config(
        tmp_path,
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / "run",
        validation=False,
    )

    with pytest.raises(TrainingError, match="records no positive longest game"):
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
@pytest.mark.parametrize("backend", training_accelerator_parameters())
def test_strict_determinism_on_an_accelerator_reproduces_the_same_parameters(
    backend: str,
    tmp_path: Path,
) -> None:
    """The correctness path has to be the real one, on the real device.

    An accelerator that accepts the strict selection is claiming its whole
    backward pass has deterministic kernels. Nothing about a single run would
    reveal a claim that is false, so this runs the same configuration twice and
    compares the weights it arrived at. A backend that cannot make the claim
    must refuse the selection instead of quietly training anyway.
    """

    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )

    def train(name: str) -> str:
        config_path = _write_training_config(
            tmp_path / f"{name}-config",
            normalized=prepared.normalized_path,
            manifest=prepared.manifest_path,
            output=tmp_path / name,
            validation=False,
            steps=3,
            device=backend,
            determinism="strict",
        )
        result = run_training(load_config(TrainingConfig, path=config_path))
        return str(result.final_parameter_sha256)

    if backend not in STRICT_DETERMINISM_BACKENDS:
        with pytest.raises(TrainingError, match="strict determinism is not supported"):
            train("refused")
        return

    assert train("first") == train("second")


@pytest.mark.gpu
@pytest.mark.parametrize("backend", training_accelerator_parameters())
def test_real_accelerator_forward_backward_update_and_validation(
    backend: str,
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
        validation=True,
        device=backend,
        determinism="relaxed",
        extra="profile_phases = true\n",
    )

    result = run_training(load_config(TrainingConfig, path=config_path))

    run_record = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert result.initial_parameter_sha256 != result.final_parameter_sha256
    assert result.checkpoint_path.is_file()
    assert result.validation is not None
    assert run_record["execution"]["backend"] == backend
    metric = json.loads(
        result.metrics_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    assert metric["peak_sampled_allocated_memory_bytes"] > 0
    assert metric["peak_sampled_driver_memory_bytes"] > 0
    # Every profiled phase is separately attributed, so a slow accelerator run
    # can be blamed on the pipeline it is actually spending time in rather than
    # on one bucket that quietly holds four of them.
    assert all(
        metric[phase] > 0.0
        for phase in (
            "data_seconds",
            "transfer_seconds",
            "compute_seconds",
            "optimizer_seconds",
        )
    )


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


def _train_at_declared_context(
    tmp_path: Path,
    maximum_context_plies: int,
    *,
    train_selection: str = "",
) -> None:
    """Train on the sample corpus against one declared context length."""

    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "data",
        load_config(PrepareConfig, path=SAMPLE_DATA_CONFIG),
    )
    config_path = _write_training_config(
        tmp_path / f"context-{maximum_context_plies}",
        normalized=prepared.normalized_path,
        manifest=prepared.manifest_path,
        output=tmp_path / f"run-{maximum_context_plies}",
        validation=False,
        maximum_context_plies=maximum_context_plies,
        train_selection=train_selection,
    )
    run_training(load_config(TrainingConfig, path=config_path))


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
    maximum_context_plies: int | None = None,
    shuffle: bool = False,
    device: str = "cpu",
    precision: str = "float32",
    determinism: str = "strict",
    extra: str = "",
    train_selection: str = "",
    train_streaming: str = "",
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "training.toml"
    resume_selection = (
        f"resume_from = {json.dumps(str(resume_from))}\n"
        if resume_from is not None
        else ""
    )
    context_declaration = (
        ""
        if maximum_context_plies is None
        else f"maximum_context_plies = {maximum_context_plies}\n"
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
{context_declaration}
[train]
normalized = {json.dumps(str(normalized))}
manifest = {json.dumps(str(manifest))}

[train.loader]
split = "train"
batch_size = 1
shuffle = {str(shuffle).lower()}
{train_selection}{train_streaming}{validation_selection}
""",
        encoding="utf-8",
    )
    return config_path
