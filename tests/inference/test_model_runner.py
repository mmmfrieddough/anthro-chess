from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import chess
import pytest
import torch

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    action_vocabulary_identity,
)
from anthro_chess.data import build_decision_context, encoding_identity
from anthro_chess.inference import (
    MODEL_SELECTION_FILE,
    CheckpointModelRunner,
    InferenceDeviceCapabilities,
    ModelRunnerConfig,
    ModelRunnerError,
    ModelSelectionError,
    resolve_model_selection,
    write_model_selection,
)
from anthro_chess.models import CausalMoveModel, MoveModelBatch, MoveModelConfig
from anthro_chess.training.checkpoints import save_training_checkpoint


def test_inference_package_does_not_import_training_orchestration() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import anthro_chess.inference; "
                "assert 'anthro_chess.training.runner' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_cpu_runner_loads_and_recomputes_complete_target_free_history(
    tmp_path: Path,
) -> None:
    checkpoint = _write_run(tmp_path / "run", seed=11)
    runner = CheckpointModelRunner.load(
        ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu")
    )
    empty = build_decision_context(chess.Board(), (), target_rating=1650)
    board, moves = _position(("e2e4", "e7e5", "g1f3"))
    developed = build_decision_context(board, moves, target_rating=1650)

    seen_lengths: list[int] = []
    hook = runner._model.register_forward_pre_hook(  # noqa: SLF001
        lambda _module, arguments: seen_lengths.append(
            arguments[0].attention_mask.shape[1]
        )
    )
    try:
        empty_logits = runner.predict(empty)
        developed_logits = runner.predict(developed)
        repeated_logits = runner.predict(developed)
    finally:
        hook.remove()

    assert seen_lengths == [1, 4, 4]
    assert empty_logits.shape == (ACTION_VOCABULARY_SIZE,)
    assert empty_logits.device.type == "cpu"
    assert empty_logits.dtype == torch.float32
    assert torch.isfinite(empty_logits).all()
    assert not torch.equal(empty_logits, developed_logits)
    torch.testing.assert_close(developed_logits, repeated_logits, rtol=0.0, atol=0.0)
    assert runner.selection.checkpoint_path == checkpoint.resolve()
    assert runner.selection.as_record()["source"] == "explicit-checkpoint"


def test_decision_tensorization_rates_only_the_current_decision() -> None:
    board, moves = _position(("d2d4", "d7d5"))
    context = build_decision_context(board, moves, target_rating=1800)

    batch = MoveModelBatch.from_decision_context(context)

    assert batch.inputs.target_rating.present.tolist() == [[False, False, True]]
    assert batch.inputs.target_rating.values.tolist() == [[0, 0, 1800]]
    assert batch.inputs.previous_action_id.present.tolist() == [[False, True, True]]
    assert batch.ply_indices.tolist() == [[0, 1, 2]]
    assert not batch.action_loss_mask.any()
    assert batch.causal_attention_mask.tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]


def test_batched_tensorization_pads_past_the_end_of_shorter_histories() -> None:
    short_board, short_moves = _position(("d2d4",))
    long_board, long_moves = _position(("d2d4", "d7d5", "c2c4"))
    contexts = (
        build_decision_context(short_board, short_moves, target_rating=1800),
        build_decision_context(long_board, long_moves, target_rating=1200),
    )

    batch = MoveModelBatch.from_decision_contexts(contexts)

    assert batch.attention_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    # Each history's rating marks its own last real timestep, and the padded
    # columns carry no inputs at all.
    assert batch.inputs.target_rating.values.tolist() == [
        [0, 1800, 0, 0],
        [0, 0, 0, 1200],
    ]
    assert batch.inputs.previous_action_id.present.tolist() == [
        [False, True, False, False],
        [False, True, True, True],
    ]
    assert batch.ply_indices.tolist() == [[0, 1, 0, 0], [0, 1, 2, 3]]


def test_a_batched_prediction_serves_every_pending_decision_in_one_pass(
    tmp_path: Path,
) -> None:
    checkpoint = _write_run(tmp_path / "run", seed=13)
    runner = CheckpointModelRunner.load(
        ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu")
    )
    board, moves = _position(("e2e4", "e7e5", "g1f3"))
    contexts = (
        build_decision_context(chess.Board(), (), target_rating=1650),
        build_decision_context(board, moves, target_rating=1650),
    )
    separate = tuple(runner.predict(context) for context in contexts)

    widths: list[int] = []
    hook = runner._model.register_forward_pre_hook(  # noqa: SLF001
        lambda _module, arguments: widths.append(arguments[0].attention_mask.shape[1])
    )
    try:
        together = runner.predict_batch(contexts)
    finally:
        hook.remove()

    assert widths == [4]
    assert len(together) == len(contexts)
    # Batching changes which floating-point kernels run, so a padded row agrees
    # with its own single-history prediction to float32 precision rather than
    # bit for bit.
    for batched, alone in zip(together, separate, strict=True):
        assert batched.shape == (ACTION_VOCABULARY_SIZE,)
        torch.testing.assert_close(batched, alone, rtol=1e-5, atol=1e-5)
    assert not torch.equal(together[0], together[1])


def test_a_prediction_batch_cannot_be_empty(tmp_path: Path) -> None:
    checkpoint = _write_run(tmp_path / "run", seed=13)
    runner = CheckpointModelRunner.load(
        ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu")
    )

    with pytest.raises(ModelRunnerError, match="at least one context"):
        runner.predict_batch(())


def test_default_and_explicit_run_selection_have_deliberate_precedence(
    tmp_path: Path,
) -> None:
    first = _write_run(tmp_path / "runs" / "first", seed=3)
    second = _write_run(tmp_path / "runs" / "second", seed=5)
    selection_path = write_model_selection(tmp_path / "runs", run="first")

    default = resolve_model_selection(ModelRunnerConfig(), run_root=tmp_path / "runs")
    explicit_run = resolve_model_selection(
        ModelRunnerConfig(run_path=Path("second"), checkpoint=second.name),
        run_root=tmp_path / "runs",
    )
    explicit_checkpoint = resolve_model_selection(
        ModelRunnerConfig(checkpoint_path=second),
        run_root=tmp_path / "unrelated-root",
    )
    default_runner = CheckpointModelRunner.load(
        ModelRunnerConfig(device="cpu"),
        run_root=tmp_path / "runs",
    )

    assert selection_path == tmp_path / "runs" / MODEL_SELECTION_FILE
    assert default.checkpoint_path == first.resolve()
    assert default.source == "default-selection"
    assert explicit_run.checkpoint_path == second.resolve()
    assert explicit_run.source == "explicit-run"
    assert explicit_checkpoint.checkpoint_path == second.resolve()
    assert explicit_checkpoint.source == "explicit-checkpoint"
    assert default_runner.selection.checkpoint_path == first.resolve()


def test_selection_rejects_missing_stale_and_escaping_records(tmp_path: Path) -> None:
    with pytest.raises(ModelSelectionError, match="run root is required"):
        resolve_model_selection(ModelRunnerConfig())
    with pytest.raises(ModelSelectionError, match="cannot resolve default"):
        resolve_model_selection(ModelRunnerConfig(), run_root=tmp_path)

    (tmp_path / MODEL_SELECTION_FILE).write_text(
        json.dumps({"version": 1, "run": "../outside", "checkpoint": "latest"}),
        encoding="utf-8",
    )
    with pytest.raises(ModelSelectionError, match="escapes the run root"):
        resolve_model_selection(ModelRunnerConfig(), run_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("action", "action vocabulary is incompatible"),
        ("encoding", "model-facing encoding is incompatible"),
        ("model", "model is incompatible"),
        ("resolved-config", "configuration disagrees"),
        ("precision", "parameter precision is unsupported"),
    ],
)
def test_runner_rejects_incompatible_artifact_contracts(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    checkpoint_path = _write_run(tmp_path / mutation, seed=7)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    run_path = checkpoint_path.parents[1] / "run.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))

    if mutation == "action":
        incompatible = {"name": "other-actions"}
        payload["metadata"]["action_vocabulary"] = incompatible
        payload["compatibility"]["action_vocabulary"] = incompatible
        run_record["action_vocabulary"] = incompatible
    elif mutation == "encoding":
        incompatible = {"name": "other-encoding"}
        payload["metadata"]["encoding"] = incompatible
        payload["compatibility"]["encoding"] = incompatible
        run_record["encoding"] = incompatible
    elif mutation == "model":
        for model in (
            payload["metadata"]["model"],
            payload["compatibility"]["model"],
            run_record["model"],
        ):
            model["rating_conditioning"] = "history-rating"
    elif mutation == "resolved-config":
        payload["metadata"]["resolved_config"]["config"]["model"]["model_dim"] = 18
        run_record["resolved_config"]["config"]["model"]["model_dim"] = 18
    else:
        payload["metadata"]["execution"]["precision"] = "float16"
        payload["metadata"]["execution"]["parameter_dtype"] = "float16"
        run_record["execution"]["precision"] = "float16"
        run_record["execution"]["parameter_dtype"] = "float16"

    torch.save(payload, checkpoint_path)
    run_path.write_text(json.dumps(run_record), encoding="utf-8")

    with pytest.raises(ModelRunnerError, match=message):
        CheckpointModelRunner.load(
            ModelRunnerConfig(checkpoint_path=checkpoint_path, device="cpu")
        )


def test_runner_rejects_unavailable_explicit_device(tmp_path: Path) -> None:
    checkpoint = _write_run(tmp_path / "run", seed=13)

    with pytest.raises(ModelRunnerError, match="MPS is not available"):
        CheckpointModelRunner.load(
            ModelRunnerConfig(checkpoint_path=checkpoint, device="mps"),
            capabilities=InferenceDeviceCapabilities(
                mps_built=True,
                mps_available=False,
            ),
        )


def test_runner_accepts_checkpoint_from_a_different_recorded_backend(
    tmp_path: Path,
) -> None:
    checkpoint = _write_run(tmp_path / "run", seed=17)
    run_path = checkpoint.parents[1] / "run.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    run_record["execution"]["device"] = "mps"
    run_record["execution"]["backend"] = "mps"
    run_path.write_text(json.dumps(run_record), encoding="utf-8")

    runner = CheckpointModelRunner.load(
        ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu")
    )

    assert runner.device.type == "cpu"


def test_model_runner_config_rejects_ambiguous_or_unknown_selection() -> None:
    with pytest.raises(ValueError, match="authoritative"):
        ModelRunnerConfig(
            checkpoint_path=Path("checkpoint.pt"),
            run_path=Path("run"),
        )
    with pytest.raises(ValueError, match="step-########"):
        ModelRunnerConfig(run_path=Path("run"), checkpoint="newest.pt")


def _write_run(path: Path, *, seed: int) -> Path:
    torch.manual_seed(seed)
    path.mkdir(parents=True)
    config = MoveModelConfig(
        piece_embedding_dim=2,
        action_embedding_dim=2,
        model_dim=4,
        attention_heads=1,
        transformer_layers=1,
        feedforward_dim=8,
        dropout=0.0,
    )
    model = CausalMoveModel(config)
    model_identity = model.identity()
    resolved_config = {
        "config": {"model": config.model_dump(mode="json")},
        "provenance": {"source": None, "overrides": []},
    }
    metadata = {
        "resolved_config": copy.deepcopy(resolved_config),
        "code": {"package_version": "test", "git_revision": "test"},
        "data": {},
        "model": copy.deepcopy(model_identity),
        "action_vocabulary": action_vocabulary_identity(),
        "encoding": encoding_identity(),
        "execution": {
            "device": "cpu",
            "backend": "cpu",
            "precision": "float32",
            "parameter_dtype": "float32",
            "determinism": "strict",
            "gradient_accumulation_steps": 1,
            "phase_profiling": False,
        },
    }
    compatibility = {
        "training_config": {},
        "data": {},
        "model": copy.deepcopy(model_identity),
        "action_vocabulary": action_vocabulary_identity(),
        "encoding": encoding_identity(),
    }
    checkpoint = path / "checkpoints" / "step-00000001.pt"
    save_training_checkpoint(
        checkpoint,
        global_step=1,
        counters={"processed_positions": 1},
        model_state=model.state_dict(),
        optimizer_state={},
        scheduler_state=None,
        scaler_state=None,
        loader_state={},
        compatibility=compatibility,
        metadata=metadata,
        device="cpu",
    )
    run_record = {
        "version": 3,
        "resolved_config": copy.deepcopy(resolved_config),
        "model": copy.deepcopy(model_identity),
        "action_vocabulary": action_vocabulary_identity(),
        "encoding": encoding_identity(),
        "execution": copy.deepcopy(metadata["execution"]),
    }
    (path / "run.json").write_text(
        json.dumps(run_record, sort_keys=True),
        encoding="utf-8",
    )
    return checkpoint


def _position(moves: tuple[str, ...]) -> tuple[chess.Board, tuple[chess.Move, ...]]:
    board = chess.Board()
    history = tuple(chess.Move.from_uci(move) for move in moves)
    for move in history:
        board.push(move)
    return board, history
