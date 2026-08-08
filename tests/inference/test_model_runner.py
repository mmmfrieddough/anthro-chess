from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import chess
import pytest
import torch

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    action_vocabulary_identity,
    encode_move,
)
from anthro_chess.data import (
    build_decision_context,
    en_passant_token,
    encoding_identity,
    previous_action_token,
)
from anthro_chess.inference import (
    MODEL_SELECTION_FILE,
    CheckpointModelRunner,
    InferenceDevice,
    InferenceDeviceCapabilities,
    ModelRunnerConfig,
    ModelRunnerError,
    ModelSelectionError,
    resolve_model_selection,
    write_model_selection,
)
from anthro_chess.machine import RUN_ROOT_VARIABLE
from anthro_chess.models import CausalMoveModel, MoveModelBatch, MoveModelConfig
from anthro_chess.training.checkpoints import save_training_checkpoint

from accelerators import inference_accelerator_parameters


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
    assert batch.inputs.previous_action_token[0, 0] == previous_action_token(None)
    assert batch.ply_indices.tolist() == [[0, 1, 2]]
    assert not batch.action_loss_mask.any()


def test_every_tensorized_column_carries_the_input_its_plies_name() -> None:
    """A column read from the wrong place would still be a plausible batch."""

    board, moves = _position(
        ("e2e4", "d7d5", "e4d5", "e7e5", "g1f3", "f8e7", "f1c4", "g8f6", "e1g1")
    )
    context = build_decision_context(board, moves, target_rating=2000)

    batch = MoveModelBatch.from_decision_context(context)

    inputs = batch.inputs
    for index, ply in enumerate(context.plies):
        position = ply.board
        assert inputs.piece_ids[0, index].tolist() == list(position.piece_ids)
        assert inputs.side_to_move[0, index] == position.side_to_move
        assert inputs.castling_rights[0, index] == position.castling_rights
        assert inputs.halfmove_clock[0, index] == position.halfmove_clock
        assert inputs.fullmove_number[0, index] == position.fullmove_number
        assert batch.ply_indices[0, index] == ply.ply_index
        assert inputs.en_passant_token[0, index] == en_passant_token(
            position.en_passant_square
        )
        assert inputs.previous_action_token[0, index] == previous_action_token(
            ply.previous_action_id
        )

    # Castling and en passant both occur, so neither column is being read as a
    # constant that happens to match.
    assert inputs.castling_rights[0].unique().numel() > 1
    assert (inputs.en_passant_token[0] != en_passant_token(None)).any()


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
    # A padded timestep has no previous action, so it reads the same row as a
    # game's first ply rather than the move a zero fill would name.
    absent = previous_action_token(None)
    opening, reply, advance = (
        encode_move(chess.Move.from_uci(move)) for move in ("d2d4", "d7d5", "c2c4")
    )
    assert batch.inputs.previous_action_token.tolist() == [
        [absent, opening, absent, absent],
        [absent, opening, reply, advance],
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


def test_a_prediction_batch_rejects_non_finite_logits(tmp_path: Path) -> None:
    runner = CheckpointModelRunner.load(
        ModelRunnerConfig(
            checkpoint_path=_write_run(tmp_path / "run", seed=13), device="cpu"
        )
    )
    context = build_decision_context(chess.Board(), (), target_rating=1650)

    with _replaced_logits(runner, lambda logits: logits * torch.inf):
        with pytest.raises(ModelRunnerError, match="non-finite action logits"):
            runner.predict_batch((context,))


def test_a_prediction_batch_never_asks_the_device_for_a_scalar(
    tmp_path: Path,
    device_read_trap: Callable[[Any], Any],
) -> None:
    """The finite check reads the host copy the caller was already getting.

    Asking the device instead blocks on the whole queued forward pass once per
    generated move, which the rollout, ladder, and termination benchmarks pay
    for at every ply. A CPU run cannot show that, so this asserts that nothing
    reaches back across the boundary.
    """

    runner = CheckpointModelRunner.load(
        ModelRunnerConfig(
            checkpoint_path=_write_run(tmp_path / "run", seed=13), device="cpu"
        )
    )
    context = build_decision_context(chess.Board(), (), target_rating=1650)
    (expected,) = runner.predict_batch((context,))

    with _replaced_logits(runner, device_read_trap):
        (trapped,) = runner.predict_batch((context,))

    torch.testing.assert_close(trapped, expected, rtol=0.0, atol=0.0)


@contextmanager
def _replaced_logits(
    runner: CheckpointModelRunner,
    transform: Callable[[Any], Any],
) -> Iterator[None]:
    """Substitute what the model hands back, leaving the runner path intact."""

    handle = runner._model.register_forward_hook(  # noqa: SLF001
        lambda _module, _arguments, output: transform(output)
    )
    try:
        yield
    finally:
        handle.remove()


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
    # An unconfigured run root and a configured one holding nothing are
    # different failures, and the message is the only place they differ.
    with pytest.raises(ModelSelectionError, match=RUN_ROOT_VARIABLE):
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


def test_a_checkpoint_rebuilds_at_the_context_length_its_run_declared(
    tmp_path: Path,
) -> None:
    """The identity is the only record the runner rebuilds from.

    A value missing from it becomes that field's default, which the model then
    enforces as its own bound — accepting histories the run never trained on,
    or refusing ones it declared.
    """

    checkpoint = _write_run(tmp_path / "run", seed=5, maximum_context_plies=512)

    runner = CheckpointModelRunner.load(
        ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu")
    )

    assert runner._model.config.maximum_context_plies == 512  # noqa: SLF001


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


@pytest.mark.gpu
@pytest.mark.parametrize("accelerator", inference_accelerator_parameters())
def test_a_cpu_checkpoint_decides_the_same_way_on_this_hosts_accelerator(
    accelerator: str,
    tmp_path: Path,
) -> None:
    """Load one checkpoint on CPU and on the accelerator, and compare.

    This is the acceptance check that a selection which merely resolves also
    runs: the weights survive the transfer, the forward pass reaches the same
    decision, and the logits still come back on the host.
    """

    checkpoint = _write_run(tmp_path / "run", seed=23)
    board, moves = _position(("e2e4", "e7e5", "g1f3"))
    context = build_decision_context(board, moves, target_rating=1650)
    selection = cast(InferenceDevice, accelerator)

    cpu = CheckpointModelRunner.load(
        ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu")
    )
    accelerated = CheckpointModelRunner.load(
        ModelRunnerConfig(checkpoint_path=checkpoint, device=selection)
    )

    assert accelerated.device.type == accelerator
    assert accelerated.parameter_sha256() == cpu.parameter_sha256()
    logits = accelerated.predict(context)
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, cpu.predict(context), rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("accelerator", inference_accelerator_parameters())
def test_automatic_selection_takes_the_accelerator_this_host_has(
    accelerator: str,
    tmp_path: Path,
) -> None:
    checkpoint = _write_run(tmp_path / "run", seed=29)

    runner = CheckpointModelRunner.load(
        ModelRunnerConfig(checkpoint_path=checkpoint, device="auto")
    )

    assert runner.device.type == accelerator


def test_an_exhausted_accelerator_fails_the_way_every_other_load_does(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out of memory arrives as a ``ModelRunnerError`` like any other failure.

    Torch raises it from the transfer as a ``RuntimeError`` subclass, so the
    behavior is inherited rather than written. Pinned because a caller that
    catches the package's own error would otherwise miss the one failure a
    large checkpoint on a busy GPU is most likely to produce, and reproducing
    it by genuinely exhausting a device is not something a suite should do.
    """

    checkpoint = _write_run(tmp_path / "run", seed=31)

    def refuse(self: CausalMoveModel, *args: Any, **kwargs: Any) -> CausalMoveModel:
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(CausalMoveModel, "to", refuse)

    with pytest.raises(ModelRunnerError, match="out of memory"):
        CheckpointModelRunner.load(
            ModelRunnerConfig(checkpoint_path=checkpoint, device="cpu")
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


def _write_run(
    path: Path,
    *,
    seed: int,
    maximum_context_plies: int | None = None,
) -> Path:
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
    if maximum_context_plies is not None:
        config = config.model_copy(
            update={"maximum_context_plies": maximum_context_plies}
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
