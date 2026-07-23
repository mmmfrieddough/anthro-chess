"""Checkpoint-backed full-history model execution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import ValidationError
from torch import Tensor

from anthro_chess.chess import ACTION_VOCABULARY_SIZE, action_vocabulary_identity
from anthro_chess.data import DecisionContext, encoding_identity
from anthro_chess.inference.config import ModelRunnerConfig
from anthro_chess.inference.selection import (
    ModelSelectionError,
    ResolvedModelSelection,
    resolve_model_selection,
)
from anthro_chess.models import CausalMoveModel, MoveModelBatch, MoveModelConfig
from anthro_chess.training.checkpoints import CheckpointError, load_training_checkpoint


class ModelRunnerError(ValueError):
    """Raised when a selected checkpoint cannot serve model decisions safely."""


@dataclass(frozen=True)
class InferenceDeviceCapabilities:
    """Hardware capabilities relevant to model-runner device selection."""

    mps_built: bool
    mps_available: bool


class CheckpointModelRunner:
    """Run one compatible action model with full trajectory recomputation."""

    def __init__(
        self,
        model: CausalMoveModel,
        *,
        selection: ResolvedModelSelection,
        device: torch.device,
    ) -> None:
        self._model = model
        self.selection = selection
        self.device = device

    @classmethod
    def load(
        cls,
        config: ModelRunnerConfig,
        *,
        run_root: Path | None = None,
        capabilities: InferenceDeviceCapabilities | None = None,
    ) -> CheckpointModelRunner:
        """Resolve, validate, and load a retained training checkpoint."""

        try:
            selection = resolve_model_selection(config, run_root=run_root)
            checkpoint = load_training_checkpoint(selection.checkpoint_path)
            run_record = _load_run_record(selection.run_record_path)
            model_config = _validate_artifact_contract(checkpoint, run_record)
            device = _resolve_device(config.device, capabilities=capabilities)
            model = CausalMoveModel(model_config)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.to(device=device, dtype=torch.float32)
            model.eval()
        except (
            CheckpointError,
            ModelSelectionError,
            OSError,
            RuntimeError,
            ValidationError,
            ValueError,
        ) as error:
            if isinstance(error, ModelRunnerError):
                raise
            raise ModelRunnerError(f"cannot load model runner: {error}") from error
        return cls(model, selection=selection, device=device)

    def predict(self, context: DecisionContext) -> Tensor:
        """Return current-decision raw action logits on CPU."""

        try:
            batch = MoveModelBatch.from_decision_context(
                context,
                device=self.device,
            )
            with torch.inference_mode():
                logits = self._model(batch)[0, -1]
        except (RuntimeError, ValueError) as error:
            raise ModelRunnerError(f"model inference failed: {error}") from error
        if logits.shape != (ACTION_VOCABULARY_SIZE,):
            raise ModelRunnerError("model returned an invalid action-logit shape")
        if not torch.isfinite(logits).all():
            raise ModelRunnerError("model returned non-finite action logits")
        return cast(
            Tensor,
            logits.detach().to(device="cpu", dtype=torch.float32).clone(),
        )


def _load_run_record(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelRunnerError(f"cannot load run record {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ModelRunnerError(f"run record is not an object: {path}")
    return cast(dict[str, Any], loaded)


def _validate_artifact_contract(
    checkpoint: Mapping[str, Any],
    run_record: Mapping[str, Any],
) -> MoveModelConfig:
    metadata = _mapping(checkpoint.get("metadata"), "checkpoint metadata")
    compatibility = _mapping(
        checkpoint.get("compatibility"),
        "checkpoint compatibility metadata",
    )
    checkpoint_model = _mapping(metadata.get("model"), "checkpoint model identity")
    compatibility_model = _mapping(
        compatibility.get("model"),
        "checkpoint compatible model identity",
    )
    run_model = _mapping(run_record.get("model"), "run model identity")

    config_record = _mapping(checkpoint_model.get("config"), "model configuration")
    model_config = MoveModelConfig.model_validate(config_record)
    expected_model = CausalMoveModel(model_config).identity()
    identities = (
        (checkpoint_model, "checkpoint model"),
        (compatibility_model, "checkpoint compatible model"),
        (run_model, "run model"),
    )
    for actual, label in identities:
        if dict(actual) != expected_model:
            raise ModelRunnerError(f"{label} is incompatible with this model runner")

    expected_action = action_vocabulary_identity()
    expected_encoding = encoding_identity()
    for container, label in (
        (metadata, "checkpoint metadata"),
        (compatibility, "checkpoint compatibility"),
        (run_record, "run record"),
    ):
        if container.get("action_vocabulary") != expected_action:
            raise ModelRunnerError(f"{label} action vocabulary is incompatible")
        if container.get("encoding") != expected_encoding:
            raise ModelRunnerError(f"{label} model-facing encoding is incompatible")

    resolved = _mapping(metadata.get("resolved_config"), "resolved configuration")
    training_config = _mapping(
        resolved.get("config"),
        "resolved training configuration",
    )
    run_resolved = _mapping(
        run_record.get("resolved_config"),
        "run resolved configuration",
    )
    run_training_config = _mapping(
        run_resolved.get("config"),
        "run resolved training configuration",
    )
    expected_config = model_config.model_dump(mode="json")
    if training_config.get("model") != expected_config:
        raise ModelRunnerError(
            "resolved training configuration disagrees with the model identity"
        )
    if run_training_config.get("model") != expected_config:
        raise ModelRunnerError(
            "run resolved training configuration disagrees with the model identity"
        )
    checkpoint_execution = _mapping(
        metadata.get("execution"),
        "checkpoint execution metadata",
    )
    run_execution = _mapping(
        run_record.get("execution"),
        "run execution metadata",
    )
    for execution, label in (
        (checkpoint_execution, "checkpoint"),
        (run_execution, "run"),
    ):
        if (
            execution.get("precision") != "float32"
            or execution.get("parameter_dtype") != "float32"
        ):
            raise ModelRunnerError(f"{label} parameter precision is unsupported")
    if expected_model.get("rating_conditioning") != (
        "post-transformer-feature-modulation"
    ):
        raise ModelRunnerError("checkpoint uses an unsupported rating context contract")
    if (
        expected_model.get("timing_inputs") is not False
        or expected_model.get("timing_head") is not False
    ):
        raise ModelRunnerError("checkpoint timing output is unsupported")
    return model_config


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelRunnerError(f"{label} is missing or invalid")
    return cast(Mapping[str, Any], value)


def _resolve_device(
    requested: str,
    *,
    capabilities: InferenceDeviceCapabilities | None,
) -> torch.device:
    observed = capabilities or InferenceDeviceCapabilities(
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
    )
    if requested == "auto":
        backend = "mps" if observed.mps_available else "cpu"
    elif requested == "mps":
        if not observed.mps_available:
            availability = (
                "the installed Torch build has no MPS support"
                if not observed.mps_built
                else "MPS is not available on this host"
            )
            raise ModelRunnerError(
                f"model runner device mps was requested but {availability}"
            )
        backend = "mps"
    elif requested == "cpu":
        backend = "cpu"
    else:
        raise ModelRunnerError(f"unknown model runner device: {requested}")
    return torch.device(backend)
