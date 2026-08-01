"""Checkpoint-backed full-history model execution."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
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
from anthro_chess.training.checkpoints import (
    CheckpointError,
    load_training_checkpoint,
    parameter_sha256,
)

logger = logging.getLogger(__name__)


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
        global_step: int,
        metadata: Mapping[str, Any],
        run_record: Mapping[str, Any],
    ) -> None:
        self._model = model
        self.selection = selection
        self.device = device
        self.global_step = global_step
        self.metadata = dict(metadata)
        self.run_record = dict(run_record)

    def parameter_sha256(self) -> str:
        """Return the digest identifying the loaded parameters."""

        return parameter_sha256(self._model)

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
            logger.info(
                "Loading selected model checkpoint %s",
                selection.checkpoint_path,
            )
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
        logger.info("Loaded model runner on %s", device.type)
        return cls(
            model,
            selection=selection,
            device=device,
            global_step=int(checkpoint["global_step"]),
            metadata=checkpoint["metadata"],
            run_record=run_record,
        )

    def action_logits(self, batch: MoveModelBatch) -> Tensor:
        """Return raw action logits for one aligned evaluation batch.

        Offline benchmarks score whole batches of held-out positions rather
        than one decision at a time, and they need the raw logits before any
        legal masking so legality itself stays measurable.
        """

        try:
            with torch.inference_mode():
                logits = self._model(batch)
        except (RuntimeError, ValueError) as error:
            raise ModelRunnerError(f"model inference failed: {error}") from error
        expected = (*batch.action_targets.shape, ACTION_VOCABULARY_SIZE)
        if logits.shape != expected:
            raise ModelRunnerError("model returned an invalid action-logit shape")
        return cast(Tensor, logits.detach().to(dtype=torch.float32).clone())

    def predict(self, context: DecisionContext) -> Tensor:
        """Return current-decision raw action logits on CPU."""

        return self.predict_batch((context,))[0]

    def predict_batch(self, contexts: Sequence[DecisionContext]) -> tuple[Tensor, ...]:
        """Return current-decision raw action logits for several histories.

        One padded forward pass serves every pending decision, which is what
        lets a benchmark playing many games at once keep the model fed instead
        of running it one position at a time. Each history is read at its own
        length, since padding sits past the end of the shorter ones.
        """

        if not contexts:
            raise ModelRunnerError("a prediction batch needs at least one context")
        lengths = tuple(len(context.plies) for context in contexts)
        try:
            batch = MoveModelBatch.from_decision_contexts(
                contexts,
                device=self.device,
            )
            with torch.inference_mode():
                predicted = self._model(batch)
            decisions = torch.as_tensor(
                [length - 1 for length in lengths],
                device=predicted.device,
            )
            rows = torch.arange(len(contexts), device=predicted.device)
            logits = predicted[rows, decisions]
        except (RuntimeError, ValueError) as error:
            raise ModelRunnerError(f"model inference failed: {error}") from error
        if logits.shape != (len(contexts), ACTION_VOCABULARY_SIZE):
            raise ModelRunnerError("model returned an invalid action-logit shape")
        # Checked on the host copy the caller was getting anyway. Asking the
        # device instead would block on the whole queued forward pass once per
        # generated move, for an answer this transfer already brings back.
        resolved = logits.detach().to(device="cpu", dtype=torch.float32).clone()
        if not torch.isfinite(resolved).all():
            raise ModelRunnerError("model returned non-finite action logits")
        return tuple(cast(Tensor, row) for row in resolved.unbind(0))


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
