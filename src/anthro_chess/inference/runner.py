"""Checkpoint-backed model execution for one decision at a time."""

from __future__ import annotations

import copy
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
from anthro_chess.inference.config import InferenceDevice, ModelRunnerConfig
from anthro_chess.inference.selection import (
    ModelSelectionError,
    ResolvedModelSelection,
    resolve_model_selection,
)
from anthro_chess.models import MoveModel, MoveModelBatch, MoveModelConfig
from anthro_chess.models.move_model import model_identity
from anthro_chess.training.checkpoints import (
    CheckpointError,
    load_training_checkpoint,
    parameter_sha256,
    training_identity_sha256,
)

logger = logging.getLogger(__name__)


class ModelRunnerError(ValueError):
    """Raised when a selected checkpoint cannot serve model decisions safely."""


#: Accelerator backends an automatic selection considers, in preference order.
#: In practice at most one is ever present, because the CUDA and MPS builds
#: target different platforms. The order therefore settles nothing real and
#: exists so ``auto`` is defined rather than left to whichever check runs first.
AUTO_ACCELERATORS = ("cuda", "mps")


@dataclass(frozen=True)
class InferenceDeviceCapabilities:
    """Hardware capabilities relevant to model-runner device selection.

    The CUDA fields default to absent so a fixture describing a host without
    one can stay silent about it, which is how most of them read.
    """

    mps_built: bool
    mps_available: bool
    cuda_built: bool = False
    cuda_available: bool = False

    def __post_init__(self) -> None:
        for backend in AUTO_ACCELERATORS:
            if self.available(backend) and not self.built(backend):
                raise ValueError(
                    f"available {backend.upper()} requires a "
                    f"{backend.upper()}-enabled Torch build"
                )

    def built(self, backend: str) -> bool:
        """Whether the installed Torch build carries support for ``backend``."""

        return self._support(backend)[0]

    def available(self, backend: str) -> bool:
        """Whether ``backend`` can actually run work on this host."""

        return self._support(backend)[1]

    def _support(self, backend: str) -> tuple[bool, bool]:
        """Return whether ``backend`` is built and whether it is available.

        An unknown name fails rather than falling through to some other
        backend's answer, which would report a host that does not exist.
        """

        if backend == "cuda":
            return self.cuda_built, self.cuda_available
        if backend == "mps":
            return self.mps_built, self.mps_available
        raise ValueError(f"unknown accelerator backend: {backend}")

    def unavailability(self, backend: str) -> str:
        """Say why ``backend`` cannot be used, distinguishing build from host.

        The two are different problems with different fixes — an installation
        without the backend compiled in, against a machine with no such device
        — and a message that conflated them would send a reader to the wrong
        one.
        """

        name = backend.upper()
        if not self.built(backend):
            return f"the installed Torch build has no {name} support"
        return f"{name} is not available on this host"


def detect_inference_device_capabilities() -> InferenceDeviceCapabilities:
    """Read this process's model-runner device capabilities from Torch."""

    return InferenceDeviceCapabilities(
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
        cuda_built=torch.backends.cuda.is_built(),
        cuda_available=torch.cuda.is_available(),
    )


class CheckpointModelRunner:
    """Run one compatible action model over a bounded window of boards."""

    def __init__(
        self,
        model: MoveModel,
        *,
        selection: ResolvedModelSelection,
        device: torch.device,
        global_step: int,
        processed_positions: int,
        metadata: Mapping[str, Any],
        training_sha256: str,
    ) -> None:
        self._model = model
        self.selection = selection
        self.device = device
        self.global_step = global_step
        self.processed_positions = processed_positions
        self.metadata = dict(metadata)
        self.training_sha256 = training_sha256

    def parameter_sha256(self) -> str:
        """Return the digest identifying the loaded parameters."""

        return parameter_sha256(self._model)

    def replicated(self, device: torch.device) -> CheckpointModelRunner:
        """Return the same loaded checkpoint on another device.

        A reading that spreads its batches over several accelerators needs one
        model per device. These are copies of the parameters already in memory
        rather than a second read of the checkpoint, so every replica answers
        for the same selection and the same digest.
        """

        model = copy.deepcopy(self._model).to(device=device)
        model.eval()
        return CheckpointModelRunner(
            model,
            selection=self.selection,
            device=device,
            global_step=self.global_step,
            processed_positions=self.processed_positions,
            metadata=self.metadata,
            training_sha256=self.training_sha256,
        )

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
            training_sha256 = training_identity_sha256(checkpoint["compatibility"])
            device = resolve_inference_device(config.device, capabilities=capabilities)
            model = MoveModel(model_config)
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
            processed_positions=checkpoint["counters"]["processed_positions"],
            metadata=checkpoint["metadata"],
            training_sha256=training_sha256,
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
        try:
            batch = MoveModelBatch.from_decision_contexts(
                contexts,
                device=self.device,
            )
        except (RuntimeError, ValueError) as error:
            raise ModelRunnerError(f"model inference failed: {error}") from error
        return self.decision_logits(batch, self.decision_indices(contexts))

    def decision_indices(self, contexts: Sequence[DecisionContext]) -> Tensor:
        """Return the ply each context is deciding at, one per row."""

        return torch.as_tensor(
            [len(context.plies) - 1 for context in contexts],
            device=self.device,
        )

    def decision_logits(
        self,
        batch: MoveModelBatch,
        decisions: Tensor,
    ) -> tuple[Tensor, ...]:
        """Return served logits for a batch and decision rows already built.

        A seam :meth:`predict_batch` also goes through, so a benchmark can time
        the model work alone against the same call the engine makes. Timing the whole
        forward pass instead would score every historical ply, which serving
        does not do, and the difference would read as batch-construction cost.
        """

        try:
            with torch.inference_mode():
                logits = self._model.decide_at(batch, decisions)
        except (RuntimeError, ValueError) as error:
            raise ModelRunnerError(f"model inference failed: {error}") from error
        if logits.shape != (decisions.shape[0], ACTION_VOCABULARY_SIZE):
            raise ModelRunnerError("model returned an invalid action-logit shape")
        # Checked on the host copy the caller was getting anyway. Asking the
        # device instead would block on the whole queued forward pass once per
        # generated move, for an answer this transfer already brings back.
        resolved = logits.detach().to(device="cpu", dtype=torch.float32).clone()
        if not torch.isfinite(resolved).all():
            raise ModelRunnerError("model returned non-finite action logits")
        return resolved.unbind(0)


def _load_run_record(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelRunnerError(f"cannot load run record {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ModelRunnerError(f"run record is not an object: {path}")
    return cast(dict[str, Any], loaded)


def run_record_incompatibility(path: Path) -> str | None:
    """Return why the run record at ``path`` no longer loads, or ``None``.

    Every gate :func:`_validate_artifact_contract` applies to a run record,
    which is what lets a survey of a whole run root say which runs are still
    usable without opening one checkpoint. The expected identity is derived
    from the record's own configuration rather than from a checkpoint's, so
    this is the weaker answer by exactly one thing: it cannot see a checkpoint
    disagreeing with the record it sits beside, and it does not read the
    payload version, which lives in each ``.pt``. A run passing here can still
    fail to load, and a caller reporting the result should not promise more.
    """

    try:
        run_record = _load_run_record(path)
        run_model = _mapping(run_record.get("model"), "run model identity")
        config_record = _mapping(run_model.get("config"), "model configuration")
        expected = model_identity(MoveModelConfig.model_validate(config_record))
        return _run_record_incompatibility(run_record, expected)
    except (ModelRunnerError, ValidationError) as error:
        return str(error)


def _run_record_incompatibility(
    run_record: Mapping[str, Any],
    expected_model: Mapping[str, object],
) -> str | None:
    """Return the first gate this run record fails, or ``None``.

    Shared with the survey above so that a gate added here reaches the listing
    that says which runs still load, instead of leaving it to keep printing a
    verdict this function would refuse.
    """

    run_model = _mapping(run_record.get("model"), "run model identity")
    if dict(run_model) != dict(expected_model):
        version = run_model.get("version")
        current = expected_model.get("version")
        if version != current:
            return f"model identity {version}, and this code loads {current}"
        return "run model is incompatible with this model runner"
    if run_record.get("action_vocabulary") != action_vocabulary_identity():
        return "run record action vocabulary is incompatible"
    if run_record.get("encoding") != encoding_identity():
        return "run record model-facing encoding is incompatible"
    execution = _mapping(run_record.get("execution"), "run execution metadata")
    if (
        execution.get("precision") != "float32"
        or execution.get("parameter_dtype") != "float32"
    ):
        return "run parameter precision is unsupported"
    return None


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

    config_record = _mapping(checkpoint_model.get("config"), "model configuration")
    model_config = MoveModelConfig.model_validate(config_record)
    expected_model = model_identity(model_config)
    identities = (
        (checkpoint_model, "checkpoint model"),
        (compatibility_model, "checkpoint compatible model"),
    )
    for actual, label in identities:
        if dict(actual) != expected_model:
            raise ModelRunnerError(f"{label} is incompatible with this model runner")

    expected_action = action_vocabulary_identity()
    expected_encoding = encoding_identity()
    for container, label in (
        (metadata, "checkpoint metadata"),
        (compatibility, "checkpoint compatibility"),
    ):
        if container.get("action_vocabulary") != expected_action:
            raise ModelRunnerError(f"{label} action vocabulary is incompatible")
        if container.get("encoding") != expected_encoding:
            raise ModelRunnerError(f"{label} model-facing encoding is incompatible")

    checkpoint_execution = _mapping(
        metadata.get("execution"),
        "checkpoint execution metadata",
    )
    if (
        checkpoint_execution.get("precision") != "float32"
        or checkpoint_execution.get("parameter_dtype") != "float32"
    ):
        raise ModelRunnerError("checkpoint parameter precision is unsupported")
    if expected_model.get("rating_conditioning") != "square-token-input-embedding":
        raise ModelRunnerError("checkpoint uses an unsupported rating context contract")
    if (
        expected_model.get("timing_inputs") is not False
        or expected_model.get("timing_head") is not False
    ):
        raise ModelRunnerError("checkpoint timing output is unsupported")
    # The run record is gated through the same function the machine report
    # reads, against the identity this checkpoint rebuilds to — so the report
    # cannot fall behind a gate added here, and this side still catches a
    # record that disagrees with the checkpoint it sits beside.
    blocker = _run_record_incompatibility(run_record, expected_model)
    if blocker is not None:
        raise ModelRunnerError(blocker)
    return model_config


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelRunnerError(f"{label} is missing or invalid")
    return cast(Mapping[str, Any], value)


def resolve_inference_device(
    requested: InferenceDevice,
    *,
    capabilities: InferenceDeviceCapabilities | None = None,
) -> torch.device:
    """Resolve automatic or explicit inference selection without silent fallback.

    Public because the resolution is worth checking against a described host
    rather than only against the one running the suite: a machine cannot be
    asked what it would have chosen with a different accelerator.
    """

    observed = capabilities or detect_inference_device_capabilities()
    if requested == "auto":
        backend = next(
            (name for name in AUTO_ACCELERATORS if observed.available(name)),
            "cpu",
        )
    elif requested in AUTO_ACCELERATORS:
        if not observed.available(requested):
            raise ModelRunnerError(
                f"model runner device {requested} was requested but "
                f"{observed.unavailability(requested)}"
            )
        backend = requested
    elif requested == "cpu":
        backend = "cpu"
    else:
        raise ModelRunnerError(f"unknown model runner device: {requested}")
    return torch.device(backend)
