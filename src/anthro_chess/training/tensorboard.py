"""Best-effort TensorBoard projection of the authoritative metrics stream."""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType

from torch.utils.tensorboard import SummaryWriter

from anthro_chess.training.cadence import CadenceReading
from anthro_chess.training.health import StepHealth

logger = logging.getLogger(__name__)

TENSORBOARD_DIRECTORY = "tensorboard"


class TrainingTensorBoard:
    """Mirror training records into a disposable TensorBoard event stream.

    Observability is never a training correctness boundary. Construction and
    emission failures disable this projection while the JSONL metrics stream
    continues to be written as the source of truth.
    """

    def __init__(self, directory: Path, *, purge_step: int | None = None) -> None:
        self.directory = directory
        self._writer: SummaryWriter | None = None
        try:
            self._writer = SummaryWriter(
                log_dir=str(directory),
                purge_step=purge_step,
            )
        except Exception as error:
            logger.warning(
                "TensorBoard output is unavailable for this run: %s",
                error,
            )

    def write_step(
        self,
        *,
        global_step: int,
        move_loss: float,
        learning_rate: float,
        health: StepHealth | None,
    ) -> None:
        """Write one optimizer-step reading at its authoritative step."""

        scalars = {
            "training/move_loss": move_loss,
            "training/learning_rate": learning_rate,
        }
        if health is not None:
            scalars.update(
                {
                    "training_health/gradient_norm": health.gradient_norm,
                    "training_health/gradient_norm_interval_maximum": (
                        health.gradient_norm_interval_maximum
                    ),
                }
            )
            if health.update_to_weight_ratio is not None:
                scalars["training_health/update_to_weight_ratio"] = (
                    health.update_to_weight_ratio
                )
        self._write_scalars(scalars, global_step=global_step)

    def write_evaluation(self, reading: CadenceReading) -> None:
        """Write one declared in-training evaluation reading."""

        self._write_scalars(
            {
                f"evaluation/{measurement.metric}": measurement.value
                for envelope in reading.envelopes
                for measurement in envelope.measurements
            },
            global_step=reading.global_step,
        )

    def close(self) -> None:
        """Flush and close the projection without making training depend on it."""

        if self._writer is None:
            return
        try:
            self._writer.close()
        except Exception as error:
            logger.warning("TensorBoard output stopped before close: %s", error)
        finally:
            self._writer = None

    def __enter__(self) -> TrainingTensorBoard:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _write_scalars(
        self,
        scalars: dict[str, float],
        *,
        global_step: int,
    ) -> None:
        if self._writer is None:
            return
        try:
            for tag, value in scalars.items():
                self._writer.add_scalar(tag, value, global_step)
        except Exception as error:
            logger.warning(
                "TensorBoard output stopped after a write failure: %s",
                error,
            )
            self.close()


__all__ = ["TENSORBOARD_DIRECTORY", "TrainingTensorBoard"]
