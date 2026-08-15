"""Disposable TensorBoard projection of the committed results store.

TensorBoard does not understand Anthro Chess series fingerprints.  This
projection preserves that boundary structurally: every raw fingerprint gets
its own run, while runs for the same metric emit the same scalar tag.  The UI
can therefore overlay comparable checkpoint history without drawing a line
through a series break.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from anthro_chess.evaluation.results.metrics import metric_definition
from anthro_chess.evaluation.results.records import ResultEnvelope
from anthro_chess.evaluation.results.store import checkpoint_labels

PROJECTION_MARKER = ".anthro-chess-results-tensorboard"
PROJECTION_VERSION = 1


class TensorBoardProjectionError(ValueError):
    """Raised when a safe, faithful projection cannot be written."""


@dataclass(frozen=True)
class TensorBoardProjection:
    """Summary of one regenerated TensorBoard view."""

    output: Path
    checkpoints: int
    runs: int
    points: int


@dataclass(frozen=True)
class _Point:
    ordinal: int
    value: float


def project_results(
    results: Sequence[ResultEnvelope],
    output: Path,
    *,
    store_root: Path,
) -> TensorBoardProjection:
    """Regenerate a TensorBoard view over ``results`` at ``output``.

    Checkpoint ordinals follow first appearance in the authoritative store.
    Re-scoring an older checkpoint therefore replaces its point at the same
    step instead of appending it to the end of history.
    """

    destination = output.expanduser().resolve()
    committed_store = store_root.expanduser().resolve()
    _require_outside_store(destination, committed_store)
    _require_owned_or_empty(destination)

    labels = checkpoint_labels(results)
    ordinals = {label: ordinal for ordinal, label in enumerate(labels)}
    series = _series_points(results, ordinals)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-",
            dir=destination.parent,
        )
    )
    try:
        point_count = _write_projection(temporary, series)
        (temporary / PROJECTION_MARKER).write_text(
            f"anthro-chess-results-tensorboard-v{PROJECTION_VERSION}\n",
            encoding="utf-8",
        )
        # Recheck immediately before replacement. An initially empty or owned
        # directory may have gained unrelated content while event files were
        # being generated.
        _require_owned_or_empty(destination)
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                raise TensorBoardProjectionError(
                    f"TensorBoard output {destination} is not a directory"
                )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return TensorBoardProjection(
        output=destination,
        checkpoints=len(labels),
        runs=len(series),
        points=point_count,
    )


def _series_points(
    results: Sequence[ResultEnvelope],
    ordinals: dict[str, int],
) -> dict[tuple[str, str, str], tuple[_Point, ...]]:
    """Return the latest reading at each checkpoint on each raw series."""

    latest: dict[tuple[str, str, str, str], _Point] = {}
    ordered = sorted(results, key=lambda item: (item.recorded_at, item.result_id))
    for envelope in ordered:
        checkpoint = envelope.checkpoint.label
        ordinal = ordinals[checkpoint]
        for measurement in envelope.measurements:
            definition = metric_definition(measurement.metric)
            key = (
                definition.family,
                definition.identifier,
                measurement.fingerprint,
                checkpoint,
            )
            latest[key] = _Point(ordinal=ordinal, value=measurement.value)

    grouped: dict[tuple[str, str, str], list[_Point]] = {}
    for (family, metric, fingerprint, _checkpoint), point in latest.items():
        grouped.setdefault((family, metric, fingerprint), []).append(point)
    return {
        key: tuple(sorted(points, key=lambda point: point.ordinal))
        for key, points in sorted(grouped.items())
    }


def _write_projection(
    root: Path,
    series: dict[tuple[str, str, str], tuple[_Point, ...]],
) -> int:
    points = 0
    for (family, metric, fingerprint), readings in series.items():
        run_directory = root / family / metric / fingerprint
        tag = f"{family}/{metric}"
        try:
            with SummaryWriter(log_dir=str(run_directory)) as writer:
                for reading in readings:
                    writer.add_scalar(tag, reading.value, reading.ordinal)
                    points += 1
        except Exception as error:
            raise TensorBoardProjectionError(
                f"cannot write TensorBoard run {run_directory}: {error}"
            ) from error
    return points


def _require_outside_store(output: Path, store: Path) -> None:
    if output == store or output in store.parents or store in output.parents:
        raise TensorBoardProjectionError(
            "TensorBoard output must be outside the results store it projects"
        )


def _require_owned_or_empty(output: Path) -> None:
    if not output.exists():
        return
    if not output.is_dir():
        raise TensorBoardProjectionError(
            f"TensorBoard output {output} is not a directory"
        )
    entries = tuple(output.iterdir())
    if not entries:
        return
    marker = output / PROJECTION_MARKER
    expected = f"anthro-chess-results-tensorboard-v{PROJECTION_VERSION}\n"
    if not marker.is_file() or marker.read_text(encoding="utf-8") != expected:
        raise TensorBoardProjectionError(
            f"refusing to replace non-projection directory {output}"
        )


__all__ = [
    "PROJECTION_MARKER",
    "TensorBoardProjection",
    "TensorBoardProjectionError",
    "project_results",
]
