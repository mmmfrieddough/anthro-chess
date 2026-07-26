"""Series fingerprints built from realized inputs.

A series is one metric measured one way over one set of inputs. Two results
may be compared or plotted on the same line only when their fingerprints
match.

A fingerprint covers what a measurement actually consumed and how it was
computed: the metric's definition version, and a digest over the content of
the games scored. It deliberately excludes configuration text, software
versions, file layout, and command shape, so a refactor or a new flag leaves
every series intact while a change to what was measured breaks one
automatically.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from anthro_chess.data.schema import NormalizedColumn
from anthro_chess.evaluation.results.metrics import (
    DataProjection,
    MetricDefinition,
    MetricRegistryError,
    data_projection,
    metric_definition,
)

FINGERPRINT_ALGORITHM = "anthro-series-fingerprint-v1"
CONTENT_DIGEST_ALGORITHM = "anthro-projection-digest-v1"


class FingerprintError(ValueError):
    """Raised when a fingerprint cannot be computed from its declared inputs."""


@dataclass(frozen=True)
class DataComponent:
    """The realized-input half of a fingerprint.

    ``content_sha256`` digests only the projection a benchmark consumes, so a
    schema field no metric reads cannot end an unrelated series.
    """

    projection: str
    projection_version: int
    content_sha256: str
    games: int

    def as_record(self) -> dict[str, object]:
        """Return the stable record carried with every result."""

        return {
            "algorithm": CONTENT_DIGEST_ALGORITHM,
            "projection": self.projection,
            "projection_version": self.projection_version,
            "content_sha256": self.content_sha256,
            "games": self.games,
        }

    def fingerprint_component(self) -> dict[str, object]:
        """Return only the fields fingerprint identity depends on.

        The scored game count is provenance rather than identity: the content
        digest already changes whenever the scored set does.
        """

        return {
            "algorithm": CONTENT_DIGEST_ALGORITHM,
            "projection": self.projection,
            "projection_version": self.projection_version,
            "content_sha256": self.content_sha256,
        }


def projection_content_digest(
    rows: Iterable[Mapping[str, Any]],
    projection: DataProjection,
) -> DataComponent:
    """Digest the projected content of the games a benchmark scored.

    Row order does not matter and neither does any column outside the
    projection, so two benchmarks scoring the same games through the same
    projection agree even when they read the pool differently.
    """

    entries: dict[int, str] = {}
    for row in rows:
        try:
            game_id = int(row[NormalizedColumn.GAME_ID.value])
        except KeyError:
            raise FingerprintError(
                "projected rows must carry the normalized game id"
            ) from None
        missing = tuple(column for column in projection.columns if column not in row)
        if missing:
            raise FingerprintError(
                f"projection {projection.name!r} needs column(s) missing from "
                f"game {game_id}: {', '.join(missing)}"
            )
        if game_id in entries:
            raise FingerprintError(f"game {game_id} appears more than once")
        content = {column: row[column] for column in projection.columns}
        entries[game_id] = _canonical_digest(content)

    if not entries:
        raise FingerprintError("a data component needs at least one scored game")

    digest = sha256()
    digest.update(
        _canonical_bytes(
            {
                "algorithm": CONTENT_DIGEST_ALGORITHM,
                "projection": projection.name,
                "projection_version": projection.version,
                "columns": list(projection.columns),
            }
        )
    )
    for game_id in sorted(entries):
        digest.update(f"\n{game_id}:{entries[game_id]}".encode())
    return DataComponent(
        projection=projection.name,
        projection_version=projection.version,
        content_sha256=digest.hexdigest(),
        games=len(entries),
    )


def series_fingerprint(
    metric: str | MetricDefinition,
    data: DataComponent | None,
) -> str:
    """Return the fingerprint identifying one metric's series.

    A metric with no data dependency must pass ``None``. Substituting an
    empty view for a null data component would tie a structurally immune
    metric to evaluation inputs it never read.
    """

    definition = (
        metric if isinstance(metric, MetricDefinition) else metric_definition(metric)
    )
    _validate_data_component(definition, data)
    payload = {
        "algorithm": FINGERPRINT_ALGORITHM,
        "metric": definition.identifier,
        "definition_version": definition.definition_version,
        "data": None if data is None else data.fingerprint_component(),
    }
    return sha256(_canonical_bytes(payload)).hexdigest()


def _validate_data_component(
    definition: MetricDefinition,
    data: DataComponent | None,
) -> None:
    if definition.projection is None:
        if data is not None:
            raise FingerprintError(
                f"metric {definition.identifier!r} has no data dependency and "
                "must carry a null data component"
            )
        return
    if data is None:
        raise FingerprintError(
            f"metric {definition.identifier!r} consumes projection "
            f"{definition.projection!r} and needs a data component"
        )
    try:
        projection = data_projection(definition.projection)
    except MetricRegistryError as error:  # pragma: no cover - registry invariant
        raise FingerprintError(str(error)) from error
    if data.projection != projection.name:
        raise FingerprintError(
            f"metric {definition.identifier!r} consumes projection "
            f"{projection.name!r}, not {data.projection!r}"
        )
    if data.projection_version != projection.version:
        raise FingerprintError(
            f"projection {projection.name!r} is at version {projection.version}; "
            f"the data component reports version {data.projection_version}"
        )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_canonical_default,
    ).encode()


def _canonical_digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _canonical_default(value: object) -> object:
    """Normalize container types Parquet readers may hand back."""

    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, bytes):
        return value.hex()
    raise FingerprintError(f"cannot digest a value of type {type(value).__name__}")
