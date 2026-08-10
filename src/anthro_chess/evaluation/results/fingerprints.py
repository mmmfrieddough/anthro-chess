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

Efficiency and generated-play metrics add one more realized input: the
**declared workload**. A latency figure taken at forty plies and one taken at
eighty measure different quantities, exactly as two different pools do, and so
do rollouts played at two temperatures, so the workload digest belongs in
identity and those metrics carry a :class:`WorkloadComponent`.

The machine deliberately does not. A cross-machine latency delta is not
meaningless — it is perfectly interpretable, just attributable to the
environment rather than to the model — and ending a series for it would
fragment efficiency history at every hardware change, Torch bump, and cloud
session, leaving no way to ask whether the shipped thing is getting slower.
Whether two numbers are safe to subtract is a question for a report, which can
answer it from the environment each result records. See
``docs/decisions/0018-workload-scoped-efficiency-series.md``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from anthro_chess.data.schema import (
    row_game_id,
)
from anthro_chess.evaluation.results.metrics import (
    DataProjection,
    MetricDefinition,
    MetricRegistryError,
    data_projection,
    metric_definition,
)

FINGERPRINT_ALGORITHM = "anthro-series-fingerprint-v1"
CONTENT_DIGEST_ALGORITHM = "anthro-projection-digest-v1"
WORKLOAD_DIGEST_ALGORITHM = "anthro-workload-digest-v1"


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


@dataclass(frozen=True)
class WorkloadComponent:
    """The declared-settings half of a workload-scoped metric's fingerprint.

    A workload says *what* was measured under which declared conditions: the
    ply depth a latency figure was taken at, the batch size a throughput figure
    was declared for, the rating and temperature a rollout figure was played
    at. Change it and the number measures a different quantity, which is the
    same test that puts scored content into identity.

    How many samples were taken is deliberately absent. Measuring more
    decisions, or generating more games, estimates the same quantity more
    precisely, in the same way that scoring more games does, so sample counts
    stay provenance rather than identity.
    """

    sha256: str

    def as_record(self) -> dict[str, object]:
        """Return the stable record carried with every efficiency result."""

        return {
            "algorithm": WORKLOAD_DIGEST_ALGORITHM,
            **self.fingerprint_component(),
        }

    def fingerprint_component(self) -> dict[str, object]:
        """Return only the fields fingerprint identity depends on."""

        return {"sha256": self.sha256}


def workload_digest(workload: Mapping[str, Any]) -> str:
    """Digest the declared workload of a workload-scoped benchmark.

    A benchmark passes only the settings that decide what was measured. Passing
    its whole configuration would put warmup counts, concurrency, and output
    paths into series identity and break the series on every unrelated flag.

    Benchmark cost is the deliberate exception and does pass the whole
    configuration, because a warmup count is an input to what an invocation
    cost rather than an unrelated flag. See
    ``docs/decisions/0031-committed-benchmark-cost.md``.
    """

    return _canonical_digest(
        {
            "algorithm": WORKLOAD_DIGEST_ALGORITHM,
            "workload": dict(workload),
        }
    )


def _projected_game_id(row: Mapping[str, Any]) -> int:
    """Return the identity a projected row is keyed by.

    Two kinds of row reach this boundary. One is a normalized corpus row, whose
    identity the schema derives from its source and source game key. The other
    is synthesized by a benchmark over something that is not a corpus game at
    all — a puzzle — and carries the identity it chose. Preferring the carried
    one leaves a benchmark's own series keyed the way it always was.
    """

    carried = row.get("game_id")
    if carried is not None:
        return int(carried)
    return row_game_id(row)


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
            game_id = _projected_game_id(row)
        except KeyError:
            raise FingerprintError(
                "projected rows must carry a game id or the columns deriving one"
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
    workload: WorkloadComponent | None = None,
) -> str:
    """Return the fingerprint identifying one metric's series.

    A metric with no data dependency must pass ``None``. Substituting an
    empty view for a null data component would tie a structurally immune
    metric to evaluation inputs it never read. The same holds in the other
    direction for the workload: only a metric that declares itself
    execution-sensitive may carry one, so an ordinary quality metric cannot
    acquire a workload dependency by accident.
    """

    definition = (
        metric if isinstance(metric, MetricDefinition) else metric_definition(metric)
    )
    _validate_data_component(definition, data)
    _validate_workload_component(definition, workload)
    payload: dict[str, object] = {
        "algorithm": FINGERPRINT_ALGORITHM,
        "metric": definition.identifier,
        "definition_version": definition.definition_version,
        "data": None if data is None else data.fingerprint_component(),
    }
    # Absent rather than null for a metric with no workload, so adding this
    # component leaves every existing series fingerprint bit-identical.
    if workload is not None:
        payload["workload"] = workload.fingerprint_component()
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


def _validate_workload_component(
    definition: MetricDefinition,
    workload: WorkloadComponent | None,
) -> None:
    if definition.execution_sensitive:
        if workload is None:
            raise FingerprintError(
                f"metric {definition.identifier!r} is execution-sensitive and "
                "needs a workload component; without one, two different "
                "measurements would share a series"
            )
        return
    if workload is not None:
        raise FingerprintError(
            f"metric {definition.identifier!r} is not execution-sensitive and "
            "must carry no workload component"
        )


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_canonical_default,
        ).encode()
    # A ``ValueError`` subclass, so the clause below would otherwise re-wrap
    # what ``_canonical_default`` already refused by name.
    except FingerprintError:
        raise
    except (RecursionError, TypeError, ValueError) as error:
        # A declared workload carries settings straight from configuration, and
        # a `--set` override can make a float one non-finite.
        raise FingerprintError(f"cannot digest these inputs: {error}") from error


def _canonical_digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _canonical_default(value: object) -> object:
    """Normalize container types Parquet readers may hand back."""

    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, bytes):
        return value.hex()
    raise FingerprintError(f"cannot digest a value of type {type(value).__name__}")
