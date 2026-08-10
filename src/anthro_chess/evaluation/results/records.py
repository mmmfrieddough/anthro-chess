"""The artifact envelope every benchmark writes and every report reads.

One envelope carries checkpoint, configuration, dataset, action, encoding,
environment, and benchmark provenance, so a new benchmark kind is a new
``kind`` string rather than a schema change. Each envelope records enough to
recompute its own fingerprints, which is what lets a reader verify a series
without the environment that produced it.

The envelope is the summary tier and is committed to the repository. It holds
scalar headline measurements only; bulk diagnostics live in the machine-local
detail tier behind a reference. That boundary is enforced here rather than
left to convention, because the natural pressure is always to commit one more
useful diagnostic.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from anthro_chess.chess import action_vocabulary_identity
from anthro_chess.data import encoding_identity
from anthro_chess.evaluation.results.fingerprints import (
    DataComponent,
    FingerprintError,
    WorkloadComponent,
    series_fingerprint,
    workload_digest,
)
from anthro_chess.evaluation.results.metrics import (
    MetricRegistryError,
    metric_definition,
)
from anthro_chess.provenance import code_provenance, environment_provenance

#: Version 6 carries a measurement's own dispersion in place of a floor built
#: from it, so a delta is floored by combining the two readings it compares.
#: Version 5 records the training identity a training noise floor is scoped to.
#: Version 4 names the estimator behind a stored noise floor.
ENVELOPE_VERSION = 6
BRIDGE_VERSION = 1

#: Cap on one committed summary record. Generous for scalar headlines and far
#: too small for per-position diagnostics, which is the point: the tier
#: boundary fails loudly instead of eroding.
MAXIMUM_SUMMARY_BYTES = 64 * 1024

Sha256Hex = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")]

#: How a noise floor was characterized. Conflating these is the usual mistake,
#: so a stored floor has to say which one it is. ``execution`` is the machine's
#: own contribution — scheduler contention, thermal state, allocator and kernel
#: warmth — which no resampling of an already-computed number can estimate.
NoiseFloorKind = Literal["evaluation", "data-sampling", "training", "execution"]


class ResultRecordError(ValueError):
    """Raised when a result record violates the store's contract."""


class ResultModel(BaseModel):
    """Base for immutable, code-owned result records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        # A freeform field carries whatever a benchmark put in it, and pydantic
        # renders a non-finite float there as ``null`` by default — so a record
        # would commit, and digest, a value its configuration never held.
        # Keeping the float is what lets :func:`canonical_json` refuse it.
        ser_json_inf_nan="constants",
    )

    def as_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record written to the store.

        A record that cannot be rendered is a recording failure like any other,
        so it is raised as one rather than as the serializer's own error, which
        no benchmark declares and which therefore ends a whole sweep.
        """

        try:
            return self.model_dump(mode="json")
        except ValueError as error:
            # ``PydanticSerializationError``, which pydantic does not re-export.
            raise ResultRecordError(
                f"cannot serialize {type(self).__name__}: {error}"
            ) from error


class BenchmarkReference(ResultModel):
    """Which benchmark produced a result, and at which implementation version."""

    name: Identifier
    version: int = Field(ge=1)


class CheckpointReference(ResultModel):
    """Which model a result describes."""

    label: Identifier
    step: int | None = Field(default=None, ge=0)
    run_id: str | None = Field(default=None, min_length=1)
    parameter_sha256: Sha256Hex | None = None
    #: The training configuration that decided these weights, digested without
    #: the initialization seed. ``parameter_sha256`` names one model and this
    #: names the set of models a seed could have produced, which is the scope a
    #: training noise floor describes and the only thing that stops one from
    #: qualifying a delta between configurations it never measured. Absent on a
    #: reading recorded before the identity existed, and on one whose runner was
    #: supplied rather than loaded from a checkpoint.
    training_sha256: Sha256Hex | None = None


class ConfigurationReference(ResultModel):
    """How a benchmark was configured.

    Only a digest and the selection provenance are committed. Configuration
    text is deliberately absent from fingerprints, so recording it in full
    would grow the committed tier without making any series more identifiable.
    """

    sha256: Sha256Hex
    source: str | None = None
    overrides: tuple[str, ...] = ()


class ProjectionDigest(ResultModel):
    """A content digest over one projection of the games a benchmark scored."""

    projection: Identifier
    projection_version: int = Field(ge=1)
    content_sha256: Sha256Hex
    games: int = Field(ge=1)

    def as_component(self) -> DataComponent:
        """Return the fingerprint component this digest stands for."""

        return DataComponent(
            projection=self.projection,
            projection_version=self.projection_version,
            content_sha256=self.content_sha256,
            games=self.games,
        )

    @classmethod
    def from_component(cls, component: DataComponent) -> ProjectionDigest:
        """Return the stored record for a computed data component."""

        return cls(
            projection=component.projection,
            projection_version=component.projection_version,
            content_sha256=component.content_sha256,
            games=component.games,
        )


class DatasetReference(ResultModel):
    """Which evaluation inputs a result was computed over."""

    pool_id: Identifier
    pool_version: int = Field(ge=1)
    view: Identifier
    selected_games: int = Field(ge=1)
    game_ids_sha256: Sha256Hex
    components: tuple[ProjectionDigest, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_components(self) -> DatasetReference:
        names = [component.projection for component in self.components]
        if len(set(names)) != len(names):
            raise ValueError("a dataset reference may digest each projection once")
        if names != sorted(names):
            raise ValueError("dataset components must be ordered by projection name")
        return self

    def component(self, projection: str) -> ProjectionDigest | None:
        """Return the digest for one projection, if this dataset carries it."""

        for component in self.components:
            if component.projection == projection:
                return component
        return None


class EnvironmentRecord(ResultModel):
    """Which build and machine produced a result."""

    package_version: str = Field(min_length=1)
    git_revision: str | None = None
    python_version: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    dependencies: dict[str, str | None] = Field(default_factory=dict)

    @classmethod
    def capture(cls) -> EnvironmentRecord:
        """Capture the current code and environment provenance."""

        code = code_provenance()
        environment = environment_provenance()
        return cls(
            package_version=code.package_version,
            git_revision=code.git_revision,
            python_version=environment.python_version,
            platform=environment.platform,
            dependencies=dict(environment.dependencies),
        )


#: Execution fields a report compares to decide whether the environment moved.
#: ``platform`` is deliberately absent: it carries the full OS version string,
#: so including it would mark every delta as confounded after an operating
#: system patch that changed no hardware. ``platform_key`` carries the part
#: that matters.
ENVIRONMENT_FIELDS: tuple[str, ...] = (
    "device",
    "device_name",
    "precision",
    "torch_version",
    "platform_key",
    "cpu_threads",
)


class ExecutionRecord(ResultModel):
    """The conditions a workload-scoped result was measured under.

    Three parts with different jobs, and the split between the first two is the
    whole design.

    The **workload** says what was timed and is part of series identity, so a
    reader can recompute an efficiency series fingerprint from this record
    alone. Only settings that make a delta *meaningless* belong here.

    The **coordinates** are settings that change the number without changing
    what it measures. A bigger model trains fewer positions per second, and
    that difference is the answer to a question rather than a category error,
    so it must stay subtractable. Coordinates are recorded and diffed, never
    digested.

    The **environment** — device, precision, Torch version, platform — is
    coordinates too, kept as named fields because every efficiency benchmark
    has the same ones.

    Both mappings are kept in full beside the digest, because "why is this
    slower" is unanswerable from a hash, and because they are a handful of
    scalars rather than a diagnostic payload.
    """

    device: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    precision: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    #: The coarse machine identity, such as ``Darwin-arm64``. This is what an
    #: environment comparison keys on.
    platform_key: str = Field(min_length=1)
    #: The full platform string, kept as provenance for the reader who needs to
    #: know the exact OS build.
    platform: str = Field(min_length=1)
    cpu_threads: int | None = Field(default=None, ge=1)
    workload: dict[str, Any] = Field(default_factory=dict)
    workload_sha256: Sha256Hex
    #: Deliberately absent from every digest. A benchmark whose conditions are
    #: worth comparing across puts them here rather than in ``workload``.
    coordinates: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_workload_digest(self) -> ExecutionRecord:
        """Keep the readable workload and the digest that identifies it agreed.

        The digest is what series identity is built from and the mapping is
        what a reader consults. Letting them drift would mean a record whose
        stated workload is not the one its series was named for.
        """

        if workload_digest(self.workload) != self.workload_sha256:
            raise ValueError(
                "the recorded workload does not produce the recorded digest"
            )
        return self

    def workload_component(self) -> WorkloadComponent:
        """Return the fingerprint component this record's workload stands for."""

        return WorkloadComponent(sha256=self.workload_sha256)

    def environment(self) -> dict[str, str | None]:
        """Return the machine coordinates a report attributes a delta to."""

        return {
            field: _environment_value(getattr(self, field))
            for field in ENVIRONMENT_FIELDS
        }

    def declared_coordinates(self) -> dict[str, str | None]:
        """Return the benchmark-declared coordinates, rendered comparably."""

        return {
            key: _environment_value(value)
            for key, value in sorted(self.coordinates.items())
        }

    def environment_label(self) -> str:
        """Return a short human label for where this ran.

        The Torch version is included because it is a coordinate an
        optimization actually varies. A label showing only the machine would
        render two sides of a software comparison identically, which is
        precisely the comparison this label most often heads.
        """

        return f"{self.device_name} ({self.device}, torch {self.torch_version})"


class NoiseFloor(ResultModel):
    """How large a delta has to be before it is a finding rather than noise."""

    value: float = Field(ge=0.0)
    kind: NoiseFloorKind
    source: str | None = Field(default=None, min_length=1)
    #: Which estimator produced the value, named rather than described. One
    #: kind can be estimated more than one way, and the ways are not
    #: interchangeable: a data-sampling floor bootstrapped over one reading's
    #: own games and one bootstrapped over two checkpoints' paired differences
    #: answer different questions and differ by a known factor. ``source``
    #: carries that in prose for a reader; this carries it for a reader who has
    #: to tell the two apart without parsing a sentence.
    estimator: Identifier | None = None


class MetricDispersion(ResultModel):
    """How far one reading's own units move the metric it reports.

    A reading stores its spread rather than a floor built from it. A floor
    computed inside one reading has to assume the other operand's spread equals
    it, and the two committed readings of one metric have differed by two orders
    of magnitude; combining the two dispersions in front of a delta is the same
    arithmetic without that assumption.
    ``docs/decisions/0043-a-delta-floor-is-combined-from-the-two-readings-it-compares.md``
    owns the design.
    """

    value: float = Field(ge=0.0)
    #: The conservative upper limit on ``value`` a floor is combined from, per
    #: decision 0026. Bounded here rather than at comparison time because the
    #: replicates behind the estimate are the reading's own, and stored beside
    #: the estimate so how much of a wide floor is spread and how much is
    #: ignorance stays visible.
    bound: float = Field(ge=0.0)
    #: How many independent sampling units the spread was read over, when it
    #: scales with that count. Set where the units are the games a bootstrap
    #: resampled, which is what makes the sizing question computable; absent
    #: where re-measuring does not redraw a sample, and deliberately not the
    #: measurement's own ``sample_size``, which counts scored positions.
    units: int | None = Field(default=None, ge=1)
    kind: NoiseFloorKind
    source: str | None = Field(default=None, min_length=1)
    estimator: Identifier | None = None

    @model_validator(mode="after")
    def _validate_bound(self) -> MetricDispersion:
        if not math.isfinite(self.value) or not math.isfinite(self.bound):
            raise ValueError("a dispersion and its bound must be finite numbers")
        if self.bound < self.value:
            raise ValueError(
                "a dispersion bound below the dispersion it bounds is not a "
                "conservative limit"
            )
        return self


class Measurement(ResultModel):
    """One metric value and the series it belongs to."""

    metric: str = Field(min_length=1)
    value: float
    fingerprint: Sha256Hex
    sample_size: int | None = Field(default=None, ge=1)
    dispersion: MetricDispersion | None = None

    @model_validator(mode="after")
    def _validate_value(self) -> Measurement:
        if not math.isfinite(self.value):
            raise ValueError(f"measurement {self.metric} must be a finite number")
        return self


class DetailReference(ResultModel):
    """Where the machine-local bulk diagnostics for a result live."""

    path: str = Field(min_length=1)
    sha256: Sha256Hex
    bytes: int = Field(ge=0)
    description: str | None = Field(default=None, min_length=1)


class ResultEnvelope(ResultModel):
    """One benchmark result, in the shape every consumer reads."""

    envelope_version: int = Field(ge=1)
    result_id: str = Field(pattern=r"^[a-f0-9]{16}$")
    recorded_at: datetime
    kind: Identifier
    benchmark: BenchmarkReference
    checkpoint: CheckpointReference
    configuration: ConfigurationReference | None = None
    data: DatasetReference | None = None
    action_vocabulary: dict[str, Any]
    encoding: dict[str, Any]
    environment: EnvironmentRecord
    #: Present only on results whose declared settings are inputs to their
    #: value: efficiency and generated play. ``environment`` describes how any
    #: result was produced; this describes what such a result measured.
    execution: ExecutionRecord | None = None
    measurements: tuple[Measurement, ...] = Field(min_length=1)
    detail: DetailReference | None = None

    @model_validator(mode="after")
    def _validate_measurements(self) -> ResultEnvelope:
        metrics = [measurement.metric for measurement in self.measurements]
        if len(set(metrics)) != len(metrics):
            raise ValueError("a result may report each metric once")
        if metrics != sorted(metrics):
            raise ValueError("measurements must be ordered by metric identifier")
        if self.recorded_at.tzinfo is None:
            raise ValueError("recorded_at must carry a time zone")
        return self

    def verify(self, *, recording: bool = True) -> None:
        """Recompute every fingerprint from the provenance the result carries.

        A result that cannot reproduce its own series identity is unusable for
        comparison, so this is checked rather than trusted.

        Reading is deliberately more forgiving than recording. A metric that
        has since left the registry leaves a dead series, and decision 0013
        expects those to stay readable and honestly labeled rather than to
        make the surrounding history unloadable. The same applies to the size
        budget, which can only be exceeded by a record written when the budget
        was larger, and to the serialization it is measured over: ``json.loads``
        accepts literals the canonical writer refuses, so re-encoding on the way
        in would let one stored record make the whole history unreadable.
        """

        if self.envelope_version > ENVELOPE_VERSION:
            raise ResultRecordError(
                f"result {self.result_id} uses envelope version "
                f"{self.envelope_version}; this build understands "
                f"{ENVELOPE_VERSION}"
            )
        for measurement in self.measurements:
            try:
                metric_definition(measurement.metric)
            except MetricRegistryError as error:
                if recording:
                    raise ResultRecordError(str(error)) from error
                continue
            expected = self.expected_fingerprint(measurement.metric)
            if expected != measurement.fingerprint:
                raise ResultRecordError(
                    f"result {self.result_id} records a fingerprint for "
                    f"{measurement.metric} that its own provenance does not "
                    "reproduce"
                )
        if not recording:
            return
        size = len(canonical_json(self.as_record()))
        if size > MAXIMUM_SUMMARY_BYTES:
            raise ResultRecordError(
                f"result {self.result_id} is {size} bytes; the committed "
                f"summary tier caps a record at {MAXIMUM_SUMMARY_BYTES}. Move "
                "bulk diagnostics to the machine-local detail tier."
            )

    def expected_fingerprint(self, metric: str) -> str:
        """Return the fingerprint this result's own provenance implies."""

        try:
            definition = metric_definition(metric)
        except MetricRegistryError as error:
            raise ResultRecordError(str(error)) from error
        component: DataComponent | None = None
        if definition.projection is not None:
            digest = self.data.component(definition.projection) if self.data else None
            if digest is None:
                raise ResultRecordError(
                    f"result {self.result_id} reports {metric}, which consumes "
                    f"projection {definition.projection!r}, without a matching "
                    "content digest"
                )
            component = digest.as_component()
        workload: WorkloadComponent | None = None
        if definition.execution_sensitive:
            if self.execution is None:
                raise ResultRecordError(
                    f"result {self.result_id} reports {metric}, which is "
                    "execution-sensitive, without recording the execution it "
                    "was measured under"
                )
            workload = self.execution.workload_component()
        try:
            return series_fingerprint(definition, component, workload)
        except FingerprintError as error:
            raise ResultRecordError(str(error)) from error

    def measurement(self, metric: str) -> Measurement | None:
        """Return one measurement by metric identifier."""

        for measurement in self.measurements:
            if measurement.metric == metric:
                return measurement
        return None


class Bridge(ResultModel):
    """An explicit assertion that two fingerprints name the same series.

    Breaking a series is automatic; rejoining one is not. A bridge is
    legitimate only when the fingerprint moved for a reason provably
    independent of the measured quantity. It records who asserted that and
    why, it is stored with the results, and revoking it is a reviewable diff.
    """

    bridge_version: int = Field(ge=1)
    bridge_id: str = Field(pattern=r"^[a-f0-9]{16}$")
    recorded_at: datetime
    from_fingerprint: Sha256Hex
    to_fingerprint: Sha256Hex
    reason: str = Field(min_length=1)
    author: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_endpoints(self) -> Bridge:
        if self.from_fingerprint == self.to_fingerprint:
            raise ValueError("a bridge must join two different fingerprints")
        if self.recorded_at.tzinfo is None:
            raise ValueError("recorded_at must carry a time zone")
        return self


def default_checkpoint_label(run_id: str, global_step: int) -> str:
    """Return the conventional label for one checkpoint of one run.

    Every reading of the same checkpoint has to agree on this, or an
    in-training preview and the later canonical evaluation of the same
    parameters would appear in a report as two unrelated checkpoints.
    """

    slug = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in run_id.lower()
    ).strip("-")
    prefix = slug or "run"
    if not prefix[0].isalnum():
        prefix = f"run-{prefix}"
    return f"{prefix}-step-{global_step:08d}"


def canonical_json(value: Any) -> bytes:
    """Serialize a record the one way the store and its digests agree on.

    What the serializer refuses is raised as a record error, which
    :meth:`ResultRecording.__exit__` and :meth:`ResultsStore.append` already
    convert into the running benchmark's own. A bare ``ValueError`` from here
    is declared by nothing and ends the sweep instead of the step, and it
    arrives after the benchmark has finished measuring.
    """

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (RecursionError, TypeError, ValueError) as error:
        raise ResultRecordError(f"cannot serialize a record: {error}") from error


def build_result(
    *,
    kind: str,
    benchmark: BenchmarkReference,
    checkpoint: CheckpointReference,
    measurements: Sequence[Measurement],
    configuration: ConfigurationReference | None = None,
    data: DatasetReference | None = None,
    detail: DetailReference | None = None,
    environment: EnvironmentRecord | None = None,
    execution: ExecutionRecord | None = None,
    recorded_at: datetime | None = None,
) -> ResultEnvelope:
    """Assemble a verified result envelope with a content-derived identity.

    The identity is derived from the record itself, so recording the same
    result twice is idempotent rather than a duplicate history entry.
    """

    ordered = tuple(sorted(measurements, key=lambda item: item.metric))
    envelope = ResultEnvelope(
        envelope_version=ENVELOPE_VERSION,
        result_id="0" * 16,
        recorded_at=recorded_at or datetime.now(tz=UTC),
        kind=kind,
        benchmark=benchmark,
        checkpoint=checkpoint,
        configuration=configuration,
        data=data,
        action_vocabulary=action_vocabulary_identity(),
        encoding=encoding_identity(),
        environment=environment or EnvironmentRecord.capture(),
        execution=execution,
        measurements=ordered,
        detail=detail,
    )
    identified = envelope.model_copy(update={"result_id": _record_id(envelope)})
    identified.verify()
    return identified


def build_bridge(
    *,
    from_fingerprint: str,
    to_fingerprint: str,
    reason: str,
    author: str,
    recorded_at: datetime | None = None,
) -> Bridge:
    """Assemble a bridge with a content-derived identity."""

    bridge = Bridge(
        bridge_version=BRIDGE_VERSION,
        bridge_id="0" * 16,
        recorded_at=recorded_at or datetime.now(tz=UTC),
        from_fingerprint=from_fingerprint,
        to_fingerprint=to_fingerprint,
        reason=reason,
        author=author,
    )
    return bridge.model_copy(update={"bridge_id": _record_id(bridge)})


def measurement(
    metric: str,
    value: float,
    *,
    data: DataComponent | None = None,
    workload: WorkloadComponent | None = None,
    sample_size: int | None = None,
    dispersion: MetricDispersion | None = None,
) -> Measurement:
    """Return one measurement with its fingerprint computed from the registry."""

    try:
        definition = metric_definition(metric)
        fingerprint = series_fingerprint(definition, data, workload)
    except (MetricRegistryError, FingerprintError) as error:
        raise ResultRecordError(str(error)) from error
    return Measurement(
        metric=definition.identifier,
        value=value,
        fingerprint=fingerprint,
        sample_size=sample_size,
        dispersion=dispersion,
    )


def dataset_reference(
    *,
    pool_id: str,
    pool_version: int,
    view: str,
    selected_games: int,
    game_ids_sha256: str,
    components: Sequence[DataComponent],
) -> DatasetReference:
    """Return the dataset reference for one benchmark's realized inputs."""

    digests = sorted(
        (ProjectionDigest.from_component(component) for component in components),
        key=lambda digest: digest.projection,
    )
    return DatasetReference(
        pool_id=pool_id,
        pool_version=pool_version,
        view=view,
        selected_games=selected_games,
        game_ids_sha256=game_ids_sha256,
        components=tuple(digests),
    )


def execution_reference(
    *,
    device: str,
    device_name: str,
    precision: str,
    torch_version: str,
    platform_key: str,
    platform: str,
    workload: Mapping[str, Any],
    coordinates: Mapping[str, Any] | None = None,
    cpu_threads: int | None = None,
) -> ExecutionRecord:
    """Return the execution record for one efficiency benchmark's conditions.

    The workload digest is computed here rather than by each caller, so two
    benchmarks declaring the same workload cannot end up on different series
    through a difference in how they hashed it. ``coordinates`` never reaches
    the digest, which is the point of passing it separately.
    """

    try:
        digest = workload_digest(workload)
    except FingerprintError as error:
        raise ResultRecordError(str(error)) from error
    return ExecutionRecord(
        device=device,
        device_name=device_name,
        precision=precision,
        torch_version=torch_version,
        platform_key=platform_key,
        platform=platform,
        cpu_threads=cpu_threads,
        workload=dict(workload),
        workload_sha256=digest,
        coordinates=dict(coordinates or {}),
    )


def configuration_reference(
    resolved: Mapping[str, Any],
    *,
    source: str | None = None,
    overrides: Sequence[str] = (),
) -> ConfigurationReference:
    """Digest a resolved configuration record for the summary tier."""

    return ConfigurationReference(
        sha256=sha256(canonical_json(dict(resolved))).hexdigest(),
        source=source,
        overrides=tuple(overrides),
    )


def _environment_value(value: object) -> str | None:
    """Render one environment coordinate as a comparable string."""

    return None if value is None else str(value)


def _record_id(record: ResultModel) -> str:
    payload = record.as_record()
    payload.pop("result_id", None)
    payload.pop("bridge_id", None)
    return sha256(canonical_json(payload)).hexdigest()[:16]
