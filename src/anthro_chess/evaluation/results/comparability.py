"""The one place that decides whether two results may be compared.

Comparability is a property of a series, not of the project. Fingerprint
mismatch breaks a series automatically; a recorded bridge may rejoin two
fingerprints that moved for a reason provably independent of the measured
quantity. Every consumer asks these functions rather than reimplementing the
rule, so a report, a chart, and a regression gate cannot disagree.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from anthro_chess.evaluation.results.records import (
    Bridge,
    Measurement,
    ResultEnvelope,
)


class Comparability(StrEnum):
    """Whether two measurements sit on the same line."""

    SAME_SERIES = "same_series"
    BRIDGED = "bridged"
    INCOMPARABLE = "incomparable"


@dataclass(frozen=True)
class SeriesComparison:
    """The verdict for one pair of measurements, with its supporting bridges."""

    comparability: Comparability
    bridges: tuple[Bridge, ...] = ()

    @property
    def is_comparable(self) -> bool:
        """Return whether a delta between the two results means anything."""

        return self.comparability is not Comparability.INCOMPARABLE


class BridgeIndex:
    """Resolve fingerprints into series, following recorded bridges.

    Bridging is an equivalence: asserting that two fingerprints name the same
    series is symmetric, and a chain of bridges joins its whole chain. Anything
    weaker would make a report's answer depend on which of two equivalent
    results happened to be the baseline.
    """

    def __init__(self, bridges: Iterable[Bridge] = ()) -> None:
        self._bridges = tuple(
            sorted(bridges, key=lambda bridge: (bridge.recorded_at, bridge.bridge_id))
        )
        self._parent: dict[str, str] = {}
        for bridge in self._bridges:
            self._union(bridge.from_fingerprint, bridge.to_fingerprint)

    @property
    def bridges(self) -> tuple[Bridge, ...]:
        """Return the indexed bridges in recording order."""

        return self._bridges

    def series(self, fingerprint: str) -> str:
        """Return the canonical fingerprint for the series of a fingerprint."""

        return self._find(fingerprint)

    def compare(self, baseline: str, current: str) -> SeriesComparison:
        """Return whether two fingerprints may be compared, and on what basis."""

        if baseline == current:
            return SeriesComparison(Comparability.SAME_SERIES)
        if self._find(baseline) != self._find(current):
            return SeriesComparison(Comparability.INCOMPARABLE)
        return SeriesComparison(
            Comparability.BRIDGED,
            self._supporting_bridges(baseline, current),
        )

    def compare_measurements(
        self,
        baseline: Measurement,
        current: Measurement,
    ) -> SeriesComparison:
        """Return the verdict for two recorded measurements of one metric."""

        if baseline.metric != current.metric:
            return SeriesComparison(Comparability.INCOMPARABLE)
        return self.compare(baseline.fingerprint, current.fingerprint)

    def _supporting_bridges(self, baseline: str, current: str) -> tuple[Bridge, ...]:
        """Return the shortest chain of bridges joining two fingerprints."""

        neighbors: dict[str, list[Bridge]] = {}
        for bridge in self._bridges:
            for endpoint in (bridge.from_fingerprint, bridge.to_fingerprint):
                neighbors.setdefault(endpoint, []).append(bridge)

        queue: deque[tuple[str, tuple[Bridge, ...]]] = deque([(baseline, ())])
        visited = {baseline}
        while queue:
            fingerprint, chain = queue.popleft()
            if fingerprint == current:
                return chain
            for bridge in neighbors.get(fingerprint, []):
                other = (
                    bridge.to_fingerprint
                    if bridge.from_fingerprint == fingerprint
                    else bridge.from_fingerprint
                )
                if other not in visited:
                    visited.add(other)
                    queue.append((other, (*chain, bridge)))
        return ()  # pragma: no cover - the union-find already proved a path

    def _find(self, fingerprint: str) -> str:
        parent = self._parent.get(fingerprint, fingerprint)
        if parent == fingerprint:
            return fingerprint
        root = self._find(parent)
        self._parent[fingerprint] = root
        return root

    def _union(self, left: str, right: str) -> None:
        left_root = self._find(left)
        right_root = self._find(right)
        if left_root == right_root:
            return
        # Order the representative so a series name does not depend on the
        # order bridges happened to be recorded in.
        low, high = sorted((left_root, right_root))
        self._parent[high] = low


@dataclass(frozen=True)
class ProvenanceDifference:
    """One recorded difference between the provenance of two results."""

    field: str
    baseline: str | None
    current: str | None


def provenance_differences(
    baseline: ResultEnvelope,
    current: ResultEnvelope,
) -> tuple[ProvenanceDifference, ...]:
    """Report how two results were produced differently.

    None of these break a series on their own. A changed package version or a
    changed encoding says the model or the code moved, which is usually the
    point of the comparison; they are reported so a surprising delta has
    somewhere to be explained from.
    """

    differences: list[ProvenanceDifference] = []
    for field, baseline_value, current_value in (
        ("benchmark", _benchmark(baseline), _benchmark(current)),
        (
            "package_version",
            baseline.environment.package_version,
            current.environment.package_version,
        ),
        (
            "git_revision",
            baseline.environment.git_revision,
            current.environment.git_revision,
        ),
        ("platform", baseline.environment.platform, current.environment.platform),
        ("encoding", _encoding(baseline), _encoding(current)),
        (
            "action_vocabulary",
            _action_vocabulary(baseline),
            _action_vocabulary(current),
        ),
        ("dataset", _dataset(baseline), _dataset(current)),
        ("execution", _execution(baseline), _execution(current)),
        ("configuration", _configuration(baseline), _configuration(current)),
    ):
        if baseline_value != current_value:
            differences.append(
                ProvenanceDifference(
                    field=field,
                    baseline=baseline_value,
                    current=current_value,
                )
            )
    return tuple(differences)


class AxisChange(StrEnum):
    """Whether one coordinate of a comparison moved between two results."""

    UNCHANGED = "unchanged"
    CHANGED = "changed"
    #: Neither result recorded enough to say. Distinct from unchanged, because
    #: an unverifiable claim of sameness is what an attribution must not make.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Attribution:
    """Which of a comparison's coordinates moved.

    An efficiency delta is only a statement about the model when everything
    else held still. Reporting the axes separately is what lets a reader — or
    an agent — tell "my change made this slower" from "I measured it on a
    different machine" or "the corpus was regenerated underneath it".

    ``conditions`` covers benchmark-declared coordinates such as the model
    architecture, the effective batch, and the training corpus. They move the
    number without changing what it measures, so they confound a comparison
    rather than invalidating it.
    """

    model: AxisChange
    environment: AxisChange
    workload: AxisChange
    conditions: AxisChange = AxisChange.UNKNOWN

    @property
    def attributable_to_model(self) -> bool:
        """Return whether the delta can be read as a property of the model."""

        return (
            self.environment is AxisChange.UNCHANGED
            and self.workload is AxisChange.UNCHANGED
            and self.conditions is not AxisChange.CHANGED
        )

    def as_record(self) -> dict[str, str]:
        """Return the machine-readable attribution."""

        return {
            "model": self.model.value,
            "environment": self.environment.value,
            "workload": self.workload.value,
            "conditions": self.conditions.value,
        }


def attribute(baseline: ResultEnvelope, current: ResultEnvelope) -> Attribution:
    """Return which coordinates moved between two results."""

    return Attribution(
        model=_axis(
            baseline.checkpoint.parameter_sha256,
            current.checkpoint.parameter_sha256,
        ),
        environment=_environment_axis(baseline, current),
        workload=_axis(
            None if baseline.execution is None else baseline.execution.workload_sha256,
            None if current.execution is None else current.execution.workload_sha256,
        ),
        conditions=_conditions_axis(baseline, current),
    )


def training_axis(baseline: ResultEnvelope, current: ResultEnvelope) -> AxisChange:
    """Return whether the training configuration behind the two models moved.

    Asked of every comparison rather than only of an efficiency one, unlike the
    axes :func:`attribute` reports. A corpus or a hyperparameter that moved
    between two checkpoints confounds a quality delta exactly as it confounds a
    timing one, and the fingerprint that breaks a series covers the measurement
    side alone.

    The digest says only that something moved, never how much, so a match is
    the verifiable half: it says nothing on the training side moved at all.
    """

    return _axis(
        baseline.checkpoint.training_sha256,
        current.checkpoint.training_sha256,
    )


def condition_differences(
    baseline: ResultEnvelope,
    current: ResultEnvelope,
) -> tuple[ProvenanceDifference, ...]:
    """Report which declared coordinates differ between two results.

    These are the settings a benchmark chose to keep out of series identity
    because a delta across them is interesting rather than meaningless. Naming
    them is what stops a corpus regeneration from reading as a slowdown.
    """

    if baseline.execution is None or current.execution is None:
        return ()
    before = baseline.execution.declared_coordinates()
    after = current.execution.declared_coordinates()
    return tuple(
        ProvenanceDifference(
            field=field,
            baseline=before.get(field),
            current=after.get(field),
        )
        for field in sorted({*before, *after})
        if before.get(field) != after.get(field)
    )


def environment_differences(
    baseline: ResultEnvelope,
    current: ResultEnvelope,
) -> tuple[ProvenanceDifference, ...]:
    """Report which execution coordinates differ between two results.

    Unlike :func:`provenance_differences`, this is scoped to the fields that
    decide an efficiency number, so a report can name the cause of a delta it
    refuses to credit to the model.
    """

    if baseline.execution is None or current.execution is None:
        return ()
    before = baseline.execution.environment()
    after = current.execution.environment()
    return tuple(
        ProvenanceDifference(field=field, baseline=before[field], current=after[field])
        for field in sorted(before)
        if before[field] != after[field]
    )


def _environment_axis(baseline: ResultEnvelope, current: ResultEnvelope) -> AxisChange:
    if baseline.execution is None or current.execution is None:
        return AxisChange.UNKNOWN
    if baseline.execution.environment() != current.execution.environment():
        return AxisChange.CHANGED
    return AxisChange.UNCHANGED


def _conditions_axis(baseline: ResultEnvelope, current: ResultEnvelope) -> AxisChange:
    """Return whether the declared coordinates moved.

    A benchmark that declares none has nothing to say here, which is reported
    as unknown rather than as an unverifiable claim that they held still.
    """

    if baseline.execution is None or current.execution is None:
        return AxisChange.UNKNOWN
    before = baseline.execution.declared_coordinates()
    after = current.execution.declared_coordinates()
    if not before and not after:
        return AxisChange.UNKNOWN
    return AxisChange.CHANGED if before != after else AxisChange.UNCHANGED


def _axis(baseline: str | None, current: str | None) -> AxisChange:
    if baseline is None or current is None:
        return AxisChange.UNKNOWN
    return AxisChange.UNCHANGED if baseline == current else AxisChange.CHANGED


def latest_measurement(
    envelopes: Sequence[ResultEnvelope],
    metric: str,
) -> tuple[ResultEnvelope, Measurement] | None:
    """Return the most recently recorded measurement of one metric.

    Blind to series by construction, so a caller that may hold several results
    for one checkpoint should reach for :func:`measurements_by_workload`
    instead. This stays for the readings where one result per checkpoint is
    guaranteed by the benchmark rather than assumed by the reader.
    """

    ordered = sorted(
        envelopes,
        key=lambda envelope: (envelope.recorded_at, envelope.result_id),
    )
    for envelope in reversed(ordered):
        found = envelope.measurement(metric)
        if found is not None:
            return envelope, found
    return None


#: Group key for a reading whose metric declares no workload. One group, which
#: is the case every benchmark writing a single envelope per checkpoint stays
#: in.
UNSCOPED_WORKLOAD = ""


def measurements_by_workload(
    envelopes: Sequence[ResultEnvelope],
    metric: str,
    *,
    workload_scoped: bool,
) -> dict[str, tuple[ResultEnvelope, Measurement]]:
    """Return one checkpoint's most recent reading of a metric per workload.

    A benchmark that writes one envelope per matrix cell records several
    results for one checkpoint that share a metric identifier and differ only
    by declared workload. Choosing the single most recent one would present one
    arbitrary cell as the checkpoint's value and hide the rest, so the caller
    is handed every cell and decides how to present them.

    ``workload_scoped`` is the metric's property rather than the envelope's. A
    result may carry an execution record while reporting a metric whose value
    does not depend on it — a training-efficiency run also reporting held-out
    quality is the case — and splitting that metric by workload would show two
    rows for readings that are genuinely on one series.
    """

    grouped: dict[str, tuple[ResultEnvelope, Measurement]] = {}
    for envelope in sorted(
        envelopes,
        key=lambda item: (item.recorded_at, item.result_id),
    ):
        found = envelope.measurement(metric)
        if found is None:
            continue
        key = UNSCOPED_WORKLOAD
        if workload_scoped and envelope.execution is not None:
            key = envelope.execution.workload_sha256
        grouped[key] = (envelope, found)
    return grouped


def recorded_series(
    envelopes: Sequence[ResultEnvelope],
    metric: str,
    bridges: BridgeIndex,
) -> frozenset[str]:
    """Return the distinct series one metric was recorded on."""

    return frozenset(
        bridges.series(found.fingerprint)
        for found in (envelope.measurement(metric) for envelope in envelopes)
        if found is not None
    )


def _benchmark(envelope: ResultEnvelope) -> str:
    return f"{envelope.benchmark.name} v{envelope.benchmark.version}"


def _encoding(envelope: ResultEnvelope) -> str:
    return f"{envelope.encoding.get('name')} v{envelope.encoding.get('version')}"


def _action_vocabulary(envelope: ResultEnvelope) -> str:
    identity = envelope.action_vocabulary
    return f"{identity.get('name')} v{identity.get('version')}"


def _dataset(envelope: ResultEnvelope) -> str | None:
    if envelope.data is None:
        return None
    return (
        f"{envelope.data.pool_id} v{envelope.data.pool_version} "
        f"view {envelope.data.view} ({envelope.data.selected_games} games)"
    )


def _execution(envelope: ResultEnvelope) -> str | None:
    """Describe what an efficiency result was measured on."""

    execution = envelope.execution
    if execution is None:
        return None
    return (
        f"{execution.environment_label()} {execution.precision} "
        f"torch {execution.torch_version} "
        f"workload {execution.workload_sha256[:12]}"
    )


def _configuration(envelope: ResultEnvelope) -> str | None:
    if envelope.configuration is None:
        return None
    return envelope.configuration.sha256
