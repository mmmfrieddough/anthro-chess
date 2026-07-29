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


def latest_measurement(
    envelopes: Sequence[ResultEnvelope],
    metric: str,
) -> tuple[ResultEnvelope, Measurement] | None:
    """Return the most recently recorded measurement of one metric."""

    ordered = sorted(
        envelopes,
        key=lambda envelope: (envelope.recorded_at, envelope.result_id),
    )
    for envelope in reversed(ordered):
        found = envelope.measurement(metric)
        if found is not None:
            return envelope, found
    return None


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
    """Describe what an efficiency result was measured on.

    An efficiency series already breaks on a device or workload change,
    because both are in its fingerprint. This is what turns the resulting
    "incomparable" into an explanation.
    """

    execution = envelope.execution
    if execution is None:
        return None
    return (
        f"{execution.device} ({execution.device_name}) {execution.precision} "
        f"workload {execution.workload_sha256[:12]}"
    )


def _configuration(envelope: ResultEnvelope) -> str | None:
    if envelope.configuration is None:
        return None
    return envelope.configuration.sha256
