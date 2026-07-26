"""Metric identity is a contract, so the registry defends it."""

from __future__ import annotations

from typing import Any, cast

import pytest

from anthro_chess.evaluation.results import (
    DataProjection,
    MetricDefinition,
    MetricDirection,
    MetricFamily,
    MetricRegistryError,
    metric_definition,
    register_family,
    register_metric,
    register_projection,
    registered_families,
    registered_metrics,
    registry_record,
    registry_snapshot,
    restore_registry,
)


def _definition(**overrides: object) -> MetricDefinition:
    fields: dict[str, object] = {
        "identifier": "legality.probe",
        "family": "legality",
        "direction": MetricDirection.LOWER_IS_BETTER,
        "definition_version": 1,
        "summary": "A probe metric.",
        "projection": None,
    }
    fields.update(overrides)
    return MetricDefinition(**fields)  # type: ignore[arg-type]


def test_every_registered_metric_declares_a_family_and_a_direction() -> None:
    for metric in registered_metrics():
        assert metric.family in {family.identifier for family in registered_families()}
        assert metric.direction in set(MetricDirection)
        assert metric.definition_version >= 1
        assert metric.summary


def test_the_registry_covers_the_families_reports_read() -> None:
    identifiers = {family.identifier for family in registered_families()}

    assert {
        "training-health",
        "legality",
        "rating-behavior",
        "generated-play",
    } <= identifiers


def test_a_later_family_registers_additively() -> None:
    family = register_family(
        MetricFamily(
            identifier="preference-controls",
            title="Preference controls",
            summary="Whether soft preference sliders behave as tendencies.",
        )
    )

    assert family in registered_families()
    assert registered_metrics(family.identifier) == ()


def test_a_conflicting_redefinition_is_refused() -> None:
    """A changed definition needs a new identity, not an edited one."""

    register_metric(_definition())
    register_metric(_definition())

    with pytest.raises(MetricRegistryError, match="already registered differently"):
        register_metric(_definition(definition_version=2))


def test_an_unknown_family_or_projection_is_refused() -> None:
    with pytest.raises(MetricRegistryError, match="unknown family"):
        register_metric(_definition(family="not-a-family"))
    with pytest.raises(MetricRegistryError, match="unknown projection"):
        register_metric(_definition(projection="not-a-projection"))


def test_a_metric_identifier_is_dotted_lowercase() -> None:
    with pytest.raises(MetricRegistryError, match="dotted lowercase"):
        register_metric(_definition(identifier="Legality Probe"))


def test_a_projection_names_only_normalized_columns() -> None:
    with pytest.raises(MetricRegistryError, match="unknown normalized"):
        register_projection(
            DataProjection(name="probe", version=1, columns=("not_a_column",))
        )
    with pytest.raises(MetricRegistryError, match="sorted and unique"):
        register_projection(
            DataProjection(
                name="probe",
                version=1,
                columns=("ruleset", "action_ids"),
            )
        )


def test_unknown_lookups_are_refused() -> None:
    with pytest.raises(MetricRegistryError, match="unknown metric"):
        metric_definition("legality.nonexistent")
    with pytest.raises(MetricRegistryError, match="unknown metric family"):
        registered_metrics("not-a-family")


def test_a_snapshot_undoes_temporary_registration() -> None:
    snapshot = registry_snapshot()
    register_metric(_definition())
    assert metric_definition("legality.probe")

    restore_registry(snapshot)

    with pytest.raises(MetricRegistryError):
        metric_definition("legality.probe")


def test_the_registry_record_names_every_metric_and_projection() -> None:
    record = registry_record()
    families = {
        family["identifier"]: family
        for family in cast(list[dict[str, Any]], record["families"])
    }

    legality = families["legality"]
    metrics = {metric["identifier"]: metric for metric in legality["metrics"]}
    assert metrics["legality.mask_penalty"]["direction"] == "lower_is_better"
    assert metrics["legality.legal_mass"]["direction"] == "higher_is_better"
    assert metrics["legality.mask_penalty"]["projection"] == "move_prediction"
    projections = cast(list[dict[str, Any]], record["projections"])
    assert {projection["name"] for projection in projections} >= {
        "move_prediction",
        "move_timing",
    }
