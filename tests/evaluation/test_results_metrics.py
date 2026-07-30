"""Metric identity is a contract, so the registry defends it."""

from __future__ import annotations

from typing import Any, cast

import pytest

from anthro_chess.evaluation.results import (
    NOMINAL_REPEATED_PASSES,
    DataProjection,
    MetricCost,
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
from anthro_chess.evaluation.results.reporting import METRIC_COLUMN_WIDTH


def _definition(**overrides: object) -> MetricDefinition:
    fields: dict[str, object] = {
        "identifier": "legality.probe",
        "family": "legality",
        "direction": MetricDirection.LOWER_IS_BETTER,
        "definition_version": 1,
        "summary": "A probe metric.",
        "cost": MetricCost.FREE,
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


def test_every_registered_metric_prices_itself_against_its_data_dependency() -> None:
    """A schedule can only reject an unaffordable pairing if cost is declared."""

    for metric in registered_metrics():
        assert metric.cost in set(MetricCost)
        assert metric.cost.reads_data == (metric.projection is not None)


def test_a_cost_that_contradicts_the_data_dependency_is_refused() -> None:
    with pytest.raises(MetricRegistryError, match="reads no data"):
        register_metric(
            _definition(
                identifier="legality.mispriced",
                cost=MetricCost.FREE,
                projection="move_prediction",
            )
        )
    with pytest.raises(MetricRegistryError, match="has to name the projection"):
        register_metric(
            _definition(
                identifier="training_health.mispriced",
                family="training-health",
                cost=MetricCost.SINGLE_PASS,
                projection=None,
            )
        )


def test_an_execution_sensitive_metric_must_measure_execution() -> None:
    """Only a timed measurement may claim the device as a realized input.

    Otherwise a quality metric could acquire a machine dependency, which would
    end its series on every move between machines for no reason.
    """

    with pytest.raises(MetricRegistryError, match="execution-sensitive"):
        register_metric(
            _definition(
                identifier="held_out.mislabelled",
                cost=MetricCost.SINGLE_PASS,
                projection="move_prediction",
                execution_sensitive=True,
            )
        )


def test_a_free_metric_costs_no_view_passes_and_a_generated_one_is_unbounded() -> None:
    assert MetricCost.FREE.view_passes == 0
    assert MetricCost.SINGLE_PASS.view_passes == 1
    assert MetricCost.REPEATED_PASS.view_passes == NOMINAL_REPEATED_PASSES
    assert MetricCost.GENERATED.view_passes is None
    assert MetricCost.MEASURED_EXECUTION.view_passes == 0


def test_measured_execution_is_not_free_despite_passing_over_no_view() -> None:
    """Both read no data; only one is derived from work already done."""

    assert not MetricCost.MEASURED_EXECUTION.reads_data
    assert not MetricCost.FREE.reads_data
    assert MetricCost.MEASURED_EXECUTION is not MetricCost.FREE


def test_the_registry_covers_the_families_reports_read() -> None:
    identifiers = {family.identifier for family in registered_families()}

    assert {
        "training-health",
        "legality",
        "rating-behavior",
        "generated-play",
        "decision-decomposition",
    } <= identifiers


def test_decision_decomposition_metrics_declare_no_direction() -> None:
    """Every one of them moves with temperature, which is a trade, not a gain."""

    metrics = registered_metrics("decision-decomposition")

    assert metrics
    for metric in metrics:
        assert metric.direction is MetricDirection.INFORMATIONAL
        assert metric.cost is MetricCost.GENERATED
        assert metric.cost.view_passes is None


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
    assert metrics["legality.mask_penalty"]["cost"] == "single_pass"
    projections = cast(list[dict[str, Any]], record["projections"])
    assert {projection["name"] for projection in projections} >= {
        "move_prediction",
        "move_timing",
    }


def test_inference_metric_identifiers_fit_the_report_column() -> None:
    """A longer identifier pushes its whole row out of alignment.

    Scoped to this family rather than the registry: four dependency and
    legality identifiers already overflow the column, which is a pre-existing
    rendering defect rather than something this test should adopt.
    """

    too_long = [
        metric.identifier
        for metric in registered_metrics("inference-efficiency")
        if len(metric.identifier) > METRIC_COLUMN_WIDTH
    ]

    assert not too_long
