"""Fingerprint identity: what breaks a series and what deliberately does not."""

from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from typing import Any

import pytest

from anthro_chess.evaluation.results import (
    FINGERPRINT_ALGORITHM,
    DataComponent,
    ExecutionComponent,
    FingerprintError,
    MetricCost,
    MetricDefinition,
    MetricDirection,
    metric_definition,
    projection_content_digest,
    register_metric,
    series_fingerprint,
    workload_digest,
)
from anthro_chess.evaluation.results.metrics import (
    MOVE_PREDICTION_PROJECTION,
    MOVE_TIMING_PROJECTION,
)

RowFactory = Callable[..., dict[str, Any]]
Digest = Callable[..., DataComponent]


def test_content_digest_ignores_row_order(
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    forward = move_prediction_component(
        [scored_row(1), scored_row(2), scored_row(3)],
    )
    backward = move_prediction_component(
        [scored_row(3), scored_row(1), scored_row(2)],
    )

    assert forward.content_sha256 == backward.content_sha256
    assert forward.games == 3


def test_content_digest_ignores_columns_the_projection_does_not_read(
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    """An unread schema field must not end a series it never touched."""

    baseline = move_prediction_component([scored_row(1), scored_row(2)])
    changed = move_prediction_component(
        [
            scored_row(1, clock_remaining_ms=[1, 2, 3], result="0-1"),
            scored_row(2, clock_remaining_ms=None, result="1/2-1/2"),
        ]
    )

    assert baseline.content_sha256 == changed.content_sha256


def test_content_digest_treats_tuples_and_lists_alike(
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    """Two readers of the same pool must not disagree over container type."""

    listed = move_prediction_component([scored_row(1, action_ids=[4, 5, 6])])
    tupled = move_prediction_component([scored_row(1, action_ids=(4, 5, 6))])

    assert listed.content_sha256 == tupled.content_sha256


def test_content_digest_changes_when_scored_content_changes(
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    baseline = move_prediction_component([scored_row(1), scored_row(2)])
    changed_move = move_prediction_component([scored_row(1, action_ids=[9])])
    grown = move_prediction_component(
        [scored_row(1), scored_row(2), scored_row(3)],
    )
    shrunk = move_prediction_component([scored_row(1)])

    digests = {
        baseline.content_sha256,
        changed_move.content_sha256,
        grown.content_sha256,
        shrunk.content_sha256,
    }
    assert len(digests) == 4


def test_content_digest_separates_projections(scored_row: RowFactory) -> None:
    rows = [scored_row(1), scored_row(2)]

    move = projection_content_digest(rows, MOVE_PREDICTION_PROJECTION)
    timing = projection_content_digest(rows, MOVE_TIMING_PROJECTION)

    assert move.content_sha256 != timing.content_sha256


def test_content_digest_rejects_incomplete_input(
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    with pytest.raises(FingerprintError, match="game id"):
        move_prediction_component([{"ruleset": "standard"}])
    with pytest.raises(FingerprintError, match="needs column"):
        move_prediction_component([{"game_id": 1}])
    with pytest.raises(FingerprintError, match="more than once"):
        move_prediction_component([scored_row(1), scored_row(1)])
    with pytest.raises(FingerprintError, match="at least one scored game"):
        move_prediction_component([])


def test_fingerprint_accepts_an_identifier_or_a_definition(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()

    assert series_fingerprint("held_out.move_loss", component) == series_fingerprint(
        metric_definition("held_out.move_loss"), component
    )


def test_fingerprint_separates_metrics_and_definition_versions(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    original = series_fingerprint("held_out.move_loss", component)

    assert series_fingerprint("legality.mask_penalty", component) != original

    probe = register_metric(
        MetricDefinition(
            identifier="held_out.move_loss_probe",
            family="held-out-prediction",
            direction=MetricDirection.LOWER_IS_BETTER,
            definition_version=2,
            summary="A probe proving that a definition bump starts a new series.",
            cost=MetricCost.SINGLE_PASS,
            projection=MOVE_PREDICTION_PROJECTION.name,
        )
    )
    version_one = MetricDefinition(
        identifier=probe.identifier,
        family=probe.family,
        direction=probe.direction,
        definition_version=1,
        summary=probe.summary,
        cost=probe.cost,
        projection=probe.projection,
    )

    assert series_fingerprint(probe, component) != series_fingerprint(
        version_one, component
    )


def test_fingerprint_breaks_when_scored_content_changes(
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    baseline = series_fingerprint(
        "held_out.move_loss",
        move_prediction_component([scored_row(1)]),
    )
    grown = series_fingerprint(
        "held_out.move_loss",
        move_prediction_component([scored_row(1), scored_row(2)]),
    )

    assert baseline != grown


def test_metric_without_data_dependency_carries_a_null_component(
    move_prediction_component: Digest,
) -> None:
    """A synthetic empty view would tie an immune metric to evaluation inputs."""

    fingerprint = series_fingerprint("training_health.gradient_norm", None)

    assert fingerprint == series_fingerprint("training_health.gradient_norm", None)
    with pytest.raises(FingerprintError, match="null data component"):
        series_fingerprint(
            "training_health.gradient_norm",
            move_prediction_component(),
        )


def test_data_dependent_metric_requires_a_component() -> None:
    with pytest.raises(FingerprintError, match="needs a data component"):
        series_fingerprint("held_out.move_loss", None)


def test_fingerprint_rejects_a_mismatched_projection(scored_row: RowFactory) -> None:
    timing = projection_content_digest([scored_row(1)], MOVE_TIMING_PROJECTION)

    with pytest.raises(FingerprintError, match="consumes projection"):
        series_fingerprint("held_out.move_loss", timing)


def test_fingerprint_rejects_a_stale_projection_version(
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    stale = DataComponent(
        projection=component.projection,
        projection_version=component.projection_version + 1,
        content_sha256=component.content_sha256,
        games=component.games,
    )

    with pytest.raises(FingerprintError, match="version"):
        series_fingerprint("held_out.move_loss", stale)


def test_scored_game_count_is_provenance_rather_than_identity(
    move_prediction_component: Digest,
) -> None:
    """The content digest already moves whenever the scored set does."""

    component = move_prediction_component()
    miscounted = DataComponent(
        projection=component.projection,
        projection_version=component.projection_version,
        content_sha256=component.content_sha256,
        games=component.games + 5,
    )

    assert series_fingerprint("held_out.move_loss", component) == series_fingerprint(
        "held_out.move_loss", miscounted
    )


def _execution(**overrides: Any) -> ExecutionComponent:
    fields: dict[str, Any] = {
        "device": "cpu",
        "device_name": "arm",
        "precision": "float32",
        "torch_version": "2.7.0",
        "platform": "macOS-15",
        "workload_sha256": workload_digest({"plies": 40}),
        "cpu_threads": 8,
    }
    fields.update(overrides)
    return ExecutionComponent(**fields)


def test_an_efficiency_series_breaks_on_the_machine_it_was_measured_on() -> None:
    """A latency figure is a property of a checkpoint on a machine.

    Without this, two laptops would share one line and a faster machine would
    read as an optimization.
    """

    baseline = series_fingerprint(
        "inference.move_latency_p50_ms",
        None,
        _execution(),
    )

    assert (
        series_fingerprint(
            "inference.move_latency_p50_ms",
            None,
            _execution(device="mps", device_name="arm64-mps", cpu_threads=None),
        )
        != baseline
    )
    assert (
        series_fingerprint(
            "inference.move_latency_p50_ms",
            None,
            _execution(cpu_threads=4),
        )
        != baseline
    )


def test_an_efficiency_series_breaks_when_the_declared_workload_changes() -> None:
    baseline = series_fingerprint(
        "inference.move_latency_p50_ms",
        None,
        _execution(workload_sha256=workload_digest({"plies": 40})),
    )
    deeper = series_fingerprint(
        "inference.move_latency_p50_ms",
        None,
        _execution(workload_sha256=workload_digest({"plies": 80})),
    )

    assert deeper != baseline


def test_an_efficiency_series_survives_an_unrelated_provenance_change() -> None:
    """Only what decides the number belongs in its identity."""

    assert series_fingerprint(
        "inference.move_latency_p50_ms", None, _execution()
    ) == series_fingerprint("inference.move_latency_p50_ms", None, _execution())


def test_an_execution_component_is_required_exactly_where_it_applies(
    move_prediction_component: Digest,
) -> None:
    with pytest.raises(FingerprintError, match="needs an execution component"):
        series_fingerprint("inference.move_latency_p50_ms", None, None)
    with pytest.raises(FingerprintError, match="not execution-sensitive"):
        series_fingerprint(
            "held_out.move_loss",
            move_prediction_component(),
            _execution(),
        )


def test_adding_execution_leaves_an_insensitive_series_bit_identical(
    move_prediction_component: Digest,
) -> None:
    """The component is absent, not null, for a metric that has no execution.

    Every series recorded before efficiency metrics existed has to keep its
    fingerprint, or this change would silently end all of them.
    """

    component = move_prediction_component()
    payload = {
        "algorithm": FINGERPRINT_ALGORITHM,
        "metric": "held_out.move_loss",
        "definition_version": metric_definition(
            "held_out.move_loss"
        ).definition_version,
        "data": component.fingerprint_component(),
    }
    expected = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert series_fingerprint("held_out.move_loss", component) == expected
