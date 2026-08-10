"""The committed summary tier, its detail-tier boundary, and its write safety."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from anthro_chess.evaluation.results import (
    MAXIMUM_SUMMARY_BYTES,
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
    DetailReference,
    DetailStore,
    ExecutionRecord,
    Measurement,
    MetricCost,
    MetricDefinition,
    MetricDirection,
    MetricDispersion,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    ResultsStoreError,
    build_bridge,
    build_result,
    configuration_reference,
    dataset_reference,
    execution_reference,
    measurement,
    register_metric,
    registry_snapshot,
    resolve_detail_root,
    resolve_optional_detail_root,
    resolve_store_root,
    restore_registry,
)
from anthro_chess.evaluation.results.store import (
    DETAIL_ROOT_VARIABLE,
    LOCK_FILE_NAME,
    STORE_ROOT_VARIABLE,
)

ResultFactory = Callable[..., ResultEnvelope]


def test_append_writes_one_readable_file_per_result(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    store = ResultsStore(tmp_path / "results")
    result = recorded_result()

    path = store.append(result)
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent == store.records_directory
    assert raw["result_id"] == result.result_id
    assert raw["checkpoint"]["label"] == "checkpoint-a"
    assert store.results() == (result,)


def test_appending_the_same_result_twice_is_idempotent(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    store = ResultsStore(tmp_path / "results")
    result = recorded_result()

    first = store.append(result)
    second = store.append(result)

    assert first == second
    assert len(store.results()) == 1


def test_a_different_result_never_overwrites_a_recorded_one(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    store = ResultsStore(tmp_path / "results")
    result = recorded_result()
    store.append(result)
    conflicting = result.model_copy(update={"measurements": (result.measurements[0],)})

    with pytest.raises(ResultsStoreError, match="already recorded"):
        store.append(conflicting)


def test_a_held_lock_fails_clearly_instead_of_corrupting_the_store(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    store = ResultsStore(tmp_path / "results")
    store.root.mkdir(parents=True)
    (store.root / LOCK_FILE_NAME).write_text("1234\n")

    with pytest.raises(ResultsStoreError, match="another process is writing"):
        store.append(recorded_result())
    assert not store.records_directory.exists()


def test_a_failed_write_releases_the_lock(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    store = ResultsStore(tmp_path / "results")
    result = recorded_result()
    store.append(result)

    with pytest.raises(ResultsStoreError):
        store.append(
            result.model_copy(update={"measurements": result.measurements[:1]})
        )

    assert not (store.root / LOCK_FILE_NAME).exists()
    assert store.append(recorded_result(label="checkpoint-b")).is_file()


def test_results_are_returned_in_recording_order(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    store = ResultsStore(tmp_path / "results")
    later = recorded_result(
        label="checkpoint-b",
        recorded_at=datetime(2026, 7, 3, tzinfo=UTC),
    )
    earlier = recorded_result(
        label="checkpoint-a",
        recorded_at=datetime(2026, 7, 2, tzinfo=UTC),
    )
    store.append(later)
    store.append(earlier)

    assert [item.checkpoint.label for item in store.results()] == [
        "checkpoint-a",
        "checkpoint-b",
    ]


def test_a_result_that_cannot_reproduce_its_fingerprint_is_rejected(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    """A result carries enough provenance to recompute its own series."""

    store = ResultsStore(tmp_path / "results")
    path = store.append(recorded_result())
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["measurements"][0]["fingerprint"] = "b" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ResultsStoreError, match="does not reproduce"):
        store.results()


def test_a_committed_record_is_capped_at_the_summary_tier_budget(
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    """The tier boundary is enforced rather than left to good intentions."""

    component = move_prediction_component()
    bulk = measurement("held_out.move_loss", 3.5, data=component)
    with pytest.raises(ResultRecordError, match="detail tier"):
        build_result(
            kind="held-out-prediction",
            benchmark=BenchmarkReference(name="move-validation", version=1),
            checkpoint=CheckpointReference(label="checkpoint-a"),
            data=dataset_reference(
                pool_id="fixture-pool",
                pool_version=1,
                view="canonical",
                selected_games=component.games,
                game_ids_sha256="a" * 64,
                components=[component],
            ),
            measurements=[bulk],
            configuration=configuration_reference(
                {"padding": "x" * (MAXIMUM_SUMMARY_BYTES + 1)},
                source="x" * (MAXIMUM_SUMMARY_BYTES + 1),
            ),
        )


def test_bulk_diagnostics_may_not_be_written_into_the_committed_tier(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    store = ResultsStore(tmp_path / "results")
    inside = DetailReference(
        path=str(store.root / "records" / "positions.json"),
        sha256="c" * 64,
        bytes=10,
    )

    with pytest.raises(ResultsStoreError, match="machine-local detail tier"):
        store.append(recorded_result().model_copy(update={"detail": inside}))


def test_detail_store_round_trips_a_referenced_payload(tmp_path: Path) -> None:
    detail = DetailStore(tmp_path / "detail")

    reference = detail.write(
        "checkpoint-a/positions.json",
        {"positions": [{"game_id": 1, "mask_penalty": 0.5}]},
        description="per-position legality diagnostics",
    )

    assert detail.read(reference) == {
        "positions": [{"game_id": 1, "mask_penalty": 0.5}]
    }
    (detail.root / reference.path).write_text("{}", encoding="utf-8")
    with pytest.raises(ResultsStoreError, match="checksum mismatch"):
        detail.read(reference)


def test_detail_store_refuses_to_escape_its_root(tmp_path: Path) -> None:
    detail = DetailStore(tmp_path / "detail")

    with pytest.raises(ResultsStoreError, match="beneath the detail root"):
        detail.write("../escaped.json", {})


def test_a_refused_detail_write_is_the_store_s_own_error(tmp_path: Path) -> None:
    detail = DetailStore(tmp_path / "detail")
    destination = detail.root / "checkpoint-a" / "positions.json"
    destination.mkdir(parents=True)

    with pytest.raises(ResultsStoreError, match="cannot write"):
        detail.write("checkpoint-a/positions.json", {"positions": []})

    # The caller carries on past this error, so nothing else will ever remove
    # the half-written copy.
    assert list(destination.parent.glob(".*.tmp")) == []


def test_a_detail_payload_the_serializer_refuses_is_the_store_s_own_error(
    tmp_path: Path,
) -> None:
    # A committed measurement cannot be non-finite, but a detail payload is
    # freeform: a rate over zero samples reaches the serializer unchecked.
    detail = DetailStore(tmp_path / "detail")

    # Indenting forces the pure-Python encoder, which recurses once per level,
    # so past the interpreter's limit the payload exhausts the stack instead.
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(2 * sys.getrecursionlimit()):
        deeper: dict[str, object] = {}
        cursor["next"] = deeper
        cursor = deeper

    with pytest.raises(ResultsStoreError, match="cannot serialize"):
        detail.write("checkpoint-a/rates.json", {"rate": float("nan")})

    with pytest.raises(ResultsStoreError, match="cannot serialize"):
        detail.write("checkpoint-a/slopes.json", {"slope": object()})

    with pytest.raises(ResultsStoreError, match="cannot serialize"):
        detail.write("checkpoint-a/tree.json", nested)

    # The caller carries on past this error, so a directory made for a payload
    # that never arrives is one nothing will ever remove.
    assert not detail.root.exists()


def _execution(
    *,
    workload: Mapping[str, object] | None = None,
    **coordinates: object,
) -> ExecutionRecord:
    """Return one benchmark's declared conditions, sound unless overridden.

    Coordinates are the committed tier's one unvalidated freeform slot: a
    workload is digested on the way in, an encoding and an action vocabulary are
    code-owned, and a measurement is checked finite.
    """

    return execution_reference(
        device="cpu",
        device_name="fixture-cpu",
        precision="float32",
        torch_version="2.0.0",
        platform_key="Linux-x86_64",
        platform="Linux-x86_64-fixture",
        workload={"reference_plies": 40} if workload is None else workload,
        coordinates=coordinates,
    )


def test_a_committed_record_the_serializer_refuses_is_the_store_s_own_error(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    # The committed tier's twin of the detail-tier refusal above, and it arrives
    # at the same late moment: after the benchmark has finished measuring.
    store = ResultsStore(tmp_path / "results")
    refused = recorded_result().model_copy(
        update={"execution": _execution(slope=object())}
    )

    with pytest.raises(ResultsStoreError, match="cannot serialize"):
        store.append(refused)

    assert store.results() == ()


def test_a_non_finite_freeform_value_is_refused_rather_than_rewritten(
    recorded_result: ResultFactory,
) -> None:
    # Rendering it as `null` would be the quieter failure: the record would
    # commit, and digest, a value its configuration never held.
    tainted = recorded_result().model_copy(
        update={"execution": _execution(rate=float("nan"))}
    )

    assert math.isnan(tainted.as_record()["execution"]["coordinates"]["rate"])
    with pytest.raises(ResultRecordError, match="cannot serialize"):
        tainted.verify()


def test_a_record_that_cannot_be_serialized_still_reads_back(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    # `json.loads` accepts `NaN`, and reading re-encodes nothing, so one record
    # carrying a literal the canonical writer refuses does not take the whole
    # committed history with it.
    store = ResultsStore(tmp_path / "results")
    path = store.append(recorded_result())
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["execution"] = {
        **_execution().as_record(),
        "coordinates": {"rate": float("nan")},
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    (loaded,) = store.results()

    assert loaded.execution is not None
    assert math.isnan(loaded.execution.coordinates["rate"])


def test_a_declared_workload_the_serializer_refuses_is_a_record_error() -> None:
    # A declared workload carries settings straight from configuration — the
    # termination benchmark puts `guardrails.premature_material_balance` in
    # one — and reaches its digest before it reaches any record.
    with pytest.raises(ResultRecordError, match="cannot digest"):
        _execution(workload={"temperature": float("nan")})


def test_a_record_that_cannot_be_read_back_is_the_store_s_own_error(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    # Appending compares against what is already recorded before it writes, so
    # the read is as much a part of the append as the write is.
    store = ResultsStore(tmp_path / "results")
    result = recorded_result()
    path = store.append(result)
    path.unlink()
    path.mkdir()

    with pytest.raises(ResultsStoreError, match="cannot read"):
        store.append(result)


def test_bridges_are_recorded_beside_results_and_can_be_revoked(
    tmp_path: Path,
) -> None:
    store = ResultsStore(tmp_path / "results")
    bridge = build_bridge(
        from_fingerprint="a" * 64,
        to_fingerprint="b" * 64,
        reason="storage format change; the scored games are identical",
        author="maintainer",
        recorded_at=datetime(2026, 7, 5, tzinfo=UTC),
    )

    path = store.append_bridge(bridge)
    assert store.bridges() == (bridge,)

    store.revoke_bridge(bridge.bridge_id)
    assert store.bridges() == ()
    assert not path.exists()
    with pytest.raises(ResultsStoreError, match="no bridge is recorded"):
        store.revoke_bridge(bridge.bridge_id)


def test_a_bridge_must_join_two_different_fingerprints() -> None:
    with pytest.raises(ValueError, match="two different fingerprints"):
        build_bridge(
            from_fingerprint="a" * 64,
            to_fingerprint="a" * 64,
            reason="no-op",
            author="maintainer",
        )


def test_recorded_output_is_deterministic(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    """Two stores given the same result produce byte-identical files."""

    first = ResultsStore(tmp_path / "first")
    second = ResultsStore(tmp_path / "second")
    result = recorded_result()

    assert first.append(result).read_bytes() == second.append(result).read_bytes()


def test_measurements_are_ordered_and_unique(
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    component = move_prediction_component()
    duplicated = measurement("held_out.move_loss", 3.5, data=component)

    with pytest.raises(ValueError, match="each metric once"):
        build_result(
            kind="held-out-prediction",
            benchmark=BenchmarkReference(name="move-validation", version=1),
            checkpoint=CheckpointReference(label="checkpoint-a"),
            data=dataset_reference(
                pool_id="fixture-pool",
                pool_version=1,
                view="canonical",
                selected_games=component.games,
                game_ids_sha256="a" * 64,
                components=[component],
            ),
            measurements=[duplicated, duplicated],
        )


def test_a_measurement_carries_its_own_dispersion(
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    component = move_prediction_component()

    value = measurement(
        "held_out.move_loss",
        3.5,
        data=component,
        sample_size=12_000,
        dispersion=MetricDispersion(
            value=0.01,
            bound=0.012,
            source="bootstrap over 12000 game(s)",
        ),
    )

    assert isinstance(value, Measurement)
    assert value.dispersion is not None
    assert value.dispersion.source == "bootstrap over 12000 game(s)"


def test_a_dispersion_bound_below_the_spread_it_bounds_is_refused() -> None:
    # A reading's stored bound is what a delta floor is combined from, so one
    # written under the spread it bounds would understate every floor it
    # reaches.
    with pytest.raises(ValueError, match="not a conservative limit"):
        MetricDispersion(value=0.2, bound=0.1)


def test_a_non_finite_measurement_is_rejected(
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    with pytest.raises(ValueError, match="finite"):
        measurement(
            "held_out.move_loss",
            float("nan"),
            data=move_prediction_component(),
        )


def test_store_roots_resolve_from_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(STORE_ROOT_VARIABLE, str(tmp_path / "store"))
    monkeypatch.setenv(DETAIL_ROOT_VARIABLE, str(tmp_path / "detail"))

    assert resolve_store_root() == tmp_path / "store"
    assert resolve_detail_root() == tmp_path / "detail"
    assert resolve_optional_detail_root() == tmp_path / "detail"
    assert resolve_store_root(tmp_path / "explicit") == tmp_path / "explicit"

    monkeypatch.delenv(DETAIL_ROOT_VARIABLE)
    monkeypatch.setenv("ANTHRO_CHESS_RUN_ROOT", str(tmp_path / "runs"))
    assert resolve_detail_root() == tmp_path / "runs" / "benchmark-detail"
    assert resolve_optional_detail_root() == tmp_path / "runs" / "benchmark-detail"

    monkeypatch.delenv("ANTHRO_CHESS_RUN_ROOT")
    with pytest.raises(ResultsStoreError, match="must be set"):
        resolve_detail_root()
    assert resolve_optional_detail_root() is None


def test_a_new_benchmark_kind_needs_no_schema_change(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    """A later benchmark family is a new kind string, not a new record shape."""

    store = ResultsStore(tmp_path / "results")
    rollout = recorded_result(
        kind="generated-game-rollout",
        label="checkpoint-a",
        measurements=[
            measurement("training_health.gradient_norm", 1.25),
        ],
    )

    store.append(rollout)
    stored = store.results()[0]

    assert stored.kind == "generated-game-rollout"
    assert stored.measurement("training_health.gradient_norm") is not None


def test_a_metric_without_a_matching_content_digest_is_refused(
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    component = move_prediction_component()
    value = measurement("held_out.move_loss", 3.5, data=component)

    with pytest.raises(ResultRecordError, match="without a matching"):
        build_result(
            kind="held-out-prediction",
            benchmark=BenchmarkReference(name="move-validation", version=1),
            checkpoint=CheckpointReference(label="checkpoint-a"),
            measurements=[value],
        )


def test_a_retired_metric_leaves_history_readable(
    tmp_path: Path,
    recorded_result: ResultFactory,
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    """A dead series stays readable rather than making the store unloadable."""

    snapshot = registry_snapshot()
    probe = register_metric(
        MetricDefinition(
            identifier="legality.retired_probe",
            family="legality",
            direction=MetricDirection.LOWER_IS_BETTER,
            definition_version=1,
            summary="A metric that later leaves the registry.",
            cost=MetricCost.SINGLE_PASS,
            projection="move_prediction",
        )
    )
    component = move_prediction_component()
    store = ResultsStore(tmp_path / "results")
    store.append(
        recorded_result(
            measurements=[measurement(probe.identifier, 0.5, data=component)],
            component=component,
        )
    )

    restore_registry(snapshot)

    stored = store.results()
    assert stored[0].measurement(probe.identifier) is not None
