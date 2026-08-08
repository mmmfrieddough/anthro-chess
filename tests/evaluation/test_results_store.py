"""The committed summary tier, its detail-tier boundary, and its write safety."""

from __future__ import annotations

import json
from collections.abc import Callable
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
    Measurement,
    MetricCost,
    MetricDefinition,
    MetricDirection,
    NoiseFloor,
    ResultEnvelope,
    ResultRecordError,
    ResultsStore,
    ResultsStoreError,
    build_bridge,
    build_result,
    configuration_reference,
    dataset_reference,
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


def test_a_measurement_carries_its_noise_floor(
    move_prediction_component: Callable[..., DataComponent],
) -> None:
    component = move_prediction_component()

    value = measurement(
        "held_out.move_loss",
        3.5,
        data=component,
        sample_size=12_000,
        noise_floor=NoiseFloor(value=0.01, kind="training", source="five seeds"),
    )

    assert isinstance(value, Measurement)
    assert value.noise_floor is not None
    assert value.noise_floor.kind == "training"


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
