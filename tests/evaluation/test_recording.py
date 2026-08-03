"""The recording tail every benchmark shares once it has finished measuring."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from anthro_chess.config import ConfigModel, ConfigProvenance, ResolvedConfig
from anthro_chess.evaluation.recording import ResultRecorder
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
    DatasetReference,
    DetailStore,
    FloorEntry,
    Measurement,
    ResultEnvelope,
    ResultsStore,
    build_characterization,
    dataset_reference,
    measurement,
    series_fingerprint,
)
from anthro_chess.evaluation.results.store import LOCK_FILE_NAME

Digest = Callable[..., DataComponent]

KIND = "held-out-prediction"
BENCHMARK = BenchmarkReference(name="move-validation", version=1)
CHECKPOINT = CheckpointReference(label="checkpoint-a", step=8000)
METRIC = "held_out.move_loss"


class _Config(ConfigModel):
    """A minimal selection, so the configuration reference has something real."""

    view: str = "canonical"


@dataclass(frozen=True)
class _Result:
    """The three fields every benchmark result carries out of the tail."""

    envelopes: tuple[ResultEnvelope, ...] = ()
    recorded_paths: tuple[Path, ...] = ()
    detail_paths: tuple[Path, ...] = ()


class _BenchmarkError(ValueError):
    """The error type one benchmark declares in the suite's registry."""


@pytest.fixture
def resolved() -> ResolvedConfig[_Config]:
    return ResolvedConfig(
        value=_Config(),
        provenance=ConfigProvenance(source="selection.toml", overrides=()),
    )


def _recorder(
    resolved: ResolvedConfig[_Config],
    *,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> ResultRecorder:
    return ResultRecorder(
        resolved,
        kind=KIND,
        benchmark=BENCHMARK,
        checkpoint=CHECKPOINT,
        store=store,
        detail=detail,
        error=_BenchmarkError,
    )


def _dataset(component: DataComponent) -> DatasetReference:
    """Return the pool identity the measured content came from."""

    return dataset_reference(
        pool_id="fixture-pool",
        pool_version=1,
        view="canonical",
        selected_games=component.games,
        game_ids_sha256="a" * 64,
        components=[component],
    )


def _add_reading(
    recorder: ResultRecorder,
    component: DataComponent,
    *,
    payload: Callable[[], Mapping[str, object]] = dict,
    measurements: Sequence[Measurement] | None = None,
) -> None:
    """Record one ordinary unit, which is what most of these tests need."""

    recorder.add(
        [measurement(METRIC, 3.5, data=component)]
        if measurements is None
        else measurements,
        payload=payload,
        description="Slice tables for one evaluation.",
        data=_dataset(component),
    )


def test_a_reading_lands_in_the_store_and_the_detail_tier(
    tmp_path: Path,
    resolved: ResolvedConfig[_Config],
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    store = ResultsStore(tmp_path / "results")
    detail = DetailStore(tmp_path / "detail")

    with _recorder(resolved, store=store, detail=detail) as recorder:
        _add_reading(recorder, component, payload=lambda: {"slices": []})
    result = _Result(**recorder.fields)

    (envelope,) = result.envelopes
    (recorded,) = result.recorded_paths
    (written,) = result.detail_paths
    assert store.results() == (envelope,)
    assert recorded.is_file()
    # Absolute, and under the kind and the checkpoint the reading belongs to,
    # so a caller can hand the path straight to a reader.
    assert written.is_absolute()
    assert written.parent == detail.root / KIND / CHECKPOINT.label
    assert json.loads(written.read_text(encoding="utf-8")) == {"slices": []}
    assert envelope.detail is not None
    assert detail.root / envelope.detail.path == written
    assert envelope.configuration is not None
    assert envelope.configuration.source == "selection.toml"


def test_one_kind_scopes_both_the_series_and_the_payload_directory(
    tmp_path: Path,
    resolved: ResolvedConfig[_Config],
    move_prediction_component: Digest,
) -> None:
    # The two are named once and cannot drift apart. A benchmark writing more
    # than one kind from a single reading is why the override exists at all.
    other_kind = "rating-dependency"
    component = move_prediction_component()
    detail = DetailStore(tmp_path / "detail")

    with _recorder(resolved, detail=detail) as recorder:
        recorder.add(
            [measurement("dependency.rating_absent_degradation", 0.5, data=component)],
            payload=dict,
            description="Cross-conditioning tables.",
            kind=other_kind,
            benchmark=BenchmarkReference(name=other_kind, version=1),
            data=_dataset(component),
        )
    result = _Result(**recorder.fields)

    (envelope,) = result.envelopes
    (written,) = result.detail_paths
    assert envelope.kind == other_kind
    assert written.parent == detail.root / other_kind / CHECKPOINT.label


def test_a_payload_is_not_built_when_there_is_no_detail_store(
    resolved: ResolvedConfig[_Config],
    move_prediction_component: Digest,
) -> None:
    # `--no-record` is how a shakedown reading is taken, and some of these
    # payloads cost real work, so the caller's is never invoked there.
    component = move_prediction_component()
    built = 0

    def payload() -> dict[str, object]:
        nonlocal built
        built += 1
        return {}

    with _recorder(resolved) as recorder:
        _add_reading(recorder, component, payload=payload)
    result = _Result(**recorder.fields)

    assert built == 0
    assert len(result.envelopes) == 1
    assert result.recorded_paths == ()
    assert result.detail_paths == ()


def test_measuring_without_a_store_still_writes_the_detail_payload(
    tmp_path: Path,
    resolved: ResolvedConfig[_Config],
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()

    with _recorder(resolved, detail=DetailStore(tmp_path / "detail")) as recorder:
        _add_reading(recorder, component)
    result = _Result(**recorder.fields)

    assert len(result.envelopes) == 1
    assert result.recorded_paths == ()
    assert len(result.detail_paths) == 1


def test_measuring_nothing_keeps_the_payload_and_commits_no_record(
    tmp_path: Path,
    resolved: ResolvedConfig[_Config],
) -> None:
    # The committed tier cannot hold a record with no measurement in it, and a
    # pass that came back empty is exactly when the evidence for why matters.
    detail = DetailStore(tmp_path / "detail")
    store = ResultsStore(tmp_path / "results")

    with _recorder(resolved, store=store, detail=detail) as recorder:
        recorder.add(
            (),
            payload=lambda: {"resignations": 0},
            description="Held-out resignation.",
            slug="held-out-resignation",
        )
    result = _Result(**recorder.fields)

    assert result.envelopes == ()
    assert result.recorded_paths == ()
    (written,) = result.detail_paths
    assert written.is_file()


def test_a_slug_distinguishes_the_payloads_one_reading_writes(
    tmp_path: Path,
    resolved: ResolvedConfig[_Config],
) -> None:
    with _recorder(resolved, detail=DetailStore(tmp_path / "detail")) as recorder:
        recorder.add((), payload=dict, description="One cell.", slug="cell-r1200")
        recorder.add((), payload=dict, description="Another.", slug="cell-r1800")
    result = _Result(**recorder.fields)

    first, second = result.detail_paths
    assert first.name.endswith("-cell-r1200.json")
    assert second.name.endswith("-cell-r1800.json")


def test_a_noise_floor_is_appended_after_the_envelopes(
    tmp_path: Path,
    resolved: ResolvedConfig[_Config],
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    store = ResultsStore(tmp_path / "results")
    characterization = build_characterization(
        kind="data-sampling",
        method="bootstrap",
        replicates=200,
        source="one evaluation view",
        floors=[
            FloorEntry(
                metric=METRIC,
                fingerprint=series_fingerprint(METRIC, component),
                floor=0.1,
                dispersion=0.05,
                dispersion_bound=0.05,
                degrees_of_freedom=5,
            )
        ],
    )

    with _recorder(resolved, store=store) as recorder:
        _add_reading(recorder, component)
        recorder.characterize(characterization)
        # A benchmark whose floor is disabled hands over nothing rather than
        # branching at every call site.
        recorder.characterize(None)
    result = _Result(**recorder.fields)

    first, second = result.recorded_paths
    assert first.parent == store.records_directory
    assert second.parent == store.floors_directory
    assert store.characterizations() == (characterization,)


def test_a_store_failure_becomes_the_benchmark_s_own_error(
    tmp_path: Path,
    resolved: ResolvedConfig[_Config],
    move_prediction_component: Digest,
) -> None:
    # A sweep converts only the error types a benchmark's registry entry
    # declares, so a store error escaping as itself would end the whole sweep
    # rather than failing one step. The append happens on the way out of the
    # block, so this is the conversion covering it there.
    component = move_prediction_component()
    root = tmp_path / "results"
    root.mkdir(parents=True)
    (root / LOCK_FILE_NAME).write_text("1234\n", encoding="utf-8")

    with pytest.raises(_BenchmarkError, match="another process is writing"):
        with _recorder(resolved, store=ResultsStore(root)) as recorder:
            _add_reading(recorder, component)


def test_a_failed_reading_appends_nothing(
    tmp_path: Path,
    resolved: ResolvedConfig[_Config],
    move_prediction_component: Digest,
) -> None:
    # A partial record of a broken measurement is worse than none, so a block
    # that raised leaves the store as it found it.
    component = move_prediction_component()
    store = ResultsStore(tmp_path / "results")

    with pytest.raises(RuntimeError):
        with _recorder(resolved, store=store) as recorder:
            _add_reading(recorder, component)
            raise RuntimeError("the pool went away")

    assert store.results() == ()
    assert recorder.fields["recorded_paths"] == ()


def test_building_a_measurement_is_covered_by_the_same_conversion(
    resolved: ResolvedConfig[_Config],
) -> None:
    # The conversion spans the whole block rather than only the calls made into
    # the recorder: a benchmark builds its measurements there, and `measurement`
    # raises the same errors the store does.
    with pytest.raises(_BenchmarkError, match="unregistered.metric"):
        with _recorder(resolved) as recorder:
            recorder.add(
                [measurement("unregistered.metric", 1.0)],
                payload=dict,
                description="Slice tables.",
            )


def test_an_envelope_that_cannot_reproduce_its_series_is_converted_too(
    resolved: ResolvedConfig[_Config],
    move_prediction_component: Digest,
) -> None:
    # This metric consumes a projection, so the envelope needs the dataset the
    # digest came from. `build_result` verifies that rather than trusting it.
    with pytest.raises(_BenchmarkError, match="content digest"):
        with _recorder(resolved) as recorder:
            recorder.add(
                [measurement(METRIC, 3.5, data=move_prediction_component())],
                payload=dict,
                description="Slice tables.",
            )


def test_an_error_the_recording_tail_does_not_own_is_left_alone(
    resolved: ResolvedConfig[_Config],
) -> None:
    # Only the store's own errors are converted. A benchmark failing to measure
    # is not a recording failure, and dressing it as one would report a broken
    # instrument as a bad write.
    with pytest.raises(RuntimeError, match="the pool went away"):
        with _recorder(resolved):
            raise RuntimeError("the pool went away")
