"""TensorBoard projection of durable checkpoint history."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore[import-untyped]
    EventAccumulator,
)

from anthro_chess.evaluation.results import DataComponent, ResultEnvelope
from anthro_chess.evaluation.tensorboard import (
    PROJECTION_MARKER,
    TensorBoardProjectionError,
    project_results,
)

ResultFactory = Callable[..., ResultEnvelope]
Digest = Callable[..., DataComponent]


def _event_runs(output: Path) -> dict[str, EventAccumulator]:
    runs = {}
    for event_file in output.rglob("events.out.tfevents.*"):
        run = event_file.parent.relative_to(output).as_posix()
        accumulator = EventAccumulator(str(event_file.parent))
        accumulator.Reload()
        runs[run] = accumulator
    return runs


def test_series_are_separate_lines_at_stable_checkpoint_ordinals(
    tmp_path: Path,
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    first_series = move_prediction_component()
    second_series = move_prediction_component(
        [
            {
                "game_id": 3,
                "ruleset": "standard",
                "initial_position": "startpos",
                "action_ids": [1, 2, 3],
                "white_normalized_rating": 1500,
                "black_normalized_rating": 1500,
            }
        ]
    )
    results = (
        recorded_result(
            label="checkpoint-a",
            move_loss=3.5,
            component=first_series,
            recorded_at=datetime(2026, 7, 1, tzinfo=UTC),
        ),
        recorded_result(
            label="checkpoint-b",
            move_loss=3.2,
            component=first_series,
            recorded_at=datetime(2026, 7, 2, tzinfo=UTC),
        ),
        recorded_result(
            label="checkpoint-c",
            move_loss=3.1,
            component=second_series,
            recorded_at=datetime(2026, 7, 3, tzinfo=UTC),
        ),
        # A later re-score replaces checkpoint-a at ordinal zero.
        recorded_result(
            label="checkpoint-a",
            move_loss=3.4,
            component=first_series,
            recorded_at=datetime(2026, 7, 4, tzinfo=UTC),
        ),
    )
    output = tmp_path / "history"

    projection = project_results(results, output, store_root=tmp_path / "results")

    runs = {
        name: events
        for name, events in _event_runs(output).items()
        if "/held_out.move_loss/" in name
    }
    assert projection.checkpoints == 3
    assert len(runs) == 2
    tag = "held-out-prediction/held_out.move_loss"
    scalar_lines = sorted(
        (
            [event.step for event in events.Scalars(tag)],
            [event.value for event in events.Scalars(tag)],
        )
        for events in runs.values()
    )
    assert scalar_lines == [
        ([0, 1], pytest.approx([3.4, 3.2])),
        ([2], pytest.approx([3.1])),
    ]
    assert all(
        name.startswith("held-out-prediction/held_out.move_loss/") for name in runs
    )


def test_regeneration_is_equivalent_and_does_not_accumulate_events(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    output = tmp_path / "history"
    results = (recorded_result(),)

    first = project_results(results, output, store_root=tmp_path / "results")
    first_scalars = {
        run: {
            tag: [(event.step, event.value) for event in events.Scalars(tag)]
            for tag in events.Tags()["scalars"]
        }
        for run, events in _event_runs(output).items()
    }
    second = project_results(results, output, store_root=tmp_path / "results")
    second_scalars = {
        run: {
            tag: [(event.step, event.value) for event in events.Scalars(tag)]
            for tag in events.Tags()["scalars"]
        }
        for run, events in _event_runs(output).items()
    }

    assert second == first
    assert second_scalars == first_scalars
    assert (output / PROJECTION_MARKER).is_file()


def test_an_empty_store_produces_an_owned_empty_view(tmp_path: Path) -> None:
    output = tmp_path / "history"

    projection = project_results((), output, store_root=tmp_path / "results")

    assert projection.checkpoints == 0
    assert projection.runs == 0
    assert projection.points == 0
    assert (output / PROJECTION_MARKER).is_file()
    assert not tuple(output.rglob("events.out.tfevents.*"))


def test_projection_refuses_unowned_output_and_the_committed_store(
    tmp_path: Path,
    recorded_result: ResultFactory,
) -> None:
    output = tmp_path / "history"
    output.mkdir()
    (output / "notes.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(TensorBoardProjectionError, match="non-projection"):
        project_results(
            (recorded_result(),),
            output,
            store_root=tmp_path / "results",
        )
    with pytest.raises(TensorBoardProjectionError, match="outside"):
        project_results(
            (recorded_result(),),
            tmp_path / "results" / "tensorboard",
            store_root=tmp_path / "results",
        )

    assert (output / "notes.txt").read_text(encoding="utf-8") == "keep me"
